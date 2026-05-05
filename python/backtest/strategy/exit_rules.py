"""
止盈 / 止损 / 移动止损（Exit Rules）
====================================
在日频回测中，仅能在「已知决策时点价格」上判定触发，无盘中 tick。
参考价默认使用 exchange 提供的收盘价（与 qlib 日频撮合一致）。

设计约束：
  - 不读取未来数据：估价区间使用 trade_start_time / trade_end_time
  - 参考入场价 `_entry_ref` 在首次建仓或加仓后由策略更新（见 QuantMLWeightStrategy）
  - 本模块核心逻辑可单测，不依赖 qlib Position
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class ExitRulesConfig:
    """止盈止损参数（均为正小数，如 0.08 表示 8%）。"""

    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None

    def is_enabled(self) -> bool:
        return any(
            x is not None and x > 0
            for x in (
                self.stop_loss_pct,
                self.take_profit_pct,
                self.trailing_stop_pct,
            )
        )


def parse_exit_rules_cfg(raw: Optional[Mapping[str, object]]) -> ExitRulesConfig:
    """从 strategy kwargs 中的 exit_rules 字典解析配置。"""
    if not raw:
        return ExitRulesConfig()
    def _f(key: str) -> Optional[float]:
        v = raw.get(key)
        if v is None:
            return None
        x = float(v)
        if x <= 0:
            raise ValueError(f"exit_rules.{key} 必须 > 0 或未设置，当前: {v}")
        return x
    return ExitRulesConfig(
        stop_loss_pct=_f("stop_loss_pct"),
        take_profit_pct=_f("take_profit_pct"),
        trailing_stop_pct=_f("trailing_stop_pct"),
    )


def evaluate_exit_triggers(
    *,
    mark: float,
    entry_ref: float,
    peak: float,
    cfg: ExitRulesConfig,
) -> Tuple[bool, str]:
    """
    单标的判断是否触发任一退出条件。

    Returns:
        (should_exit, reason)  reason 如 "stop_loss" / "take_profit" / "trailing_stop" / ""
    """
    if mark <= 0 or entry_ref <= 0:
        return False, ""

    sl = cfg.stop_loss_pct
    if sl is not None and sl > 0 and mark <= entry_ref * (1.0 - sl):
        return True, "stop_loss"

    tp = cfg.take_profit_pct
    if tp is not None and tp > 0 and mark >= entry_ref * (1.0 + tp):
        return True, "take_profit"

    tr = cfg.trailing_stop_pct
    if tr is not None and tr > 0 and peak > 0 and mark <= peak * (1.0 - tr):
        return True, "trailing_stop"

    return False, ""


def strip_exited_symbols(
    target: MutableMapping[str, float],
    held_codes: List[str],
    marks: Mapping[str, float],
    entry_ref: MutableMapping[str, float],
    peak: MutableMapping[str, float],
    cfg: ExitRulesConfig,
) -> List[Dict[str, Any]]:
    """
    对当前持仓逐个检查止盈止损；触发的标的从 target 中移除（赋 0 并 pop），
    并清理 entry_ref / peak。

    Returns:
        本步被强制平仓的事件列表（便于日志/持久化），每元素含：
          code / reason / mark / entry_ref / peak / pnl_pct
    """
    closed: List[Dict[str, Any]] = []
    if not cfg.is_enabled():
        return closed

    for code in held_codes:
        mark = marks.get(code)
        if mark is None or not np.isfinite(mark) or mark <= 0:
            continue
        ref = entry_ref.get(code, mark)
        pk = peak.get(code, ref)
        pk = max(pk, mark)
        peak[code] = pk

        exit_ok, reason = evaluate_exit_triggers(
            mark=mark, entry_ref=ref, peak=pk, cfg=cfg
        )
        if exit_ok:
            closed.append(
                {
                    "code": str(code),
                    "reason": reason,
                    "mark": float(mark),
                    "entry_ref": float(ref),
                    "peak": float(pk),
                    "pnl_pct": (float(mark) - float(ref)) / float(ref)
                    if ref > 0
                    else 0.0,
                }
            )
            target.pop(code, None)
            entry_ref.pop(code, None)
            peak.pop(code, None)
    return closed


def renormalize_target(
    target: MutableMapping[str, float],
    eps: float = 1e-12,
) -> Dict[str, float]:
    """强制清仓后将其余权重按比例缩放到和为 1（满仓风险度仍由 risk_degree 决定）。"""
    s = sum(max(0.0, float(v)) for v in target.values())
    if s <= eps:
        return {}
    return {str(k): max(0.0, float(v)) / s for k, v in target.items() if float(v) > eps}
