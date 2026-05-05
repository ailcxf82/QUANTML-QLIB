"""
策略层（Strategy Layer）
=========================
职责：
  - 接收预测信号，结合策略参数，构造 Qlib 策略配置字典
  - 对策略参数做基本合法性检查（topk、n_drop 范围等）
  - 提供策略描述供日志与报告使用

输入: strategy_cfg（dict，来自 YAML）+ pred_score（pd.Series）
输出: Qlib 可直接使用的策略配置 dict

支持策略:
  - TopkDropoutStrategy: 选取预测分前 topk 只，每期换出 n_drop 只（Qlib 内置）
  - QuantMLWeightStrategy: 五段式管线（候选 → 加权 → 约束 → 提交，本仓库扩展）
  （扩展时实现 build_custom 即可，不影响现有流程）
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from backtest.strategy.exit_rules import parse_exit_rules_cfg


class StrategyLayer:
    """
    策略层：将预测信号与策略参数合并，构造 Qlib 策略配置。

    设计原则:
      - 信号注入在此层完成，下游（ExecutionLayer）只看执行配置
      - 参数校验在此层报错，保证错误信息友好
    """

    def build(
        self,
        strategy_cfg: Dict[str, Any],
        signal: pd.Series,
    ) -> Dict[str, Any]:
        """
        构造 Qlib strategy 配置字典。

        Args:
            strategy_cfg: 来自 YAML 的策略配置（含 class/module_path/kwargs）
            signal:       模型预测打分序列（MultiIndex datetime×instrument）

        Returns:
            注入了 signal 的策略配置 dict，可直接传入 qlib.backtest.backtest()
        """
        class_name = strategy_cfg["class"]
        self._validate_params(class_name, strategy_cfg.get("kwargs", {}))

        return {
            "class": class_name,
            "module_path": strategy_cfg["module_path"],
            "kwargs": {
                **strategy_cfg.get("kwargs", {}),
                "signal": signal,
            },
        }

    def describe(self, strategy_cfg: Dict[str, Any]) -> str:
        """返回策略的可读描述字符串，用于日志和报告。"""
        class_name = strategy_cfg.get("class", "Unknown")
        kwargs = strategy_cfg.get("kwargs", {})

        if class_name == "TopkDropoutStrategy":
            topk = kwargs.get("topk", "?")
            n_drop = kwargs.get("n_drop", "?")
            hold_thresh = kwargs.get("hold_thresh", 0.5)
            return (
                f"TopkDropoutStrategy("
                f"持仓={topk}只, 每期换出最多={n_drop}只, "
                f"持有阈值={hold_thresh})"
            )
        if class_name == "QuantMLWeightStrategy":
            topk = kwargs.get("topk", "?")
            sq = kwargs.get("score_quantile", 0.0)
            rebal = kwargs.get("rebalance_freq", 1)
            wcfg = kwargs.get("weighter_cfg", {})
            rcfg = kwargs.get("risk_cfg", {})
            er = kwargs.get("exit_rules") or {}
            er_parts = []
            for label, key in (
                ("SL", "stop_loss_pct"),
                ("TP", "take_profit_pct"),
                ("Trail", "trailing_stop_pct"),
            ):
                v = er.get(key)
                if v is not None:
                    er_parts.append(f"{label}={v}")
            er_str = ", ".join(er_parts) if er_parts else "off"
            n_drop = kwargs.get("n_drop")
            n_drop_str = (
                f"≤ {n_drop} 只/期"
                if isinstance(n_drop, int) and n_drop > 0
                else "off"
            )
            return (
                f"QuantMLWeightStrategy(topk={topk}, score_q={sq}, "
                f"rebal_freq={rebal}步)\n"
                f"    权重方案 : {wcfg.get('type', 'inverse_vol')}\n"
                f"    单票上限 : {rcfg.get('max_weight', 'N/A')}\n"
                f"    板块上限 : {rcfg.get('sector_max_weight', 'N/A')}\n"
                f"    换手上限 : {rcfg.get('max_turnover', 'N/A')}\n"
                f"    每期换出 : {n_drop_str}\n"
                f"    止盈止损 : {er_str}"
            )
        return f"{class_name}({kwargs})"

    def _validate_params(self, class_name: str, kwargs: Dict[str, Any]) -> None:
        """对已知策略类型做参数范围校验。"""
        if class_name == "TopkDropoutStrategy":
            topk = kwargs.get("topk", 50)
            n_drop = kwargs.get("n_drop", 5)
            if not isinstance(topk, int) or topk <= 0:
                raise ValueError(f"TopkDropoutStrategy.topk 必须为正整数，当前: {topk}")
            if not isinstance(n_drop, int) or n_drop < 0:
                raise ValueError(f"TopkDropoutStrategy.n_drop 必须为非负整数，当前: {n_drop}")
            if n_drop >= topk:
                raise ValueError(
                    f"TopkDropoutStrategy.n_drop({n_drop}) 不应 >= topk({topk})，"
                    "否则每期会清空全部持仓"
                )
        elif class_name == "QuantMLWeightStrategy":
            self._validate_quantml_params(kwargs)

    @staticmethod
    def _validate_quantml_params(kwargs: Dict[str, Any]) -> None:
        """QuantMLWeightStrategy 参数校验。"""
        topk = kwargs.get("topk", 25)
        if not isinstance(topk, int) or topk <= 0:
            raise ValueError(
                f"QuantMLWeightStrategy.topk 必须为正整数，当前: {topk}"
            )
        sq = kwargs.get("score_quantile", 0.0)
        if not isinstance(sq, (int, float)) or not 0.0 <= sq < 1.0:
            raise ValueError(
                f"QuantMLWeightStrategy.score_quantile 必须 ∈ [0, 1)，当前: {sq}"
            )
        rebal = kwargs.get("rebalance_freq", 10)
        if not isinstance(rebal, int) or rebal < 1:
            raise ValueError(
                f"QuantMLWeightStrategy.rebalance_freq 必须为正整数，当前: {rebal}"
            )
        vol_lb = kwargs.get("vol_lookback", 60)
        if not isinstance(vol_lb, int) or vol_lb < 5:
            raise ValueError(
                f"QuantMLWeightStrategy.vol_lookback 必须为 ≥5 的整数，当前: {vol_lb}"
            )
        # weighter_cfg
        wcfg = kwargs.get("weighter_cfg") or {"type": "inverse_vol"}
        if not isinstance(wcfg, dict) or "type" not in wcfg:
            raise ValueError(
                f"QuantMLWeightStrategy.weighter_cfg 必须含 'type' 字段，当前: {wcfg!r}"
            )
        valid_types = {"equal", "score", "inverse_vol", "risk_parity"}
        if wcfg["type"] not in valid_types:
            raise ValueError(
                f"QuantMLWeightStrategy.weighter_cfg.type 必须 ∈ {sorted(valid_types)}，"
                f"当前: {wcfg['type']!r}"
            )
        # risk_cfg
        rcfg = kwargs.get("risk_cfg") or {}
        if not isinstance(rcfg, dict):
            raise ValueError(
                f"QuantMLWeightStrategy.risk_cfg 必须为 dict，当前: {type(rcfg).__name__}"
            )
        n_drop = kwargs.get("n_drop")
        if n_drop is not None and (not isinstance(n_drop, int) or n_drop < 0):
            raise ValueError(
                f"QuantMLWeightStrategy.n_drop 必须为非负整数或省略，当前: {n_drop!r}"
            )
        if isinstance(n_drop, int) and n_drop > 0 and n_drop > topk:
            raise ValueError(
                f"QuantMLWeightStrategy.n_drop({n_drop}) 不应 > topk({topk})，"
                "否则等同于不限制（一次最多卖出 = topk 只持仓）"
            )
        for key, lo, hi in [
            ("max_weight", 0.0, 1.0),
            ("sector_max_weight", 0.0, 1.0),
            ("max_turnover", 0.0, 2.0),
        ]:
            v = rcfg.get(key)
            if v is not None and not (isinstance(v, (int, float)) and lo < v <= hi):
                raise ValueError(
                    f"QuantMLWeightStrategy.risk_cfg.{key} 必须在 ({lo}, {hi}]，"
                    f"当前: {v}"
                )
        er = kwargs.get("exit_rules")
        if er is not None:
            if not isinstance(er, dict):
                raise ValueError(
                    "QuantMLWeightStrategy.exit_rules 必须为 dict 或省略，"
                    f"当前: {type(er).__name__}"
                )
            try:
                parse_exit_rules_cfg(er)
            except ValueError as exc:
                raise ValueError(
                    f"QuantMLWeightStrategy.exit_rules 非法: {exc}"
                ) from exc
