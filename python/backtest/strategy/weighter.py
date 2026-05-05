"""
权重方案层（Weighter）
======================
职责：
  - 为候选股票分配组合权重（sum = 1.0）
  - 提供 4 种可切换方案，统一通过 build_weighter(cfg) 工厂方法构造
  - 不做风险约束（交给 RiskConstraints）

输入：
  - candidates: List[str] 候选股票（来自 Selector）
  - score:      pd.Series  本期截面分（用于 Score-weighted）
  - ret_history: pd.DataFrame
        index = datetime, columns = instrument, value = 日收益率
        用于 InverseVol / RiskParity 计算波动率与协方差

输出：pd.Series（index = candidates，权重和为 1，无 NaN）

四种方案：
  - EqualWeighter:        等权（基线对照）
  - ScoreWeighter:        softmax(score / temperature) 加权
  - InverseVolWeighter:   w_i ∝ 1 / σ_i（基于 lookback 日波动率）
  - RiskParityWeighter:   等风险贡献，迭代求解 w_i × (Σw)_i = const
                          协方差用 Ledoit-Wolf 收缩防奇异

设计权衡：
  - 第一阶段默认 InverseVolWeighter：数学最稳定、不需协方差求逆，
    实证上夏普表现接近 RiskParity 但实现复杂度低一个数量级。
  - RiskParity 数值不稳时退化为 InverseVol（兜底逻辑见类内 _safe_solve）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, runtime_checkable

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 接口定义
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class Weighter(Protocol):
    """权重方案统一协议。"""

    def weight(
        self,
        candidates: List[str],
        score: pd.Series,
        ret_history: pd.DataFrame,
    ) -> pd.Series:
        """返回 index=candidates、和为 1 的权重序列。"""
        ...

    def describe(self) -> str:
        """返回可读描述字符串。"""
        ...


# ─────────────────────────────────────────────────────────────────────────────
# 实现 1: EqualWeighter
# ─────────────────────────────────────────────────────────────────────────────

class EqualWeighter:
    """等权方案：每只候选权重 = 1 / N。基线对照用。"""

    def weight(
        self,
        candidates: List[str],
        score: pd.Series,
        ret_history: pd.DataFrame,
    ) -> pd.Series:
        if not candidates:
            return pd.Series(dtype=float, name="weight")
        n = len(candidates)
        return pd.Series(1.0 / n, index=candidates, name="weight")

    def describe(self) -> str:
        return "EqualWeighter（等权基线）"


# ─────────────────────────────────────────────────────────────────────────────
# 实现 2: ScoreWeighter
# ─────────────────────────────────────────────────────────────────────────────

class ScoreWeighter:
    """
    按预测分加权：w_i ∝ exp(score_i / T) 后归一。

    Args:
        temperature: softmax 温度（越大越接近等权；越小越集中头部）
    """

    def __init__(self, temperature: float = 1.0) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature 必须 > 0，当前: {temperature}")
        self.temperature = float(temperature)

    def weight(
        self,
        candidates: List[str],
        score: pd.Series,
        ret_history: pd.DataFrame,
    ) -> pd.Series:
        if not candidates:
            return pd.Series(dtype=float, name="weight")

        sub = score.reindex(candidates)
        # 缺失分用候选最小分填充，保证后续 softmax 仍能计算
        if sub.isna().any():
            fill = float(sub.min(skipna=True)) if sub.notna().any() else 0.0
            sub = sub.fillna(fill)

        # 数值稳定：减去最大值再做 softmax
        x = sub.values.astype(float) / self.temperature
        x_shift = x - x.max()
        exp = np.exp(x_shift)
        denom = exp.sum()
        if denom <= 0 or not np.isfinite(denom):
            return pd.Series(1.0 / len(candidates), index=candidates, name="weight")
        w = exp / denom
        return pd.Series(w, index=candidates, name="weight")

    def describe(self) -> str:
        return f"ScoreWeighter(softmax, T={self.temperature:.2f})"


# ─────────────────────────────────────────────────────────────────────────────
# 实现 3: InverseVolWeighter
# ─────────────────────────────────────────────────────────────────────────────

class InverseVolWeighter:
    """
    波动率倒数加权：w_i ∝ 1 / σ_i。

    σ_i 取最近 lookback 个交易日的日收益标准差；
    历史不足或全 NaN 时退化为该股票的截面中位数 σ。

    Args:
        lookback: 计算波动率的回看天数（默认 60 ≈ 三个月）
        min_vol: 波动率下限，避免低波动票获得极端权重
    """

    def __init__(self, lookback: int = 60, min_vol: float = 1e-4) -> None:
        if not isinstance(lookback, int) or lookback < 5:
            raise ValueError(f"lookback 必须为 ≥5 的整数，当前: {lookback}")
        if min_vol <= 0:
            raise ValueError(f"min_vol 必须 > 0，当前: {min_vol}")
        self.lookback = lookback
        self.min_vol = float(min_vol)

    def weight(
        self,
        candidates: List[str],
        score: pd.Series,
        ret_history: pd.DataFrame,
    ) -> pd.Series:
        if not candidates:
            return pd.Series(dtype=float, name="weight")

        vols = self._compute_vol(candidates, ret_history)
        # 极端低波动钉到下限，防止单票占满
        vols = vols.clip(lower=self.min_vol)
        inv = 1.0 / vols
        total = inv.sum()
        if total <= 0 or not np.isfinite(total):
            return pd.Series(1.0 / len(candidates), index=candidates, name="weight")
        return (inv / total).rename("weight")

    def _compute_vol(
        self,
        candidates: List[str],
        ret_history: pd.DataFrame,
    ) -> pd.Series:
        """计算每只候选最近 lookback 日的收益率波动率。"""
        if ret_history is None or ret_history.empty:
            # 无历史时全部用 min_vol → 退化为等权
            return pd.Series(self.min_vol, index=candidates)

        recent = ret_history.tail(self.lookback)
        # 仅保留候选列；缺失列用全 NaN 填充
        sub = recent.reindex(columns=candidates)
        vol = sub.std(axis=0, skipna=True)
        # 缺失/NaN 用同期截面中位数兜底
        median_vol = float(vol.dropna().median()) if vol.notna().any() else self.min_vol
        vol = vol.fillna(median_vol)
        return vol

    def describe(self) -> str:
        return f"InverseVolWeighter(lookback={self.lookback}日, min_vol={self.min_vol})"


# ─────────────────────────────────────────────────────────────────────────────
# 实现 4: RiskParityWeighter
# ─────────────────────────────────────────────────────────────────────────────

class RiskParityWeighter:
    """
    等风险贡献（Equal Risk Contribution / Risk Parity）权重方案。

    数学定义：每只股票 i 满足 w_i × (Σw)_i = c（常数），即风险贡献相等。
    迭代解法（cyclic coordinate descent，Spinu 2013）：
      重复 w_i ← (1/σ_i²_{|w}) × (1/N)，其中 σ²_{|w} 为给定其它权重时该股票的边际波动。

    Args:
        lookback: 估计协方差的回看天数（≥10）
        max_iter: 最大迭代次数
        tol: 收敛容差（权重 L1 变化）
        shrink: Ledoit-Wolf 收缩强度（0~1，0 = 样本协方差，1 = 对角阵）。
                自动估计时设为 None；估计失败兜底用 0.1。
    """

    def __init__(
        self,
        lookback: int = 60,
        max_iter: int = 200,
        tol: float = 1e-6,
        shrink: float | None = None,
    ) -> None:
        if not isinstance(lookback, int) or lookback < 10:
            raise ValueError(f"lookback 必须为 ≥10 的整数，当前: {lookback}")
        if max_iter < 1:
            raise ValueError(f"max_iter 必须 ≥1，当前: {max_iter}")
        if tol <= 0:
            raise ValueError(f"tol 必须 > 0，当前: {tol}")
        if shrink is not None and not 0.0 <= shrink <= 1.0:
            raise ValueError(f"shrink 必须在 [0, 1] 或 None，当前: {shrink}")
        self.lookback = lookback
        self.max_iter = max_iter
        self.tol = tol
        self.shrink = shrink
        self._fallback = InverseVolWeighter(lookback=lookback)

    def weight(
        self,
        candidates: List[str],
        score: pd.Series,
        ret_history: pd.DataFrame,
    ) -> pd.Series:
        if not candidates:
            return pd.Series(dtype=float, name="weight")
        if len(candidates) == 1:
            return pd.Series(1.0, index=candidates, name="weight")

        cov = self._estimate_cov(candidates, ret_history)
        if cov is None:
            # 样本严重不足 → 退化为 InverseVol
            return self._fallback.weight(candidates, score, ret_history)

        w = self._solve_rp(cov)
        if w is None or not np.all(np.isfinite(w)) or w.sum() <= 0:
            return self._fallback.weight(candidates, score, ret_history)
        w = w / w.sum()
        return pd.Series(w, index=candidates, name="weight")

    def _estimate_cov(
        self,
        candidates: List[str],
        ret_history: pd.DataFrame,
    ) -> np.ndarray | None:
        """估计候选股票的收益率协方差矩阵（含 Ledoit-Wolf 收缩）。"""
        if ret_history is None or ret_history.empty:
            return None
        recent = ret_history.tail(self.lookback).reindex(columns=candidates)
        # 行内 NaN 用 0 填充（保守做法：缺数据当当日无收益）
        recent = recent.fillna(0.0)
        if recent.shape[0] < 5:
            return None

        sample_cov = np.cov(recent.values, rowvar=False, ddof=1)
        if sample_cov.ndim == 0:  # 单股票
            return np.array([[float(sample_cov)]])

        n = sample_cov.shape[0]
        diag_target = np.diag(np.diag(sample_cov))
        if not np.all(np.isfinite(sample_cov)):
            return None

        alpha = 0.1 if self.shrink is None else self.shrink
        cov = (1.0 - alpha) * sample_cov + alpha * diag_target

        # 数值兜底：加入很小对角扰动，避免严格奇异
        cov = cov + np.eye(n) * 1e-10
        return cov

    def _solve_rp(self, cov: np.ndarray) -> np.ndarray | None:
        """
        Maillard-Roncalli (2010) multiplicative 更新求解 ERC（Equal Risk Contribution）。

        定义：RC_i = w_i × (Σw)_i，目标 RC_i = total_var / N（每只股票贡献相等）。

        迭代公式：
            w_i ← w_i × sqrt(target_share / current_share_i)
            (target_share = 1/N, current_share = RC_i / sum(RC))
        每步后归一化 sum(w)=1，长尾收敛但保证单调下降到不动点。

        参考：Maillard, Roncalli, Teiletche (2010)
            "The Properties of Equally Weighted Risk Contribution Portfolios."
        """
        n = cov.shape[0]
        w = np.full(n, 1.0 / n)
        target_share = 1.0 / n

        for _ in range(self.max_iter):
            w_prev = w.copy()
            marginal = cov @ w           # (Σw)_i
            rc = w * marginal            # RC_i = w_i × (Σw)_i
            total_rc = float(rc.sum())
            if total_rc <= 0 or not np.isfinite(total_rc):
                return None
            rc_share = rc / total_rc     # 各股 RC 占比
            # multiplicative 更新：低贡献的票放大权重，高贡献缩小
            ratio = target_share / np.maximum(rc_share, 1e-12)
            w = w * np.sqrt(ratio)
            w = w / w.sum()
            if np.linalg.norm(w - w_prev, ord=1) < self.tol:
                break

        return w

    def describe(self) -> str:
        shrink_str = "auto(0.1)" if self.shrink is None else f"{self.shrink:.2f}"
        return (
            f"RiskParityWeighter(lookback={self.lookback}日, "
            f"max_iter={self.max_iter}, shrink={shrink_str})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 工厂方法
# ─────────────────────────────────────────────────────────────────────────────

_WEIGHTER_REGISTRY: Dict[str, type] = {
    "equal": EqualWeighter,
    "score": ScoreWeighter,
    "inverse_vol": InverseVolWeighter,
    "risk_parity": RiskParityWeighter,
}


def build_weighter(cfg: Dict[str, Any]) -> Weighter:
    """
    从配置字典构造 Weighter 实例。

    Args:
        cfg: {"type": "inverse_vol", "kwargs": {...}}

    Returns:
        Weighter 实例

    Raises:
        ValueError: 类型未知或参数非法
    """
    if not isinstance(cfg, dict) or "type" not in cfg:
        raise ValueError(
            f"weighter_cfg 必须含 'type' 字段，当前: {cfg!r}"
        )
    weighter_type = str(cfg["type"]).lower()
    if weighter_type not in _WEIGHTER_REGISTRY:
        raise ValueError(
            f"未知 weighter type '{weighter_type}'，"
            f"可选: {sorted(_WEIGHTER_REGISTRY.keys())}"
        )
    cls = _WEIGHTER_REGISTRY[weighter_type]
    kwargs = cfg.get("kwargs") or {}
    if not isinstance(kwargs, dict):
        raise ValueError(
            f"weighter_cfg.kwargs 必须为 dict，当前: {type(kwargs).__name__}"
        )
    return cls(**kwargs)
