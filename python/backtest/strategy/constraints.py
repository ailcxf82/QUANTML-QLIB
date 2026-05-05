"""
风险约束层（Risk Constraints）
================================
职责：
  - 对 Weighter 输出的目标权重序列施加顺序约束：
      ① 单票上限（max_weight）：超出按比例截断，余量按比例放大其他票
      ② 板块/行业上限（sector_max_weight）：板块总权重超限按板块内部按比例缩放
      ③ 换手率上限（max_turnover）：当 ‖w_target − w_prev‖₁ > 上限时，
          线性插值 w' = w_prev + α(w_target − w_prev)，求最大 α 使换手 ≤ 上限
  - 输出最终归一化权重序列（sum = 1）

设计取舍：
  - 三层约束顺序串联（pipeline 风格），简单可解释；
    不采用 QP 求解器（需 cvxpy 等重依赖），中频场景 25 票内 pipeline 已够稳定。
  - 板块映射默认复用 trade_layer.classify_market（基于代码前缀 5 类粗粒度），
    可通过 sector_provider 注入申万一级（28 类）等更细粒度映射。
  - 换手率定义：sum_i |w_i_new − w_i_prev| / 2（双边），与 qlib 一致；
    未持有过的票视为 prev 权重 0。

输入：
  - target_w: pd.Series，候选股票目标权重（来自 Weighter）
  - prev_w:   pd.Series，上期组合权重（来自策略状态，无持仓时传 None 或空）

输出：pd.Series，调整后权重，sum = 1
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pandas as pd

# 复用 trade_layer 中的板块分类函数，避免重复实现
from ..trade_layer import classify_market


class RiskConstraints:
    """
    顺序约束管线：单票 → 板块 → 换手 → 归一化。

    Args:
        max_weight: 单票权重上限（0~1，None 表示不约束）
        sector_max_weight: 单板块/行业总权重上限（0~1，None 表示不约束）
        max_turnover: 单期换手上限（0~1，None 表示不约束；qlib 双边定义）
        sector_provider: 板块查询函数 (stock_id -> sector_name)，
                         默认用 classify_market（板块粗粒度 5 类）
    """

    def __init__(
        self,
        max_weight: Optional[float] = 0.08,
        sector_max_weight: Optional[float] = 0.30,
        max_turnover: Optional[float] = 0.30,
        sector_provider: Optional[Callable[[str], str]] = None,
    ) -> None:
        if max_weight is not None and not 0 < max_weight <= 1:
            raise ValueError(f"max_weight 必须在 (0, 1]，当前: {max_weight}")
        if sector_max_weight is not None and not 0 < sector_max_weight <= 1:
            raise ValueError(
                f"sector_max_weight 必须在 (0, 1]，当前: {sector_max_weight}"
            )
        if max_turnover is not None and not 0 < max_turnover <= 2.0:
            raise ValueError(
                f"max_turnover 必须在 (0, 2.0]，当前: {max_turnover}"
            )
        self.max_weight = max_weight
        self.sector_max_weight = sector_max_weight
        self.max_turnover = max_turnover
        self.sector_provider = sector_provider or classify_market

    # ─────────────────────────────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────────────────────────────

    def apply(
        self,
        target_w: pd.Series,
        prev_w: Optional[pd.Series] = None,
    ) -> pd.Series:
        """
        顺序应用三层约束。

        Args:
            target_w: 目标权重（候选股票，来自 Weighter）
            prev_w:   上期权重（可为 None / 空 / 含 target_w 之外的标的）

        Returns:
            调整后权重序列，sum = 1（除非 target_w 为空）
        """
        if target_w is None or target_w.empty:
            return pd.Series(dtype=float, name="weight")

        w = target_w.astype(float).copy()
        # 0. 先归一化（防御 Weighter 偶尔不归一）
        s = float(w.sum())
        if s <= 0 or not np.isfinite(s):
            return pd.Series(dtype=float, name="weight")
        w = w / s

        # 1. 单票上限
        if self.max_weight is not None:
            w = self._cap_max_weight(w, self.max_weight)

        # 2. 板块上限
        if self.sector_max_weight is not None:
            w = self._cap_sector_weight(w, self.sector_max_weight)

        # 3. 换手率约束（基于"已就位"权重 vs 上期）
        if self.max_turnover is not None and prev_w is not None and not prev_w.empty:
            w = self._cap_turnover(w, prev_w, self.max_turnover)

        # 4. 终极归一化（防御浮点误差累计）
        s = float(w.sum())
        if s > 0:
            w = w / s
        return w.rename("weight")

    # ─────────────────────────────────────────────────────────────────────
    # 内部约束算子
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _cap_max_weight(w: pd.Series, cap: float) -> pd.Series:
        """
        反复迭代：单票截断 → 余量按未截断票的相对权重重新分配。
        最多 50 次循环（数学上一定收敛，因为每次截断单票数严格增加）。
        """
        w = w.copy()
        for _ in range(50):
            over = w > cap + 1e-12
            if not over.any():
                break
            # 把超额部分截到 cap，剩余 budget 按未截断票的现权重比例分配
            excess = float((w[over] - cap).sum())
            w[over] = cap
            below = ~over
            if below.sum() == 0 or w[below].sum() <= 0:
                # 全部都到 cap 了，无处可分，留作 cash（sum < 1）
                break
            w[below] = w[below] + excess * (w[below] / w[below].sum())
        return w

    def _cap_sector_weight(self, w: pd.Series, cap: float) -> pd.Series:
        """
        板块总权重不超过 cap：
          1. 计算每个板块的总权重 g_s
          2. 若 g_s > cap：把该板块所有票按 (cap / g_s) 等比缩放
          3. 缩出来的余量按未超限板块的现权重比例放回（保持 sum=1）
          4. 最多迭代 10 次防发散（板块数有限）
        """
        w = w.copy()
        sectors = pd.Series(
            {idx: self.sector_provider(idx) for idx in w.index},
            name="sector",
        )

        for _ in range(10):
            grouped = w.groupby(sectors)
            sector_w = grouped.sum()
            over = sector_w > cap + 1e-12
            if not over.any():
                break

            excess_total = 0.0
            for sec in sector_w[over].index:
                members = sectors[sectors == sec].index
                gs = float(sector_w[sec])
                scale = cap / gs
                scaled = w.loc[members] * scale
                excess_total += float((w.loc[members] - scaled).sum())
                w.loc[members] = scaled

            # 余量按未超限板块的现权重重新分配
            under = ~sectors.isin(sector_w[over].index)
            under_idx = w.index[under.values]
            under_total = float(w.loc[under_idx].sum())
            if excess_total > 0 and under_total > 0:
                w.loc[under_idx] = w.loc[under_idx] + excess_total * (
                    w.loc[under_idx] / under_total
                )
            else:
                # 没有可放的位置，留作 cash（sum < 1）
                break

        return w

    @staticmethod
    def _cap_turnover(
        w_target: pd.Series,
        prev_w: pd.Series,
        cap: float,
    ) -> pd.Series:
        """
        若 target 与 prev 的双边换手率 ≤ cap，原样返回 target；
        否则线性插值 w' = prev + α(target − prev)，求 α = cap / full_to 使换手 = cap。

        关键语义：
          - 输出权重序列定义在 target ∪ prev 的并集上
          - prev 中但不在 target 的旧持仓获得 (1-α)×prev_w 的残余权重
            （自然降仓，避免被强制清出造成"虚换手"）
          - target 中但不在 prev 的新仓获得 α×target_w
        换手率定义：sum |w_new − w_prev| / 2（双边，与 qlib 一致）。
        """
        union_idx = w_target.index.union(prev_w.index)
        t_full = w_target.reindex(union_idx).fillna(0.0)
        p_full = prev_w.reindex(union_idx).fillna(0.0)
        # 防御：prev 归一化
        p_sum = float(p_full.sum())
        if p_sum > 0:
            p_full = p_full / p_sum

        delta = t_full - p_full
        full_to = float(delta.abs().sum()) / 2.0
        if full_to <= cap + 1e-12:
            return w_target

        alpha = cap / full_to if full_to > 0 else 0.0
        alpha = float(np.clip(alpha, 0.0, 1.0))
        w_new = p_full + alpha * delta
        # 返回 union 域权重（含残余持仓），由上层在最终归一化前决定
        # 残余票留在结果里 → Qlib 会按权重生成"卖出 prev 残余 + 买入 target 部分"订单
        return w_new[w_new > 1e-9]

    # ─────────────────────────────────────────────────────────────────────
    # 描述
    # ─────────────────────────────────────────────────────────────────────

    def describe(self) -> str:
        parts = []
        if self.max_weight is not None:
            parts.append(f"单票≤{self.max_weight*100:.1f}%")
        if self.sector_max_weight is not None:
            parts.append(f"板块≤{self.sector_max_weight*100:.0f}%")
        if self.max_turnover is not None:
            parts.append(f"换手≤{self.max_turnover*100:.0f}%/期")
        return "RiskConstraints(" + ", ".join(parts) + ")"
