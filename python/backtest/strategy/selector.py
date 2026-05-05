"""
候选选择层（Selector）
======================
职责：
  - 在每日截面对预测分 score 进行筛选，输出本期候选股票列表
  - 双重筛选：取头部 topk + 要求分数高于截面分位阈值
  - 不做权重分配（交给 Weighter），不做风险约束（交给 RiskConstraints）

输入：pd.Series（index = instrument，value = score；同一调仓日的截面）
输出：List[str] 候选股票代码，按 score 降序

设计要点：
  - 分位阈值兜底"弱信号日"：若 score 整体偏低（IC 衰减期），即便 topk 仍能纳入，
    也可通过 score_quantile=0.7 强制只在头部 30% 中再选 topk，避免被弱信号牵动调仓。
  - score 含 NaN 的样本自动剔除，不计入分位计算。
"""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


class TopKSelector:
    """
    截面 TopK 选择器。

    Args:
        topk: 目标候选数量（>0）
        score_quantile: 分位阈值（0 = 不过滤；0.7 = 仅在前 30% 中再取 topk）

    Raises:
        ValueError: 参数非法
    """

    def __init__(self, topk: int, score_quantile: float = 0.0) -> None:
        if not isinstance(topk, int) or topk <= 0:
            raise ValueError(f"topk 必须为正整数，当前: {topk}")
        if not 0.0 <= score_quantile < 1.0:
            raise ValueError(
                f"score_quantile 必须在 [0, 1)，当前: {score_quantile}"
            )
        self.topk = topk
        self.score_quantile = score_quantile

    def select(self, score: pd.Series) -> List[str]:
        """
        对单期截面分进行筛选。

        Args:
            score: 当日截面预测分序列，index 为 instrument

        Returns:
            候选股票代码列表（最多 topk 只，按 score 降序）
        """
        if score is None or score.empty:
            return []

        # 1) 剔除 NaN（避免污染 quantile 计算）
        clean = score.dropna()
        if clean.empty:
            return []

        # 2) 分位过滤
        if self.score_quantile > 0.0:
            threshold = float(np.quantile(clean.values, self.score_quantile))
            clean = clean[clean >= threshold]
            if clean.empty:
                return []

        # 3) TopK
        head = clean.sort_values(ascending=False).head(self.topk)
        return [str(idx) for idx in head.index]

    def describe(self) -> str:
        """返回可读描述字符串，供策略层日志输出。"""
        if self.score_quantile > 0.0:
            return (
                f"TopKSelector(topk={self.topk}, "
                f"score_quantile={self.score_quantile:.2f} "
                f"→ 前 {(1 - self.score_quantile) * 100:.0f}% 中取 top{self.topk})"
            )
        return f"TopKSelector(topk={self.topk}, 无分位过滤)"
