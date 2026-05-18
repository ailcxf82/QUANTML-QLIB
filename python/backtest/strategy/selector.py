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
  - 可选 AdaptiveTopKCfg：根据每日截面信号强度动态调整 topk 和 score_quantile。
    强信号日集中持仓（小 topk），弱信号日分散（大 topk），降低噪声票的 σ_idio 贡献。
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np
import pandas as pd

_logger = logging.getLogger("TopKSelector")


# ─────────────────────────────────────────────────────────────────────────────
# 自适应 TopK 配置
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AdaptiveTopKCfg:
    """
    信号强度自适应 TopK 配置。

    信号强度定义（每日截面）：
        raw_strength = (mean(top_q scores) - median(all scores)) / max(std(all scores), ε)

    与最近 strength_window 期的滚动历史比较，计算 z-score：
        z = (raw_strength - rolling_mean) / max(rolling_std, ε)

    触发规则：
        z > strength_high → 强信号日：topk=strong_topk, quantile=strong_quantile
        z < strength_low  → 弱信号日：topk=weak_topk,   quantile=weak_quantile
        否则              → 默认：使用 TopKSelector 初始化时的 topk / score_quantile

    冷启动（历史不足 cold_start_min 期）：回退到默认，不做自适应。

    Args:
        strong_topk:      强信号日持仓数（更集中，通常 < 默认 topk）
        weak_topk:        弱信号日持仓数（更分散，通常 > 默认 topk）
        strong_quantile:  强信号日截面分位阈值（更严格）
        weak_quantile:    弱信号日截面分位阈值（更宽松）
        strength_high:    触发"强信号"的 z-score 阈值（建议 0.5~1.5）
        strength_low:     触发"弱信号"的 z-score 阈值（建议 -1.0~0.0）
        strength_window:  滚动历史窗口大小（建议 10~30）
        cold_start_min:   冷启动保护：历史不足此值时回退默认
    """
    strong_topk: int = 3
    weak_topk: int = 8
    strong_quantile: float = 0.85
    weak_quantile: float = 0.60
    strength_high: float = 1.0
    strength_low: float = -0.3
    strength_window: int = 20
    cold_start_min: int = 5


# ─────────────────────────────────────────────────────────────────────────────
# TopKSelector
# ─────────────────────────────────────────────────────────────────────────────

class TopKSelector:
    """
    截面 TopK 选择器，支持可选的信号强度自适应模式。

    Args:
        topk: 默认候选数量（>0）
        score_quantile: 默认分位阈值（0 = 不过滤；0.7 = 仅在前 30% 中再取 topk）
        adaptive_topk_cfg: 可选自适应配置（None = 固定 topk，不做自适应）

    Raises:
        ValueError: 参数非法
    """

    def __init__(
        self,
        topk: int,
        score_quantile: float = 0.0,
        adaptive_topk_cfg: Optional[AdaptiveTopKCfg] = None,
    ) -> None:
        if not isinstance(topk, int) or topk <= 0:
            raise ValueError(f"topk 必须为正整数，当前: {topk}")
        if not 0.0 <= score_quantile < 1.0:
            raise ValueError(
                f"score_quantile 必须在 [0, 1)，当前: {score_quantile}"
            )
        self.topk = topk
        self.score_quantile = score_quantile
        self._adaptive: Optional[AdaptiveTopKCfg] = adaptive_topk_cfg

        # 滚动强度历史（用于 z-score 计算）
        window = adaptive_topk_cfg.strength_window if adaptive_topk_cfg else 1
        self._strength_history: Deque[float] = deque(maxlen=window)

    def select(self, score: pd.Series) -> List[str]:
        """
        对单期截面分进行筛选。

        若配置了 AdaptiveTopKCfg，会先计算截面信号强度，动态决定本期 topk 和
        score_quantile，再执行标准双重筛选流程。

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

        # 2) 确定本期有效 topk / score_quantile
        eff_topk, eff_quantile = self._resolve_topk_quantile(clean)

        # 3) 分位过滤
        if eff_quantile > 0.0:
            threshold = float(np.quantile(clean.values, eff_quantile))
            clean = clean[clean >= threshold]
            if clean.empty:
                return []

        # 4) TopK
        head = clean.sort_values(ascending=False).head(eff_topk)
        return [str(idx) for idx in head.index]

    def _resolve_topk_quantile(self, clean: pd.Series) -> tuple[int, float]:
        """
        根据自适应配置决定本期 topk 和 score_quantile。

        同时将本期 raw_strength 推入滚动历史（副作用，保证调用顺序一致）。

        Returns:
            (eff_topk, eff_quantile)
        """
        if self._adaptive is None:
            return self.topk, self.score_quantile

        cfg = self._adaptive

        # 计算本期截面信号强度：(mean(top-K) - mean(rest)) / std
        # 直接比较 top-topk 与其余股票的均值差距，对"少数明显优质"的截面更敏感。
        # 这比 top_q%-median 更精准：后者对极端值不敏感，本指标直接测量 topK 的相对优势。
        n = len(clean)
        if n < 2:
            self._strength_history.append(0.0)
            return self.topk, self.score_quantile

        sorted_scores = clean.sort_values(ascending=False)
        effective_topk = min(self.topk, n - 1)
        top_mean = float(sorted_scores.iloc[:effective_topk].mean())
        rest_mean = float(sorted_scores.iloc[effective_topk:].mean())
        std = float(clean.std(ddof=1))
        raw_strength = (top_mean - rest_mean) / max(std, 1e-8)

        # z-score 基于历史（不含本期），避免本期值拉低自身 z-score
        hist_len = len(self._strength_history)
        if hist_len < cfg.cold_start_min:
            # 冷启动：历史不足，先积累历史再做自适应
            self._strength_history.append(raw_strength)
            _logger.debug(
                "[AdaptiveTopK] 冷启动（%d/%d），使用默认 topk=%d q=%.2f",
                hist_len + 1, cfg.cold_start_min, self.topk, self.score_quantile,
            )
            return self.topk, self.score_quantile

        hist_arr = np.array(self._strength_history)
        rolling_mean = float(hist_arr.mean())
        rolling_std = float(hist_arr.std(ddof=1)) if hist_len > 1 else 0.0
        z = (raw_strength - rolling_mean) / max(rolling_std, 1e-8)

        # 本期值在 z-score 计算后再 append，保证下一期历史是最新的
        self._strength_history.append(raw_strength)

        if z > cfg.strength_high:
            eff_topk, eff_q = cfg.strong_topk, cfg.strong_quantile
            regime = "strong"
        elif z < cfg.strength_low:
            eff_topk, eff_q = cfg.weak_topk, cfg.weak_quantile
            regime = "weak"
        else:
            eff_topk, eff_q = self.topk, self.score_quantile
            regime = "normal"

        _logger.debug(
            "[AdaptiveTopK] raw_strength=%.3f z=%.3f regime=%s topk=%d q=%.2f",
            raw_strength, z, regime, eff_topk, eff_q,
        )
        return eff_topk, eff_q

    def reset_history(self) -> None:
        """清空滚动强度历史（跨回测实例复用时调用）。"""
        self._strength_history.clear()

    def describe(self) -> str:
        """返回可读描述字符串，供策略层日志输出。"""
        base = (
            f"TopKSelector(topk={self.topk}, "
            f"score_quantile={self.score_quantile:.2f} "
            f"→ 前 {(1 - self.score_quantile) * 100:.0f}% 中取 top{self.topk})"
            if self.score_quantile > 0.0
            else f"TopKSelector(topk={self.topk}, 无分位过滤)"
        )
        if self._adaptive is not None:
            cfg = self._adaptive
            base += (
                f" [自适应: 强信号→topk={cfg.strong_topk}/q={cfg.strong_quantile:.2f}"
                f"(z>{cfg.strength_high}), 弱信号→topk={cfg.weak_topk}/q={cfg.weak_quantile:.2f}"
                f"(z<{cfg.strength_low}), window={cfg.strength_window}]"
            )
        return base
