"""
信号层（Signal Layer）
======================
职责：
  - 验证模型预测信号的完整性与合法性
  - 计算 IC（信息系数）/ ICIR 评估信号预测能力
  - 输出 SignalSummary 供后续层与分析层使用

输入: pd.Series，MultiIndex(datetime, instrument)，值为预测打分
输出: SignalSummary（含信号本体 + 质量统计）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class SignalSummary:
    """信号质量汇总。"""
    signal: pd.Series
    n_dates: int
    n_instruments: int
    nan_ratio: float
    # IC 指标（仅当传入 label 时有效）
    ic_mean: float = float("nan")
    ic_std: float = float("nan")
    icir: float = float("nan")
    ic_positive_ratio: float = float("nan")
    ic_series: Optional[pd.Series] = None


class SignalLayer:
    """
    信号层：校验预测序列并计算截面 IC 系列。

    IC 计算方法: 每日截面 Spearman 秩相关（预测分 vs 实际标签）
    """

    def validate(self, signal: pd.Series, run_id: str = "unknown") -> None:
        """检查信号合法性，违规则抛异常，警告不终止。"""
        if signal.empty:
            raise ValueError(f"[{run_id}] 预测信号序列为空，请检查模型与数据集配置")
        if not isinstance(signal.index, pd.MultiIndex):
            raise TypeError(
                f"[{run_id}] 预测信号索引必须是 MultiIndex(datetime, instrument)，"
                f"当前为 {type(signal.index).__name__}"
            )
        nan_ratio = float(signal.isna().mean())
        if nan_ratio > 0.5:
            import warnings
            warnings.warn(
                f"[{run_id}] 信号 NaN 占比 {nan_ratio*100:.1f}%（超过 50%），"
                "请检查特征提取与模型输出",
                RuntimeWarning,
                stacklevel=2,
            )

    def compute_ic_series(
        self,
        signal: pd.Series,
        label: pd.Series,
        min_stocks: int = 5,
    ) -> pd.Series:
        """
        逐日计算截面 Spearman IC。

        Args:
            signal: 预测打分序列，MultiIndex(datetime, instrument)
            label:  实际收益标签序列，MultiIndex(datetime, instrument)
            min_stocks: 每日最少有效股票数，低于此则跳过该日

        Returns:
            每日 IC 序列（pd.Series，index=datetime）
        """
        common_idx = signal.index.intersection(label.index)
        if len(common_idx) < min_stocks:
            return pd.Series(dtype=float)

        sig = signal.loc[common_idx].dropna()
        lbl = label.loc[common_idx].dropna()

        dates = sig.index.get_level_values(0).unique().sort_values()
        ic_records: List[Tuple[pd.Timestamp, float]] = []

        for dt in dates:
            try:
                s = sig.xs(dt, level=0)
                l = lbl.xs(dt, level=0)
                common = s.index.intersection(l.index)
                if len(common) < min_stocks:
                    continue
                ic_val = float(
                    s.loc[common].rank().corr(l.loc[common].rank(), method="pearson")
                )
                if not np.isnan(ic_val):
                    ic_records.append((dt, ic_val))
            except Exception:
                continue

        if not ic_records:
            return pd.Series(dtype=float)

        dts, vals = zip(*ic_records)
        return pd.Series(vals, index=pd.DatetimeIndex(dts), name="IC")

    def process(
        self,
        signal: pd.Series,
        label: Optional[pd.Series] = None,
        run_id: str = "unknown",
    ) -> SignalSummary:
        """
        主入口：校验信号并计算质量指标。

        Args:
            signal: 预测打分序列
            label:  实际标签序列（可选，传入时计算 IC/ICIR）
            run_id: 实验 ID，用于日志标识

        Returns:
            SignalSummary
        """
        self.validate(signal, run_id)

        dates = signal.index.get_level_values(0).unique()
        instruments = signal.index.get_level_values(1).unique()
        nan_ratio = float(signal.isna().mean())

        ic_mean = ic_std = icir = ic_pos_ratio = float("nan")
        ic_series: Optional[pd.Series] = None

        if label is not None:
            ic_series = self.compute_ic_series(signal, label)
            if not ic_series.empty:
                arr = ic_series.values
                ic_mean = float(np.nanmean(arr))
                ic_std = float(np.nanstd(arr))
                icir = ic_mean / ic_std if ic_std > 1e-8 else float("nan")
                ic_pos_ratio = float((arr > 0).mean())

        return SignalSummary(
            signal=signal,
            n_dates=len(dates),
            n_instruments=len(instruments),
            nan_ratio=nan_ratio,
            ic_mean=ic_mean,
            ic_std=ic_std,
            icir=icir,
            ic_positive_ratio=ic_pos_ratio,
            ic_series=ic_series,
        )
