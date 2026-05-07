"""
Walk-Forward 切分模块（Rolling Module）
========================================
职责：
  - 定义 FoldSpec 数据类（一个 fold 的 train/valid/test 时间段）
  - build_walk_forward_folds：扩展窗口 + 可配步长，含 purge/embargo 防泄漏
  - infer_purge_days：从 label Qlib 表达式自动推断最小 purge 天数
  - concat_pred_scores：将多 fold 的 OOS pred_score 沿 datetime 拼接

输入输出契约：
  - 日期字符串格式均为 "YYYY-MM-DD"
  - 返回的 FoldSpec 保证 train.end < valid.start ≤ valid.end < test.start ≤ test.end
  - 拼接后的 pred_score 满足 datetime 单调递增且无重复 (datetime, instrument) 对

失败处理：
  - train_min 不满足时（数据不够建第一个 fold）→ 返回空列表（让调用方决定策略）
  - 解析失败的 label 表达式 → infer_purge_days 返回安全默认值 3
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 数据类
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class FoldSpec:
    """一个 walk-forward fold 的完整时间段描述。

    Attributes:
        fold_id:    1-based fold 编号
        train:      (start, end) 字符串，日期含右端点
        valid:      (start, end) 早停验证集
        test:       (start, end) OOS 测试集
        purge_days: 该 fold 实际使用的 purge 天数（纯日历日）
        embargo_days: 该 fold 实际使用的 embargo 天数
    """
    fold_id: int
    train: Tuple[str, str]
    valid: Tuple[str, str]
    test: Tuple[str, str]
    purge_days: int
    embargo_days: int

    @property
    def train_start(self) -> str:
        return self.train[0]

    @property
    def train_end(self) -> str:
        return self.train[1]

    @property
    def valid_start(self) -> str:
        return self.valid[0]

    @property
    def valid_end(self) -> str:
        return self.valid[1]

    @property
    def test_start(self) -> str:
        return self.test[0]

    @property
    def test_end(self) -> str:
        return self.test[1]


# ─────────────────────────────────────────────────────────────────────────────
# 工具：解析 Ny / Nm / Nd 字符串为 DateOffset
# ─────────────────────────────────────────────────────────────────────────────

def _parse_offset(s: str) -> pd.DateOffset:
    """将 "4y" / "6m" / "20d" 等字符串解析为 pd.DateOffset。

    支持格式：
      - "Ny" → N 年（YearEnd 对齐）→ relativedelta(years=N)
      - "Nm" → N 个月
      - "Nd" → N 天（自然日）
    """
    s = s.strip().lower()
    m = re.fullmatch(r"(\d+)([ymd])", s)
    if m is None:
        raise ValueError(f"无法解析时间偏移字符串 {s!r}，合法格式示例：'4y', '6m', '20d'")
    n, unit = int(m.group(1)), m.group(2)
    if unit == "y":
        return pd.DateOffset(years=n)
    if unit == "m":
        return pd.DateOffset(months=n)
    return pd.DateOffset(days=n)


def _fmt(ts: pd.Timestamp) -> str:
    return ts.strftime("%Y-%m-%d")


# ─────────────────────────────────────────────────────────────────────────────
# 核心：build_walk_forward_folds
# ─────────────────────────────────────────────────────────────────────────────

def build_walk_forward_folds(
    *,
    data_start: str,
    data_end: str,
    train_min: str = "4y",
    valid_size: str = "6m",
    test_size: str = "6m",
    step: str = "6m",
    purge_days: int = 2,
    embargo_days: int = 5,
    oos_start: Optional[str] = None,
) -> List[FoldSpec]:
    """生成扩展窗口 Walk-Forward 切分序列。

    Args:
        data_start:   全量数据起始日（训练集固定从此开始）
        data_end:     全量数据截止日
        train_min:    最小训练窗口（如 "4y"）；第一个 fold 的 train 必须 >= 此长度
        valid_size:   每 fold 的验证集长度（如 "6m"）
        test_size:    每 fold 的 OOS 测试集长度（如 "6m"）
        step:         每 fold 向后滑动的步长（如 "6m"），通常 = test_size
        purge_days:   train_end 距 valid_start 的 gap（纯日历日），防 label 泄漏
        embargo_days: purge 后额外 embargo（应对特征自相关）
        oos_start:    可选的 OOS 起始日（如 "2025-01-01"）。设置后，第一个 fold 的
                      test_start 会被抬高到该日期（若该日期晚于默认起点）。

    Returns:
        FoldSpec 列表；若数据不够第一个 fold 则返回 []。

    Fold 时间结构（单个 fold）：
      ╔════════════════════════════╗
      ║ train (expanding window)   ║
      ╚════════════════════════════╝
      gap = purge + embargo 天
      ╔══════════╗
      ║  valid   ║
      ╚══════════╝
      ╔══════════╗
      ║  test    ║ ← OOS
      ╚══════════╝
    """
    ts_start = pd.Timestamp(data_start)
    ts_end = pd.Timestamp(data_end)

    off_train_min = _parse_offset(train_min)
    off_valid = _parse_offset(valid_size)
    off_test = _parse_offset(test_size)
    off_step = _parse_offset(step)
    gap = pd.Timedelta(days=purge_days + embargo_days)

    folds: List[FoldSpec] = []
    fold_id = 1

    # test_start 从 "最小训练 + gap + valid" 开始，每次滑 step
    # 即 fold1.test_start = ts_start + train_min + gap + valid_size
    first_test_start = ts_start + off_train_min + gap + off_valid
    if oos_start:
        first_test_start = max(first_test_start, pd.Timestamp(oos_start))
    test_start = first_test_start

    while True:
        test_end = test_start + off_test - pd.Timedelta(days=1)
        if test_start > ts_end:
            break
        test_end = min(test_end, ts_end)

        valid_end = test_start - pd.Timedelta(days=1)
        valid_start = valid_end - off_valid + pd.Timedelta(days=1)

        train_end = valid_start - gap - pd.Timedelta(days=1)
        # train_start 固定（扩展窗口）
        train_start = ts_start

        if train_end < train_start:
            break

        folds.append(
            FoldSpec(
                fold_id=fold_id,
                train=(_fmt(train_start), _fmt(train_end)),
                valid=(_fmt(valid_start), _fmt(valid_end)),
                test=(_fmt(test_start), _fmt(test_end)),
                purge_days=purge_days,
                embargo_days=embargo_days,
            )
        )
        fold_id += 1
        test_start = test_start + off_step

    return folds


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：从 label 表达式推断 purge_days
# ─────────────────────────────────────────────────────────────────────────────

def infer_purge_days(label_expr: str) -> int:
    """从 Qlib label 表达式推断所需最小 purge 天数。

    解析规则：扫描所有 Ref($xxx, -k) 形式，取最大 |k| 再 +1（安全余量）。
    失败或无匹配时返回保守默认值 3。

    示例：
      "Ref($close_qfq,-2)/Ref($close_qfq,-1)-1" → max(2, 1) + 1 = 3
    """
    try:
        matches = re.findall(r"Ref\s*\([^,]+,\s*(-?\d+)\s*\)", label_expr, re.IGNORECASE)
        if not matches:
            return 3
        max_lag = max(abs(int(k)) for k in matches)
        return max_lag + 1
    except Exception:
        return 3


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：拼接多 fold 的 OOS pred_score
# ─────────────────────────────────────────────────────────────────────────────

def concat_oos_pred_scores(fold_preds: List[pd.Series]) -> pd.Series:
    """将多个 fold 的 OOS pred_score 沿 datetime 轴拼接。

    前置检查：
      - 所有 fold 均为 MultiIndex(datetime, instrument)
      - fold 间 datetime 不允许重叠（walk-forward 保证）
    返回单调递增 datetime 的完整 OOS pred_score。
    """
    if not fold_preds:
        raise ValueError("fold_preds 为空，无法拼接")
    combined = pd.concat(fold_preds)
    combined = combined.sort_index()

    # 检查 (datetime, instrument) 组合重复（同一只股票在同一天被两个 fold 预测）
    if combined.index.duplicated().any():
        n_dup = int(combined.index.duplicated().sum())
        raise ValueError(
            f"拼接后发现 {n_dup} 条重复 (datetime, instrument) 对，"
            "请检查 fold 切分是否存在重叠。"
        )
    combined.name = "score"
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 辅助：将 FoldSpec 列表转为 rolling 摘要
# ─────────────────────────────────────────────────────────────────────────────

def folds_to_summary(
    folds: List[FoldSpec],
    per_fold_metrics: Optional[List[dict]] = None,
) -> dict:
    """将 fold 列表和可选的 per-fold 指标汇总成 rolling_summary.json 结构。

    Args:
        folds:            build_walk_forward_folds 返回的列表
        per_fold_metrics: 每个 fold 的 metrics_dict（含 ic_mean/icir/annualized_return 等）

    Returns:
        rolling_summary dict，可直接 json.dump 写文件
    """
    fold_records = []
    for i, fold in enumerate(folds):
        record: dict = {
            "fold_id": fold.fold_id,
            "train": {"start": fold.train_start, "end": fold.train_end},
            "valid": {"start": fold.valid_start, "end": fold.valid_end},
            "test":  {"start": fold.test_start,  "end": fold.test_end},
            "purge_days": fold.purge_days,
            "embargo_days": fold.embargo_days,
        }
        if per_fold_metrics and i < len(per_fold_metrics):
            record["metrics"] = per_fold_metrics[i]
        fold_records.append(record)

    summary: dict = {
        "fold_count": len(folds),
        "mode": "walk_forward",
        "data_range": {
            "start": folds[0].train_start if folds else None,
            "end":   folds[-1].test_end if folds else None,
        },
        "oos_range": {
            "start": folds[0].test_start if folds else None,
            "end":   folds[-1].test_end if folds else None,
        },
        "folds": fold_records,
    }

    if per_fold_metrics:
        import numpy as np
        key_metrics = ["ic_mean", "icir", "annualized_return", "max_drawdown",
                       "sharpe_ratio", "avg_turnover"]
        stats: dict = {}
        for key in key_metrics:
            vals = [m.get(key) for m in per_fold_metrics if m.get(key) is not None]
            if vals:
                stats[key] = {
                    "mean": float(np.mean(vals)),
                    "std":  float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                    "min":  float(np.min(vals)),
                    "max":  float(np.max(vals)),
                }
        summary["fold_stats"] = stats

    return summary
