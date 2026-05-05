"""
单测：python/backtest/rolling.py
覆盖 fold 切分、purge/embargo 边界、OOS pred_score 拼接。
"""
from __future__ import annotations

import pandas as pd
import pytest

from backtest.rolling import (
    FoldSpec,
    build_walk_forward_folds,
    concat_oos_pred_scores,
    folds_to_summary,
    infer_purge_days,
)


# ─────────────────────────────────────────────────────────────────
# 常量：与计划文档一致的标准参数
# ─────────────────────────────────────────────────────────────────

STD_PARAMS = dict(
    data_start="2020-01-01",
    data_end="2026-04-28",
    train_min="4y",
    valid_size="6m",
    test_size="6m",
    step="6m",
    purge_days=2,
    embargo_days=5,
)


# ─────────────────────────────────────────────────────────────────
# build_walk_forward_folds：切出 4 个 fold
# ─────────────────────────────────────────────────────────────────

class TestBuildFolds:
    def test_standard_params_yield_4_folds(self):
        """标准参数（6 年数据 + 4y train_min + 6m step）应切出 4 个 fold。"""
        folds = build_walk_forward_folds(**STD_PARAMS)
        assert len(folds) == 4

    def test_fold_ids_are_1_based_sequential(self):
        folds = build_walk_forward_folds(**STD_PARAMS)
        assert [f.fold_id for f in folds] == [1, 2, 3, 4]

    def test_train_start_is_fixed(self):
        """扩展窗口：所有 fold 的 train_start 固定在 data_start。"""
        folds = build_walk_forward_folds(**STD_PARAMS)
        for fold in folds:
            assert fold.train_start == "2020-01-01"

    def test_train_end_expands_each_fold(self):
        """每个 fold 的 train_end 应单调递增。"""
        folds = build_walk_forward_folds(**STD_PARAMS)
        ends = [pd.Timestamp(f.train_end) for f in folds]
        assert all(ends[i] < ends[i + 1] for i in range(len(ends) - 1))

    def test_oos_no_overlap(self):
        """相邻 fold 的 test 区间不得重叠。"""
        folds = build_walk_forward_folds(**STD_PARAMS)
        for i in range(len(folds) - 1):
            this_end = pd.Timestamp(folds[i].test_end)
            next_start = pd.Timestamp(folds[i + 1].test_start)
            assert this_end < next_start, (
                f"fold {folds[i].fold_id} test_end={this_end} "
                f">= fold {folds[i+1].fold_id} test_start={next_start}"
            )

    def test_last_fold_test_end_capped_at_data_end(self):
        """最后一个 fold 的 test_end 不超过 data_end。"""
        folds = build_walk_forward_folds(**STD_PARAMS)
        assert pd.Timestamp(folds[-1].test_end) <= pd.Timestamp(STD_PARAMS["data_end"])

    def test_fold1_oos_starts_after_2024_01(self):
        """基于计划文档：fold1 test 应在 2024 年下半年开始。"""
        folds = build_walk_forward_folds(**STD_PARAMS)
        fold1 = folds[0]
        test_start = pd.Timestamp(fold1.test_start)
        assert test_start >= pd.Timestamp("2024-06-01"), (
            f"fold1.test_start={fold1.test_start} 早于预期（应 >= 2024-06-01）"
        )

    def test_insufficient_data_returns_empty(self):
        """数据范围太短（不够 train_min），应返回空列表。"""
        folds = build_walk_forward_folds(
            data_start="2020-01-01",
            data_end="2021-06-30",  # 仅 1.5 年，不够 4y train_min
            train_min="4y",
            valid_size="6m",
            test_size="6m",
            step="6m",
            purge_days=2,
            embargo_days=5,
        )
        assert folds == []

    def test_shorter_data_cuts_fewer_folds(self):
        """数据缩短 3 个月（到 2026-01-28），应只切出 3 个 fold。"""
        folds = build_walk_forward_folds(
            **{**STD_PARAMS, "data_end": "2025-10-31"}
        )
        assert len(folds) == 3

    def test_foldspec_is_frozen(self):
        """FoldSpec 是不可变的（frozen dataclass）。"""
        folds = build_walk_forward_folds(**STD_PARAMS)
        with pytest.raises((AttributeError, TypeError)):
            folds[0].fold_id = 99  # type: ignore[misc]


# ─────────────────────────────────────────────────────────────────
# purge/embargo 正确性
# ─────────────────────────────────────────────────────────────────

class TestPurgeEmbargoGap:
    def test_train_end_before_valid_start_by_gap(self):
        """train_end + gap（purge + embargo 天）< valid_start。"""
        folds = build_walk_forward_folds(**STD_PARAMS)
        gap = STD_PARAMS["purge_days"] + STD_PARAMS["embargo_days"]
        for fold in folds:
            train_end = pd.Timestamp(fold.train_end)
            valid_start = pd.Timestamp(fold.valid_start)
            actual_gap = (valid_start - train_end).days
            assert actual_gap >= gap, (
                f"fold {fold.fold_id}: train_end-valid_start gap={actual_gap}d "
                f"< 期望 {gap}d"
            )

    def test_purge_days_stored_in_foldspec(self):
        folds = build_walk_forward_folds(**STD_PARAMS)
        for fold in folds:
            assert fold.purge_days == STD_PARAMS["purge_days"]
            assert fold.embargo_days == STD_PARAMS["embargo_days"]

    def test_zero_purge_zero_embargo_still_works(self):
        """purge=0, embargo=0 时 train_end 可以紧贴 valid_start - 1 天。"""
        folds = build_walk_forward_folds(**{**STD_PARAMS, "purge_days": 0, "embargo_days": 0})
        assert len(folds) > 0
        fold1 = folds[0]
        train_end = pd.Timestamp(fold1.train_end)
        valid_start = pd.Timestamp(fold1.valid_start)
        assert train_end < valid_start


# ─────────────────────────────────────────────────────────────────
# infer_purge_days
# ─────────────────────────────────────────────────────────────────

class TestInferPurgeDays:
    def test_standard_label(self):
        """标准 label：Ref($close_qfq,-2)/Ref($close_qfq,-1)-1 → max(2,1)+1=3。"""
        expr = "Ref($close_qfq,-2)/Ref($close_qfq,-1)-1"
        assert infer_purge_days(expr) == 3

    def test_single_ref(self):
        """单个 Ref($x,-5)→ 5+1=6。"""
        assert infer_purge_days("Ref($close,-5)") == 6

    def test_no_ref_returns_default(self):
        """无 Ref 表达式时返回保守默认值 3。"""
        assert infer_purge_days("$close/$open - 1") == 3

    def test_invalid_expr_returns_default(self):
        """解析失败时不抛异常，返回 3。"""
        assert infer_purge_days("") == 3
        assert infer_purge_days("garbage_expr") == 3

    def test_case_insensitive(self):
        """Ref 大小写不敏感。"""
        assert infer_purge_days("ref($close,-3)") == 4


# ─────────────────────────────────────────────────────────────────
# concat_oos_pred_scores
# ─────────────────────────────────────────────────────────────────

def _make_pred(dates: list[str], instruments: list[str], base_score: float = 1.0) -> pd.Series:
    """生成测试用 MultiIndex pred_score。"""
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), instruments],
        names=["datetime", "instrument"],
    )
    return pd.Series(base_score, index=idx, name="score")


class TestConcatOosPredScores:
    def test_basic_concat(self):
        """两个不重叠 fold 的 pred 拼接后 datetime 单调递增。"""
        pred1 = _make_pred(["2024-07-01", "2024-07-02"], ["A.SZ", "B.SZ"])
        pred2 = _make_pred(["2025-01-01", "2025-01-02"], ["A.SZ", "B.SZ"])
        result = concat_oos_pred_scores([pred1, pred2])
        datetimes = result.index.get_level_values("datetime")
        assert list(datetimes) == sorted(datetimes), "拼接后 datetime 应单调递增"

    def test_no_duplicate_datetime_instrument(self):
        """拼接后不应有重复 (datetime, instrument) 对。"""
        pred1 = _make_pred(["2024-07-01"], ["A.SZ"])
        pred2 = _make_pred(["2025-01-01"], ["B.SZ"])
        result = concat_oos_pred_scores([pred1, pred2])
        assert not result.index.duplicated().any()

    def test_overlapping_datetime_raises(self):
        """两个 fold 同一只股票出现在相同 datetime 时应抛出 ValueError。"""
        pred1 = _make_pred(["2024-07-01", "2024-07-02"], ["A.SZ"])
        pred2 = _make_pred(["2024-07-02", "2024-07-03"], ["A.SZ"])
        with pytest.raises(ValueError, match="重复"):
            concat_oos_pred_scores([pred1, pred2])

    def test_empty_list_raises(self):
        with pytest.raises(ValueError):
            concat_oos_pred_scores([])

    def test_result_name_is_score(self):
        pred = _make_pred(["2024-07-01"], ["A.SZ"])
        result = concat_oos_pred_scores([pred])
        assert result.name == "score"

    def test_four_fold_concat_covers_oos(self):
        """模拟 4 个 fold 拼接，检查覆盖范围。"""
        fold_preds = [
            _make_pred(["2024-07-01"], ["A.SZ"]),
            _make_pred(["2025-01-01"], ["A.SZ"]),
            _make_pred(["2025-07-01"], ["A.SZ"]),
            _make_pred(["2026-01-01"], ["A.SZ"]),
        ]
        result = concat_oos_pred_scores(fold_preds)
        datetimes = result.index.get_level_values("datetime")
        assert datetimes.min() == pd.Timestamp("2024-07-01")
        assert datetimes.max() == pd.Timestamp("2026-01-01")


# ─────────────────────────────────────────────────────────────────
# folds_to_summary
# ─────────────────────────────────────────────────────────────────

class TestFoldsToSummary:
    def test_basic_structure(self):
        folds = build_walk_forward_folds(**STD_PARAMS)
        summary = folds_to_summary(folds)
        assert summary["fold_count"] == len(folds)
        assert summary["mode"] == "walk_forward"
        assert "folds" in summary
        assert len(summary["folds"]) == len(folds)

    def test_oos_range_correct(self):
        folds = build_walk_forward_folds(**STD_PARAMS)
        summary = folds_to_summary(folds)
        assert summary["oos_range"]["start"] == folds[0].test_start
        assert summary["oos_range"]["end"] == folds[-1].test_end

    def test_with_per_fold_metrics_computes_stats(self):
        folds = build_walk_forward_folds(**STD_PARAMS)
        per_fold = [
            {"ic_mean": 0.05, "icir": 0.5, "annualized_return": 0.12},
            {"ic_mean": 0.04, "icir": 0.4, "annualized_return": 0.08},
            {"ic_mean": 0.06, "icir": 0.6, "annualized_return": 0.15},
            {"ic_mean": 0.03, "icir": 0.3, "annualized_return": 0.05},
        ]
        summary = folds_to_summary(folds, per_fold_metrics=per_fold)
        assert "fold_stats" in summary
        stats = summary["fold_stats"]
        assert "ic_mean" in stats
        import numpy as np
        assert abs(stats["ic_mean"]["mean"] - float(np.mean([0.05, 0.04, 0.06, 0.03]))) < 1e-9

    def test_foldspec_properties(self):
        """验证 FoldSpec 的属性快捷方式。"""
        folds = build_walk_forward_folds(**STD_PARAMS)
        fold1 = folds[0]
        assert fold1.train_start == fold1.train[0]
        assert fold1.train_end == fold1.train[1]
        assert fold1.valid_start == fold1.valid[0]
        assert fold1.test_end == fold1.test[1]
