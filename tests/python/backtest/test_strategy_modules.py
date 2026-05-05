"""
策略子包单元测试（Strategy Sub-package Unit Tests）
=====================================================
覆盖：
  - TopKSelector：基础 / 分位过滤 / NaN 处理 / 参数校验
  - EqualWeighter / ScoreWeighter / InverseVolWeighter / RiskParityWeighter
    的权重正确性、归一化、退化场景
  - RiskConstraints：单票上限、板块上限、换手控制、顺序串联
  - QuantMLWeightStrategy：调仓节流、exit_rules 非模型日行为
  - exit_rules：止盈 / 止损 / 移动止损纯逻辑
  - StrategyLayer：QuantMLWeightStrategy 参数校验

执行：
  pytest tests/python/backtest/test_strategy_modules.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.strategy import (
    TopKSelector,
    EqualWeighter,
    ScoreWeighter,
    InverseVolWeighter,
    RiskParityWeighter,
    RiskConstraints,
    QuantMLWeightStrategy,
    build_weighter,
)
from backtest.strategy.exit_rules import (
    ExitRulesConfig,
    evaluate_exit_triggers,
    parse_exit_rules_cfg,
    renormalize_target,
    strip_exited_symbols,
)


# ─────────────────────────────────────────────────────────────────────────────
# Exit rules（止盈止损纯逻辑）
# ─────────────────────────────────────────────────────────────────────────────


class TestExitRules:
    def test_evaluate_stop_loss(self) -> None:
        cfg = ExitRulesConfig(stop_loss_pct=0.10)
        ok, reason = evaluate_exit_triggers(
            mark=89.0, entry_ref=100.0, peak=100.0, cfg=cfg
        )
        assert ok and reason == "stop_loss"

    def test_evaluate_take_profit(self) -> None:
        cfg = ExitRulesConfig(take_profit_pct=0.20)
        ok, reason = evaluate_exit_triggers(
            mark=121.0, entry_ref=100.0, peak=121.0, cfg=cfg
        )
        assert ok and reason == "take_profit"

    def test_evaluate_trailing(self) -> None:
        cfg = ExitRulesConfig(trailing_stop_pct=0.10)
        ok, reason = evaluate_exit_triggers(
            mark=85.0, entry_ref=80.0, peak=100.0, cfg=cfg
        )
        assert ok and reason == "trailing_stop"

    def test_strip_and_renormalize(self) -> None:
        cfg = ExitRulesConfig(stop_loss_pct=0.10)
        target: dict[str, float] = {"A": 0.5, "B": 0.5}
        entry = {"A": 100.0, "B": 100.0}
        peak = {"A": 100.0, "B": 100.0}
        marks = {"A": 85.0, "B": 100.0}
        closed = strip_exited_symbols(
            target, ["A", "B"], marks, entry, peak, cfg
        )
        assert [e["code"] for e in closed] == ["A"]
        assert closed[0]["reason"] == "stop_loss"
        assert closed[0]["pnl_pct"] == pytest.approx(-0.15)
        out = renormalize_target(target)
        assert pytest.approx(out["B"]) == 1.0
        assert "A" not in entry

    def test_parse_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_exit_rules_cfg({"stop_loss_pct": 0.0})


# ─────────────────────────────────────────────────────────────────────────────
# TopKSelector
# ─────────────────────────────────────────────────────────────────────────────

class TestTopKSelector:
    """TopKSelector 的截面筛选行为。"""

    def test_select_topk_basic(self) -> None:
        score = pd.Series(
            [0.9, 0.5, 0.7, 0.1, 0.3],
            index=["A", "B", "C", "D", "E"],
        )
        selector = TopKSelector(topk=3)
        result = selector.select(score)
        assert result == ["A", "C", "B"]

    def test_select_with_quantile_filter(self) -> None:
        score = pd.Series(
            [0.9, 0.5, 0.7, 0.1, 0.3],
            index=["A", "B", "C", "D", "E"],
        )
        # np.quantile([0.1, 0.3, 0.5, 0.7, 0.9], 0.6) = 0.58（线性插值）
        # >= 0.58 的只有 0.7 和 0.9 → ["A", "C"]
        selector = TopKSelector(topk=5, score_quantile=0.6)
        result = selector.select(score)
        assert set(result) == {"A", "C"}
        assert result[0] == "A"  # 排序保持降序

    def test_select_drops_nan(self) -> None:
        score = pd.Series(
            [0.9, np.nan, 0.7, np.nan, 0.3],
            index=["A", "B", "C", "D", "E"],
        )
        selector = TopKSelector(topk=5)
        result = selector.select(score)
        assert result == ["A", "C", "E"]

    def test_select_empty_returns_empty(self) -> None:
        selector = TopKSelector(topk=10)
        assert selector.select(pd.Series(dtype=float)) == []
        assert selector.select(None) == []

    def test_select_topk_smaller_than_n(self) -> None:
        score = pd.Series([0.1, 0.2, 0.3], index=["A", "B", "C"])
        selector = TopKSelector(topk=10)
        # 只有 3 只可选，全部返回
        assert set(selector.select(score)) == {"A", "B", "C"}

    @pytest.mark.parametrize("bad_topk", [0, -1, 1.5, "5"])
    def test_invalid_topk(self, bad_topk) -> None:
        with pytest.raises(ValueError, match="topk"):
            TopKSelector(topk=bad_topk)  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_q", [-0.1, 1.0, 1.5])
    def test_invalid_quantile(self, bad_q: float) -> None:
        with pytest.raises(ValueError, match="score_quantile"):
            TopKSelector(topk=5, score_quantile=bad_q)


# ─────────────────────────────────────────────────────────────────────────────
# EqualWeighter
# ─────────────────────────────────────────────────────────────────────────────

class TestEqualWeighter:
    def test_equal_weight_sums_to_one(self) -> None:
        w = EqualWeighter().weight(
            candidates=["A", "B", "C", "D"],
            score=pd.Series(dtype=float),
            ret_history=pd.DataFrame(),
        )
        assert len(w) == 4
        assert pytest.approx(w.sum()) == 1.0
        assert all(pytest.approx(v) == 0.25 for v in w.values)

    def test_empty_candidates(self) -> None:
        w = EqualWeighter().weight([], pd.Series(dtype=float), pd.DataFrame())
        assert w.empty


# ─────────────────────────────────────────────────────────────────────────────
# ScoreWeighter
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreWeighter:
    def test_softmax_higher_score_higher_weight(self) -> None:
        score = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])
        w = ScoreWeighter(temperature=1.0).weight(
            ["A", "B", "C"], score, pd.DataFrame()
        )
        assert pytest.approx(w.sum()) == 1.0
        assert w["C"] > w["B"] > w["A"]

    def test_temperature_higher_more_uniform(self) -> None:
        score = pd.Series([1.0, 2.0, 3.0], index=["A", "B", "C"])
        w_hot = ScoreWeighter(temperature=10.0).weight(
            ["A", "B", "C"], score, pd.DataFrame()
        )
        w_cold = ScoreWeighter(temperature=0.1).weight(
            ["A", "B", "C"], score, pd.DataFrame()
        )
        # 高温 → 更接近等权（max 权重小）；低温 → 集中头部（max 权重大）
        assert w_hot.max() < w_cold.max()

    def test_invalid_temperature(self) -> None:
        with pytest.raises(ValueError):
            ScoreWeighter(temperature=0.0)
        with pytest.raises(ValueError):
            ScoreWeighter(temperature=-1.0)


# ─────────────────────────────────────────────────────────────────────────────
# InverseVolWeighter
# ─────────────────────────────────────────────────────────────────────────────

class TestInverseVolWeighter:
    def _make_history(
        self,
        vols: dict[str, float],
        n_days: int = 100,
        seed: int = 42,
    ) -> pd.DataFrame:
        """构造各股票指定波动率的历史日收益序列。"""
        rng = np.random.default_rng(seed)
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
        data = {
            stock: rng.normal(loc=0.0, scale=vol, size=n_days)
            for stock, vol in vols.items()
        }
        return pd.DataFrame(data, index=dates)

    def test_low_vol_gets_higher_weight(self) -> None:
        history = self._make_history(
            {"LowVol": 0.005, "MidVol": 0.020, "HighVol": 0.050},
            n_days=120,
        )
        w = InverseVolWeighter(lookback=60).weight(
            ["LowVol", "MidVol", "HighVol"],
            pd.Series(dtype=float),
            history,
        )
        assert pytest.approx(w.sum()) == 1.0
        assert w["LowVol"] > w["MidVol"] > w["HighVol"]
        # 数学上 w 比例约 ~ (1/0.005) : (1/0.020) : (1/0.050) = 200 : 50 : 20
        # → 比值 ~ 0.741 : 0.185 : 0.074
        assert w["LowVol"] > 0.6
        assert w["HighVol"] < 0.15

    def test_min_vol_clipping(self) -> None:
        # 三只股票波动率全部低于 min_vol，应退化为等权
        history = self._make_history(
            {"A": 1e-6, "B": 1e-6, "C": 1e-6}, n_days=100
        )
        w = InverseVolWeighter(lookback=60, min_vol=1e-3).weight(
            ["A", "B", "C"], pd.Series(dtype=float), history
        )
        assert pytest.approx(w.sum()) == 1.0
        for v in w.values:
            assert pytest.approx(v, abs=1e-9) == 1.0 / 3

    def test_missing_history_falls_back_to_equal(self) -> None:
        w = InverseVolWeighter(lookback=60).weight(
            ["A", "B"], pd.Series(dtype=float), pd.DataFrame()
        )
        assert pytest.approx(w.sum()) == 1.0
        assert pytest.approx(w["A"]) == 0.5

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError):
            InverseVolWeighter(lookback=2)
        with pytest.raises(ValueError):
            InverseVolWeighter(lookback=60, min_vol=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# RiskParityWeighter
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskParityWeighter:
    def test_diagonal_cov_matches_inverse_vol(self) -> None:
        """对角协方差时，RP 权重排序与 InverseVol 一致；样本噪声允许微小偏移。"""
        rng = np.random.default_rng(0)
        n_days = 600
        dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
        vols = {"A": 0.005, "B": 0.020, "C": 0.050}
        # 独立生成 → 总体协方差对角；600 样本下 off-diag 估计 ≈ 0
        history = pd.DataFrame(
            {s: rng.normal(0, v, n_days) for s, v in vols.items()},
            index=dates,
        )
        # 关闭 LW 收缩消除其偏移；只看求解器本身
        w_rp = RiskParityWeighter(
            lookback=n_days, max_iter=2000, tol=1e-9, shrink=0.0
        ).weight(["A", "B", "C"], pd.Series(dtype=float), history)
        w_iv = InverseVolWeighter(lookback=n_days).weight(
            ["A", "B", "C"], pd.Series(dtype=float), history
        )
        assert pytest.approx(w_rp.sum()) == 1.0
        # 排序应一致：低波动 → 高权重
        assert w_rp["A"] > w_rp["B"] > w_rp["C"]
        # 600 样本 + 0 收缩，数值差应 < 3%
        for k in ["A", "B", "C"]:
            assert abs(w_rp[k] - w_iv[k]) < 0.03

    def test_equal_risk_contribution(self) -> None:
        """风险贡献等于 total_var/N（误差 < 5%）。

        构造方式：所有股票被一个共同因子 F 驱动 + 各自特异波动，保证
        协方差矩阵全部非负元素（A 股长仓组合常见场景），long-only ERC
        有良好定义解。
        """
        rng = np.random.default_rng(1)
        n_stocks = 5
        n_days = 800
        # 共同因子 + 个股特异波动；vol 跨度 1x ~ 5x
        common = rng.normal(0, 0.012, n_days)
        idio_vols = np.array([0.005, 0.008, 0.012, 0.020, 0.025])
        rets = (
            common[:, None]
            + rng.normal(0, 1, (n_days, n_stocks)) * idio_vols[None, :]
        )
        cols = [f"S{i}" for i in range(n_stocks)]
        history = pd.DataFrame(rets, columns=cols)

        w = RiskParityWeighter(
            lookback=n_days, max_iter=5000, tol=1e-10, shrink=0.0
        ).weight(cols, pd.Series(dtype=float), history)

        # 用估计协方差验证 RC 是否等
        cov = np.cov(history.values, rowvar=False, ddof=1)
        marginal = cov @ w.values
        rc = w.values * marginal  # 风险贡献
        target = rc.sum() / n_stocks
        # 全部 RC 应同号且接近 target
        assert (rc > 0).all(), f"RC 出现负值: {rc}"
        for r in rc:
            assert abs(r - target) / abs(target) < 0.05, (
                f"RC={rc}, target={target}, 偏差超过 5%"
            )

    def test_single_candidate(self) -> None:
        w = RiskParityWeighter(lookback=20).weight(
            ["A"], pd.Series(dtype=float), pd.DataFrame()
        )
        assert len(w) == 1
        assert pytest.approx(w.iloc[0]) == 1.0

    def test_empty_candidates(self) -> None:
        w = RiskParityWeighter(lookback=20).weight(
            [], pd.Series(dtype=float), pd.DataFrame()
        )
        assert w.empty

    def test_history_too_short_falls_back(self) -> None:
        # 历史只有 3 行 → 退化为 InverseVol（lookback 默认值有保护）
        history = pd.DataFrame(
            np.random.default_rng(0).normal(0, 0.01, (3, 3)),
            columns=["A", "B", "C"],
        )
        w = RiskParityWeighter(lookback=60).weight(
            ["A", "B", "C"], pd.Series(dtype=float), history
        )
        assert pytest.approx(w.sum()) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# build_weighter 工厂
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildWeighter:
    def test_build_equal(self) -> None:
        w = build_weighter({"type": "equal"})
        assert isinstance(w, EqualWeighter)

    def test_build_inverse_vol_with_kwargs(self) -> None:
        w = build_weighter(
            {"type": "inverse_vol", "kwargs": {"lookback": 30, "min_vol": 1e-3}}
        )
        assert isinstance(w, InverseVolWeighter)
        assert w.lookback == 30

    def test_build_score_with_temperature(self) -> None:
        w = build_weighter(
            {"type": "score", "kwargs": {"temperature": 0.5}}
        )
        assert isinstance(w, ScoreWeighter)
        assert w.temperature == 0.5

    def test_build_risk_parity(self) -> None:
        w = build_weighter({"type": "risk_parity"})
        assert isinstance(w, RiskParityWeighter)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="未知 weighter type"):
            build_weighter({"type": "no_such_thing"})

    def test_missing_type_raises(self) -> None:
        with pytest.raises(ValueError, match="type"):
            build_weighter({"kwargs": {}})

    def test_kwargs_not_dict_raises(self) -> None:
        with pytest.raises(ValueError, match="kwargs"):
            build_weighter({"type": "equal", "kwargs": "not_a_dict"})  # type: ignore[arg-type]


# ─────────────────────────────────────────────────────────────────────────────
# RiskConstraints
# ─────────────────────────────────────────────────────────────────────────────

class TestRiskConstraints:
    """风险约束顺序管线：单票 → 板块 → 换手 → 归一化。"""

    def test_max_weight_clip_redistributes(self) -> None:
        # 一只票占 50%，cap=20%，余 30% 应按其他 4 只的相对比例分配
        w = pd.Series(
            [0.50, 0.10, 0.20, 0.10, 0.10],
            index=["A", "B", "C", "D", "E"],
        )
        rc = RiskConstraints(max_weight=0.20, sector_max_weight=None, max_turnover=None)
        out = rc.apply(w)
        assert pytest.approx(out.sum(), abs=1e-9) == 1.0
        # A 被截到 0.20
        assert out["A"] <= 0.20 + 1e-9
        # 其他票按原相对比例分：B/D/E 各 0.10，C 0.20 → C 是 B/D/E 的 2 倍
        # 余量 0.30 按 (0.10+0.20+0.10+0.10)=0.50 分配
        # 但分到 C 后 C 可能也超过 cap=0.20，进入第二轮迭代
        # 终态：A=C=0.20（都到上限），B/D/E 分余下 0.60 → 各 0.20
        assert (out <= 0.20 + 1e-6).all()
        assert pytest.approx(out["B"], abs=1e-6) == out["D"] == out["E"]

    def test_max_weight_iterates_when_redistribution_overflows(self) -> None:
        # 极端：3 只票 [0.7, 0.2, 0.1]，cap=0.4 → 一次截断后 B 也会超
        # 算法应迭代直到稳态
        w = pd.Series([0.7, 0.2, 0.1], index=["A", "B", "C"])
        rc = RiskConstraints(max_weight=0.40, sector_max_weight=None, max_turnover=None)
        out = rc.apply(w)
        # 全部 ≤ cap
        assert (out <= 0.40 + 1e-9).all()
        assert pytest.approx(out.sum(), abs=1e-9) == 1.0

    def test_sector_cap_with_custom_provider(self) -> None:
        # 5 只票，前 3 只属 sector1，权重和 0.7；cap=0.5
        # 注入自定义 sector_provider（避免依赖 classify_market）
        w = pd.Series(
            [0.30, 0.20, 0.20, 0.15, 0.15],
            index=["A", "B", "C", "D", "E"],
        )
        sec = {"A": "S1", "B": "S1", "C": "S1", "D": "S2", "E": "S2"}
        rc = RiskConstraints(
            max_weight=None,
            sector_max_weight=0.50,
            max_turnover=None,
            sector_provider=lambda x: sec[x],
        )
        out = rc.apply(w)
        # S1 总权重 ≤ 0.50
        s1_sum = out[["A", "B", "C"]].sum()
        assert s1_sum <= 0.50 + 1e-6
        # S2 总权重应增加到 1 - 0.50 = 0.50
        s2_sum = out[["D", "E"]].sum()
        assert pytest.approx(s2_sum, abs=1e-6) == 0.50
        assert pytest.approx(out.sum(), abs=1e-9) == 1.0

    def test_turnover_bounded(self) -> None:
        # 上期 100% 持有 X/Y/Z 等权；本期目标全部清出换 A/B/C
        # 完整换手 = 1.0，cap = 0.30 → α = 0.30
        # 输出应包含 union 域，prev 仓位残留 70%×prev_w，target 仓位获得 30%×target_w
        prev = pd.Series([1 / 3, 1 / 3, 1 / 3], index=["X", "Y", "Z"])
        target = pd.Series([1 / 3, 1 / 3, 1 / 3], index=["A", "B", "C"])
        rc = RiskConstraints(
            max_weight=None,
            sector_max_weight=None,
            max_turnover=0.30,
        )
        out = rc.apply(target, prev_w=prev)
        # 输出应包含 6 只票
        assert set(out.index) == {"A", "B", "C", "X", "Y", "Z"}
        assert pytest.approx(out.sum(), abs=1e-9) == 1.0
        # 验证实际换手 ≈ cap
        union = out.index.union(prev.index)
        prev_full = prev.reindex(union).fillna(0.0)
        out_full = out.reindex(union).fillna(0.0)
        actual_to = float((out_full - prev_full).abs().sum()) / 2.0
        assert actual_to <= 0.30 + 1e-6
        # 残余 prev：每只 0.70/3 ≈ 0.233，新仓 target：每只 0.30/3 = 0.10
        for sym in ["X", "Y", "Z"]:
            assert pytest.approx(out[sym], abs=1e-3) == 0.70 / 3
        for sym in ["A", "B", "C"]:
            assert pytest.approx(out[sym], abs=1e-3) == 0.30 / 3

    def test_turnover_no_action_when_within_limit(self) -> None:
        # target 与 prev 仅微调（换手 0.10 < cap 0.30）→ 直接返回 target
        prev = pd.Series([0.5, 0.5], index=["A", "B"])
        target = pd.Series([0.55, 0.45], index=["A", "B"])
        rc = RiskConstraints(
            max_weight=None,
            sector_max_weight=None,
            max_turnover=0.30,
        )
        out = rc.apply(target, prev_w=prev)
        assert pytest.approx(out["A"], abs=1e-6) == 0.55
        assert pytest.approx(out["B"], abs=1e-6) == 0.45

    def test_pipeline_all_three_stages(self) -> None:
        """三层约束串联：单票截断 + 板块截断 + 换手限制都触发。"""
        target = pd.Series(
            [0.40, 0.20, 0.15, 0.15, 0.10],
            index=["A", "B", "C", "D", "E"],
        )
        prev = pd.Series([0.5, 0.5], index=["X", "Y"])  # 全是新仓
        sec = {"A": "S1", "B": "S1", "C": "S1", "D": "S2", "E": "S2"}
        rc = RiskConstraints(
            max_weight=0.30,
            sector_max_weight=0.50,
            max_turnover=0.40,
            sector_provider=lambda x: sec[x],
        )
        out = rc.apply(target, prev_w=prev)
        assert pytest.approx(out.sum(), abs=1e-9) == 1.0
        # 单票 ≤ 0.30
        assert (out <= 0.30 + 1e-6).all()

    def test_empty_target_returns_empty(self) -> None:
        rc = RiskConstraints()
        out = rc.apply(pd.Series(dtype=float))
        assert out.empty

    def test_invalid_params(self) -> None:
        with pytest.raises(ValueError, match="max_weight"):
            RiskConstraints(max_weight=0.0)
        with pytest.raises(ValueError, match="max_weight"):
            RiskConstraints(max_weight=1.5)
        with pytest.raises(ValueError, match="sector_max_weight"):
            RiskConstraints(sector_max_weight=-0.1)
        with pytest.raises(ValueError, match="max_turnover"):
            RiskConstraints(max_turnover=2.5)

    def test_default_sector_provider_uses_classify_market(self) -> None:
        # 默认板块映射应能识别 .SH/.SZ 后缀
        w = pd.Series(
            [0.35, 0.35, 0.30],
            index=["600000.SH", "601000.SH", "000001.SZ"],
        )
        rc = RiskConstraints(
            max_weight=None,
            sector_max_weight=0.50,
            max_turnover=None,
        )
        out = rc.apply(w)
        # 600000 + 601000 同属"上证主板"，原和 0.70 → 截到 0.50
        sh_sum = out[["600000.SH", "601000.SH"]].sum()
        assert sh_sum <= 0.50 + 1e-6
        assert pytest.approx(out.sum(), abs=1e-9) == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# QuantMLWeightStrategy
# ─────────────────────────────────────────────────────────────────────────────

class _FakePosition:
    """模拟 qlib Position，供单测使用。"""

    def __init__(self, weight_dict: dict[str, float]) -> None:
        self._wd = dict(weight_dict)

    def get_stock_weight_dict(self, only_stock: bool = True) -> dict[str, float]:
        return dict(self._wd)


class TestQuantMLWeightStrategy:
    """重点覆盖：调仓节流、五段式管线、参数校验。"""

    @pytest.fixture
    def basic_score(self) -> pd.Series:
        return pd.Series(
            np.linspace(1.0, 0.1, 10),
            index=[f"60000{i}.SH" for i in range(10)],
        )

    @pytest.fixture
    def fake_history(self) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        cols = [f"60000{i}.SH" for i in range(10)]
        dates = pd.date_range("2024-01-01", periods=120, freq="B")
        return pd.DataFrame(
            rng.normal(0, 0.02, (120, 10)), index=dates, columns=cols
        )

    def _make_strategy(self, **overrides) -> QuantMLWeightStrategy:
        # 父类 WeightStrategyBase 强制要求 signal，传一个最小 placeholder Series
        # 单测里不会真用到该 signal（generate_target_weight_position 直接接收 score 参数）
        placeholder_signal = pd.Series(
            [0.0],
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2025-01-01"), "PLACEHOLDER")],
                names=["datetime", "instrument"],
            ),
            name="score",
        )
        cfg = dict(
            signal=placeholder_signal,
            topk=5,
            score_quantile=0.0,
            rebalance_freq=10,
            vol_lookback=60,
            weighter_cfg={"type": "equal"},
            risk_cfg={
                "max_weight": 0.30,
                "sector_max_weight": 1.0,
                "max_turnover": 1.0,
            },
        )
        cfg.update(overrides)
        return QuantMLWeightStrategy(**cfg)

    # ─── 五段式管线 ─────────────────────────────────────────────────

    def test_first_step_always_recomputes(
        self, basic_score, fake_history, monkeypatch
    ) -> None:
        """generate_target_weight_position 在每次调仓日被调用后产出权重。"""
        s = self._make_strategy()
        monkeypatch.setattr(
            s, "_fetch_return_history", lambda **kw: fake_history
        )
        out = s.generate_target_weight_position(
            score=basic_score,
            current=None,
            trade_start_time=pd.Timestamp("2025-01-15"),
            trade_end_time=pd.Timestamp("2025-01-15"),
        )
        assert len(out) == 5  # topk=5
        assert pytest.approx(sum(out.values()), abs=1e-6) == 1.0

    # ─── 调仓节流（通过 generate_trade_decision 实现）─────────────

    def test_throttle_first_step_calls_super(
        self, basic_score, fake_history, monkeypatch
    ) -> None:
        """step 1：调仓日，应调用父类 generate_trade_decision。"""
        s = self._make_strategy(rebalance_freq=10)
        called = {"n": 0}
        sentinel = object()

        def fake_super(self, execute_result=None):
            called["n"] += 1
            return sentinel

        monkeypatch.setattr(
            type(s).__bases__[0], "generate_trade_decision", fake_super
        )

        td = s.generate_trade_decision()
        assert called["n"] == 1
        assert td is sentinel  # 调仓日转发给父类

    def test_throttle_non_rebalance_returns_empty(
        self, basic_score, fake_history, monkeypatch
    ) -> None:
        """
        非调仓日：generate_trade_decision 直接返回 TradeDecisionWO([], self)
        而不调用父类（彻底杜绝再平衡微调单，是中频策略性能关键）。
        """
        from qlib.backtest.decision import TradeDecisionWO
        s = self._make_strategy(rebalance_freq=10)
        called = {"n": 0}
        sentinel = object()

        def fake_super(self, execute_result=None):
            called["n"] += 1
            return sentinel

        monkeypatch.setattr(
            type(s).__bases__[0], "generate_trade_decision", fake_super
        )
        # 同样 mock TradeDecisionWO（避免依赖 trade_calendar）
        empty_decision = object()
        monkeypatch.setattr(
            "backtest.strategy.quantml_strategy.TradeDecisionWO",
            lambda orders, strategy: empty_decision if not orders else sentinel,
        )

        # step 1：调仓日 → 调用父类
        td1 = s.generate_trade_decision()
        assert called["n"] == 1
        assert td1 is sentinel

        # step 2 ~ 10：非调仓日，9 次全部返回空订单且不调父类
        for _ in range(9):
            td_n = s.generate_trade_decision()
            assert td_n is empty_decision
        assert called["n"] == 1  # 父类没被再调

        # step 11：再次到调仓日 → 又调用一次父类
        td_11 = s.generate_trade_decision()
        assert called["n"] == 2
        assert td_11 is sentinel

    def test_throttle_non_rebalance_with_exit_rules_calls_super(
        self, monkeypatch
    ) -> None:
        """非模型日若启用 exit_rules，仍应调用父类以生成止损等订单。"""
        s = self._make_strategy(
            rebalance_freq=10,
            exit_rules={"stop_loss_pct": 0.05},
        )
        called = {"n": 0}
        sentinel = object()

        def fake_super(self, execute_result=None):
            called["n"] += 1
            return sentinel

        monkeypatch.setattr(
            type(s).__bases__[0], "generate_trade_decision", fake_super
        )
        s.generate_trade_decision()
        assert called["n"] == 1
        s.generate_trade_decision()
        assert called["n"] == 2

    def test_exit_events_recorded_and_retrievable(
        self, basic_score, monkeypatch
    ) -> None:
        """触发的平仓事件应被记录到 _exit_events，并可通过 get_exit_events 取回。"""
        s = self._make_strategy(exit_rules={"stop_loss_pct": 0.10})
        s._pending_model_rebalance = False
        s._last_target_weights = {"600000.SH": 1.0}
        s._entry_ref = {"600000.SH": 100.0}
        s._peak_ref = {"600000.SH": 100.0}
        monkeypatch.setattr(
            s, "_fetch_marks", lambda codes, ts, te: {"600000.SH": 80.0}
        )
        prev = _FakePosition({"600000.SH": 1.0})
        s.generate_target_weight_position(
            score=basic_score,
            current=prev,
            trade_start_time=pd.Timestamp("2025-01-15"),
            trade_end_time=pd.Timestamp("2025-01-15"),
        )
        df = s.get_exit_events()
        assert len(df) == 1
        assert df.iloc[0]["code"] == "600000.SH"
        assert df.iloc[0]["reason"] == "stop_loss"
        assert df.iloc[0]["pnl_pct"] == pytest.approx(-0.20)
        assert pd.Timestamp(df.iloc[0]["datetime"]) == pd.Timestamp("2025-01-15")

    def test_get_exit_events_empty_when_no_trigger(self) -> None:
        s = self._make_strategy(exit_rules={"stop_loss_pct": 0.10})
        df = s.get_exit_events()
        assert df.empty
        assert list(df.columns) == [
            "datetime", "code", "reason", "entry_ref", "mark", "peak", "pnl_pct"
        ]

    def test_exit_rules_non_model_day_strips_loser(
        self, basic_score, monkeypatch
    ) -> None:
        """非模型日：止损触发标的从目标权中剔除并归一化剩余仓位。"""
        s = self._make_strategy(
            exit_rules={"stop_loss_pct": 0.10},
        )
        s._pending_model_rebalance = False
        s._last_target_weights = {"600000.SH": 0.5, "600001.SH": 0.5}
        s._entry_ref = {"600000.SH": 100.0, "600001.SH": 100.0}
        s._peak_ref = dict(s._entry_ref)
        monkeypatch.setattr(
            s,
            "_fetch_marks",
            lambda codes, ts, te: {
                "600000.SH": 89.0,
                "600001.SH": 100.0,
            },
        )
        prev = _FakePosition({"600000.SH": 0.5, "600001.SH": 0.5})
        out = s.generate_target_weight_position(
            score=basic_score,
            current=prev,
            trade_start_time=pd.Timestamp("2025-01-15"),
            trade_end_time=pd.Timestamp("2025-01-15"),
        )
        assert set(out.keys()) == {"600001.SH"}
        assert pytest.approx(out["600001.SH"]) == 1.0

    def test_throttle_freq_one_calls_super_every_step(
        self, basic_score, fake_history, monkeypatch
    ) -> None:
        """rebalance_freq=1：每步都是调仓日。"""
        s = self._make_strategy(rebalance_freq=1)
        called = {"n": 0}
        sentinel = object()

        def fake_super(self, execute_result=None):
            called["n"] += 1
            return sentinel

        monkeypatch.setattr(
            type(s).__bases__[0], "generate_trade_decision", fake_super
        )
        for _ in range(5):
            s.generate_trade_decision()
        assert called["n"] == 5

    # ─── 五段式管线 ─────────────────────────────────────────────────

    def test_pipeline_runs_and_caps_max_weight(
        self, basic_score, fake_history, monkeypatch
    ) -> None:
        # cap=0.20 + 5 只候选 → 全部到 cap，sum 应仍 = 1
        s = self._make_strategy(
            risk_cfg={"max_weight": 0.20, "sector_max_weight": 1.0, "max_turnover": 1.0}
        )
        monkeypatch.setattr(
            s, "_fetch_return_history", lambda **kw: fake_history
        )
        out = s.generate_target_weight_position(
            score=basic_score, current=None,
            trade_start_time=pd.Timestamp("2025-01-15"),
            trade_end_time=pd.Timestamp("2025-01-15"),
        )
        assert all(v <= 0.20 + 1e-6 for v in out.values())
        assert pytest.approx(sum(out.values()), abs=1e-6) == 1.0

    def test_pipeline_with_inverse_vol(
        self, basic_score, monkeypatch
    ) -> None:
        # 构造明确的波动率差异：低波动股 idx=0 应获得最高权重
        rng = np.random.default_rng(42)
        cols = [f"60000{i}.SH" for i in range(10)]
        vols = np.linspace(0.005, 0.05, 10)
        history = pd.DataFrame(
            np.column_stack(
                [rng.normal(0, v, 120) for v in vols]
            ),
            columns=cols,
            index=pd.date_range("2024-01-01", periods=120, freq="B"),
        )
        s = self._make_strategy(
            topk=10,  # 全选
            weighter_cfg={"type": "inverse_vol", "kwargs": {"lookback": 60}},
            risk_cfg={"max_weight": 1.0, "sector_max_weight": 1.0, "max_turnover": 1.0},
        )
        monkeypatch.setattr(s, "_fetch_return_history", lambda **kw: history)
        out = s.generate_target_weight_position(
            score=basic_score, current=None,
            trade_start_time=pd.Timestamp("2025-01-15"),
            trade_end_time=pd.Timestamp("2025-01-15"),
        )
        # idx=0 是最低波动 → 权重最大
        ws = pd.Series(out)
        assert ws["600000.SH"] == ws.max()

    def test_score_quantile_filter_in_pipeline(
        self, fake_history, monkeypatch
    ) -> None:
        # 高分位过滤后只剩头部 30%，topk=5 但应只产出 3 只
        score = pd.Series(
            np.linspace(1.0, 0.1, 10),
            index=[f"60000{i}.SH" for i in range(10)],
        )
        s = self._make_strategy(topk=5, score_quantile=0.7)
        monkeypatch.setattr(s, "_fetch_return_history", lambda **kw: fake_history)
        out = s.generate_target_weight_position(
            score=score, current=None,
            trade_start_time=pd.Timestamp("2025-01-15"),
            trade_end_time=pd.Timestamp("2025-01-15"),
        )
        # 70% 分位 → 至多前 30% 的票，即 3 只
        assert len(out) <= 3

    def test_empty_score_returns_empty_or_cached(
        self, basic_score, fake_history, monkeypatch
    ) -> None:
        s = self._make_strategy()
        monkeypatch.setattr(s, "_fetch_return_history", lambda **kw: fake_history)
        # 第一步：empty score → 空结果
        out = s.generate_target_weight_position(
            score=pd.Series(dtype=float), current=None,
            trade_start_time=pd.Timestamp("2025-01-15"),
            trade_end_time=pd.Timestamp("2025-01-15"),
        )
        assert out == {}

    def test_extracts_prev_weights_from_position(
        self, basic_score, fake_history, monkeypatch
    ) -> None:
        s = self._make_strategy(
            risk_cfg={"max_weight": 1.0, "sector_max_weight": 1.0, "max_turnover": 0.30}
        )
        monkeypatch.setattr(s, "_fetch_return_history", lambda **kw: fake_history)
        # 上期持有不同的 5 只票
        prev = _FakePosition(
            {f"99999{i}.SH": 0.2 for i in range(5)}
        )
        out = s.generate_target_weight_position(
            score=basic_score, current=prev,
            trade_start_time=pd.Timestamp("2025-01-15"),
            trade_end_time=pd.Timestamp("2025-01-15"),
        )
        # 由于 max_turnover=0.30 且换手原本为 1.0 → 仅 30% 转向 target
        # 输出应包含 prev 残余 + target 一小部分
        prev_residual = sum(v for k, v in out.items() if k.startswith("99999"))
        assert prev_residual > 0.5  # ≈ 0.7
        new_holdings = sum(v for k, v in out.items() if k.startswith("60000"))
        assert new_holdings > 0  # 部分新仓建立

    # ─── 参数校验 ────────────────────────────────────────────────────

    def test_invalid_rebalance_freq(self) -> None:
        with pytest.raises(ValueError, match="rebalance_freq"):
            self._make_strategy(rebalance_freq=0)
        with pytest.raises(ValueError, match="rebalance_freq"):
            self._make_strategy(rebalance_freq=-1)

    def test_invalid_vol_lookback(self) -> None:
        with pytest.raises(ValueError, match="vol_lookback"):
            self._make_strategy(vol_lookback=3)

    def test_invalid_weighter_type_propagates(self) -> None:
        with pytest.raises(ValueError, match="未知 weighter type"):
            self._make_strategy(weighter_cfg={"type": "no_such_thing"})

    def test_invalid_n_drop_propagates(self) -> None:
        with pytest.raises(ValueError, match="n_drop"):
            self._make_strategy(n_drop=-1)

    # ─── n_drop 行为 ────────────────────────────────────────────────

    def test_n_drop_keeps_top_score_loser(
        self, fake_history, monkeypatch
    ) -> None:
        """
        prev 持有 X1/X2/X3 三只，本期 target 想全换为 A/B/C，n_drop=1：
        应只允许卖出 1 只 → 保留 2 只「计划卖出但 score 最高」的旧票。
        """
        s = self._make_strategy(
            topk=3,
            n_drop=1,
            score_quantile=0.0,
            risk_cfg={"max_weight": 1.0, "sector_max_weight": 1.0, "max_turnover": 1.0},
            weighter_cfg={"type": "equal"},
        )
        monkeypatch.setattr(s, "_fetch_return_history", lambda **kw: fake_history)
        score = pd.Series(
            {
                "A": 0.95, "B": 0.90, "C": 0.85,
                "X1": 0.50, "X2": 0.40, "X3": 0.10,
            }
        )
        prev = _FakePosition({"X1": 1 / 3, "X2": 1 / 3, "X3": 1 / 3})
        out = s.generate_target_weight_position(
            score=score,
            current=prev,
            trade_start_time=pd.Timestamp("2025-01-15"),
            trade_end_time=pd.Timestamp("2025-01-15"),
        )
        assert pytest.approx(sum(out.values()), abs=1e-6) == 1.0
        assert "X3" not in out
        assert "X1" in out and "X2" in out
        assert {"A", "B", "C"}.issubset(out.keys())

    def test_n_drop_no_effect_when_within_limit(
        self, fake_history, monkeypatch
    ) -> None:
        """计划卖出 ≤ n_drop：n_drop 不应改变 target。"""
        s = self._make_strategy(
            topk=3,
            n_drop=5,
            risk_cfg={"max_weight": 1.0, "sector_max_weight": 1.0, "max_turnover": 1.0},
            weighter_cfg={"type": "equal"},
        )
        monkeypatch.setattr(s, "_fetch_return_history", lambda **kw: fake_history)
        score = pd.Series({"A": 0.9, "B": 0.8, "C": 0.7, "X1": 0.5})
        prev = _FakePosition({"X1": 1.0})
        out = s.generate_target_weight_position(
            score=score,
            current=prev,
            trade_start_time=pd.Timestamp("2025-01-15"),
            trade_end_time=pd.Timestamp("2025-01-15"),
        )
        assert set(out.keys()) == {"A", "B", "C"}
        assert "X1" not in out

    def test_n_drop_zero_disabled(self, fake_history, monkeypatch) -> None:
        """n_drop=0 等价于 None：完全不限制。"""
        s = self._make_strategy(topk=3, n_drop=0)
        assert s.n_drop is None

    # ─── _extract_prev_weights ───────────────────────────────────────

    def test_extract_prev_handles_none(self) -> None:
        result = QuantMLWeightStrategy._extract_prev_weights(None)
        assert result.empty

    def test_extract_prev_handles_dict(self) -> None:
        result = QuantMLWeightStrategy._extract_prev_weights(
            {"A": 0.5, "B": 0.5}
        )
        assert pytest.approx(result["A"]) == 0.5

    def test_extract_prev_handles_position(self) -> None:
        result = QuantMLWeightStrategy._extract_prev_weights(
            _FakePosition({"X": 1.0})
        )
        assert result["X"] == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# StrategyLayer 调度（QuantML 路径）
# ─────────────────────────────────────────────────────────────────────────────

class TestStrategyLayerQuantML:
    """覆盖 StrategyLayer 对 QuantMLWeightStrategy 配置的校验、build、describe。"""

    def _valid_cfg(self) -> dict:
        return {
            "class": "QuantMLWeightStrategy",
            "module_path": "backtest.strategy",
            "kwargs": {
                "topk": 25,
                "score_quantile": 0.7,
                "rebalance_freq": 10,
                "vol_lookback": 60,
                "weighter_cfg": {
                    "type": "inverse_vol",
                    "kwargs": {"lookback": 60},
                },
                "risk_cfg": {
                    "max_weight": 0.08,
                    "sector_max_weight": 0.30,
                    "max_turnover": 0.30,
                },
            },
        }

    def test_build_injects_signal(self) -> None:
        from backtest.strategy_layer import StrategyLayer
        layer = StrategyLayer()
        signal = pd.Series(
            [1.0],
            index=pd.MultiIndex.from_tuples(
                [(pd.Timestamp("2025-01-01"), "600000.SH")],
                names=["datetime", "instrument"],
            ),
            name="score",
        )
        out = layer.build(self._valid_cfg(), signal)
        assert out["class"] == "QuantMLWeightStrategy"
        assert out["module_path"] == "backtest.strategy"
        assert out["kwargs"]["signal"] is signal
        assert out["kwargs"]["topk"] == 25

    def test_describe_quantml(self) -> None:
        from backtest.strategy_layer import StrategyLayer
        desc = StrategyLayer().describe(self._valid_cfg())
        assert "QuantMLWeightStrategy" in desc
        assert "inverse_vol" in desc
        assert "0.08" in desc

    def test_validate_invalid_topk(self) -> None:
        from backtest.strategy_layer import StrategyLayer
        cfg = self._valid_cfg()
        cfg["kwargs"]["topk"] = 0
        with pytest.raises(ValueError, match="topk"):
            StrategyLayer().build(cfg, pd.Series(dtype=float))

    def test_validate_invalid_quantile(self) -> None:
        from backtest.strategy_layer import StrategyLayer
        cfg = self._valid_cfg()
        cfg["kwargs"]["score_quantile"] = 1.0
        with pytest.raises(ValueError, match="score_quantile"):
            StrategyLayer().build(cfg, pd.Series(dtype=float))

    def test_validate_invalid_weighter_type(self) -> None:
        from backtest.strategy_layer import StrategyLayer
        cfg = self._valid_cfg()
        cfg["kwargs"]["weighter_cfg"]["type"] = "bogus"
        with pytest.raises(ValueError, match="weighter_cfg.type"):
            StrategyLayer().build(cfg, pd.Series(dtype=float))

    def test_validate_invalid_max_weight(self) -> None:
        from backtest.strategy_layer import StrategyLayer
        cfg = self._valid_cfg()
        cfg["kwargs"]["risk_cfg"]["max_weight"] = 1.5
        with pytest.raises(ValueError, match="max_weight"):
            StrategyLayer().build(cfg, pd.Series(dtype=float))

    def test_validate_invalid_exit_rules(self) -> None:
        from backtest.strategy_layer import StrategyLayer
        cfg = self._valid_cfg()
        cfg["kwargs"]["exit_rules"] = {"stop_loss_pct": -0.1}
        with pytest.raises(ValueError, match="exit_rules"):
            StrategyLayer().build(cfg, pd.Series(dtype=float))

    def test_validate_invalid_n_drop_negative(self) -> None:
        from backtest.strategy_layer import StrategyLayer
        cfg = self._valid_cfg()
        cfg["kwargs"]["n_drop"] = -1
        with pytest.raises(ValueError, match="n_drop"):
            StrategyLayer().build(cfg, pd.Series(dtype=float))

    def test_validate_n_drop_exceeds_topk(self) -> None:
        from backtest.strategy_layer import StrategyLayer
        cfg = self._valid_cfg()
        cfg["kwargs"]["topk"] = 5
        cfg["kwargs"]["n_drop"] = 6
        with pytest.raises(ValueError, match="n_drop"):
            StrategyLayer().build(cfg, pd.Series(dtype=float))

    def test_describe_includes_n_drop(self) -> None:
        from backtest.strategy_layer import StrategyLayer
        cfg = self._valid_cfg()
        cfg["kwargs"]["n_drop"] = 2
        desc = StrategyLayer().describe(cfg)
        assert "每期换出" in desc
        assert "≤ 2" in desc

    def test_topk_dropout_unaffected(self) -> None:
        # 老路径回归：TopkDropoutStrategy 仍能正常 build
        from backtest.strategy_layer import StrategyLayer
        cfg = {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {"topk": 8, "n_drop": 2},
        }
        signal = pd.Series([1.0], index=["X"])
        out = StrategyLayer().build(cfg, signal)
        assert out["class"] == "TopkDropoutStrategy"
        assert out["kwargs"]["signal"] is signal
