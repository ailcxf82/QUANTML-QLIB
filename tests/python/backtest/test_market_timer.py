"""
MarketTimer 单元测试
====================
覆盖：
  1. NullTimer：始终返回 1.0
  2. RollingVolTimer：高波动期返回 < 1.0，低波动期贴近上限
  3. GarchVolTimer：冷启动（obs 不足）时返回 1.0 兜底
  4. GarchVolTimer：高波动合成数据 → risk_factor 方向正确（高波动 < 低波动）
  5. build_market_timer 工厂函数：type=null / rolling / garch 分支
  6. _winsorize_a_share：涨跌停截断 + 节假日跳空处理

执行：
  pytest tests/python/backtest/test_market_timer.py -v
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.strategy.market_timer import (
    NullTimer,
    RollingVolTimer,
    GarchVolTimer,
    build_market_timer,
    _winsorize_a_share,
    _LIMIT_THRESHOLD,
)


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数：合成收益率序列
# ─────────────────────────────────────────────────────────────────────────────

def _make_returns(
    n: int,
    vol: float = 0.02,
    seed: int = 42,
    start: str = "2020-01-01",
) -> pd.Series:
    """生成指定日频波动率的随机收益率序列。"""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n)
    rets = rng.normal(0, vol, size=n)
    return pd.Series(rets, index=dates, dtype=float)


def _make_high_vol_returns(n: int = 200, seed: int = 0) -> pd.Series:
    """年化波动率 ≈ 60%（极端）的收益率序列。"""
    return _make_returns(n, vol=0.038, seed=seed)  # 0.038 × √252 ≈ 0.60


def _make_low_vol_returns(n: int = 200, seed: int = 1) -> pd.Series:
    """年化波动率 ≈ 10%（平静）的收益率序列。"""
    return _make_returns(n, vol=0.0063, seed=seed)  # 0.0063 × √252 ≈ 0.10


# ─────────────────────────────────────────────────────────────────────────────
# 1. NullTimer
# ─────────────────────────────────────────────────────────────────────────────

class TestNullTimer:
    def test_always_returns_one(self) -> None:
        timer = NullTimer()
        for date in pd.bdate_range("2024-01-02", periods=10):
            assert timer.get_risk_factor(date) == 1.0

    def test_describe(self) -> None:
        assert "1.0" in NullTimer().describe()


# ─────────────────────────────────────────────────────────────────────────────
# 2. RollingVolTimer
# ─────────────────────────────────────────────────────────────────────────────

class TestRollingVolTimer:
    """
    所有测试通过 _index_returns 注入合成数据，无 qlib 依赖。
    """

    def _make_timer(
        self,
        returns: pd.Series,
        target_vol: float = 0.15,
        min_risk: float = 0.30,
        max_risk: float = 1.0,
        lookback: int = 60,
        min_obs: int = 30,
    ) -> RollingVolTimer:
        return RollingVolTimer(
            benchmark="SH000905",
            target_vol=target_vol,
            min_risk=min_risk,
            max_risk=max_risk,
            lookback=lookback,
            min_obs=min_obs,
            _index_returns=returns,
        )

    def test_high_vol_returns_below_one(self) -> None:
        """年化波动 >> target_vol → risk_factor < 1.0"""
        rets = _make_high_vol_returns()
        timer = self._make_timer(rets, target_vol=0.15)
        # 用序列最后日期的次日作为 trade_date
        trade_date = rets.index[-1] + pd.Timedelta(days=1)
        rf = timer.get_risk_factor(trade_date)
        assert 0.0 < rf < 1.0, f"期望 < 1.0，实际 {rf:.4f}"

    def test_low_vol_clamps_to_max_risk(self) -> None:
        """年化波动 << target_vol → risk_factor 被 clip 到 max_risk"""
        rets = _make_low_vol_returns()
        timer = self._make_timer(rets, target_vol=0.15, max_risk=1.0)
        trade_date = rets.index[-1] + pd.Timedelta(days=1)
        rf = timer.get_risk_factor(trade_date)
        assert rf == pytest.approx(1.0), f"期望 = 1.0（上限），实际 {rf:.4f}"

    def test_cold_start_returns_one(self) -> None:
        """历史数据不足 min_obs 时返回 1.0（冷启动兜底）。"""
        rets = _make_returns(10, vol=0.05)  # 仅 10 个样本
        timer = self._make_timer(rets, min_obs=30)
        trade_date = rets.index[-1] + pd.Timedelta(days=1)
        rf = timer.get_risk_factor(trade_date)
        assert rf == 1.0

    def test_risk_factor_within_bounds(self) -> None:
        """返回值始终在 [min_risk, max_risk] 内。"""
        rets = _make_high_vol_returns(n=300)
        timer = self._make_timer(rets, min_risk=0.25, max_risk=0.95)
        for date in pd.bdate_range(rets.index[100], rets.index[-1]):
            rf = timer.get_risk_factor(date)
            assert 0.25 <= rf <= 0.95, f"{date}: {rf} 超出 [0.25, 0.95]"

    def test_parameter_validation(self) -> None:
        """min_risk > max_risk 应抛 ValueError。"""
        with pytest.raises(ValueError, match="min_risk"):
            RollingVolTimer(
                benchmark="SH000905",
                target_vol=0.15,
                min_risk=0.8,
                max_risk=0.3,
            )

    def test_describe_contains_key_info(self) -> None:
        timer = self._make_timer(_make_returns(60))
        desc = timer.describe()
        assert "RollingVolTimer" in desc
        assert "15.00%" in desc


# ─────────────────────────────────────────────────────────────────────────────
# 3 & 4. GarchVolTimer
# ─────────────────────────────────────────────────────────────────────────────

class TestGarchVolTimer:
    """
    GarchVolTimer 测试：通过 _index_returns 注入，无 qlib 依赖。
    当 arch 包不可用时，GarchVolTimer 自动降级为 RollingVolTimer，测试仍然覆盖接口。
    """

    def _make_timer(
        self,
        returns: pd.Series,
        target_vol: float = 0.15,
        min_risk: float = 0.30,
        max_risk: float = 1.0,
        min_obs: int = 50,
        refit_freq: int = 1,
    ) -> GarchVolTimer:
        return GarchVolTimer(
            benchmark="SH000905",
            target_vol=target_vol,
            min_risk=min_risk,
            max_risk=max_risk,
            refit_freq=refit_freq,
            min_obs=min_obs,
            _index_returns=returns,
        )

    def test_cold_start_returns_one(self) -> None:
        """观测期不足 min_obs 时，始终返回 1.0（测试3）。"""
        rets = _make_returns(30)  # 仅 30 个样本
        timer = self._make_timer(rets, min_obs=60)
        trade_date = rets.index[-1] + pd.Timedelta(days=1)
        rf = timer.get_risk_factor(trade_date)
        assert rf == 1.0, f"冷启动应返回 1.0，实际 {rf}"

    def test_high_vol_lower_than_low_vol(self) -> None:
        """高波动序列的 risk_factor < 低波动序列（测试4，方向正确性）。"""
        # 使用足够长的序列确保 GARCH 能收敛
        n = 300
        high_rets = _make_high_vol_returns(n=n)
        low_rets = _make_low_vol_returns(n=n)

        timer_high = self._make_timer(high_rets, min_obs=60)
        timer_low = self._make_timer(low_rets, min_obs=60)

        trade_date = high_rets.index[-1] + pd.Timedelta(days=1)
        rf_high = timer_high.get_risk_factor(trade_date)

        trade_date_low = low_rets.index[-1] + pd.Timedelta(days=1)
        rf_low = timer_low.get_risk_factor(trade_date_low)

        assert rf_high < rf_low, (
            f"高波动 risk_factor ({rf_high:.4f}) 应 < 低波动 ({rf_low:.4f})"
        )

    def test_risk_factor_within_bounds(self) -> None:
        """返回值始终在 [min_risk, max_risk] 内。"""
        rets = _make_high_vol_returns(n=300)
        timer = self._make_timer(rets, min_risk=0.2, max_risk=0.9, min_obs=60)
        for dt in rets.index[100::20]:
            rf = timer.get_risk_factor(dt + pd.Timedelta(days=1))
            assert 0.2 <= rf <= 0.9, f"{dt}: {rf} 超出 [0.2, 0.9]"

    def test_parameter_validation_min_obs(self) -> None:
        with pytest.raises(ValueError, match="min_obs"):
            GarchVolTimer(benchmark="SH000905", min_obs=5)

    def test_describe_contains_backend(self) -> None:
        timer = self._make_timer(_make_returns(100))
        desc = timer.describe()
        assert "GarchVolTimer" in desc


# ─────────────────────────────────────────────────────────────────────────────
# 5. build_market_timer 工厂函数
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildMarketTimer:
    def test_none_cfg_returns_none(self) -> None:
        assert build_market_timer(None) is None

    def test_null_type_returns_null_timer(self) -> None:
        t = build_market_timer({"type": "null"})
        assert isinstance(t, NullTimer)
        assert t.get_risk_factor(pd.Timestamp("2024-01-02")) == 1.0

    def test_rolling_type_returns_rolling_timer(self) -> None:
        rets = _make_returns(100)
        t = build_market_timer(
            {"type": "rolling", "target_vol": 0.20, "min_obs": 30},
            _index_returns=rets,
        )
        assert isinstance(t, RollingVolTimer)

    def test_garch_type_returns_garch_timer(self) -> None:
        rets = _make_returns(150)
        t = build_market_timer(
            {"type": "garch", "min_obs": 50},
            _index_returns=rets,
        )
        assert isinstance(t, GarchVolTimer)

    def test_unknown_type_raises(self) -> None:
        with pytest.raises(ValueError, match="未知 market_timer type"):
            build_market_timer({"type": "unknown_xyz"})

    def test_empty_cfg_returns_none(self) -> None:
        assert build_market_timer({}) is None


# ─────────────────────────────────────────────────────────────────────────────
# 6. _winsorize_a_share A 股特殊处理
# ─────────────────────────────────────────────────────────────────────────────

class TestWinsorizeAShare:
    def test_limit_up_clipped(self) -> None:
        """涨停收益（10%）应被截断到 < _LIMIT_THRESHOLD。"""
        dates = pd.bdate_range("2023-01-01", periods=5)
        rets = pd.Series([0.0, 0.10, -0.10, 0.05, -0.05], index=dates)
        out = _winsorize_a_share(rets)
        assert out.abs().max() < _LIMIT_THRESHOLD

    def test_normal_returns_unchanged(self) -> None:
        """正常收益率（±5%）不应被截断。"""
        dates = pd.bdate_range("2023-01-01", periods=5)
        rets = pd.Series([0.02, -0.03, 0.01, -0.02, 0.04], index=dates)
        out = _winsorize_a_share(rets)
        pd.testing.assert_series_equal(out, rets, check_names=False)

    def test_holiday_gap_winsorized(self) -> None:
        """节假日复市后的跳空收益（大于 3σ）应被 Winsorize。"""
        # 构造 50 个正常交易日（低波动背景），然后插入节假日后的极端跳空
        rng = np.random.default_rng(99)
        bg_dates = pd.bdate_range("2023-01-03", periods=50)
        bg_rets = rng.normal(0, 0.005, size=50)  # 背景波动 ~0.5%

        # 节后复市日（距上一交易日 8 个自然日）
        holiday_date = pd.Timestamp("2023-04-06")  # 节后第一个工作日
        prev_date = pd.Timestamp("2023-03-29")     # 节前最后一个工作日（间隔 8 天）

        # 在背景序列末尾构造含节假日跳空的序列
        all_dates = list(bg_dates[:30]) + [prev_date, holiday_date] + list(bg_dates[30:40])
        jump_rets = list(bg_rets[:30]) + [0.004, 0.060] + list(bg_rets[30:40])  # jump=6%，背景σ≈0.5%
        rets = pd.Series(jump_rets, index=pd.DatetimeIndex(all_dates), dtype=float).sort_index()

        out = _winsorize_a_share(rets)
        # 节后日收益（6%）应被压缩，因为 3σ ≈ 3×0.005 = 1.5% << 6%
        assert abs(out[holiday_date]) < abs(rets[holiday_date]), (
            f"期望节假日跳空被 Winsorize：{rets[holiday_date]:.4f} → {out[holiday_date]:.4f}"
        )

    def test_empty_series_returns_empty(self) -> None:
        out = _winsorize_a_share(pd.Series(dtype=float))
        assert out.empty
