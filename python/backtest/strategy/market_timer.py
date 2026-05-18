"""
MarketTimer（市场波动定时器）
============================
根据预测到的市场波动率动态调节总仓位缩放因子（risk_factor），实现目标波动率策略。

层级位置：Strategy 管线末端（在 RiskConstraints / exit_rules 之后）
    pred_score → Selector → Weighter → RiskConstraints → [exit_rules] → MarketTimer → 提交

核心公式：
    risk_factor = clip(σ_target / (√252 · σ̂_{t+1}), min_risk, max_risk)

其中 σ̂_{t+1} 为 GARCH(1,1) 一步前瞻年化波动率预测，σ_target 为目标年化波动率。
风险因子 < 1 时自动减仓，> 1 时放大至上限（默认 1.0，不加杠杆）。

提供三个实现：
  - NullTimer       : 始终返回 1.0，可作对比基线
  - RollingVolTimer : 基于滚动标准差，arch 包缺失时自动降级
  - GarchVolTimer   : GARCH(1,1) + Student-t 分布（主推，捕捉厚尾与波动聚类）

A 股特殊处理：
  - 涨跌停收益率（|ret| ≥ 0.095）Winsorize 到 ±0.094，避免截断效应扭曲参数
  - 节假日后复市跳空（交易日历间隔 > 3 自然日）：将复市收益 Winsorize 到 3σ
  - 冷启动期（obs < min_obs）：降级返回 1.0

反前视偏差保证：
  get_risk_factor(trade_date) 只使用 trade_date 前一个自然日之前的历史数据。
"""
from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

_logger = logging.getLogger("MarketTimer")

# 尝试导入 arch；如不可用，GARCH 将自动降级到 RollingVolTimer
try:
    from arch import arch_model as _arch_model  # type: ignore[import]
    _ARCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ARCH_AVAILABLE = False
    _arch_model = None  # type: ignore[assignment]

# 涨跌停绝对收益率上界（A 股 ±10%，留出 0.5% 缓冲）
_LIMIT_THRESHOLD: float = 0.095
# 节假日后跳空：连续两个交易日的自然日间隔阈值
_HOLIDAY_GAP_DAYS: int = 3
# 节假日收益 Winsorize 倍数（3 倍本地 σ）
_HOLIDAY_WINSOR_SIGMA: float = 3.0


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _load_index_returns(
    benchmark: str,
    hist_start: str,
    hist_end_exclusive: pd.Timestamp,
) -> pd.Series:
    """
    从 Qlib 数据层加载基准指数日收益率序列。

    Args:
        benchmark: Qlib 行情格式代码，如 "SH000905"（CSI500）
        hist_start: 历史起始日期字符串，如 "2018-01-01"
        hist_end_exclusive: 截止日期（不含）

    Returns:
        升序 DatetimeIndex 的日收益率 Series；失败时返回空 Series
    """
    try:
        from qlib.data import D  # type: ignore[import]
    except ImportError:
        _logger.warning("[MarketTimer] qlib 不可用，无法加载指数收益率")
        return pd.Series(dtype=float)

    end_dt = hist_end_exclusive - pd.Timedelta(days=1)
    try:
        df = D.features(
            instruments=[benchmark],
            fields=["$close"],
            start_time=hist_start,
            end_time=end_dt,
            freq="day",
        )
    except Exception as exc:
        _logger.warning("[MarketTimer] 加载基准 %s 数据失败: %s", benchmark, exc)
        return pd.Series(dtype=float)

    if df is None or df.empty:
        return pd.Series(dtype=float)

    try:
        close = df["$close"].unstack(level=0).iloc[:, 0].sort_index()
    except Exception:
        try:
            close = df["$close"].sort_index()
        except Exception:
            return pd.Series(dtype=float)

    rets = close.pct_change(fill_method=None).iloc[1:].dropna()
    # 确保 index 是 DatetimeIndex（Qlib 返回的时间戳在某些版本可能为 object 类型）
    rets.index = pd.DatetimeIndex(rets.index)
    return rets.rename("index_ret")


def _winsorize_a_share(returns: pd.Series) -> pd.Series:
    """
    A 股专用预处理：
      1. 将 |ret| ≥ _LIMIT_THRESHOLD 的样本截断至 ±(_LIMIT_THRESHOLD - 0.001)
      2. 识别节假日后复市跳空，将其 Winsorize 至本地 3σ

    目的：防止涨跌停极端值扭曲 GARCH 参数估计，同时保留波动率信息。
    """
    if returns.empty:
        return returns

    s = returns.copy()

    # 1. 涨跌停截断
    clip_val = _LIMIT_THRESHOLD - 0.001
    s = s.clip(lower=-clip_val, upper=clip_val)

    # 2. 节假日复市跳空处理
    if len(s) < 10:
        return s
    dates = s.index
    gaps = pd.Series(
        (dates[1:] - dates[:-1]).days, index=dates[1:]
    )
    holiday_dates = gaps[gaps > _HOLIDAY_GAP_DAYS].index
    if len(holiday_dates) == 0:
        return s

    sigma_local = float(s.std(ddof=1))
    if sigma_local <= 0 or not np.isfinite(sigma_local):
        return s

    bound = _HOLIDAY_WINSOR_SIGMA * sigma_local
    for dt in holiday_dates:
        if dt in s.index:
            s.loc[dt] = float(np.clip(s.loc[dt], -bound, bound))

    return s


# ─────────────────────────────────────────────────────────────────────────────
# 抽象基类
# ─────────────────────────────────────────────────────────────────────────────

class MarketTimerBase(ABC):
    """MarketTimer 抽象基类，统一接口签名。"""

    @abstractmethod
    def get_risk_factor(self, trade_date: pd.Timestamp) -> float:
        """
        返回 trade_date 当日的仓位缩放因子（0 < factor ≤ max_risk）。

        实现必须保证：仅使用 trade_date 之前的数据（严格无前视）。

        Args:
            trade_date: 调仓日（Qlib trade_start_time，即下单执行日）

        Returns:
            float，通常在 [min_risk, max_risk] 之间
        """

    def describe(self) -> str:
        return self.__class__.__name__


# ─────────────────────────────────────────────────────────────────────────────
# 实现 1：NullTimer（基线，始终满仓）
# ─────────────────────────────────────────────────────────────────────────────

class NullTimer(MarketTimerBase):
    """
    恒等定时器：始终返回 1.0，仓位不缩放。

    用途：作为对比基线（A/B 测试），与 RollingVolTimer / GarchVolTimer 比较
    时保持其他配置完全一致。
    """

    def get_risk_factor(self, trade_date: pd.Timestamp) -> float:  # noqa: ARG002
        return 1.0

    def describe(self) -> str:
        return "NullTimer(always=1.0)"


# ─────────────────────────────────────────────────────────────────────────────
# 实现 2：RollingVolTimer（滚动标准差，arch 缺失时降级）
# ─────────────────────────────────────────────────────────────────────────────

class RollingVolTimer(MarketTimerBase):
    """
    基于滚动标准差的目标波动率策略。

    σ̂_{t+1} = std(ret_{t-lookback:t}) × √252

    相对 GARCH 的特点：
      - 等权历史窗口，对近期波动反应较慢
      - 无参数拟合，稳健性更高但精度略低
      - 可作为 GarchVolTimer 的降级方案
    """

    def __init__(
        self,
        benchmark: str,
        target_vol: float = 0.15,
        min_risk: float = 0.30,
        max_risk: float = 1.0,
        lookback: int = 60,
        min_obs: int = 60,
        hist_start: str = "2018-01-01",
        _index_returns: Optional[pd.Series] = None,
    ) -> None:
        if not (0 < min_risk <= max_risk <= 2.0):
            raise ValueError(
                f"需满足 0 < min_risk ({min_risk}) ≤ max_risk ({max_risk}) ≤ 2.0"
            )
        if target_vol <= 0:
            raise ValueError(f"target_vol 必须 > 0，当前: {target_vol}")
        if lookback < 5:
            raise ValueError(f"lookback 必须 ≥ 5，当前: {lookback}")

        self.benchmark = benchmark
        self.target_vol = target_vol
        self.min_risk = min_risk
        self.max_risk = max_risk
        self.lookback = lookback
        self.min_obs = min_obs
        self.hist_start = hist_start

        # 允许测试时直接注入 Series，避免 qlib 依赖
        if _index_returns is not None:
            self._index_returns = _index_returns
        else:
            self._index_returns = pd.Series(dtype=float)
        self._loaded = _index_returns is not None
        self._data_end: Optional[pd.Timestamp] = None

    def _ensure_data(self, trade_date: pd.Timestamp) -> None:  # noqa: ARG002
        """一次性加载全量可用历史数据。

        关键修复：不再按 trade_date 动态扩展，改为首次调用时加载所有可用数据。
        Qlib 数据层在回测时使用冻结快照，全量一次加载即可覆盖整个回测区间。
        即使 end_time 传入远期日期，Qlib 也只返回实际存在的数据（无前视风险）。
        get_risk_factor 内部仍严格截止 trade_date - 1 day，保证无前视偏差。
        """
        if self._loaded:
            return
        # 传入远期 end_date 让 Qlib 返回全部可用数据；实际只会返回到最新数据日
        far_future = pd.Timestamp("2099-01-01")
        loaded = _load_index_returns(self.benchmark, self.hist_start, far_future)
        if not loaded.empty:
            self._index_returns = loaded
            self._data_end = loaded.index[-1]
        self._loaded = True

    def get_risk_factor(self, trade_date: pd.Timestamp) -> float:
        self._ensure_data(trade_date)

        # 严格截止 trade_date 前一日（无前视保证）
        cutoff = pd.Timestamp(trade_date) - pd.Timedelta(days=1)
        idx = pd.DatetimeIndex(self._index_returns.index)
        hist = self._index_returns[idx <= cutoff]
        hist = _winsorize_a_share(hist).tail(self.lookback)

        if len(hist) < self.min_obs:
            _logger.debug(
                "[RollingVolTimer] %s 观测不足 %d，返回 1.0", trade_date, self.min_obs
            )
            return 1.0

        sigma_daily = float(hist.std(ddof=1))
        if sigma_daily <= 0 or not np.isfinite(sigma_daily):
            return 1.0

        sigma_ann = sigma_daily * np.sqrt(252)
        factor = self.target_vol / sigma_ann
        result = float(np.clip(factor, self.min_risk, self.max_risk))
        _logger.debug(
            "[RollingVolTimer] %s σ_ann=%.4f factor=%.4f",
            trade_date, sigma_ann, result,
        )
        return result

    def describe(self) -> str:
        return (
            f"RollingVolTimer(benchmark={self.benchmark}, "
            f"target_vol={self.target_vol:.2%}, "
            f"lookback={self.lookback}日, "
            f"risk=[{self.min_risk:.2f},{self.max_risk:.2f}])"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 实现 3：GarchVolTimer（GARCH(1,1) + Student-t，主推）
# ─────────────────────────────────────────────────────────────────────────────

class GarchVolTimer(MarketTimerBase):
    """
    基于 GARCH(1,1) + Student-t 分布的目标波动率策略。

    核心优势（相对滚动 STD）：
      - 指数衰减权重：近期波动权重更高，反应更及时
      - 条件方差自回归：显式建模波动聚类（α · ε² + β · σ²）
      - Student-t 误差项：捕捉 A 股厚尾分布（比正态假设更准确）
      - 前瞻 1 步：每日产出"明日预测波动率"，而非"昨日实现波动率"

    若 arch 包不可用，自动降级为 RollingVolTimer 并记录警告。

    拟合频率：每 refit_freq 个交易日重拟合一次，平衡计算效率与模型新鲜度。
    """

    def __init__(
        self,
        benchmark: str,
        target_vol: float = 0.15,
        min_risk: float = 0.30,
        max_risk: float = 1.0,
        refit_freq: int = 5,
        min_obs: int = 120,
        hist_start: str = "2018-01-01",
        _index_returns: Optional[pd.Series] = None,
    ) -> None:
        if not (0 < min_risk <= max_risk <= 2.0):
            raise ValueError(
                f"需满足 0 < min_risk ({min_risk}) ≤ max_risk ({max_risk}) ≤ 2.0"
            )
        if target_vol <= 0:
            raise ValueError(f"target_vol 必须 > 0，当前: {target_vol}")
        if refit_freq < 1:
            raise ValueError(f"refit_freq 必须 ≥ 1，当前: {refit_freq}")
        if min_obs < 30:
            raise ValueError(f"min_obs 必须 ≥ 30，当前: {min_obs}")

        self.benchmark = benchmark
        self.target_vol = target_vol
        self.min_risk = min_risk
        self.max_risk = max_risk
        self.refit_freq = refit_freq
        self.min_obs = min_obs
        self.hist_start = hist_start

        if _index_returns is not None:
            self._index_returns = _index_returns
        else:
            self._index_returns = pd.Series(dtype=float)
        self._loaded = _index_returns is not None
        self._data_end: Optional[pd.Timestamp] = None

        # GARCH 拟合状态缓存
        self._fitted_result: Any = None
        self._last_refit_date: Optional[pd.Timestamp] = None
        self._refit_call_count: int = 0

        # arch 不可用时降级
        if not _ARCH_AVAILABLE:
            _logger.warning(
                "[GarchVolTimer] arch 包不可用，将降级为 RollingVolTimer。"
                "安装方法：pip install arch>=7.0"
            )
            self._fallback: Optional[RollingVolTimer] = RollingVolTimer(
                benchmark=benchmark,
                target_vol=target_vol,
                min_risk=min_risk,
                max_risk=max_risk,
                lookback=60,
                min_obs=60,
                hist_start=hist_start,
                _index_returns=_index_returns,
            )
        else:
            self._fallback = None

    def _ensure_data(self, trade_date: pd.Timestamp) -> None:  # noqa: ARG002
        """一次性加载全量可用历史数据（与 RollingVolTimer 相同修复）。"""
        if self._loaded:
            return
        far_future = pd.Timestamp("2099-01-01")
        loaded = _load_index_returns(self.benchmark, self.hist_start, far_future)
        if not loaded.empty:
            self._index_returns = loaded
            self._data_end = loaded.index[-1]
        self._loaded = True

    def _needs_refit(self, trade_date: pd.Timestamp) -> bool:
        if self._fitted_result is None or self._last_refit_date is None:
            return True
        self._refit_call_count += 1
        return self._refit_call_count % self.refit_freq == 0

    def _fit_garch(self, returns_pct: pd.Series) -> bool:
        """
        用百分比收益率序列拟合 GARCH(1,1)-t 模型。

        使用百分比（×100）是为了提升数值稳定性（arch 内部优化器对量级敏感）。

        Returns:
            True: 拟合成功；False: 失败（调用方应降级处理）
        """
        if not _ARCH_AVAILABLE or _arch_model is None:
            return False
        try:
            model = _arch_model(
                returns_pct,
                vol="Garch",
                p=1,
                q=1,
                dist="t",
                mean="Zero",
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(
                    disp="off",
                    show_warning=False,
                    options={"maxiter": 200, "ftol": 1e-6},
                )
            # 简单收敛检查：ω > 0，α + β < 1
            params = result.params
            omega = float(params.get("omega", params.iloc[0]))
            alpha = float(params.get("alpha[1]", params.iloc[1]))
            beta = float(params.get("beta[1]", params.iloc[2]))
            if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1.0:
                _logger.warning(
                    "[GarchVolTimer] 参数不合理(ω=%.4f α=%.4f β=%.4f)，降级",
                    omega, alpha, beta,
                )
                return False
            self._fitted_result = result
            return True
        except Exception as exc:
            _logger.warning("[GarchVolTimer] GARCH 拟合失败: %s", exc)
            return False

    def _forecast_sigma_ann(self) -> Optional[float]:
        """
        从已拟合结果中提取 1 步前瞻条件标准差（年化）。

        Returns:
            年化波动率（如 0.20 表示 20%）；失败时返回 None
        """
        if self._fitted_result is None:
            return None
        try:
            fc = self._fitted_result.forecast(horizon=1, reindex=False)
            # arch 返回的方差单位为百分比²
            variance_pct2 = float(fc.variance.iloc[-1, 0])
            if variance_pct2 <= 0 or not np.isfinite(variance_pct2):
                return None
            sigma_daily_pct = np.sqrt(variance_pct2)
            sigma_daily = sigma_daily_pct / 100.0
            return sigma_daily * np.sqrt(252)
        except Exception as exc:
            _logger.warning("[GarchVolTimer] forecast 失败: %s", exc)
            return None

    def get_risk_factor(self, trade_date: pd.Timestamp) -> float:
        # arch 不可用：降级到 RollingVolTimer
        if self._fallback is not None:
            return self._fallback.get_risk_factor(trade_date)

        self._ensure_data(trade_date)

        # 严格截止 trade_date 前一日
        cutoff = pd.Timestamp(trade_date) - pd.Timedelta(days=1)
        idx = pd.DatetimeIndex(self._index_returns.index)
        hist_raw = self._index_returns[idx <= cutoff]

        if len(hist_raw) < self.min_obs:
            _logger.debug(
                "[GarchVolTimer] %s 观测不足 %d，返回 1.0（冷启动）",
                trade_date, self.min_obs,
            )
            return 1.0

        hist = _winsorize_a_share(hist_raw)
        hist_pct = hist * 100.0

        if self._needs_refit(trade_date):
            success = self._fit_garch(hist_pct)
            if success:
                self._last_refit_date = trade_date
                _logger.debug("[GarchVolTimer] %s 重拟合 GARCH 成功", trade_date)
            else:
                # 拟合失败：用滚动 STD 兜底
                sigma_daily = float(hist.tail(60).std(ddof=1))
                sigma_ann = sigma_daily * np.sqrt(252)
                if sigma_ann <= 0 or not np.isfinite(sigma_ann):
                    return 1.0
                factor = self.target_vol / sigma_ann
                return float(np.clip(factor, self.min_risk, self.max_risk))

        sigma_ann = self._forecast_sigma_ann()
        if sigma_ann is None or sigma_ann <= 0 or not np.isfinite(sigma_ann):
            _logger.warning("[GarchVolTimer] %s forecast 无效，返回 1.0", trade_date)
            return 1.0

        factor = self.target_vol / sigma_ann
        result = float(np.clip(factor, self.min_risk, self.max_risk))
        _logger.debug(
            "[GarchVolTimer] %s σ_ann=%.4f factor=%.4f",
            trade_date, sigma_ann, result,
        )
        return result

    def describe(self) -> str:
        backend = "GARCH(1,1)-t" if _ARCH_AVAILABLE else "RollingVol(降级)"
        return (
            f"GarchVolTimer(backend={backend}, "
            f"benchmark={self.benchmark}, "
            f"target_vol={self.target_vol:.2%}, "
            f"refit_freq={self.refit_freq}日, "
            f"min_obs={self.min_obs}, "
            f"risk=[{self.min_risk:.2f},{self.max_risk:.2f}])"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────────────────────────────────────

def build_market_timer(
    cfg: Optional[Dict[str, Any]],
    _index_returns: Optional[pd.Series] = None,
) -> Optional[MarketTimerBase]:
    """
    从 YAML 配置字典构建 MarketTimer 实例。

    Args:
        cfg: YAML market_timer_cfg 节对应的 dict，例如：
            {
                "type": "garch",          # garch | rolling | null
                "benchmark": "SH000905",  # CSI500 指数代码
                "target_vol": 0.15,
                "min_risk": 0.30,
                "max_risk": 1.0,
                "refit_freq": 5,          # 仅 garch 有效
                "min_obs": 120,
                "hist_start": "2018-01-01",
            }
        _index_returns: 测试专用注入参数，生产环境不传

    Returns:
        MarketTimerBase 实例；cfg 为 None 或 type="null" 时返回 NullTimer
    """
    if not cfg:
        return None

    timer_type = str(cfg.get("type", "null")).lower()

    if timer_type == "null":
        return NullTimer()

    benchmark: str = str(cfg.get("benchmark", "000905.SZ"))  # A 股 Qlib 格式
    target_vol: float = float(cfg.get("target_vol", 0.15))
    min_risk: float = float(cfg.get("min_risk", 0.30))
    max_risk: float = float(cfg.get("max_risk", 1.0))
    min_obs: int = int(cfg.get("min_obs", 120))
    hist_start: str = str(cfg.get("hist_start", "2018-01-01"))

    if timer_type == "rolling":
        lookback: int = int(cfg.get("lookback", 60))
        return RollingVolTimer(
            benchmark=benchmark,
            target_vol=target_vol,
            min_risk=min_risk,
            max_risk=max_risk,
            lookback=lookback,
            min_obs=min(min_obs, lookback),
            hist_start=hist_start,
            _index_returns=_index_returns,
        )

    if timer_type == "garch":
        refit_freq: int = int(cfg.get("refit_freq", 5))
        return GarchVolTimer(
            benchmark=benchmark,
            target_vol=target_vol,
            min_risk=min_risk,
            max_risk=max_risk,
            refit_freq=refit_freq,
            min_obs=min_obs,
            hist_start=hist_start,
            _index_returns=_index_returns,
        )

    raise ValueError(
        f"未知 market_timer type: {timer_type!r}，"
        "支持的类型: null | rolling | garch"
    )
