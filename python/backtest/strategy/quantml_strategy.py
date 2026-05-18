"""
QuantML 自定义策略类（QuantMLWeightStrategy）
================================================
继承 Qlib `WeightStrategyBase`，串联五段式管线：
    pred_score
        → TopKSelector（候选选择）
        → Weighter（权重计算：等权 / IV / RP / Score）
        → RiskConstraints（单票 + 板块 + 换手 三层约束）
        → 返回 dict[stock_id → weight] 给 Qlib OrderGenerator

关键能力：
  1. **调仓节流（rebalance_freq）**：仅每 N 个交易步用模型重算权重；
     若配置了 **exit_rules**（止盈/止损/移动止损），非模型步仍会询价并可能下达平仓单。
  2. **历史收益缓存**：用 qlib.data.D.features 拉收盘价并转为日收益率，
     按 (trade_start_time, lookback) 缓存，TTL = 一个调仓周期
  3. **未来函数防护**：所有数据访问严格使用 trade_start_time 之前的日期
  4. **优雅降级**：拉取失败 / 候选为空时，保留上期持仓不动

接口规范（被 Qlib 在每个交易步调用）：
    generate_target_weight_position(score, current, trade_start_time, trade_end_time)
        → dict[stock_id, weight]
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

try:
    from qlib.contrib.strategy.signal_strategy import WeightStrategyBase
    from qlib.backtest.decision import TradeDecisionWO
except ImportError:  # 测试环境无 qlib 时降级，单测仅用 _compute_target_weights
    WeightStrategyBase = object  # type: ignore[misc,assignment]
    TradeDecisionWO = None  # type: ignore[assignment,misc]

_logger = logging.getLogger("QuantMLWeightStrategy")

from .selector import TopKSelector, AdaptiveTopKCfg
from .weighter import build_weighter
from .constraints import RiskConstraints
from .exit_rules import (
    ExitRulesConfig,
    parse_exit_rules_cfg,
    renormalize_target,
    strip_exited_symbols,
)
from .stock_filter import StockFilter, build_stock_filter
from .market_timer import MarketTimerBase, build_market_timer


class QuantMLWeightStrategy(WeightStrategyBase):
    """
    五段式权重策略：信号 → 选股 → 加权 → 约束 → 提交。

    Args（kwargs）:
        topk: 候选股票数量（默认 25）
        score_quantile: 截面分位过滤阈值（默认 0.0 = 不过滤；推荐 0.7）
        adaptive_topk_cfg: 信号强度自适应 topk 配置（dict or None）；
            可选键：strong_topk / weak_topk / strong_quantile / weak_quantile /
            strength_high / strength_low / strength_window / cold_start_min
        rebalance_freq: 调仓周期（每 N 个 trade_step 重算一次权重，默认 10 = 双周）
        vol_lookback: 计算波动率/协方差的回看天数（默认 60 = 三个月）
        weighter_cfg: dict {"type": "inverse_vol", "kwargs": {...}}
        risk_cfg: dict {"max_weight": 0.08, "sector_max_weight": 0.30, "max_turnover": 0.30}
        n_drop: int | None 每个模型调仓步**最多换出**的票数（仅"上期持有→本期目标外"
            的剔除计入，止盈止损平仓不占额度）。None / 0 表示不限制（默认）。
            离散控制配 `risk_cfg.max_turnover` 的连续控制可双层防护。
        exit_rules: dict | None 止盈止损，可选键：
            stop_loss_pct / take_profit_pct / trailing_stop_pct（正小数，如 0.08=8%）
        stock_filter_cfg: dict | None 股票过滤配置，可选键：
            st_csv_path (str): is_st.csv 路径
            consecutive_limit_days (int): 连续涨停天数阈值（0=不过滤）
            exclude_st (bool): 是否过滤 ST，默认 True
            exclude_limit_up (bool): 是否过滤连续涨停，默认 True
        market_timer_cfg: dict | None 市场波动定时器配置（目标波动率策略），可选键：
            type (str): garch | rolling | null（默认 null）
            benchmark (str): 基准指数代码，如 "SH000905"（CSI500）
            target_vol (float): 年化目标波动率，如 0.15（15%）
            min_risk (float): 仓位缩放下限，如 0.30（最低持仓 30%）
            max_risk (float): 仓位缩放上限，默认 1.0（不加杠杆）
            refit_freq (int): GARCH 重拟合间隔（交易日数，仅 garch 有效）
            min_obs (int): 冷启动期（观测不足时返回 1.0）
            hist_start (str): 历史数据起始日期
        其它 kwargs 传给父类（signal/trade_exchange/risk_degree 等）
    """

    def __init__(
        self,
        *,
        topk: int = 25,
        score_quantile: float = 0.0,
        adaptive_topk_cfg: Optional[Dict[str, Any]] = None,
        rebalance_freq: int = 10,
        vol_lookback: int = 60,
        weighter_cfg: Optional[Dict[str, Any]] = None,
        risk_cfg: Optional[Dict[str, Any]] = None,
        n_drop: Optional[int] = None,
        exit_rules: Optional[Dict[str, Any]] = None,
        stock_filter_cfg: Optional[Dict[str, Any]] = None,
        trade_unit: int = 100,
        market_timer_cfg: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        # 校验调仓周期与回看
        if not isinstance(rebalance_freq, int) or rebalance_freq < 1:
            raise ValueError(
                f"rebalance_freq 必须为正整数，当前: {rebalance_freq}"
            )
        if not isinstance(vol_lookback, int) or vol_lookback < 5:
            raise ValueError(
                f"vol_lookback 必须为 ≥5 的整数，当前: {vol_lookback}"
            )
        if n_drop is not None and (not isinstance(n_drop, int) or n_drop < 0):
            raise ValueError(
                f"n_drop 必须为非负整数或 None，当前: {n_drop!r}"
            )

        # 调用父类（仅当 qlib 可用时；避免单测 import 阶段失败）
        if WeightStrategyBase is not object:
            super().__init__(**kwargs)

        # 组装管线组件
        _adaptive_cfg: Optional[AdaptiveTopKCfg] = (
            AdaptiveTopKCfg(**adaptive_topk_cfg)
            if adaptive_topk_cfg and isinstance(adaptive_topk_cfg, dict)
            else None
        )
        self._selector = TopKSelector(
            topk=topk,
            score_quantile=score_quantile,
            adaptive_topk_cfg=_adaptive_cfg,
        )
        self._weighter = build_weighter(
            weighter_cfg or {"type": "inverse_vol", "kwargs": {}}
        )
        self._risk = RiskConstraints(**(risk_cfg or {}))

        self.topk = topk
        self.score_quantile = score_quantile
        self.rebalance_freq = rebalance_freq
        self.vol_lookback = vol_lookback
        # 0 与 None 等价：不限制每期换出的票数
        self.n_drop: Optional[int] = n_drop if (n_drop is not None and n_drop > 0) else None
        self._exit_cfg: ExitRulesConfig = parse_exit_rules_cfg(exit_rules)

        # 股票过滤器（ST + 连续涨停）
        self._stock_filter: Optional[StockFilter] = build_stock_filter(stock_filter_cfg)
        self._filter_exclude_st: bool = (
            stock_filter_cfg.get("exclude_st", True) if stock_filter_cfg else True
        )
        self._filter_exclude_limit_up: bool = (
            stock_filter_cfg.get("exclude_limit_up", True) if stock_filter_cfg else True
        )
        if not isinstance(trade_unit, int) or trade_unit <= 0:
            raise ValueError(f"trade_unit 必须为正整数，当前: {trade_unit}")
        self.trade_unit: int = trade_unit

        # 调仓状态
        self._step_counter: int = 0           # 已调用的 trade_step 数
        self._last_target_weights: Dict[str, float] = {}  # 上次产出的目标权重
        # 止盈止损参考（入场参考价 + 峰值，用于移动止损）；键为 stock_id
        self._entry_ref: Dict[str, float] = {}
        self._peak_ref: Dict[str, float] = {}
        # 止盈止损触发事件记录（含 datetime/code/reason/entry_ref/mark/peak/pnl_pct）
        self._exit_events: List[Dict[str, Any]] = []
        # 由 generate_trade_decision 在调用父类前设置，供 generate_target_weight_position 分支
        self._pending_model_rebalance: bool = True

        # 收益率历史缓存（按标的存储全量收盘价，避免每次调仓都重新拉取 Qlib 数据）
        # 设计：每只标的只加载一次（end_time=far_future），后续 _fetch_return_history 直接切片
        # 无前视风险：_fetch_return_history 内部严格截止 end_time_exclusive - 1day
        self._close_price_cache: Dict[str, pd.Series] = {}

        # 市场波动定时器（目标波动率策略，末端仓位缩放）
        self._market_timer: Optional[MarketTimerBase] = build_market_timer(
            market_timer_cfg
        )
        # risk_factor 滞后阈值：只有变化幅度超过此值才触发仓位调整，防止 GARCH 每日
        # 小幅波动产生大量无效微调订单。取 market_timer_cfg 中的 min_change 参数
        # （绝对值，如 0.05 = 5%）。0.0 表示每日都应用（不过滤）。
        self._rf_min_change: float = (
            float(market_timer_cfg.get("min_change", 0.05))
            if market_timer_cfg else 0.0
        )
        self._applied_risk_factor: float = 1.0  # 上次实际生效的 risk_factor

    # ─────────────────────────────────────────────────────────────────────
    # 批量预加载（供 BacktestEngine 在回测启动前调用）
    # ─────────────────────────────────────────────────────────────────────

    def preload_instruments(self, pred_score: pd.Series) -> None:
        """
        在 Qlib 回测循环启动前，一次性批量加载全部候选标的的历史收盘价。

        动机：_fetch_return_history 与 StockFilter.get_limit_up_set 均需要
        D.features 调用。若不提前加载，每出现一只新股票就触发一次独立的 D.features
        调用（7-14 秒/次），340 天回测中出现 100+ 只股票 = 700-1400 秒额外开销。
        批量预加载将所有 D.features 调用合并为 1 次，消除回测中的反复磁盘读取。

        Args:
            pred_score: 完整预测信号（MultiIndex: datetime × instrument）

        副作用：
            - 填充 self._close_price_cache（供 _fetch_return_history 使用）
            - 填充 self._stock_filter._close_cache（供 StockFilter 使用）
        """
        try:
            from qlib.data import D
        except ImportError:
            return

        if isinstance(pred_score.index, pd.MultiIndex):
            all_insts = list(pred_score.index.get_level_values("instrument").unique())
        else:
            _logger.debug("[preload] pred_score 索引非 MultiIndex，跳过预加载")
            return

        if not all_insts:
            return

        _logger.info("[preload] 批量预加载 %d 个标的收盘价历史（一次性 D.features）...", len(all_insts))
        try:
            # 同时加载复权价（InverseVolWeighter）和原始价（StockFilter 涨停检测）
            df = D.features(
                all_insts,
                fields=["$close_qfq", "$close"],
                start_time="2015-01-01",
                end_time="2099-01-01",
                freq="day",
            )
            if df is None or df.empty:
                _logger.warning("[preload] 加载结果为空，跳过预加载")
                return

            close_qfq_all = df["$close_qfq"].unstack(level=0).sort_index() if "$close_qfq" in df.columns else pd.DataFrame()
            close_all = df["$close"].unstack(level=0).sort_index() if "$close" in df.columns else pd.DataFrame()

            loaded_n = 0
            for inst in all_insts:
                # 复权价 → _fetch_return_history 缓存
                if inst not in self._close_price_cache:
                    if not close_qfq_all.empty and inst in close_qfq_all.columns:
                        self._close_price_cache[inst] = close_qfq_all[inst].dropna()
                    else:
                        self._close_price_cache[inst] = pd.Series(dtype=float)
                # 原始价 → StockFilter 缓存
                if self._stock_filter is not None and inst not in self._stock_filter._close_cache:
                    if not close_all.empty and inst in close_all.columns:
                        self._stock_filter._close_cache[inst] = close_all[inst].dropna()
                    else:
                        self._stock_filter._close_cache[inst] = pd.Series(dtype=float)
                loaded_n += 1

            _logger.info("[preload] 完成：%d 个标的 close_qfq 已缓存，%d 个 close 已缓存",
                        sum(1 for v in self._close_price_cache.values() if not v.empty),
                        sum(1 for v in (self._stock_filter._close_cache.values() if self._stock_filter else [])))
        except Exception as exc:
            _logger.warning("[preload] 批量加载失败（%s），回测中将按需加载（较慢）", exc)

    # ─────────────────────────────────────────────────────────────────────
    # Qlib 入口
    # ─────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────
    # Qlib 顶层入口：override generate_trade_decision（节流 + 可选每日平仓规则）
    # ─────────────────────────────────────────────────────────────────────

    def generate_trade_decision(self, execute_result=None):  # type: ignore[override]
        """
        在父类基础上加调仓节流，并可选每日止盈止损：

          - **仅中频、未配置 exit_rules**：非模型调仓日返回空订单，
            杜绝因价格漂移产生的维持权重微调单。
          - **配置了 exit_rules**：非模型调仓日仍走父类流程，仅在
            `generate_target_weight_position` 中沿用上期目标并应用平仓规则。
          - **模型调仓日**：走父类完整流程（信号 → 权重 → 订单）。
        """
        if TradeDecisionWO is None:  # 单测降级路径
            return None

        # 计数本次调用
        self._step_counter += 1
        is_model_step = (
            self._step_counter == 1
            or (self._step_counter - 1) % self.rebalance_freq == 0
        )
        exit_on = self._exit_cfg.is_enabled()

        # 非模型调仓日且无退出规则：零订单（中频换手控制）
        if not is_model_step and not exit_on:
            return TradeDecisionWO([], self)

        self._pending_model_rebalance = is_model_step
        raw_decision = super().generate_trade_decision(execute_result=execute_result)
        return self._apply_trade_unit_rounding(raw_decision)

    def _apply_trade_unit_rounding(self, decision: Any) -> Any:
        """
        将订单股数裁剪到 trade_unit 的整数倍（默认 A 股 100 股一手）。

        规则：
          - amount < trade_unit 的订单直接丢弃
          - 其余订单按 floor(amount / trade_unit) * trade_unit 向下取整
        """
        if decision is None or TradeDecisionWO is None:
            return decision
        order_list = getattr(decision, "order_list", None)
        if not isinstance(order_list, list) or not order_list:
            return decision

        rounded_orders: List[Any] = []
        dropped = 0
        adjusted = 0
        for od in order_list:
            try:
                amount = float(getattr(od, "amount", 0.0))
                if amount <= 0:
                    dropped += 1
                    continue
                rounded = int(amount // self.trade_unit) * self.trade_unit
                if rounded <= 0:
                    dropped += 1
                    continue
                if rounded != int(amount):
                    adjusted += 1
                    setattr(od, "amount", rounded)
                rounded_orders.append(od)
            except Exception:
                # 异常订单保守起见直接丢弃，避免提交非法股数
                dropped += 1

        if adjusted > 0 or dropped > 0:
            _logger.info(
                "[trade_unit] unit=%d adjusted=%d dropped=%d original=%d kept=%d",
                self.trade_unit, adjusted, dropped, len(order_list), len(rounded_orders)
            )
        return TradeDecisionWO(rounded_orders, self)

    def generate_target_weight_position(
        self,
        score: pd.Series,
        current: Any,
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
    ) -> Dict[str, float]:
        """
        由父类 `generate_trade_decision` 在每个**产生决策**的交易日调用。

        模型调仓日：用信号重算目标权并施加风险约束，再应用 exit_rules（若启用）。
        非模型调仓日：仅当启用 exit_rules 时会调用本函数，沿用上期目标权并检查平仓条件。
        """
        prev_w = self._extract_prev_weights(current)
        held_list = [str(c) for c, v in prev_w.items() if float(v) > 1e-12]

        if self._pending_model_rebalance:
            target = self._compute_target_weights(
                score=score,
                prev_w=prev_w,
                trade_start_time=trade_start_time,
            )
            if not target:
                cand = (
                    dict(self._last_target_weights)
                    if self._last_target_weights
                    else self._snapshot_current(current)
                )
                target = cand
        else:
            target = (
                dict(self._last_target_weights)
                if self._last_target_weights
                else self._snapshot_current(current)
            )

        target = {str(k): float(v) for k, v in target.items() if float(v) > 1e-12}

        codes_for_marks = sorted(set(held_list).union(target.keys()))
        marks = self._fetch_marks(
            codes_for_marks, trade_start_time, trade_end_time
        )

        if self._exit_cfg.is_enabled():
            eligible = [c for c in held_list if c in target]
            closed = strip_exited_symbols(
                target,
                eligible,
                marks,
                self._entry_ref,
                self._peak_ref,
                self._exit_cfg,
            )
            for ev in closed:
                ev_full = {
                    "datetime": pd.Timestamp(trade_start_time),
                    **ev,
                }
                self._exit_events.append(ev_full)
                _logger.warning(
                    "[exit] %s | code=%s reason=%s entry=%.4f mark=%.4f peak=%.4f pnl=%.2f%%",
                    ev_full["datetime"], ev_full["code"], ev_full["reason"],
                    ev_full["entry_ref"], ev_full["mark"], ev_full["peak"],
                    ev_full["pnl_pct"] * 100.0,
                )
            target = renormalize_target(target)

        # 更新入场参考与峰值（日频）；新进入 target 的标的以当前 mark 为参考价
        for code in list(target.keys()):
            mark = marks.get(code)
            px = QuantMLWeightStrategy._scalar_price(mark)
            if px is None or px <= 0:
                continue
            if code not in self._entry_ref:
                self._entry_ref[code] = px
            pk = self._peak_ref.get(code, self._entry_ref[code])
            self._peak_ref[code] = max(pk, px)

        for code in list(self._entry_ref.keys()):
            if code not in target:
                self._entry_ref.pop(code, None)
                self._peak_ref.pop(code, None)

        # MarketTimer：目标波动率仓位缩放（在 renormalize 之后，保持相对权重结构不变）
        # 滞后过滤：GARCH 每日微小波动（< rf_min_change）复用上次值，不触发微调订单
        if self._market_timer is not None and target:
            rf_new = self._market_timer.get_risk_factor(trade_start_time)
            if abs(rf_new - self._applied_risk_factor) >= self._rf_min_change:
                self._applied_risk_factor = rf_new
                _logger.debug(
                    "[MarketTimer] %s risk_factor 更新: %.4f → %.4f",
                    trade_start_time, rf_new, self._applied_risk_factor,
                )
            rf = self._applied_risk_factor
            if rf < 1.0 - 1e-6:
                target = {k: v * rf for k, v in target.items()}
                _logger.debug(
                    "[MarketTimer] %s 应用 risk_factor=%.4f（%.1f%%仓位）",
                    trade_start_time, rf, rf * 100,
                )

        self._last_target_weights = dict(target)
        return target

    def get_exit_events(self) -> pd.DataFrame:
        """
        返回回测中所有止盈止损触发记录（DataFrame，按时间排序）。

        Columns: datetime / code / reason / entry_ref / mark / peak / pnl_pct
        """
        cols = [
            "datetime", "code", "reason", "entry_ref", "mark", "peak", "pnl_pct",
        ]
        if not self._exit_events:
            return pd.DataFrame(columns=cols)
        df = pd.DataFrame(self._exit_events)
        # 兜底列序与排序
        for c in cols:
            if c not in df.columns:
                df[c] = pd.NA
        return df[cols].sort_values("datetime").reset_index(drop=True)

    @staticmethod
    def _snapshot_current(current: Any) -> Dict[str, float]:
        """从 qlib Position 提取当前权重 dict（包含现金以外的所有股票）。"""
        if current is None:
            return {}
        if hasattr(current, "get_stock_weight_dict"):
            try:
                return dict(current.get_stock_weight_dict(only_stock=True))
            except Exception:
                return {}
        if isinstance(current, dict):
            return dict(current)
        return {}

    # ─────────────────────────────────────────────────────────────────────
    # 核心计算逻辑（与 qlib 解耦，便于单测）
    # ─────────────────────────────────────────────────────────────────────

    def _compute_target_weights(
        self,
        score: pd.Series,
        prev_w: pd.Series,
        trade_start_time: pd.Timestamp,
    ) -> Dict[str, float]:
        """
        执行五段式管线（去掉 qlib 接口包装），返回 stock → weight 字典。
        所有数据访问严格小于 trade_start_time，杜绝未来函数。
        """
        if score is None or score.empty:
            return {}

        # 兼容 qlib Signal 返回的 DataFrame（含 'score' 列）
        if isinstance(score, pd.DataFrame):
            if "score" in score.columns:
                score = score["score"]
            elif score.shape[1] == 1:
                score = score.iloc[:, 0]
            else:
                return {}

        # ① Selector：候选选择
        candidates = self._selector.select(score)
        if not candidates:
            return {}

        # ① ½ 股票过滤（ST + 连续涨停），发生在加权前确保权重干净
        if self._stock_filter is not None:
            date_str = trade_start_time.strftime("%Y%m%d")
            candidates, excluded = self._stock_filter.filter_candidates(
                candidates,
                date=date_str,
                check_st=self._filter_exclude_st,
                check_limit_up=self._filter_exclude_limit_up,
            )
            if excluded:
                _logger.info(
                    "[%s] 过滤 %d 支候选: %s",
                    date_str,
                    len(excluded),
                    {k: v[:30] for k, v in excluded.items()},
                )
            if not candidates:
                _logger.warning(
                    "[%s] 过滤后候选为空，跳过本次调仓", date_str
                )
                return {}

        # ② 拉取历史收益率（仅用 trade_start_time 之前的数据）
        ret_history = self._fetch_return_history(
            candidates=candidates,
            end_time_exclusive=trade_start_time,
        )

        # ③ Weighter：权重计算
        target_w = self._weighter.weight(
            candidates=candidates,
            score=score,
            ret_history=ret_history,
        )
        if target_w.empty:
            return {}

        # ④ RiskConstraints：约束施加
        adjusted = self._risk.apply(target_w, prev_w=prev_w)
        if adjusted.empty:
            return {}

        target = {str(k): float(v) for k, v in adjusted.items() if v > 0}
        # ⑤ n_drop：限制每期换出的股票数（离散换手控制）
        return self._apply_n_drop(target=target, prev_w=prev_w, score=score)

    def _apply_n_drop(
        self,
        target: Dict[str, float],
        prev_w: pd.Series,
        score: pd.Series,
    ) -> Dict[str, float]:
        """
        若 `prev → target` 的剔除数量超过 `n_drop`，从「计划卖出」的标的中按
        score 由高到低保留 `len(want_to_sell) - n_drop` 只，重新加回 target，
        并归一化到 sum=1。

        语义：与 Qlib 内置 TopkDropoutStrategy 的 `n_drop` 一致——离散地限制
        每期换出多少只票。模型最看好的票优先进 target，被迫保留的"边缘票"
        从计划卖出列表里按 score 倒序挑。
        """
        if self.n_drop is None:
            return target
        if prev_w is None or prev_w.empty:
            return target

        prev_codes = {
            str(c) for c, v in prev_w.items() if float(v) > 1e-12
        }
        target_codes = set(target.keys())
        want_to_sell = prev_codes - target_codes
        if len(want_to_sell) <= self.n_drop:
            return target

        keep_n = len(want_to_sell) - self.n_drop

        if score is not None and not score.empty:
            score_dict = {
                str(k): float(v)
                for k, v in score.items()
                if pd.notna(v)
            }
            keep_codes = sorted(
                want_to_sell,
                key=lambda c: score_dict.get(c, float("-inf")),
                reverse=True,
            )[:keep_n]
        else:
            keep_codes = sorted(
                want_to_sell,
                key=lambda c: float(prev_w.get(c, 0.0)),
                reverse=True,
            )[:keep_n]

        new_target = dict(target)
        for code in keep_codes:
            new_target[code] = float(prev_w.get(code, 0.0))

        total = sum(max(0.0, v) for v in new_target.values())
        if total <= 1e-12:
            return target
        return {k: max(0.0, v) / total for k, v in new_target.items() if v > 0}

    def _fetch_marks(
        self,
        codes: List[str],
        trade_start_time: pd.Timestamp,
        trade_end_time: pd.Timestamp,
    ) -> Dict[str, float]:
        """按决策区间拉取各标的收盘价（与 Qlib 日频撮合一致）。"""
        ex = getattr(self, "trade_exchange", None)
        if ex is None or not codes:
            return {}
        out: Dict[str, float] = {}
        for sid in codes:
            try:
                q = ex.get_close(sid, trade_start_time, trade_end_time)
            except Exception:
                continue
            px = self._scalar_price(q)
            if px is not None and px > 0:
                out[str(sid)] = px
        return out

    @staticmethod
    def _scalar_price(x: Union[None, int, float, np.floating, Any]) -> Optional[float]:
        if x is None:
            return None
        if isinstance(x, (int, float, np.floating)):
            v = float(x)
            return v if np.isfinite(v) else None
        try:
            raw = getattr(x, "data", x)
            arr = np.asarray(raw).ravel()
            if arr.size == 0:
                return None
            v = float(arr[0])
            return v if np.isfinite(v) else None
        except Exception:
            return None

    # ─────────────────────────────────────────────────────────────────────
    # 数据访问
    # ─────────────────────────────────────────────────────────────────────

    def _fetch_return_history(
        self,
        candidates: List[str],
        end_time_exclusive: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        拉取候选股票最近 vol_lookback 个交易日的日收益率序列。

        性能优化：按标的缓存全量收盘价，每只股票只做一次 D.features 调用。
        在长回测（数百日 × 日频调仓）下可将 Qlib 查询次数从 O(T) 降至 O(K)
        （T=交易日数，K=出现过的不同标的数量）。

        关键合规：
          - end_time_exclusive 严格 *不* 包含在内（防止读未来）
          - 即使缓存包含未来价格数据，切片也只取 end_time_exclusive - 1day 之前的数据
          - 无前视风险：切片逻辑与原来完全一致

        Returns:
            DataFrame, index=DatetimeIndex（升序），columns=candidates，value=日收益率
            拉取失败返回空 DataFrame（让 Weighter 自行降级）
        """
        try:
            from qlib.data import D
        except ImportError:
            return pd.DataFrame()

        # end 之前一天为窗口右界（无前视）
        end_dt = pd.Timestamp(end_time_exclusive) - pd.Timedelta(days=1)

        # 找出尚未缓存的标的，批量加载
        missing = [c for c in candidates if c not in self._close_price_cache]
        if missing:
            # 使用远期 end_time 一次性加载所有历史，避免每天重复查询
            # Qlib 只返回实际存在的数据，不会引入未来数据
            far_start = pd.Timestamp("2015-01-01")  # 足够早，覆盖所有 lookback
            far_future = pd.Timestamp("2099-01-01")
            try:
                df = D.features(
                    instruments=missing,
                    fields=["$close_qfq"],
                    start_time=far_start,
                    end_time=far_future,
                    freq="day",
                )
                if df is not None and not df.empty:
                    try:
                        close_all = df["$close_qfq"].unstack(level=0).sort_index()
                    except Exception:
                        close_all = pd.DataFrame()
                    for inst in missing:
                        if inst in close_all.columns:
                            self._close_price_cache[inst] = close_all[inst].dropna()
                        else:
                            # 标的无数据：缓存空 Series 避免重复查询
                            self._close_price_cache[inst] = pd.Series(dtype=float)
            except Exception:
                # 加载失败：所有 missing 标的缓存空 Series
                for inst in missing:
                    if inst not in self._close_price_cache:
                        self._close_price_cache[inst] = pd.Series(dtype=float)

        # 从缓存中切片（严格截止 end_dt，无前视）
        frames: Dict[str, pd.Series] = {}
        # 窗口起始（自然日缓冲，确保有足够 lookback 个交易日）
        start_dt = end_dt - pd.Timedelta(days=int(self.vol_lookback * 2 + 30))
        for inst in candidates:
            cached = self._close_price_cache.get(inst)
            if cached is None or cached.empty:
                continue
            sliced = cached[(cached.index >= start_dt) & (cached.index <= end_dt)]
            if not sliced.empty:
                frames[inst] = sliced

        if not frames:
            return pd.DataFrame()

        close = pd.DataFrame(frames).sort_index()
        # 日收益率（显式 fill_method=None 抑制 FutureWarning）
        rets = close.pct_change(fill_method=None).iloc[1:]
        rets = rets.reindex(columns=candidates)
        return rets

    @staticmethod
    def _extract_prev_weights(current: Any) -> pd.Series:
        """
        从 qlib Position 对象中提取上期组合权重（仅股票，不含现金）。
        其它对象类型（如 dict）也兼容。
        """
        if current is None:
            return pd.Series(dtype=float)
        # qlib Position 接口
        if hasattr(current, "get_stock_weight_dict"):
            try:
                d = current.get_stock_weight_dict(only_stock=True)
                if d:
                    return pd.Series(d, dtype=float)
            except Exception:
                pass
        # 退化：dict 兼容
        if isinstance(current, dict):
            return pd.Series(current, dtype=float)
        return pd.Series(dtype=float)

    # ─────────────────────────────────────────────────────────────────────
    # 描述
    # ─────────────────────────────────────────────────────────────────────

    def describe(self) -> str:
        filter_desc = (
            self._stock_filter.describe()
            if self._stock_filter is not None
            else "off"
        )
        timer_desc = (
            self._market_timer.describe()
            if self._market_timer is not None
            else "off"
        )
        return (
            f"QuantMLWeightStrategy(topk={self.topk}, "
            f"score_q={self.score_quantile:.2f}, "
            f"rebal_freq={self.rebalance_freq}步, "
            f"vol_lookback={self.vol_lookback}日)\n"
            f"    Selector    : {self._selector.describe()}\n"
            f"    Weighter    : {self._weighter.describe()}\n"
            f"    RiskGuard   : {self._risk.describe()}\n"
            f"    NDrop       : {self.n_drop if self.n_drop is not None else 'off'}\n"
            f"    StockFilter : {filter_desc}\n"
            f"    TradeUnit   : {self.trade_unit}\n"
            f"    ExitRules   : enabled={self._exit_cfg.is_enabled()} "
            f"(sl={self._exit_cfg.stop_loss_pct}, "
            f"tp={self._exit_cfg.take_profit_pct}, "
            f"trail={self._exit_cfg.trailing_stop_pct})\n"
            f"    MarketTimer : {timer_desc}"
        )
