"""
分析层（Analysis Layer）
=========================
职责：
  - 从 PortfolioState 计算完整绩效指标（年化/回撤/夏普/IR/换手/胜率等）
  - 在终端打印结构化、易读的回测报告（无外部依赖）
  - 使用 Matplotlib 生成四张图表并保存到 artifacts/
  - 生成自包含 HTML 报告（嵌入图表路径）

输入: PortfolioState + SignalSummary（可选）+ 运行元数据
输出: PerformanceMetrics（dataclass）+ 文件输出

图表说明:
  cumulative_return.png  — 累计净值曲线（策略 vs 基准）
  drawdown.png           — 最大回撤曲线
  monthly_heatmap.png    — 月度超额收益热力图
  rolling_metrics.png    — 滚动夏普比率 & 滚动信息比率（63 日窗口）
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .portfolio_layer import PortfolioState
from .signal_layer import SignalSummary
from .trade_layer import TradeRecords


# 年化换算因子（交易期数 → 年）
_ANNUALIZE: dict = {"1day": 252, "1min": 252 * 240}


@dataclass
class PerformanceMetrics:
    """完整绩效指标集，所有百分比均以小数存储（0.1 = 10%）。"""
    # 收益
    annualized_return: float
    total_return: float
    # 风险
    annualized_volatility: float
    max_drawdown: float
    # 风险调整收益
    sharpe_ratio: float
    calmar_ratio: float
    information_ratio: float
    # 超额
    annualized_excess_return: float
    excess_volatility: float
    # 交易
    avg_turnover: float
    win_rate: float
    # 基准对比
    benchmark_annualized_return: float
    benchmark_max_drawdown: float


# ─────────────────────────────────────────────────────────────────────────────
# 指标计算
# ─────────────────────────────────────────────────────────────────────────────

class AnalysisLayer:
    """分析层：计算绩效指标，生成控制台报告、图表与 HTML。"""

    def compute_metrics(
        self,
        state: PortfolioState,
        freq_key: str = "1day",
    ) -> PerformanceMetrics:
        """计算完整绩效指标。"""
        ann_factor = _ANNUALIZE.get(freq_key, 252)
        n = state.n_periods

        returns = state.returns
        benchmark = state.benchmark
        excess = state.excess_return

        def _ann_ret(rets: pd.Series) -> float:
            total = float((1.0 + rets).prod() - 1.0)
            return float((1.0 + total) ** (ann_factor / n) - 1.0) if n > 0 else float("nan")

        def _max_dd(rets: pd.Series) -> float:
            cum = (1.0 + rets).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            return float(dd.min())

        ann_ret = _ann_ret(returns)
        total_ret = float((1.0 + returns).prod() - 1.0)
        ann_vol = float(returns.std() * math.sqrt(ann_factor))
        max_dd = _max_dd(returns)

        sharpe = (
            float(returns.mean() / returns.std() * math.sqrt(ann_factor))
            if returns.std() > 1e-9 else float("nan")
        )
        calmar = ann_ret / abs(max_dd) if abs(max_dd) > 1e-9 else float("nan")
        ir = (
            float(excess.mean() / excess.std() * math.sqrt(ann_factor))
            if excess.std() > 1e-9 else float("nan")
        )

        ann_excess = _ann_ret(excess)
        excess_vol = float(excess.std() * math.sqrt(ann_factor))
        avg_turnover = float(state.turnover.mean()) if not state.turnover.isna().all() else float("nan")
        win_rate = float((returns > 0).mean())

        bench_ann = _ann_ret(benchmark)
        bench_max_dd = _max_dd(benchmark)

        return PerformanceMetrics(
            annualized_return=ann_ret,
            total_return=total_ret,
            annualized_volatility=ann_vol,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe,
            calmar_ratio=calmar,
            information_ratio=ir,
            annualized_excess_return=ann_excess,
            excess_volatility=excess_vol,
            avg_turnover=avg_turnover,
            win_rate=win_rate,
            benchmark_annualized_return=bench_ann,
            benchmark_max_drawdown=bench_max_dd,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 控制台报告
    # ─────────────────────────────────────────────────────────────────────────

    def print_report(
        self,
        run_id: str,
        model_name: str,
        freq_name: str,
        metrics: PerformanceMetrics,
        signal_summary: Optional[SignalSummary] = None,
        state: Optional[PortfolioState] = None,
        indicator_df: Optional[pd.DataFrame] = None,
        output_dir: Optional[Path] = None,
        trade_records: Optional[TradeRecords] = None,
    ) -> None:
        """在终端输出格式化回测报告（纯 Python，无第三方依赖）。"""
        W = 72

        def _hr(ch: str = "═") -> str:
            return ch * W

        def _p(v: float, d: int = 2) -> str:
            if math.isnan(v):
                return "—"
            sign = "+" if v >= 0 else ""
            return f"{sign}{v * 100:.{d}f}%"

        def _f(v: float, d: int = 2) -> str:
            if math.isnan(v):
                return "—"
            return f"{v:.{d}f}"

        freq_display = "日频 (1day)" if state and "day" in state.freq_key else "分钟频 (1min)"
        start_dt = str(state.returns.index[0].date()) if state else "?"
        end_dt = str(state.returns.index[-1].date()) if state else "?"
        n_periods = state.n_periods if state else "?"

        # ── 标题区 ────────────────────────────────────────────────────────
        print("\n" + _hr())
        print(f"  {'回 测 结 果 分 析 报 告':^{W - 4}}")
        print(f"  {'─' * (W - 4)}")
        print(f"  run_id   : {run_id}")
        print(f"  模型     : {model_name}     频率: {freq_display}")
        print(f"  测试区间 : {start_dt} → {end_dt}  ({n_periods} 个交易期)")
        print(_hr())

        # ── 核心绩效指标 ──────────────────────────────────────────────────
        C = [26, 14, 14, 12]  # 列宽
        sep = "  " + "─" * (W - 2)

        print(f"\n  ┌─ 核心绩效指标 {'─' * (W - 17)}┐")
        _header = (
            f"  │ {'指标名称':<{C[0]}} {'策略':>{C[1]}} "
            f"{'基准(000905)':>{C[2]}} {'超额':>{C[3]}} │"
        )
        print(_header)
        print(f"  │{'─' * (W - 2)}│")

        rows = [
            ("年化收益率",
             _p(metrics.annualized_return),
             _p(metrics.benchmark_annualized_return),
             _p(metrics.annualized_excess_return)),
            ("总收益率",
             _p(metrics.total_return),
             "—", "—"),
            ("最大回撤",
             _p(metrics.max_drawdown),
             _p(metrics.benchmark_max_drawdown),
             "—"),
            ("年化波动率",
             _p(metrics.annualized_volatility),
             "—", "—"),
            ("夏普比率",
             _f(metrics.sharpe_ratio),
             "—", "—"),
            ("卡玛比率",
             _f(metrics.calmar_ratio),
             "—", "—"),
            ("信息比率 (IR)",
             _f(metrics.information_ratio),
             "—", "—"),
            ("超额年化收益",
             _p(metrics.annualized_excess_return),
             "—", "—"),
            ("超额年化波动",
             _p(metrics.excess_volatility),
             "—", "—"),
            ("日均换手率",
             _p(metrics.avg_turnover),
             "—", "—"),
            ("胜率",
             _p(metrics.win_rate),
             "—", "—"),
        ]
        for name, strat, bench, exc in rows:
            line = (
                f"  │ {name:<{C[0]}} {strat:>{C[1]}} "
                f"{bench:>{C[2]}} {exc:>{C[3]}} │"
            )
            print(line)
        print(f"  └{'─' * (W - 2)}┘")

        # ── 信号质量 ──────────────────────────────────────────────────────
        if signal_summary is not None:
            print(f"\n  ┌─ 信号质量 (IC 分析) {'─' * (W - 21)}┐")
            ic_rows = [
                ("IC 均值（越大越好，>0.03 为有效信号）",
                 _f(signal_summary.ic_mean, 4)),
                ("IC 标准差（越小越稳定）",
                 _f(signal_summary.ic_std, 4)),
                ("ICIR = IC均值 / IC标准差（>0.3 为较好）",
                 _f(signal_summary.icir, 3)),
                ("IC > 0 占比（>55% 表示预测方向稳定）",
                 _p(signal_summary.ic_positive_ratio)),
                ("NaN 比例",
                 _p(signal_summary.nan_ratio)),
                ("覆盖标的数",
                 str(signal_summary.n_instruments)),
                ("覆盖日期数",
                 str(signal_summary.n_dates)),
            ]
            for name, val in ic_rows:
                print(f"  │  {name:<{W - 12}} {val:>6} │")
            print(f"  └{'─' * (W - 2)}┘")

        # ── 月度超额收益 ──────────────────────────────────────────────────
        if state is not None:
            self._print_monthly_excess(state, W)

        # ── 交易明细摘要（聚合统计） ─────────────────────────────────────
        if trade_records is not None and not trade_records.trades.empty:
            self._print_trade_summary(trade_records, W)
            self._print_recent_trades(trade_records.trades, W, max_rows=8)
            self._print_top_pnl(trade_records.realized_pnl, W, top_n=5)
        elif state is not None:
            # 退化模式：trade_layer 为空时，沿用旧版日级摘要 / 调仓兜底
            trade_preview = self._build_trade_detail_preview(indicator_df, max_rows=5)
            print(f"\n  ┌─ 交易明细摘要 {'─' * (W - 13)}┐")
            if not trade_preview.empty:
                print(f"  │  交易记录(日级): {len(indicator_df):<{W - 22}} │")
                latest_date = str(trade_preview.iloc[0]["date"])
                avg_deal_amt = trade_preview["deal_amount"].mean()
                avg_trade_cnt = trade_preview["trade_count"].mean()
                print(f"  │  最近记录日    : {latest_date:<{W - 22}} │")
                print(f"  │  近5日平均成交额: {avg_deal_amt:>{12},.0f}{'':<{W - 35}} │")
                print(f"  │  近5日平均成交笔: {avg_trade_cnt:>{12}.1f}{'':<{W - 35}} │")
            else:
                latest_date = str(state.turnover.index[-1].date())
                avg_turnover = float(state.turnover.tail(5).mean())
                print(f"  │  交易记录(日级): 暂无明细数据，使用调仓兜底{'':<{W - 32}} │")
                print(f"  │  最近记录日    : {latest_date:<{W - 22}} │")
                print(f"  │  近5日平均换手率: {avg_turnover*100:>11.2f}%{'':<{W - 35}} │")
            print(f"  └{'─' * (W - 2)}┘")

        # ── 已生成文件 ────────────────────────────────────────────────────
        if output_dir is not None:
            charts_dir = output_dir / "charts"
            print(f"\n  ┌─ 已生成文件 {'─' * (W - 13)}┐")
            print(f"  │  图表目录  : {str(charts_dir):<{W - 17}} │")
            for fn, desc in [
                ("cumulative_return.png", "累计净值曲线（策略 vs 基准）"),
                ("drawdown.png",          "最大回撤曲线"),
                ("monthly_heatmap.png",   "月度超额收益热力图"),
                ("rolling_metrics.png",   "滚动夏普 / 信息比率（63日窗口）"),
            ]:
                # 使用 ASCII，避免 Windows GBK 控制台打印 ✓/○ 触发 UnicodeEncodeError
                exists = "*" if (charts_dir / fn).exists() else "."
                print(f"  │  {exists} {fn:<28} {desc:<{W - 36}} │")
            # report.html 此时已由 save_html_report 写入；
            # indicator/trades/realized_pnl 的 parquet 由上层 persist_artifacts
            # 在 print_report 之后才落盘。这里改用"逻辑预期标记"，
            # 避免误报"文件不存在"。
            indicator_ok = indicator_df is not None and not indicator_df.empty
            has_trades = trade_records is not None and not trade_records.trades.empty
            has_pnl = trade_records is not None and not trade_records.realized_pnl.empty
            file_marks = [
                ("report.html",          "HTML 完整报告",         (output_dir / "report.html").exists()),
                ("indicator.parquet",    "日级成交聚合数据",         indicator_ok),
                ("trades.parquet",       "每笔订单明细",           has_trades),
                ("realized_pnl.parquet", "FIFO 已实现盈亏明细",     has_pnl),
            ]
            for fn, desc, will_exist in file_marks:
                mark = "*" if will_exist else "."
                print(f"  │  {mark} {fn:<28} {desc:<{W - 36}} │")
            print(f"  └{'─' * (W - 2)}┘")

        print("\n" + _hr() + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    # 交易明细打印
    # ─────────────────────────────────────────────────────────────────────────

    def _print_trade_summary(self, records: TradeRecords, W: int = 72) -> None:
        """打印订单 + 已实现盈亏汇总统计。"""
        trades = records.trades
        pnl = records.realized_pnl

        n_trades = len(trades)
        n_buy = int((trades["direction"] == "BUY").sum())
        n_sell = int((trades["direction"] == "SELL").sum())
        n_stocks = int(trades["stock_id"].nunique())
        gross_buy = float(trades.loc[trades["direction"] == "BUY", "value"].sum())
        gross_sell = float(trades.loc[trades["direction"] == "SELL", "value"].sum())
        total_cost = float(trades["cost"].sum())

        n_close = len(pnl)
        win_rate = float((pnl["net_pnl"] > 0).mean()) if n_close else float("nan")
        avg_hold = float(pnl["holding_days"].mean()) if n_close else float("nan")
        net_pnl_sum = float(pnl["net_pnl"].sum()) if n_close else 0.0
        avg_ret = float(pnl["return_pct"].mean()) if n_close else float("nan")
        market_breakdown = trades["market"].value_counts().to_dict() if n_trades else {}

        def _row(name: str, value: str) -> str:
            return f"  │  {name:<{W - 22}} {value:>{14}} │"

        print(f"\n  ┌─ 交易明细汇总 {'─' * (W - 13)}┐")
        print(_row("订单总数", f"{n_trades:,}"))
        print(_row("买入笔数 / 卖出笔数", f"{n_buy:,} / {n_sell:,}"))
        print(_row("涉及标的数", f"{n_stocks:,}"))
        print(_row("买入总额（元）", f"{gross_buy:>14,.0f}"))
        print(_row("卖出总额（元）", f"{gross_sell:>14,.0f}"))
        print(_row("累计手续费（元）", f"{total_cost:>14,.0f}"))
        print(f"  │{'─' * (W - 2)}│")
        if n_close:
            print(_row("已平仓笔数", f"{n_close:,}"))
            print(_row("胜率（净盈利>0）", f"{win_rate * 100:>13.2f}%"))
            print(_row("平均持仓天数", f"{avg_hold:>14.2f}"))
            print(_row("累计已实现净盈亏（元）",
                      f"{'+' if net_pnl_sum >= 0 else ''}{net_pnl_sum:>13,.0f}"))
            print(_row("平均单笔收益率",
                      f"{'+' if avg_ret >= 0 else ''}{avg_ret * 100:>13.2f}%"))
        if market_breakdown:
            top = sorted(market_breakdown.items(), key=lambda kv: -kv[1])[:4]
            mkt_str = ", ".join(f"{k}:{v}" for k, v in top)
            print(_row("板块分布(订单数)", mkt_str[:14] if len(mkt_str) > 14 else mkt_str))
        print(f"  └{'─' * (W - 2)}┘")

    def _print_recent_trades(
        self, trades: pd.DataFrame, W: int = 72, max_rows: int = 8
    ) -> None:
        """打印最近 N 笔订单明细。"""
        if trades.empty:
            return
        recent = trades.sort_values("datetime", ascending=False).head(max_rows)

        col_w = {
            "date": 11, "stock": 10, "mkt": 7, "dir": 4,
            "amt": 9, "price": 9, "value": 11, "cost": 7,
        }
        header = (
            f"  │  {'日期':<{col_w['date']}} {'股票代码':<{col_w['stock']}} "
            f"{'板块':<{col_w['mkt']}} {'方向':<{col_w['dir']}} "
            f"{'数量':>{col_w['amt']}} {'成交价':>{col_w['price']}} "
            f"{'成交额':>{col_w['value']}} {'手续费':>{col_w['cost']}}"
        )
        print(f"\n  ┌─ 最近 {len(recent)} 笔订单明细 {'─' * (W - 23)}┐")
        # 中文等宽控制台宽度难以精准对齐，这里直接打印
        print(f"{header:<{W - 1}} │")
        print(f"  │{'─' * (W - 2)}│")
        for _, row in recent.iterrows():
            line = (
                f"  │  {str(row['datetime'].date()):<{col_w['date']}} "
                f"{row['stock_id']:<{col_w['stock']}} "
                f"{row['market']:<{col_w['mkt']}} "
                f"{row['direction']:<{col_w['dir']}} "
                f"{row['amount']:>{col_w['amt']},.0f} "
                f"{row['price']:>{col_w['price']},.2f} "
                f"{row['value']:>{col_w['value']},.0f} "
                f"{row['cost']:>{col_w['cost']},.1f}"
            )
            print(f"{line:<{W - 1}} │")
        print(f"  └{'─' * (W - 2)}┘")

    def _print_top_pnl(self, pnl: pd.DataFrame, W: int = 72, top_n: int = 5) -> None:
        """打印盈利与亏损最大的若干笔已实现交易。"""
        if pnl.empty:
            return
        sorted_pnl = pnl.sort_values("net_pnl", ascending=False)
        top = sorted_pnl.head(top_n)
        bottom = sorted_pnl.tail(top_n).iloc[::-1]

        def _emit(title: str, sub: pd.DataFrame) -> None:
            print(f"\n  ┌─ {title} {'─' * (W - len(title) - 5)}┐")
            print(
                f"  │  {'平仓日':<11} {'股票':<10} {'持仓天':>5} "
                f"{'买价':>8} {'卖价':>8} {'净盈亏':>11} {'收益率':>8}{'':<{W - 67}} │"
            )
            print(f"  │{'─' * (W - 2)}│")
            for _, row in sub.iterrows():
                line = (
                    f"  │  {str(row['close_date'].date()):<11} "
                    f"{row['stock_id']:<10} "
                    f"{int(row['holding_days']):>5} "
                    f"{row['buy_price']:>8,.2f} "
                    f"{row['sell_price']:>8,.2f} "
                    f"{('+' if row['net_pnl'] >= 0 else '')}{row['net_pnl']:>10,.0f} "
                    f"{('+' if row['return_pct'] >= 0 else '')}{row['return_pct']*100:>7.2f}%"
                )
                print(f"{line:<{W - 1}} │")
            print(f"  └{'─' * (W - 2)}┘")

        _emit(f"盈利前 {len(top)} 笔交易", top)
        _emit(f"亏损前 {len(bottom)} 笔交易", bottom)

    def _print_monthly_excess(self, state: PortfolioState, W: int = 72) -> None:
        """打印月度超额收益表格。"""
        try:
            excess = state.excess_return
            freq_str = "ME" if pd.__version__ >= "2.2" else "M"
            try:
                monthly = excess.resample(freq_str).apply(
                    lambda x: float((1.0 + x).prod() - 1.0)
                )
            except ValueError:
                monthly = excess.resample("M").apply(
                    lambda x: float((1.0 + x).prod() - 1.0)
                )

            monthly.index = pd.DatetimeIndex(monthly.index)
            pv = monthly.to_frame("excess")
            pv["year"] = pv.index.year
            pv["month"] = pv.index.month
            table = pv.pivot(index="year", columns="month", values="excess")

            month_labels = ["1月", "2月", "3月", "4月", "5月", "6月",
                            "7月", "8月", "9月", "10月", "11月", "12月"]
            col_w = 7

            print(f"\n  ┌─ 月度超额收益分布 {'─' * (W - 19)}┐")
            header = f"  │  {'年份':<5}" + "".join(f"{m:>{col_w}}" for m in month_labels[:12])
            # 填充到宽度
            print(f"{header:<{W - 1}} │")
            print(f"  │{'─' * (W - 2)}│")

            for yr in sorted(table.index):
                row = f"  │  {yr:<5}"
                for m in range(1, 13):
                    if m in table.columns and not pd.isna(table.loc[yr, m]):
                        val = table.loc[yr, m]
                        sign = "+" if val >= 0 else ""
                        row += f"{sign}{val*100:.1f}%".rjust(col_w)
                    else:
                        row += "—".rjust(col_w)
                print(f"{row:<{W - 1}} │")

            print(f"  └{'─' * (W - 2)}┘")
        except Exception:
            pass  # 月度统计失败不影响主报告

    # ─────────────────────────────────────────────────────────────────────────
    # 图表生成
    # ─────────────────────────────────────────────────────────────────────────

    def save_charts(
        self,
        state: PortfolioState,
        metrics: PerformanceMetrics,
        output_dir: Path,
        run_id: str,
        signal_summary: Optional[SignalSummary] = None,
    ) -> bool:
        """
        生成并保存四张图表到 output_dir/charts/。

        Returns:
            True 表示成功，False 表示 matplotlib 不可用
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.colors as mcolors
        except ImportError:
            return False

        charts_dir = output_dir / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        plt.rcParams.update({
            "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "figure.dpi": 120,
        })

        try:
            self._chart_cumulative_return(state, metrics, charts_dir, plt)
        except Exception:
            pass
        try:
            self._chart_drawdown(state, metrics, charts_dir, plt)
        except Exception:
            pass
        try:
            self._chart_monthly_heatmap(state, charts_dir, plt, mcolors)
        except Exception:
            pass
        try:
            self._chart_rolling_metrics(state, charts_dir, plt, signal_summary)
        except Exception:
            pass

        return True

    def _chart_cumulative_return(self, state, metrics, charts_dir, plt):
        """图 1: 累计净值曲线（策略 vs 基准）。"""
        fig, ax = plt.subplots(figsize=(13, 5))

        ax.plot(
            state.cumulative_return.index,
            state.cumulative_return.values,
            label="策略",
            color="#D94040",
            linewidth=1.8,
            zorder=3,
        )
        ax.plot(
            state.cumulative_benchmark.index,
            state.cumulative_benchmark.values,
            label="基准 (000905)",
            color="#4060D9",
            linewidth=1.2,
            linestyle="--",
            alpha=0.75,
            zorder=2,
        )
        # 超额净值
        excess_cum = state.cumulative_return / state.cumulative_benchmark
        ax.plot(
            excess_cum.index,
            excess_cum.values,
            label="相对净值（策略/基准）",
            color="#20A060",
            linewidth=1.0,
            linestyle=":",
            alpha=0.85,
            zorder=2,
        )
        ax.axhline(1.0, color="gray", linewidth=0.6, linestyle=":")

        ann_str = _fmt_pct(metrics.annualized_return)
        sharpe_str = _fmt_f(metrics.sharpe_ratio)
        ir_str = _fmt_f(metrics.information_ratio)
        ax.set_title(
            f"累计净值曲线  |  年化收益 {ann_str}  夏普 {sharpe_str}  IR {ir_str}",
            fontsize=11,
        )
        ax.set_ylabel("净值（期初=1）")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(charts_dir / "cumulative_return.png", bbox_inches="tight")
        plt.close(fig)

    def _chart_drawdown(self, state, metrics, charts_dir, plt):
        """图 2: 回撤曲线（策略 vs 基准）。"""
        cum = state.cumulative_return
        dd = (cum - cum.cummax()) / cum.cummax()

        bench_cum = state.cumulative_benchmark
        bench_dd = (bench_cum - bench_cum.cummax()) / bench_cum.cummax()

        fig, ax = plt.subplots(figsize=(13, 4))
        ax.fill_between(dd.index, dd.values, 0.0,
                        alpha=0.35, color="#D94040", label="策略回撤")
        ax.plot(dd.index, dd.values, color="#D94040", linewidth=0.8, alpha=0.7)
        ax.plot(bench_dd.index, bench_dd.values,
                color="#4060D9", linewidth=1.0, linestyle="--", alpha=0.55,
                label="基准回撤")

        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}%")
        )
        max_dd_str = _fmt_pct(metrics.max_drawdown)
        ax.set_title(f"回撤曲线  |  策略最大回撤 {max_dd_str}", fontsize=11)
        ax.set_ylabel("回撤幅度")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(charts_dir / "drawdown.png", bbox_inches="tight")
        plt.close(fig)

    def _chart_monthly_heatmap(self, state, charts_dir, plt, mcolors):
        """图 3: 月度超额收益热力图。"""
        excess = state.excess_return
        freq_str = "ME" if pd.__version__ >= "2.2" else "M"
        try:
            monthly = excess.resample(freq_str).apply(
                lambda x: float((1.0 + x).prod() - 1.0)
            )
        except ValueError:
            monthly = excess.resample("M").apply(
                lambda x: float((1.0 + x).prod() - 1.0)
            )

        monthly.index = pd.DatetimeIndex(monthly.index)
        pv = monthly.to_frame("excess")
        pv["year"] = pv.index.year
        pv["month"] = pv.index.month
        table = pv.pivot(index="year", columns="month", values="excess")

        n_years = len(table)
        fig_h = max(2.5, n_years * 0.75 + 1.2)
        fig, ax = plt.subplots(figsize=(14, fig_h))

        all_vals = table.values[~np.isnan(table.values)]
        vmax = max(0.04, float(np.percentile(np.abs(all_vals), 95))) if len(all_vals) else 0.04
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        im = ax.imshow(table.values, cmap="RdYlGn", norm=norm, aspect="auto")

        month_labels = [str(m) for m in range(1, 13)]
        ax.set_xticks(range(len(table.columns)))
        ax.set_xticklabels([month_labels[c - 1] for c in table.columns], fontsize=9)
        ax.set_yticks(range(n_years))
        ax.set_yticklabels(table.index.astype(str), fontsize=9)

        for i in range(n_years):
            for j, col in enumerate(table.columns):
                val = table.iloc[i, j]
                if not math.isnan(val):
                    text = f"{val * 100:+.1f}%"
                    text_color = "white" if abs(val) > vmax * 0.65 else "black"
                    ax.text(j, i, text, ha="center", va="center",
                            fontsize=7.5, color=text_color)

        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x * 100:.0f}%")
        )
        ax.set_title("月度超额收益热力图（策略 - 基准）", fontsize=11)
        ax.set_xlabel("月份")
        ax.set_ylabel("年份")
        fig.tight_layout()
        fig.savefig(charts_dir / "monthly_heatmap.png", bbox_inches="tight")
        plt.close(fig)

    def _chart_rolling_metrics(self, state, charts_dir, plt,
                                signal_summary: Optional[SignalSummary] = None):
        """图 4: 滚动夏普 & 滚动 IR（63 日窗口）。"""
        returns = state.returns
        excess = state.excess_return
        window = 63

        ann = _ANNUALIZE.get(state.freq_key, 252)

        rolling_sharpe = returns.rolling(window).apply(
            lambda x: x.mean() / x.std() * math.sqrt(ann) if x.std() > 1e-9 else float("nan"),
            raw=True,
        )
        rolling_ir = excess.rolling(window).apply(
            lambda x: x.mean() / x.std() * math.sqrt(ann) if x.std() > 1e-9 else float("nan"),
            raw=True,
        )

        n_rows = 3 if (signal_summary is not None and signal_summary.ic_series is not None
                       and not signal_summary.ic_series.empty) else 2
        fig, axes = plt.subplots(n_rows, 1, figsize=(13, n_rows * 2.8), sharex=True)

        ax1 = axes[0]
        ax1.plot(rolling_sharpe.index, rolling_sharpe.values,
                 color="#D94040", linewidth=1.0)
        ax1.axhline(0, color="gray", linewidth=0.5)
        ax1.axhline(1, color="#D94040", linewidth=0.5, linestyle="--", alpha=0.4)
        ax1.set_ylabel(f"滚动夏普（{window}日）", fontsize=9)
        ax1.grid(axis="y", alpha=0.2)

        ax2 = axes[1]
        ax2.plot(rolling_ir.index, rolling_ir.values,
                 color="#4060D9", linewidth=1.0)
        ax2.axhline(0, color="gray", linewidth=0.5)
        ax2.set_ylabel(f"滚动 IR（{window}日）", fontsize=9)
        ax2.grid(axis="y", alpha=0.2)

        if n_rows == 3:
            ax3 = axes[2]
            ic_s = signal_summary.ic_series
            rolling_ic = ic_s.rolling(window).mean()
            ax3.bar(ic_s.index, ic_s.values, color="#20A060", alpha=0.35, width=1)
            ax3.plot(rolling_ic.index, rolling_ic.values,
                     color="#20A060", linewidth=1.2, label=f"滚动IC均值（{window}日）")
            ax3.axhline(0, color="gray", linewidth=0.5)
            ax3.set_ylabel("IC", fontsize=9)
            ax3.legend(fontsize=8)
            ax3.grid(axis="y", alpha=0.2)

        fig.suptitle("滚动绩效指标", fontsize=11)
        fig.tight_layout()
        fig.savefig(charts_dir / "rolling_metrics.png", bbox_inches="tight")
        plt.close(fig)

    # ─────────────────────────────────────────────────────────────────────────
    # HTML 报告
    # ─────────────────────────────────────────────────────────────────────────

    def save_html_report(
        self,
        output_dir: Path,
        run_id: str,
        model_name: str,
        freq_name: str,
        metrics: PerformanceMetrics,
        signal_summary: Optional[SignalSummary] = None,
        state: Optional[PortfolioState] = None,
        indicator_df: Optional[pd.DataFrame] = None,
        trade_records: Optional[TradeRecords] = None,
        rolling_summary: Optional[Dict[str, Any]] = None,
    ) -> None:
        """生成自包含 HTML 回测报告。

        rolling_summary 若非 None，会在报告末尾插入"fold 间稳定性"一节，
        展示各 fold IC/年化/换手的均值±标准差。
        """
        charts_dir = output_dir / "charts"

        def _hp(v: float, d: int = 2) -> str:
            """格式化为带颜色 HTML span 的百分比。"""
            if math.isnan(v):
                return "<span>—</span>"
            sign = "+" if v >= 0 else ""
            cls = "pos" if v >= 0 else "neg"
            return f'<span class="{cls}">{sign}{v * 100:.{d}f}%</span>'

        def _hf(v: float, d: int = 2) -> str:
            """格式化为 HTML 数值。"""
            if math.isnan(v):
                return "—"
            return f"{v:.{d}f}"

        start_dt = str(state.returns.index[0].date()) if state else "?"
        end_dt = str(state.returns.index[-1].date()) if state else "?"
        n_periods = state.n_periods if state else "?"
        freq_display = "日频 (1day)" if state and "day" in state.freq_key else "分钟频 (1min)"

        # 图表卡片 HTML
        charts_html = ""
        for fn, title in [
            ("cumulative_return.png", "累计净值曲线"),
            ("drawdown.png", "回撤曲线"),
            ("monthly_heatmap.png", "月度超额收益热力图"),
            ("rolling_metrics.png", "滚动绩效指标"),
        ]:
            if (charts_dir / fn).exists():
                charts_html += f"""
          <div class="chart-card">
            <h3>{title}</h3>
            <img src="charts/{fn}" alt="{title}">
          </div>"""

        # 信号质量区块
        ic_html = ""
        if signal_summary is not None:
            def _grade_ic(v):
                if math.isnan(v):
                    return ""
                if v >= 0.06:
                    return '<span class="badge good">优</span>'
                if v >= 0.03:
                    return '<span class="badge ok">良</span>'
                return '<span class="badge bad">弱</span>'

            ic_html = f"""
        <div class="section">
          <h2>信号质量分析（IC / ICIR）</h2>
          <table>
            <tr><th>指标</th><th>数值</th><th>评级参考</th></tr>
            <tr>
              <td>IC 均值（Spearman，截面秩相关）</td>
              <td>{_hf(signal_summary.ic_mean, 4)}</td>
              <td>{_grade_ic(signal_summary.ic_mean)} &gt;0.03 有效 / &gt;0.06 优秀</td>
            </tr>
            <tr>
              <td>IC 标准差（越低越稳定）</td>
              <td>{_hf(signal_summary.ic_std, 4)}</td>
              <td>—</td>
            </tr>
            <tr>
              <td>ICIR = IC均值 / IC标准差</td>
              <td>{_hf(signal_summary.icir, 3)}</td>
              <td>&gt;0.3 较好 / &gt;0.5 优秀</td>
            </tr>
            <tr>
              <td>IC &gt; 0 占比（方向胜率）</td>
              <td>{_hp(signal_summary.ic_positive_ratio)}</td>
              <td>&gt;55% 方向稳定</td>
            </tr>
            <tr>
              <td>NaN 比例（越低越好）</td>
              <td>{_hp(signal_summary.nan_ratio)}</td>
              <td>&lt;5% 为佳</td>
            </tr>
            <tr><td>覆盖标的数</td><td>{signal_summary.n_instruments}</td><td>—</td></tr>
            <tr><td>覆盖交易日数</td><td>{signal_summary.n_dates}</td><td>—</td></tr>
          </table>
        </div>"""

        # 交易明细区块（来源: trade_records，订单级别 + 已实现盈亏）
        trade_html = self._build_trade_html(trade_records)
        if not trade_html and state is not None:
            # 退化：trade_records 为空时，沿用调仓明细兜底
            rebalance_preview = self._build_rebalance_preview(state, max_rows=30)
            rebalance_rows = ""
            for _, row in rebalance_preview.iterrows():
                rebalance_rows += (
                    "<tr>"
                    f"<td>{row['date']}</td>"
                    f"<td>{row['turnover']:.2%}</td>"
                    f"<td>{row['strategy_ret']:+.2%}</td>"
                    f"<td>{row['bench_ret']:+.2%}</td>"
                    f"<td>{row['excess_ret']:+.2%}</td>"
                    "</tr>"
                )
            trade_html = f"""
    <div class="section">
      <h2>交易明细（日级调仓摘要）</h2>
      <p style="margin:0 0 12px;color:#666;font-size:12px;">
        当前运行未返回订单明细，已降级展示最近 {len(rebalance_preview)} 日调仓与收益数据。
      </p>
      <table>
        <tr>
          <th>日期</th>
          <th>换手率</th>
          <th>策略收益</th>
          <th>基准收益</th>
          <th>超额收益</th>
        </tr>
        {rebalance_rows}
      </table>
    </div>"""

        # ── fold 间稳定性区块（walk-forward 模式） ────────────────────────────
        rolling_stability_html = ""
        if rolling_summary and rolling_summary.get("fold_count", 0) > 1:
            fold_stats = rolling_summary.get("fold_stats", {})
            fold_records = rolling_summary.get("folds", [])
            oos_start = rolling_summary.get("oos_range", {}).get("start", "?")
            oos_end = rolling_summary.get("oos_range", {}).get("end", "?")
            fold_count = rolling_summary.get("fold_count", 0)

            # per-fold IC 行
            fold_ic_rows = ""
            for fr in fold_records:
                fm = fr.get("metrics", {})
                fold_ic_rows += (
                    f"<tr>"
                    f"<td>fold {fr['fold_id']}</td>"
                    f"<td>{fr['test']['start']} ~ {fr['test']['end']}</td>"
                    f"<td>{fm.get('ic_mean', float('nan')):.4f}" if fm.get('ic_mean') is not None else "<td>—"
                    f"</td>"
                    f"<td>{fm.get('icir', float('nan')):.4f}" if fm.get('icir') is not None else "<td>—"
                    f"</td>"
                    f"</tr>"
                )

            # 汇总统计行
            def _stat_cell(key: str) -> str:
                s = fold_stats.get(key, {})
                if not s:
                    return "<td>—</td><td>—</td>"
                return (
                    f"<td>{s['mean']:.4f} ± {s['std']:.4f}</td>"
                    f"<td>[{s['min']:.4f}, {s['max']:.4f}]</td>"
                )

            rolling_stability_html = f"""
    <div class="section">
      <h2>Walk-Forward fold 间稳定性</h2>
      <p style="margin:0 0 12px;color:#666;font-size:12px;">
        共 {fold_count} 个 fold，OOS 总覆盖：{oos_start} ~ {oos_end}
      </p>
      <table>
        <tr>
          <th>指标</th><th>均值 ± 标准差</th><th>最小 ~ 最大</th>
        </tr>
        <tr><td>IC（日均）</td>{_stat_cell("ic_mean")}</tr>
        <tr><td>ICIR</td>{_stat_cell("icir")}</tr>
        <tr><td>年化收益</td>{_stat_cell("annualized_return")}</tr>
        <tr><td>最大回撤</td>{_stat_cell("max_drawdown")}</tr>
        <tr><td>Sharpe</td>{_stat_cell("sharpe_ratio")}</tr>
        <tr><td>平均换手</td>{_stat_cell("avg_turnover")}</tr>
      </table>
      <br>
      <h3>各 fold OOS 信号指标</h3>
      <table>
        <tr><th>fold</th><th>OOS 测试区间</th><th>IC 均值</th><th>ICIR</th></tr>
        {fold_ic_rows}
      </table>
    </div>"""

        html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>回测报告 — {run_id}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, "PingFang SC", "Microsoft YaHei",
                   "Helvetica Neue", Arial, sans-serif;
      margin: 0; background: #f0f2f5; color: #1a1a2e; font-size: 14px;
    }}
    .container {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}

    /* 标题 */
    .header {{
      background: linear-gradient(135deg, #1a1f3c 0%, #2d3561 100%);
      color: white; padding: 28px 36px; border-radius: 12px;
      margin-bottom: 24px;
    }}
    .header h1 {{ margin: 0 0 10px; font-size: 22px; letter-spacing: 1px; }}
    .header .meta {{ font-size: 13px; opacity: 0.75; line-height: 1.8; }}
    .header .meta span {{ margin-right: 20px; }}

    /* 卡片 */
    .section {{
      background: white; border-radius: 10px; padding: 24px 28px;
      margin-bottom: 20px;
      box-shadow: 0 2px 8px rgba(0,0,0,.06);
    }}
    h2 {{
      margin: 0 0 18px; font-size: 15px; color: #1a1f3c;
      border-left: 4px solid #D94040; padding-left: 10px;
    }}
    h3 {{ margin: 0 0 10px; font-size: 13px; color: #666; }}

    /* 表格 */
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th {{
      background: #f5f7fa; padding: 9px 14px; text-align: left;
      font-weight: 600; color: #444; border-bottom: 2px solid #e8eaf0;
    }}
    td {{ padding: 9px 14px; border-bottom: 1px solid #f0f2f5; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #fafbff; }}

    /* 数值颜色 */
    .pos {{ color: #D94040; font-weight: 600; }}
    .neg {{ color: #2E9E4F; font-weight: 600; }}

    /* 买卖方向标签 */
    .buy-tag, .sell-tag {{
      display: inline-block; padding: 1px 7px; border-radius: 4px;
      font-size: 12px; font-weight: 600;
    }}
    .buy-tag  {{ background: #fff0ed; color: #D94040; }}
    .sell-tag {{ background: #ecf7ee; color: #2E9E4F; }}

    /* 代码字体 */
    code {{
      font-family: "SF Mono", Consolas, Monaco, monospace;
      background: #f5f5f7; padding: 1px 5px; border-radius: 3px; font-size: 12px;
    }}

    /* 图表网格 */
    .charts-grid {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
    }}
    .chart-card {{
      background: white; border-radius: 10px; padding: 16px 20px;
      box-shadow: 0 2px 8px rgba(0,0,0,.06);
    }}
    .chart-card img {{ width: 100%; height: auto; border-radius: 4px; }}

    /* 评级徽章 */
    .badge {{
      display: inline-block; padding: 1px 7px; border-radius: 10px;
      font-size: 11px; font-weight: 600; margin-right: 4px;
    }}
    .badge.good {{ background: #e8f7ee; color: #2E9E4F; }}
    .badge.ok   {{ background: #fff3e0; color: #E67E22; }}
    .badge.bad  {{ background: #fdecea; color: #D94040; }}

    @media (max-width: 768px) {{
      .charts-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="container">

    <div class="header">
      <h1>&#x1F4CA; 回测结果分析报告</h1>
      <div class="meta">
        <span>run_id: <strong>{run_id}</strong></span>
        <span>模型: <strong>{model_name}</strong></span>
        <span>频率: <strong>{freq_display}</strong></span>
        <span>测试区间: <strong>{start_dt} → {end_dt}</strong>（{n_periods} 个交易期）</span>
      </div>
    </div>

    <div class="section">
      <h2>核心绩效指标</h2>
      <table>
        <tr>
          <th>指标名称</th>
          <th>策略</th>
          <th>基准 (000905)</th>
          <th>超额</th>
          <th>说明</th>
        </tr>
        <tr>
          <td>年化收益率</td>
          <td>{_hp(metrics.annualized_return)}</td>
          <td>{_hp(metrics.benchmark_annualized_return)}</td>
          <td>{_hp(metrics.annualized_excess_return)}</td>
          <td>复利年化，去除手续费后</td>
        </tr>
        <tr>
          <td>总收益率</td>
          <td>{_hp(metrics.total_return)}</td>
          <td>—</td><td>—</td>
          <td>测试期累计净值增长</td>
        </tr>
        <tr>
          <td>最大回撤</td>
          <td>{_hp(metrics.max_drawdown)}</td>
          <td>{_hp(metrics.benchmark_max_drawdown)}</td>
          <td>—</td>
          <td>峰值到谷值最大跌幅（越小越好）</td>
        </tr>
        <tr>
          <td>年化波动率</td>
          <td>{_hp(metrics.annualized_volatility)}</td>
          <td>—</td><td>—</td>
          <td>日收益率标准差 × √年化因子</td>
        </tr>
        <tr>
          <td>夏普比率</td>
          <td>{_hf(metrics.sharpe_ratio)}</td>
          <td>—</td><td>—</td>
          <td>超额收益 / 波动率（&gt;1 良好，&gt;2 优秀）</td>
        </tr>
        <tr>
          <td>卡玛比率</td>
          <td>{_hf(metrics.calmar_ratio)}</td>
          <td>—</td><td>—</td>
          <td>年化收益 / 最大回撤（&gt;1 良好）</td>
        </tr>
        <tr>
          <td>信息比率 (IR)</td>
          <td>{_hf(metrics.information_ratio)}</td>
          <td>—</td><td>—</td>
          <td>超额收益均值 / 超额收益波动（&gt;0.5 良好）</td>
        </tr>
        <tr>
          <td>超额年化收益</td>
          <td>{_hp(metrics.annualized_excess_return)}</td>
          <td>—</td><td>—</td>
          <td>相对于基准的年化超额</td>
        </tr>
        <tr>
          <td>日均换手率</td>
          <td>{_hp(metrics.avg_turnover)}</td>
          <td>—</td><td>—</td>
          <td>每日换仓比例，影响交易成本</td>
        </tr>
        <tr>
          <td>胜率</td>
          <td>{_hp(metrics.win_rate)}</td>
          <td>—</td><td>—</td>
          <td>收益为正的交易日占比</td>
        </tr>
      </table>
    </div>

    {ic_html}
    {trade_html}
    {rolling_stability_html}

    <div class="section">
      <h2>可视化图表</h2>
      <div class="charts-grid">
        {charts_html if charts_html else "<p>图表生成中或 matplotlib 不可用</p>"}
      </div>
    </div>

  </div>
</body>
</html>"""

        html_path = output_dir / "report.html"
        html_path.write_text(html, encoding="utf-8")

    @staticmethod
    def _build_trade_detail_preview(
        indicator_df: Optional[pd.DataFrame],
        max_rows: int = 30,
    ) -> pd.DataFrame:
        """将 indicator_df 标准化为交易明细预览表（按日期倒序）。"""
        if indicator_df is None or indicator_df.empty:
            return pd.DataFrame()

        df = indicator_df.copy()

        if isinstance(df.index, pd.MultiIndex):
            date_level = df.index.get_level_values(0)
            df["date"] = pd.to_datetime(date_level, errors="coerce")
        elif isinstance(df.index, pd.DatetimeIndex):
            df["date"] = pd.to_datetime(df.index, errors="coerce")
        else:
            df["date"] = pd.to_datetime(df.get("datetime"), errors="coerce")

        def _safe_col(name: str, default: float = 0.0) -> pd.Series:
            if name in df.columns:
                return pd.to_numeric(df[name], errors="coerce").fillna(default)
            return pd.Series(default, index=df.index, dtype=float)

        preview = pd.DataFrame(
            {
                "date": df["date"].dt.date.astype(str),
                "fill_ratio": _safe_col("ffr"),
                "position_ratio": _safe_col("pos"),
                "deal_amount": _safe_col("deal_amount"),
                "position_value": _safe_col("value"),
                "trade_count": _safe_col("count"),
            }
        ).dropna(subset=["date"])

        if preview.empty:
            return preview

        preview = preview.sort_values("date", ascending=False).head(max_rows)
        return preview.reset_index(drop=True)

    @staticmethod
    def _build_rebalance_preview(
        state: PortfolioState,
        max_rows: int = 30,
    ) -> pd.DataFrame:
        """基于 report_df 生成调仓明细预览（indicator 缺失时兜底）。"""
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(state.returns.index, errors="coerce").date.astype(str),
                "turnover": pd.to_numeric(state.turnover, errors="coerce").fillna(0.0).values,
                "strategy_ret": pd.to_numeric(state.returns, errors="coerce").fillna(0.0).values,
                "bench_ret": pd.to_numeric(state.benchmark, errors="coerce").fillna(0.0).values,
                "excess_ret": pd.to_numeric(state.excess_return, errors="coerce").fillna(0.0).values,
            }
        )
        return df.sort_values("date", ascending=False).head(max_rows).reset_index(drop=True)

    @staticmethod
    def _build_trade_html(trade_records: Optional[TradeRecords]) -> str:
        """渲染订单明细 + 已实现盈亏 + Top/Bottom 盈亏 三个 HTML 表。"""
        if trade_records is None or trade_records.trades.empty:
            return ""

        trades = trade_records.trades.sort_values("datetime", ascending=False)
        pnl = trade_records.realized_pnl
        n_show = 100  # 表格最多展示 100 行

        # 1) 订单明细（最近 100 笔）
        trade_rows = ""
        for _, row in trades.head(n_show).iterrows():
            dir_cls = "buy-tag" if row["direction"] == "BUY" else "sell-tag"
            dir_label = "买入" if row["direction"] == "BUY" else "卖出"
            trade_rows += (
                "<tr>"
                f"<td>{str(row['datetime'].date())}</td>"
                f"<td><code>{row['stock_id']}</code></td>"
                f"<td>{row['market']}</td>"
                f"<td><span class='{dir_cls}'>{dir_label}</span></td>"
                f"<td>{row['amount']:,.0f}</td>"
                f"<td>{row['price']:,.2f}</td>"
                f"<td>{row['value']:,.0f}</td>"
                f"<td>{row['cost']:,.2f}</td>"
                f"<td>{row['fill_ratio']:.2f}</td>"
                "</tr>"
            )
        trades_table = f"""
    <div class="section">
      <h2>订单明细（最近 {min(len(trades), n_show)} 笔，共 {len(trades)} 笔）</h2>
      <p style="margin:0 0 12px;color:#666;font-size:12px;">
        每笔订单含成交价、数量、金额、手续费、所属板块。完整明细见 trades.parquet。
      </p>
      <table>
        <tr>
          <th>日期</th>
          <th>股票代码</th>
          <th>板块</th>
          <th>方向</th>
          <th>数量（股）</th>
          <th>成交价（元）</th>
          <th>成交额（元）</th>
          <th>手续费（元）</th>
          <th>完成率</th>
        </tr>
        {trade_rows}
      </table>
    </div>"""

        if pnl.empty:
            return trades_table

        # 2) FIFO 已实现盈亏（最近 100 笔）
        pnl_recent = pnl.sort_values("close_date", ascending=False).head(n_show)
        pnl_rows = ""
        for _, row in pnl_recent.iterrows():
            pnl_cls = "pos" if row["net_pnl"] >= 0 else "neg"
            pnl_rows += (
                "<tr>"
                f"<td>{str(row['close_date'].date())}</td>"
                f"<td>{str(row['open_date'].date())}</td>"
                f"<td><code>{row['stock_id']}</code></td>"
                f"<td>{row['market']}</td>"
                f"<td>{int(row['holding_days'])}</td>"
                f"<td>{row['amount']:,.0f}</td>"
                f"<td>{row['buy_price']:,.2f}</td>"
                f"<td>{row['sell_price']:,.2f}</td>"
                f"<td>{row['trade_cost']:,.2f}</td>"
                f"<td class='{pnl_cls}'>{('+' if row['net_pnl'] >= 0 else '')}{row['net_pnl']:,.0f}</td>"
                f"<td class='{pnl_cls}'>{('+' if row['return_pct'] >= 0 else '')}{row['return_pct']*100:,.2f}%</td>"
                "</tr>"
            )
        # 汇总
        total_pnl = float(pnl["net_pnl"].sum())
        n_close = len(pnl)
        win_rate = float((pnl["net_pnl"] > 0).mean())
        avg_hold = float(pnl["holding_days"].mean())
        total_pnl_cls = "pos" if total_pnl >= 0 else "neg"
        pnl_table = f"""
    <div class="section">
      <h2>已实现盈亏明细（FIFO 配对）</h2>
      <p style="margin:0 0 12px;color:#666;font-size:12px;">
        共 {n_close} 笔已平仓交易，胜率 {win_rate*100:.2f}%，平均持仓 {avg_hold:.1f} 天，
        累计净盈亏 <span class="{total_pnl_cls}">{('+' if total_pnl >= 0 else '')}{total_pnl:,.0f}</span> 元。
        完整数据见 realized_pnl.parquet。
      </p>
      <table>
        <tr>
          <th>平仓日</th>
          <th>开仓日</th>
          <th>股票代码</th>
          <th>板块</th>
          <th>持仓天数</th>
          <th>数量（股）</th>
          <th>买入价</th>
          <th>卖出价</th>
          <th>手续费</th>
          <th>净盈亏（元）</th>
          <th>收益率</th>
        </tr>
        {pnl_rows}
      </table>
    </div>"""

        # 3) Top/Bottom 盈亏排行
        top = pnl.sort_values("net_pnl", ascending=False).head(10)
        bottom = pnl.sort_values("net_pnl", ascending=True).head(10)

        def _ranking(title: str, sub: pd.DataFrame, color_cls: str) -> str:
            if sub.empty:
                return ""
            rows_html = ""
            for _, row in sub.iterrows():
                rows_html += (
                    "<tr>"
                    f"<td>{str(row['close_date'].date())}</td>"
                    f"<td><code>{row['stock_id']}</code></td>"
                    f"<td>{row['market']}</td>"
                    f"<td>{int(row['holding_days'])}</td>"
                    f"<td>{row['buy_price']:,.2f}</td>"
                    f"<td>{row['sell_price']:,.2f}</td>"
                    f"<td class='{color_cls}'>{('+' if row['net_pnl'] >= 0 else '')}{row['net_pnl']:,.0f}</td>"
                    f"<td class='{color_cls}'>{('+' if row['return_pct'] >= 0 else '')}{row['return_pct']*100:,.2f}%</td>"
                    "</tr>"
                )
            return f"""
      <h3 style="margin-top:18px;font-size:13px;">{title}</h3>
      <table>
        <tr>
          <th>平仓日</th>
          <th>股票</th>
          <th>板块</th>
          <th>持仓天数</th>
          <th>买入价</th>
          <th>卖出价</th>
          <th>净盈亏</th>
          <th>收益率</th>
        </tr>
        {rows_html}
      </table>"""

        ranking_table = f"""
    <div class="section">
      <h2>盈亏排行 Top / Bottom</h2>
      {_ranking(f"盈利前 {len(top)} 笔", top, "pos")}
      {_ranking(f"亏损前 {len(bottom)} 笔", bottom, "neg")}
    </div>"""

        return trades_table + pnl_table + ranking_table


# ─────────────────────────────────────────────────────────────────────────────
# 私有格式化工具（模块内部使用）
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_pct(v: float, d: int = 2) -> str:
    if math.isnan(v):
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.{d}f}%"


def _fmt_f(v: float, d: int = 2) -> str:
    if math.isnan(v):
        return "—"
    return f"{v:.{d}f}"
