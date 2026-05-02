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
from typing import Optional

import numpy as np
import pandas as pd

from .portfolio_layer import PortfolioState
from .signal_layer import SignalSummary


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
        output_dir: Optional[Path] = None,
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
                exists = "✓" if (charts_dir / fn).exists() else "○"
                print(f"  │  {exists} {fn:<28} {desc:<{W - 36}} │")
            html = output_dir / "report.html"
            h_exists = "✓" if html.exists() else "○"
            print(f"  │  {h_exists} {'report.html':<28} {'HTML 完整报告':<{W - 36}} │")
            print(f"  └{'─' * (W - 2)}┘")

        print("\n" + _hr() + "\n")

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
    ) -> None:
        """生成自包含 HTML 回测报告。"""
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
