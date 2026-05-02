"""
回测引擎（BacktestEngine）
===========================
编排各功能层，完成端到端回测与分析。

层级调用顺序:
    pred_score (模型输出)
        ↓
    [1] SignalLayer    — 验证信号合法性，计算 IC/ICIR
        ↓
    [2] StrategyLayer  — 注入信号，构造 Qlib 策略配置
        ↓
    [3] ExecutionLayer — 构建执行器（日频/分钟频）
        ↓
    [4] MarketLayer    — 构建交易成本 exchange_kwargs
        ↓
    qlib.backtest.backtest()  （Qlib 内部撮合引擎）
        ↓
    [5] PortfolioLayer — 解析回测结果，提取净值/换手序列
        ↓
    [6] AnalysisLayer  — 计算绩效指标，打印报告，保存图表+HTML

输入:  pred_score（pd.Series）+ cfg（合并后配置字典）
输出:  BacktestResult（包含 PortfolioState + PerformanceMetrics + SignalSummary）
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from .signal_layer import SignalLayer, SignalSummary
from .strategy_layer import StrategyLayer
from .execution_layer import ExecutionLayer
from .market_layer import MarketLayer
from .portfolio_layer import PortfolioLayer, PortfolioState
from .analysis_layer import AnalysisLayer, PerformanceMetrics


@dataclass
class BacktestResult:
    """
    回测结果汇总。

    Attributes:
        run_id:          实验唯一标识
        portfolio_state: 组合净值/换手/基准等时间序列
        metrics:         完整绩效指标
        signal_summary:  信号质量统计（IC/ICIR 等）
        report_df:       宽格式 DataFrame（含 return/bench/turnover/excess/cum_*）
        indicator_df:    Qlib indicator 指标（成交率等，可为空）
    """
    run_id: str
    portfolio_state: PortfolioState
    metrics: PerformanceMetrics
    signal_summary: SignalSummary
    report_df: pd.DataFrame
    indicator_df: pd.DataFrame


class BacktestEngine:
    """
    回测引擎：按层编排 SignalLayer → StrategyLayer → ExecutionLayer
             → MarketLayer → [qlib backtest] → PortfolioLayer → AnalysisLayer。

    使用示例:
        engine = BacktestEngine()
        result = engine.run(pred_score, cfg, output_dir=run_ctx.output_dir, label=test_label)
    """

    def __init__(self) -> None:
        self._signal_layer = SignalLayer()
        self._strategy_layer = StrategyLayer()
        self._execution_layer = ExecutionLayer()
        self._market_layer = MarketLayer()
        self._portfolio_layer = PortfolioLayer()
        self._analysis_layer = AnalysisLayer()

    def run(
        self,
        pred_score: pd.Series,
        cfg: Dict[str, Any],
        output_dir: Optional[Path] = None,
        label: Optional[pd.Series] = None,
    ) -> BacktestResult:
        """
        端到端回测主流程。

        Args:
            pred_score:  模型预测打分，MultiIndex(datetime, instrument)
            cfg:         合并后的完整配置字典
            output_dir:  结果输出目录（传入时生成图表与 HTML 报告）
            label:       测试集实际标签序列（传入时计算 IC/ICIR）

        Returns:
            BacktestResult
        """
        from qlib.backtest import backtest as qlib_backtest

        run_id = cfg.get("experiment", {}).get("run_id", "unknown")
        freq_key: str = cfg.get("model_meta", {}).get("freq_key", "1day")
        model_name: str = cfg.get("model_meta", {}).get("name", "unknown")
        freq_name: str = cfg.get("freq_meta", {}).get("name", "unknown")

        # ──────────────────────────────────────────────────────────────────
        # [1] 信号层：校验信号，计算 IC/ICIR
        # ──────────────────────────────────────────────────────────────────
        signal_summary = self._signal_layer.process(
            pred_score, label=label, run_id=run_id
        )

        # ──────────────────────────────────────────────────────────────────
        # [2] 策略层：信号 + 策略参数 → Qlib 策略配置
        # ──────────────────────────────────────────────────────────────────
        strategy_config = self._strategy_layer.build(cfg["strategy"], pred_score)

        # ──────────────────────────────────────────────────────────────────
        # [3] 执行层：构建执行器（日频/分钟频）
        # ──────────────────────────────────────────────────────────────────
        executor_config = self._execution_layer.build(cfg["executor"], freq_key)

        # ──────────────────────────────────────────────────────────────────
        # [4] 市场层：构建交易成本 exchange_kwargs
        # ──────────────────────────────────────────────────────────────────
        exchange_kwargs = self._market_layer.build(cfg["backtest"], freq_key)

        # ──────────────────────────────────────────────────────────────────
        # 调用 Qlib 回测引擎
        # ──────────────────────────────────────────────────────────────────
        exp_cfg = cfg["experiment"]
        portfolio_metric_dict, indicator_dict = qlib_backtest(
            start_time=cfg["dataset"]["kwargs"]["segments"]["test"][0],
            end_time=exp_cfg["end_time"],
            strategy=strategy_config,
            executor=executor_config,
            benchmark=exp_cfg["benchmark"],
            account=exp_cfg["account"],
            exchange_kwargs=exchange_kwargs,
        )

        # ──────────────────────────────────────────────────────────────────
        # [5] 组合状态层：解析 portfolio_metric_dict
        # ──────────────────────────────────────────────────────────────────
        portfolio_state = self._portfolio_layer.parse(portfolio_metric_dict, freq_key)

        # 解析 indicator_dict（成交率等指标）
        indicator_df = self._parse_indicator_dict(indicator_dict)

        # ──────────────────────────────────────────────────────────────────
        # [6] 分析层：计算指标 + 生成报告
        # ──────────────────────────────────────────────────────────────────
        metrics = self._analysis_layer.compute_metrics(portfolio_state, freq_key)

        if output_dir is not None:
            self._analysis_layer.save_charts(
                portfolio_state, metrics, output_dir, run_id, signal_summary
            )
            self._analysis_layer.save_html_report(
                output_dir, run_id, model_name, freq_name,
                metrics, signal_summary, portfolio_state
            )

        self._analysis_layer.print_report(
            run_id=run_id,
            model_name=model_name,
            freq_name=freq_name,
            metrics=metrics,
            signal_summary=signal_summary,
            state=portfolio_state,
            output_dir=output_dir,
        )

        # 宽格式 report_df（供持久化）
        report_df = self._portfolio_layer.to_dataframe(portfolio_state)

        return BacktestResult(
            run_id=run_id,
            portfolio_state=portfolio_state,
            metrics=metrics,
            signal_summary=signal_summary,
            report_df=report_df,
            indicator_df=indicator_df,
        )

    def metrics_to_dict(self, result: BacktestResult) -> Dict[str, float]:
        """将 PerformanceMetrics 转为 dict，便于写入 metrics.json。"""
        m = result.metrics
        s = result.signal_summary
        return {
            "annualized_return": m.annualized_return,
            "total_return": m.total_return,
            "annualized_volatility": m.annualized_volatility,
            "max_drawdown": m.max_drawdown,
            "sharpe_ratio": m.sharpe_ratio,
            "calmar_ratio": m.calmar_ratio,
            "information_ratio": m.information_ratio,
            "annualized_excess_return": m.annualized_excess_return,
            "excess_volatility": m.excess_volatility,
            "avg_turnover": m.avg_turnover,
            "win_rate": m.win_rate,
            "benchmark_annualized_return": m.benchmark_annualized_return,
            "benchmark_max_drawdown": m.benchmark_max_drawdown,
            # 信号质量
            "ic_mean": s.ic_mean,
            "ic_std": s.ic_std,
            "icir": s.icir,
            "ic_positive_ratio": s.ic_positive_ratio,
        }

    @staticmethod
    def _parse_indicator_dict(indicator_dict: Dict[str, Any]) -> pd.DataFrame:
        """从 indicator_dict 中提取 DataFrame，找不到时返回空 DataFrame。"""
        dfs = []
        for _fk, ind_tuple in indicator_dict.items():
            if isinstance(ind_tuple, (tuple, list)) and len(ind_tuple) > 0:
                ind_df = ind_tuple[0]
                if isinstance(ind_df, pd.DataFrame) and not ind_df.empty:
                    dfs.append(ind_df)
        return pd.concat(dfs) if dfs else pd.DataFrame()
