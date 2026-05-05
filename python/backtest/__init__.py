"""
回测模块（Backtest Module）
============================
分层架构，每层职责独立、接口清晰：

    模型预测信号
        ↓
    SignalLayer     — 验证信号质量，计算 IC/ICIR
        ↓
    StrategyLayer   — 将信号+策略参数组合为 Qlib 策略配置
        ↓
    ExecutionLayer  — 构建执行器（日频/分钟频）
        ↓
    MarketLayer     — 封装 A 股交易成本模型
        ↓  (调用 qlib.backtest.backtest)
    PortfolioLayer  — 解析回测结果，提取净值/换手/基准序列
        ↓
    AnalysisLayer   — 计算绩效指标，生成控制台报告+图表+HTML

使用方式:
    from backtest import BacktestEngine
    engine = BacktestEngine()
    result = engine.run(pred_score, cfg, output_dir=run_ctx.output_dir, label=label)
"""

from .engine import BacktestEngine, BacktestResult
from .signal_layer import SignalLayer, SignalSummary
from .strategy_layer import StrategyLayer
from .execution_layer import ExecutionLayer
from .market_layer import MarketLayer, CostModel
from .portfolio_layer import PortfolioLayer, PortfolioState
from .analysis_layer import AnalysisLayer, PerformanceMetrics
# 五段式策略管线（中频机构化）：候选 → 加权 → 约束 → 提交
from . import strategy as strategy_pipeline  # noqa: F401  保证子包 import 路径可用

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "SignalLayer",
    "SignalSummary",
    "StrategyLayer",
    "ExecutionLayer",
    "MarketLayer",
    "CostModel",
    "PortfolioLayer",
    "PortfolioState",
    "AnalysisLayer",
    "PerformanceMetrics",
    "strategy_pipeline",
]
