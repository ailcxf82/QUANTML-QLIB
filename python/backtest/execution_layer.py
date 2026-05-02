"""
执行层（Execution Layer）
==========================
职责：
  - 根据回测频率构建 Qlib Executor 配置
  - 日频: SimulatorExecutor（time_per_step="day"）
  - 分钟频: NestedExecutor（外层日频调仓 + 内层分钟 TWAP 执行）
  - 提供执行配置描述供日志使用

输入: executor_cfg（dict，来自 YAML）+ freq_key（"1day" / "1min"）
输出: Qlib 可直接使用的 executor 配置 dict

分钟频嵌套执行器设计:
  NestedExecutor（外层，time_per_step="day"）
    └── SimulatorExecutor（内层，time_per_step="1min"）
        ← 内层策略: TWAPStrategy（将日订单均匀拆分到分钟级别执行）
  当前 data_ready=false 时自动回退为日频执行器。
"""
from __future__ import annotations

from typing import Any, Dict


class ExecutionLayer:
    """
    执行层：根据频率选择并构建 Qlib Executor 配置。

    日频场景: 直接使用配置文件中的 SimulatorExecutor，无需修改。
    分钟频场景: 将 SimulatorExecutor 包装进 NestedExecutor，实现
               日频信号 → 分钟级 TWAP 拆单执行的真实模拟。
    """

    # 分钟频内层策略：TWAP 均匀分配订单到每根分钟 bar
    _TWAP_STRATEGY_CFG: Dict[str, Any] = {
        "class": "TWAPStrategy",
        "module_path": "qlib.contrib.strategy.signal_strategy",
    }

    def build(
        self,
        executor_cfg: Dict[str, Any],
        freq_key: str,
    ) -> Dict[str, Any]:
        """
        构建执行器配置。

        Args:
            executor_cfg: 来自 YAML 的基础执行器配置
            freq_key:     频率标识（"1day" 或 "1min"）

        Returns:
            Qlib 可直接使用的 executor 配置 dict
        """
        if freq_key == "1min":
            return self._build_nested_minute(executor_cfg)
        return executor_cfg

    def _build_nested_minute(self, inner_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        分钟频嵌套执行器：
          - 外层（NestedExecutor, time_per_step="day"）负责接收每日持仓决策
          - 内层（SimulatorExecutor, time_per_step="1min"）负责在日内按 TWAP 执行
          - generate_portfolio_metrics=True 保证外层能输出组合指标
        """
        return {
            "class": "NestedExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
                "inner_executor": inner_cfg,
                "inner_strategy": self._TWAP_STRATEGY_CFG,
            },
        }

    def describe(self, freq_key: str) -> str:
        """返回执行器的可读描述，用于日志和报告。"""
        if freq_key == "1min":
            return (
                "NestedExecutor（外层日频调仓 → 内层分钟 TWAP 执行）\n"
                "    外层: time_per_step=day，接收组合信号\n"
                "    内层: time_per_step=1min，TWAP 均匀拆单"
            )
        return "SimulatorExecutor（日频，time_per_step=day）"
