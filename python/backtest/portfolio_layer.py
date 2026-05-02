"""
组合状态层（Portfolio State Layer）
=====================================
职责：
  - 解析 Qlib backtest() 返回的 portfolio_metric_dict
  - 提取干净的每期收益率、基准收益率、换手率序列
  - 计算累计净值（从 1 起算）
  - 兼容日频（freq_key="1day"）和分钟频（freq_key="1min"）

输入:  portfolio_metric_dict（qlib.backtest 返回），freq_key
输出:  PortfolioState（干净的 pd.Series + 统计字段）

Qlib portfolio_metric_dict 结构:
  {
    "1day": (report_df, risk_dict),   # report_df 含 return/bench/cost/turnover 列
    "1min": (...),
  }
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd


@dataclass
class PortfolioState:
    """
    回测后组合状态的完整快照。

    Attributes:
        returns:              策略每期净值收益率（去掉手续费后）
        benchmark:            基准每期收益率
        turnover:             每期换手率（0~1，NaN 表示不可用）
        excess_return:        超额收益率 = returns - benchmark
        cumulative_return:    策略累计净值（期初=1）
        cumulative_benchmark: 基准累计净值（期初=1）
        n_periods:            总交易期数
        freq_key:             频率标识（"1day" / "1min"）
        raw_report_df:        原始 report_df，保留备用
    """
    returns: pd.Series
    benchmark: pd.Series
    turnover: pd.Series
    excess_return: pd.Series
    cumulative_return: pd.Series
    cumulative_benchmark: pd.Series
    n_periods: int
    freq_key: str
    raw_report_df: pd.DataFrame


class PortfolioLayer:
    """
    组合状态层：将 Qlib 内部数据结构转换为干净的 PortfolioState。

    隔离了 Qlib 返回格式变化对上层分析代码的影响。
    """

    def parse(
        self,
        portfolio_metric_dict: Dict[str, Any],
        freq_key: str = "1day",
    ) -> PortfolioState:
        """
        解析 portfolio_metric_dict，提取组合状态。

        Args:
            portfolio_metric_dict: qlib.backtest.backtest() 的第一个返回值
            freq_key:              频率键（"1day" / "1min"），用于索引字典

        Returns:
            PortfolioState

        Raises:
            ValueError: 如果无法从字典中找到有效的 report_df
        """
        # 优先使用 freq_key 对应的条目，找不到则取第一个
        port_tuple = portfolio_metric_dict.get(freq_key)
        if port_tuple is None:
            if not portfolio_metric_dict:
                raise ValueError("portfolio_metric_dict 为空，回测未产生任何组合数据")
            port_tuple = next(iter(portfolio_metric_dict.values()))

        report_df: pd.DataFrame = (
            port_tuple[0] if isinstance(port_tuple, (tuple, list)) else port_tuple
        )

        if not isinstance(report_df, pd.DataFrame) or report_df.empty:
            raise ValueError(
                f"无法从 portfolio_metric_dict[{freq_key!r}] 中提取有效的 report_df。"
                f"实际类型: {type(report_df)}"
            )

        # 提取核心序列，缺失列做安全降级
        returns = report_df["return"].fillna(0.0)
        benchmark = report_df["bench"].fillna(0.0)

        if "turnover" in report_df.columns:
            turnover = report_df["turnover"].fillna(float("nan"))
        else:
            turnover = pd.Series(float("nan"), index=returns.index, name="turnover")

        excess_return = returns - benchmark
        cumulative_return = (1.0 + returns).cumprod()
        cumulative_benchmark = (1.0 + benchmark).cumprod()

        return PortfolioState(
            returns=returns,
            benchmark=benchmark,
            turnover=turnover,
            excess_return=excess_return,
            cumulative_return=cumulative_return,
            cumulative_benchmark=cumulative_benchmark,
            n_periods=len(returns),
            freq_key=freq_key,
            raw_report_df=report_df,
        )

    def to_dataframe(self, state: PortfolioState) -> pd.DataFrame:
        """将 PortfolioState 转为宽格式 DataFrame，方便持久化。"""
        return pd.DataFrame(
            {
                "return": state.returns,
                "bench": state.benchmark,
                "turnover": state.turnover,
                "excess": state.excess_return,
                "cum_return": state.cumulative_return,
                "cum_bench": state.cumulative_benchmark,
            }
        )
