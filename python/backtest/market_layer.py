"""
市场模拟层（Market Simulation Layer）
======================================
职责：
  - 封装 A 股真实交易成本模型（佣金、印花税、滑点、最小费用）
  - 处理涨跌停限制（limit_threshold）
  - 构造 Qlib exchange_kwargs，注入频率信息
  - 提供成本说明描述供报告使用

输入: backtest_cfg（dict，来自 YAML）+ freq_key（"1day" / "1min"）
输出: exchange_kwargs dict，可直接传入 qlib.backtest.backtest()

A 股成本结构（默认值）:
  买入:  佣金 0.05%（open_cost=0.0005），最低 5 元
  卖出:  佣金 0.05% + 印花税 0.10% = 0.15%（close_cost=0.0015），最低 5 元
  滑点:  通过 deal_price="close" 隐含（开盘/收盘不同场景可配置）
  涨跌停: 9.5% 阈值（A 股 ±10% 停牌前保护）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CostModel:
    """
    A 股交易成本模型配置。

    Attributes:
        open_cost:       买入费率（佣金，默认 0.05%）
        close_cost:      卖出费率（印花税 0.10% + 佣金 0.05%，默认 0.15%）
        min_cost:        单笔最低费用（元）
        limit_threshold: 涨跌停保护阈值（0.095 表示 ±9.5%，触达则无法成交）
        deal_price:      成交价字段（"close"/"open"/"vwap"）
    """
    open_cost: float = 0.0005
    close_cost: float = 0.0015
    min_cost: float = 5.0
    limit_threshold: float = 0.095
    deal_price: str = "close"

    @property
    def roundtrip_cost(self) -> float:
        """双边手续费合计（不含最低费用效应）。"""
        return self.open_cost + self.close_cost

    def describe(self) -> str:
        return (
            f"A 股成本模型\n"
            f"    买入手续费 : {self.open_cost*100:.3f}%\n"
            f"    卖出手续费 : {self.close_cost*100:.3f}%（含 0.10% 印花税）\n"
            f"    双边合计   : {self.roundtrip_cost*100:.3f}%\n"
            f"    最低费用   : {self.min_cost:.0f} 元\n"
            f"    成交价     : {self.deal_price}\n"
            f"    涨跌停阈值 : ±{self.limit_threshold*100:.1f}%"
        )


class MarketLayer:
    """
    市场模拟层：从配置中读取交易成本参数，构建 exchange_kwargs。

    Qlib Exchange 负责底层撮合逻辑（报价匹配、涨跌停过滤），
    此层的职责是参数化和文档化成本模型，确保可追溯。
    """

    def build(
        self,
        backtest_cfg: Dict[str, Any],
        freq_key: str,
    ) -> Dict[str, Any]:
        """
        构建 exchange_kwargs。

        Args:
            backtest_cfg: 来自 YAML 的 backtest 配置段（含 exchange_kwargs 子段）
            freq_key:     频率标识（"1day" 或 "1min"），用于自动填写 freq 字段

        Returns:
            Qlib exchange_kwargs dict
        """
        exchange_kwargs = dict(backtest_cfg.get("exchange_kwargs", {}))
        # 确保 freq 字段与当前回测频率一致
        exchange_kwargs["freq"] = "1min" if freq_key == "1min" else "day"
        return exchange_kwargs

    def extract_cost_model(self, backtest_cfg: Dict[str, Any]) -> CostModel:
        """从配置中提取 CostModel，方便报告中展示成本参数。"""
        ekw = backtest_cfg.get("exchange_kwargs", {})
        return CostModel(
            open_cost=ekw.get("open_cost", 0.0005),
            close_cost=ekw.get("close_cost", 0.0015),
            min_cost=ekw.get("min_cost", 5.0),
            limit_threshold=ekw.get("limit_threshold", 0.095),
            deal_price=ekw.get("deal_price", "close"),
        )
