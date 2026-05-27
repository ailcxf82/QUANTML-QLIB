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

A 股成本结构（W1 修正后默认值，2026-05-26）：
  买入  open_cost  = 0.0035 = 佣金 0.05% + 过户费 0.10% + 滑点 0.10% + 缓冲 0.20%
  卖出  close_cost = 0.0045 = 佣金 0.05% + 印花 0.10% + 过户 0.10% + 滑点 0.10% + 缓冲 0.20%
  双边合计 ≈ 0.80%（含 10bp×2 滑点估计，对应 csi500 中小盘 topk=8 单票实际滑点）
  最低费用 5 元 / 单边
  涨跌停  ±9.5%（A 股 ±10% 阈值前保护）

历史变更：
  - 之前（2026-05 前）：open_cost=0.0025, close_cost=0.0035，未含滑点
    实盘 vs 回测 sharpe gap ≈ 0.3 的主要来源之一
  - W1 修正（2026-05-26，docs/lessons/2026-05-26_w1_cost_alignment.md）：
    双边各 +10bp 滑点估计，对应 csi500 中小盘 5min VWAP 实测分布的 90% 分位
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CostModel:
    """
    A 股交易成本模型配置。

    Attributes:
        open_cost:       买入费率（默认 0.35%：佣金+过户+滑点+缓冲）
        close_cost:      卖出费率（默认 0.45%：佣金+印花+过户+滑点+缓冲）
        min_cost:        单笔最低费用（元）
        limit_threshold: 涨跌停保护阈值（0.095 表示 ±9.5%，触达则无法成交）
        deal_price:      成交价字段（"$close_qfq" / "$open_qfq" / "vwap"）
    """
    open_cost: float = 0.0035
    close_cost: float = 0.0045
    min_cost: float = 5.0
    limit_threshold: float = 0.095
    deal_price: str = "$close_qfq"

    @property
    def roundtrip_cost(self) -> float:
        """双边手续费合计（不含最低费用效应）。"""
        return self.open_cost + self.close_cost

    @property
    def slippage_estimate(self) -> float:
        """双边滑点估计（成本中扣除佣金/印花/过户后的剩余部分）。"""
        # 固定项：佣金 0.05%×2 + 印花 0.10%（仅卖）+ 过户 0.10%×2 = 0.40%
        fixed = 0.0005 * 2 + 0.0010 + 0.0010 * 2
        return max(0.0, self.roundtrip_cost - fixed)

    def describe(self) -> str:
        return (
            f"A 股成本模型\n"
            f"    买入手续费   : {self.open_cost*100:.3f}%\n"
            f"    卖出手续费   : {self.close_cost*100:.3f}%（含 0.10% 印花税）\n"
            f"    双边合计     : {self.roundtrip_cost*100:.3f}%\n"
            f"    其中滑点估计 : {self.slippage_estimate*100:.3f}%（双边）\n"
            f"    最低费用     : {self.min_cost:.0f} 元\n"
            f"    成交价       : {self.deal_price}\n"
            f"    涨跌停阈值   : ±{self.limit_threshold*100:.1f}%"
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
        """从配置中提取 CostModel，方便报告中展示成本参数。

        默认值与 W1 修正后的生产配置对齐（含滑点估计）；
        历史 yaml 显式提供时以 yaml 为准。
        """
        ekw = backtest_cfg.get("exchange_kwargs", {})
        return CostModel(
            open_cost=ekw.get("open_cost", 0.0035),
            close_cost=ekw.get("close_cost", 0.0045),
            min_cost=ekw.get("min_cost", 5.0),
            limit_threshold=ekw.get("limit_threshold", 0.095),
            deal_price=ekw.get("deal_price", "$close_qfq"),
        )
