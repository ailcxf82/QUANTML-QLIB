"""Market simulation settings and cost descriptions.

Daily runs use explicit commission, stamp tax, transfer fee and slippage.
Historical configurations retain their aggregate rates without inventing a
component breakdown. Slippage is a cash-equivalent assumption, not observed fills.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class CostModel:
    """
    A 股交易成本模型配置。

    Attributes:
        open_cost:       历史合并买入费率
        close_cost:      历史合并卖出费率
        min_cost:        单笔最低费用（元）
        limit_threshold: 涨跌停保护阈值（0.095 表示 ±9.5%，触达则无法成交）
        deal_price:      成交价字段（"$close_qfq" / "$open_qfq" / "vwap"）
    """
    fee_model: dict | None = None
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
        """Explicit slippage only; legacy aggregate costs have no known split."""
        return self.fee_model['slippage_rate'] * 2 if self.fee_model is not None else None

    def describe(self) -> str:
        if self.fee_model is not None:
            from .fees import AShareFees
            fee = AShareFees(**self.fee_model)
            return (f"A股分项成本：佣金双边 {fee.commission_rate:.3%}（含规费），"
                    f"最低每笔 {fee.minimum_commission:g} 元；印花税仅卖出 {fee.stamp_tax_sell_rate:.3%}；"
                    f"过户费双边 {fee.transfer_rate:.3%}；滑点假设单边 {fee.slippage_rate:.3%}，"
                    "按成交金额单独扣款，计入净收益。")
        return (f"历史合并成本：买入 {self.open_cost:.3%}，卖出 {self.close_cost:.3%}，"
                f"合并最低费用 {self.min_cost:g} 元；无可靠分项，不推定滑点或税费。")



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
        *,
        start_time=None,
        end_time=None,
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
        contract = backtest_cfg.get('quote_contract')
        if backtest_cfg.get('fee_model') is not None:
            if contract != 'tushare_qfq_v1':
                raise ValueError('Explicit fees require the validated daily quote contract')
            from .fees import AShareFees
            from dataclasses import asdict
            fees = AShareFees(**backtest_cfg['fee_model'])
            if any(k in exchange_kwargs for k in ('open_cost', 'close_cost', 'min_cost')):
                raise ValueError('Do not mix explicit fees with legacy aggregate costs')
            exchange_kwargs['fee_model'] = asdict(fees)
        if contract is not None:
            if contract != 'tushare_qfq_v1':
                raise ValueError(f'Unknown quote contract: {contract}')
            if start_time is None or end_time is None:
                raise ValueError('Quote contract requires explicit backtest dates')
            if 'exchange' in exchange_kwargs:
                raise ValueError('quote_contract cannot override a custom exchange')
            exchange_kwargs.update(start_time=start_time, end_time=end_time)
            return {'exchange': {'class': 'TushareQfqExchange',
                                 'module_path': 'backtest.quote_contract',
                                 'kwargs': exchange_kwargs}}
        return exchange_kwargs

    def extract_cost_model(self, backtest_cfg: Dict[str, Any]) -> CostModel:
        """从配置中提取 CostModel，方便报告中展示成本参数。

        分项配置优先；历史合并费率只展示总额，不推断组成。
        """
        ekw = backtest_cfg.get("exchange_kwargs", {})
        if backtest_cfg.get('fee_model') is not None:
            from .fees import AShareFees
            from dataclasses import asdict
            fee = AShareFees(**backtest_cfg['fee_model'])
            return CostModel(fee_model=asdict(fee), open_cost=fee.rate(), close_cost=fee.rate(True),
                             min_cost=fee.minimum_commission,
                             limit_threshold=ekw.get('limit_threshold', .095),
                             deal_price=ekw.get('deal_price', '$close_qfq'))
        return CostModel(
            open_cost=ekw.get("open_cost", 0.0035),
            close_cost=ekw.get("close_cost", 0.0045),
            min_cost=ekw.get("min_cost", 5.0),
            limit_threshold=ekw.get("limit_threshold", 0.095),
            deal_price=ekw.get("deal_price", "$close_qfq"),
        )
