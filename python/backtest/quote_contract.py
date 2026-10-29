"""Adapt Tushare full-field daily data to Qlib's execution quote contract.

Tushare `change` is a CNY difference and `pct_chg` is in percent. Qlib uses
`$change` as a fractional return and `$close` to mark positions. Adapt only
the Exchange's in-memory quotes, never the provider or prediction features.
"""
import numpy as np

from qlib.backtest.exchange import Exchange
from qlib.backtest.decision import Order
from .fees import AShareFees


class TushareQfqExchange(Exchange):
    """Daily QFQ execution/valuation with percentage-based limit checks."""

    def __init__(self, *, freq='day', deal_price='$close_qfq', subscribe_fields=None, fee_model=None, **kwargs):
        if freq not in ('day', '1day') or deal_price != '$close_qfq':
            raise ValueError('tushare_qfq_v1 requires daily $close_qfq execution')
        if kwargs.get('extra_quote') is not None:
            raise ValueError('extra_quote must have a separately validated quote contract')
        self.fees = AShareFees(**fee_model) if fee_model is not None else None
        if self.fees is not None:
            if kwargs.get('impact_cost', 0) != 0:
                raise ValueError('Explicit fees require impact_cost=0; slippage is charged separately')
            if any(key in kwargs for key in ('open_cost', 'close_cost', 'min_cost')):
                raise ValueError('Do not mix explicit fees with legacy aggregate costs')
            # Qlib strategies use these proportional rates for initial sizing.
            kwargs.update(open_cost=self.fees.rate(), close_cost=self.fees.rate(True),
                          min_cost=self.fees.minimum_commission, impact_cost=0)
        fields = list(dict.fromkeys([*(subscribe_fields or []), '$pct_chg', '$close_qfq']))
        super().__init__(freq=freq, deal_price=deal_price, subscribe_fields=fields, **kwargs)

    def _calc_trade_info_by_order(self, order, position, dealt_order_amount):
        if self.fees is None:
            return super()._calc_trade_info_by_order(order, position, dealt_order_amount)
        # Preserve Qlib quote, volume, position and lot constraints; apply the
        # commission floor separately from taxes and slippage during cash sizing.
        price = float(self.get_deal_price(order.stock_id, order.start_time, order.end_time,
                                          direction=order.direction))
        order.factor = self.get_factor(order.stock_id, order.start_time, order.end_time)
        order.deal_amount = order.amount
        self._clip_amount_by_volume(order, dealt_order_amount)
        if order.direction == Order.BUY:
            if position is not None:
                available = self.fees.affordable_buy_value(position.get_cash()) / price
                order.deal_amount = min(order.deal_amount, available)
            order.deal_amount = self.round_amount_by_trade_unit(order.deal_amount, order.factor)
        elif order.direction == Order.SELL:
            if position is not None:
                current = position.get_stock_amount(order.stock_id) if position.check_stock(order.stock_id) else 0
                if not np.isclose(order.deal_amount, current):
                    order.deal_amount = self.round_amount_by_trade_unit(min(current, order.deal_amount), order.factor)
                value = order.deal_amount * price
                if position.get_cash() + value < self.fees.total(value, sell=True):
                    order.deal_amount = 0
        else:
            raise ValueError(f'Unsupported trade direction: {order.direction}')
        value = order.deal_amount * price
        if value <= 1e-5:
            order.deal_amount = 0
            return price, 0.0, 0.0
        return price, value, self.fees.total(value, sell=order.direction == Order.SELL)

    def get_quote_from_qlib(self):
        super().get_quote_from_qlib()
        frame = self.quote_df.copy()
        qfq = frame['$close_qfq']
        valid_price = np.isfinite(qfq) & (qfq > 0)
        frame['$close'] = qfq.where(valid_price)
        frame['$change'] = frame['$pct_chg'] / 100.0
        self.quote_df = frame
        # Base constructor creates the fast quote accessor only after this hook.
        self._update_limit(self.limit_threshold)
        unknown = ~valid_price | ~np.isfinite(frame['$change'])
        self.quote_df.loc[unknown, ['limit_buy', 'limit_sell']] = True
        self.quote_contract_audit = {
            'name': 'tushare_qfq_v1', 'rows': len(frame),
            'invalid_price_rows': int((~valid_price).sum()),
            'invalid_change_rows': int((~np.isfinite(frame['$change'])).sum()),
            'deal_and_valuation_field': '$close_qfq', 'change_expression': '$pct_chg/100',
        }
