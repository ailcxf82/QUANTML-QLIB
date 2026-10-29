"""Explicit A-share cash costs; slippage is an assumption, not a broker fee.

Tax reference (checked 2026-10-29): https://one.sse.com.cn/onething/gptz/
Rates apply to this replay's 2025-01-02..2026-10-15 period.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AShareFees:
    commission_rate: float = 0.00035  # assumed inclusive of regulatory fees
    minimum_commission: float = 5.0
    stamp_tax_sell_rate: float = 0.0005
    transfer_rate: float = 0.00001
    slippage_rate: float = 0.001  # each side; fixed cash-equivalent deduction

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in self.__dict__.values()):
            raise ValueError('Fee parameters must be finite and nonnegative')
        if self.rate(True) >= 1:
            raise ValueError('Total proportional cost must be less than 100%')

    def rate(self, sell=False):
        return (self.commission_rate + self.transfer_rate + self.slippage_rate
                + (self.stamp_tax_sell_rate if sell else 0.0))

    def components(self, value, sell=False):
        if not math.isfinite(value) or value < 0:
            raise ValueError('Trade value must be finite and nonnegative')
        return {
            'commission': max(value * self.commission_rate, self.minimum_commission) if value > 0 else 0.0,
            'stamp_tax': value * self.stamp_tax_sell_rate if sell else 0.0,
            'transfer_fee': value * self.transfer_rate,
            'slippage': value * self.slippage_rate,
        }

    def total(self, value, sell=False):
        return sum(self.components(value, sell).values())

    def affordable_buy_value(self, cash):
        # Both inequalities must hold: V*(1+r_all)<=cash and
        # V*(1+r_noncommission)+minimum_commission<=cash.
        return max(0.0, min(cash / (1 + self.rate()),
                           (cash - self.minimum_commission) /
                           (1 + self.transfer_rate + self.slippage_rate)))

    def annotate(self, trades):
        """Split actual deductions and fail if the exchange used another formula."""
        import numpy as np
        import pandas as pd
        result = trades.copy()
        parts = pd.DataFrame([self.components(float(row.value), row.direction == 'SELL')
                              for row in result.itertuples()], index=result.index,
                             columns=['commission', 'stamp_tax', 'transfer_fee', 'slippage'])
        if not np.allclose(parts.sum(axis=1), result.cost, atol=1e-8, rtol=1e-10):
            raise ValueError('Trade component costs do not reconcile to exchange deductions')
        result[parts.columns] = parts
        result['explicit_fees'] = result.commission + result.stamp_tax + result.transfer_fee
        return result
