"""Accounting regressions: known cash flows, missing evidence and export round trips."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.analysis_layer import AnalysisLayer
from backtest.portfolio_layer import PortfolioLayer

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/audit_saved_costs.py"
spec = importlib.util.spec_from_file_location("audit_saved_costs", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def ledger():
    # Cash account 1,000 -> 1,095 -> 1,095.90; exact fees 5 and 21.
    dates = pd.date_range("2026-01-05", periods=2)
    report = pd.DataFrame({
        "return": [0.10, 0.02], "cost": [5 / 1000, 21 / 1095],
        "bench": [0.01, -0.01], "turnover": [0.5, 200 / 1095],
        "account": [1095.0, 1095.9], "cash": [595.0, 400.0],
    }, index=dates)
    trades = pd.DataFrame({"datetime": dates, "value": [500.0, 200.0], "cost": [5.0, 21.0]})
    return report, trades


def test_net_returns_match_account_and_export_is_not_double_charged(ledger):
    report, _ = ledger
    layer = PortfolioLayer()
    state = layer.parse({"1day": (report, {})})
    np.testing.assert_allclose(state.cumulative_return, report.account / 1000)
    values = AnalysisLayer().compute_metrics(state)
    assert values.total_return == pytest.approx(0.0959)
    assert values.annualized_return == pytest.approx(1.0959 ** 126 - 1)
    assert values.sharpe_ratio == pytest.approx(state.returns.mean() / state.returns.std() * np.sqrt(252))
    exported = layer.to_dataframe(state)
    pd.testing.assert_series_equal(exported['return'], report['return'])
    pd.testing.assert_series_equal(exported['cost'], report['cost'])
    pd.testing.assert_series_equal(exported['cash'], report['cash'])
    pd.testing.assert_series_equal(layer.parse({'1day': exported}).returns, state.returns)
    np.testing.assert_allclose(exported.excess, state.returns - report.bench)


@pytest.mark.parametrize("bad_cost", [None, np.nan, np.inf, -0.01])
def test_unknown_or_invalid_cost_is_not_zero(ledger, bad_cost):
    report, _ = ledger
    if bad_cost is None:
        report = report.drop(columns='cost')
    else:
        report.loc[report.index[0], 'cost'] = bad_cost
    with pytest.raises(ValueError):
        PortfolioLayer().parse({'1day': report})


def test_zero_cost_remains_gross(ledger):
    report, _ = ledger
    report['cost'] = 0.0
    state = PortfolioLayer().parse({'1day': report})
    np.testing.assert_allclose(state.returns, report['return'])


def test_restore_legacy_export_from_independent_turnover(ledger):
    report, trades = ledger
    legacy = report.drop(columns=['cost', 'account', 'cash'])
    restored, evidence = audit.reconstruct_costs(legacy, trades)
    assert evidence['initial_account_inferred'] == pytest.approx(1000)
    assert evidence['cash_cost_sum'] == pytest.approx(26)
    np.testing.assert_allclose(restored.account, report.account)
    np.testing.assert_allclose(restored.cost, report.cost)
    assert audit.metrics(restored)['total_return'] == pytest.approx(0.0959)
    assert audit.metrics(restored, gross=True)['total_return'] == pytest.approx(0.122)
    pd.testing.assert_frame_equal(legacy, report.drop(columns=['cost', 'account', 'cash']))


def test_restore_checks_original_cost_and_account_when_available(ledger):
    report, trades = ledger
    audit.reconstruct_costs(report, trades)
    report.loc[report.index[-1], 'account'] += 10
    with pytest.raises(ValueError, match='stored account'):
        audit.reconstruct_costs(report, trades)


def test_missing_trade_is_rejected(ledger):
    report, trades = ledger
    with pytest.raises(ValueError, match='reconciliation failed'):
        audit.reconstruct_costs(report, trades.iloc[:1])


def test_trade_outside_coverage_is_rejected(ledger):
    report, trades = ledger
    trades.loc[0, 'datetime'] = pd.Timestamp('2025-12-31')
    with pytest.raises(ValueError, match='Trade dates'):
        audit.reconstruct_costs(report, trades)


def test_cash_only_days_before_first_trade_and_between_trades(ledger):
    report, trades = ledger
    dates = pd.to_datetime(['2026-01-02', '2026-01-05', '2026-01-06', '2026-01-07'])
    expanded = pd.DataFrame({
        'return': [0, .1, 0, .02], 'bench': [0, .01, 0, -.01],
        'turnover': [0, .5, 0, 200 / 1095],
    }, index=dates)
    trades.loc[1, 'datetime'] = dates[-1]
    restored, evidence = audit.reconstruct_costs(expanded, trades)
    assert evidence['initial_account_inferred'] == pytest.approx(1000)
    np.testing.assert_allclose(restored.account, [1000, 1095, 1095, 1095.9])


def test_legacy_mlp_metrics_use_net_return(ledger, monkeypatch):
    import run_mlp_daily_backtest as legacy

    report, _ = ledger
    observed = {}

    def capture(series):
        observed['excess'] = series
        return pd.DataFrame({'risk': [0.0]}, index=['annualized_return'])

    monkeypatch.setattr(legacy, 'risk_analysis', capture)
    result = legacy.compute_metrics(report)
    np.testing.assert_allclose(observed['excess'], report['return'] - report.cost - report.bench)
    assert result['win_rate'] == 1.0
