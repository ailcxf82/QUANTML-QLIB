"""Read-only fixed-fill slippage sensitivity. No model or backtest imports.

Keep all saved trade dates, quantities, execution/marking prices and explicit
fees fixed. Change only the cash-equivalent slippage debit. Saved cash plus
the cumulative debit difference gives the new cash and account paths. Savings
remain idle (zero interest, no reinvestment). Negative cash is an infeasible
unfinanced path, not a new executable strategy result.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    hashes = {str(p): digest(p) for p in source.rglob('*') if p.is_file()}
    trades = pd.read_parquet(source / 'trades.parquet')
    report = pd.read_parquet(source / 'backtest_report.parquet')
    metrics = json.loads((source / 'metrics.json').read_text(encoding='utf-8'))
    initial = 300000.0
    old_rate = metrics['fee_model']['slippage_rate']
    assert old_rate == .001
    assert len(report) == 430
    assert str(report.index[-1].date()) == '2026-10-15'
    assert (trades.value > 0).all()
    np.testing.assert_allclose(trades.slippage, trades.value * old_rate)
    np.testing.assert_allclose(trades.cost, trades.explicit_fees + trades.slippage)
    values = trades.groupby('datetime').value.sum().reindex(report.index, fill_value=0)
    explicit = trades.groupby('datetime').explicit_fees.sum().reindex(report.index, fill_value=0)
    previous_account = report.account.shift(fill_value=initial)
    np.testing.assert_allclose(report.cost * previous_account, explicit + values * old_rate, atol=1e-7)
    # Independent daily gross currency P&L from the stored Qlib gross return.
    gross_pnl = report['return'] * previous_account
    rows, paths = [], []
    for bp in [0, 2, 5, 10, 15, 20]:
        slip = values * bp / 10000
        delta = (values * old_rate - slip).cumsum()
        nav, cash = report.account + delta, report.cash + delta
        direct = initial + (gross_pnl - explicit - slip).cumsum()
        np.testing.assert_allclose(nav, direct, atol=1e-7, rtol=1e-12)
        returns = nav / nav.shift(fill_value=initial) - 1
        np.testing.assert_allclose(initial * (1 + returns).cumprod(), nav, atol=1e-7)
        drawdown = nav / nav.cummax().clip(lower=initial) - 1
        total_return = float(nav.iloc[-1] / initial - 1)
        negative_days = int((cash < -1e-8).sum())
        rows.append(dict(slippage_bp_per_side=bp, slippage_rate=bp/10000,
                         explicit_fees=float(explicit.sum()), slippage_cost=float(slip.sum()),
                         total_cost=float(explicit.sum()+slip.sum()),
                         savings_vs_10bp=float(delta.iloc[-1]), ending_account=float(nav.iloc[-1]),
                         total_return=total_return, annualized_return=(1+total_return)**(252/len(report))-1,
                         max_drawdown=float(drawdown.min()), min_end_of_day_cash=float(cash.min()),
                         negative_cash_days=negative_days,
                         status='saved_actual' if bp==10 else ('unfunded_stress_only' if negative_days else 'fixed_fill_cash_adjustment')))
        paths.append(pd.DataFrame({'slippage_bp_per_side':bp,'account':nav,'cash':cash,
                                   'slippage':slip,'explicit_fees':explicit,'net_return':returns,
                                   'drawdown':drawdown}).rename_axis('datetime').reset_index())
    summary = pd.DataFrame(rows)
    actual = summary[summary.slippage_bp_per_side == 10].iloc[0]
    assert np.isclose(actual.total_return, metrics['metrics']['total_return'])
    assert np.isclose(actual.max_drawdown, metrics['metrics']['max_drawdown'])
    summary.to_csv(out / 'scenarios.csv', index=False)
    pd.concat(paths, ignore_index=True).to_parquet(out / 'daily_fixed_fill_scenarios.parquet', index=False)
    trades.groupby('direction')[['value','commission','stamp_tax','transfer_fee','slippage','cost']].sum().to_csv(out/'cost_by_side.csv')
    evidence = {
        'source':str(source),'start':str(report.index[0].date()),'end':str(report.index[-1].date()),
        'days':len(report),'trades':len(trades),'training_runs':0,'prediction_runs':0,'backtest_runs':0,
        'initial_capital':initial,'total_traded_value':float(values.sum()),
        'slippage_fraction_of_all_cost':float(trades.slippage.sum()/trades.cost.sum()),
        'each_1bp_cost':float(values.sum()/10000),
        'each_1bp_total_return_pp':float(values.sum()/10000/initial*100),
        'method':__doc__, 'base_10bp_reconciles':True,'gross_currency_pnl_reconciles':True,
        'zero_slippage_is_counterfactual_not_realistic_claim':True,
        'negative_cash_check':'End-of-day only; no intraday funding or transaction-order validation.',
        'source_checks':[
            {'path':'docs/lessons/2026-05-26_w1_cost_alignment_and_optuna_v2.md','lines':'35-36',
             'finding':'States CSI500 5min VWAP 90th percentile without sample dates, counts, raw fills or quantile calculation.'},
            {'path':'docs/时序模型优化.md','lines':'39',
             'finding':'Describes fixed 10bp per side as a coarse assumption; per-stock calibration remains planned.'},
            {'path':'artifacts/_audit/w1_cost_sanity_20260526_150236.json',
             'finding':'Changes close to close_qfq and aggregate fee rates simultaneously; no measured execution-price slippage sample.'},
            {'path':'git:676388d:configs/freq/daily.yaml',
             'finding':'Historical comment compares 5min VWAP with open for topk=8; current replay uses close_qfq and topk=5.'}],
        'calibration_conclusion':'No reproducible empirical calibration found in the inspected project documents, scripts and named W1 audit outputs. 10bp remains an unvalidated scenario assumption.',
        'source_sha256':hashes,
        'scenarios':rows}
    for file, old in hashes.items():
        assert digest(Path(file)) == old, file
    evidence['source_files_unchanged']=True
    (out/'analysis.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'analysis_source.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    print(summary.to_string(index=False))
    print(json.dumps({k:evidence[k] for k in ['total_traded_value','each_1bp_cost','each_1bp_total_return_pp','source_files_unchanged']},indent=2))


if __name__ == '__main__':
    main()
