"""Read-only decomposition of saved results; no fit, predict, or backtest calls.

Holdings are cumulative executed quantities, never simulated orders. Top-five
comparisons are descriptive rankings, not replacement strategy evaluations.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from audit_saved_costs import reconstruct_costs


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def total(s):
    return float(np.expm1(np.log1p(s).sum()))


def holdings(trades, dates):
    signed = trades.assign(quantity=trades.amount*np.where(trades.direction=='BUY',1,-1))
    changes = signed.pivot_table(index='datetime', columns='stock_id', values='quantity',aggfunc='sum',fill_value=0)
    h = changes.reindex(dates,fill_value=0).cumsum()
    if h.min().min() < -1e-5:
        raise ValueError('Executed trades imply negative holdings')
    return h.where(h>1e-6,0)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    old=ROOT/'artifacts/lgbm_optuna_daily_f2c798bb'
    new=ROOT/'artifacts/frozen_model_replay/20261029_lgbm_optuna_to_20261015_v2'
    repair=ROOT/'artifacts/data_repair/20261029_frozen_lgbm_v1'
    provider=Path('D:/qlib_data/qlib_data')
    sources=[p/n for p in [old,new] for n in ['backtest_report.parquet','signals.parquet','trades.parquet','realized_pnl.parquet']]
    sources += [old/'model.pkl',new/'frozen_model.pkl',new/'universe.json',new/'resolved_config.yaml',repair/'summary.json',repair/'verification.json',provider/'calendars/day.txt']
    hashes={str(p):sha(p) for p in sources}
    ot=pd.read_parquet(old/'trades.parquet'); nt=pd.read_parquet(new/'trades.parquet')
    o, audit = reconstruct_costs(pd.read_parquet(old/'backtest_report.parquet'),ot)
    n=pd.read_parquet(new/'backtest_report.parquet')
    a=pd.read_parquet(old/'signals.parquet').score;b=pd.read_parquet(new/'signals.parquet').score
    common=o.index.intersection(n.index); internal=n.index[(n.index<=o.index.max())&~n.index.isin(o.index)]; extension=n.index[n.index>o.index.max()]
    aligned=pd.DataFrame({f'{name}_{col}':frame.loc[common,col] for name,frame in [('old',o),('new',n)] for col in ['return','net_return','cost','turnover']})
    aligned['gross_log_difference']=np.log1p(aligned.new_return)-np.log1p(aligned.old_return)
    aligned['net_log_difference']=np.log1p(aligned.new_net_return)-np.log1p(aligned.old_net_return)
    aligned.to_csv(out/'aligned_daily.csv')
    windows={
        'old_common':o.loc[common], 'new_common':n.loc[common],
        'new_through_old_end':n.loc[:o.index.max()], 'new_extension':n.loc[extension], 'new_full':n,
        'old_before_financial_repair':o.loc[:'2026-03-31'], 'new_before_financial_repair':n.loc[:'2026-03-31'],
        'old_financial_repair_overlap':o.loc['2026-04-01':], 'new_financial_repair_overlap_common':n.loc[common[common>='2026-04-01']]}
    rows=[]
    for key,d in windows.items():
        cash_cost = d.cash_cost.sum() if 'cash_cost' in d else (d.cost*n.account.shift(fill_value=300000).reindex(d.index)).sum()
        rows.append(dict(window=key,start=str(d.index.min().date()),end=str(d.index.max().date()),days=len(d),gross_return=total(d['return']),net_return=total(d.net_return),cash_cost=float(cash_cost),mean_turnover=float(d.turnover.mean())))
    window=pd.DataFrame(rows).set_index('window');window.to_csv(out/'windows.csv')
    levels=[total(o['return']),total(o.net_return),total(n.loc[common,'net_return']),total(n.loc[:o.index.max(),'net_return']),total(n.net_return)]
    labels=['旧结果毛收益','旧结果扣费后','新结果同332日扣费后','新结果补入5月8日后','新结果延长到10月15日']
    bridge=pd.DataFrame({'stage':labels,'total_return':levels,'change_percentage_points':np.r_[0,np.diff(levels)*100]})
    bridge.to_csv(out/'return_bridge.csv',index=False)
    assert np.isclose(np.diff(levels).sum(),levels[-1]-levels[0])
    # Exact additive log decomposition: common-date gross path + fee wedge.
    logparts={'gross_path_log_difference':float(aligned.gross_log_difference.sum()),
        'cost_wedge_log_difference':float((np.log1p(aligned.new_net_return)-np.log1p(aligned.new_return)-np.log1p(aligned.old_net_return)+np.log1p(aligned.old_return)).sum()),
        'net_log_difference':float(aligned.net_log_difference.sum())}
    assert np.isclose(logparts['gross_path_log_difference']+logparts['cost_wedge_log_difference'],logparts['net_log_difference'])
    # Read existing quote bins without Qlib init, caching or data retrieval.
    calendar=pd.DatetimeIndex(pd.to_datetime((provider/'calendars/day.txt').read_text().splitlines()))
    universe_old=set(a.index.get_level_values('instrument')); universe_new=set(b.index.get_level_values('instrument'))
    stocks=sorted(universe_new|set(ot.stock_id)|set(nt.stock_id))
    quotes={}
    replay_stats=json.loads((new/'provider_file_stats_before.json').read_text(encoding='utf-8'))
    for field in ['close','close_qfq','change','pct_chg']:
        values={}
        for stock in stocks:
            p=provider/'features'/stock.lower()/f'{field}.day.bin'
            full=np.full(len(calendar),np.nan)
            if p.exists():
                rel=str(p.relative_to(provider)); stat=p.stat()
                if rel in replay_stats and [stat.st_size,stat.st_mtime_ns]!=replay_stats[rel]:
                    raise ValueError('Quote source differs from saved replay metadata: '+rel)
                raw=np.fromfile(p,dtype='<f4'); offset=int(raw[0]);end=min(len(calendar),offset+len(raw)-1)
                full[offset:end]=raw[1:1+end-offset];hashes[str(p)]=sha(p)
            values[stock]=full
        quotes[field]=pd.DataFrame(values,index=calendar)
    rank_rows=[]; top_examples=[]
    for date in sorted(set(a.index.get_level_values('datetime'))&set(b.index.get_level_values('datetime'))):
        x=a.loc[date];y=b.loc[date];shared=x.index.intersection(y.index)
        topold=set(x.nlargest(5).index);topnew=set(y.nlargest(5).index);restricted=set(y.loc[shared].nlargest(5).index)
        rank_rows.append(dict(datetime=date,common_stocks=len(shared),score_correlation=x.loc[shared].corr(y.loc[shared]),
            rank_correlation=x.loc[shared].rank().corr(y.loc[shared].rank()),top5_overlap=len(topold&topnew),
            top5_overlap_same_universe=len(topold&restricted),new_top5_added_universe=len(topnew-universe_old),
            score_mae=float((x.loc[shared]-y.loc[shared]).abs().mean())))
    ranks=pd.DataFrame(rank_rows).set_index('datetime');ranks.to_csv(out/'daily_rank_comparison.csv')
    # Compare saved executed holdings; arithmetic ledger reconstruction only.
    oh=holdings(ot,o.index);nh=holdings(nt,n.index)
    holdrows=[]
    for date in common:
        os=set(oh.columns[oh.loc[date]>0]);ns=set(nh.columns[nh.loc[date]>0])
        if os or ns:
            holdrows.append(dict(datetime=date,old_count=len(os),new_count=len(ns),overlap=len(os&ns),jaccard=len(os&ns)/len(os|ns),old_stocks=';'.join(sorted(os)),new_stocks=';'.join(sorted(ns))))
    hold=pd.DataFrame(holdrows).set_index('datetime');hold.to_csv(out/'holding_comparison.csv')
    # Current $change is in CNY, but exchange compares it to a fractional 0.095.
    filterrows=[]; first=[]
    all_dates=n.index
    for date in all_dates[1:]:
        previous=all_dates[all_dates.get_loc(date)-1]
        scores=b.loc[previous]
        ids=scores.index
        ch=quotes['change'].loc[date,ids];pct=quotes['pct_chg'].loc[date,ids]/100
        close=quotes['close'].loc[date,ids]
        valid=ch.notna()&pct.notna()&close.notna()
        blocked=ch.abs()>=.095; pctblocked=pct.abs()>=.095
        falseblocked=blocked&~pctblocked&valid
        top=scores.nlargest(5).index
        filterrows.append(dict(datetime=date,signal_date=previous,available=int(valid.sum()),blocked_absolute=int((blocked&valid).sum()),
            blocked_percentage=int((pctblocked&valid).sum()),false_blocked=int(falseblocked.sum()),top5_false_blocked=int(falseblocked.reindex(top).sum())))
        if date==all_dates[1]:
            for stock in top:
                first.append(dict(stock_id=stock,score=float(scores[stock]),change_cny=float(ch[stock]),pct_change=float(pct[stock]),blocked=bool(blocked[stock]),executed=bool(stock in set(nt.loc[nt.datetime==date,'stock_id']))))
    filters=pd.DataFrame(filterrows).set_index('datetime');filters.to_csv(out/'tradability_semantics.csv')
    pd.DataFrame(first).to_csv(out/'first_day_top5_exclusions.csv',index=False)
    # Validate marking against saved account: no changed transactions or sizing.
    price=quotes['close'].reindex(n.index)[nh.columns].ffill()
    adj=quotes['close_qfq'].reindex(n.index)[nh.columns].ffill()
    raw_mark=(nh*price).sum(axis=1); adj_mark=(nh*adj).sum(axis=1)
    mark_gap=raw_mark-adj_mark
    gross_pnl=n['return']*n.account.shift(fill_value=300000)
    valuation=pd.DataFrame({'saved_account':n.account,'saved_cash':n.cash,'raw_mark':raw_mark,'adjusted_mark':adj_mark,'mixed_price_gap':mark_gap,
        'gap_daily_change':mark_gap.diff().fillna(mark_gap.iloc[0]),'reported_gross_pnl':gross_pnl})
    valuation['account_check_error']=raw_mark+n.cash-n.account
    if valuation.account_check_error.abs().max()>1e-4:
        raise ValueError('Executed-holding reconstruction does not reconcile')
    valuation.to_csv(out/'valuation_diagnostic.csv')
    first_trades=nt[nt.datetime==n.index[1]].copy()
    first_trades['raw_close']=[quotes['close'].loc[n.index[1],s] for s in first_trades.stock_id]
    first_trades['adjusted_close']=[quotes['close_qfq'].loc[n.index[1],s] for s in first_trades.stock_id]
    first_trades['mixed_price_pnl']=first_trades.amount*(first_trades.raw_close-first_trades.price)
    first_trades.to_csv(out/'first_day_executed_prices.csv',index=False)
    coverage_old_path=new.parent/'20261029_lgbm_optuna_to_20261015_v1/feature_coverage.csv'
    coverage_new_path=new/'feature_coverage.csv'
    hashes[str(coverage_old_path)]=sha(coverage_old_path);hashes[str(coverage_new_path)]=sha(coverage_new_path)
    co=pd.read_csv(coverage_old_path,index_col=0);cn=pd.read_csv(coverage_new_path,index_col=0)
    repaired_features=['NET_AMT','earnings_growth_smooth','quarterly_eps','roe_momentum_60d','roe_pb_ratio']
    coverage_pair=pd.DataFrame({'before_repair':co.loc['2026-05-22',repaired_features],'after_repair':cn.loc['2026-05-22',repaired_features]})
    coverage_pair.to_csv(out/'repair_coverage_old_end.csv')
    # Attributed cash from completed saved lots; incomplete positions not inferred.
    pnl=pd.concat([pd.read_parquet(p/'realized_pnl.parquet').assign(run=name) for p,name in [(old,'old'),(new,'new')]])
    pnl=pnl[pnl.close_date<=o.index.max()]
    pnl.groupby(['run','stock_id'])[['gross_pnl','trade_cost','net_pnl']].sum().to_csv(out/'closed_lot_pnl_by_stock.csv')
    # Source repair has no direct modified cells before 2026-04-01.
    monthly=aligned[['gross_log_difference','net_log_difference']].groupby(aligned.index.to_period('M')).sum();monthly.to_csv(out/'monthly_log_gap.csv')
    summaries={}
    for name,sel in [('all',slice(None)),('before_financial_repair',slice(None,'2026-03-31')),('financial_repair_overlap',slice('2026-04-01',None))]:
        r=ranks.loc[sel];h=hold.loc[sel]
        summaries[name]={'days':len(r),'mean_top5_overlap':float(r.top5_overlap.mean()),'mean_top5_overlap_same_universe':float(r.top5_overlap_same_universe.mean()),
            'mean_rank_correlation':float(r.rank_correlation.mean()),'mean_score_mae':float(r.score_mae.mean()),'mean_holding_overlap':float(h.overlap.mean()),
            'identical_holding_days':int((h.jaccard==1).sum()),'holding_days':len(h),'days_new_top5_has_added_stock':int((r.new_top5_added_universe>0).sum())}
    nf=filters.loc[:o.index.max()]
    fees=[]
    for name,t in [('old',ot),('new_common_period',nt[nt.datetime<=o.index.max()]),('new_extension',nt[nt.datetime>o.index.max()])]:
        for side,g in t.groupby('direction'):
            fees.append({'run':name,'side':side,'trades':len(g),'value':float(g.value.sum()),'cash_cost':float(g.cost.sum()),'effective_rate':float(g.cost.sum()/g.value.sum())})
    pd.DataFrame(fees).to_csv(out/'trade_cost_rates.csv',index=False)
    extra=universe_new-universe_old
    summary={'training_runs':0,'prediction_runs':0,'backtest_runs':0,'source_model_identical':hashes[str(old/'model.pkl')]==hashes[str(new/'frozen_model.pkl')],
        'dates':{'common_days':len(common),'old_days':len(o),'new_days':len(n),'internal_extra_dates':[str(x.date()) for x in internal],'extension_days':len(extension),'extension_start':str(extension.min().date()),'end':str(n.index.max().date())},
        'window_metrics':window.reset_index().to_dict('records'),'return_bridge':bridge.to_dict('records'),'common_date_log_decomposition':logparts,
        'universe':{'old_observed':len(universe_old),'new_observed':len(universe_new),'added':len(extra),'removed':len(universe_old-universe_new),
            'added_stocks':sorted(extra),'added_stocks_traded_common_period':sorted(set(nt.loc[nt.datetime<=o.index.max(),'stock_id'])&extra)},
        'rank_and_holding_comparison':summaries,
        'tradability_common_period':{'candidate_sessions':int(nf.available.sum()),'absolute_change_blocked':int(nf.blocked_absolute.sum()),'fractional_change_blocked':int(nf.blocked_percentage.sum()),
            'false_blocked':int(nf.false_blocked.sum()),'false_block_rate':float(nf.false_blocked.sum()/nf.available.sum()),'mean_top5_false_blocked':float(nf.top5_false_blocked.mean())},
        'first_day_top5':first,'valuation':{'first_trade_day':str(n.index[1].date()),'first_trade_day_mixed_gap':float(mark_gap.iloc[1]),'max_mixed_gap':float(mark_gap.max()),
            'max_mixed_gap_date':str(mark_gap.idxmax().date()),'terminal_gap':float(mark_gap.iloc[-1]),'max_account_check_error':float(valuation.account_check_error.abs().max())},
        'fee_rates':fees,'financial_repair_start':'2026-04-01','moneyflow_repair_start':'2026-07-10',
        'repaired_feature_coverage_before_april_max_difference':float((co.loc[:'2026-03-31',repaired_features]-cn.loc[:'2026-03-31',repaired_features]).abs().max().max()),
        'repair_coverage_old_end':coverage_pair.to_dict('index'),
        'limitations':['Exact same-date compound products omit new 2026-05-08 only for comparison, not a counterfactual strategy.',
            'No saved pre-repair prediction/replay exists. Repair and decision-path causal return contributions are not separately identified.',
            'Universe/rank/holding changes and price/tradability semantics overlap; their effects cannot be added as independent causes.',
            'Mixed-mark gap is an accounting diagnostic on fixed executed quantities, not a corrected backtest; sizing and orders could change after a correction.']}
    if any(sha(p)!=h for p,h in hashes.items()):raise ValueError('Source files changed')
    (out/'source_sha256.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    (out/'diagnosis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')
    (out/'verification.json').write_text(json.dumps({'sources_unchanged':True,'source_files_checked':len(hashes),'bridge_reconciles':True,'log_decomposition_reconciles':True,'saved_holdings_account_reconciles':True,'training_runs':0,'prediction_runs':0,'backtest_runs':0},indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,default=str),flush=True)


if __name__=='__main__':main()
