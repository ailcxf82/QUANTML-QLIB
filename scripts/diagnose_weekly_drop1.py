"""Read-only attribution of saved weekly-drop1 results; no engine/model imports."""
from pathlib import Path
import hashlib
import json
from collections import defaultdict
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'artifacts/weekly_strategy/20261029_all_boards_slip5bp_v1/weekly_drop1'
OUT = ROOT / 'artifacts/weekly_diagnostics/20261029_drop1_yearly_attribution_v1'

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True, exist_ok=False)
    files = ['backtest_report.parquet', 'trades.parquet', 'holdings_reconstructed.parquet',
             'source_execution_quotes.parquet', 'realized_pnl.parquet', 'signals.parquet']
    before = {f: digest(SOURCE/f) for f in files}
    r = pd.read_parquet(SOURCE/files[0]); t = pd.read_parquet(SOURCE/files[1])
    h = pd.read_parquet(SOURCE/files[2]); q = pd.read_parquet(SOURCE/files[3])
    lots = pd.read_parquet(SOURCE/files[4])
    h.index = pd.to_datetime(h.index)
    dates, codes = r.index, h.columns
    p = q['$close_qfq'].unstack('instrument').reindex(index=dates, columns=codes).ffill()
    assert not ((h.abs() > 1e-8) & p.isna()).any().any()
    mv = (h * p).fillna(0)
    def pivot(col, frame=t):
        return frame.pivot_table(index='datetime', columns='stock_id', values=col, aggfunc='sum').reindex(index=dates, columns=codes).fillna(0)
    buys = pivot('value', t[t.direction=='BUY']); sells = pivot('value', t[t.direction=='SELL'])
    costs = {k: pivot(k) for k in ['cost','commission','stamp_tax','transfer_fee','slippage']}
    gross = mv.diff().fillna(mv.iloc[0]) + sells - buys
    net = gross - costs['cost']
    start_nav = r.account.shift().fillna(300000)
    pnl_error = net.sum(axis=1) - (r.account-start_nav)
    fee_error = costs['cost'].sum(axis=1) - r.cost*start_nav
    value_error = mv.sum(axis=1)-r.value
    np.testing.assert_allclose(pnl_error, 0, atol=1e-6)
    np.testing.assert_allclose(fee_error, 0, atol=1e-6)
    np.testing.assert_allclose(value_error, 0, atol=1e-6)
    np.testing.assert_allclose(sum(costs[k] for k in ['commission','stamp_tax','transfer_fee','slippage']), costs['cost'], atol=1e-7)
    # Continuous holding episodes, including sale day; assert no additional buying while held.
    episodes, daily = [], []
    for code in codes:
        ep = None
        for i, day in enumerate(dates):
            was_held = i>0 and h.iloc[i-1][code]>1e-8
            held = h.loc[day,code]>1e-8
            if buys.loc[day,code]>0:
                assert not was_held, (code, day, 'additional buy requires lot-level attribution')
                assert ep is None
                ep = {'episode_id':f'{code}_{day:%Y%m%d}', 'stock_id':code, 'open_date':day,
                      'close_date':pd.NaT, 'closed':False, 'entry_value':buys.loc[day,code],
                      'net_pnl':0., 'gross_pnl':0., 'cost':0., 'net_2025':0., 'net_2026':0.}
            if ep is None:
                assert abs(net.loc[day,code])<1e-6
                continue
            age = (day-ep['open_date']).days
            age_bin = '00-07' if age<=7 else '08-28' if age<=28 else '29-56' if age<=56 else '57+'
            row = {'datetime':day,'year':day.year,'stock_id':code,'episode_id':ep['episode_id'],
                   'age_calendar_days':age,'age_bin':age_bin,'gross_pnl':gross.loc[day,code],
                   'net_pnl':net.loc[day,code], 'market_value':mv.loc[day,code],
                   'previous_market_value':mv.iloc[i-1][code] if i else 0.,
                   'buy_value':buys.loc[day,code],'sell_value':sells.loc[day,code]}
            for k,v in costs.items(): row[k]=v.loc[day,code]
            daily.append(row)
            for k in ['net_pnl','gross_pnl','cost']: ep[k]+=row[k]
            ep[f'net_{day.year}']+=row['net_pnl']
            ep['last_date']=day; ep['holding_calendar_days']=age
            if was_held and not held:
                ep['close_date']=day; ep['closed']=True
                episodes.append(ep); ep=None
        if ep is not None: episodes.append(ep)
    d=pd.DataFrame(daily); e=pd.DataFrame(episodes)
    closed=e[e.closed]
    np.testing.assert_allclose(closed.net_pnl.sum(),lots.net_pnl.sum(),atol=1e-6)
    assert len(closed)==len(lots)
    np.testing.assert_allclose(e.net_pnl.sum(),r.account.iloc[-1]-300000,atol=1e-6)
    periods={'2025':('2025-01-02','2025-12-31'), '2026_to_1015':('2026-01-01','2026-10-15'),
             '2025_to_1015':('2025-01-02','2025-10-15')}
    summaries=[]; stock_tables=[]; concentration=[]
    for label,(lo,hi) in periods.items():
        ix=(dates>=lo)&(dates<=hi); dr=d[d.datetime.between(lo,hi)]
        cap=float(start_nav.loc[ix].iloc[0]); end=float(r.account.loc[ix].iloc[-1])
        tr=t[t.datetime.between(lo,hi)]
        stock=dr.groupby('stock_id')[['net_pnl','gross_pnl','cost','commission','stamp_tax','transfer_fee','slippage','buy_value','sell_value']].sum().sort_values('net_pnl',ascending=False)
        stock['contribution_return']=stock.net_pnl/cap
        stock['held_sessions']=dr[dr.market_value>1e-6].groupby('stock_id').size().reindex(stock.index,fill_value=0)
        stock['mean_weight_all_sessions']=(mv.loc[ix].div(r.account.loc[ix],axis=0)).mean().reindex(stock.index)
        stock['max_weight']=(mv.loc[ix].div(r.account.loc[ix],axis=0)).max().reindex(stock.index)
        stock['period']=label; stock_tables.append(stock.reset_index())
        wins=stock[stock.net_pnl>0].net_pnl; losses=stock[stock.net_pnl<0].net_pnl
        sold=closed[closed.close_date.between(lo,hi)]
        s={'period':label,'start_account':cap,'end_account':end,'sessions':int(ix.sum()),
           'net_pnl':end-cap,'net_return':end/cap-1,'gross_pnl_same_path':float(stock.gross_pnl.sum()),
           'gross_return_cash_addback':float(stock.gross_pnl.sum()/cap),
           'fees':float(stock.cost.sum()),'fee_return_drag':float(stock.cost.sum()/cap),
           'positive_pnl':float(wins.sum()),'negative_pnl':float(losses.sum()),
           'winning_stocks':len(wins),'losing_stocks':len(losses),'stocks':len(stock),
           'orders':len(tr),'trade_days':tr.datetime.nunique(),'traded_value':float(tr.value.sum()),
           'closed_episodes':len(sold),'closed_win_rate':float((sold.net_pnl>0).mean()),
           'closed_median_days':float(sold.holding_calendar_days.median()),
           'closed_mean_days':float(sold.holding_calendar_days.mean()),
           'closed_winner_mean_days':float(sold.loc[sold.net_pnl>0,'holding_calendar_days'].mean()),
           'closed_loser_mean_days':float(sold.loc[sold.net_pnl<0,'holding_calendar_days'].mean()),
           'closed_winner_mean_pnl':float(sold.loc[sold.net_pnl>0,'net_pnl'].mean()),
           'closed_loser_mean_pnl':float(sold.loc[sold.net_pnl<0,'net_pnl'].mean())}
        for k in ['commission','stamp_tax','transfer_fee','slippage']: s[k]=float(stock[k].sum())
        for n in [1,3,5,10]:
            top=float(stock.head(n).net_pnl.sum())
            concentration.append({'period':label,'top_n':n,'top_net_pnl':top,
                 'share_of_net_pnl':top/s['net_pnl'],'share_of_positive_pnl':top/wins.sum(),
                 'contribution_return':top/cap,'residual_pnl_accounting_only':s['net_pnl']-top,
                 'residual_return_accounting_only':(s['net_pnl']-top)/cap})
        summaries.append(s)
    stocks=pd.concat(stock_tables,ignore_index=True)
    by_age=d.groupby(['year','age_bin'])[['gross_pnl','net_pnl','cost','previous_market_value']].sum().reset_index()
    by_age['pnl_per_1000_exposure_session']=by_age.gross_pnl/by_age.previous_market_value*1000
    month=d.assign(month=d.datetime.dt.strftime('%Y-%m')).groupby('month')[['net_pnl','gross_pnl','cost']].sum()
    month['start_account']=start_nav.groupby(dates.strftime('%Y-%m')).first()
    month['net_return']=month.net_pnl/month.start_account
    boundary=dates[dates.year==2025][-1]
    crosscodes=h.loc[boundary][h.loc[boundary]>1e-8].index
    cross=stocks[stocks.stock_id.isin(crosscodes)&stocks.period.isin(['2025','2026_to_1015'])].copy()
    cross['boundary_value']=cross.stock_id.map(mv.loc[boundary])
    carry=e[(e.open_date<=boundary)&(e.last_date>boundary)].copy()
    tables={'period_summary':pd.DataFrame(summaries),'stock_year_attribution':stocks,
            'concentration':pd.DataFrame(concentration),'holding_episodes':e,'daily_stock_attribution':d,
            'holding_age_attribution':by_age,'monthly_attribution':month.reset_index(),
            'cross_year_stocks':cross,'cross_year_episodes':carry}
    for name,frame in tables.items(): frame.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    after={f:digest(SOURCE/f) for f in files}; assert before==after
    verification={'training_runs':0,'prediction_runs':0,'backtest_runs':0,'source':str(SOURCE),
        'source_sha256':before,'sources_unchanged':True,'max_daily_pnl_error':float(pnl_error.abs().max()),
        'max_daily_cost_error':float(fee_error.abs().max()),'max_daily_value_error':float(value_error.abs().max()),
        'episodes':len(e),'closed_episodes':len(closed),'open_episodes':int((~e.closed).sum()),
        'method':'Daily per-stock PnL = delta end market value + sells - buys - actual cash costs. Annual contributions divide by annual starting NAV. Cross-year lots marked at year-end. Cost addback and top-N removal are fixed-path accounting, not counterfactual backtests.'}
    (OUT/'verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    print(pd.DataFrame(summaries).to_string(index=False))
    print('\nTOP STOCKS', stocks.groupby('period',sort=False).head(6).to_string(index=False))
    print('\nCONCENTRATION',tables['concentration'].to_string(index=False))
    print('\nAGE',by_age.to_string(index=False))
    print('\nCARRY',carry.to_string(index=False))
    print('\nMONTHS',month.to_string())
    print('\nTOP EPISODES',e.nlargest(8,'net_pnl').to_string(index=False))

if __name__=='__main__': main()
