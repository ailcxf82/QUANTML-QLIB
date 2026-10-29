"""Saved-score event study and decision attribution. No fit, inference or replay."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'artifacts/weekly_strategy/20261029_all_boards_slip5bp_v1/weekly_drop1'
PRIOR=ROOT/'artifacts/weekly_diagnostics/20261029_drop1_yearly_attribution_v1'
OUT=ROOT/'artifacts/weekly_diagnostics/20261029_drop1_exit_signals_v1'

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if OUT.exists() and '--refresh-diagnostic' not in sys.argv:
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True,exist_ok=True)
    protocol={'scope':'Saved predictions, positions and quotes only; no new training, inference or backtest.',
      'rules':{'rank20':'Previous-session pool rank > 20 on a scheduled weekly review',
               'rank20_twice':'Pool rank > 20 on two consecutive scheduled reviews while already held',
               'rank20_bottom2':'Pool rank > 20 and among the bottom two of existing holdings'},
      'evaluation':'First tradable scheduled review strictly before actual exit or censoring date. Future price movement is an event-study outcome, not a portfolio counterfactual. No threshold optimization.',
      'price_contract':'Saved QFQ close and pct_chg; reproduce existing 9.5% both-direction blocking for attribution only.',
      'pool':'Rank among all nonmissing saved scores, not a broad-market rank.'}
    protocol['timing_probe']='Same rank > 20 threshold checked daily, to separate weekly sampling from one-sale quota; descriptive only.'
    (OUT/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
    paths=[SRC/f for f in ['signals.parquet','labels.parquet','holdings_reconstructed.parquet','trades.parquet',
        'source_execution_quotes.parquet','review_schedule.csv','backtest_report.parquet','resolved_config.yaml',
        'weekly_topk_source.py']]+[PRIOR/'holding_episodes.csv',PRIOR/'daily_stock_attribution.csv']
    before={str(p):sha(p) for p in paths}
    sig=pd.read_parquet(SRC/'signals.parquet').score.unstack('instrument').sort_index()
    ranks=sig.rank(axis=1,ascending=False,method='min'); sizes=sig.notna().sum(axis=1)
    h=pd.read_parquet(SRC/'holdings_reconstructed.parquet');dates=pd.DatetimeIndex(h.index)
    t=pd.read_parquet(SRC/'trades.parquet')
    q=pd.read_parquet(SRC/'source_execution_quotes.parquet')
    raw=q['$close_qfq'].unstack('instrument').reindex(index=dates,columns=sig.columns)
    prices=raw.ffill();change=q['$pct_chg'].unstack('instrument').reindex_like(raw)/100
    eligible=raw.notna()&(raw>0)&change.notna()&(change.abs()<.095)
    schedule=pd.read_csv(SRC/'review_schedule.csv',parse_dates=['review_date','signal_date'])
    reviews=dict(zip(schedule.review_date,schedule.signal_date))
    ep=pd.read_csv(PRIOR/'holding_episodes.csv',parse_dates=['open_date','close_date','last_date'])
    dp=pd.read_csv(PRIOR/'daily_stock_attribution.csv',parse_dates=['datetime'])
    bydate={day:g for day,g in t.groupby('datetime')}; review_context={}; mismatch=[]
    for _,v in schedule.iterrows():
        day,sd=v.review_date,v.signal_date
        assert sd<day and sd==dates[dates.get_loc(day)-1]
        prev=h.loc[sd]; held=prev[prev>1e-8].index.tolist()
        score=sig.loc[sd]; heldscore=score.reindex(held); heldrank=heldscore.rank(ascending=False,method='min')
        candidates=score.drop(index=held,errors='ignore').dropna().sort_values(ascending=False)
        candidates=candidates[eligible.loc[day].reindex(candidates.index).fillna(False)]
        challenger=candidates.index[0] if len(candidates) else None
        comb=heldscore.copy()
        if challenger is not None: comb.loc[challenger]=score[challenger]
        comb=comb[eligible.loc[day].reindex(comb.index).fillna(False)].sort_values(ascending=False)
        worst=comb.index[-1] if len(comb) else None
        expected=worst if worst in held else None
        sold=bydate[day].loc[bydate[day].direction=='SELL','stock_id'].tolist() if day in bydate else []
        if ([expected] if expected else [])!=sold: mismatch.append({'day':str(day),'expected':expected,'actual':sold})
        review_context[day]={'held':held,'held_rank':heldrank,'challenger':challenger,'expected':expected,'sold':sold}
    assert not mismatch,mismatch
    rows=[]
    for _,a in ep.iterrows():
        code=a.stock_id; start=dates.get_loc(a.open_date); end=dates.get_loc(a.last_date)
        buy=t[(t.datetime==a.open_date)&(t.stock_id==code)&(t.direction=='BUY')].iloc[0]
        for i in range(start,end+1):
            day=dates[i]; sd=dates[i-1];score=sig.loc[sd,code];rank=ranks.loc[sd,code]
            review=day in reviews; existing=day>a.open_date
            ctx=review_context.get(day,{})
            heldrank=ctx.get('held_rank',pd.Series(dtype=float)).get(code,np.nan)
            pool_n=int(sizes.loc[sd]);sold=ctx.get('sold',[])
            row={'episode_id':a.episode_id,'stock_id':code,'open_date':a.open_date,'end_date':a.last_date,
              'entry_year':a.open_date.year,'datetime':day,'signal_date':sd,'score':score,'pool_rank':rank,
              'pool_size':pool_n,'is_review':review,'existing_holding':existing,'held_rank':heldrank,
              'held_count':len(ctx.get('held',[])),'tradable_existing_contract':bool(eligible.loc[day,code]),
              'actual_sold':code in sold,'sold_other':','.join(x for x in sold if x!=code),
              'challenger':ctx.get('challenger'),'price':prices.loc[day,code],
              'gross_return_since_entry':prices.loc[day,code]/buy.price-1,
              'remaining_gross_return':prices.loc[a.last_date,code]/prices.loc[day,code]-1,
              'remaining_gross_pnl':float(buy.amount*(prices.loc[a.last_date,code]-prices.loc[day,code])),
              'calendar_days_remaining':(a.last_date-day).days,'actual_episode_net_pnl':a.net_pnl,
              'closed':a.closed,'quantity':buy.amount}
            if review and existing:
                if code in sold: reason='actual_exit'
                elif not row['tradable_existing_contract']:reason='untradable_at_review'
                elif pd.isna(rank):reason='missing_score'
                elif rank>20 and sold:reason='weak_rank_waiting_behind_other_sale'
                elif rank>20:reason='weak_rank_no_sale_other_reason'
                else:reason='rank_still_top20'
                row['retention_reason']=reason
            else:row['retention_reason']='entry_or_nonreview'
            rows.append(row)
    traj=pd.DataFrame(rows)
    rev=traj[traj.is_review&traj.existing_holding].copy()
    rev['rank20']=rev.pool_rank>20
    rev['rank20_twice']=rev.rank20 & rev.groupby('episode_id').rank20.shift(fill_value=False)
    rev['rank20_bottom2']=rev.rank20 & (rev.held_rank>=rev.held_count-1)
    alerts=[]
    for (eid,group) in rev.groupby('episode_id',sort=False):
        for rule in protocol['rules']:
            fired=group[group[rule]&group.tradable_existing_contract&(group.datetime<group.end_date)]
            if len(fired):
                a=fired.iloc[0].to_dict();a['rule']=rule;alerts.append(a)
    alerts=pd.DataFrame(alerts)
    dailyalerts=traj[traj.existing_holding & (traj.pool_rank>20) & traj.tradable_existing_contract & (traj.datetime<traj.end_date)].groupby('episode_id',sort=False).head(1).copy()
    dailyalerts['rule']='rank20_daily_timing_probe'
    allalerts=pd.concat([alerts,dailyalerts],ignore_index=True)
    alertsum=[]
    for year in [2025,2026]:
        cohort=ep[ep.open_date.dt.year==year]
        for rule in [*protocol['rules'],'rank20_daily_timing_probe']:
            a=allalerts[(allalerts.entry_year==year)&(allalerts.rule==rule)]
            for state in ['all_marked','closed_only','closed_same_year']:
                co=cohort if state=='all_marked' else cohort[cohort.closed]
                if state=='closed_same_year':co=co[co.close_date.dt.year==year]
                z=a[a.episode_id.isin(co.episode_id)]
                losers=co[co.net_pnl<0];winners=co[co.net_pnl>=0]
                alertsum.append({'entry_year':year,'rule':rule,'state':state,'cohort_n':len(co),'alerted_n':len(z),
                  'losing_episodes':len(losers),'losers_alerted':int(z.episode_id.isin(losers.episode_id).sum()),
                  'winning_episodes':len(winners),'winners_alerted':int(z.episode_id.isin(winners.episode_id).sum()),
                  'subsequent_fall_n':int((z.remaining_gross_return<0).sum()),
                  'subsequent_rise_n':int((z.remaining_gross_return>0).sum()),
                  'mean_subsequent_return':z.remaining_gross_return.mean(),'median_subsequent_return':z.remaining_gross_return.median(),
                  'subsequent_negative_gross_pnl':z.loc[z.remaining_gross_pnl<0,'remaining_gross_pnl'].sum(),
                  'subsequent_positive_gross_pnl':z.loc[z.remaining_gross_pnl>0,'remaining_gross_pnl'].sum(),
                  'median_days_before_end':z.calendar_days_remaining.median()})
    # Model ranking quality at the trained horizon, and entry outcomes at fixed horizons.
    labels=pd.read_parquet(SRC/'labels.parquet').label.unstack('instrument').reindex_like(sig)
    forward_label=raw.shift(-2)/raw.shift(-1)-1
    valid=labels.notna()&forward_label.notna()
    label_diff=float((labels-forward_label).where(valid).abs().max().max())
    rankic=sig.corrwith(forward_label,axis=1,method='spearman')
    icdaily=pd.DataFrame({'signal_date':sig.index,'rank_ic':rankic.values,
        'top5_label':forward_label.where(ranks<=5).mean(axis=1).values,
        'pool_equal_weight_label':forward_label.mean(axis=1).values,
        'valid_label_n':forward_label.notna().sum(axis=1).values})
    icdaily['execution_date']=pd.Series(dates,index=dates).shift(-1).values
    icdaily['outcome_date']=pd.Series(dates,index=dates).shift(-2).values
    icdaily['top5_minus_pool']=icdaily.top5_label-icdaily.pool_equal_weight_label
    icdaily['is_used_on_weekly_review']=icdaily.signal_date.isin(schedule.signal_date)
    quality=[]
    for year in [2025,2026]:
        for window in ['full_available','through_1015']:
            z=icdaily[icdaily.execution_date.dt.year==year].dropna(subset=['rank_ic'])
            if window=='through_1015':z=z[z.outcome_date<=pd.Timestamp(f'{year}-10-15')]
            for freq in ['daily','weekly_review']:
                x=z if freq=='daily' else z[z.is_used_on_weekly_review]
                quality.append({'year':year,'window':window,'signal_sample':freq,'n':len(x),
                    'mean_rank_ic':x.rank_ic.mean(),'positive_rank_ic_fraction':(x.rank_ic>0).mean(),
                    'top5_mean_1d_return':x.top5_label.mean(),'pool_mean_1d_return':x.pool_equal_weight_label.mean(),
                    'top5_minus_pool':x.top5_minus_pool.mean()})
    entries=[]
    for _,a in ep.iterrows():
        i=dates.get_loc(a.open_date);sd=dates[i-1];code=a.stock_id
        row={'episode_id':a.episode_id,'stock_id':code,'open_date':a.open_date,'entry_year':a.open_date.year,
          'entry_rank':ranks.loc[sd,code],'score':sig.loc[sd,code],'closed':a.closed,'actual_net_pnl':a.net_pnl}
        for n in [1,5,20]:
            if i+n<len(dates):
                row[f'forward_{n}d_end_date']=dates[i+n]
                fwd=raw.iloc[i+n]/raw.iloc[i]-1
                row[f'forward_{n}d_return']=fwd[code]
                row[f'forward_{n}d_pool_return']=fwd.mean()
                row[f'forward_{n}d_excess']=fwd[code]-fwd.mean()
            else:
                row[f'forward_{n}d_end_date']=pd.NaT
                for suffix in ['return','pool_return','excess']:row[f'forward_{n}d_{suffix}']=np.nan
        entries.append(row)
    entries=pd.DataFrame(entries);entry_stats=[]
    for year in [2025,2026]:
        for window in ['full_available','through_1015']:
            z=entries[entries.entry_year==year]
            if window=='through_1015': z=z[z.open_date<=pd.Timestamp(f'{year}-10-15')]
            for n in [1,5,20]:
                x=z.dropna(subset=[f'forward_{n}d_return'])
                if window=='through_1015':x=x[x[f'forward_{n}d_end_date']<=pd.Timestamp(f'{year}-10-15')]
                entry_stats.append({'year':year,'window':window,'horizon':n,'n':len(x),
                   'mean_return':x[f'forward_{n}d_return'].mean(),'median_return':x[f'forward_{n}d_return'].median(),
                   'positive_fraction':(x[f'forward_{n}d_return']>0).mean(),
                   'mean_pool_return':x[f'forward_{n}d_pool_return'].mean(),'mean_excess':x[f'forward_{n}d_excess'].mean()})
    # Attribute realized observed movements to the latest review state, no altered holdings.
    intervals=[]
    for eid,g in traj.groupby('episode_id',sort=False):
        current='entry_before_first_review'
        dr=dp[dp.episode_id==eid].set_index('datetime')
        for _,a in g.iterrows():
            # Close-to-close movement ending today belongs to yesterday's review state.
            if a.datetime==a.open_date:
                intervals.append({'episode_id':eid,'entry_year':a.entry_year,'datetime':a.datetime,
                    'review_state':'entry_day_fees','gross_pnl':dr.loc[a.datetime,'gross_pnl'],
                    'cost':dr.loc[a.datetime,'cost'],'net_pnl':dr.loc[a.datetime,'net_pnl']})
            else:
                intervals.append({'episode_id':eid,'entry_year':a.entry_year,'datetime':a.datetime,
                    'review_state':current,'gross_pnl':dr.loc[a.datetime,'gross_pnl'],
                    'cost':dr.loc[a.datetime,'cost'],'net_pnl':dr.loc[a.datetime,'net_pnl']})
            if a.is_review and a.existing_holding: current=a.retention_reason
    intervals=pd.DataFrame(intervals)
    state_summary=intervals.groupby(['entry_year','review_state'])[['gross_pnl','cost','net_pnl']].sum().reset_index()
    reasons=rev.groupby(['entry_year','retention_reason']).size().reset_index(name='held_review_observations')
    episode_summary=[]
    for _,a in ep.iterrows():
        z=rev[rev.episode_id==a.episode_id];g=traj[traj.episode_id==a.episode_id];early=z[z.datetime<z.end_date]
        w=early[early.rank20 & early.tradable_existing_contract]
        row=a.to_dict();row.update({'entry_rank':g.iloc[0].pool_rank,'held_weekly_reviews':len(z),
          'top20_held_reviews':int((z.pool_rank<=20).sum()),'quota_wait_reviews':int((early.retention_reason=='weak_rank_waiting_behind_other_sale').sum()),
          'first_rank20_alert':w.iloc[0].datetime if len(w) else pd.NaT,
          'first_alert_rank':w.iloc[0].pool_rank if len(w) else np.nan,
          'first_alert_held_rank':w.iloc[0].held_rank if len(w) else np.nan,
          'return_at_first_alert':w.iloc[0].gross_return_since_entry if len(w) else np.nan,
          'return_after_first_alert':w.iloc[0].remaining_gross_return if len(w) else np.nan,
          'gross_pnl_after_first_alert':w.iloc[0].remaining_gross_pnl if len(w) else np.nan})
        episode_summary.append(row)
    tables={'daily_rank_trajectory':traj,'weekly_rank_reviews':rev,'first_exit_alerts':allalerts,
       'alert_summary':pd.DataFrame(alertsum),'daily_model_quality':icdaily,'model_quality_summary':pd.DataFrame(quality),
       'entry_forward_returns':entries,'entry_forward_summary':pd.DataFrame(entry_stats),
       'review_state_intervals':intervals,'review_state_pnl':state_summary,'review_reason_counts':reasons,
       'episode_rank_summary':pd.DataFrame(episode_summary)}
    for name,df in tables.items():df.to_csv(OUT/f'{name}.csv',index=False,encoding='utf-8-sig')
    assert all(sha(Path(p))==v for p,v in before.items())
    np.testing.assert_allclose(ep.net_pnl.sum(),dp.net_pnl.sum(),atol=1e-6)
    np.testing.assert_allclose(intervals.net_pnl.sum(),ep.net_pnl.sum(),atol=1e-6)
    assert label_diff<1e-5,label_diff
    v={'training_runs':0,'prediction_runs':0,'backtest_runs':0,'sources_unchanged':True,'source_sha256':before,
       'review_decision_attribution_mismatches':mismatch,'reviews_checked':len(schedule),
       'saved_label_vs_saved_price_max_difference':label_diff,'pool_score_count_min':int(sizes.min()),'pool_score_count_max':int(sizes.max()),
       'rules_count':len(protocol['rules']),'not_a_counterfactual_backtest':True}
    (OUT/'verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'analysis_source.py').write_bytes(Path(__file__).read_bytes())
    for name in ['model_quality_summary','entry_forward_summary','alert_summary','review_reason_counts','review_state_pnl']:
        print('\n'+name+'\n'+tables[name].to_string(index=False))
    a=tables['episode_rank_summary'];print('\nLOSERS2026\n'+a[a.open_date.dt.year==2026].nsmallest(12,'net_pnl').to_string(index=False))

if __name__=='__main__':main()
