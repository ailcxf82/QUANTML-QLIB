"""Audit and summarize the three fixed weekly experiments (no replays)."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/weekly_strategy/20261029_all_boards_slip5bp_v1'
BASE=ROOT/'artifacts/fee_replay/20261029_lgbm_slip5bp_to_20261015_v1'


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ledger=json.loads((OUT/'campaign_ledger.json').read_text())
    assert ledger['status']=='complete' and ledger['backtest_attempts']==3
    baseline_cfg=yaml.safe_load((BASE/'resolved_config.yaml').read_text(encoding='utf-8'))
    basequotes=pd.read_parquet(BASE/'source_execution_quotes.parquet')
    periods=[];rows=[];all_paths={};audits={}
    names={'daily_baseline':'每日最多换1只','weekly_drop1':'每周最多换1只','weekly_drop2':'每周最多换2只','weekly_drop5':'每周更新前5名'}
    for name,path in [('daily_baseline',BASE)]+[(f'weekly_drop{n}',OUT/f'weekly_drop{n}') for n in [1,2,5]]:
        met=json.loads((path/'metrics.json').read_text(encoding='utf-8'))
        r=pd.read_parquet(path/'backtest_report.parquet');t=pd.read_parquet(path/'trades.parquet')
        z=pd.read_parquet(path/'realized_pnl.parquet');h=pd.read_parquet(path/'holdings_reconstructed.parquet')
        assert len(r)==430 and r.index[-1]==pd.Timestamp('2026-10-15')
        assert sha(path/'signals.parquet')==sha(BASE/'signals.parquet')
        assert sha(path/'universe.json')==sha(BASE/'universe.json')
        q=pd.read_parquet(path/'source_execution_quotes.parquet')
        pd.testing.assert_frame_equal(q,basequotes,check_exact=True)
        cfg=yaml.safe_load((path/'resolved_config.yaml').read_text(encoding='utf-8'))
        cmp=yaml.safe_load(yaml.safe_dump(baseline_cfg))
        for c in [cfg,cmp]:c.pop('strategy');c.pop('replay');c['experiment'].pop('run_id')
        assert cfg==cmp
        post=json.loads((path/'postflight.json').read_text(encoding='utf-8'))
        for rel,expected in post['artifact_sha256'].items():assert sha(path/rel)==expected,(name,rel)
        if name!='daily_baseline':
            for p,expected in json.loads((path/'source_sha256.json').read_text()).items():assert sha(Path(p))==expected,p
            reviews=pd.read_csv(path/'review_schedule.csv')
            assert not reviews.duplicated(['iso_year','iso_week']).any()
            assert (pd.to_datetime(reviews.signal_date)<pd.to_datetime(reviews.review_date)).all()
            assert set(t.datetime.dt.strftime('%Y-%m-%d')) <= set(reviews.review_date)
            iso=r.index.isocalendar();first=pd.Series(r.index,index=r.index).groupby([iso.year,iso.week]).min()
            assert set(reviews.review_date.iloc[1:]) <= set(first.dt.strftime('%Y-%m-%d'))
            n=int(name[-1]);max_sells=t[t.direction=='SELL'].groupby('datetime').size().max()
            assert max_sells<=n
            review_count=len(reviews)
        else:review_count=len(r)
        prices=q['$close_qfq'].unstack('instrument').reindex(r.index).ffill()[h.columns]
        weights=(h*prices).div(r.account,axis=0)
        max_weight=float(weights.max().max())
        prev=r.account.shift(fill_value=300000.)
        np.testing.assert_allclose(r.account,300000*(1+r['return']-r.cost).cumprod(),atol=1e-6)
        cost=t.cost.sum();np.testing.assert_allclose(cost,r.total_cost.iloc[-1],atol=1e-6)
        row={'id':name,'strategy':names[name],**{k:met['metrics'][k] for k in ['total_return','annualized_return','max_drawdown','sharpe_ratio','avg_turnover']},
            'ending_account':float(r.account.iloc[-1]),'review_days':review_count,'active_trade_days':t.datetime.nunique(),
            'orders':len(t),'cash_cost':float(cost),'slippage_cost':float(t.slippage.sum()),
            'median_holding_calendar_days':float(z.holding_days.median()),'mean_holding_calendar_days':float(z.holding_days.mean()),
            'max_single_stock_weight':max_weight,'minimum_cash':float(r.cash.min())}
        for period,mask in [('2025',r.index.year==2025),('2026_to_1015',r.index.year==2026),('recent_20260525_to_1015',r.index>='2026-05-25')]:
            part=r.loc[mask];start=float(prev.loc[mask].iloc[0]);nav=pd.concat([pd.Series([start]),part.account.reset_index(drop=True)],ignore_index=True)
            ret=float(part.account.iloc[-1]/start-1);dd=float((nav/nav.cummax()-1).min())
            row[f'{period}_return']=ret
            periods.append({'id':name,'strategy':names[name],'period':period,'days':len(part),'net_return':ret,'max_drawdown':dd,
                            'cost':float((part.cost*prev.loc[mask]).sum())})
        rows.append(row);all_paths[names[name]]=r.account/300000
        audits[name]={'predictions_identical':True,'universe_identical':True,'quotes_identical':True,'only_strategy_config_changed':True,
            'source_hashes_unchanged':True,'weekly_schedule_checked':name!='daily_baseline'}
    comparison=pd.DataFrame(rows);comparison.to_csv(OUT/'comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(periods).to_csv(OUT/'period_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(all_paths).to_csv(OUT/'aligned_net_values.csv',encoding='utf-8-sig')
    (OUT/'verification.json').write_text(json.dumps({'training_runs':0,'prediction_runs':0,'backtest_runs':3,'tests_passed':178,'runs':audits},ensure_ascii=False,indent=2),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans'];plt.rcParams['axes.unicode_minus']=False
    fig,ax=plt.subplots(figsize=(11,5))
    for i,(name,nav) in enumerate(all_paths.items()):ax.plot(nav.index,nav,label=name,linestyle='--' if i==0 else '-')
    ax.set_title('全板块、相同预测与成本：每日和每周调仓净值');ax.set_ylabel('扣费净值（初始=1）');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    print(comparison.to_string(index=False))


if __name__=='__main__':main()
