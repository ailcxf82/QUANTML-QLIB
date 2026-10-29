"""Replay immutable saved signals after correcting execution quote semantics.

No model loading, fitting, feature regeneration, or prediction. Write a fresh
directory and compare identical dates to the supplied saved replay.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def save(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str,allow_nan=False),encoding='utf-8')


def provider_stats(provider):
    stats={}
    for folder in (provider/'features').iterdir():
        if folder.is_dir():
            for field in ['close','close_qfq','change','pct_chg','factor','volume']:
                path=folder/f'{field}.day.bin'
                if path.exists():
                    stat=path.stat();stats[str(path.relative_to(provider))]=[stat.st_size,stat.st_mtime_ns]
    return stats


def find_frozen_model(source):
    """Follow saved replay provenance without loading a model or assuming one hop."""
    seen=set()
    current=Path(source).resolve()
    expected=None
    while current not in seen:
        seen.add(current)
        cfg=yaml.safe_load((current/'resolved_config.yaml').read_text(encoding='utf-8'))
        lineage=cfg.get('replay',{})
        recorded=lineage.get('source_model_sha256')
        if expected is not None and recorded is not None and recorded!=expected:
            raise ValueError('Model hash differs across replay provenance')
        expected=expected or recorded
        candidate=current/'frozen_model.pkl'
        if candidate.is_file():
            if expected is not None and sha(candidate)!=expected:
                raise ValueError('Frozen model hash changed')
            return candidate
        if not lineage.get('source_run'):
            raise ValueError('No frozen model in replay provenance')
        current=Path(lineage['source_run']).resolve()
    raise ValueError('Cycle in replay provenance')


def excluded_board(code):
    """Board classification for the saved A-share stock universe."""
    if code.endswith('.SZ') and code.startswith('3'):
        return '创业板'
    if code.endswith('.SH') and code.startswith(('688','689')):
        return '科创板'
    return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'artifacts/frozen_model_replay/20261029_lgbm_optuna_to_20261015_v2')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fee-config',type=Path,help='YAML containing backtest.fee_model; change costs only')
    parser.add_argument('--exclude-growth-boards',action='store_true',help='Exclude ChiNext and STAR; retain saved scores without recomputing predictions')
    parser.add_argument('--strategy-config',type=Path,help='YAML strategy only; preserve baseline universe and fees')
    args=parser.parse_args();source=args.source.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    ledger={'status':'preparing','training_runs':0,'prediction_runs':0,'backtest_runs':0,'source':str(source)}

    def stage(name):
        ledger.update(status=name,elapsed_seconds=round(time.monotonic()-started,2))
        save(out/'run_ledger.json',ledger);print(json.dumps(ledger,ensure_ascii=False),flush=True)

    try:
        from backtest.engine import BacktestEngine
        from backtest.analysis_layer import AnalysisLayer
        from backtest.portfolio_layer import PortfolioLayer
        import qlib
        from qlib.data import D

        protected=[p for p in source.rglob('*') if p.is_file()]
        protected+=list((ROOT/'artifacts/models').glob('latest_*'))
        hashes={str(p):sha(p) for p in protected}
        for name in ['signals.parquet','labels.parquet','universe.json']:
            shutil.copy2(source/name,out/name)
        pred_frame=pd.read_parquet(out/'signals.parquet')
        board_filter=None
        if args.exclude_growth_boards:
            if args.fee_config:
                raise ValueError('Board-only comparison cannot also change fee configuration')
            codes=pred_frame.index.get_level_values('instrument')
            removed=sorted(c for c in codes.unique() if excluded_board(c))
            keep=~codes.isin(removed)
            shutil.copy2(source/'signals.parquet',out/'source_signals.parquet')
            pred_frame=pred_frame.loc[keep].copy()
            pred_frame.to_parquet(out/'signals.parquet')
            universe=json.loads((out/'universe.json').read_text(encoding='utf-8'))
            original_count=universe['n_instruments']
            universe['instruments']=[c for c in universe['instruments'] if not excluded_board(c)]
            universe['n_instruments']=len(universe['instruments'])
            universe['excluded_boards']=['创业板','科创板']
            save(out/'universe.json',universe)
            board_filter={'original_instruments':original_count,'remaining_instruments':universe['n_instruments'],
                'removed_instruments':len(removed),'removed_by_board':{b:sum(excluded_board(c)==b for c in removed) for b in ['创业板','科创板']},
                'remaining_prediction_rows':len(pred_frame),'removed_codes':removed,
                'method':'Filter saved prediction rows before Topk selection; retained scores unchanged. No re-standardization or new prediction.'}
            save(out/'board_filter.json',board_filter)
        pred=pred_frame['score']
        labels=pd.read_parquet(out/'labels.parquet').iloc[:,0]
        if args.exclude_growth_boards:
            labels=labels.reindex(pred.index)
            labels.to_frame().to_parquet(out/'labels.parquet')
        before=pd.read_parquet(source/'backtest_report.parquet')
        if pred.index.has_duplicates or not np.isfinite(pred.values).all():
            raise ValueError('Invalid saved predictions')
        if not (before.net_return == before['return']-before.cost).all():
            raise ValueError('Reference report is not on the corrected cost basis')
        cfg=yaml.safe_load((source/'resolved_config.yaml').read_text(encoding='utf-8'))
        if args.strategy_config:
            if args.fee_config or args.exclude_growth_boards:
                raise ValueError('Strategy comparison must preserve universe and fee model')
            replacement=yaml.safe_load(args.strategy_config.read_text(encoding='utf-8'))
            cfg['strategy']=replacement['strategy']
        model_path=find_frozen_model(source)
        model_hash=sha(model_path)
        if cfg.get('replay',{}).get('source_model_sha256',model_hash)!=model_hash:
            raise ValueError('Frozen model hash changed')
        hashes[str(model_path)]=model_hash
        if args.fee_config:
            if cfg['backtest'].get('quote_contract') != 'tushare_qfq_v1':
                raise ValueError('Fee comparison requires an already repaired quote baseline')
            from backtest.fees import AShareFees
            fees=AShareFees(**yaml.safe_load(args.fee_config.read_text(encoding='utf-8'))['backtest']['fee_model'])
            cfg['backtest']['fee_model']=asdict(fees)
            for key in ['open_cost','close_cost','min_cost']:
                cfg['backtest']['exchange_kwargs'].pop(key,None)
            if cfg['backtest']['exchange_kwargs'].get('impact_cost',0)!=0:
                raise ValueError('Unexpected legacy impact cost')
        else:
            from backtest.fees import AShareFees
            fees=AShareFees(**cfg['backtest']['fee_model']) if cfg['backtest'].get('fee_model') is not None else None
        change_label='费用口径' if fees is not None else '撮合报价'
        old_fees=yaml.safe_load((source/'resolved_config.yaml').read_text(encoding='utf-8'))['backtest'].get('fee_model')
        before_label=f"单边滑点{old_fees['slippage_rate']:.2%}（扣费后）" if fees is not None and old_fees is not None else '修复前（扣费后）'
        after_label=f"单边滑点{fees.slippage_rate:.2%}（扣费后）" if fees is not None and old_fees is not None else '修复后（扣费后）'
        if args.exclude_growth_boards:
            if cfg['backtest'].get('quote_contract')!='tushare_qfq_v1':
                raise ValueError('Board-only comparison requires repaired execution quotes')
            change_label='剔除创业板和科创板'
            before_label='原全板块（扣费后）'
            after_label='剔除创业板和科创板（扣费后）'
            cfg['replay_universe_filter']={'excluded_boards':['创业板','科创板']}
        if args.strategy_config:
            change_label='周调仓策略'
            before_label='每日最多换1只（扣费后）'
            after_label=f"每周最多换{cfg['strategy']['kwargs']['n_drop']}只（扣费后）"
        cfg['experiment']['run_id']=out.name
        cfg['backtest']['quote_contract']='tushare_qfq_v1'
        start,end=cfg['dataset']['kwargs']['segments']['test']
        if str(end)!='2026-10-15':
            raise ValueError('Expected frozen replay ending 2026-10-15')
        provider=Path(cfg['qlib_init']['provider_uri'])
        for path in [provider/'calendars/day.txt',provider/'instruments/all.txt',provider/'instruments/csi500.txt']:
            hashes[str(path)]=sha(path)
        stats=provider_stats(provider)
        save(out/'provider_file_stats_before.json',stats)
        prior_stats=json.loads((source/'provider_file_stats_before.json').read_text(encoding='utf-8'))
        if any(prior_stats[p]!=v for p,v in stats.items() if p in prior_stats):
            raise ValueError('Market data changed since reference replay')
        private=out/'calendar_provider'
        shutil.copytree(source/'calendar_provider',private)
        calendar=pd.DatetimeIndex(pd.to_datetime((private/'calendars/day.txt').read_text().splitlines()))
        expected=calendar[(calendar>=pd.Timestamp(start))&(calendar<=pd.Timestamp(end))]
        dates=pd.DatetimeIndex(pred.index.get_level_values('datetime').unique()).sort_values()
        if not dates.equals(expected) or not before.index.equals(expected):
            raise ValueError('Saved predictions/reference dates are not aligned')
        cfg['replay']={'source_prediction_sha256':sha(out/'signals.parquet'),'source_model_sha256':model_hash,
            'source_run':str(source),'training_runs':0,'prediction_runs':0,'private_calendar':str(private)}
        if board_filter is not None:
            cfg['replay']['original_prediction_sha256']=sha(source/'signals.parquet')
            cfg['replay']['prediction_transform']=board_filter['method']
        (out/'resolved_config.yaml').write_text(yaml.safe_dump(cfg,allow_unicode=True,sort_keys=False),encoding='utf-8')
        for path in [Path(__file__),ROOT/'python/backtest/quote_contract.py',ROOT/'python/backtest/engine.py',ROOT/'python/backtest/market_layer.py',ROOT/'python/backtest/portfolio_layer.py',ROOT/'python/backtest/fees.py']:
            shutil.copy2(path,out/(path.stem+'_source.py'))
        if args.strategy_config:
            shutil.copy2(ROOT/'python/backtest/strategy/weekly_topk.py',out/'weekly_topk_source.py')
        save(out/'source_sha256.json',hashes)
        save(out/'protocol.json',{'source':str(source),'start':str(start),'end':str(end),
             'training_runs':0,'prediction_runs':0,'planned_backtest_runs':1,
             'change':change_label,'fee_model':asdict(fees) if fees is not None else None,
             'board_filter':board_filter,
             'strategy':cfg['strategy'],
             'fee_reference':'https://one.sse.com.cn/onething/gptz/',
             'slippage_method':'fixed cash-equivalent cost per side; included in net returns',
             'prediction_sha256':sha(out/'signals.parquet'),'model_sha256':model_hash})
        qlib.init(provider_uri=str(provider),region='cn',kernels=1,expression_cache=None,dataset_cache=None,
            calendar_provider={'class':'LocalCalendarProvider','kwargs':{'backend':{'class':'FileCalendarStorage',
                'module_path':'qlib.data.storage.file_storage','kwargs':{'provider_uri':str(private)}}}})
        stage('checking_execution_quotes')
        instruments=sorted(pred.index.get_level_values('instrument').unique())
        quotes=D.features(instruments,['$close','$close_qfq','$pct_chg','$change'],start_time=start,end_time=end)
        quotes.to_parquet(out/'source_execution_quotes.parquet')
        daily=quotes.notna().groupby(level='datetime').mean()
        daily.to_csv(out/'quote_coverage.csv')
        if (daily[['$close_qfq','$pct_chg']].min()<.9).any():
            raise ValueError('Insufficient execution price or percent-change coverage')
        ledger['backtest_runs']=1
        stage('backtest_started')
        engine=BacktestEngine();result=engine.run(pred,cfg,output_dir=out,label=labels)
        after=result.report_df
        for name,frame in [('backtest_report',after),('indicator',result.indicator_df),('trades',result.trades_df),('realized_pnl',result.realized_pnl_df)]:
            frame.to_parquet(out/f'{name}.parquet')
        if not after.index.equals(expected):
            raise ValueError('Replay omitted a requested trading day')
        stage('reconciling_execution_and_costs')
        trades=result.trades_df
        if args.exclude_growth_boards and trades.stock_id.map(excluded_board).notna().any():
            raise ValueError('An excluded-board stock traded')
        idx=pd.MultiIndex.from_arrays([trades.stock_id,trades.datetime],names=['instrument','datetime'])
        actual=quotes.reindex(idx)
        prices=actual['$close_qfq'].to_numpy()
        fractions=actual['$pct_chg'].to_numpy()/100
        if not np.isfinite(prices).all() or not np.allclose(prices,trades.price,atol=1e-7,rtol=1e-10):
            raise ValueError('Trades are not at the frozen QFQ closing prices')
        if not np.isfinite(fractions).all() or (np.abs(fractions)>=cfg['backtest']['exchange_kwargs']['limit_threshold']).any():
            raise ValueError('A fill violates the unchanged percentage threshold')
        trade_audit=trades.copy()
        trade_audit['pct_change_fraction']=fractions
        trade_audit['source_close_qfq']=prices
        trade_audit.to_parquet(out/'executed_quote_audit.parquet')
        signed=trades.assign(quantity=trades.amount*np.where(trades.direction=='BUY',1,-1))
        holdings=signed.pivot_table(index='datetime',columns='stock_id',values='quantity',aggfunc='sum',fill_value=0).reindex(expected,fill_value=0).cumsum()
        if holdings.min().min() < -1e-5:
            raise ValueError('Negative reconstructed stock quantity')
        holdings=holdings.where(holdings>1e-6,0)
        price_matrix=quotes['$close_qfq'].unstack('instrument').reindex(expected).ffill()[holdings.columns]
        if (price_matrix.isna()&(holdings>0)).any().any():
            raise ValueError('Missing marking price for a held stock')
        independent_value=(holdings*price_matrix).sum(axis=1)
        account_error=independent_value+after.cash-after.account
        initial=float(cfg['experiment']['account'])
        net_error=initial*(1+after['return']-after.cost).cumprod()-after.account
        actual_cash_cost=trades.groupby('datetime').cost.sum().reindex(expected,fill_value=0)
        cost_error=after.cost*after.account.shift(fill_value=initial)-actual_cash_cost
        if max(account_error.abs().max(),net_error.abs().max(),cost_error.abs().max())>1e-5:
            raise ValueError('Independent valuation/cost reconciliation failed')
        if not np.isclose(trades.cost.sum(),after.total_cost.iloc[-1],atol=1e-5,rtol=1e-10):
            raise ValueError('Cumulative trade cost mismatch')
        if fees is not None:
            components=trades[['commission','stamp_tax','transfer_fee','slippage']]
            totals={k:float(v) for k,v in components.sum().items()}
            totals['explicit_fees']=sum(totals[k] for k in ['commission','stamp_tax','transfer_fee'])
            totals['total']=totals['explicit_fees']+totals['slippage']
            totals['minimum_commission_orders']=int((trades.value*fees.commission_rate < fees.minimum_commission).sum())
            save(out/'cost_components.json',totals)
            daily_components=trades.groupby('datetime')[components.columns].sum().reindex(expected,fill_value=0)
            daily_components.to_csv(out/'daily_cost_components.csv')
            if not np.allclose(daily_components.sum(axis=1),actual_cash_cost,atol=1e-8,rtol=1e-10):
                raise ValueError('Daily component fees do not reconcile')
        first_trade=trades.datetime.min()
        if abs(after.loc[first_trade,'return'])>1e-10:
            raise ValueError('First close-price purchase produced artificial gross PnL')
        holdings.to_parquet(out/'holdings_reconstructed.parquet')
        pd.DataFrame({'qfq_stock_value':independent_value,'account_error':account_error,'net_curve_error':net_error,'cash_cost_error':cost_error}).to_parquet(out/'reconciliation.parquet')
        if not np.allclose(before.bench,after.bench,atol=1e-12,rtol=1e-10):
            raise ValueError('Benchmark changed in an execution-only comparison')
        analysis=AnalysisLayer();portfolio=PortfolioLayer()
        rows=[]
        for period,dates in [('full',expected),('through_20260522',expected[expected<='2026-05-22']),('extension',expected[expected>'2026-05-22'])]:
            for basis,frame in [('before',before),('after',after)]:
                part=frame.loc[dates]
                met=asdict(analysis.compute_metrics(portfolio.parse({'1day':part})))
                prev=frame.account.shift(fill_value=initial).reindex(dates)
                rows.append(dict(period=period,version=basis,days=len(dates),start=str(dates.min().date()),end=str(dates.max().date()),
                    cash_cost=float((part.cost*prev).sum()),**met))
        comparison=pd.DataFrame(rows)
        comparison.to_csv(out/'comparison_net.csv',index=False)
        aligned=pd.DataFrame({f'{version}_{col}':frame[col] for version,frame in [('before',before),('after',after)]
            for col in ['return','cost','net_return','account','cum_return','turnover']})
        aligned.to_parquet(out/'aligned_comparison.parquet')
        net=engine.metrics_to_dict(result)
        gross=asdict(analysis.compute_metrics(portfolio.parse({'1day':after.assign(cost=0.0)})))
        notes=[
            '使用同一份已保存预测、相同643只候选集合、相同策略和费用，仅修复撮合报价口径；未训练、未生成预测。',
            '涨跌幅使用pct_chg/100；成交价和持仓估值统一为close_qfq；无有效涨跌幅或价格时禁止成交。',
            '保留原9.5%统一阈值，这是原策略的保护规则，尚未升级为逐证券、逐日期的交易所涨跌停规则。',
            '保留原Qlib复权数量及小数股行为，未新增整手、容量或完整实盘可执行性约束。',
            '原基准000905.SZ不是中证500指数；股票池仍为区间成分集合，非逐日PIT成分。基准及股票池问题未在本轮变更。',
            '财务历史版本和复权数据仍沿用原预测对应的数据版本，本轮不是独立前瞻样本外检验。',
        ]
        if fees is not None:
            notes[0]='复用完全相同的已保存预测、候选集合、策略和报价规则，仅修正费用；未训练、未生成预测。'
            notes += ['佣金双边万3.5，按含规费假设，最低每笔5元仅作用于佣金；印花税卖出万5，过户费双边十万分之一。',
                      f'滑点采用单边{fees.slippage_rate*10000:g}bp假设，以现金等价成本独立扣款并计入净收益，不改变成交价；没有独立缓冲或二次冲击成本。',
                      '费用改变可用现金、持仓数量及复利路径；本对比是同预测重新回测，不是旧费用直接加回。']
        if board_filter is not None:
            notes[0]=f"仅在选股前过滤创业板和科创板：候选股票从{board_filter['original_instruments']}只降至{board_filter['remaining_instruments']}只；保留分数逐值不变，不重新计算特征、标准化或预测。"
            notes[-1]='其余策略、报价、佣金税费和滑点与基线一致；从剩余股票中重新选取Top5，剔除股票的资金可买入替代股票。'
        if args.strategy_config:
            notes[0]='回到未剔除板块的643只候选池，预测文件逐字节复用；仅调整调仓策略，不训练、不生成预测。'
            notes[-1]=f"首次有可用信号时建仓，此后每个ISO周首个可用交易日检查，最多替换{cfg['strategy']['kwargs']['n_drop']}只；使用前一交易日分数，周内不重试成交或调仓，不增加止损规则。"
        metrics={'run_id':out.name,'source_run':source.name,'return_basis':'net_of_transaction_costs','actual_start':str(expected.min().date()),
            'actual_end':str(expected.max().date()),'n_days':len(expected),'training_runs':0,'prediction_runs':0,'backtest_runs':1,
            'metrics':net,'gross_metrics_same_trades':gross,'ending_account':float(after.account.iloc[-1]),'cash_cost_total':float(after.total_cost.iloc[-1]),
            'before_net_metrics':json.loads((source/'metrics.json').read_text(encoding='utf-8'))['metrics'],'limitations':notes}
        metrics['comparison_labels']={'before':before_label,'after':after_label}
        metrics['board_filter']=board_filter
        metrics['strategy']=cfg['strategy']
        if fees is not None:
            metrics['fee_model']=asdict(fees)
            metrics['cost_components']=totals
            metrics['fee_reference']='https://one.sse.com.cn/onething/gptz/'
        save(out/'metrics.json',metrics)
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
        plt.rcParams['axes.unicode_minus']=False
        fig,ax=plt.subplots(figsize=(11,4.5))
        ax.plot(expected,before.cum_return,label=before_label,linestyle='--',color='#777777')
        ax.plot(expected,after.cum_return,label=after_label,color='#1769aa')
        ax.axhline(1,color='#999999',linewidth=.7);ax.set_ylabel('账户净值（初始=1）');ax.legend();ax.grid(alpha=.2)
        ax.set_title(f'复用预测、相同日期：{change_label}对比');fig.tight_layout();fig.savefig(out/'comparison_net.png',dpi=160);plt.close(fig)
        # Put the exact paired comparison before the inherited single-run report.
        oldmet=metrics['before_net_metrics']
        fields=[('total_return','累计收益','{:.2%}'),('annualized_return','年化收益','{:.2%}'),('max_drawdown','最大回撤','{:.2%}'),('sharpe_ratio','夏普比率','{:.3f}')]
        table=''.join('<tr><td>'+label+'</td><td>'+fmt.format(oldmet[k])+'</td><td>'+fmt.format(net[k])+'</td></tr>' for k,label,fmt in fields)
        panel=f'<section style="margin:24px;padding:24px;background:#eef5fb;color:#172633"><h2>复用预测：{change_label}对比（均已扣费）</h2>'
        panel+=f'<p>{expected.min().date()} 至 {expected.max().date()}，{len(expected)} 个交易日；训练 0 次，预测 0 次，回测 1 次。</p>'
        panel+=f'<table style="width:100%;text-align:left"><thead><tr><th>指标</th><th>{before_label}</th><th>{after_label}</th></tr></thead><tbody>'+table+'</tbody></table>'
        panel+=f'<p>初始资金 {initial:,.2f} 元；修复后期末 {after.account.iloc[-1]:,.2f} 元；累计费用 {after.total_cost.iloc[-1]:,.2f} 元。{engine._market_layer.extract_cost_model(cfg["backtest"]).describe()}</p>'
        if fees is not None:
            names={'commission':'佣金','stamp_tax':'印花税','transfer_fee':'过户费','slippage':'滑点假设','explicit_fees':'显性费税合计','total':'全部成本'}
            panel+='<h3>费用分项（元）</h3><table>'+''.join(f'<tr><td>{v}</td><td>{totals[k]:,.2f}</td></tr>' for k,v in names.items())+'</table>'
            panel+='<p>佣金按含规费口径；<a href="https://one.sse.com.cn/onething/gptz/">印花税与过户费标准来源：上交所</a>（2026-10-29核对）。滑点为假设，不是券商手续费。</p>'
            panel+='<p><a href="trade_cost_components.csv">逐笔费用明细</a> · <a href="daily_cost_components.csv">逐日费用明细</a> · <a href="comparison_net.csv">同区间净收益对比</a></p>'
        panel+='<img src="comparison_net.png" style="width:100%;max-width:1100px" alt="同预测扣费净值对比"><ul>'+''.join('<li>'+html.escape(note)+'</li>' for note in notes)+'</ul></section>'
        report=out/'report.html';report.write_text(report.read_text(encoding='utf-8').replace('<body>','<body>'+panel,1),encoding='utf-8')
        if any(sha(p)!=h for p,h in hashes.items()):
            raise ValueError('A protected reference artifact changed')
        if stats!=provider_stats(provider):
            raise ValueError('Shared execution data changed')
        if board_filter is not None:
            original=pd.read_parquet(source/'signals.parquet')
            retained=original.loc[~original.index.get_level_values('instrument').map(excluded_board).notna()]
            pd.testing.assert_frame_equal(pd.read_parquet(out/'signals.parquet'),retained,check_exact=True)
            if sha(out/'source_signals.parquet')!=hashes[str(source/'signals.parquet')]:
                raise ValueError('Original prediction copy changed')
        elif sha(out/'signals.parquet')!=hashes[str(source/'signals.parquet')]:
            raise ValueError('Saved prediction bytes changed')
        checks={'source_sha256_unchanged':True,'source_files_checked':len(hashes),'shared_data_stats_unchanged':True,'shared_quote_files_checked':len(stats),
            'prediction_bytes_identical':board_filter is None,'retained_prediction_values_identical':True,
            'excluded_boards_absent':True if board_filter is not None else None,
            'model_loaded':False,'training_runs':0,'prediction_runs':0,'backtest_runs':1,
            'trade_prices_match_qfq':True,'all_fills_pass_original_percent_threshold':True,'first_trade_gross_return':float(after.loc[first_trade,'return']),
            'max_valuation_account_error':float(account_error.abs().max()),'max_net_account_error':float(net_error.abs().max()),
            'max_daily_cash_cost_error':float(cost_error.abs().max()),'benchmark_unchanged':True,
            'artifact_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['run_ledger.json','postflight.json']}}
        save(out/'postflight.json',checks)
        stage('complete');print(json.dumps(metrics,ensure_ascii=False),flush=True)
    except Exception as exc:
        ledger['error']=f'{type(exc).__name__}: {exc}';stage('failed');raise


if __name__=='__main__':main()
