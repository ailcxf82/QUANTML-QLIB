"""Export investment detail from a completed replay; never run a backtest.

Read saved trades, FIFO realized P&L, quotes and account reports. Reconstruct
remaining FIFO purchase costs and mark open positions at saved QFQ closes.
"""
import argparse
from collections import defaultdict, deque
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


LABELS = {
    'datetime':'日期','stock_id':'证券代码','market':'板块','direction':'方向',
    'amount':'复权数量','price':'成交价_前复权','value':'成交金额_元',
    'commission':'佣金_元','stamp_tax':'印花税_元','transfer_fee':'过户费_元',
    'slippage':'滑点_元','cost':'总成本_元','explicit_fees':'显性费税_元',
    'cash_flow':'净现金流_元','close_date':'卖出日期','open_date':'买入日期',
    'holding_days':'持有自然日','buy_price':'买入价_前复权','sell_price':'卖出价_前复权',
    'gross_pnl':'毛盈亏_元','trade_cost':'分摊买卖成本_元','net_pnl':'已实现净盈亏_元',
    'return_pct':'净盈亏占买入成交额','quantity':'期末复权数量',
    'mark_price':'估值价_前复权','market_value':'市值_元','remaining_buy_value':'剩余买入成交额_元',
    'remaining_buy_cost':'剩余买入费用_元','book_cost':'持仓成本含买入费用_元',
    'unrealized_pnl':'浮动盈亏含买入费用_元','weight':'账户权重',
    'first_open_date':'最早未平仓买入日','account':'总资产_元','cash':'现金_元',
    'stock_value':'股票市值_元','daily_net_pnl':'当日净盈亏_元',
    'net_return':'当日净收益率','cumulative_net_return':'累计净收益率',
    'turnover':'双边换手率','trade_count':'成交笔数','buy_value':'累计买入金额_元',
    'sell_value':'累计卖出金额_元','realized_pnl':'已实现净盈亏_元',
    'total_net_pnl':'合计净盈亏_元', 'position_count':'持仓数',
}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    args=ap.parse_args(); run=args.run.resolve(); out=run/'investment_details'
    out.mkdir(exist_ok=False)
    files=['trades.parquet','realized_pnl.parquet','backtest_report.parquet',
           'source_execution_quotes.parquet','metrics.json','holdings_reconstructed.parquet']
    hashes={name:sha(run/name) for name in files}
    trades=pd.read_parquet(run/'trades.parquet')
    realized=pd.read_parquet(run/'realized_pnl.parquet')
    report=pd.read_parquet(run/'backtest_report.parquet')
    quotes=pd.read_parquet(run/'source_execution_quotes.parquet')
    holdings=pd.read_parquet(run/'holdings_reconstructed.parquet')
    metrics=json.loads((run/'metrics.json').read_text(encoding='utf-8'))
    old_path=Path(json.loads((run/'protocol.json').read_text(encoding='utf-8'))['source'])
    old_metrics=json.loads((old_path/'metrics.json').read_text(encoding='utf-8'))
    labels=metrics.get('comparison_labels',{'before':'原单边10bp','after':'新单边5bp'})
    board_comparison=metrics.get('board_filter') is not None
    title='剔除创业板和科创板：投资明细' if board_comparison else '5bp滑点：回测投资明细'
    if metrics.get('strategy',{}).get('class')=='WeeklyTopkDropoutStrategy':
        title=f"每周最多换{metrics['strategy']['kwargs']['n_drop']}只：投资明细"
    initial=300000.0
    assert not trades.duplicated(['datetime','stock_id']).any(), 'Need explicit intraday ordering for repeated stock fills'
    marks=quotes['$close_qfq'].unstack('instrument').reindex(report.index).ffill()
    groups={dt:g for dt,g in trades.groupby('datetime')}
    lots=defaultdict(deque); holding_rows=[]; closed_pnl=0.0
    for dt in report.index:
        for row in groups.get(dt,pd.DataFrame()).itertuples():
            if row.direction=='BUY':
                lots[row.stock_id].append({'quantity':row.amount,'price':row.price,
                    'cost_per_unit':row.cost/row.amount,'date':dt})
            else:
                remaining=row.amount
                while remaining>1e-7 and lots[row.stock_id]:
                    lot=lots[row.stock_id][0]; matched=min(remaining,lot['quantity'])
                    closed_pnl += matched*(row.price-lot['price']-lot['cost_per_unit']-row.cost/row.amount)
                    remaining-=matched;lot['quantity']-=matched
                    if lot['quantity']<1e-7: lots[row.stock_id].popleft()
                assert remaining<1e-6, 'Sell quantity exceeds saved buys'
        for code, queue in lots.items():
            if not queue: continue
            qty=sum(l['quantity'] for l in queue)
            mark=float(marks.loc[dt,code]); assert np.isfinite(mark)
            principal=sum(l['quantity']*l['price'] for l in queue)
            fees=sum(l['quantity']*l['cost_per_unit'] for l in queue)
            holding_rows.append(dict(datetime=dt,stock_id=code,quantity=qty,mark_price=mark,
                market_value=qty*mark,remaining_buy_value=principal,remaining_buy_cost=fees,
                book_cost=principal+fees,unrealized_pnl=qty*mark-principal-fees,
                weight=qty*mark/report.loc[dt,'account'],first_open_date=min(l['date'] for l in queue)))
    daily_holdings=pd.DataFrame(holding_rows)
    ending=daily_holdings[daily_holdings.datetime==report.index[-1]].sort_values('market_value',ascending=False).copy()
    np.testing.assert_allclose(daily_holdings.groupby('datetime').market_value.sum().reindex(report.index,fill_value=0),
                               report.account-report.cash,atol=1e-6)
    rebuilt=daily_holdings.pivot(index='datetime',columns='stock_id',values='quantity').reindex(index=holdings.index,columns=holdings.columns).fillna(0)
    np.testing.assert_allclose(rebuilt,holdings,atol=1e-6)
    assert np.isclose(closed_pnl,realized.net_pnl.sum(),atol=1e-6)
    final_profit=float(report.account.iloc[-1]-initial)
    assert np.isclose(realized.net_pnl.sum()+ending.unrealized_pnl.sum(),final_profit,atol=1e-6)
    transactions=trades.copy()
    transactions['cash_flow']=np.where(transactions.direction=='BUY',-transactions.value,transactions.value)-transactions.cost
    daily_cash=initial+transactions.groupby('datetime').cash_flow.sum().reindex(report.index,fill_value=0).cumsum()
    np.testing.assert_allclose(daily_cash,report.cash,atol=1e-6)
    transactions=transactions.drop(columns=['fill_ratio'])
    transactions.direction=transactions.direction.map({'BUY':'买入','SELL':'卖出'})
    daily=report[['account','cash','net_return','turnover']].copy()
    daily['stock_value']=daily.account-daily.cash
    daily['daily_net_pnl']=daily.account-daily.account.shift(fill_value=initial)
    daily['cumulative_net_return']=daily.account/initial-1
    daily['position_count']=daily_holdings.groupby('datetime').size().reindex(report.index,fill_value=0)
    for col in ['commission','stamp_tax','transfer_fee','slippage','cost']:
        daily[col]=trades.groupby('datetime')[col].sum().reindex(report.index,fill_value=0)
    daily=daily.rename_axis('datetime').reset_index()
    stock=trades.groupby('stock_id').agg(trade_count=('value','size'),cost=('cost','sum'),
        commission=('commission','sum'),stamp_tax=('stamp_tax','sum'),transfer_fee=('transfer_fee','sum'),slippage=('slippage','sum'))
    for side,col in [('BUY','buy_value'),('SELL','sell_value')]:
        stock[col]=trades[trades.direction==side].groupby('stock_id').value.sum()
    stock['realized_pnl']=realized.groupby('stock_id').net_pnl.sum()
    for col in ['market_value','unrealized_pnl']:
        stock[col]=ending.set_index('stock_id')[col]
    stock=stock.fillna(0)
    stock['total_net_pnl']=stock.realized_pnl+stock.unrealized_pnl
    stock=stock.sort_values('total_net_pnl',ascending=False).reset_index()
    assert np.isclose(stock.total_net_pnl.sum(),final_profit,atol=1e-6)
    tables={'ending_holdings':('期末持仓',ending),'transactions':('逐笔买卖与费用',transactions),
            'realized_pnl':('已实现盈亏（FIFO配对）',realized),'stock_pnl':('个股盈亏汇总',stock),
            'daily_holdings':('每日持仓',daily_holdings),'daily_account':('每日账户',daily)}
    sections=[]
    percent={'weight','return_pct','net_return','cumulative_net_return','turnover'}
    for key,(title,df) in tables.items():
        export=df.copy()
        for c in export.select_dtypes(include=['datetime','datetimetz']).columns:
            export[c]=export[c].dt.strftime('%Y-%m-%d')
        export.rename(columns=LABELS).to_csv(out/f'{key}.csv',index=False,encoding='utf-8-sig',float_format='%.8f')
        view=export.copy()
        for col in view.select_dtypes(include='number'):
            if col in percent: view[col]=view[col].map(lambda x:f'{x:.2%}')
            elif col in ['price','mark_price','buy_price','sell_price','amount','quantity']:
                view[col]=view[col].map(lambda x:f'{x:,.4f}')
            elif col not in ['trade_count','holding_days','position_count']:
                view[col]=view[col].map(lambda x:f'{x:,.2f}')
        content=view.rename(columns=LABELS).to_html(index=False,border=0,escape=True,table_id=key)
        sections.append(f'<details {"open" if key in ["ending_holdings","transactions"] else ""}><summary>{title} · {len(df):,} 行</summary>'
                        f'<p><a href="{key}.csv" download>下载完整CSV</a> <label>筛选代码或日期 <input data-target="{key}" placeholder="例如 600000.SH 或 2026-10-15"></label> <span class="count"></span></p><div class="scroll">{content}</div></details>')
    fields=[('total_return','累计净收益'),('annualized_return','年化净收益'),('max_drawdown','最大回撤')]
    compare=''.join(f'<tr><td>{name}</td><td>{old_metrics["metrics"][key]:.2%}</td><td>{metrics["metrics"][key]:.2%}</td></tr>' for key,name in fields)
    compare+=f'<tr><td>期末资产</td><td>{old_metrics["ending_account"]:,.2f}</td><td>{metrics["ending_account"]:,.2f}</td></tr>'
    compare+=f'<tr><td>总成本（含滑点）</td><td>{old_metrics["cash_cost_total"]:,.2f}</td><td>{metrics["cash_cost_total"]:,.2f}</td></tr>'
    notes=(f'已实现净盈亏 {realized.net_pnl.sum():,.2f} 元 + 期末浮动盈亏 {ending.unrealized_pnl.sum():,.2f} 元 = 账户净利润 {final_profit:,.2f} 元。'
           '浮动盈亏已扣分摊买入费用，未预扣未来卖出费用；已实现盈亏按FIFO配对。'
           '价格均为前复权价，数量为回测复权单位，可能含小数，不等于实盘股数。'
           '日内成交顺序未记录，同一天的表格顺序不代表实际先后。未提供证券名称，代码为权威标识。')
    doc='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>5bp滑点回测投资明细</title>
<style>body{font:15px/1.6 system-ui,"Microsoft YaHei",sans-serif;color:#172633;background:#f5f7fa;margin:24px}main{max-width:1500px;margin:auto}h1{font-size:26px}a{color:#0864ad}table{border-collapse:collapse;white-space:nowrap;width:100%;font-variant-numeric:tabular-nums}th,td{padding:8px 12px;border-bottom:1px solid #d8e1e8;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e9f0f7;position:sticky;top:0}details{background:white;padding:16px;margin:16px 0;border-radius:8px}summary{font-size:18px;cursor:pointer}input{padding:7px;min-width:240px;max-width:100%;margin:0 8px}.scroll{overflow:auto;max-height:650px}.overview{max-width:760px}.note{background:#eaf3fb;padding:16px}nav{display:flex;gap:18px;flex-wrap:wrap}tr:hover{background:#f1f7fc}</style></head><body><main>
<h1>5bp滑点：回测投资明细</h1><p>2025-01-02 至 2026-10-15 · 初始资金 300,000 元 · 相同已保存预测 · 训练0次、预测0次、回测1次</p>
<nav><a href="../report.html">完整回测报告</a><a href="../comparison_net.csv">分区间对比</a></nav>
<div class="overview"><table><thead><tr><th>指标</th><th>原单边10bp</th><th>新单边5bp</th></tr></thead><tbody>'''+compare+'''</tbody></table></div>
<p>佣金双边万3.5（含规费、每笔最低5元），印花税仅卖出0.05%，过户费双边0.001%；滑点双边各0.05%，单独计入成本。</p>
<p class="note">'''+html.escape(notes)+'''</p>'''+''.join(sections)+'''
<p>原统一9.5%涨跌停保护、区间股票池、复权小数数量及错误的基准代码均沿用，以保持本次费用对比一致；不能把本明细视为实盘交割单或下单建议。</p>
</main><script>document.querySelectorAll('input[data-target]').forEach(input=>input.addEventListener('input',()=>{const rows=document.getElementById(input.dataset.target).tBodies[0].rows;let n=0;const q=input.value.trim().toLowerCase();for(const row of rows){const show=row.textContent.toLowerCase().includes(q);row.hidden=!show;if(show)n++;}input.closest('p').querySelector('.count').textContent=`显示 ${n} 行`;}));</script></body></html>'''
    doc=doc.replace('5bp滑点回测投资明细',title).replace('5bp滑点：回测投资明细',title)
    doc=doc.replace('原单边10bp',html.escape(labels['before'])).replace('新单边5bp',html.escape(labels['after']))
    if board_comparison:
        doc=doc.replace('相同已保存预测','保留股票的既有预测分数不变')
    (out/'index.html').write_text(doc,encoding='utf-8')
    summary={'source':str(run),'training_runs':0,'prediction_runs':0,'backtest_runs':0,
        'detail_generation_only':True,'ending_cash':float(report.cash.iloc[-1]),
        'ending_stock_value':float(ending.market_value.sum()),'ending_account':float(report.account.iloc[-1]),
        'realized_net_pnl':float(realized.net_pnl.sum()),'unrealized_net_pnl':float(ending.unrealized_pnl.sum()),
        'total_net_pnl':final_profit,'row_counts':{k:len(df) for k,(_,df) in tables.items()},
        'daily_cash_reconciles':True,'daily_holdings_reconcile':True,'fifo_pnl_reconciles':True,
        'source_sha256':hashes,'source_files_unchanged':all(sha(run/name)==h for name,h in hashes.items())}
    assert summary['source_files_unchanged']
    (out/'verification.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    (out/'builder_source.py').write_text(Path(__file__).read_text(encoding='utf-8'),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print(ending.to_string(index=False))


if __name__=='__main__':main()
