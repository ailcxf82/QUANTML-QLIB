"""Render saved accounting diagnostics without running a model or backtest."""
from pathlib import Path
import json
import hashlib
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'artifacts/weekly_diagnostics/20261029_drop1_yearly_attribution_v1'

def main():
    assert not (P/'report.html').exists(), 'Do not overwrite report'
    def read(n): return pd.read_csv(P/f'{n}.csv')
    s=read('period_summary').set_index('period'); stocks=read('stock_year_attribution')
    e=read('holding_episodes'); d=read('daily_stock_attribution'); m=read('monthly_attribution')
    c=read('concentration'); age=read('holding_age_attribution'); carry=read('cross_year_episodes')
    y25=s.loc['2025']; y26=s.loc['2026_to_1015']; same=s.loc['2025_to_1015']
    cohort=e.assign(open_year=pd.to_datetime(e.open_date).dt.year).groupby('open_year').net_2026.sum()
    pd.DataFrame({'open_year':cohort.index,'net_pnl_in_2026':cohort.values}).to_csv(P/'2026_entry_cohorts.csv',index=False,encoding='utf-8-sig')
    topcodes=stocks[stocks.period=='2025'].nlargest(10,'net_pnl').stock_id
    leaders=stocks.pivot(index='stock_id',columns='period',values='net_pnl').reindex(topcodes).fillna(0)[['2025','2026_to_1015']]
    leaders.to_csv(P/'2025_leaders_followthrough.csv',encoding='utf-8-sig')
    fonts={f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams['font.sans-serif']=[next((f for f in ['Microsoft YaHei','SimHei','Noto Sans CJK SC'] if f in fonts),'DejaVu Sans')]
    plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(2,2,figsize=(16,10),layout='constrained')
    fig.suptitle('周换1只：2025年盈利集中，2026年新开仓贡献转负',fontsize=19)
    ax=axs[0,0]; x=np.arange(3); w=.24
    view=s.loc[['2025','2025_to_1015','2026_to_1015']]
    for off,key,label,color in [(-w,'gross_return_cash_addback','加回实际费用','#789fbd'),(0,'fee_return_drag','费用拖累','#e69b53'),(w,'net_return','扣费后','#245b75')]:
        bars=ax.bar(x+off,view[key]*100,w,label=label,color=color)
        ax.bar_label(bars,fmt='%.2f',fontsize=9,padding=3)
    ax.set_xticks(x,['2025全年','2025至10/15','2026至10/15']); ax.set_ylabel('期初净值贡献（%）');ax.legend(fontsize=9)
    ax.set_title('相同截止日差异仍有91.42个百分点'); ax.set_ylim(0,112)
    ax=axs[0,1]; a=stocks[stocks.period=='2025'].head(10).iloc[::-1]
    ax.barh(a.stock_id,a.net_pnl/10000,color='#245b75'); ax.set_xlabel('年度净利润贡献（万元）');ax.set_title('2025年前十只贡献98.9%的净利润')
    ax=axs[1,0]; a=m[m.month.str.startswith('2026')]
    ax.bar(a.month.str[5:],a.net_pnl/10000,color=['#245b75' if v>=0 else '#bd604f' for v in a.net_pnl]);ax.axhline(0,color='#888',lw=.7)
    ax.set_xlabel('2026年月（10月仅至15日）');ax.set_ylabel('月度净利润（万元）');ax.set_title('波段盈利反复被亏损月份抵消')
    ax=axs[1,1]; x=np.arange(4)
    for offset,yr,col in [(-.18,2025,'#245b75'),(.18,2026,'#e69b53')]:
        a=age[age.year==yr].set_index('age_bin').reindex(['00-07','08-28','29-56','57+'])
        ax.bar(x+offset,a.net_pnl/10000,.36,label=str(yr),color=col)
    ax.axhline(0,color='#888',lw=.7);ax.set_xticks(x,['0–7天','8–28天','29–56天','57天以上']);ax.set_ylabel('当日持仓年龄分组净贡献（万元）')
    ax.set_title('2026年持仓第8–28天累计贡献为负');ax.legend()
    fig.savefig(P/'diagnostic_overview.png',dpi=160);plt.close(fig)
    def table(frame): return frame.to_html(index=False,border=0,classes='data',escape=True)
    def fmt(v): return f'{v:,.0f}'
    def pct(v): return f'{v:.2%}'
    annual=[]
    for key,label in [('2025','2025全年'),('2025_to_1015','2025至10/15'),('2026_to_1015','2026至10/15')]:
        a=s.loc[key]; annual.append({'区间':label,'交易日':int(a.sessions),'期初资金':fmt(a.start_account),'净利润':fmt(a.net_pnl),
           '扣费后收益':pct(a.net_return),'加回费用的贡献':pct(a.gross_return_cash_addback),'费用拖累':pct(a.fee_return_drag),'费用合计':fmt(a.fees)})
    fee=[]
    for label in ['2025','2025_to_1015','2026_to_1015']:
        a=s.loc[label];fee.append({'区间':label,'佣金':fmt(a.commission),'印花税':fmt(a.stamp_tax),'过户费':fmt(a.transfer_fee),
             '滑点':fmt(a.slippage),'合计':fmt(a.fees),'成交笔数':int(a.orders),'成交总额':fmt(a.traded_value)})
    def stocktable(period,bottom=False):
        a=stocks[stocks.period==period].sort_values('net_pnl',ascending=bottom).head(10)
        return pd.DataFrame({'代码':a.stock_id,'净利润贡献':a.net_pnl.map(fmt),'收益贡献':a.contribution_return.map(pct),
                             '费用':a.cost.map(fmt),'有持仓交易日':a.held_sessions.astype(int),'最高账户权重':a.max_weight.map(pct)})
    ct=c[c.period=='2025'].copy()
    ct=pd.DataFrame({'盈利股票累计':ct.top_n.map(lambda n:f'前{n}只'),'净利润贡献':ct.top_net_pnl.map(fmt),'占年度净利润':ct.share_of_net_pnl.map(pct),
                     '占盈利股票正贡献':ct.share_of_positive_pnl.map(pct),'扣除这些贡献后的账面收益':ct.residual_return_accounting_only.map(pct)})
    durations=[]
    for key in ['2025','2026_to_1015']:
        a=s.loc[key]; subset=e[pd.to_datetime(e.close_date).dt.year==(2025 if key=='2025' else 2026)].copy()
        subset['ret']=subset.net_pnl/subset.entry_value
        durations.append({'卖出年份':key,'已平仓笔数':int(a.closed_episodes),'盈利笔数占比':pct(a.closed_win_rate),
          '持仓中位数/日':f'{a.closed_median_days:.0f}','盈利笔平均持有/日':f'{a.closed_winner_mean_days:.1f}',
          '亏损笔平均持有/日':f'{a.closed_loser_mean_days:.1f}',
          '盈利笔平均净回报':pct(subset.loc[subset.net_pnl>0,'ret'].mean()),'亏损笔平均净回报':pct(subset.loc[subset.net_pnl<0,'ret'].mean())})
    ca=carry[['stock_id','open_date','close_date','net_2025','net_2026']].copy()
    ca.columns=['代码','开仓日期','平仓日期','2025净贡献','2026净贡献']
    for k in ['2025净贡献','2026净贡献']:ca[k]=ca[k].map(fmt)
    ep=e.nlargest(8,'net_pnl')[['stock_id','open_date','close_date','holding_calendar_days','net_pnl']].copy()
    ep['net_pnl']=ep.net_pnl.map(fmt);ep.columns=['代码','开仓日期','平仓日期','持有自然日','完整持仓净利润']
    worst=e.nsmallest(6,'net_2026')[['stock_id','open_date','close_date','holding_calendar_days','net_2026']].copy()
    worst['net_2026']=worst.net_2026.map(fmt);worst.columns=['代码','开仓日期','平仓日期','持有自然日','2026净贡献']
    latest=leaders.reset_index();latest.columns=['代码','2025净贡献','2026净贡献']
    for k in latest.columns[1:]:latest[k]=latest[k].map(fmt)
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
    <title>周换1只：年度收益归因</title><style>
    body{{font-family:"Microsoft YaHei",sans-serif;background:#f1f5f7;color:#183243;margin:0;line-height:1.7}}main{{max-width:1250px;margin:auto;padding:34px}}
    h1{{font-size:32px}}h2{{margin-top:35px;border-bottom:2px solid #c4d5df;padding-bottom:8px}}.card{{background:white;border-radius:12px;padding:24px;margin:20px 0}}
    .hint{{background:#fff2d9;padding:15px;border-left:4px solid #d29c48}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px 11px;border-bottom:1px solid #e0e8ec;text-align:right;white-space:nowrap}}th{{background:#eaf1f5}}th:first-child,td:first-child{{text-align:left}}img{{width:100%;height:auto}}a{{color:#176d98}}small{{color:#526b79}}details{{margin:18px 0}}summary{{cursor:pointer;font-weight:bold}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.stat{{font-size:25px;color:#245b75}}@media(max-width:700px){{main{{padding:15px}}.grid{{grid-template-columns:1fr}}}}
    </style><main><small>已有结果的只读归因 · 2026-10-29 · 不新增训练 / 预测 / 回测</small>
    <h1>周换1只：高收益依赖盈利尾部，2026年新开仓拖累明显</h1>
    <p>全年2025扣费后收益为91.19%，2026截至10月15日为1.26%。相同截止日比较仍是92.68%对1.26%，所以差异不能用“2026还没走完一年”解释。</p>
    <div class="grid"><div class="card">2025前三只占净利润<div class="stat">53.74%</div>前三只合计14.70万元</div><div class="card">2026新开持仓净贡献<div class="stat">−48,666元</div>含期末未平仓持仓的浮动盈亏</div><div class="card">2026加回全部费用<div class="stat">+3.22%</div>同一路径现金费用加回，非无成本回测</div></div>
    <div class="card"><h2>1. 收益差异的账面拆分</h2><div class="scroll">{table(pd.DataFrame(annual))}</div>
    <p>2025盈利股票合计赚357,321元，亏损股票合计亏83,750元；2026分别为144,569元和137,317元。2026的盈利几乎全被亏损抵消。按相同截止日和各年期初资金标准化：2025正贡献109.28个百分点、负贡献−16.60个百分点；2026为25.21和−23.94个百分点。</p>
    <p>同期净收益差距91.42个百分点，其中加回费用后的贡献下降92.07个百分点，费用拖累反而减少0.65个百分点。因此主要差异来自持仓涨跌，而非费率上升。</p>
    <img src="diagnostic_overview.png" alt="收益、集中度、月度利润与持仓年龄归因图"></div>
    <div class="card"><h2>2. 是否依赖少数牛股？有明显集中，但不是只靠一只</h2>
    <p>2025共涉及44只股票，27只净贡献为正。前三只贡献53.74%的年度净利润，前五只70.04%，前十只98.91%。若只扣除第一只的贡献，剩余账面收益仍有69.89%；若扣除前十只，则只剩1.00%。可见结果依赖一组大赢家形成的盈利尾部。</p>
    <div class="scroll">{table(ct)}</div><p class="hint">上述“扣除贡献”只是对已发生账户利润的算术拆分，未重新配置资金、改变持仓或节省费用，不能当成剔除这些股票后的策略回测。股票也是事后按贡献排序，不能事前识别这些赢家。</p>
    <h3>2025主要盈利股票</h3><div class="scroll">{table(stocktable('2025'))}</div>
    <p>2025前三只为603185.SH、688411.SH、688313.SH，分别贡献63,903元、47,965元、35,156元。2025年7—8月两个月贡献135,673元，占全年净利润49.59%。利润在个股和时间上都集中。</p>
    <details><summary>2025前十只在2026的贡献</summary>{table(latest)}<p>0表示本策略在该年没有对应持仓盈亏，不代表该股票市场涨跌为0。2025前三大赢家2026均未持有；600867.SH由2025赚21,962元转为2026亏24,821元。未持有不等于漏买：本诊断没有比较未持有期间的行情收益。</p></details>
    <h3>2026盈利和亏损股票</h3><div class="scroll">{table(stocktable('2026_to_1015'))}</div><div class="scroll">{table(stocktable('2026_to_1015',True))}</div>
    <p>2026有18只盈利、18只亏损。前五大盈利贡献90,660元，占全部正贡献62.71%；净利润分母仅7,252元，因此不用“占净利润1250%”判断集中度。扣除第一大盈利股票的贡献后账面净利润已为−19,129元。</p></div>
    <div class="card"><h2>3. 跨年拆分：2026的新开仓没有接上2025的盈利</h2>
    <p>2025年末带入的5笔持仓，按2025年末市值划分，其<strong>2026年内</strong>净贡献为55,918元；2026年新开的所有持仓合计−48,666元，两者相加就是7,252元。前者并非把2025已赚的钱重复计入2026。</p>
    <div class="scroll">{table(ca)}</div><p>跨年贡献主要来自000951.SZ和603233.SH，2026年内分别贡献25,424元、25,958元。按股票合并后的000951.SZ贡献为26,380元，还包含2026另外一次交易。</p></div>
    <div class="card"><h2>4. 持仓时间：2025留住赢家，2026亏损持仓反而更久</h2><div class="scroll">{table(pd.DataFrame(durations))}</div>
    <p>两年已平仓持仓的中位数均为28个自然日。2025盈利持仓平均39.1天、亏损持仓20.9天；2026变为盈利31.1天、亏损41.6天。盈利笔平均净回报由13.07%降至5.21%，亏损笔平均净回报由−5.06%变为−5.55%。2026的问题同时体现在胜率下降、赢家幅度缩小和亏损持仓拖延。</p>
    <p class="hint">这张表按卖出年份归类完整持仓，跨年持仓会包含前一年部分盈亏；它用于观察持仓行为，不用于计算年度收益。年度收益始终使用逐日盯市归因。持仓长短与盈亏相关，不能据此推断“多持有几天必然增加收益”。</p>
    <h3>完整盈利持仓示例</h3><div class="scroll">{table(ep)}</div><h3>2026亏损持仓示例</h3><div class="scroll">{table(worst)}</div>
    <p>按每个交易日当时的持仓年龄归因，2025第29天以后贡献188,290元（全年净利润68.83%）；2026第29天以后仍贡献101,841元，但前28天合计−94,589元，尤其第8—28天贡献−86,875元。长持仓有盈利也有亏损，例如002557.SZ持有140天亏13,714元、600927.SH持有98天亏11,097元；现有结果不支持简单地把持仓期限缩短或延长作为修复。</p></div>
    <div class="card"><h2>5. 交易成本：放大弱收益，但不是年度落差的主因</h2><div class="scroll">{table(pd.DataFrame(fee))}</div>
    <p>保持既有费率：佣金单边万3.5、每笔最低5元；卖出印花税万5；双边过户费万0.1；滑点单边0.05%。2026成交80笔，少于2025全年107笔，但资金规模更大，成交总额1,010.79万元，高于2025的954.12万元，费用因此略高。</p>
    <p>2026同路径毛利润18,474元，扣除11,223元费用后剩7,252元，费用吃掉毛利润60.75%；其中滑点5,054元，佣金和税费6,169元。即使全部费用加回，收益也仅3.22%。相同截止日看，费用占期初资金的拖累从2.61%降为1.96%，无法解释收益大幅下降。</p></div>
    <div class="card"><h2>6. 对这个战法的判断</h2>
    <p>现有样本表明，周换1只的优势是减少换手并给赢家延续上涨的时间；2025一组较大的盈利持仓兑现了这种优势。2026新开仓表现较弱、赢家幅度收窄、亏损持仓更久，导致此前优势没有稳定延续。它可以继续作为低换手研究候选，但93.61%的全区间收益不能作为每年都能盈利的证据。</p>
    <p>本次只确定了账户收益来自哪里，尚未识别市场风格变化、模型预测质量变化各自的因果作用。后续应优先持续观察新开仓贡献、胜率、盈亏幅度与亏损持仓时长，不能凭这两年结果事后挑掉亏损股票或直接认定某个止损阈值有效。</p></div>
    <div class="card"><h2>口径、核对与文件</h2>
    <p>逐股逐日净利润 = 当日持仓市值变化 + 卖出成交额 − 买入成交额 − 实际费用。年度贡献除以各年期初净值。所有430个交易日的个股利润合计、持仓市值合计和费用合计均与原账户核对通过；2025年末持仓按年末市值分段。原始输入SHA-256在分析前后相同。</p>
    <p>佣金、印花税、过户费、滑点沿用既有回测的现金扣减实现；“加回费用”是同路径金额归因，不是资金再投资后的零费用复利回测。持仓年龄为自然日，个股持仓交易日为年度多次持有的合计。未额外验证原始行情和历史可交易性；原回测统一涨跌停阈值、复权数量及股票池历史可得性等既有限制仍适用。</p>
    <ul><li><a href="period_summary.csv">年度与同期收益、费用和持仓统计 CSV</a></li><li><a href="stock_year_attribution.csv">全部个股年度贡献 CSV</a></li>
    <li><a href="holding_episodes.csv">96笔完整或期末未平仓持仓明细 CSV</a></li><li><a href="daily_stock_attribution.csv">逐日逐股贡献 CSV</a></li>
    <li><a href="holding_age_attribution.csv">按当日持仓年龄归因 CSV</a></li><li><a href="monthly_attribution.csv">月度收益归因 CSV</a></li><li><a href="concentration.csv">集中度与算术扣除贡献 CSV</a></li><li><a href="cross_year_episodes.csv">跨年持仓分段明细 CSV</a></li><li><a href="verification.json">核对结果与来源哈希</a></li></ul>
    <small>数据来源：artifacts/weekly_strategy/20261029_all_boards_slip5bp_v1/weekly_drop1。保存到新诊断目录，未覆盖原结果。</small></div></main></html>'''
    (P/'report.html').write_text(content,encoding='utf-8')
    (P/'report_source.py').write_bytes(Path(__file__).read_bytes())
    # Validate local references and preserve a manifest of all diagnostic outputs.
    import re
    for ref in re.findall(r'(?:href|src)="([^"]+)"',content): assert (P/ref).is_file(),ref
    checks=json.loads((P/'verification.json').read_text(encoding='utf-8'))
    for f,sha in checks['source_sha256'].items():
        assert hashlib.sha256((Path(checks['source'])/f).read_bytes()).hexdigest()==sha
    (P/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.iterdir() if p.is_file()},indent=2),encoding='utf-8')
    print(str(P/'report.html'))

if __name__=='__main__': main()
