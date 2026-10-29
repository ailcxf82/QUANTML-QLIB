"""Render the existing-score exit-signal diagnostic; no backtest routines."""
from pathlib import Path
import json
import hashlib
import re
import html
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, dates as mdates

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'artifacts/weekly_diagnostics/20261029_drop1_exit_signals_v1'

def main():
    assert not (P/'report.html').exists(), 'Do not overwrite a completed report'
    def read(n):return pd.read_csv(P/f'{n}.csv')
    e=read('episode_rank_summary');e['entry_year']=pd.to_datetime(e.open_date).dt.year
    d=read('daily_rank_trajectory');d.datetime=pd.to_datetime(d.datetime)
    r=read('weekly_rank_reviews');a=read('first_exit_alerts');stats=read('alert_summary')
    q=read('model_quality_summary');en=read('entry_forward_summary');states=read('review_state_pnl')
    cases=['600867.SH_20260420','002557.SZ_20260511','600927.SH_20260601','300573.SZ_20260810']
    fonts={f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams['font.sans-serif']=[next((f for f in ['Microsoft YaHei','SimHei','Noto Sans CJK SC'] if f in fonts),'DejaVu Sans')]
    plt.rcParams['axes.unicode_minus']=False
    fig,axs=plt.subplots(2,2,figsize=(16,10),layout='constrained')
    fig.suptitle('亏损持仓：有的排名早已转弱，有的周检排名长期靠前',fontsize=19)
    for ax,eid in zip(axs.flat,cases):
        z=d[d.episode_id==eid];w=z[z.is_review];ep=e[e.episode_id==eid].iloc[0]
        ax.plot(z.datetime,z.pool_rank,color='#b8b4aa',lw=1,label='每日可用排名',alpha=.9)
        ax.plot(w.datetime,w.pool_rank,'o-',color='#b45e3e',markersize=4,lw=1.5,label='周检排名')
        ax.axhline(20,color='#956bac',linestyle='--',lw=1,label='第20名')
        ax.set_yscale('log');ax.set_ylim(650,.8);ax.set_yticks([1,5,20,100,600],[1,5,20,100,600])
        ax.set_ylabel('股票池排名（越上越好，对数轴）',fontsize=10)
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4,maxticks=6));ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax2=ax.twinx();ax2.plot(z.datetime,z.gross_return_since_entry*100,color='#1e6484',lw=2,label='买入以来价格涨跌')
        ax2.axhline(0,color='#d5d5d5',lw=.8);ax2.set_ylabel('价格涨跌（%，未扣费用）',fontsize=10)
        ax.set_title(f'{ep.stock_id}  {ep.open_date[5:]}—{ep.last_date[5:]}  净利润 {ep.net_pnl:,.0f}元',fontsize=12)
        ax.set_xlabel('2026年可执行日（排名使用前一交易日预测）',fontsize=9)
        if eid==cases[0]:
            handles,labels=ax.get_legend_handles_labels();h2,l2=ax2.get_legend_handles_labels()
            ax.legend(handles+h2,labels+l2,fontsize=8,loc='lower left')
    fig.savefig(P/'rank_and_price_cases.png',dpi=160);plt.close(fig)
    def money(v):return '—' if pd.isna(v) else f'{v:,.0f}'
    def pct(v):return '—' if pd.isna(v) else f'{v:.2%}'
    def num(v):return '—' if pd.isna(v) else f'{v:.0f}'
    def table(df):return df.to_html(index=False,border=0,classes='data',escape=True)
    quality=[]
    for freq in ['daily','weekly_review']:
        for year in [2025,2026]:
            row=q[(q.year==year)&(q.window=='through_1015')&(q.signal_sample==freq)].iloc[0]
            quality.append({'同截止日样本':'全部交易日' if freq=='daily' else '实际周检日','年份':year,'样本日数':int(row.n),
               '平均RankIC':f'{row.mean_rank_ic:.4f}','Top5次日均值':pct(row.top5_mean_1d_return),
               '同池次日均值':pct(row.pool_mean_1d_return),'Top5减同池':pct(row.top5_minus_pool)})
    entries=[]
    for n in [1,5,20]:
        for yr in [2025,2026]:
            row=en[(en.year==yr)&(en.window=='through_1015')&(en.horizon==n)].iloc[0]
            entries.append({'年份':yr,'买入后交易日':n,'完整观测笔数':int(row.n),'所买股票平均涨跌':pct(row.mean_return),
                 '同批股票池平均涨跌':pct(row.mean_pool_return),'相对同池差值':pct(row.mean_excess),'上涨比例':pct(row.positive_fraction)})
    reasonmap={'entry_day_fees':'买入当日费用','entry_before_first_review':'买入后至首次周检',
      'rank_still_top20':'最近周检仍在前20名且可交易','weak_rank_waiting_behind_other_sale':'最近周检已掉出前20，名额用于更弱持仓',
      'untradable_at_review':'最近周检不可交易（不等于原本应卖出）'}
    st=states[states.entry_year==2026].copy()
    st['状态']=st.review_state.map(reasonmap);st['后续实际价格损益']=st.gross_pnl.map(money);st['实际费用']=st.cost.map(money);st['扣费后贡献']=st.net_pnl.map(money)
    rules={'rank20':'周检跌出前20','rank20_twice':'连续两次周检跌出前20','rank20_bottom2':'跌出前20且为持仓倒数两名','rank20_daily_timing_probe':'每日跌出前20（频率对照）'}
    alerttab=[]
    for rule,title in rules.items():
        z=stats[(stats.entry_year==2026)&(stats.state=='closed_only')&(stats.rule==rule)].iloc[0]
        alerttab.append({'警报':title,'提前触发笔数':int(z.alerted_n),'覆盖亏损笔':f'{int(z.losers_alerted)}/19',
              '触及盈利笔':f'{int(z.winners_alerted)}/16','警报后跌/涨':f'{int(z.subsequent_fall_n)}/{int(z.subsequent_rise_n)}',
              '警报后平均涨跌':pct(z.mean_subsequent_return),'提前自然日中位数':num(z.median_days_before_end)})
    loss=e[e.entry_year==2026].nsmallest(12,'net_pnl')
    losstable=pd.DataFrame({'代码':loss.stock_id,'开仓':loss.open_date,'净利润':loss.net_pnl.map(money),
       '入场排名':loss.entry_rank.map(num),'首次提前周度警报':loss.first_rank20_alert.fillna('无'),
       '警报排名':loss.first_alert_rank.map(num),'当时累计价格涨跌':loss.return_at_first_alert.map(pct),
       '警报后至退出价格涨跌':loss.return_after_first_alert.map(pct),'名额等待次数':loss.quota_wait_reviews.map(num)})
    details=[]
    for _,ep in e[e.entry_year==2026].sort_values('net_pnl').iterrows():
        z=d[(d.episode_id==ep.episode_id)&d.is_review].copy()
        def state(row):
            if not row.existing_holding:return '买入'
            return {'actual_exit':'实际卖出','rank_still_top20':'排名仍在前20','weak_rank_waiting_behind_other_sale':'较弱，名额用于其他股票','untradable_at_review':'当日不可交易'}.get(row.retention_reason,row.retention_reason)
        tab=pd.DataFrame({'周检日期':z.datetime.dt.strftime('%Y-%m-%d'),'预测日期':z.signal_date,
            '全池排名':z.pool_rank.map(num),'持仓内排名':z.held_rank.map(num),'累计价格涨跌':z.gross_return_since_entry.map(pct),
            '决策状态':z.apply(state,axis=1),'当日卖出的其他股票':z.sold_other.fillna('—')})
        details.append(f'<details class="episode" data-code="{ep.stock_id}"><summary>{ep.stock_id}｜{ep.open_date} → {ep.last_date}｜净利润 {ep.net_pnl:,.0f}元｜{"已平仓" if ep.closed else "期末未平仓"}</summary><div class="scroll">{table(tab)}</div></details>')
    compare=pd.read_csv(ROOT/'artifacts/weekly_strategy/20261029_all_boards_slip5bp_v1/period_comparison.csv')
    co=compare[compare.period=='2026_to_1015'].copy()
    ct=pd.DataFrame({'既有策略':co.strategy,'2026扣费后收益':co.net_return.map(pct),'2026费用':co.cost.map(money)})
    (P/'existing_strategy_comparison.csv').write_bytes(co.to_csv(index=False).encode('utf-8-sig'))
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>2026新开仓：选股与退出信号诊断</title>
    <style>body{{font-family:"Microsoft YaHei",sans-serif;background:#f2f5f7;color:#173143;line-height:1.75;margin:0}}main{{max-width:1320px;margin:auto;padding:30px}}h1{{font-size:32px}}h2{{margin-top:30px}}.card{{background:white;border-radius:12px;padding:24px;margin:20px 0}}.hint{{background:#fff3df;border-left:4px solid #d6a352;padding:14px}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px;border-bottom:1px solid #dfe6eb;text-align:right;white-space:nowrap}}th{{background:#edf3f6}}th:first-child,td:first-child{{text-align:left}}img{{width:100%;height:auto}}a{{color:#17698f}}small{{color:#5e7684}}details{{margin:14px 0}}summary{{cursor:pointer;font-weight:600}}input{{padding:12px;font-size:16px;width:300px;max-width:90%;border:1px solid #b5c8d3;border-radius:6px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.big{{font-size:25px;color:#1e6484}}@media(max-width:700px){{main{{padding:12px}}.grid{{grid-template-columns:1fr}}}}</style>
    <main><small>2026-10-29 · 仅使用保存的预测、持仓与报价 · 新增训练/预测/回测均为0</small><h1>选股表现减弱与退出迟缓同时存在，排名警报尚不足以直接卖出</h1>
    <p>2026年40笔新开仓合计净亏48,666元。所有买入时的预测排名都在第1—4名，问题并非买入了低排名股票。后续有些股票排名明显变差却排队等待退出，另一些亏损时模型仍持续给出较高排名；每日预测与每周交易之间也存在时间不匹配。</p>
    <div class="grid"><div class="card">实际周检日平均RankIC<div class="big">0.0974 → 0.0206</div>2025和2026均以10月15日为截止</div><div class="card">2026新开仓亏损中的警报覆盖<div class="big">14 / 19笔</div>已平仓亏损笔中，提前周检跌出前20</div><div class="card">同一警报对2025赢家的干扰<div class="big">18 / 22笔后续上涨</div>2025年内开平仓的提前警报样本</div></div>
    <div class="card"><h2>1. 模型是否变差？每周交易时点的表现减弱更明显</h2><div class="scroll">{table(pd.DataFrame(quality))}</div>
    <p>RankIC是预测排序与次日实际涨跌排序的相关性。全体交易日从0.0444降到0.0377，仍为正；实际周检对应时点从0.0974降到0.0206，Top5相对同池的平均次日表现从+0.35个百分点变为−0.16个百分点。2026的周度取样没有稳定承接模型的日频排序优势，但40次周检不足以证明这种差异会持续。</p>
    <p>模型训练目标是“信号日之后第1个交易日收盘，到第2个交易日收盘”的单日收益，并非未来一个月收益。不能把入场排名第1理解为该股票未来几周都值得持有。</p>
    <h3>实际买入股票的固定期限表现</h3><div class="scroll">{table(pd.DataFrame(entries))}</div>
    <p>2026买入后20个交易日平均跌0.91%，但同批可观测股票池也平均跌0.75%，相对差值只有−0.16个百分点。较长持有期的绝对表现变差，部分来自这些入场时点对应的股票池整体走弱，不能全部归因于模型选股。实际买入股票5日相对同池仍为正，也不支持“模型全面失效”的结论。</p>
    <p class="hint">以上为无费用的价格事件研究，不是持仓收益或新回测。固定期限内即使原策略已卖出，也继续观察原股票价格。每个期限只保留截至各年10月15日已有完整价格的样本；同池是保存预测覆盖的股票等权均值，不是沪深300或中证500基准。样本重叠且有关联，不作统计显著性结论。</p></div>
    <div class="card"><h2>2. 退出为什么迟缓？分别看名额、周频取样和高排名误判</h2>
    <p>原规则把当前5只持仓与一个新候选比较，每周最多卖掉其中排名最差的可交易持仓；没有“掉出前20必须卖”的规定。93次周检的卖出选择均可由保存的预测、原持仓与原交易约束解释，未发现卖出逻辑与既有规则不一致。</p>
    <img src="rank_and_price_cases.png" alt="四笔主要亏损持仓每日排名、周检排名与累计价格涨跌">
    <p><strong>名额限制确实延误了部分股票退出。</strong>600867.SH在4月20日以第3名买入；4月27日已降至44名，股价仅较买入低0.99%，但它在持仓内排第3，当周卖掉的是更弱的603882.SH。随后周检排名为149、54，直到5月18日持仓内排末位才卖出。从第一次警报到退出，价格又跌13.33%。600129.SH在5月11日跌至58名时仍浮盈2.38%，同样因另有更弱持仓而保留，之后至退出又跌9.13%。</p>
    <p><strong>另一些亏损，周检排名本身报警太晚。</strong>002557.SZ的5月11日开仓持有140天，20次持仓周检中16次仍在前20；首次提前周度警报直到8月31日，累计已跌13.54%，之后至退出只再跌1.50%。600927.SH持有98天，14次周检中12次仍在前20；首次提前警报时已跌6.80%，后续再跌1.09%。这两笔的主要跌幅并不是“排名早已很差却一直没卖”。</p>
    <p><strong>每周检查还会漏过中途走弱。</strong>300573.SZ在8月10日买入排名第2；8月11日可用的前一日预测已降到31名，当时股价较买入上涨0.23%，此后至退出跌11.13%。但后两次周检排名又回到20、12，直到卖出日才降到70。这属于周频取样与日频排名波动的问题，不能都归为“只换1只”的名额问题。</p>
    <h3>主要亏损持仓的提前警报</h3><div class="scroll">{table(losstable)}</div>
    <p>“无”只表示在实际退出之前没有符合这条周度警报的可交易观察日，不表示每天都没有警报。警报当天卖出的持仓不计为提前识别。</p>
    <h3>按最近一次周检状态拆分随后实际盈亏</h3><div class="scroll">{table(st[['状态','后续实际价格损益','实际费用','扣费后贡献']])}</div>
    <p>2026新开仓出现46次“已跌出前20、但名额用于另一只股票”的持仓周检观察，相关后续区间净贡献−12,938元；最近周检仍在前20的后续区间净贡献−28,514元。另有一次当日不可交易的观察，但该股票当时仍排全池第5、持仓第2，不能把随后亏损认定为无法卖出造成。</p>
    <p class="hint">状态从周检日收盘后生效，周检当天的价格变化归入此前状态，避免使用当天收盘价格事后归因。表中各项加总为−48,666元；这是实际路径的状态分组，不是模型与退出机制各自的因果损失，也不是可以直接挽回的金额。</p></div>
    <div class="card"><h2>3. 有没有可提前识别的退出信号？有预警，但误报不可忽略</h2>
    <p>固定检查三条周度警报，不做阈值择优：①跌出前20；②连续两次周检跌出前20；③跌出前20且处于现有持仓倒数两名。另用同一阈值做每日观察，单独检查频率影响。均使用前一交易日预测，每笔只统计第一次早于实际退出、且符合原交易约束的警报。</p>
    <p>以下主表仅包含2026新开且已平仓的35笔，其中19笔亏损、16笔盈利；另5笔未平仓的结果单列保存在CSV中。</p><div class="scroll">{table(pd.DataFrame(alerttab))}</div>
    <p>单次周检掉出前20能提前覆盖14/19笔亏损，也会触及5/16笔盈利；19个警报中12个之后继续跌、7个之后涨，平均后续涨跌仅−0.37%。连续两次确认并没有更好：12个警报里8个之后上涨，平均上涨2.65%。每日检查则对35笔已平仓持仓全部报警，几乎失去区分作用。</p>
    <p>包括期末未平仓在内，单次周检掉出前20对40笔中的20笔报警，13笔后续下跌、7笔上涨。固定数量下这些下跌对应−76,248元价格损益，上涨对应+56,997元，合计−19,251元；这只是警报后的原持仓价格变化，未处理提前卖出费用、替代买入、现金和后续仓位路径，不能据此宣称新规则能多赚19,251元。</p>
    <h3>必须同时看到的2025反例</h3>
    <p>2025年内开平仓的51笔中，同一周度警报提前触发22笔，其中18笔之后上涨。例如688411.SH的6月30日持仓，在7月7日周检已降至199名，此后到原退出日仍涨45.30%；688313.SH在4月14日周检降至48名，此后到退出涨66.56%；603185.SH的7月28日持仓，8月4日降至144名，此后仍涨30.20%。</p>
    <p><strong>因此现有证据支持“排名恶化值得提示”，尚不支持“排名恶化就自动清仓”。</strong>只看2026亏损案例会得到很诱人的卖出规则，但同样规则可能破坏2025的主要盈利来源。</p></div>
    <div class="card"><h2>4. 已有更快换仓结果也不能证明放宽名额就能解决</h2><div class="scroll">{table(ct)}</div>
    <p>这些结果均来自之前已完成的回测，本次只读取。2026周换2只为−0.15%，周换5只为−16.20%，均没有超过周换1只的+1.26%。不过它们的买入股票、资金路径和费用均不同，不能当成“只加快退出、其他保持不变”的实验。</p>
    <p>当前可保留的结论是：存在退出排队案例，也存在高排名持续误判及周中警报被周频检查漏过；没有一条已验证、能兼顾两年赢家和亏损控制的排名退出规则。可以把“当前排名、持仓内排名、连续走弱次数、累计价格涨跌”列为周度审查信息。本次不修改策略，也不把前20名阈值升级为自动卖出条件。</p></div>
    <div class="card"><h2>5. 2026全部40笔新开仓的周检排名明细</h2><p>按净利润从低到高排列；持仓内排名1最好，5最差。排名池每日覆盖636—641只有效预测。</p><input id="filter" aria-label="按证券代码筛选" placeholder="输入证券代码筛选，例如 600867"><div id="episodes">{''.join(details)}</div></div>
    <div class="card"><h2>计算与核对</h2>
    <p>预测只读自signals.parquet；持仓、成交、报价均来自原weekly_drop1目录。所有可执行日均使用上一交易日的预测，未使用未来预测触发警报。未来价格仅作为警报后的评价结果。模型标签与保存价格重算标签一致；93次周检卖出选择归因全部吻合；状态分组盈亏与原持仓净利润核对一致；输入哈希在处理前后不变。</p>
    <p>“可交易”仅复用原回测统一9.5%阈值与报价有效性限制，不代表真实盘口一定可成交。未重新审计股票池历史可得性、复权股数与其他原回测限制。已平仓样本与未平仓样本分开；2025入场组跨年平仓的扩展统计仍保存在CSV，正文反例统计限定2025年内开平仓。所有阈值是诊断假设，未在独立样本上验证。</p>
    <ul><li><a href="episode_rank_summary.csv">每笔持仓排名与首次周度警报摘要 CSV</a></li><li><a href="weekly_rank_reviews.csv">逐次周检排名、卖出股票与保留原因 CSV</a></li><li><a href="daily_rank_trajectory.csv">逐日排名与价格轨迹 CSV</a></li><li><a href="first_exit_alerts.csv">所有首次警报及后续涨跌 CSV</a></li><li><a href="alert_summary.csv">警报覆盖与误报对照 CSV</a></li><li><a href="model_quality_summary.csv">模型排序表现对照 CSV</a></li><li><a href="entry_forward_returns.csv">实际新开仓固定期限表现 CSV</a></li><li><a href="review_state_pnl.csv">按周检状态拆分盈亏 CSV</a></li><li><a href="protocol.json">诊断口径</a> · <a href="verification.json">核对结果及来源哈希</a></li></ul>
    </div></main><script>document.getElementById('filter').addEventListener('input',e=>{{const q=e.target.value.trim().toUpperCase();document.querySelectorAll('.episode').forEach(el=>{{el.hidden=!el.dataset.code.includes(q);if(q&&!el.hidden)el.open=true;}});}});</script></html>'''
    (P/'report.html').write_text(content,encoding='utf-8')
    (P/'report_source.py').write_bytes(Path(__file__).read_bytes())
    for ref in re.findall(r'(?:href|src)="([^"]+)"',content):assert (P/ref).is_file(),ref
    v=json.loads((P/'verification.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==digest for f,digest in v['source_sha256'].items())
    (P/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in P.iterdir() if p.is_file() and p.name!='manifest.json'},indent=2),encoding='utf-8')
    print(P/'report.html')

if __name__=='__main__':main()
