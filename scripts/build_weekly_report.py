"""Build the weekly strategy decision report from saved, verified comparisons."""
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/weekly_strategy/20261029_all_boards_slip5bp_v1'


def main():
    t=pd.read_csv(OUT/'comparison.csv')
    names={r.id:r.strategy for r in t.itertuples()}
    def table(columns):
        df=t[[k for k,v in columns]].copy()
        for k,v in columns:
            if k.endswith('_return') or k in ['total_return','annualized_return','max_drawdown','max_single_stock_weight']:
                df[k]=df[k].map(lambda x:f'{x:.2%}')
            elif k in ['cash_cost','ending_account']:df[k]=df[k].map(lambda x:f'{x:,.2f}')
            elif k=='sharpe_ratio':df[k]=df[k].map(lambda x:f'{x:.3f}')
        return df.rename(columns=dict(columns)).to_html(index=False,border=0,escape=True)
    performance=table([('strategy','策略'),('total_return','累计净收益'),('annualized_return','年化净收益'),
        ('max_drawdown','最大回撤'),('sharpe_ratio','夏普'),('ending_account','期末资产_元')])
    effort=table([('strategy','策略'),('review_days','检查日'),('active_trade_days','交易日'),('orders','成交笔数'),
        ('cash_cost','全部成本_元'),('median_holding_calendar_days','持有自然日中位数'),('max_single_stock_weight','历史最大单票权重')])
    periods=table([('strategy','策略'),('2025_return','2025净收益'),('2026_to_1015_return','2026至10月15日'),
                   ('recent_20260525_to_1015_return','5月25日至10月15日')])
    chosen=t.set_index('id').loc['weekly_drop1'];baseline=t.set_index('id').loc['daily_baseline']
    save={'candidate':'weekly_drop1','reason':'User prefers once per week and fewer trades. Among the three predeclared candidates, drop1 has the fewest orders and lowest costs, strongest full-period and 2026 return; drop2 has smaller drawdown and stronger recent segment.',
          'order_reduction':float(1-chosen.orders/baseline.orders),'cost_reduction':float(1-chosen.cash_cost/baseline.cash_cost),
          'status':'historical_research_candidate_not_live_validated','training_runs':0,'prediction_runs':0,'backtest_runs':3,
          'limitations':['Repeatedly observed historical period, no fresh out-of-sample selection test.',
          'One-session prediction model reused without retraining; weekly holding horizon is an execution experiment.',
          'Inherited non-PIT universe, wrong benchmark code, uniform price limit and adjusted fractional-share assumptions remain.',
          'Holding concentration can drift; drop1 historical maximum single-stock weight 30.59%.',
          '2025 drives most performance; 2026 YTD 1.26%, recent period -4.13%.']}
    (OUT/'candidate_decision.json').write_text(json.dumps(save,ensure_ascii=False,indent=2),encoding='utf-8')
    shutil.copy2(ROOT/'configs/strategy/weekly_top5_drop1.yaml',OUT/'candidate_strategy.yaml')
    doc='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>每周一次：低换手策略对比</title>
<style>body{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;background:#f5f7fa;color:#152c40;margin:24px}main{max-width:1200px;margin:auto}section{background:white;padding:22px;margin:20px 0;border-radius:8px}h1{font-size:28px}h2{font-size:21px}table{border-collapse:collapse;min-width:700px;width:100%;font-variant-numeric:tabular-nums}th,td{padding:10px;border-bottom:1px solid #d5e0e8;text-align:right;white-space:nowrap}th:first-child,td:first-child{text-align:left}th{background:#e8f0f7}.scroll{overflow:auto}a{color:#0864ac}.note{background:#fff3dc;padding:15px}img{width:100%;height:auto}nav{display:flex;flex-wrap:wrap;gap:20px}</style></head><body><main>
<h1>每周检查一次：保留强势股，最多替换1只</h1>
<p>2025-01-02 至 2026-10-15，共430个交易日。恢复原643只候选股票，包含创业板和科创板；持仓5只。复用相同预测与报价，未训练、未生成预测，预先固定3组方案并各回测1次。</p>
<nav><a href="weekly_drop1/investment_details/index.html">候选方案投资明细</a><a href="candidate_strategy.yaml">候选策略配置</a><a href="comparison.csv">完整比较表</a><a href="period_comparison.csv">分段结果</a></nav>
<section><h2>为何选周换1只作为候选</h2><p>它最符合“每周一次、尽量少调仓”的目标：实际交易日92天、成交187笔，而原策略为428天、859笔。总成本由85,073.01元降至21,742.41元。累计净收益93.61%，最大回撤16.63%。周换2只的最大回撤更小（14.43%），但成交367笔、成本41,672.17元。</p>
<p class="note">这不是已验证的实盘收益目标。周换1只在2025年收益91.19%，2026年至10月15日只有1.26%，最近一段为−4.13%。不应按47.28%的历史年化收益外推未来。</p></section>
<section><h2>同日期、同费率的收益比较</h2><div class="scroll">'''+performance+'''</div><img src="comparison.png" alt="四种策略扣费净值对比"></section>
<section><h2>操作次数、持仓时间与成本</h2><div class="scroll">'''+effort+'''</div><p>成本包括佣金、印花税、过户费与滑点假设。佣金双边万3.5（每笔最低5元、含规费）；印花税卖出0.05%；过户费双边0.001%；滑点双边各0.05%。持有天数统计已平仓FIFO批次，单位为自然日。</p></section>
<section><h2>不同阶段是否一致</h2><div class="scroll">'''+periods+'''</div><p>最近一段是2026年数据的子集，不能相加；各段按段首账户资产计算净收益。所有区间均已被观察，只能作为历史稳定性诊断，不能称为独立验证集。</p></section>
<section><h2>候选战法的执行规则</h2><ol>
<li>首次有可用信号时建仓；此后每个ISO周首个交易日检查一次，假期顺延。</li>
<li>读取前一交易日的预测排名，维持最多5只持仓；不剔除创业板和科创板。</li>
<li>在可交易股票中比较当前持仓和外部高分候选，最多卖出1只较弱持仓，并用更高分候选替代；没有符合条件的替代时不换。</li>
<li>其余持仓继续持有，不为了恢复等权而反复买卖。周内不重试、不新增止盈止损交易。</li>
<li>沿用停牌、涨跌停保护与原资金分配规则。95%是原始资金投入参数，并非每日强制95%仓位或单票20%上限。</li>
</ol><p>已平仓持仓的中位持有期约28个自然日；历史最大单票权重30.59%。少换仓可能使权重集中，不能将“五只股票”直接理解成始终均分风险。</p></section>
<section><h2>结论的边界与下一步</h2><p>保留这组简单参数作为下一阶段模拟执行候选，固定每周一次，不继续根据本区间反复调参。需要先用后续未参与选择的时间段检验，并处理既有股票池时点、基准代码、分板块涨跌停及复权小数数量等问题，再评估实际执行。</p>
<p>Qlib官方允许日频估值配合周频下单：<a href="https://github.com/microsoft/qlib/blob/main/docs/component/strategy.rst">策略接口说明</a>。本轮保留全部比较方案，以降低只展示历史赢家的误导；相关方法背景见<a href="https://www.davidhbailey.com/dhbtalks/dhb-jbcc-fin.pdf">Bailey：策略选择与回测过拟合</a>。</p></section>
<section><h2>逐方案报告</h2><ul><li><a href="weekly_drop1/report.html">每周最多替换1只</a></li><li><a href="weekly_drop2/report.html">每周最多替换2只</a></li><li><a href="weekly_drop5/report.html">每周更新前5名</a></li></ul><p>预测与候选池文件逐字节一致，报价逐值一致；所有周策略均验证每周至多一个检查日、成交日期只出现在检查日。178项测试通过，原结果未覆盖。</p></section>
</main></body></html>'''
    (OUT/'report.html').write_text(doc,encoding='utf-8')
    print(json.dumps(save,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
