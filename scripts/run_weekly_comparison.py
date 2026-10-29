"""Three predeclared weekly replays of saved all-board signals; no training."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import yaml

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/weekly_strategy/20261029_all_boards_slip5bp_v1'
SOURCE=ROOT/'artifacts/fee_replay/20261029_lgbm_slip5bp_to_20261015_v1'


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    variants=[{'id':f'weekly_drop{n}','max_replacements':n} for n in [1,2,5]]
    save(OUT/'protocol.json',{
        'baseline':str(SOURCE),'planned_backtest_runs':3,'planned_training_runs':0,'planned_prediction_runs':0,
        'window':['2025-01-02','2026-10-15'],'variants':variants,
        'review_rule':'First usable saved signal initially, then first tradable session of each ISO week; previous-session signal; no intraweek trades.',
        'preserved':'All 643 securities, original score bytes, topk=5, risk_degree=0.95, quote rules, fees and 5bp slippage.',
        'evaluation':'Compare transaction days/count, costs, net returns, drawdown, 2025/2026 and recent segment. No additional parameter search after seeing outcomes.',
        'limitations':'Previously observed historical window, not fresh out-of-sample. Existing one-session model and incomplete execution/PIT assumptions remain.',
        'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            ROOT/'python/backtest/strategy/weekly_topk.py',ROOT/'scripts/replay_saved_predictions.py',SOURCE/'signals.parquet']}})
    for variant in variants:
        cfg={'strategy':{'class':'WeeklyTopkDropoutStrategy','module_path':'backtest.strategy.weekly_topk',
            'kwargs':{'topk':5,'n_drop':variant['max_replacements'],'only_tradable':True,'forbid_all_trade_at_limit':True}}}
        (OUT/f"{variant['id']}.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    ledger={'status':'running','training_runs':0,'prediction_runs':0,'backtest_attempts':0,'runs':[]}
    save(OUT/'campaign_ledger.json',ledger)
    for variant in variants:
        name=variant['id'];ledger['backtest_attempts']+=1
        item={'id':name,'status':'started'};ledger['runs'].append(item);save(OUT/'campaign_ledger.json',ledger)
        with (OUT/f'{name}.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/replay_saved_predictions.py'),
                '--source',str(SOURCE),'--output',str(OUT/name),'--strategy-config',str(OUT/f'{name}.yaml')],
                stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
        item['return_code']=result.returncode
        item['status']='complete' if result.returncode==0 else 'failed'
        save(OUT/'campaign_ledger.json',ledger)
        print(json.dumps(item),flush=True)
        if result.returncode:
            ledger['status']='failed';save(OUT/'campaign_ledger.json',ledger)
            raise RuntimeError(f'{name} failed; no automatic retry')
    ledger['status']='complete';save(OUT/'campaign_ledger.json',ledger)


if __name__=='__main__':main()
