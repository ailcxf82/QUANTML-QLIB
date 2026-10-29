"""Predict with a frozen 39-feature model and run one isolated, cost-aware replay.

No fit/finetune or model promotion. The exact external factor implementation is
identified by comparing its historical output to the model's archived input.
All new files live in a new output directory; the shared provider is read-only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'python'))
FACTOR_NAME = 'VolRegimeResidMom_90D'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')


def calculate_external(close):
    """Same arithmetic/order as archived factor 556b1289...; input is adjusted close."""
    close = close.sort_index()
    # Explicit ffill reproduces the archived pandas pct_change default.
    returns = close.groupby(level='instrument').ffill().groupby(level='instrument').pct_change(fill_method=None)
    residual = returns - returns.groupby(level='datetime').transform('mean')
    mom = residual.groupby(level='instrument').transform(lambda x: x.rolling(90, min_periods=90).sum())
    short = returns.groupby(level='instrument').transform(lambda x: x.rolling(20, min_periods=20).std())
    long = returns.groupby(level='instrument').transform(lambda x: x.rolling(120, min_periods=120).std())
    ratio_rank = (short / long).groupby(level='datetime').rank(pct=True)
    weighted = mom * (1 - ratio_rank)
    smooth = weighted.groupby(level='instrument').transform(lambda x: x.rolling(20, min_periods=20).mean())
    return smooth.groupby(level='instrument').shift(1).rename(FACTOR_NAME)


def source_stats(provider, fields):
    """Record size/mtime of all relevant bins; content hashes of generated inputs follow."""
    result = {}
    for folder in sorted((provider / 'features').iterdir()):
        if folder.is_dir():
            for field in sorted(fields):
                path = folder / f'{field}.day.bin'
                if path.is_file():
                    stat = path.stat()
                    result[str(path.relative_to(provider))] = [stat.st_size, stat.st_mtime_ns]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-run', type=Path, default=ROOT/'artifacts/lgbm_optuna_daily_f2c798bb')
    parser.add_argument('--factor-workspace', type=Path, required=True)
    parser.add_argument('--data-repair', type=Path, help='Completed isolated repair bundle')
    parser.add_argument('--start', default='2025-01-01')
    parser.add_argument('--end', default='2026-10-15')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    ledger = {'status': 'preparing', 'training_runs': 0, 'prediction_runs': 0, 'backtest_runs': 0,
              'source_run': str(args.source_run.resolve()), 'requested_start': args.start, 'requested_end': args.end}

    def stage(name):
        ledger['status'] = name
        ledger['elapsed_seconds'] = round(time.monotonic() - started, 2)
        write_json(out/'run_ledger.json', ledger)
        print(f'[{name}] {ledger["elapsed_seconds"]}s', flush=True)

    try:
        from features import build_feature_config
        from features.combined_json_factors import merge_features_from_combined_json
        from run_experiment import load_and_merge_configs
        from backtest.engine import BacktestEngine
        from backtest.analysis_layer import AnalysisLayer
        from backtest.portfolio_layer import PortfolioLayer
        import qlib
        from qlib.data import D
        from qlib.data.dataset.loader import QlibDataLoader
        from qlib.utils import init_instance_by_config

        cfg = load_and_merge_configs('lgbm_optuna', 'daily', None)
        provider = Path(cfg['qlib_init']['provider_uri'])
        sources = [args.source_run/'model.pkl', args.source_run/'model_meta.json',
                   args.source_run/'signals.parquet', args.source_run/'backtest_report.parquet',
                   args.source_run/'metrics.json', args.factor_workspace/'factor.py',
                   args.factor_workspace/'result.h5', ROOT/'configs/base.yaml',
                   ROOT/'configs/models/lgbm_optuna.yaml', ROOT/'configs/freq/daily.yaml',
                   ROOT/'configs/combined_factors_df.json', provider/'calendars/day.txt',
                   provider/'instruments/all.txt', provider/'instruments/csi500.txt']
        sources += list((ROOT/'artifacts/models').glob('latest_*'))
        source_hashes = {str(p.resolve()): digest(p) for p in sources if p.is_file()}
        overlay_config = {}
        repair_summary = None
        if args.data_repair:
            repair = args.data_repair.resolve()
            repair_summary = json.loads((repair/'summary.json').read_text(encoding='utf-8'))
            if repair_summary['status'] != 'complete':
                raise ValueError('Data repair is not complete')
            original_repair_hashes = json.loads((repair/'original_sha256.json').read_text(encoding='utf-8'))
            for path, expected_hash in original_repair_hashes.items():
                if digest(path) != expected_hash:
                    raise ValueError('Original provider changed since repair: '+path)
            overlay_hashes = json.loads((repair/'overlay_sha256.json').read_text(encoding='utf-8'))
            for relative, expected_hash in overlay_hashes.items():
                path = repair/relative
                if digest(path) != expected_hash:
                    raise ValueError('Data repair hash mismatch: '+relative)
                source_hashes[str(path)] = expected_hash
            for name in ['summary.json', 'overlay_sha256.json', 'original_sha256.json']:
                source_hashes[str(repair/name)] = digest(repair/name)
            overlay_config = {'feature_provider': {'class': 'OverlayFeatureProvider',
                'module_path': 'features.overlay_provider', 'kwargs': {'overlay': str(repair/'provider')}}}
            write_json(out/'data_repair_summary.json', repair_summary)
        meta = json.loads((args.source_run/'model_meta.json').read_text(encoding='utf-8'))
        source_external = Path(meta['external_feature_parquets'][0])
        source_hashes[str(source_external)] = digest(source_external)
        write_json(out/'source_sha256.json', source_hashes)
        shutil.copy2(args.source_run/'model.pkl', out/'frozen_model.pkl')
        shutil.copy2(args.source_run/'model_meta.json', out/'source_model_meta.json')
        shutil.copy2(args.factor_workspace/'factor.py', out/'source_external_factor.py')
        model = joblib.load(out/'frozen_model.pkl')
        if model.model.num_feature() != 39:
            raise ValueError('This runner requires the identified 39-feature frozen model')
        model_text_hash = hashlib.sha256(model.model.model_to_string().encode()).hexdigest()
        reference = pd.read_pickle(source_external).iloc[:, 0].dropna()
        original = pd.read_hdf(args.factor_workspace/'result.h5', key='data').iloc[:, 0].dropna()
        common = reference.index.intersection(original.index)
        if len(common) != len(reference) or not np.array_equal(reference.loc[common].values, original.loc[common].values):
            raise ValueError('Factor implementation provenance does not match archived model input')
        ledger['factor_provenance_exact_rows'] = len(common)
        del reference, original, common
        gc.collect()

        # A private future calendar supplies only the next step boundary; no quote
        # or return is invented for this sentinel, and historical day.txt is intact.
        calendar = pd.DatetimeIndex(pd.to_datetime((provider/'calendars/day.txt').read_text().splitlines()))
        end = pd.Timestamp(args.end)
        if end not in calendar:
            raise ValueError('Requested end is not in the observed provider calendar')
        private = out/'calendar_provider'
        (private/'calendars').mkdir(parents=True)
        shutil.copy2(provider/'calendars/day.txt', private/'calendars/day.txt')
        future = calendar
        if calendar.max() == end:
            boundary = end + pd.Timedelta(days=1)
            future = calendar.append(pd.DatetimeIndex([boundary]))
            ledger['calendar_boundary_only'] = str(boundary.date())
        (private/'calendars/day_future.txt').write_text('\n'.join(str(d.date()) for d in future)+'\n', encoding='utf-8')
        qlib.init(provider_uri=str(provider), region='cn', kernels=1, expression_cache=None, dataset_cache=None,
                  **overlay_config,
                  calendar_provider={'class': 'LocalCalendarProvider', 'kwargs': {'backend': {
                      'class': 'FileCalendarStorage', 'module_path': 'qlib.data.storage.file_storage',
                      'kwargs': {'provider_uri': str(private)}}}})
        exprs, names = build_feature_config()
        merge_features_from_combined_json(exprs, names, ROOT/cfg['model_meta']['combined_factors_json'])
        fields = set(re.findall(r'\$([A-Za-z_0-9]+)', ' '.join(exprs))) | {'close_qfq', 'close', 'open', 'factor', 'volume', 'change'}
        before_stats = source_stats(provider, fields)
        write_json(out/'provider_file_stats_before.json', before_stats)
        stage('rebuilding_external_factor')
        # The archived daily_pv uses forward-adjusted prices; use the provider's
        # explicit adjusted field, not its raw $close. Recompute with current data
        # throughout, rather than splice incompatible historical adjustment vintages.
        close = D.features(D.instruments('all'), ['$close_qfq'], start_time='2022-01-04', end_time=args.end)
        close = close.swaplevel().sort_index().iloc[:, 0]
        external = calculate_external(close)
        external.to_frame().to_parquet(out/'external_factor_recomputed.parquet')
        del close
        gc.collect()
        stage('building_inference_features')
        raw = QlibDataLoader(config={'feature': [exprs, names],
                                     'label': [['Ref($close_qfq,-2)/Ref($close_qfq,-1)-1'], ['LABEL0']]}, freq='day').load('all', args.start, args.end)
        raw[('feature', FACTOR_NAME)] = external.reindex(raw.index)
        del external
        # NestedDataLoader sorted columns when this model was trained.
        raw = raw.sort_index(axis=1)
        expected_names = sorted(names + [FACTOR_NAME])
        if list(raw['feature'].columns) != expected_names:
            raise ValueError('Feature ordering mismatch with training loader contract')
        coverage = raw['feature'].notna().groupby(level='datetime').mean()
        coverage.to_csv(out/'feature_coverage.csv')
        raw.to_parquet(out/'raw_inference_inputs.parquet')
        if (coverage.loc[end] == 0).any() or coverage.loc[end, FACTOR_NAME] < .8:
            raise ValueError('Incomplete last-day feature coverage')
        if args.data_repair:
            repaired_names = ['NET_AMT', 'earnings_growth_smooth', 'quarterly_eps', 'roe_momentum_60d', 'roe_pb_ratio']
            if (coverage.loc[end, repaired_names] < .8).any():
                raise ValueError('Repaired features cover fewer than 80% of last-day instruments')
        labels = raw[('label', 'LABEL0')].copy().rename('label')
        for processor in cfg['dataset']['kwargs']['handler']['kwargs']['infer_processors']:
            raw = init_instance_by_config(processor)(raw)
        x = raw['feature']
        if not np.isfinite(x.to_numpy()).all():
            raise ValueError('Non-finite inference features after original processors')
        ledger['prediction_runs'] = 1
        stage('predicting_frozen_model')
        pred = pd.Series(model.model.predict(x.values, num_threads=4), index=x.index, name='score')
        pred.to_frame().to_parquet(out/'signals_all.parquet')
        write_json(out/'feature_schema.json', {'names_in_model_order': expected_names,
                    'n_features': len(expected_names), 'inference_processors': cfg['dataset']['kwargs']['handler']['kwargs']['infer_processors'],
                    'external_price_field': '$close_qfq', 'factor_calculation_start': '2022-01-04'})
        del raw, x
        gc.collect()
        # Preserve the old backtest's union-over-window universe convention.
        universe = D.list_instruments(D.instruments('csi500'), start_time=args.start, end_time=args.end, as_list=True)
        if not universe:
            raise ValueError('Empty investment universe; refusing fallback to all')
        pred = pred.loc[pred.index.get_level_values('instrument').isin(universe)]
        expected_dates = calendar[(calendar >= pd.Timestamp(args.start)) & (calendar <= end)]
        if not expected_dates.equals(pd.DatetimeIndex(pred.index.get_level_values('datetime').unique()).sort_values()):
            raise ValueError('Prediction dates do not cover all requested sessions')
        pred.to_frame().to_parquet(out/'signals.parquet')
        labels = labels.reindex(pred.index)
        labels.to_frame().to_parquet(out/'labels.parquet')
        universe_frame = pd.read_csv(provider/'instruments/csi500.txt', sep='\t', header=None, names=['instrument','start','end'])
        write_json(out/'universe.json', {'mode': 'legacy_union_over_test_window', 'n_instruments': len(universe),
                                       'membership_last_date': universe_frame.end.max(), 'instruments': universe})
        old = pd.read_parquet(args.source_run/'signals.parquet').iloc[:, 0]
        common = old.index.intersection(pred.index)
        overlap = {'n_rows': len(common), 'correlation': float(old.loc[common].corr(pred.loc[common])),
                   'mean_absolute_score_difference': float((old.loc[common]-pred.loc[common]).abs().mean())}
        write_json(out/'historical_prediction_comparison.json', overlap)
        cfg['experiment']['run_id'] = out.name
        cfg['experiment']['end_time'] = args.end
        cfg['dataset']['kwargs']['segments']['test'] = [args.start, args.end]
        cfg['dataset']['kwargs']['handler']['kwargs']['end_time'] = args.end
        cfg['replay'] = {'source_model_sha256': digest(out/'frozen_model.pkl'), 'training_runs': 0,
                         'external_factor_source': str(args.factor_workspace), 'private_calendar': str(private),
                         'data_repair': str(args.data_repair) if args.data_repair else None}
        (out/'resolved_config.yaml').write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
        ledger['backtest_runs'] = 1
        stage('backtest_started')
        engine = BacktestEngine()
        result = engine.run(pred, cfg, output_dir=out, label=labels)
        report = result.report_df
        if not report.index.equals(expected_dates):
            raise ValueError('Backtest did not include every requested session through end date')
        for name, frame in [('backtest_report', report), ('indicator', result.indicator_df),
                            ('trades', result.trades_df), ('realized_pnl', result.realized_pnl_df)]:
            frame.to_parquet(out/f'{name}.parquet')
        expected_account = float(cfg['experiment']['account']) * report.cum_return
        if not np.allclose(expected_account, report.account, rtol=1e-10, atol=1e-5):
            raise ValueError('Net return curve does not reconcile to Qlib account value')
        if not np.isclose(result.trades_df.cost.sum(), report.total_cost.iloc[-1], rtol=1e-10, atol=1e-5):
            raise ValueError('Trade costs do not reconcile to Qlib total cost')
        # Never transform gross figures into an imaginary zero-cost trading run.
        gross_state = PortfolioLayer().parse({'1day': report.assign(cost=0.0)})
        net_metrics = engine.metrics_to_dict(result)
        gross_metrics = asdict(AnalysisLayer().compute_metrics(gross_state))
        pd.DataFrame([dict(basis='gross_same_trades', **gross_metrics),
                      dict(basis='net_of_transaction_costs', **net_metrics)]).to_csv(out/'cost_comparison.csv', index=False)
        notes = [
            'Fixed historical model, no fit or finetune. Features recomputed from current provider; this is not an untouched prospective OOS claim.',
            'Preserved legacy csi500 union-over-window selection, not daily point-in-time membership; membership data ends '+str(universe_frame.end.max())+'.',
            'Preserved configured benchmark 000905.SZ. This is not the CSI500 index 000905.SH; index-relative metrics must not be interpreted as CSI500 excess returns.',
            'Provider factor fields are missing for some instruments; existing Qlib adjusted-price/fractional-share behavior is preserved, not certified live execution.',
            'Last two forward labels may be unavailable; portfolio returns through requested end are evaluated independently of label availability.',
        ]
        if repair_summary:
            notes.append('Missing financial/moneyflow inputs filled from an isolated Tushare bundle. '+repair_summary['policy'])
        write_json(out/'metrics.json', {'run_id': out.name, 'source_run': args.source_run.name,
            'return_basis': 'net_of_transaction_costs', 'actual_start': str(report.index.min().date()),
            'actual_end': str(report.index.max().date()), 'n_days': len(report), 'metrics': net_metrics,
            'gross_metrics_same_trades': gross_metrics, 'cash_cost_total': float(report.total_cost.iloc[-1]),
            'ending_account': float(report.account.iloc[-1]), 'limitations': notes})
        changed = [path for path, h in source_hashes.items() if digest(path) != h]
        after_stats = source_stats(provider, fields)
        if before_stats != after_stats:
            raise ValueError('Provider input file metadata changed during replay')
        if changed or digest(out/'frozen_model.pkl') != source_hashes[str((args.source_run/'model.pkl').resolve())]:
            raise ValueError(f'Protected input hashes changed: {changed}')
        if hashlib.sha256(model.model.model_to_string().encode()).hexdigest() != model_text_hash:
            raise ValueError('In-memory model changed')
        # Put limitations prominently in the generated report, not only in metadata.
        import html
        report_path = out/'report.html'
        page = report_path.read_text(encoding='utf-8')
        rows = ''.join('<tr><td>'+label+'</td><td>'+fmt.format(gross_metrics[key])+'</td><td>'+fmt.format(net_metrics[key])+'</td></tr>'
            for key,label,fmt in [('total_return','累计收益','{:.2%}'),('annualized_return','年化收益','{:.2%}'),
                                  ('max_drawdown','最大回撤','{:.2%}'),('sharpe_ratio','夏普比率','{:.3f}')])
        notice = '<section style="margin:24px;padding:20px;background:#fff2d5;color:#312200"><h2>冻结 LightGBM · 扣费核对</h2>'
        notice += f'<p>{report.index.min().date()} 至 {args.end}，共 {len(report)} 个交易日。训练 0 次，预测 1 次，回测 1 次。</p>'
        notice += '<table style="width:100%;text-align:left"><thead><tr><th>指标</th><th>同笔交易扣费前</th><th>扣费后</th></tr></thead><tbody>'+rows+'</tbody></table>'
        notice += f'<p>初始资金 300,000.00 元；期末账户 {report.account.iloc[-1]:,.2f} 元；累计现金交易费用 {report.total_cost.iloc[-1]:,.2f} 元。日净收益 = Qlib return − cost；净值与逐日账户一致，费用与逐笔成交一致。</p>'
        notice += '<p>扣费前数字为同笔交易的毛收益序列，不能理解为重新执行了一次零成本策略。沿用买入费率 0.35%、卖出费率 0.45%、最低单笔 5 元的原配置。</p>'
        notice += '<p>新报告使用补齐后的当前数据重新预测，与旧报告的差异包含数据版本与新增区间影响。原基准配置 000905.SZ 不是中证500指数；下方基准/超额指标不可作为中证500比较结论。股票池仍沿用区间成分集合，非逐日成分股。</p>'
        if repair_summary:
            notice += '<p>补齐值来自独立数据目录。财务新增值按公告后下一交易日生效，已有历史非空值保留；这不代表原历史财务数据与修订版本已经完整通过时点审计。</p>'
        notice += '<details><summary>完整口径与局限</summary><ul>'+''.join('<li>'+html.escape(n)+'</li>' for n in notes)+'</ul></details></section>'
        page = page.replace('<body>', '<body>'+notice, 1)
        report_path.write_text(page, encoding='utf-8')
        write_json(out/'postflight.json', {'source_sha256_unchanged': True, 'provider_file_stats_unchanged': True,
            'provider_files_checked': len(before_stats), 'model_unchanged': True, 'net_account_reconciliation': True,
            'trade_cost_reconciliation': True, 'training_runs': 0, 'backtest_runs': 1,
            'artifact_sha256': {p.name: digest(p) for p in out.iterdir() if p.is_file() and p.name != 'run_ledger.json'}})
        stage('complete')
        print(json.dumps({'output': str(out), 'metrics': net_metrics, 'ending_account': float(report.account.iloc[-1])}, ensure_ascii=False), flush=True)
    except Exception as exc:
        ledger['error'] = f'{type(exc).__name__}: {exc}'
        stage('failed')
        raise


if __name__ == '__main__':
    main()
