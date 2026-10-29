"""Download missing frozen-replay inputs into a private, append-only data bundle.

Never changes the shared provider. Only missing cells after the known source
cutoffs are filled. Financial additions are available AFTER their announcement;
this does not certify the historical provider's financial data as point-in-time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

FIN = ['roe', 'q_eps', 'q_profit_yoy']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def financial_daily(records, calendar):
    """Most recent disclosed period, with revisions effective after announcement.

    Ambiguous same-day values are excluded, never chosen by API row order.
    An older period's subsequent revision cannot replace a newer period.
    """
    records = records.drop_duplicates().copy()
    records['ann_date'] = pd.to_datetime(records.ann_date, format='%Y%m%d', errors='coerce')
    records['end_date'] = pd.to_datetime(records.end_date, format='%Y%m%d', errors='coerce')
    valid = records.ann_date.notna() & records.end_date.notna() & (records.ann_date >= records.end_date)
    valid &= records.ann_date < calendar.max()
    rejected = records.loc[~valid].copy()
    records = records.loc[valid].copy()
    key = ['ts_code', 'ann_date', 'end_date']
    conflicts = records.groupby(key)[FIN].nunique(dropna=False).max(axis=1)
    bad = conflicts[conflicts > 1].index
    ambiguous = pd.MultiIndex.from_frame(records[key]).isin(bad)
    rejected = pd.concat([rejected, records.loc[ambiguous]])
    records = records.loc[~ambiguous].drop_duplicates(key).sort_values(key)
    result = {}
    for code, group in records.groupby('ts_code', sort=False):
        latest_period = pd.Timestamp.min
        events = []
        for row in group.itertuples():
            if row.end_date < latest_period:
                continue
            latest_period = row.end_date
            index = calendar.searchsorted(row.ann_date, side='right')
            if index < len(calendar):
                events.append([calendar[index], row.roe, row.q_eps, row.q_profit_yoy])
        if events:
            frame = pd.DataFrame(events, columns=['datetime']+FIN).drop_duplicates('datetime', keep='last').set_index('datetime')
            # Reindex with ffill carries the event row, including its real nulls.
            result[code] = frame.reindex(calendar, method='ffill')
    return result, rejected


def read_bin(path, length):
    values = np.full(length, np.nan, dtype='<f4')
    if path.exists():
        raw = np.fromfile(path, dtype='<f4')
        if len(raw):
            start = int(raw[0])
            end = min(length, start+len(raw)-1)
            values[start:end] = raw[1:1+end-start]
    return values


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--provider', type=Path, default=Path('D:/qlib_data/qlib_data'))
    parser.add_argument('--resume', action='store_true', help='Reuse immutable downloaded pages in this bundle')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    cache = out/'downloads'
    cache.mkdir(exist_ok=args.resume)
    calendar = pd.DatetimeIndex(pd.to_datetime((args.provider/'calendars/day.txt').read_text().splitlines()))
    calendar = calendar[calendar <= '2026-10-15']
    token = os.environ.get('TUSHARE_API_KEY') or os.environ.get('TUSHARE_TOKEN')
    if not token:
        raise ValueError('Tushare credential is unavailable')
    import tushare as ts
    pro = ts.pro_api(token, timeout=60)
    receipt = out/'api_calls.jsonl'

    def fetch(api, **params):
        pages = []
        seen = set()
        for offset in range(0, 100000, 5000):
            tag = '_'.join(str(v) for v in params.values())
            path = cache/f'{api}_{tag}_{offset}.parquet'
            if path.exists():
                frame = pd.read_parquet(path)
            else:
                start = time.monotonic()
                try:
                    frame = getattr(pro, api)(**params, limit=5000, offset=offset)
                except Exception as exc:
                    raise RuntimeError(str(exc).replace(token, '[REDACTED]')) from None
                frame.to_parquet(path, index=False)
                with receipt.open('a', encoding='utf-8') as f:
                    f.write(json.dumps({'api': api, 'params': params, 'offset': offset, 'rows': len(frame), 'sha256': sha(path), 'seconds': time.monotonic()-start})+'\n')
                time.sleep(.35)
            if frame.empty:
                break
            fingerprint = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).values.tobytes()).hexdigest()
            if fingerprint in seen:
                raise ValueError(f'{api} pagination did not advance')
            seen.add(fingerprint)
            pages.append(frame)
            if len(frame) < 5000:
                break
        else:
            raise ValueError('API pagination bound exhausted')
        result = pd.concat(pages, ignore_index=True) if pages else pd.DataFrame()
        print(api, params, 'rows', len(result), flush=True)
        return result

    financial = pd.concat([fetch('fina_indicator_vip', period=p,
        fields='ts_code,ann_date,end_date,roe,q_eps,q_profit_yoy,update_flag')
        for p in ['20250930', '20251231', '20260331', '20260630', '20260930']], ignore_index=True)
    financial.to_parquet(out/'financial_source.parquet', index=False)
    daily, rejected = financial_daily(financial, calendar)
    rejected.to_parquet(out/'financial_rejected.parquet', index=False)
    flows = []
    # One overlap day validates units and the original source before extension.
    for date in calendar[calendar >= '2026-07-09']:
        frame = fetch('moneyflow_dc', trade_date=date.strftime('%Y%m%d'), fields='ts_code,trade_date,net_amount')
        if frame.empty or frame.trade_date.nunique() != 1 or str(frame.trade_date.iloc[0]) != date.strftime('%Y%m%d'):
            raise ValueError(f'Invalid/empty moneyflow date {date}')
        if frame.ts_code.duplicated().any():
            raise ValueError('Duplicate moneyflow source keys')
        flows.append(frame)
    flow = pd.concat(flows, ignore_index=True)
    flow['datetime'] = pd.to_datetime(flow.trade_date, format='%Y%m%d')
    flow.to_parquet(out/'moneyflow_source.parquet', index=False)
    flow_groups = {k: g.set_index('datetime').net_amount.reindex(calendar).to_numpy() for k,g in flow.groupby('ts_code')}
    bins = out/'provider/features'
    bins.mkdir(parents=True, exist_ok=args.resume)
    records, originals, overlap = [], {}, []
    for folder in sorted((args.provider/'features').iterdir()):
        if not folder.is_dir():
            continue
        instrument = folder.name.lower()
        if len(instrument) != 9 or instrument[-2:] not in ['sh','sz','bj']:
            continue
        code = instrument.upper()
        for field in FIN+['net_amount']:
            path = folder/f'{field}.day.bin'
            old = read_bin(path, len(calendar))
            if path.exists():
                originals[str(path)] = sha(path)
            if field == 'net_amount':
                added = flow_groups.get(code, np.full(len(calendar), np.nan))
                same = (calendar == '2026-07-09') & np.isfinite(old) & np.isfinite(added)
                if same.any():
                    overlap.append(float(np.max(np.abs(old[same]-added[same].astype('<f4')))))
                eligible = calendar > '2026-07-09'
            else:
                added = daily[code][field].to_numpy() if code in daily else np.full(len(calendar), np.nan)
                eligible = calendar > '2026-03-31'
            mask = eligible & ~np.isfinite(old) & np.isfinite(added)
            new = old.copy()
            new[mask] = added[mask]
            if mask.any():
                target = bins/instrument/f'{field}.day.bin'
                target.parent.mkdir(exist_ok=True)
                if target.exists():
                    raise ValueError('Overlay already exists; use a fresh output for rebuilding')
                np.r_[np.float32(0), new].astype('<f4').tofile(target)
            records.append({'instrument': instrument, 'field': field, 'filled_cells': int(mask.sum()),
                            'last_day_available': bool(np.isfinite(new[-1]))})
    if len(overlap) < 4000 or max(overlap) > .01:
        raise ValueError(f'Moneyflow overlap failed: {len(overlap)} rows, max diff {max(overlap,default=-1)}')
    pd.DataFrame(records).to_csv(out/'repair_coverage.csv',index=False)
    if any(sha(p) != h for p,h in originals.items()):
        raise ValueError('Original feature files changed')
    (out/'original_sha256.json').write_text(json.dumps(originals,indent=2),encoding='utf-8')
    manifest = {str(p.relative_to(out)): sha(p) for p in bins.rglob('*.bin')}
    (out/'overlay_sha256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    summary = {'status': 'complete', 'original_files_unchanged': True, 'overlay_bins': len(manifest),
               'financial_rows':len(financial), 'financial_rejected_rows':len(rejected),
               'moneyflow_rows':len(flow), 'moneyflow_overlap_rows':len(overlap), 'moneyflow_overlap_max_abs_diff':max(overlap),
               'filled_cells': pd.DataFrame(records).groupby('field').filled_cells.sum().to_dict(),
               'policy':'Only missing cells after financial 2026-03-31 / moneyflow 2026-07-09; financial events effective next trading session. Original historical financial cells preserved, not certified PIT; vendor historical revision vintages not frozen.',
               'moneyflow_units':'10,000 CNY, same moneyflow_dc endpoint as original',
               'sources':['https://tushare.pro/document/2?doc_id=79','https://tushare.pro/document/2?doc_id=349']}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
