import importlib.util
from pathlib import Path

import pandas as pd
import numpy as np

spec = importlib.util.spec_from_file_location('repair', Path(__file__).resolve().parents[3]/'scripts/repair_replay_data.py')
repair = importlib.util.module_from_spec(spec)
spec.loader.exec_module(repair)


def test_financial_announcements_and_revisions():
    calendar = pd.bdate_range('2026-04-01', '2026-04-15')
    rows = pd.DataFrame([
        ['X', '20260402', '20251231', 1, 1, 1, '1'],
        ['X', '20260406', '20260331', 2, 2, 2, '1'],
        ['X', '20260407', '20251231', 9, 9, 9, '1'],
        ['X', '20260420', '20260331', 8, 8, 8, '1'],
        ['X', '20260403', '20260630', 7, 7, 7, '1'],
    ], columns=['ts_code','ann_date','end_date']+repair.FIN+['update_flag'])
    daily, rejected = repair.financial_daily(rows, calendar)
    frame = daily['X']
    assert pd.isna(frame.loc['2026-04-02', 'roe'])
    assert frame.loc['2026-04-03', 'roe'] == 1
    assert frame.loc['2026-04-06', 'roe'] == 1
    assert (frame.loc['2026-04-07':, 'roe'] == 2).all()
    assert len(rejected) == 2


def test_overlay_routes_repaired_field_and_keeps_original_prices(tmp_path):
    import qlib
    from qlib.data import D
    base, overlay = tmp_path/'base', tmp_path/'overlay'
    (base/'calendars').mkdir(parents=True)
    (base/'calendars/day.txt').write_text('2026-10-13\n2026-10-14\n2026-10-15\n')
    for root in [base, overlay]:
        (root/'features/000001.sz').mkdir(parents=True)
    np.array([0, 1, 2, 3],dtype='<f4').tofile(base/'features/000001.sz/close.day.bin')
    np.array([0, 4, 5, 6],dtype='<f4').tofile(base/'features/000001.sz/roe.day.bin')
    np.array([0, 4, 50, 60],dtype='<f4').tofile(overlay/'features/000001.sz/roe.day.bin')
    qlib.init(provider_uri=str(base), region='cn', kernels=1, expression_cache=None, dataset_cache=None,
        feature_provider={'class':'OverlayFeatureProvider', 'module_path':'features.overlay_provider',
                          'kwargs':{'overlay':str(overlay)}})
    result = D.features(['000001.SZ'], ['$close', '$roe', '$roe-Ref($roe,1)'], '2026-10-14', '2026-10-15')
    np.testing.assert_allclose(result.values, [[2,50,46],[3,60,10]])


def test_ambiguous_same_day_values_are_not_selected():
    rows = pd.DataFrame([
        ['X', '20260402', '20251231', 1, 1, 1],
        ['X', '20260402', '20251231', 2, 2, 2],
    ], columns=['ts_code','ann_date','end_date']+repair.FIN)
    daily, rejected = repair.financial_daily(rows, pd.bdate_range('2026-04-01','2026-04-15'))
    assert daily == {}
    assert len(rejected) == 2
