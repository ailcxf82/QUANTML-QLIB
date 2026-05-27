"""诊断 daily_t1_open 实验结果差的原因。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "python"))


def main() -> None:
    import qlib
    from qlib.data import D
    from qlib.data.dataset.loader import QlibDataLoader

    uri = "D:/qlib_data/qlib_data_train_20260516"
    qlib.init(provider_uri=uri, region="cn")

    sample = ["000001.SZ", "000002.SZ", "600000.SH", "600519.SH", "000858.SZ"]
    for field in ["$close_qfq", "$open_qfq"]:
        df = D.features(sample, [field], start_time="2026-04-01", end_time="2026-05-22", freq="day")
        print(f"\n=== {field} === shape={df.shape} nan%={df.isna().mean().mean():.4f}")
        print(df.tail(6))

    for name, expr in [
        ("close_label", "Ref($close_qfq,-2)/Ref($close_qfq,-1)-1"),
        ("open_label", "Ref($open_qfq,-2)/Ref($open_qfq,-1)-1"),
    ]:
        loader = QlibDataLoader(config={"feature": [], "label": [[expr], ["LABEL0"]]})
        df = loader.load(instruments="csi500", start_time="2026-04-01", end_time="2026-05-22")
        col = df.iloc[:, 0]
        print(f"\n=== {name} csi500 === rows={len(df)} nan%={col.isna().mean()*100:.2f}%")
        if len(df) > 0:
            dates = df.index.get_level_values("datetime").unique()
            print(f"  date range: {dates.min()} ~ {dates.max()} ({len(dates)} days)")

    # 日历最后交易日
    from qlib.data.data import Cal
    cal = Cal.calendar(freq="day", future=False)
    print(f"\n=== calendar last dates ===")
    print(cal[-10:])

    # 长 OOS 区间数据覆盖（应对齐 daily.yaml test，而非 monthly 短 test）
    df_long = D.features(sample, ["$open_qfq", "$close_qfq"], start_time="2025-01-01", end_time="2026-04-30", freq="day")
    last = df_long.groupby(level="instrument").apply(lambda x: x.index.get_level_values("datetime").max())
    print(f"\n=== long OOS 2025-01~2026-04 per-instrument last date (sample) ===")
    print(last)

    # 对比 run 产物
    for rid in [
        "lgbm_optuna_daily_t1_open_143e3614",
        "lgbm_optuna_daily_retrain_4de4eb8b",
    ]:
        p = _REPO / "artifacts" / rid
        if not p.exists():
            continue
        print(f"\n=== artifact {rid} ===")
        with (p / "metrics.json").open(encoding="utf-8") as f:
            m = json.load(f)
        met = m["metrics"]
        t = m.get("timing_seconds", {})
        print(f"  sharpe={met.get('sharpe_ratio')} ann_ret={met.get('annualized_return')}")
        print(f"  ic_mean={met.get('ic_mean'):.4f} per_seed_ic={t.get('ensemble_per_seed_ic')}")
        s = pd.read_parquet(p / "signals.parquet").iloc[:, 0]
        d = s.index.get_level_values("datetime")
        print(f"  signals: {d.min().date()} ~ {d.max().date()} ({d.nunique()} days)")


if __name__ == "__main__":
    main()
