"""
Build daily 5-minute aggregate feature packages for the LGBM daily pipeline.

Input:
  D:/qlib_data/csv_data_5min/<INSTRUMENT>.csv with columns:
  symbol,date,open,high,low,close,volume,amount

Output:
  artifacts/intraday_agg/<name>/features.parquet
  artifacts/intraday_agg/<name>/meta.json

The parquet index is MultiIndex(datetime, instrument), so it can be injected into
run_experiment.py via --external-feature-parquet without changing the daily
Qlib feature registry or base.yaml.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


EXPECTED_BARS_PER_DAY = 48
FEATURE_COLUMNS = [
    "M5_OPEN30_RET",
    "M5_TAIL30_RET",
    "M5_AM_RET",
    "M5_PM_RET",
    "M5_RV",
    "M5_VOL_AM_RATIO",
    "M5_AMT_AM_RATIO",
    "M5_CLOSE_VWAP_DEV",
    "M5_HIGH_TIME_IDX",
    "M5_LOW_TIME_IDX",
    "M5_GAP_OPEN",
    "M5_INTRADAY_RANGE",
]


def _safe_div(a: float, b: float) -> float:
    if b == 0 or not np.isfinite(b):
        return np.nan
    return float(a / b)


def _features_for_symbol(path: Path, expected_bars: int) -> tuple[pd.DataFrame, Dict[str, int]]:
    df = pd.read_csv(path)
    required = {"symbol", "date", "open", "high", "low", "close", "volume", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

    df["date"] = pd.to_datetime(df["date"])
    df["day"] = df["date"].dt.normalize()
    df = df.sort_values(["symbol", "date"], kind="stable")
    symbol = str(df["symbol"].iloc[0])

    rows: List[dict] = []
    stats = {"days_total": 0, "days_used": 0, "days_bad_bar_count": 0}
    prev_close = np.nan

    for day, g in df.groupby("day", sort=True):
        stats["days_total"] += 1
        if len(g) != expected_bars:
            stats["days_bad_bar_count"] += 1
            # Keep prev_close from complete days only, so gap features are auditable.
            continue

        o = g["open"].to_numpy(dtype=float)
        h = g["high"].to_numpy(dtype=float)
        l = g["low"].to_numpy(dtype=float)
        c = g["close"].to_numpy(dtype=float)
        v = g["volume"].to_numpy(dtype=float)
        amt = g["amount"].to_numpy(dtype=float)

        if np.any(~np.isfinite(o)) or np.any(~np.isfinite(c)) or np.any(c <= 0):
            continue
        if np.any(v < 0) or np.any(amt < 0):
            continue

        n = len(g)
        first6 = min(6, n)
        last6_start = max(0, n - 6)
        am_end = n // 2
        vwap = _safe_div(float(np.nansum(amt)), float(np.nansum(v)))
        bar_ret = c[1:] / c[:-1] - 1.0

        rows.append(
            {
                "datetime": pd.Timestamp(day),
                "instrument": symbol,
                "M5_OPEN30_RET": _safe_div(c[first6 - 1], o[0]) - 1.0,
                "M5_TAIL30_RET": _safe_div(c[-1], o[last6_start]) - 1.0,
                "M5_AM_RET": _safe_div(c[am_end - 1], o[0]) - 1.0,
                "M5_PM_RET": _safe_div(c[-1], o[am_end]) - 1.0,
                "M5_RV": float(np.sqrt(np.nansum(np.square(bar_ret)))),
                "M5_VOL_AM_RATIO": _safe_div(float(np.nansum(v[:am_end])), float(np.nansum(v))),
                "M5_AMT_AM_RATIO": _safe_div(float(np.nansum(amt[:am_end])), float(np.nansum(amt))),
                "M5_CLOSE_VWAP_DEV": _safe_div(c[-1], vwap) - 1.0,
                "M5_HIGH_TIME_IDX": float(np.nanargmax(h) / max(1, n - 1)),
                "M5_LOW_TIME_IDX": float(np.nanargmin(l) / max(1, n - 1)),
                "M5_GAP_OPEN": _safe_div(o[0], prev_close) - 1.0 if np.isfinite(prev_close) else np.nan,
                "M5_INTRADAY_RANGE": _safe_div(float(np.nanmax(h)), float(np.nanmin(l))) - 1.0,
            }
        )
        prev_close = float(c[-1])
        stats["days_used"] += 1

    out = pd.DataFrame(rows)
    return out, stats


def build(
    csv_dir: Path,
    out_dir: Path,
    expected_bars: int,
    min_instruments: int,
    *,
    dry_run: bool = False,
) -> dict:
    files = sorted(csv_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    frames: List[pd.DataFrame] = []
    per_symbol: Dict[str, Dict[str, int]] = {}
    for path in files:
        frame, stats = _features_for_symbol(path, expected_bars)
        if not frame.empty:
            frames.append(frame)
            per_symbol[path.stem] = stats

    if len(frames) < min_instruments:
        raise RuntimeError(
            f"Only {len(frames)} instruments produced features; required >= {min_instruments}"
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.set_index(["datetime", "instrument"]).sort_index()
    df = df[FEATURE_COLUMNS].astype("float32")

    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "features.parquet"
    if not dry_run:
        try:
            df.to_parquet(parquet_path)
        except ImportError as exc:
            raise RuntimeError(
                "Writing parquet requires pyarrow or fastparquet. Run this script in "
                "the qlib_zhengshi environment, or use --dry-run for metadata-only validation."
            ) from exc

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "csv_dir": str(csv_dir),
        "parquet": None if dry_run else str(parquet_path),
        "dry_run": dry_run,
        "n_rows": int(len(df)),
        "n_instruments": int(df.index.get_level_values("instrument").nunique()),
        "date_range": [
            str(df.index.get_level_values("datetime").min().date()),
            str(df.index.get_level_values("datetime").max().date()),
        ],
        "expected_bars_per_day": expected_bars,
        "columns": FEATURE_COLUMNS,
        "per_symbol": per_symbol,
        "note": "Features at day T use only T-day 5min bars and are safe for T+1 prediction.",
    }
    (out_dir / ("meta.dry_run.json" if dry_run else "meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Build daily 5min aggregate feature parquet")
    parser.add_argument("--csv-dir", default="D:/qlib_data/csv_data_5min")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--name", default="dev")
    parser.add_argument("--expected-bars", type=int, default=EXPECTED_BARS_PER_DAY)
    parser.add_argument(
        "--min-instruments",
        type=int,
        default=1,
        help="Minimum instruments required. Use a high value for production gates.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Compute metadata but do not write parquet")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path("artifacts") / "intraday_agg" / args.name
    meta = build(
        Path(args.csv_dir),
        out_dir,
        args.expected_bars,
        args.min_instruments,
        dry_run=args.dry_run,
    )
    print(json.dumps({k: meta[k] for k in ("parquet", "dry_run", "n_rows", "n_instruments", "date_range")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
