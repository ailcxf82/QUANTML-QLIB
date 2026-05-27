"""列出 artifacts/ 下匹配模式的 run，标注健康度，按 OOS 覆盖范围排序。

用法：
    python scripts/_pick_run.py --pattern lgbm_optuna_daily_retrain_
    python scripts/_pick_run.py --pattern lgbm_optuna_daily_retrain_ --healthy-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent


def health_status(metrics: dict, timing: dict) -> tuple[bool, str]:
    """根据 metrics.json 判定 run 健康度。

    返回 (健康?, 原因描述)
    """
    issues = []
    ann_ret = metrics.get("annualized_return", 0.0)
    win_rate = metrics.get("win_rate", 0.0)
    sharpe = metrics.get("sharpe_ratio")
    per_seed_ic = timing.get("ensemble_per_seed_ic", []) if timing else []

    if ann_ret == 0.0 and metrics.get("annualized_volatility", 0.0) == 0.0:
        issues.append("零下单(ann_ret=0)")
    if win_rate == 0.0:
        issues.append("win_rate=0")
    if sharpe is None or (isinstance(sharpe, float) and sharpe != sharpe):  # NaN check
        issues.append("sharpe=NaN")
    if per_seed_ic and all(ic < 0 for ic in per_seed_ic):
        issues.append(f"per_seed_ic 全负({per_seed_ic})")

    if not issues:
        return True, "OK"
    return False, "; ".join(issues)


def collect(pattern: str) -> list[dict]:
    rows = []
    for run_dir in (_REPO_ROOT / "artifacts").glob(f"{pattern}*"):
        if not run_dir.is_dir():
            continue
        concat_sig = run_dir / "concatenated" / "signals.parquet"
        sig = run_dir / "signals.parquet"
        if concat_sig.exists():
            sig_path, mode = concat_sig, "WF"
            metrics_path = run_dir / "concatenated" / "metrics.json"
        elif sig.exists():
            sig_path, mode = sig, "single"
            metrics_path = run_dir / "metrics.json"
        else:
            continue

        df = pd.read_parquet(sig_path)
        s = df.iloc[:, 0] if hasattr(df, "iloc") else df
        dates = s.index.get_level_values("datetime")
        n_days = dates.nunique()

        metrics, timing = {}, {}
        if metrics_path.exists():
            try:
                with metrics_path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                metrics = raw.get("metrics", raw) if "metrics" in raw else raw
                timing = raw.get("timing_seconds", {})
            except Exception:
                pass

        healthy, reason = health_status(metrics, timing)
        rows.append({
            "run_id": run_dir.name,
            "mode": mode,
            "n_days": n_days,
            "date_range": f"{dates.min().date()}~{dates.max().date()}",
            "sharpe": metrics.get("sharpe_ratio"),
            "ann_ret": metrics.get("annualized_return"),
            "win_rate": metrics.get("win_rate"),
            "mtime": run_dir.stat().st_mtime,
            "healthy": healthy,
            "reason": reason,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="lgbm_optuna_daily_retrain_")
    parser.add_argument("--healthy-only", action="store_true")
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()

    rows = collect(args.pattern)
    if args.healthy_only:
        rows = [r for r in rows if r["healthy"]]
    rows.sort(key=lambda r: (r["n_days"], r["mtime"]), reverse=True)

    print(f"\n{'run_id':<42} {'mode':>6} {'days':>5} {'date_range':>22} "
          f"{'sharpe':>8} {'ann_ret':>8} {'win':>6}  status")
    print("-" * 120)
    for r in rows[:args.top]:
        sharpe = f"{r['sharpe']:.3f}" if isinstance(r['sharpe'], (int, float)) and r['sharpe'] == r['sharpe'] else "n/a"
        ann = f"{r['ann_ret'] * 100:.1f}%" if isinstance(r['ann_ret'], (int, float)) else "n/a"
        win = f"{r['win_rate'] * 100:.0f}%" if isinstance(r['win_rate'], (int, float)) else "n/a"
        flag = "OK" if r["healthy"] else "BAD"
        print(f"{r['run_id']:<42} {r['mode']:>6} {r['n_days']:>5} {r['date_range']:>22} "
              f"{sharpe:>8} {ann:>8} {win:>6}  [{flag}] {r['reason']}")


if __name__ == "__main__":
    main()
