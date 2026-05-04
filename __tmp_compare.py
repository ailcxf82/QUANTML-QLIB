import json
from pathlib import Path
import sys

rows = []
for p in sorted(Path("artifacts").glob("lgbm_daily_*/metrics.json")):
    with p.open(encoding="utf-8") as f:
        d = json.load(f)
    m = d.get("metrics", {})
    t = d.get("timing_seconds", {})
    rows.append({
        "run_id": d["run_id"][-8:],
        "ds_sec": t.get("dataset_build_sec", 0),
        "train_sec": t.get("train_sec", 0),
        "rc": t.get("rolling_chunks", 0) or 0,
        "ic": m.get("ic_mean", 0),
        "icir": m.get("icir", 0),
        "rank_ic_pos": m.get("ic_positive_ratio", 0),
        "ann_ret": m.get("annualized_return", 0),
        "mdd": m.get("max_drawdown", 0),
        "sharpe": m.get("sharpe_ratio", 0),
        "calmar": m.get("calmar_ratio", 0),
        "ir": m.get("information_ratio", 0),
        "exc_ret": m.get("annualized_excess_return", 0),
        "win": m.get("win_rate", 0),
        "turn": m.get("avg_turnover", 0),
        "mtime": Path(p).stat().st_mtime,
    })

rows.sort(key=lambda x: x["mtime"])

cols = [
    ("run_id", 8, "s"),
    ("ds_sec", 7, ".0f"),
    ("train_sec", 9, ".0f"),
    ("rc", 3, "d"),
    ("ic", 7, ".4f"),
    ("icir", 7, ".4f"),
    ("rank_ic_pos", 6, ".3f"),
    ("ann_ret", 8, ".3f"),
    ("mdd", 8, ".3f"),
    ("sharpe", 7, ".2f"),
    ("ir", 7, ".3f"),
    ("exc_ret", 8, ".3f"),
    ("win", 6, ".3f"),
    ("turn", 6, ".3f"),
]

header = " ".join([f"{c:>{w}}" for c, w, _ in cols])
print(header)
print("-" * len(header))
for r in rows:
    line = " ".join([f"{r[c]:>{w}{f}}" for c, w, f in cols])
    print(line)
