"""按归档时间排序，把 artifacts/lgbm_daily_*/metrics.json 横向对比成宽表。

用途:
    实验闭环阶段 D3 / D4 大量小步迭代时，需要快速对照各 run_id 的
    "信号端 IC/ICIR" 与 "策略端 年化/IR/Sharpe/MDD/换手" 是否同向漂移。
    本脚本不依赖 mlflow，仅读取每个 run 的 metrics.json，输出
    单屏宽表，便于排查"信号涨而策略跌"或"换手率失控"等异常。

执行:
    python python/audit_compare_runs.py

字段:
    ds_sec      dataset_build_sec（数据加载 + processor 流水线总耗时）
    train_sec   model.fit 总耗时（ensemble 模式累加各 seed）
    rc          rolling_chunks（滚动训练段数；0 = 一刀切单切分）
    ic / icir   截面 daily IC 均值与 IR（基于 indicator.parquet）
    ann_ret     年化收益率（策略端，已扣交易成本）
    mdd         最大回撤
    sharpe      年化 Sharpe（rf=0）
    ir          相对 benchmark 的信息比率
    exc_ret     年化超额收益（策略 - benchmark）
    win         日胜率（策略日收益 > 0 占比）
    turn        平均日换手率
"""
from __future__ import annotations

import json
from pathlib import Path

# 输出列顺序 = (字段名, 列宽, 格式化说明)
_COLS: list[tuple[str, int, str]] = [
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


def _collect(artifacts_root: Path) -> list[dict]:
    """读取 artifacts 下所有 lgbm_daily_*/metrics.json 拼成行列表。

    Walk-Forward 模式产生的 run 结构为：
      artifacts/lgbm_daily_wf_<hash>/concatenated/metrics.json
    此函数优先读 concatenated/ 子目录下的 metrics.json（包含完整指标），
    若不存在则回落到根级 metrics.json（单切分模式）。

    缺失字段统一以 0 填充以保证对齐；按 metrics.json 的 mtime 升序，
    便于"最新 run 在最下"的工作流。
    """
    rows: list[dict] = []
    for run_dir in sorted(artifacts_root.glob("lgbm_daily_*")):
        if not run_dir.is_dir():
            continue
        # 优先使用 walk-forward 的 concatenated/metrics.json
        concat_p = run_dir / "concatenated" / "metrics.json"
        root_p = run_dir / "metrics.json"
        p = concat_p if concat_p.exists() else root_p
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            d = json.load(f)
        m = d.get("metrics", {})
        t = d.get("timing_seconds", {})
        rolling = d.get("rolling", {})
        rows.append({
            "run_id": d["run_id"][-8:],
            "ds_sec": t.get("dataset_build_sec", 0),
            "train_sec": t.get("train_sec", 0),
            "rc": rolling.get("fold_count") or t.get("rolling_chunks", 0) or 0,
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
            "mtime": p.stat().st_mtime,
        })
    rows.sort(key=lambda x: x["mtime"])
    return rows


def main() -> None:
    workspace_root = Path(__file__).resolve().parent.parent
    rows = _collect(workspace_root / "artifacts")
    if not rows:
        print("[compare_runs] 未发现任何 artifacts/lgbm_daily_* 目录")
        return

    header = " ".join(f"{name:>{width}}" for name, width, _ in _COLS)
    print(header)
    print("-" * len(header))
    for r in rows:
        line = " ".join(f"{r[name]:>{width}{spec}}" for name, width, spec in _COLS)
        print(line)


if __name__ == "__main__":
    main()
