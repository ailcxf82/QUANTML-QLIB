"""
B1：LGBM × gru_raw rank 融合回测验收
============================================================

承接 B0（scripts/accept_b0_ts.py 已验证「信息源分离 + 融合 IC 增量」），
B1 的目标是把 rank 融合信号写成真实 signals.parquet，复用既有回测引擎
（python/backtest/engine.py）跑端到端回测，验证 §8.5.8 的 Sharpe / 换手门禁。

为什么必须单独回测（而非沿用 B0 的离线 IC）：
  见 docs/lessons —— IC 升 ≠ Alpha 升。融合后 IC 提升不代表 TopK 组合
  超额收益/Sharpe 提升；gru_raw 单模型 Sharpe 偏低（1.68）有可能拖累融合。
  只有对融合信号实跑撮合回测，得到 Sharpe / 换手，才能给出上线结论。

口径要求（与 B0 一致）：
  三个 run 必须同 freq、同 OOS。脚本默认取 --freq daily（test 长窗），
  并复用 lgbm_optuna 的合并配置（base→lgbm_optuna→daily）作为回测 cfg，
  因为回测引擎只消费 strategy/executor/backtest/experiment/segments 段，
  与具体模型无关 —— pred_score 换成融合信号即可。

门禁（docs/时序模型优化.md §8.5.8）：
  - B1-S 融合 OOS Sharpe ≥ max(LGBM 单模型, 时序单模型) + 0.05
  - B1-T 融合换手率与 LGBM 主力相比 ±20% 以内

用法：
  # 自动取 artifacts 下最新的 lgbm_optuna_daily / gru_raw 两个 run
  python scripts/run_b1_blend_backtest.py --auto --freq daily

  # 显式指定 run_id 与权重
  python scripts/run_b1_blend_backtest.py --freq daily \
      --run-lgbm   lgbm_optuna_daily_30ad95e7 \
      --run-gru-raw gru_raw_daily_b2e02085 \
      --w-lgbm 0.7 --w-ts 0.3

输出：
  - artifacts/<run_id>/  （signals.parquet + metrics.json + 图表 + report.html）
  - artifacts/_audit/b1_blend_backtest_<timestamp>.json  （门禁结论）
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

# B0 验收脚本中的信号/标签辅助函数（避免重复实现，保持口径一致）
from accept_b0_ts import (  # noqa: E402
    _load_label_expr,
    _resolve_provider_uri,
    align_signals,
    find_latest_run,
    load_label_for_signal,
    load_signals,
    rank_blend,
    signal_ic,
)

import qlib  # noqa: E402

from run_experiment import (  # noqa: E402
    RunContext,
    load_and_merge_configs,
    persist_artifacts,
    run_backtest,
    validate_provider_uri,
)


# ── 门禁阈值（与 docs/时序模型优化.md §8.5.8 对齐）────────────────────
GATE_SHARPE_MIN_DELTA = 0.05
GATE_TURNOVER_REL_TOL = 0.20
DEFAULT_W_LGBM = 0.7
DEFAULT_W_TS = 0.3


def _read_metrics(run_dir: Path) -> Dict[str, Any]:
    """读取单模型 run 的 metrics.json 的 metrics 子字典。"""
    p = run_dir / "metrics.json"
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        doc = json.load(f)
    inner = doc.get("metrics", doc)
    return inner if isinstance(inner, dict) else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="B1 LGBM × gru_raw rank 融合回测验收")
    parser.add_argument("--auto", action="store_true", help="自动选取最新 run")
    parser.add_argument("--freq", default="daily", help="频率配置名（解析 provider_uri + 回测 cfg）")
    parser.add_argument("--run-lgbm", default=None, help="LGBM 主力 run_id（默认自动取最新 lgbm_optuna_daily）")
    parser.add_argument("--run-gru-raw", default=None, help="gru_raw run_id（默认自动取最新 gru_raw）")
    parser.add_argument("--w-lgbm", type=float, default=DEFAULT_W_LGBM)
    parser.add_argument("--w-ts", type=float, default=DEFAULT_W_TS)
    parser.add_argument("--provider-uri", default=None)
    parser.add_argument("--output", default=None, help="门禁 JSON 输出路径")
    args = parser.parse_args()

    if not args.auto and not (args.run_lgbm and args.run_gru_raw):
        parser.error("请指定 --auto，或同时提供 --run-lgbm 与 --run-gru-raw")

    # ── 定位两个单模型 run ────────────────────────────────────────────
    if args.run_lgbm:
        lgbm_dir = _REPO_ROOT / "artifacts" / args.run_lgbm
        if not lgbm_dir.is_dir():
            raise FileNotFoundError(f"run 目录不存在: {lgbm_dir}")
    else:
        lgbm_dir = find_latest_run("lgbm_optuna_daily")
        if lgbm_dir is None:
            raise FileNotFoundError("未找到 lgbm_optuna_daily_* 且含 signals.parquet 的 run")

    if args.run_gru_raw:
        gru_raw_dir = _REPO_ROOT / "artifacts" / args.run_gru_raw
        if not gru_raw_dir.is_dir():
            raise FileNotFoundError(f"run 目录不存在: {gru_raw_dir}")
    else:
        gru_raw_dir = find_latest_run("gru_raw_")
        if gru_raw_dir is None:
            raise FileNotFoundError("未找到 gru_raw_* 且含 signals.parquet 的 run")

    print(f"LGBM 主力 run : {lgbm_dir.name}")
    print(f"gru_raw  run : {gru_raw_dir.name}")
    print(f"融合权重     : {args.w_lgbm:.2f}×rank(lgbm) + {args.w_ts:.2f}×rank(gru_raw)")

    # ── 复用 lgbm_optuna 合并配置作为回测 cfg（回测引擎与模型无关）─────────
    cfg: Dict[str, Any] = load_and_merge_configs("lgbm_optuna", args.freq, None)
    cfg.setdefault("model_meta", {})["name"] = "b1_blend"
    base_run_id = f"b1_blend_lgbm_gru_raw_{args.freq}"
    cfg.setdefault("experiment", {})["run_id"] = base_run_id

    provider_uri = args.provider_uri or cfg["qlib_init"]["provider_uri"]
    validate_provider_uri(provider_uri)
    qlib.init(provider_uri=provider_uri, region=cfg["qlib_init"]["region"])

    # ── 加载并对齐信号（单模型 signals.parquet 已做 universe 过滤）─────────
    sig_lgbm = load_signals(lgbm_dir)
    sig_raw = load_signals(gru_raw_dir)
    sig_lgbm, sig_raw = align_signals(sig_lgbm, sig_raw)

    blend_sig = rank_blend(
        {"lgbm": sig_lgbm, "gru_raw": sig_raw},
        {"lgbm": args.w_lgbm, "gru_raw": args.w_ts},
    )

    # ── 标签（用于回测 IC/ICIR；Sharpe/换手不依赖它）────────────────────
    label_expr = _load_label_expr()
    label = load_label_for_signal(blend_sig, label_expr, provider_uri)
    ic_blend, icir_blend, n_days = signal_ic(blend_sig, label)

    # ── 跑融合信号的端到端回测 ─────────────────────────────────────────
    unique_run_id = f"{base_run_id}_{uuid.uuid4().hex[:8]}"
    run_ctx = RunContext(
        run_id=unique_run_id,
        output_dir=_REPO_ROOT / "artifacts" / unique_run_id,
    )
    print(f"\n开始回测融合信号 → {run_ctx.output_dir}\n")
    report_df, indicator_df, trades_df, realized_pnl_df, metrics = run_backtest(
        cfg, blend_sig, output_dir=run_ctx.output_dir, label=label,
    )
    persist_artifacts(
        run_ctx, cfg, metrics, report_df, indicator_df, blend_sig,
        trades_df=trades_df, realized_pnl_df=realized_pnl_df,
    )

    # ── 单模型基线 + 门禁评估 ──────────────────────────────────────────
    m_lgbm = _read_metrics(lgbm_dir)
    m_raw = _read_metrics(gru_raw_dir)

    blend_sharpe = float(metrics.get("sharpe_ratio", float("nan")))
    blend_turnover = float(metrics.get("avg_turnover", float("nan")))
    lgbm_sharpe = float(m_lgbm.get("sharpe_ratio", float("nan")))
    raw_sharpe = float(m_raw.get("sharpe_ratio", float("nan")))
    lgbm_turnover = float(m_lgbm.get("avg_turnover", float("nan")))

    sharpe_baseline = max(lgbm_sharpe, raw_sharpe)
    sharpe_gate = blend_sharpe >= sharpe_baseline + GATE_SHARPE_MIN_DELTA

    turnover_lo = lgbm_turnover * (1 - GATE_TURNOVER_REL_TOL)
    turnover_hi = lgbm_turnover * (1 + GATE_TURNOVER_REL_TOL)
    turnover_gate = turnover_lo <= blend_turnover <= turnover_hi

    gates = [
        {
            "gate_id": "B1-S",
            "name": "融合 OOS Sharpe",
            "value": round(blend_sharpe, 4),
            "threshold": f">= max({lgbm_sharpe:.3f}, {raw_sharpe:.3f}) + {GATE_SHARPE_MIN_DELTA} = {sharpe_baseline + GATE_SHARPE_MIN_DELTA:.3f}",
            "passed": bool(sharpe_gate),
            "interpretation": "融合相对单模型最优 Sharpe 需净增 ≥0.05，否则融合无正向贡献。",
        },
        {
            "gate_id": "B1-T",
            "name": "融合换手率 vs LGBM 主力",
            "value": round(blend_turnover, 4),
            "threshold": f"[{turnover_lo:.4f}, {turnover_hi:.4f}] (LGBM {lgbm_turnover:.4f} ±{GATE_TURNOVER_REL_TOL:.0%})",
            "passed": bool(turnover_gate),
            "interpretation": "换手率偏离主力过大会改变成本结构；超出需重新调融合权重。",
        },
    ]

    overall = all(g["passed"] for g in gates)

    # ── 控制台报告 ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("B1 rank 融合回测验收报告")
    print("=" * 72)
    print(f"\n[融合信号] run={run_ctx.run_id}")
    print(f"  OOS 交易日={n_days}  融合 IC={ic_blend:+.4f}  ICIR={icir_blend:+.3f}")
    print("\n[Sharpe 对照]")
    print(f"  LGBM 主力 : {lgbm_sharpe:+.4f}  (turnover {lgbm_turnover:.4f})")
    print(f"  gru_raw   : {raw_sharpe:+.4f}  (turnover {float(m_raw.get('avg_turnover', float('nan'))):.4f})")
    print(f"  融合      : {blend_sharpe:+.4f}  (turnover {blend_turnover:.4f})")
    print("\n[门禁]")
    n_pass = sum(1 for g in gates if g["passed"])
    for g in gates:
        mark = "PASS" if g["passed"] else "FAIL"
        print(f"  [{mark}] {g['gate_id']} {g['name']}: value={g['value']} (要求 {g['threshold']})")
        print(f"         → {g['interpretation']}")
    print("\n" + "-" * 72)
    print(f"总体结论: {'B1 验收通过' if overall else 'B1 验收未通过'} ({n_pass}/{len(gates)} 项通过)")
    print("=" * 72 + "\n")

    # ── JSON 报告 ─────────────────────────────────────────────────────
    report = {
        "generated_at": datetime.now().isoformat(),
        "freq": args.freq,
        "provider_uri": provider_uri,
        "weights": {"lgbm": args.w_lgbm, "gru_raw": args.w_ts},
        "blend_run_id": run_ctx.run_id,
        "runs": {
            "lgbm": {"run_id": lgbm_dir.name, "sharpe": lgbm_sharpe, "turnover": lgbm_turnover,
                     "ic_mean": m_lgbm.get("ic_mean")},
            "gru_raw": {"run_id": gru_raw_dir.name, "sharpe": raw_sharpe,
                        "turnover": m_raw.get("avg_turnover"), "ic_mean": m_raw.get("ic_mean")},
        },
        "blend_metrics": {
            "sharpe_ratio": blend_sharpe,
            "avg_turnover": blend_turnover,
            "ic_mean": ic_blend,
            "icir": icir_blend,
            "n_days": n_days,
            "annualized_return": metrics.get("annualized_return"),
            "max_drawdown": metrics.get("max_drawdown"),
            "information_ratio": metrics.get("information_ratio"),
        },
        "gates": gates,
        "overall_pass": overall,
    }
    out_path = Path(args.output) if args.output else (
        _REPO_ROOT / "artifacts" / "_audit" / f"b1_blend_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"JSON 已写入: {out_path}")
    print(f"回测产物目录: {run_ctx.output_dir}")


if __name__ == "__main__":
    main()
