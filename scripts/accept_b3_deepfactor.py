"""
B3 Deep Factor 验收（架构 B：日内 hidden → PCA → 注入 LGBM）
==============================================================
对照「纯 LGBM 主力基线」评估 lgbm_deepfactor 的回测表现，按 docs/时序模型优化.md §8.5.8
+ 防泄漏门禁输出 PASS/FAIL。承接 B1 的 IC≠Alpha 教训：以**回测 Sharpe** 为准，不以 IC 结论。

门禁：
  B3-S    融合 Sharpe ≥ 基线 Sharpe + 0.05            （核心）
  B3-IC   融合 IC ≥ 基线 IC（不低于）
  B3-T    换手在基线 ±20% 内
  B3-Leak PCA 仅在 train segment 拟合（读 deep_factor meta 校验）
  B3-Imp  deep factor 列（DF*）在 LGBM feature_importance > 0（确认被使用）

前置：
  python python/run_experiment.py --model lgbm_optuna   --freq daily            # 基线（若无）
  python python/run_experiment.py --model lgbm_deepfactor --freq daily \\
      --deep-factor-parquet artifacts/deep_factors/<...>/deep_factor.parquet     # 融合

用法：
  python scripts/accept_b3_deepfactor.py --auto
  python scripts/accept_b3_deepfactor.py --deepfactor-run <dir> --baseline-run <dir>

输出：控制台报告 + artifacts/_audit/b3_acceptance_<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python"))
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from accept_b0_ts import find_latest_run  # noqa: E402

# ── 门禁阈值 ──────────────────────────────────────────────────────────
GATE_SHARPE_MIN_DELTA = 0.05
GATE_IC_MIN_DELTA = 0.0
GATE_TURNOVER_TOL = 0.20


def _read_metrics(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "metrics.json"
    if not p.exists():
        raise FileNotFoundError(f"缺少 metrics.json: {run_dir}")
    data = json.loads(p.read_text(encoding="utf-8"))
    m = data.get("metrics", {})
    return {
        "run_id": data.get("run_id", run_dir.name),
        "sharpe": m.get("sharpe_ratio"),
        "ic_mean": m.get("ic_mean"),
        "icir": m.get("icir"),
        "turnover": m.get("avg_turnover"),
        "ann_return": m.get("annualized_return"),
        "max_dd": m.get("max_drawdown"),
    }


def _check_leak(deepfactor_run: Path) -> Dict[str, Any]:
    """从 deepfactor run 的 model_meta 追溯 deep_factor.parquet → meta.json，校验 PCA 仅 train fit。"""
    mm_path = deepfactor_run / "model_meta.json"
    parquet = None
    if mm_path.exists():
        mm = json.loads(mm_path.read_text(encoding="utf-8"))
        parquet = mm.get("deep_factor_parquet")
    result: Dict[str, Any] = {"parquet": parquet, "passed": False, "reason": ""}
    if not parquet:
        result["reason"] = "model_meta 未记录 deep_factor_parquet，无法校验"
        return result
    meta_path = Path(parquet).parent / "meta.json"
    if not meta_path.exists():
        result["reason"] = f"deep_factor meta.json 不存在: {meta_path}"
        return result
    dfm = json.loads(meta_path.read_text(encoding="utf-8"))
    fit_seg = dfm.get("pca_fit_segment")
    train_seg = (dfm.get("segments", {}) or {}).get("train")
    fit_rows = dfm.get("pca_fit_rows", 0)
    ok = bool(fit_seg) and fit_seg == train_seg and fit_rows > 0
    result.update(
        passed=ok,
        pca_fit_segment=fit_seg,
        train_segment=train_seg,
        pca_fit_rows=fit_rows,
        k=dfm.get("k"),
        explained_var=dfm.get("explained_variance_sum"),
        reason="PCA 仅在 train segment 拟合" if ok else "PCA fit segment 与 train 不一致或行数为0",
    )
    return result


def _check_importance(deepfactor_run: Path) -> Dict[str, Any]:
    """加载 lgbm_deepfactor 的 model.pkl，检查 DF* 列 feature_importance>0。"""
    result: Dict[str, Any] = {"passed": False, "reason": "", "df_importance": {}}
    model_path = deepfactor_run / "model.pkl"
    if not model_path.exists():
        result["reason"] = f"model.pkl 不存在: {model_path}"
        return result
    try:
        import joblib

        model = joblib.load(model_path)
        booster = getattr(model, "model", None)
        if booster is None or not hasattr(booster, "feature_name"):
            result["reason"] = "模型非 LightGBM Booster，跳过 importance 校验"
            return result
        names = list(booster.feature_name())
        imp = list(booster.feature_importance(importance_type="gain"))
        df_imp = {n: float(v) for n, v in zip(names, imp) if "DF" in n.upper()}
        total = sum(df_imp.values())
        result.update(
            passed=total > 0,
            df_importance=df_imp,
            df_importance_sum=total,
            n_df_cols=len(df_imp),
            reason=f"DF 列 gain 合计={total:.1f}" if total > 0 else "未发现 DF 列或其 importance=0",
        )
    except Exception as exc:  # noqa: BLE001
        result["reason"] = f"读取 importance 失败: {exc}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="B3 Deep Factor 验收")
    parser.add_argument("--auto", action="store_true", help="自动选取最新 deepfactor / baseline run")
    parser.add_argument("--deepfactor-run", default=None, help="lgbm_deepfactor run 目录")
    parser.add_argument("--baseline-run", default=None, help="纯 LGBM 基线 run 目录（lgbm_optuna_daily）")
    args = parser.parse_args()

    if args.deepfactor_run:
        df_run = Path(args.deepfactor_run)
    else:
        df_run = find_latest_run("lgbm_deepfactor_daily")
    if args.baseline_run:
        base_run = Path(args.baseline_run)
    else:
        base_run = find_latest_run("lgbm_optuna_daily", exclude_substr="retrain")

    if df_run is None or base_run is None:
        raise SystemExit(
            f"未找到 run：deepfactor={df_run} baseline={base_run}。"
            f"请先跑 lgbm_deepfactor 与 lgbm_optuna 回测，或用 --deepfactor-run/--baseline-run 指定。"
        )

    df_m = _read_metrics(df_run)
    base_m = _read_metrics(base_run)

    gates: List[Dict[str, Any]] = []

    # B3-S Sharpe
    s_req = (base_m["sharpe"] or 0) + GATE_SHARPE_MIN_DELTA
    gates.append({
        "gate": "B3-S", "name": "融合 Sharpe ≥ 基线+0.05",
        "deepfactor": df_m["sharpe"], "baseline": base_m["sharpe"], "required": round(s_req, 4),
        "passed": df_m["sharpe"] is not None and df_m["sharpe"] >= s_req,
    })
    # B3-IC
    ic_req = (base_m["ic_mean"] or 0) + GATE_IC_MIN_DELTA
    gates.append({
        "gate": "B3-IC", "name": "融合 IC ≥ 基线",
        "deepfactor": df_m["ic_mean"], "baseline": base_m["ic_mean"], "required": round(ic_req, 5),
        "passed": df_m["ic_mean"] is not None and df_m["ic_mean"] >= ic_req,
    })
    # B3-T turnover ±20%
    base_to = base_m["turnover"]
    lo, hi = (base_to * (1 - GATE_TURNOVER_TOL), base_to * (1 + GATE_TURNOVER_TOL)) if base_to else (None, None)
    gates.append({
        "gate": "B3-T", "name": "换手在基线 ±20% 内",
        "deepfactor": df_m["turnover"], "baseline": base_to,
        "required": f"[{lo:.4f}, {hi:.4f}]" if lo is not None else "n/a",
        "passed": (df_m["turnover"] is not None and lo is not None and lo <= df_m["turnover"] <= hi),
    })
    # B3-Leak
    leak = _check_leak(df_run)
    gates.append({
        "gate": "B3-Leak", "name": "PCA 仅 train fit（防泄漏）",
        "detail": leak.get("reason"), "passed": leak["passed"],
    })
    # B3-Imp
    imp = _check_importance(df_run)
    gates.append({
        "gate": "B3-Imp", "name": "DF 列被 LGBM 使用（importance>0）",
        "detail": imp.get("reason"), "passed": imp["passed"],
    })

    n_pass = sum(1 for g in gates if g["passed"])
    overall = all(g["passed"] for g in gates)

    # ── 控制台报告 ──────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("B3 Deep Factor 验收报告（架构 B：日内 hidden → PCA → LGBM）")
    print("=" * 72)
    print(f"deepfactor run: {df_m['run_id']}")
    print(f"baseline   run: {base_m['run_id']}")
    print(f"\n指标对比：")
    print(f"  {'metric':<12}{'deepfactor':>14}{'baseline':>14}")
    for key, label in (("sharpe", "Sharpe"), ("ic_mean", "IC"), ("icir", "ICIR"),
                       ("turnover", "Turnover"), ("ann_return", "AnnRet"), ("max_dd", "MaxDD")):
        dv = df_m[key]
        bv = base_m[key]
        dv_s = f"{dv:.4f}" if isinstance(dv, (int, float)) else "n/a"
        bv_s = f"{bv:.4f}" if isinstance(bv, (int, float)) else "n/a"
        print(f"  {label:<12}{dv_s:>14}{bv_s:>14}")
    print(f"\n门禁（{n_pass}/{len(gates)} 通过）：")
    for g in gates:
        flag = "PASS" if g["passed"] else "FAIL"
        extra = g.get("detail") or f"df={g.get('deepfactor')} base={g.get('baseline')} req={g.get('required')}"
        print(f"  [{flag}] {g['gate']:<8} {g['name']:<28} {extra}")
    print("\n结论：", "B3 验收通过 ✓" if overall else "B3 验收未通过 ✗")
    print("=" * 72)

    # ── 写审计 JSON ─────────────────────────────────────────────────
    audit_dir = _REPO_ROOT / "artifacts" / "_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    audit_path = audit_dir / f"b3_acceptance_{ts}.json"
    audit_path.write_text(json.dumps({
        "timestamp": ts,
        "deepfactor": df_m,
        "baseline": base_m,
        "gates": gates,
        "leak_detail": leak,
        "importance_detail": {k: v for k, v in imp.items() if k != "df_importance"},
        "overall_pass": overall,
    }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"审计报告：{audit_path}")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
