"""
B0 时序模型验收自动化（gru vs gru_raw [+ 可选 lgbm 主力]）
============================================================

对照 docs/时序模型优化.md §8.5.6 / §8.5.8，基于 artifacts/<run_id>/signals.parquet
离线计算 IC、预测相关性、rank 融合 IC，并输出 PASS/FAIL 门禁结论。

前置：需先完成同 freq、同数据口径下的实验，例如：
  python python/run_experiment.py --model gru     --freq daily_retrain
  python python/run_experiment.py --model gru_raw --freq daily_retrain
  python python/run_experiment.py --model lgbm_optuna --freq daily_retrain  # 可选

用法：
  # 自动选取 artifacts 下最新的 gru / gru_raw / lgbm_optuna run
  python scripts/accept_b0_ts.py --auto --freq daily_retrain

  # 显式指定 run_id
  python scripts/accept_b0_ts.py \\
      --run-gru gru_daily_retrain_xxxx \\
      --run-gru-raw gru_raw_daily_retrain_yyyy \\
      --run-lgbm lgbm_optuna_daily_retrain_zzzz

  # 仅检查因子注册表（不依赖 run 产物）
  python scripts/accept_b0_ts.py --check-registry-only

输出：
  控制台报告 + artifacts/_audit/b0_acceptance_<timestamp>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python"))

from backtest.signal_layer import SignalLayer  # noqa: E402


# ── 门禁阈值（与 docs/时序模型优化.md §8.5.8 对齐）────────────────────
GATE_MIN_TRADING_DAYS = 120
GATE_RHO_LGBM_RAW_MAX = 0.85
GATE_IC_GRU_RAW_VS_GRU_MIN_DELTA = -0.005
GATE_IC_BLEND_MIN_DELTA = 0.003
DEFAULT_W_LGBM = 0.7
DEFAULT_W_TS = 0.3


@dataclass
class GateResult:
    gate_id: str
    name: str
    value: Any
    threshold: str
    passed: bool
    interpretation: str


@dataclass
class RunSnapshot:
    run_id: str
    model: str
    freq: str
    n_days: int
    date_range: str
    n_features: Optional[int]
    ic_mean: float
    icir: float
    sharpe: Optional[float]
    metrics_ic_mean: Optional[float]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_provider_uri(freq_name: str) -> str:
    freq_path = _REPO_ROOT / "configs" / "freq" / f"{freq_name}.yaml"
    if freq_path.exists():
        freq_cfg = _load_yaml(freq_path)
        uri = freq_cfg.get("qlib_init", {}).get("provider_uri")
        if uri:
            return str(uri)
    base = _load_yaml(_REPO_ROOT / "configs" / "base.yaml")
    return str(base["qlib_init"]["provider_uri"])


def _load_label_expr() -> str:
    base = _load_yaml(_REPO_ROOT / "configs" / "base.yaml")
    label_block = (
        base["dataset"]["kwargs"]["handler"]["kwargs"]["data_loader"]["kwargs"]["config"]["label"]
    )
    return str(label_block[0][0])


def load_signals(run_dir: Path) -> pd.Series:
    for rel in ("signals.parquet", "concatenated/signals.parquet"):
        p = run_dir / rel
        if p.exists():
            df = pd.read_parquet(p)
            s = df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df
            s = s.rename("score")
            if not isinstance(s.index, pd.MultiIndex):
                raise TypeError(f"{p}: 索引不是 MultiIndex(datetime, instrument)")
            return s
    raise FileNotFoundError(f"未找到 signals.parquet: {run_dir}")


def load_run_meta(run_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"run_id": run_dir.name}
    for name in ("metrics.json", "model_meta.json"):
        p = run_dir / name
        if p.exists():
            with p.open(encoding="utf-8") as f:
                out[name.replace(".json", "")] = json.load(f)
    return out


def find_latest_run(pattern: str, *, exclude_substr: Optional[str] = None) -> Optional[Path]:
    candidates = sorted(
        (_REPO_ROOT / "artifacts").glob(f"{pattern}*"),
        key=lambda p: p.stat().st_mtime if p.is_dir() else 0,
        reverse=True,
    )
    for d in candidates:
        if not d.is_dir():
            continue
        if exclude_substr and exclude_substr in d.name:
            continue
        try:
            load_signals(d)
            return d
        except FileNotFoundError:
            continue
    return None


def load_label_for_signal(
    signal: pd.Series,
    label_expr: str,
    provider_uri: str,
) -> pd.Series:
    import qlib
    from qlib.data.dataset.loader import QlibDataLoader

    qlib.init(provider_uri=provider_uri, region="cn")
    dates = signal.index.get_level_values(0)
    instruments = signal.index.get_level_values(1).unique().tolist()
    start = pd.Timestamp(dates.min()).strftime("%Y-%m-%d")
    end = pd.Timestamp(dates.max()) + pd.Timedelta(days=5)
    end_s = end.strftime("%Y-%m-%d")

    # 仅取 label；不可传空的 feature 组，否则 QlibDataLoader 报
    # "The size of fields must be greater than 0"。
    loader = QlibDataLoader(
        config={"label": [[label_expr], ["LABEL0"]]},
    )
    df = loader.load(instruments=instruments, start_time=start, end_time=end_s)
    lbl = df.iloc[:, 0]
    lbl.name = "label"
    return lbl


def signal_ic(signal: pd.Series, label: pd.Series) -> Tuple[float, float, int]:
    layer = SignalLayer()
    summary = layer.process(signal, label=label, run_id="b0_acceptance")
    return summary.ic_mean, summary.icir, summary.n_dates


def mean_daily_rank_corr(a: pd.Series, b: pd.Series, min_stocks: int = 30) -> float:
    """逐日截面 Spearman（秩相关）均值，衡量两路预测排序相似度。"""
    joined = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if joined.empty:
        return float("nan")

    def _day_corr(g: pd.DataFrame) -> float:
        if len(g) < min_stocks:
            return np.nan
        return float(g["a"].rank().corr(g["b"].rank(), method="pearson"))

    daily = joined.groupby(level=0, group_keys=False).apply(_day_corr)
    return float(np.nanmean(daily.values))


def rank_blend(
    signals: Dict[str, pd.Series],
    weights: Dict[str, float],
) -> pd.Series:
    """按日截面 pct-rank 加权融合（rank-space，避免 raw score 尺度问题）。"""
    names = list(signals.keys())
    wide = pd.concat([signals[n].rename(n) for n in names], axis=1).dropna(how="any")
    ranked = wide.groupby(level=0).rank(pct=True)
    score = sum(weights[n] * ranked[n] for n in names)
    score.name = "score"
    return score


def align_signals(*series: pd.Series) -> List[pd.Series]:
    common = series[0].index
    for s in series[1:]:
        common = common.intersection(s.index)
    if len(common) == 0:
        raise ValueError("信号无公共 (datetime, instrument) 交集，无法对比")
    return [s.loc[common] for s in series]


def snapshot_from_run(run_dir: Path, label: pd.Series) -> RunSnapshot:
    meta = load_run_meta(run_dir)
    sig = load_signals(run_dir)
    ic_mean, icir, n_days = signal_ic(sig, label)
    file_metrics = meta.get("metrics", {})
    inner: Dict[str, Any] = {}
    if isinstance(file_metrics, dict):
        inner = file_metrics.get("metrics", file_metrics)
    m_ic = inner.get("ic_mean") if isinstance(inner, dict) else None
    mm = meta.get("model_meta", {})
    model_name = (
        file_metrics.get("model")
        if isinstance(file_metrics, dict)
        else None
    ) or mm.get("model") or run_dir.name.split("_")[0]
    freq_name = (
        file_metrics.get("freq")
        if isinstance(file_metrics, dict)
        else None
    ) or mm.get("freq") or "?"
    return RunSnapshot(
        run_id=run_dir.name,
        model=str(model_name),
        freq=str(freq_name),
        n_days=n_days,
        date_range=(
            f"{sig.index.get_level_values(0).min().date()}~"
            f"{sig.index.get_level_values(0).max().date()}"
        ),
        n_features=mm.get("n_features"),
        ic_mean=ic_mean,
        icir=icir,
        sharpe=inner.get("sharpe_ratio") if isinstance(inner, dict) else None,
        metrics_ic_mean=float(m_ic) if m_ic is not None else None,
    )


def check_registry_only() -> Dict[str, Any]:
    from features import build_feature_config, get_registry

    exprs_raw, names_raw = build_feature_config(groups=["raw_ohlcv"])
    all_exprs, all_names = build_feature_config(groups=None)
    return {
        "raw_ohlcv_count": len(names_raw),
        "raw_ohlcv_names": names_raw,
        "default_all_count": len(all_names),
        "open_r_in_default": "OPEN_R" in all_names,
        "registry_ok": len(names_raw) == 5 and "OPEN_R" not in all_names,
    }


def evaluate_gates(
    snaps: Dict[str, RunSnapshot],
    rho: Dict[str, float],
    ic_blend: float,
    n_days_overlap: int,
    w_lgbm: float,
    w_ts: float,
    min_trading_days: int = GATE_MIN_TRADING_DAYS,
) -> List[GateResult]:
    gates: List[GateResult] = []
    gru = snaps.get("gru")
    gru_raw = snaps.get("gru_raw")
    lgbm = snaps.get("lgbm")

    gates.append(
        GateResult(
            gate_id="G0",
            name="OOS 交易日长度",
            value=n_days_overlap,
            threshold=f">= {min_trading_days}",
            passed=n_days_overlap >= min_trading_days,
            interpretation="公共信号区间交易日过少时，IC/Sharpe 不稳定，结论不可用于上线决策。",
        )
    )

    rho_lgbm_raw = rho.get("lgbm_vs_gru_raw")
    if rho_lgbm_raw is not None and not np.isnan(rho_lgbm_raw):
        gates.append(
            GateResult(
                gate_id="G1",
                name="LGBM vs gru_raw 预测秩相关",
                value=round(rho_lgbm_raw, 4),
                threshold=f"< {GATE_RHO_LGBM_RAW_MAX}",
                passed=rho_lgbm_raw < GATE_RHO_LGBM_RAW_MAX,
                interpretation="ρ 越低说明信息源越分离；过高表示 gru_raw 与 LGBM 冗余，ensemble 增量有限。",
            )
        )

    if gru and gru_raw:
        delta = gru_raw.ic_mean - gru.ic_mean
        gates.append(
            GateResult(
                gate_id="G2",
                name="gru_raw IC 相对 gru",
                value=round(delta, 4),
                threshold=f">= {GATE_IC_GRU_RAW_VS_GRU_MIN_DELTA}",
                passed=delta >= GATE_IC_GRU_RAW_VS_GRU_MIN_DELTA,
                interpretation="B0 换输入后 IC 不应明显劣化；略降可接受，大幅为负需排查特征/训练。",
            )
        )

    if lgbm and gru_raw:
        base_ic = max(lgbm.ic_mean, gru_raw.ic_mean)
        blend_delta = ic_blend - base_ic
        gates.append(
            GateResult(
                gate_id="G3",
                name=f"rank 融合 IC 增量 ({w_lgbm:.0%} LGBM + {w_ts:.0%} gru_raw)",
                value=round(blend_delta, 4),
                threshold=f">= {GATE_IC_BLEND_MIN_DELTA}",
                passed=blend_delta >= GATE_IC_BLEND_MIN_DELTA,
                interpretation="融合 IC 相对单模型最优值的提升；仅 IC 升而回测超额不升仍可能红灯（见 lessons IC≠Alpha）。",
            )
        )

    if gru and "gru_vs_gru_raw" in rho:
        gates.append(
            GateResult(
                gate_id="G4",
                name="gru vs gru_raw 预测秩相关（信息性）",
                value=round(rho["gru_vs_gru_raw"], 4),
                threshold="参考值（非硬门禁）",
                passed=True,
                interpretation="应与 gru 明显不同；若接近 1，说明 raw_ohlcv 与旧特征组产出几乎相同排序。",
            )
        )

    return gates


def print_report(
    snaps: Dict[str, RunSnapshot],
    rho: Dict[str, float],
    ic_blend: float,
    gates: List[GateResult],
    registry: Optional[Dict[str, Any]],
) -> None:
    print("\n" + "=" * 72)
    print("B0 时序模型验收报告")
    print("=" * 72)

    if registry is not None:
        print("\n[因子注册表]")
        print(f"  raw_ohlcv 因子数: {registry['raw_ohlcv_count']} {registry['raw_ohlcv_names']}")
        print(f"  默认全量因子数: {registry['default_all_count']} (不含 OPEN_R: {not registry['open_r_in_default']})")
        print(f"  注册表检查: {'PASS' if registry['registry_ok'] else 'FAIL'}")

    print("\n[各 run 信号质量（脚本重算 IC，与 metrics.json 可略有差异）]")
    for key, snap in snaps.items():
        print(
            f"  {key:<10} run={snap.run_id}\n"
            f"             days={snap.n_days} range={snap.date_range} n_feat={snap.n_features}\n"
            f"             IC={snap.ic_mean:+.4f} ICIR={snap.icir:+.3f} "
            f"metrics_ic={snap.metrics_ic_mean if snap.metrics_ic_mean is not None else 'n/a'} "
            f"sharpe={snap.sharpe if snap.sharpe is not None else 'n/a'}"
        )

    print("\n[预测秩相关 ρ（日均截面 Spearman 均值）]")
    for name, val in sorted(rho.items()):
        print(f"  {name}: {val:+.4f}")

    if not np.isnan(ic_blend):
        print(f"\n[rank 融合] 离线 IC = {ic_blend:+.4f}")

    print("\n[门禁]")
    n_pass = sum(1 for g in gates if g.passed)
    for g in gates:
        mark = "PASS" if g.passed else "FAIL"
        print(f"  [{mark}] {g.gate_id} {g.name}: value={g.value} (要求 {g.threshold})")
        print(f"         → {g.interpretation}")

    overall = all(g.passed for g in gates if g.gate_id in {"G0", "G1", "G2", "G3"})
    print("\n" + "-" * 72)
    print(f"总体结论: {'B0 验收通过' if overall else 'B0 验收未通过'} ({n_pass}/{len(gates)} 项通过)")
    print("说明: G3 仅验证 IC 增量；完整 Sharpe/换手门禁需对融合信号再跑回测。")
    print("=" * 72 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="B0 时序模型验收（gru / gru_raw / lgbm）")
    parser.add_argument("--auto", action="store_true", help="自动选取最新 run")
    parser.add_argument("--freq", default="daily_retrain", help="用于解析 provider_uri")
    parser.add_argument("--run-gru", default=None)
    parser.add_argument("--run-gru-raw", default=None)
    parser.add_argument("--run-lgbm", default=None, help="可选，建议 lgbm_optuna 主力 run")
    parser.add_argument("--provider-uri", default=None)
    parser.add_argument("--w-lgbm", type=float, default=DEFAULT_W_LGBM)
    parser.add_argument("--w-ts", type=float, default=DEFAULT_W_TS)
    parser.add_argument("--min-days", type=int, default=GATE_MIN_TRADING_DAYS)
    parser.add_argument("--check-registry-only", action="store_true")
    parser.add_argument("--output", default=None, help="JSON 输出路径")
    args = parser.parse_args()

    if args.check_registry_only:
        reg = check_registry_only()
        print(json.dumps(reg, ensure_ascii=False, indent=2))
        print("registry_ok:", reg["registry_ok"])
        return

    if not args.auto and not args.run_gru and not args.run_gru_raw:
        parser.error("请指定 --auto 或同时提供 --run-gru 与 --run-gru-raw")

    if args.run_gru:
        gru_dir = _REPO_ROOT / "artifacts" / args.run_gru
        if not gru_dir.is_dir():
            raise FileNotFoundError(f"run 目录不存在: {gru_dir}")
    else:
        gru_dir = find_latest_run("gru_", exclude_substr="gru_raw")
        if gru_dir is None:
            raise FileNotFoundError("未找到 gru_*（排除 gru_raw）且含 signals.parquet 的 run")
    if "gru_raw" in gru_dir.name:
        raise ValueError(f"gru run 不能是 gru_raw: {gru_dir.name}")

    if args.run_gru_raw:
        gru_raw_dir = _REPO_ROOT / "artifacts" / args.run_gru_raw
        if not gru_raw_dir.is_dir():
            raise FileNotFoundError(f"run 目录不存在: {gru_raw_dir}")
    else:
        gru_raw_dir = find_latest_run("gru_raw_")
        if gru_raw_dir is None:
            raise FileNotFoundError("未找到 gru_raw_* 且含 signals.parquet 的 run")

    lgbm_dir: Optional[Path] = None
    if args.run_lgbm:
        lgbm_dir = _REPO_ROOT / "artifacts" / args.run_lgbm
    elif args.auto:
        lgbm_dir = find_latest_run("lgbm_optuna_daily")

    provider_uri = args.provider_uri or _resolve_provider_uri(args.freq)
    label_expr = _load_label_expr()

    print(f"provider_uri: {provider_uri}")
    print(f"label: {label_expr}")

    sig_gru = load_signals(gru_dir)
    sig_raw = load_signals(gru_raw_dir)
    sig_lgbm = load_signals(lgbm_dir) if lgbm_dir else None

    aligned = align_signals(sig_gru, sig_raw, *( [sig_lgbm] if sig_lgbm is not None else [] ))
    sig_gru, sig_raw = aligned[0], aligned[1]
    if sig_lgbm is not None:
        sig_lgbm = aligned[2]

    label = load_label_for_signal(sig_raw, label_expr, provider_uri)

    snaps: Dict[str, RunSnapshot] = {
        "gru": snapshot_from_run(gru_dir, label),
        "gru_raw": snapshot_from_run(gru_raw_dir, label),
    }
    if lgbm_dir is not None and sig_lgbm is not None:
        snaps["lgbm"] = snapshot_from_run(lgbm_dir, label)

    rho: Dict[str, float] = {
        "gru_vs_gru_raw": mean_daily_rank_corr(sig_gru, sig_raw),
    }
    if sig_lgbm is not None:
        rho["lgbm_vs_gru_raw"] = mean_daily_rank_corr(sig_lgbm, sig_raw)
        rho["lgbm_vs_gru"] = mean_daily_rank_corr(sig_lgbm, sig_gru)

    ic_blend = float("nan")
    if sig_lgbm is not None:
        blend_sig = rank_blend(
            {"lgbm": sig_lgbm, "gru_raw": sig_raw},
            {"lgbm": args.w_lgbm, "gru_raw": args.w_ts},
        )
        ic_blend, _, _ = signal_ic(blend_sig, label)

    n_days = sig_raw.index.get_level_values(0).nunique()
    gates = evaluate_gates(
        snaps, rho, ic_blend, n_days, args.w_lgbm, args.w_ts, min_trading_days=args.min_days,
    )
    registry = check_registry_only()

    report = {
        "generated_at": datetime.now().isoformat(),
        "freq": args.freq,
        "provider_uri": provider_uri,
        "label_expr": label_expr,
        "weights": {"lgbm": args.w_lgbm, "gru_raw": args.w_ts},
        "registry": registry,
        "runs": {k: asdict(v) for k, v in snaps.items()},
        "rho": rho,
        "ic_blend": ic_blend,
        "gates": [asdict(g) for g in gates],
        "overall_pass": all(g.passed for g in gates if g.gate_id in {"G0", "G1", "G2", "G3"}),
    }

    out_path = Path(args.output) if args.output else (
        _REPO_ROOT / "artifacts" / "_audit" / f"b0_acceptance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print_report(snaps, rho, ic_blend, gates, registry)
    print(f"JSON 已写入: {out_path}")


if __name__ == "__main__":
    main()
