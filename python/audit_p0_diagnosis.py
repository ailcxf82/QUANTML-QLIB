"""P0 诊断脚本：逐因子单 IC + 两 run 月度对照 + new vs old 集合分布。

层级位置: Data -> Feature -> Model -> Signal -> [Diagnosis] -> Backtest

输入:
  - 当前 lgbm/daily 配置（与 run_experiment 一致）
  - 两个待对照的 run 目录（默认 4d43003f vs f435f29a）

输出:
  artifacts/_audit/
    factor_ic_per_segment.csv     单因子 IC/ICIR（按特征 × 段）
    factor_ic_summary.txt         按 |test ICIR| 降序的纯文本表
    monthly_ic_compare.csv        两 run 的月度 IC
    monthly_excess_compare.csv    两 run 的月度超额收益

终端打印:
  - new vs old 集合 ICIR 分布统计
  - test 段 |ICIR| Top 30 特征清单
  - 月度 IC 对照表（仅 test 段）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from features import _BUILTIN_ALPHAS
from features.combined_json_factors import _EXPR_BY_JSON_NAME, _DISABLED_NAMES
from run_experiment import (
    inject_feature_config,
    load_and_merge_configs,
    patch_dataset_class,
    validate_provider_uri,
    build_dataset,
)

import qlib  # noqa: E402


# 原工程的"老 26 列"集合（手工记录，便于 new vs old 对比）。
# 任何不在此集合的因子视为 P0 新加。
_ORIGINAL_26_NAMES: frozenset[str] = frozenset({
    # momentum (3)
    "RET1", "RET2", "RET5",
    # volume (5; NET_AMT 已 disabled，但仍属于"老集合")
    "VOL_CHG", "VOL5", "VOL20", "TURN", "NET_AMT",
    # technical (10)
    "KMID", "KLEN", "KSFT", "STD5", "STD20", "MA5_20", "MA10_60",
    "RSI12", "MACD_DIF", "KDJK",
    # fundamental (2)
    "PB", "MV",
    # combined json (6; quarterly_eps 已 disabled，但仍属于"老集合")
    "ValueMR_20D", "volume_zscore_20d", "turnover_acceleration_5d",
    "volume_ratio_ma_15d", "roe_momentum_60d", "quarterly_eps",
})


def _section_ic(feature: pd.Series, label: pd.Series) -> Dict[str, float]:
    """按日截面计算 corr(feature, label)，再聚合 mean/std/icir/posratio。

    feature, label 必须共享 (datetime, instrument) 多重索引。
    """
    df = pd.concat([feature.rename("x"), label.rename("y")], axis=1).dropna()
    if df.empty:
        return {"ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan, "ic_pos_ratio": np.nan, "n_days": 0}
    by_day = df.groupby(level="datetime")
    ics: List[float] = []
    for _, sub in by_day:
        if len(sub) < 5:
            continue
        x = sub["x"].to_numpy()
        y = sub["y"].to_numpy()
        if x.std() < 1e-12 or y.std() < 1e-12:
            continue
        ic = float(np.corrcoef(x, y)[0, 1])
        if np.isfinite(ic):
            ics.append(ic)
    if not ics:
        return {"ic_mean": np.nan, "ic_std": np.nan, "icir": np.nan, "ic_pos_ratio": np.nan, "n_days": 0}
    ics_arr = np.asarray(ics, dtype=float)
    mean = float(ics_arr.mean())
    std = float(ics_arr.std(ddof=1)) if len(ics_arr) > 1 else np.nan
    icir = mean / std * np.sqrt(252) if std and std > 0 else np.nan
    return {
        "ic_mean": mean,
        "ic_std": std,
        "icir": icir,
        "ic_pos_ratio": float((ics_arr > 0).mean()),
        "n_days": len(ics),
    }


def _segment_filter(
    df_feat: pd.DataFrame,
    df_label: pd.DataFrame,
    start: str,
    end: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    dts = df_feat.index.get_level_values("datetime")
    mask = (dts >= start) & (dts <= end)
    return df_feat.loc[mask], df_label.loc[mask]


def per_factor_ic(
    df_feat: pd.DataFrame,
    label: pd.Series,
    segments: Dict[str, Tuple[str, str]],
) -> pd.DataFrame:
    """逐特征 × 逐段 IC 表。"""
    rows: List[Dict[str, float]] = []
    feat_cols = list(df_feat.columns)
    for seg_name, (start, end) in segments.items():
        sub_feat, sub_label = _segment_filter(df_feat, label.to_frame(), start, end)
        sub_label = sub_label.iloc[:, 0]
        for col in feat_cols:
            stat = _section_ic(sub_feat[col], sub_label)
            stat.update({"feature": col, "segment": seg_name})
            rows.append(stat)
    return pd.DataFrame(rows)


def _decide_group(name: str) -> str:
    return "old" if name in _ORIGINAL_26_NAMES else "new"


def summarize_new_vs_old(ic_df: pd.DataFrame) -> str:
    """对 test 段，比较 new vs old 集合的 ICIR 分布。"""
    test_df = ic_df[ic_df["segment"] == "test"].copy()
    test_df["set"] = test_df["feature"].map(_decide_group)

    out_lines: List[str] = []
    for grp_name in ("old", "new"):
        sub = test_df[test_df["set"] == grp_name]
        if sub.empty:
            out_lines.append(f"  {grp_name:<4s}: 空")
            continue
        abs_icir = sub["icir"].abs()
        out_lines.append(
            f"  {grp_name:<4s}: n={len(sub):3d} "
            f"|ICIR|: mean={abs_icir.mean():.3f} median={abs_icir.median():.3f} "
            f"max={abs_icir.max():.3f} "
            f"|ICIR|>1.0: {int((abs_icir>1.0).sum()):3d} "
            f"|ICIR|>0.5: {int((abs_icir>0.5).sum()):3d} "
            f"|ICIR|<0.2: {int((abs_icir<0.2).sum()):3d}"
        )
    return "\n".join(out_lines)


def monthly_ic_for_run(signals_path: Path, label: pd.Series) -> pd.DataFrame:
    """从 run 的 signals.parquet 读取 score，与 label 拼接后按月聚合 IC。"""
    if not signals_path.is_file():
        return pd.DataFrame()
    sig = pd.read_parquet(signals_path)
    score_col = sig.columns[0]
    score = sig[score_col].rename("score")
    df = pd.concat([score, label.rename("y")], axis=1).dropna()
    df = df.reset_index()
    df["month"] = df["datetime"].dt.to_period("M").astype(str)

    # 先按 (datetime) 计算日 IC，再按月聚合
    daily_records: List[Dict[str, float]] = []
    for date, sub in df.groupby("datetime"):
        if len(sub) < 5 or sub["score"].std() < 1e-12 or sub["y"].std() < 1e-12:
            continue
        ic = float(np.corrcoef(sub["score"], sub["y"])[0, 1])
        if np.isfinite(ic):
            daily_records.append({"datetime": pd.Timestamp(date), "ic": ic})
    daily = pd.DataFrame(daily_records)
    if daily.empty:
        return pd.DataFrame()
    daily["month"] = daily["datetime"].dt.to_period("M").astype(str)
    monthly = daily.groupby("month").agg(
        ic_mean=("ic", "mean"),
        ic_std=("ic", "std"),
        n_days=("ic", "size"),
    )
    return monthly


def monthly_excess_for_run(report_path: Path) -> pd.DataFrame:
    """从 backtest_report.parquet 读取每日策略/基准收益，按月汇总超额。"""
    if not report_path.is_file():
        return pd.DataFrame()
    rep = pd.read_parquet(report_path)
    if "return" not in rep.columns or "bench" not in rep.columns:
        return pd.DataFrame()
    df = rep[["return", "bench"]].copy()
    df.index = pd.to_datetime(df.index)
    df["excess"] = df["return"] - df["bench"]
    df["month"] = df.index.to_period("M").astype(str)
    monthly = df.groupby("month").agg(
        ret_strategy=("return", lambda s: float((1 + s).prod() - 1)),
        ret_bench=("bench", lambda s: float((1 + s).prod() - 1)),
        excess_arith=("excess", "sum"),
    )
    return monthly


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lgbm")
    parser.add_argument("--freq", default="daily")
    parser.add_argument("--run-old", default="lgbm_daily_4d43003f", help="P0 之前的基线 run id")
    parser.add_argument("--run-new", default="lgbm_daily_f435f29a", help="P0 之后的 run id")
    args = parser.parse_args()

    cfg = load_and_merge_configs(model_name=args.model, freq_name=args.freq, full_config=None)
    cfg = patch_dataset_class(cfg)
    cfg = inject_feature_config(cfg)

    provider_uri = cfg["qlib_init"]["provider_uri"]
    validate_provider_uri(provider_uri)
    qlib.init(provider_uri=provider_uri, region=cfg["qlib_init"]["region"])

    segments = cfg["dataset"]["kwargs"]["segments"]
    seg_dict: Dict[str, Tuple[str, str]] = {k: (str(v[0]), str(v[1])) for k, v in segments.items()}
    print(f"\n[diag] segments: {seg_dict}\n")

    dataset = build_dataset(cfg)
    dataset.setup_data()
    handler = dataset.handler

    df_raw_feat = handler.fetch(col_set="feature", data_key=handler.DK_R)
    df_raw_label_full = handler.fetch(col_set="label", data_key=handler.DK_R)
    label_raw = df_raw_label_full.iloc[:, 0]
    print(f"[diag] raw feature shape: {df_raw_feat.shape}; raw label shape: {label_raw.shape}")

    out_dir = Path("artifacts/_audit")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 单因子 IC ────────────────────────────────────────
    print("[diag] 计算逐因子分段 IC（约 30s × 段数）...")
    ic_df = per_factor_ic(df_raw_feat, label_raw, seg_dict)
    ic_df.to_csv(out_dir / "factor_ic_per_segment.csv", index=False, encoding="utf-8-sig")

    # 透视成 feature × segment 的 |ICIR| 与 ic_mean
    icir_pivot = ic_df.pivot_table(index="feature", columns="segment", values="icir", aggfunc="mean")
    ic_pivot = ic_df.pivot_table(index="feature", columns="segment", values="ic_mean", aggfunc="mean")
    seg_order = [s for s in ["train", "valid", "test"] if s in icir_pivot.columns]
    icir_pivot = icir_pivot[seg_order]
    ic_pivot = ic_pivot[seg_order]

    # 标注 set
    icir_pivot["set"] = icir_pivot.index.map(_decide_group)
    ic_pivot["set"] = ic_pivot.index.map(_decide_group)

    # ── new vs old 集合统计 ───────────────────────────────
    print("\n=== test 段 |ICIR| 分布对比（new vs old 集合）===")
    print(summarize_new_vs_old(ic_df))

    # ── Top 30 by |test ICIR| ────────────────────────────
    test_sorted = icir_pivot.copy()
    test_sorted["abs_icir_test"] = test_sorted["test"].abs()
    test_sorted = test_sorted.sort_values("abs_icir_test", ascending=False)

    print("\n=== test 段 |ICIR| Top 30（含 set 标签）===")
    print(test_sorted.head(30).drop(columns=["abs_icir_test"]).to_string(
        float_format=lambda x: f"{x:>+.3f}"
    ))

    print("\n=== test 段 |ICIR| 最差 15（最可能是噪声/过拟合源）===")
    print(test_sorted.tail(15).drop(columns=["abs_icir_test"]).to_string(
        float_format=lambda x: f"{x:>+.3f}"
    ))

    # ── train ↔ test ICIR 一致性（"在 train 强但 test 弱"=过拟合候选）────
    consist = icir_pivot.copy()
    if "train" in consist.columns and "test" in consist.columns:
        consist["sign_match"] = (np.sign(consist["train"]) == np.sign(consist["test"])).astype(int)
        consist["icir_decay"] = consist["train"].abs() - consist["test"].abs()
        unstable = consist[(consist["sign_match"] == 0) & (consist["train"].abs() > 0.5)]
        print(f"\n=== train↔test ICIR 符号反转的因子（共 {len(unstable)} 个）===")
        if not unstable.empty:
            print(unstable[["train", "valid", "test", "set"]].to_string(
                float_format=lambda x: f"{x:>+.3f}"
            ))

    # ── 写综合 summary ────────────────────────────────────
    summary_path = out_dir / "factor_ic_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("[factor_ic_summary] segment-wise ICIR pivot (sorted by |test|)\n\n")
        f.write(test_sorted.drop(columns=["abs_icir_test"]).to_string(
            float_format=lambda x: f"{x:>+.3f}"
        ))
        f.write("\n\n=== test 段 new vs old 集合统计 ===\n")
        f.write(summarize_new_vs_old(ic_df))
    print(f"\n[diag] 单因子 IC 详表已保存: {summary_path}")

    # ── 两 run 月度 IC 与超额收益对照 ────────────────────
    runs_root = Path("artifacts")
    label_for_signal = label_raw  # 与 signals.parquet 同口径的 raw label
    monthly_ic_old = monthly_ic_for_run(runs_root / args.run_old / "signals.parquet", label_for_signal)
    monthly_ic_new = monthly_ic_for_run(runs_root / args.run_new / "signals.parquet", label_for_signal)
    monthly_ex_old = monthly_excess_for_run(runs_root / args.run_old / "backtest_report.parquet")
    monthly_ex_new = monthly_excess_for_run(runs_root / args.run_new / "backtest_report.parquet")

    monthly_ic = pd.concat({"old_26": monthly_ic_old["ic_mean"], "new_83": monthly_ic_new["ic_mean"]}, axis=1)
    monthly_ic["delta_new_minus_old"] = monthly_ic["new_83"] - monthly_ic["old_26"]
    monthly_ic.to_csv(out_dir / "monthly_ic_compare.csv", encoding="utf-8-sig")

    monthly_ex = pd.concat({
        "old_26_strategy": monthly_ex_old["ret_strategy"],
        "new_83_strategy": monthly_ex_new["ret_strategy"],
        "bench": monthly_ex_old["ret_bench"],
        "old_26_excess": monthly_ex_old["ret_strategy"] - monthly_ex_old["ret_bench"],
        "new_83_excess": monthly_ex_new["ret_strategy"] - monthly_ex_new["ret_bench"],
    }, axis=1)
    monthly_ex["delta_excess_new_minus_old"] = monthly_ex["new_83_excess"] - monthly_ex["old_26_excess"]
    monthly_ex.to_csv(out_dir / "monthly_excess_compare.csv", encoding="utf-8-sig")

    print("\n=== 月度 IC 对照（test 期内）===")
    print(monthly_ic.to_string(float_format=lambda x: f"{x:>+.4f}"))

    print("\n=== 月度超额收益对照（new_83 - old_26 < 0 即恶化月）===")
    print(monthly_ex.to_string(float_format=lambda x: f"{x:>+.4f}"))

    # ── 数据驱动结论 ────────────────────────────────────
    test_only = icir_pivot.dropna(subset=["test"])
    old_test = test_only[test_only["set"] == "old"]["test"].abs()
    new_test = test_only[test_only["set"] == "new"]["test"].abs()
    old_strong = int((old_test > 0.5).sum())
    new_strong = int((new_test > 0.5).sum())
    new_noise = int((new_test < 0.2).sum())

    print("\n────────────── 诊断结论（机械归纳，需结合业务再判断）────────────")
    print(f"  原 26 列中 |test ICIR| > 0.5 的 = {old_strong} / {len(old_test)}")
    print(f"  新 57 列中 |test ICIR| > 0.5 的 = {new_strong} / {len(new_test)}")
    print(f"  新 57 列中 |test ICIR| < 0.2 的 = {new_noise} / {len(new_test)} （噪声候选）")
    print(f"  新 57 列 vs 原 26 列 中位数 |test ICIR|: "
          f"{new_test.median():.3f} vs {old_test.median():.3f}")

    if not monthly_ex.empty:
        bad_months = monthly_ex[monthly_ex["delta_excess_new_minus_old"] < -0.02]
        good_months = monthly_ex[monthly_ex["delta_excess_new_minus_old"] > 0.02]
        print(f"  月度超额: new < old 超过 2pp 的月数 = {len(bad_months)}; "
              f"new > old 超过 2pp 的月数 = {len(good_months)}")


if __name__ == "__main__":
    main()
