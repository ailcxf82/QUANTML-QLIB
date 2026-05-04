"""因子 NaN 覆盖率审计脚本。

层级位置: Data -> [Feature] -> Model -> Signal -> Portfolio -> Backtest -> Evaluation
职责:
  - 实例化与 run_experiment 完全一致的 dataset（不训练模型，仅 setup_data）
  - 输出每列特征在 train/valid/test 三段的非 NaN 占比
  - 用于决定哪些因子需要 enabled=False（覆盖率过低）

调用:
    python python/audit_features.py
    python python/audit_features.py --model lgbm --freq daily
    python python/audit_features.py --threshold 0.7   # 仅打印覆盖率<70% 的列

输出:
    控制台 + artifacts/_audit/feature_nan_coverage.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from run_experiment import (
    inject_feature_config,
    load_and_merge_configs,
    patch_dataset_class,
    validate_provider_uri,
    build_dataset,
)

import qlib  # noqa: E402


def _coverage_by_segment(
    df_feat: pd.DataFrame,
    segments: Dict[str, Tuple[str, str]],
) -> pd.DataFrame:
    """计算每列在各 segment 时间区间内的非 NaN 占比。"""
    rows: List[Dict[str, float]] = []
    for seg_name, (start, end) in segments.items():
        sub = df_feat.loc[(df_feat.index.get_level_values("datetime") >= start)
                          & (df_feat.index.get_level_values("datetime") <= end)]
        if sub.empty:
            continue
        total = len(sub)
        for col in sub.columns:
            non_na = sub[col].notna().sum()
            rows.append({
                "feature": col,
                "segment": seg_name,
                "rows": int(total),
                "non_na": int(non_na),
                "coverage": float(non_na) / float(total),
            })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lgbm")
    parser.add_argument("--freq", default="daily")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="低于此覆盖率的列会被特别标记")
    args = parser.parse_args()

    cfg = load_and_merge_configs(model_name=args.model, freq_name=args.freq, full_config=None)
    cfg = patch_dataset_class(cfg)
    cfg = inject_feature_config(cfg)

    provider_uri = cfg["qlib_init"]["provider_uri"]
    validate_provider_uri(provider_uri)
    qlib.init(provider_uri=provider_uri, region=cfg["qlib_init"]["region"])

    segments = cfg["dataset"]["kwargs"]["segments"]
    seg_dict: Dict[str, Tuple[str, str]] = {k: (str(v[0]), str(v[1])) for k, v in segments.items()}
    print(f"\n[audit] segments: {seg_dict}\n")

    dataset = build_dataset(cfg)
    dataset.setup_data()
    handler = dataset.handler

    df_raw = handler.fetch(
        col_set="feature",
        data_key=handler.DK_R,
    )
    print(f"[audit] raw feature shape (no processors): {df_raw.shape}")

    df_infer = handler.fetch(col_set="feature", data_key=handler.DK_I)
    print(f"[audit] infer feature shape (after processors): {df_infer.shape}\n")

    cov_raw = _coverage_by_segment(df_raw, seg_dict)
    cov_infer = _coverage_by_segment(df_infer, seg_dict)

    out_dir = Path("artifacts/_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    cov_raw.to_csv(out_dir / "feature_nan_coverage_raw.csv", index=False, encoding="utf-8-sig")
    cov_infer.to_csv(out_dir / "feature_nan_coverage_infer.csv", index=False, encoding="utf-8-sig")
    print(f"[audit] csv 已保存到: {out_dir.resolve()}")

    print("\n=== RAW（未处理）train 段覆盖率 < 阈值 的列 ===")
    train_cov = cov_raw[cov_raw["segment"] == "train"].copy()
    train_cov = train_cov.sort_values("coverage")
    bad = train_cov[train_cov["coverage"] < args.threshold]
    if bad.empty:
        print(f"  无 (阈值={args.threshold:.2f})")
    else:
        for _, row in bad.iterrows():
            print(f"  {row['feature']:<24s}  cov={row['coverage']:.3f}  rows={row['rows']}")

    print("\n=== RAW（未处理）按特征 train/valid/test 全表 ===")
    pivot_raw = train_cov.pivot_table(index="feature", columns="segment", values="coverage", aggfunc="mean") \
        if False else cov_raw.pivot_table(index="feature", columns="segment", values="coverage", aggfunc="mean")
    pivot_raw = pivot_raw[[c for c in ["train", "valid", "test"] if c in pivot_raw.columns]]
    pivot_raw = pivot_raw.sort_values("train") if "train" in pivot_raw.columns else pivot_raw
    print(pivot_raw.to_string(float_format=lambda x: f"{x:.3f}"))

    print(f"\n[audit] 完成。建议: train 段 coverage < {args.threshold:.2f} 的因子在 features/*.py 中设 enabled=False。")


if __name__ == "__main__":
    main()
