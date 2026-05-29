"""
Deep Factor 提取 + PCA 降维（B3 架构 B / 阶段 4）
==============================================================
加载已训练的日内 GRU，对每个 (交易日, 股票) 前向取 hidden out[:, -1, :]（hidden_size 维），
PCA 仅在 train segment 上 fit（防泄漏）后 transform 全量，降到 k 维，输出 deep_factor.parquet
（MultiIndex(datetime, instrument) × DF0..DF{k-1}），供 LGBM 经 StaticDataLoader 注入。

防泄漏要点：
  - PCA 与标准化 scaler 只用 train segment 行拟合；
  - deep factor 在 day T 的取值仅由 day T 的日内 bar 决定，标签用 T+1→T+2，无未来函数；
  - datetime 取交易日（00:00），与日频 LGBM 索引对齐。

用法：
  python scripts/extract_deep_factor.py \
      --model-dir artifacts/intraday_models/gru_intraday_<hash> --k 8
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PY_ROOT = Path(__file__).resolve().parents[1] / "python"
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

from data.intraday_dataset import build_intraday_dataset  # noqa: E402
from models.intraday_gru import IntradayGRU  # noqa: E402

logger = logging.getLogger("extract_deep_factor")


def _extract_hidden(model, X: np.ndarray, device, batch: int = 4096) -> np.ndarray:
    import torch

    model.eval()
    H = np.empty((X.shape[0], model.hidden_size), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[i : i + batch]).to(device)
            H[i : i + batch] = model.extract_hidden(xb).detach().cpu().numpy()
    return H


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep Factor 提取 + train-fit PCA")
    parser.add_argument("--model-dir", required=True, help="artifacts/intraday_models/<name>_<hash>")
    parser.add_argument("--k", type=int, default=8, help="PCA 目标维度")
    parser.add_argument("--out-root", default="artifacts/deep_factors")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    import torch
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    model_dir = Path(args.model_dir)
    meta = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))

    # 用与训练完全一致的特征与区间重建数据集（顺序对齐）
    import features  # noqa: E402

    exprs, names = features.build_feature_config(groups=meta["feature_groups"], include_rdagent=False)
    if names != meta["feature_names"]:
        raise RuntimeError("特征名与训练时不一致，注册表可能已变更，提取中止以防错位。")

    segments = meta["segments"]
    ds = build_intraday_dataset(
        provider_uri=meta["provider_uri"],
        instruments=meta["market"],
        start=segments["train"][0],
        end=segments["test"][1],
        feature_exprs=exprs,
        feature_names=names,
        step_len=int(meta["step_len"]),
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = IntradayGRU(
        d_feat=int(meta["d_feat"]),
        hidden_size=int(meta["hidden_size"]),
        num_layers=int(meta["num_layers"]),
        dropout=float(meta.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device))
    logger.info("model loaded hidden=%d device=%s", model.hidden_size, device)

    H = _extract_hidden(model, ds.X, device)
    logger.info("hidden extracted H=%s", H.shape)

    # 防泄漏：scaler + PCA 仅在 train segment 行拟合
    train_mask = ds.segment_mask(*segments["train"])
    k = min(int(args.k), H.shape[1])
    scaler = StandardScaler().fit(H[train_mask])
    pca = PCA(n_components=k, random_state=42).fit(scaler.transform(H[train_mask]))
    DF = pca.transform(scaler.transform(H)).astype(np.float32)
    evr = pca.explained_variance_ratio_
    logger.info("pca k=%d fit_rows=%d explained_var=%.3f", k, int(train_mask.sum()), float(evr.sum()))

    cols = [f"DF{i}" for i in range(k)]
    idx = pd.MultiIndex.from_arrays(
        [pd.to_datetime(ds.days), ds.instruments], names=["datetime", "instrument"]
    )
    df = pd.DataFrame(DF, index=idx, columns=cols).sort_index()

    out_dir = Path(args.out_root) / f"{meta['name']}_{meta['config_hash']}_k{k}"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "deep_factor.parquet"
    df.to_parquet(parquet_path)

    out_meta = {
        "source_model_dir": str(model_dir),
        "source_config_hash": meta["config_hash"],
        "k": k,
        "hidden_size": int(meta["hidden_size"]),
        "columns": cols,
        "explained_variance_ratio": [float(x) for x in evr],
        "explained_variance_sum": float(evr.sum()),
        "pca_fit_segment": segments["train"],
        "pca_fit_rows": int(train_mask.sum()),
        "n_rows": int(len(df)),
        "date_range": f"{df.index.get_level_values('datetime').min()}~{df.index.get_level_values('datetime').max()}",
        "n_instruments": int(df.index.get_level_values("instrument").nunique()),
        "label_note": "deep factor@day T 仅用 T 当日 bar；LGBM 标签 T+1→T+2，无泄漏",
        "segments": segments,
    }
    (out_dir / "meta.json").write_text(json.dumps(out_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({"parquet": str(parquet_path), "k": k,
                      "explained_var": round(float(evr.sum()), 3),
                      "n_rows": int(len(df))}, ensure_ascii=False))
    print(f"[extract] saved → {out_dir}")


if __name__ == "__main__":
    main()
