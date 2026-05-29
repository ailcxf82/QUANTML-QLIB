"""
日内 5min 时序模型训练（B3 架构 B / Deep Factor 阶段 3）
==============================================================
按 (交易日, 股票) 把当日 48 根 5min bar 重塑为 [48, 6] 序列，训练 IntradayGRU，
标签为日度前向收益（close_{T+2}/close_{T+1}-1）。产物供 extract_deep_factor.py 提取
hidden。本脚本不做撮合回测，单模型 OOS 日度 IC 仅作 sanity（不要求达标）。

用法（数据就绪后，在本机）：
  python scripts/train_intraday_model.py \
      --model-config configs/models/gru_intraday.yaml \
      --freq-config  configs/freq/fivemin.yaml \
      --market csi500

产物：artifacts/intraday_models/<name>_<hash>/{model.pt, meta.json}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import yaml

# 绝对导入（python/ 加入 sys.path）
_PY_ROOT = Path(__file__).resolve().parents[1] / "python"
if str(_PY_ROOT) not in sys.path:
    sys.path.insert(0, str(_PY_ROOT))

import features  # noqa: E402
from data.intraday_dataset import build_intraday_dataset, IntradayDataset  # noqa: E402
from models.intraday_gru import IntradayGRU  # noqa: E402

logger = logging.getLogger("train_intraday")


def _load_yaml(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _config_hash(*objs) -> str:
    h = hashlib.sha1(json.dumps(objs, sort_keys=True, default=str).encode("utf-8"))
    return h.hexdigest()[:8]


def _daily_ic(pred: np.ndarray, y: np.ndarray, days: np.ndarray) -> float:
    """按日 Spearman(rank-IC) 的均值。"""
    df = pd.DataFrame({"pred": pred, "y": y, "day": days}).dropna()
    if df.empty:
        return float("nan")
    ics = df.groupby("day").apply(
        lambda g: g["pred"].corr(g["y"], method="spearman") if len(g) > 2 else np.nan
    )
    return float(np.nanmean(ics.to_numpy()))


def _batched_predict(model, X: np.ndarray, device, batch: int = 4096) -> np.ndarray:
    import torch

    model.eval()
    out = np.empty(X.shape[0], dtype=np.float32)
    with torch.no_grad():
        for i in range(0, X.shape[0], batch):
            xb = torch.from_numpy(X[i : i + batch]).to(device)
            out[i : i + batch] = model(xb).detach().cpu().numpy()
    return out


def train(
    ds: IntradayDataset,
    segments: Dict[str, list],
    mkw: dict,
    out_dir: Path,
) -> dict:
    import torch
    import torch.nn as nn

    seed = int(mkw.get("seed", 42))
    np.random.seed(seed)
    torch.manual_seed(seed)

    gpu = int(mkw.get("GPU", -1))
    device = torch.device(f"cuda:{gpu}" if (gpu >= 0 and torch.cuda.is_available()) else "cpu")
    logger.info("device=%s cuda_available=%s", device, torch.cuda.is_available())

    label_ok = ds.valid_label_mask()
    tr = ds.segment_mask(*segments["train"]) & label_ok
    va = ds.segment_mask(*segments["valid"]) & label_ok
    te = ds.segment_mask(*segments["test"]) & label_ok
    logger.info("split train=%d valid=%d test=%d", tr.sum(), va.sum(), te.sum())
    if tr.sum() == 0 or va.sum() == 0:
        raise ValueError("train/valid 样本为空，检查 segments 与数据区间。")

    Xtr, ytr = ds.X[tr], ds.y[tr]
    Xva, yva = ds.X[va], ds.y[va]

    model = IntradayGRU(
        d_feat=ds.X.shape[2],
        hidden_size=int(mkw["hidden_size"]),
        num_layers=int(mkw["num_layers"]),
        dropout=float(mkw.get("dropout", 0.0)),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(mkw["lr"]))
    loss_fn = nn.MSELoss()

    n_epochs = int(mkw.get("n_epochs", 60))
    batch = int(mkw.get("batch_size", 2048))
    early_stop = int(mkw.get("early_stop", 12))

    yva_t = torch.from_numpy(yva).to(device)
    Xva_t = torch.from_numpy(Xva).to(device) if Xva.nbytes < 1.5e9 else None

    best_va = float("inf")
    best_state = None
    bad = 0
    n = Xtr.shape[0]
    t0 = time.perf_counter()
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = np.random.permutation(n)
        ep_loss = 0.0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            xb = torch.from_numpy(Xtr[idx]).to(device)
            yb = torch.from_numpy(ytr[idx]).to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(idx)
        ep_loss /= n

        # valid loss
        model.eval()
        with torch.no_grad():
            if Xva_t is not None:
                va_loss = loss_fn(model(Xva_t), yva_t).item()
            else:
                pv = _batched_predict(model, Xva, device)
                va_loss = float(np.mean((pv - yva) ** 2))
        logger.info("epoch=%d train_loss=%.6f valid_loss=%.6f", epoch, ep_loss, va_loss)

        if va_loss < best_va - 1e-7:
            best_va = va_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= early_stop:
                logger.info("early_stop at epoch=%d (best_valid=%.6f)", epoch, best_va)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # test sanity IC
    test_ic = float("nan")
    if te.sum() > 0:
        pte = _batched_predict(model, ds.X[te], device)
        test_ic = _daily_ic(pte, ds.y[te], ds.days[te])
    logger.info("train_done elapsed=%.0fs best_valid_loss=%.6f test_daily_ic=%.4f",
                time.perf_counter() - t0, best_va, test_ic)

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    return {"best_valid_loss": best_va, "test_daily_ic": test_ic, "device": str(device)}


def main() -> None:
    parser = argparse.ArgumentParser(description="日内 5min GRU 训练（Deep Factor 阶段3）")
    parser.add_argument("--model-config", default="configs/models/gru_intraday.yaml")
    parser.add_argument("--freq-config", default="configs/freq/fivemin.yaml")
    parser.add_argument("--market", default="csi500", help="训练 universe（5min 库内的 instruments 名）")
    parser.add_argument("--out-root", default="artifacts/intraday_models")
    parser.add_argument("--allow-not-ready", action="store_true", help="忽略 freq data_ready=false")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    mcfg = _load_yaml(args.model_config)
    fcfg = _load_yaml(args.freq_config)

    if fcfg.get("freq_meta", {}).get("data_ready") is False and not args.allow_not_ready:
        raise SystemExit(
            f"freq 配置 {args.freq_config} data_ready=false：请先 dump+校验 5min 数据并改为 true，"
            f"或加 --allow-not-ready 强制运行。"
        )

    provider_uri = fcfg["qlib_init"]["provider_uri"]
    segments = fcfg["dataset_segments"]
    groups = mcfg["model_meta"]["feature_groups"]
    step_len = int(mcfg["model_meta"]["step_len"])
    mkw = mcfg["model"]["kwargs"]

    exprs, names = features.build_feature_config(groups=groups, include_rdagent=False)
    logger.info("provider=%s groups=%s F=%d step_len=%d", provider_uri, groups, len(names), step_len)

    ds = build_intraday_dataset(
        provider_uri=provider_uri,
        instruments=args.market,
        start=segments["train"][0],
        end=segments["test"][1],
        feature_exprs=exprs,
        feature_names=names,
        step_len=step_len,
    )

    cfg_hash = _config_hash(mcfg, segments, args.market, names)
    name = mcfg["model_meta"]["name"]
    out_dir = Path(args.out_root) / f"{name}_{cfg_hash}"

    stats = train(ds, segments, mkw, out_dir)

    meta = {
        "name": name,
        "config_hash": cfg_hash,
        "provider_uri": provider_uri,
        "market": args.market,
        "feature_groups": groups,
        "feature_names": names,
        "step_len": step_len,
        "d_feat": len(names),
        "hidden_size": int(mkw["hidden_size"]),
        "num_layers": int(mkw["num_layers"]),
        "dropout": float(mkw.get("dropout", 0.0)),
        "segments": segments,
        "label": "close_{T+2}/close_{T+1}-1",
        **stats,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("name", "config_hash", "test_daily_ic", "best_valid_loss")},
                     ensure_ascii=False))
    print(f"[train] saved → {out_dir}")


if __name__ == "__main__":
    main()
