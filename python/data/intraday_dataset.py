"""
日内 5min 序列数据集构建（B3 架构 B / Deep Factor 共享层）
==============================================================
层级位置: qlib_data_5min → intraday_dataset → 日内时序模型训练 / Deep Factor 提取

为什么自定义而非 qlib TSDatasetH：
  qlib 的 TSDatasetH 在 5min 日历上是「逐 5min bar 滑窗」采样，得到的样本数是
  bar 数级别、窗口会跨日；而架构 B 需要的是「每个 (交易日, 股票) 一个样本，
  输入为当日 48 根 5min bar 的 [48, F] 序列」。本模块做这种日内池化重塑，
  并产出与日频 LGBM 对齐的「日度前向收益」标签，供 train / extract 复用，
  保证两端的样本顺序与索引完全一致（防泄漏、可对齐）。

标签口径（与日频主链路 base.yaml 一致）：
  label_T = close_{T+2}/close_{T+1} - 1   （T+1 收盘买入、T+2 收盘卖出）
  其中 close_* 取每日末根 5min bar 的收盘价（与日频 close 同为复权序列，
  前向收益对复权方式不敏感）。

输出（IntradayDataset）：
  X            np.float32 [G, step_len, F]   日内特征序列
  y            np.float32 [G]                日度前向收益（缺失为 NaN）
  days         np.datetime64 [G]             交易日（已 normalize 到 00:00）
  instruments  np.ndarray[str] [G]           标的代码（如 '000006.SZ'）
  feature_names list[str]                    F 个特征名
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger("data.intraday_dataset")


@dataclass
class IntradayDataset:
    X: np.ndarray
    y: np.ndarray
    days: np.ndarray
    instruments: np.ndarray
    feature_names: List[str]

    def __len__(self) -> int:
        return self.X.shape[0]

    def segment_mask(self, start: str, end: str) -> np.ndarray:
        """返回 days ∈ [start, end] 的布尔掩码。"""
        s, e = np.datetime64(pd.Timestamp(start)), np.datetime64(pd.Timestamp(end))
        return (self.days >= s) & (self.days <= e)

    def valid_label_mask(self) -> np.ndarray:
        """标签非 NaN 的布尔掩码（训练/评估用）。"""
        return ~np.isnan(self.y)


def build_intraday_dataset(
    provider_uri: str,
    instruments,
    start: str,
    end: str,
    feature_exprs: Sequence[str],
    feature_names: Sequence[str],
    step_len: int = 48,
    close_field: str = "$close",
    clip: float = 20.0,
    qlib_initialized: bool = False,
) -> IntradayDataset:
    """从 5min 库构建日内序列数据集。

    Args:
        provider_uri: 5min qlib 库路径（D:/qlib_data/qlib_data_5min）。
        instruments:  标的列表或 qlib market 名（如 'csi500'）。
        start, end:   日期区间（自然日端点）。
        feature_exprs/feature_names: 日内特征表达式与列名（intraday_raw 组）。
        step_len:     单日 bar 数，A 股 5min = 48；只保留恰好 step_len 根的交易日。
        close_field:  用于推导日度标签的收盘价字段。
        clip:         特征绝对值裁剪阈值（去除复权/缺口异常）。
        qlib_initialized: 调用方是否已 qlib.init（避免重复初始化）。

    Returns:
        IntradayDataset。
    """
    import qlib
    from qlib.data import D

    if not qlib_initialized:
        qlib.init(provider_uri=provider_uri, region="cn")

    if isinstance(instruments, str):
        inst_list = D.instruments(market=instruments)
    else:
        inst_list = list(instruments)

    fields = list(feature_exprs) + [close_field]
    names = list(feature_names) + ["_CLOSE"]
    logger.info("intraday_fetch start insts=%s range=%s~%s F=%d",
                instruments if isinstance(instruments, str) else len(inst_list), start, end, len(feature_exprs))

    df = D.features(inst_list, fields, start_time=start, end_time=end, freq="5min")
    if df is None or df.empty:
        raise ValueError("5min 特征为空：检查 provider_uri / instruments / 日期区间。")
    df.columns = names

    # 索引: (instrument, datetime)。重置并标注交易日。
    df = df.reset_index()
    inst_col = df.columns[0]   # 'instrument'
    dt_col = df.columns[1]     # 'datetime'
    df = df.rename(columns={inst_col: "instrument", dt_col: "datetime"})
    df["day"] = df["datetime"].dt.normalize()
    df = df.sort_values(["instrument", "datetime"], kind="stable")

    # 仅保留当日恰好 step_len 根 bar 的 (instrument, day)
    grp_size = df.groupby(["instrument", "day"], sort=False)["datetime"].transform("size")
    df = df[grp_size == step_len]
    if df.empty:
        raise ValueError(f"无任何交易日满足 {step_len} 根/日，检查 5min 数据完整性。")
    if len(df) % step_len != 0:
        raise RuntimeError("过滤后行数非 step_len 整数倍（不应发生）。")

    feat_names = list(feature_names)
    feat_mat = df[feat_names].to_numpy(dtype=np.float32, copy=False)
    feat_mat = np.nan_to_num(feat_mat, nan=0.0, posinf=clip, neginf=-clip)
    np.clip(feat_mat, -clip, clip, out=feat_mat)

    G = len(df) // step_len
    F = len(feat_names)
    X = feat_mat.reshape(G, step_len, F)

    # 每组首行索引 → (instrument, day)；末根收盘 → 日度收盘
    head = df.iloc[::step_len]
    g_inst = head["instrument"].to_numpy()
    g_day = head["day"].to_numpy()
    last_close = df["_CLOSE"].to_numpy()[step_len - 1::step_len]  # 每组第 48 根

    # 标签：fwd = close.shift(-2)/close.shift(-1) - 1。
    # df 已按 (instrument, datetime) 稳定排序，故组顺序天然 = (instrument, day) 升序，
    # 与 X 的第 0 维一一对应。此处 groupby(sort=False) 不改变顺序，shift 在组内沿
    # 既有(升序)顺序计算前向收益，保证 y 与 X 行对齐（不可再 sort，否则错位）。
    meta = pd.DataFrame({"instrument": g_inst, "day": g_day, "last_close": last_close})
    lc = meta.groupby("instrument", sort=False)["last_close"]
    meta["y"] = lc.shift(-2) / lc.shift(-1) - 1.0

    ds = IntradayDataset(
        X=X,
        y=meta["y"].to_numpy(dtype=np.float32),
        days=meta["day"].to_numpy(),
        instruments=meta["instrument"].to_numpy(),
        feature_names=feat_names,
    )
    logger.info("intraday_built G=%d step_len=%d F=%d label_valid=%d",
                G, step_len, F, int(ds.valid_label_mask().sum()))
    return ds
