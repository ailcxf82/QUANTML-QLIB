"""
实盘预测脚本（Live Prediction）
================================
层级位置: Model → Signal → 实盘选股输出（无回测，无自动下单）

功能:
  - predict-only  模式: 加载已训练模型，对指定日期生成 topK 选股推荐
  - retrain-and-predict 模式: 用最新数据重训 LGBM，保存模型，再生成预测

输入:
  --model lgbm --freq daily --mode predict-only [--date YYYY-MM-DD]

输出（predictions/<date>/）:
  selection.csv     主推荐列表（code/score/rank/weight/change/name）
  pred_score.parquet  csi500 内全量打分（供归因分析）
  model_meta.json   使用的模型版本信息
  report.html       人可读 HTML 报告

约束:
  - 数据截止 --date 当日，不使用未来数据
  - 选股范围限定 tradable_universe（csi500），与回测口径一致
  - 所有中间结果持久化到 predictions/<date>/，方便历史回溯
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# 将 python/ 目录加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import build_feature_config
from features.combined_json_factors import merge_features_from_combined_json

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_BASE = WORKSPACE_ROOT / "predictions"
LIVE_CFG_PATH = WORKSPACE_ROOT / "configs" / "live" / "daily_live.yaml"
VERSION = "v1"


# ─────────────────────────────────────────────────────────────────────────────
# 配置加载工具
# ─────────────────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_live_config(
    model_name: str,
    freq_name: str,
    predict_date: str,
) -> Dict[str, Any]:
    """加载并合并实盘配置（base.yaml + models/<model>.yaml + live/daily_live.yaml）。

    注: 实盘不使用 freq/daily.yaml（含 dataset_segments），
    改用 live/daily_live.yaml 并由本函数动态注入 segments。
    """
    configs_dir = WORKSPACE_ROOT / "configs"

    base_cfg = _load_yaml(configs_dir / "base.yaml")
    model_cfg_path = configs_dir / "models" / f"{model_name}.yaml"
    if not model_cfg_path.exists():
        raise FileNotFoundError(f"模型配置不存在: {model_cfg_path}")
    model_cfg = _load_yaml(model_cfg_path)
    live_cfg = _load_yaml(LIVE_CFG_PATH)

    cfg = _deep_merge(base_cfg, model_cfg)
    cfg = _deep_merge(cfg, live_cfg)

    # 动态注入 end_time = predict_date
    cfg["experiment"]["end_time"] = predict_date
    cfg["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = predict_date

    # 实盘 segments：train 覆盖全段历史，"test" = [predict_date, predict_date]
    train_start = cfg["experiment"].get("start_time", "2020-01-01")
    # train 段结束 = predict_date 前一个自然日
    train_end_dt = pd.Timestamp(predict_date) - pd.Timedelta(days=1)
    train_end = train_end_dt.strftime("%Y-%m-%d")

    cfg["dataset"]["kwargs"]["segments"] = {
        "train": [train_start, train_end],
        "valid": [train_start, train_end],  # 实盘不用 valid，同 train 避免空段
        "test":  [predict_date, predict_date],
    }

    # 注入 model_meta（freq 信息）
    cfg.setdefault("model_meta", {})
    cfg["model_meta"]["name"] = model_name
    cfg["model_meta"].setdefault("dataset_type", "flat")
    cfg["model_meta"].setdefault("freq_key", "1day")
    cfg.setdefault("freq_meta", {})
    cfg["freq_meta"]["name"] = freq_name

    # 注入 run_id（实盘标识）
    cfg["experiment"]["run_id"] = f"{model_name}_{freq_name}_live"

    return cfg


def inject_feature_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """注入动态因子特征（复用 run_experiment.py 的逻辑）。"""
    cfg = copy.deepcopy(cfg)
    feature_groups: Optional[List[str]] = cfg.get("model_meta", {}).get("feature_groups")
    exprs, names = build_feature_config(groups=feature_groups, include_rdagent=True)
    combined_rel = cfg.get("model_meta", {}).get("combined_factors_json")
    if combined_rel:
        merge_features_from_combined_json(exprs, names, WORKSPACE_ROOT / str(combined_rel))

    dl_cfg = cfg["dataset"]["kwargs"]["handler"]["kwargs"]["data_loader"]["kwargs"]["config"]
    dl_cfg["feature"] = [exprs, names]

    # 同步模型维度
    n_feat = len(names)
    model_kwargs = cfg.get("model", {}).get("kwargs", {})
    for dim_key in ("input_dim", "d_feat"):
        if dim_key in model_kwargs:
            model_kwargs[dim_key] = n_feat
    pt_kwargs = model_kwargs.get("pt_model_kwargs", {})
    if "input_dim" in pt_kwargs:
        pt_kwargs["input_dim"] = n_feat

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# 最近交易日推断
# ─────────────────────────────────────────────────────────────────────────────

def _last_trading_day(provider_uri: str) -> str:
    """从 calendars/day.txt 读取最后一个交易日（Qlib 数据包含的最新日期）。"""
    cal_path = Path(provider_uri) / "calendars" / "day.txt"
    if not cal_path.exists():
        raise FileNotFoundError(f"calendars/day.txt 不存在: {cal_path}")
    lines = [l.strip() for l in cal_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not lines:
        raise RuntimeError("calendars/day.txt 为空")
    return lines[-1]


# ─────────────────────────────────────────────────────────────────────────────
# 模型加载 / 保存
# ─────────────────────────────────────────────────────────────────────────────

def load_latest_model(model_name: str, freq_name: str) -> Tuple[Any, Dict[str, Any]]:
    """加载 artifacts/models/latest_<model>_<freq>.pkl 及对应 meta。

    Returns:
        (model, meta_dict)
    """
    try:
        import joblib
    except ImportError as exc:
        raise ImportError("未安装 joblib，请执行: pip install joblib") from exc

    models_dir = WORKSPACE_ROOT / "artifacts" / "models"
    model_path = models_dir / f"latest_{model_name}_{freq_name}.pkl"
    meta_path = models_dir / f"latest_{model_name}_{freq_name}_meta.json"

    if not model_path.exists():
        raise FileNotFoundError(
            f"模型文件不存在: {model_path}\n"
            "请先运行 run_experiment.py 训练并保存模型，或使用 --mode retrain-and-predict。"
        )

    model = joblib.load(model_path)
    meta: Dict[str, Any] = {}
    if meta_path.exists():
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)

    print(f"  模型加载: {model_path}")
    print(f"  训练截止: {meta.get('train_end', '未知')}  版本: {meta.get('run_id', '未知')}")
    return model, meta


def train_and_save_model(
    cfg: Dict[str, Any],
    dataset: Any,
    model_name: str,
    freq_name: str,
    run_id: str,
) -> Tuple[Any, Dict[str, Any]]:
    """重训 LGBM 并保存到 artifacts/models/latest_*.pkl。"""
    try:
        import joblib
        from qlib.utils import init_instance_by_config
    except ImportError as exc:
        raise ImportError(f"依赖缺失: {exc}") from exc

    print(f"  [retrain] 开始训练 {model_name}...")
    t0 = time.perf_counter()
    model = init_instance_by_config(cfg["model"])
    model.fit(dataset)
    train_sec = round(time.perf_counter() - t0, 2)
    print(f"  [retrain] 训练完成 [{train_sec}s]")

    train_seg = cfg.get("dataset", {}).get("kwargs", {}).get("segments", {}).get("train", [])
    train_end = train_seg[1] if len(train_seg) >= 2 else "unknown"

    models_dir = WORKSPACE_ROOT / "artifacts" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    latest_path = models_dir / f"latest_{model_name}_{freq_name}.pkl"
    joblib.dump(model, latest_path)

    meta = {
        "run_id": run_id,
        "model": model_name,
        "freq": freq_name,
        "train_end": train_end,
        "train_sec": train_sec,
        "saved_at": pd.Timestamp.now().isoformat(),
        "version": VERSION,
    }
    meta_path = models_dir / f"latest_{model_name}_{freq_name}_meta.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"  [retrain] 模型已保存: {latest_path}")
    return model, meta


# ─────────────────────────────────────────────────────────────────────────────
# 信号生成
# ─────────────────────────────────────────────────────────────────────────────

def generate_pred_score(
    model: Any,
    cfg: Dict[str, Any],
    predict_date: str,
) -> pd.Series:
    """用已训练模型对 predict_date 的全市场生成预测分。

    Returns:
        pred_score: MultiIndex(datetime, instrument) 的 pd.Series，值为模型打分
    """
    from qlib.utils import init_instance_by_config

    print(f"  构建数据集（test=[{predict_date}, {predict_date}]）...")
    t0 = time.perf_counter()
    dataset = init_instance_by_config(cfg["dataset"])
    ds_sec = round(time.perf_counter() - t0, 2)
    print(f"  数据集就绪 [{ds_sec}s]")

    pred = model.predict(dataset, segment="test")
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    pred = pred.rename("score")

    if pred.empty:
        raise RuntimeError(f"预测结果为空，请检查 {predict_date} 数据是否存在。")
    return pred


def filter_to_universe(
    pred_score: pd.Series,
    universe: str,
    predict_date: str,
    run_id: str = "live",
) -> pd.Series:
    """将预测打分限定到可投资 universe（csi500）。"""
    if not universe or universe.lower() == "all":
        return pred_score
    try:
        import qlib
        from qlib.data import D
        inst_cfg = {"market": universe}
        tradable = D.list_instruments(
            inst_cfg, start_time=predict_date, end_time=predict_date, as_list=True
        )
        if not tradable:
            print(f"  警告: {universe} 在 {predict_date} 为空，跳过过滤")
            return pred_score
        tradable_set = set(tradable)
        mask = pred_score.index.get_level_values(1).isin(tradable_set)
        filtered = pred_score[mask]
        print(f"  universe 过滤: {universe} 保留 {mask.sum()} / {len(pred_score)} 条")
        return filtered
    except Exception as exc:
        print(f"  警告: universe 过滤失败（{exc}），使用全量预测")
        return pred_score


# ─────────────────────────────────────────────────────────────────────────────
# 特殊股票过滤（ST / 连续涨停）— 统一使用 StockFilter 模块
# ─────────────────────────────────────────────────────────────────────────────

def _build_stock_filter(live_cfg_raw: Dict[str, Any]) -> Any:
    """从 live 配置构建 StockFilter 实例。

    ST CSV 路径优先读取 filter.st_csv_path，
    其次自动拼接 qlib_init.provider_uri/is_st.csv。
    """
    from backtest.strategy.stock_filter import StockFilter

    filter_cfg = live_cfg_raw.get("filter", {})
    provider_uri = live_cfg_raw.get("qlib_init", {}).get("provider_uri", "")

    # 优先用显式配置，其次自动推断
    st_csv = filter_cfg.get("st_csv_path") or (
        str(Path(provider_uri) / "is_st.csv") if provider_uri else None
    )
    consecutive_limit_days = int(filter_cfg.get("consecutive_limit_days", 3))

    return StockFilter(
        st_csv_path=st_csv,
        consecutive_limit_days=consecutive_limit_days,
    )


def filter_and_refill(
    selection: pd.DataFrame,
    pred_score: pd.Series,
    predict_date: str,
    topk: int,
    score_quantile: float = 0.0,
    min_score_threshold: Optional[float] = None,
    stock_filter: Optional[Any] = None,
    exclude_st: bool = True,
    exclude_limit_up: bool = True,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """过滤 ST 和连续涨停股票，并从剩余候选池中补充替代股票。

    过滤逻辑：
      - ST 类：从 is_st.csv 按日期查询，财务风险高，日限幅仅 ±5%
      - 连续涨停：封单状态大概率无法在 T+1 买入，泡沫回撤风险高

    Args:
        selection:     select_topk 的初步结果（不含 drop_out 行）
        pred_score:    universe 内全量打分（MultiIndex: datetime/instrument）
        predict_date:  预测日期
        topk:          目标持仓数量
        stock_filter:  StockFilter 实例（None 则跳过过滤）
        exclude_st:    是否过滤 ST 股票
        exclude_limit_up: 是否过滤连续涨停股票

    Returns:
        (updated_selection, exclusion_log)
    """
    exclusion_log: List[Dict[str, Any]] = []

    if stock_filter is None:
        # 无过滤器：直接截取 topk
        result_df = selection.head(topk).copy()
        result_df["rank"] = range(1, len(result_df) + 1)
        return result_df, exclusion_log

    # ── 构建今日截面完整排序列表（用于补充替代股） ─────────────────
    if isinstance(pred_score.index, pd.MultiIndex):
        dt_level = pred_score.index.get_level_values("datetime")
        date_ts = pd.Timestamp(predict_date)
        avail = sorted(dt_level.unique())
        actual_date = max((d for d in avail if d <= date_ts), default=avail[-1])
        scores_today = pred_score[dt_level == actual_date].copy()
        scores_today.index = scores_today.index.get_level_values("instrument")
    else:
        scores_today = pred_score.copy()

    scores_today = scores_today.dropna().sort_values(ascending=False)
    if score_quantile > 0:
        scores_today = scores_today[scores_today >= scores_today.quantile(score_quantile)]
    if min_score_threshold is not None:
        scores_today = scores_today[scores_today >= min_score_threshold]

    # ── 批量检测候选池（当前选股 + topk*5 备用池） ──────────────────
    check_pool = scores_today.index.tolist()[: topk * 6]
    print(f"  [过滤] 批量检测候选池 {len(check_pool)} 支...")
    excluded_reasons = stock_filter.get_excluded_with_reasons(
        check_pool,
        date=predict_date,
        check_st=exclude_st,
        check_limit_up=exclude_limit_up,
    )
    if excluded_reasons:
        print(f"  [过滤] 命中 {len(excluded_reasons)} 支: {list(excluded_reasons.keys())}")
    else:
        print("  [过滤] 候选池内未发现 ST / 连续涨停股票")

    # ── 过滤当前选股 ──────────────────────────────────────────────
    keep_rows: List[pd.Series] = []
    excluded_instruments: set = set()
    for _, row in selection.iterrows():
        inst = str(row["instrument"])
        reason = excluded_reasons.get(inst)
        if reason:
            exclusion_log.append(
                {
                    "instrument": inst,
                    "reason": reason,
                    "original_rank": int(row.get("rank", 999)),
                    "score": float(row["score"]),
                    "replacement": None,
                }
            )
            excluded_instruments.add(inst)
            print(f"  [过滤] 排除 {inst}（原排名 {int(row.get('rank', 999))}）: {reason}")
        else:
            keep_rows.append(row)

    result_df = (
        pd.DataFrame(keep_rows).reset_index(drop=True)
        if keep_rows
        else pd.DataFrame(columns=selection.columns)
    )

    # ── 从候选池中补充替代股票 ────────────────────────────────────
    already_selected = set(result_df["instrument"].tolist()) if not result_df.empty else set()
    need_fill = topk - len(result_df)
    filled_count = 0
    log_idx = 0

    if need_fill > 0:
        print(f"  [过滤] 需补充 {need_fill} 支替代股票...")
        for inst, score in scores_today.items():
            if filled_count >= need_fill:
                break
            inst_str = str(inst)
            if inst_str in already_selected or inst_str in excluded_instruments:
                continue
            if inst_str in excluded_reasons:
                excluded_instruments.add(inst_str)
                continue

            pct = round(float((scores_today < score).mean()) * 100, 1)
            new_rank = len(keep_rows) + filled_count + 1
            new_row = {
                "instrument": inst_str,
                "score": float(score),
                "rank": new_rank,
                "suggested_weight": round(1.0 / topk, 4),
                "percentile": pct,
                "predict_date": predict_date,
            }
            if log_idx < len(exclusion_log):
                exclusion_log[log_idx]["replacement"] = inst_str
                log_idx += 1

            result_df = pd.concat([result_df, pd.DataFrame([new_row])], ignore_index=True)
            already_selected.add(inst_str)
            filled_count += 1
            print(f"  [过滤] 补充替代股 {inst_str}（新排名 {new_rank}，打分={score:.4f}）")

        if filled_count < need_fill:
            print(f"  [过滤] 警告：仅补充 {filled_count} 支，不足 {need_fill} 支")

    # ── 重新排名 & 权重 ───────────────────────────────────────────
    if not result_df.empty:
        result_df = result_df.sort_values("score", ascending=False).reset_index(drop=True)
        result_df["rank"] = range(1, len(result_df) + 1)
        result_df["suggested_weight"] = round(1.0 / max(len(result_df), 1), 4)

    return result_df, exclusion_log


# ─────────────────────────────────────────────────────────────────────────────
# 选股逻辑
# ─────────────────────────────────────────────────────────────────────────────

def select_topk(
    pred_score: pd.Series,
    predict_date: str,
    topk: int = 5,
    score_quantile: float = 0.0,
    min_score_threshold: Optional[float] = None,
) -> pd.DataFrame:
    """从 pred_score 中提取指定日期的截面打分，输出 topK 推荐 DataFrame。

    Returns:
        DataFrame: [instrument, score, rank, suggested_weight, percentile]
    """
    # 取指定日期的截面
    if isinstance(pred_score.index, pd.MultiIndex):
        dt_level = pred_score.index.get_level_values("datetime")
        date_ts = pd.Timestamp(predict_date)
        mask = dt_level == date_ts
        if not mask.any():
            # 尝试最近有效日期
            available = sorted(dt_level.unique())
            if available:
                date_ts = available[-1]
                mask = dt_level == date_ts
                print(f"  注意: {predict_date} 无数据，使用最近可用日期 {date_ts.date()}")
            else:
                raise RuntimeError(f"pred_score 为空，无法选股")
        scores_today = pred_score[mask]
        scores_today.index = scores_today.index.get_level_values("instrument")
    else:
        scores_today = pred_score

    scores_today = scores_today.dropna().sort_values(ascending=False)

    if len(scores_today) == 0:
        return pd.DataFrame(columns=["instrument", "score", "rank", "suggested_weight", "percentile"])

    # 分位过滤（在全截面内）
    if score_quantile > 0:
        threshold = scores_today.quantile(score_quantile)
        scores_today = scores_today[scores_today >= threshold]

    # 最低分阈值过滤
    if min_score_threshold is not None:
        scores_today = scores_today[scores_today >= min_score_threshold]

    # 取 topK
    topk_scores = scores_today.head(topk)

    # 计算百分位（在 csi500 全截面内的排名百分位）
    all_sorted = scores_today.rank(pct=True, ascending=True)

    # 构建结果 DataFrame
    result = pd.DataFrame({
        "instrument": topk_scores.index,
        "score": topk_scores.values,
    })
    result["rank"] = range(1, len(result) + 1)

    # 建议权重（等权分配）
    result["suggested_weight"] = round(1.0 / max(len(result), 1), 4)

    # 在 universe 内的百分位
    result["percentile"] = result["instrument"].map(
        lambda inst: round(float(all_sorted.get(inst, 0.5)) * 100, 1)
    )

    result["predict_date"] = str(date_ts.date())
    return result.reset_index(drop=True)


def load_prev_selection(predict_date: str) -> Optional[pd.DataFrame]:
    """加载前一个有效预测日的 selection.csv（用于计算 change 字段）。"""
    preds_dir = PREDICTIONS_BASE
    if not preds_dir.exists():
        return None

    # 找最近的历史预测目录（排除当天）
    date_dirs = sorted(
        [d for d in preds_dir.iterdir()
         if d.is_dir() and d.name < predict_date and (d / "selection.csv").exists()],
        reverse=True,
    )
    if not date_dirs:
        return None

    prev_path = date_dirs[0] / "selection.csv"
    try:
        df = pd.read_csv(prev_path, dtype={"instrument": str})
        return df
    except Exception:
        return None


def add_change_field(
    selection: pd.DataFrame,
    prev_selection: Optional[pd.DataFrame],
    pred_score_today: pd.Series,
    drop_signal_quantile: float = 0.3,
) -> pd.DataFrame:
    """追加 change（new_in/hold/drop_out）和 prev_score 字段。

    drop_out 逻辑：前日持仓股票，若今日打分低于 csi500 截面 drop_signal_quantile 分位，
    标记为 drop_out（建议减仓/清仓）。
    """
    selection = selection.copy()
    today_codes = set(selection["instrument"].tolist())

    if prev_selection is None or prev_selection.empty:
        selection["change"] = "new_in"
        selection["prev_score"] = float("nan")
        return selection

    prev_codes = set(prev_selection["instrument"].tolist())
    prev_score_map: Dict[str, float] = dict(
        zip(prev_selection["instrument"], prev_selection.get("score", pd.Series(dtype=float)))
    )

    selection["change"] = selection["instrument"].apply(
        lambda c: "hold" if c in prev_codes else "new_in"
    )
    selection["prev_score"] = selection["instrument"].map(
        lambda c: prev_score_map.get(c, float("nan"))
    )

    # 计算今日全截面分位，识别 drop_out（前日持仓但今日不在 topK）
    if isinstance(pred_score_today.index, pd.MultiIndex):
        today_flat = pred_score_today.copy()
        today_flat.index = today_flat.index.get_level_values("instrument")
    else:
        today_flat = pred_score_today

    if len(today_flat) > 0:
        q_threshold = today_flat.quantile(drop_signal_quantile)
        drop_out_candidates = [
            c for c in prev_codes
            if c not in today_codes and today_flat.get(c, float("nan")) < q_threshold
        ]
    else:
        drop_out_candidates = list(prev_codes - today_codes)

    # 拼接 drop_out 行
    if drop_out_candidates:
        drop_rows = []
        for code in drop_out_candidates:
            score_now = float(today_flat.get(code, float("nan")))
            score_prev = float(prev_score_map.get(code, float("nan")))
            pct = 0.0
            if len(today_flat) > 0 and not math.isnan(score_now):
                pct = round(float((today_flat < score_now).mean()) * 100, 1)
            drop_rows.append({
                "instrument": code,
                "score": score_now,
                "rank": 999,
                "suggested_weight": 0.0,
                "percentile": pct,
                "predict_date": selection["predict_date"].iloc[0] if len(selection) > 0 else "",
                "change": "drop_out",
                "prev_score": score_prev,
            })
        selection = pd.concat([selection, pd.DataFrame(drop_rows)], ignore_index=True)

    return selection


def add_stock_names(selection: pd.DataFrame) -> pd.DataFrame:
    """尝试从 Qlib 查询股票名称，失败则用空字符串填充。"""
    selection = selection.copy()
    selection["name"] = ""
    try:
        import qlib
        from qlib.data import D
        for idx, row in selection.iterrows():
            try:
                # Qlib 不直接提供中文名，此处留位置供扩展
                selection.at[idx, "name"] = row["instrument"]
            except Exception:
                pass
    except Exception:
        pass
    return selection


# ─────────────────────────────────────────────────────────────────────────────
# 持久化
# ─────────────────────────────────────────────────────────────────────────────

def persist_prediction(
    predict_date: str,
    selection: pd.DataFrame,
    pred_score: pd.Series,
    model_meta: Dict[str, Any],
    live_cfg: Dict[str, Any],
    exclusion_log: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """将今日预测结果写入 predictions/<date>/ 目录。

    Returns:
        output_dir: 实际写入的目录路径
    """
    output_dir = PREDICTIONS_BASE / predict_date
    output_dir.mkdir(parents=True, exist_ok=True)

    # selection.csv
    selection.to_csv(output_dir / "selection.csv", index=False, encoding="utf-8-sig")

    # pred_score.parquet（全量打分）
    if not pred_score.empty:
        pred_score.to_frame("score").to_parquet(output_dir / "pred_score.parquet")

    # model_meta.json
    with (output_dir / "model_meta.json").open("w", encoding="utf-8") as f:
        json.dump(model_meta, f, ensure_ascii=False, indent=2)

    # exclusion_log.json（过滤记录，方便审计）
    if exclusion_log:
        with (output_dir / "exclusion_log.json").open("w", encoding="utf-8") as f:
            json.dump(exclusion_log, f, ensure_ascii=False, indent=2)

    # report.html
    html = _build_report_html(
        predict_date, selection, pred_score, model_meta, live_cfg,
        exclusion_log=exclusion_log or [],
    )
    (output_dir / "report.html").write_text(html, encoding="utf-8")

    return output_dir


# ─────────────────────────────────────────────────────────────────────────────
# HTML 报告生成
# ─────────────────────────────────────────────────────────────────────────────

def _build_report_html(
    predict_date: str,
    selection: pd.DataFrame,
    pred_score: pd.Series,
    model_meta: Dict[str, Any],
    live_cfg: Dict[str, Any],
    exclusion_log: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """生成完整 HTML 预测报告。"""

    topk_df = selection[selection["change"] != "drop_out"].copy()
    drop_df = selection[selection["change"] == "drop_out"].copy()
    history_html = _build_history_table(predict_date, live_cfg)
    exclusion_log = exclusion_log or []

    def _change_badge(c: str) -> str:
        if c == "new_in":
            return '<span style="background:#2ecc71;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px">新入</span>'
        if c == "drop_out":
            return '<span style="background:#e74c3c;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px">退出</span>'
        return '<span style="background:#3498db;color:#fff;padding:2px 8px;border-radius:4px;font-size:11px">持续</span>'

    def _score_bar(score: float, max_score: float) -> str:
        if math.isnan(score) or max_score == 0:
            return "—"
        pct = min(100, round(abs(score / max_score) * 100))
        color = "#2ecc71" if score > 0 else "#e74c3c"
        return (
            f'<div style="display:flex;align-items:center;gap:6px">'
            f'<div style="width:{pct}px;height:12px;background:{color};border-radius:3px;min-width:4px"></div>'
            f'<span>{score:.4f}</span></div>'
        )

    # 今日推荐表
    max_s = float(topk_df["score"].max()) if not topk_df.empty else 1.0
    rec_rows = ""
    for _, row in topk_df.iterrows():
        prev_s = row.get("prev_score", float("nan"))
        trend = ""
        if not math.isnan(float(prev_s if prev_s is not None else float("nan"))):
            delta = float(row["score"]) - float(prev_s)
            trend = f'<span style="color:{"#2ecc71" if delta>=0 else "#e74c3c"}">{delta:+.4f}</span>'
        rec_rows += f"""
        <tr>
          <td style="font-weight:600">{row.get('name', row['instrument'])}<br>
            <small style="color:#999;font-weight:normal">{row['instrument']}</small></td>
          <td style="text-align:center">{int(row['rank'])}</td>
          <td>{_score_bar(float(row['score']), max_s)}</td>
          <td style="text-align:center">{row.get('percentile',0):.1f}%</td>
          <td style="text-align:center">{float(row['suggested_weight']):.1%}</td>
          <td style="text-align:center">{_change_badge(row['change'])}</td>
          <td style="text-align:center">{trend if trend else '—'}</td>
        </tr>"""

    # 退出提示表
    drop_rows_html = ""
    for _, row in drop_df.iterrows():
        score_str = f"{float(row['score']):.4f}" if not math.isnan(float(row['score'])) else "—"
        drop_rows_html += f"""
        <tr>
          <td>{row['instrument']}</td>
          <td style="text-align:center">{score_str}</td>
          <td style="text-align:center">{row.get('percentile', 0):.1f}%</td>
          <td style="text-align:center">
            {f"{float(row['prev_score']):.4f}" if not math.isnan(float(row.get('prev_score', float('nan')))) else '—'}
          </td>
        </tr>"""

    drop_section = ""
    if drop_rows_html:
        drop_section = f"""
    <div class="section">
      <h2>建议减仓/退出标的</h2>
      <p style="color:#666;font-size:12px;margin:0 0 12px">
        以下标的为昨日推荐但今日打分明显下滑（低于 {live_cfg.get('live',{}).get('drop_signal_quantile',0.3)*100:.0f} 分位），建议关注风险。
      </p>
      <table>
        <tr><th>股票代码</th><th>今日打分</th><th>今日百分位</th><th>昨日打分</th></tr>
        {drop_rows_html}
      </table>
    </div>"""

    # 过滤说明区块
    filter_section = ""
    if exclusion_log:
        filter_rows_html = ""
        for entry in exclusion_log:
            replacement = entry.get("replacement") or "—"
            filter_rows_html += f"""
        <tr>
          <td style="font-weight:600">{entry['instrument']}</td>
          <td style="text-align:center">{entry['original_rank']}</td>
          <td style="color:{{'ST' in entry['reason'] and '#e67e22' or '#e74c3c'}}">{entry['reason']}</td>
          <td style="text-align:center;font-weight:600">{replacement}</td>
        </tr>"""
        filter_section = f"""
    <div class="section">
      <h2>过滤与替换说明</h2>
      <p style="color:#666;font-size:12px;margin:0 0 12px">
        以下股票因 ST 风险或连续涨停被从推荐列表中移除，已自动补充得分次高的替代股票。
      </p>
      <table>
        <tr>
          <th>被排除股票</th>
          <th>原始排名</th>
          <th>排除原因</th>
          <th>替代股票</th>
        </tr>
        {filter_rows_html}
      </table>
    </div>"""

    # 信号分布摘要
    if not pred_score.empty:
        try:
            if isinstance(pred_score.index, pd.MultiIndex):
                flat = pred_score.copy()
                flat.index = flat.index.get_level_values("instrument")
            else:
                flat = pred_score
            n_pos = int((flat > 0).sum())
            n_neg = int((flat <= 0).sum())
            score_mean = float(flat.mean())
            score_std = float(flat.std())
            dist_html = (
                f"<span>总标的数 {len(flat)}</span> &nbsp;|&nbsp; "
                f"<span>正信号 {n_pos} ({n_pos/len(flat)*100:.1f}%)</span> &nbsp;|&nbsp; "
                f"<span>负信号 {n_neg}</span> &nbsp;|&nbsp; "
                f"<span>均值 {score_mean:.4f} ± {score_std:.4f}</span>"
            )
        except Exception:
            dist_html = "信号分布计算失败"
    else:
        dist_html = "—"

    trained_end = model_meta.get("train_end", "未知")
    run_id_str = model_meta.get("run_id", "未知")
    saved_at = model_meta.get("saved_at", "未知")[:19] if model_meta.get("saved_at") else "未知"
    topk_n = live_cfg.get("live", {}).get("topk", 5)
    universe = live_cfg.get("experiment", {}).get("tradable_universe", "csi500")

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>实盘预测报告 {predict_date}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
      margin: 0; background: #f0f2f5; color: #1a1a2e; font-size: 14px;
    }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
    .header {{
      background: linear-gradient(135deg, #1a1f3c 0%, #2d3561 100%);
      color: white; padding: 24px 32px; border-radius: 12px; margin-bottom: 20px;
    }}
    .header h1 {{ margin: 0 0 8px; font-size: 20px; }}
    .header .meta {{ font-size: 12px; opacity: .75; line-height: 2; }}
    .header .meta span {{ margin-right: 20px; }}
    .badge-live {{
      display: inline-block; background: #e74c3c; color: #fff;
      padding: 3px 10px; border-radius: 20px; font-size: 11px;
      margin-left: 10px; vertical-align: middle; letter-spacing: 1px;
    }}
    .section {{
      background: white; border-radius: 10px; padding: 20px 24px;
      margin-bottom: 16px; box-shadow: 0 2px 6px rgba(0,0,0,.06);
    }}
    h2 {{
      margin: 0 0 14px; font-size: 14px; color: #1a1f3c;
      border-left: 4px solid #D94040; padding-left: 10px;
    }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th {{
      background: #f5f7fa; padding: 8px 12px; text-align: left;
      font-weight: 600; color: #444; border-bottom: 2px solid #e8eaf0;
    }}
    td {{ padding: 9px 12px; border-bottom: 1px solid #f0f0f0; vertical-align: middle; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #fafbff; }}
    .dist-bar {{ font-size: 12px; color: #666; padding: 8px 0; }}
    .dist-bar span {{ margin-right: 0; }}
    .note {{ font-size: 11px; color: #999; margin-top: 12px; }}
  </style>
</head>
<body>
<div class="container">
  <div class="header">
    <h1>实盘预测报告 — {predict_date} <span class="badge-live">LIVE</span></h1>
    <div class="meta">
      <span>模型: {model_meta.get('model', '—')}/{model_meta.get('freq', '—')}</span>
      <span>训练截止: {trained_end}</span>
      <span>版本: {run_id_str}</span>
      <span>生成时间: {saved_at}</span>
      <span>选股池: {universe}</span>
      <span>推荐数量: Top {topk_n}</span>
    </div>
  </div>

  <div class="section">
    <h2>今日推荐持仓（T+1 参考执行）</h2>
    <div class="dist-bar">信号分布: {dist_html}</div>
    <br>
    <table>
      <tr>
        <th>股票</th>
        <th>排名</th>
        <th>模型打分</th>
        <th>百分位</th>
        <th>建议权重</th>
        <th>变化</th>
        <th>打分变化</th>
      </tr>
      {rec_rows if rec_rows else '<tr><td colspan="7" style="text-align:center;color:#999">无推荐标的</td></tr>'}
    </table>
    <p class="note">
      注：推荐结果基于收盘后模型推理，次日开盘参考执行（T+1）。
      建议权重为等权分配参考值，实际仓位请结合个人风险偏好调整。
      股票存在涨跌停、停牌等情况时可能无法按预期成交。
    </p>
  </div>

  {filter_section}

  {drop_section}

  {history_html}

  <div class="section">
    <h2>使用说明</h2>
    <table>
      <tr><th>字段</th><th>说明</th></tr>
      <tr><td>排名</td><td>在 {universe} 内按模型打分降序排列</td></tr>
      <tr><td>模型打分</td><td>LGBM 输出的预测收益率（正值=模型认为将上涨）</td></tr>
      <tr><td>百分位</td><td>在 {universe} 全截面中的打分百分位（越高越好）</td></tr>
      <tr><td>建议权重</td><td>等权分配（1/topK），实际可按风控需求调整</td></tr>
      <tr><td>变化</td><td>新入=昨日不在推荐中；持续=昨日已推荐；退出=打分明显下滑</td></tr>
    </table>
  </div>
</div>
</body>
</html>"""

    return html


def _build_history_table(predict_date: str, live_cfg: Dict[str, Any]) -> str:
    """读取历史预测记录，构建近 N 日预测摘要 HTML 表格。"""
    history_days = live_cfg.get("tracking", {}).get("history_days", 20)
    if not PREDICTIONS_BASE.exists():
        return ""

    date_dirs = sorted(
        [d for d in PREDICTIONS_BASE.iterdir()
         if d.is_dir() and d.name <= predict_date and (d / "selection.csv").exists()],
        reverse=True,
    )[:history_days]

    if len(date_dirs) < 2:
        return ""

    rows = ""
    for d in date_dirs:
        try:
            df = pd.read_csv(d / "selection.csv", dtype={"instrument": str})
            top = df[df["change"] != "drop_out"]
            codes = ", ".join(top["instrument"].head(5).tolist())
            n_new = int((top["change"] == "new_in").sum())
            n_hold = int((top["change"] == "hold").sum())
            avg_score = top["score"].mean() if not top.empty else float("nan")
            score_str = f"{avg_score:.4f}" if not math.isnan(avg_score) else "—"
            rows += f"""
        <tr>
          <td>{d.name}</td>
          <td>{codes}</td>
          <td style="text-align:center">{score_str}</td>
          <td style="text-align:center">{n_new}</td>
          <td style="text-align:center">{n_hold}</td>
        </tr>"""
        except Exception:
            continue

    if not rows:
        return ""

    return f"""
    <div class="section">
      <h2>近期预测历史（最近 {len(date_dirs)} 日）</h2>
      <table>
        <tr>
          <th>预测日期</th>
          <th>推荐标的</th>
          <th>平均打分</th>
          <th>新入数</th>
          <th>持续数</th>
        </tr>
        {rows}
      </table>
    </div>"""


# ─────────────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="实盘预测脚本 — 每日收盘后运行，输出 topK 选股推荐",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 日常使用（每日收盘后，加载已有模型）
  python python/predict_live.py --model lgbm --freq daily --mode predict-only

  # 指定预测日期（默认自动读取 Qlib 数据中最新可用日期）
  python python/predict_live.py --model lgbm --freq daily --mode predict-only --date 2026-05-05

  # 每周重训一次（周一或月初）
  python python/predict_live.py --model lgbm --freq daily --mode retrain-and-predict

  # 不更新数据，仅重跑已有日期的预测
  python python/predict_live.py --model lgbm --freq daily --mode predict-only --no-data-update
        """,
    )
    parser.add_argument("--model", default="lgbm", help="模型名（lgbm/mlp/lstm，默认 lgbm）")
    parser.add_argument("--freq", default="daily", help="频率（daily，默认 daily）")
    parser.add_argument(
        "--mode",
        choices=["predict-only", "retrain-and-predict"],
        default="predict-only",
        help=(
            "predict-only: 加载已保存模型快速推理（秒级）；"
            "retrain-and-predict: 重训模型后推理（~21秒）"
        ),
    )
    parser.add_argument(
        "--date",
        default=None,
        help="预测日期 YYYY-MM-DD（默认: 自动读取 Qlib 数据中最新可用日期）",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="推荐股票数（默认读 configs/live/daily_live.yaml 的 live.topk）",
    )
    args = parser.parse_args()

    wall_start = time.perf_counter()
    model_name = args.model.lower()
    freq_name = args.freq.lower()

    print(f"\n{'='*60}")
    print(f"  实盘预测 — {model_name}/{freq_name} | mode={args.mode}")
    print(f"{'='*60}")

    # ── 加载配置 ──────────────────────────────────────────────────
    live_cfg_raw = _load_yaml(LIVE_CFG_PATH)
    live_params = live_cfg_raw.get("live", {})
    topk = args.topk or live_params.get("topk", 5)
    score_quantile = live_params.get("score_quantile", 0.0)
    min_score_threshold = live_params.get("min_score_threshold")
    drop_signal_quantile = live_params.get("drop_signal_quantile", 0.3)
    tradable_universe = live_cfg_raw.get("experiment", {}).get("tradable_universe", "csi500")
    provider_uri = live_cfg_raw.get("qlib_init", {}).get("provider_uri", "D:/qlib_data/qlib_data")

    # ── qlib 初始化 ────────────────────────────────────────────────
    try:
        import qlib
        qlib.init(
            provider_uri=provider_uri,
            region=live_cfg_raw.get("qlib_init", {}).get("region", "cn"),
        )
    except ImportError as exc:
        raise ImportError(
            "无法导入 qlib，请在 qlib_zhengshi 环境中运行。"
        ) from exc

    # ── 确定预测日期 ───────────────────────────────────────────────
    if args.date:
        predict_date = args.date
    else:
        predict_date = _last_trading_day(provider_uri)
        print(f"\n  预测日期: {predict_date}（自动读取 Qlib 最新可用日期）")
    print(f"  预测日期: {predict_date}")

    # ── 加载配置（注入动态 date） ──────────────────────────────────
    cfg = load_live_config(model_name, freq_name, predict_date)
    cfg = inject_feature_config(cfg)

    run_id = f"{model_name}_{freq_name}_live_{predict_date}"

    # ── 模型获取（加载 or 重训） ───────────────────────────────────
    print(f"\n  模式: {args.mode}")
    if args.mode == "predict-only":
        model, model_meta = load_latest_model(model_name, freq_name)
    else:
        # retrain-and-predict：先构建 dataset（full train），再训练
        from qlib.utils import init_instance_by_config
        print("  构建训练数据集...")
        dataset = init_instance_by_config(cfg["dataset"])
        model, model_meta = train_and_save_model(
            cfg, dataset, model_name, freq_name, run_id
        )

    # ── 推理 ─────────────────────────────────────────────────────
    print(f"\n  开始推理（date={predict_date}）...")

    if args.mode == "predict-only":
        # predict-only：重新构建 dataset（仅 test 段）
        pred_score = generate_pred_score(model, cfg, predict_date)
    else:
        # retrain-and-predict：dataset 已构建（复用）
        from qlib.utils import init_instance_by_config
        pred = model.predict(dataset, segment="test")
        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0]
        pred_score = pred.rename("score")

    # ── universe 过滤 ──────────────────────────────────────────────
    pred_score = filter_to_universe(
        pred_score, tradable_universe, predict_date, run_id
    )

    # ── 构建过滤器 + 选股 ─────────────────────────────────────────
    filter_cfg = live_cfg_raw.get("filter", {})
    exclude_st = filter_cfg.get("exclude_st", True)
    exclude_limit_up = filter_cfg.get("exclude_limit_up", True)
    consecutive_limit_days = int(filter_cfg.get("consecutive_limit_days", 3))

    stock_filter = _build_stock_filter(live_cfg_raw)
    filter_active = (
        (exclude_st and stock_filter.has_st_data)
        or (exclude_limit_up and consecutive_limit_days > 0)
    )

    # 初始选股数量适当扩大（为过滤留出余量），实际以 topk 为准
    prefetch_k = topk + max(consecutive_limit_days, 3) * 2 if filter_active else topk
    print(f"\n  TopK 选股（topk={topk}，score_quantile={score_quantile}，预选 {prefetch_k} 支）...")
    selection = select_topk(
        pred_score,
        predict_date=predict_date,
        topk=prefetch_k,
        score_quantile=score_quantile,
        min_score_threshold=min_score_threshold,
    )

    # ── 过滤 ST / 连续涨停，补充替代股票 ─────────────────────────
    print(
        f"\n  [过滤] ST={exclude_st}（has_data={stock_filter.has_st_data}）"
        f"  涨停={exclude_limit_up}（阈值={consecutive_limit_days}天）"
    )
    selection, exclusion_log = filter_and_refill(
        selection=selection,
        pred_score=pred_score,
        predict_date=predict_date,
        topk=topk,
        score_quantile=score_quantile,
        min_score_threshold=min_score_threshold,
        stock_filter=stock_filter if filter_active else None,
        exclude_st=exclude_st,
        exclude_limit_up=exclude_limit_up,
    )
    if exclusion_log:
        print(f"  [过滤] 共排除 {len(exclusion_log)} 支，已补充替代股票")

    # ── 与前日比较，计算 change 字段 ─────────────────────────────
    prev_selection = load_prev_selection(predict_date)
    selection = add_change_field(
        selection, prev_selection, pred_score,
        drop_signal_quantile=drop_signal_quantile,
    )

    selection = add_stock_names(selection)

    # ── 持久化 ─────────────────────────────────────────────────────
    output_dir = persist_prediction(
        predict_date, selection, pred_score, model_meta, live_cfg_raw,
        exclusion_log=exclusion_log,
    )

    # ── 打印结果摘要 ──────────────────────────────────────────────
    total_sec = round(time.perf_counter() - wall_start, 2)
    top_df = selection[selection["change"] != "drop_out"]
    drop_df = selection[selection["change"] == "drop_out"]

    print(f"\n{'='*60}")
    print(f"  预测日期: {predict_date}   耗时: {total_sec}s")
    print(f"  输出目录: {output_dir}")
    print(f"  {'─'*50}")
    print(f"  {'排名':<6} {'股票代码':<14} {'打分':>10} {'百分位':>8} {'权重':>8} {'变化'}")
    print(f"  {'─'*50}")
    for _, row in top_df.iterrows():
        change_str = {"new_in": "★新入", "hold": "  持续", "drop_out": "↓退出"}.get(
            row["change"], row["change"]
        )
        pct_str = f"{row.get('percentile', 0):.1f}%"
        w_str = f"{float(row['suggested_weight']):.1%}"
        print(
            f"  {int(row['rank']):<6} {row['instrument']:<14} "
            f"{float(row['score']):>10.4f} {pct_str:>8} {w_str:>8} {change_str}"
        )
    if not drop_df.empty:
        print(f"  {'─'*50}")
        print(f"  建议减仓/退出（打分下滑明显）:")
        for _, row in drop_df.iterrows():
            score_str = f"{float(row['score']):.4f}" if not math.isnan(float(row["score"])) else "—"
            print(f"    ↓ {row['instrument']}  今日打分={score_str}")
    print(f"{'='*60}")
    print(f"  查看完整报告: {output_dir / 'report.html'}\n")


if __name__ == "__main__":
    main()
