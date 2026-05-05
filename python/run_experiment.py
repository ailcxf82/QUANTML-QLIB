"""
通用实验入口：支持任意模型（MLP/LSTM/GRU/Transformer/LightGBM）× 任意频率（日频/分钟频）。

层级位置: Data -> Feature -> Model -> Signal -> Portfolio -> Backtest -> Evaluation
输入:
  方式1（组合式）: --model mlp --freq daily
  方式2（完整配置）: --config configs/mlp_daily_backtest.yaml
输出: artifacts/<run_id>/
    metrics.json             核心指标（含 IC/ICIR）+ 阶段耗时
    backtest_report.parquet  逐日组合收益序列（含 return/bench/turnover/excess/cum_*）
    signals.parquet          测试集预测分数
    indicator.parquet        日级成交聚合指标（如有）
    trades.parquet           每笔订单明细（datetime/stock_id/direction/price/value/cost/...）
    realized_pnl.parquet     FIFO 配对后的已实现盈亏明细
    charts/                  四张可视化图表
    report.html              HTML 完整回测报告

配置合并顺序: base.yaml → models/<model>.yaml → freq/<freq>.yaml
后者覆盖前者（深度递归合并，list 整体替换）。

回测分层架构（python/backtest/ 模块）:
  SignalLayer     → 验证信号 + IC/ICIR 计算
  StrategyLayer   → 信号注入策略配置
  ExecutionLayer  → 日频/分钟频执行器构建
  MarketLayer     → A 股交易成本封装
  PortfolioLayer  → 解析 Qlib 回测结果
  AnalysisLayer   → 指标计算 + 图表 + HTML 报告

特征注入流程:
  model_meta.feature_groups → features.build_feature_config() → 注入 data_loader.config.feature
  → 自动同步 model.kwargs.input_dim / d_feat，消除手动硬编码。
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 将 python/ 目录加入 sys.path，确保 from features import ... 可正常解析
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import yaml

from features import build_feature_config, feature_count, print_registry
from features.combined_json_factors import merge_features_from_combined_json

try:
    import qlib
    from qlib.data.dataset import DatasetH
    from qlib.log import get_module_logger
    from qlib.model.base import Model
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
except ImportError as exc:
    raise ImportError(
        "无法导入 Microsoft Qlib (pyqlib)。"
        "请在 qlib_zhengshi 虚拟环境中执行：pip uninstall -y qlib && pip install pyqlib"
    ) from exc

from backtest import BacktestEngine
from backtest.rolling import (
    FoldSpec,
    build_walk_forward_folds,
    concat_oos_pred_scores,
    folds_to_summary,
    infer_purge_days,
)


LOGGER = get_module_logger("run_experiment")
VERSION = "v2"
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


# ─────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────

@dataclass
class RunContext:
    run_id: str
    output_dir: Path
    timing: Dict[str, float] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────
# 配置加载与合并
# ─────────────────────────────────────────────────────────────────

# 模型名别名映射（统一到 configs/models/ 下的文件名）
_MODEL_ALIASES: Dict[str, str] = {
    # lgbm
    "lgbm": "lgbm",
    "lightgbm": "lgbm",
    "gbdt": "lgbm",
    # mlp
    "mlp": "mlp",
    "dnn": "mlp",
    # lstm
    "lstm": "lstm",
    # gru
    "gru": "gru",
    # transformer
    "transformer": "transformer",
}

# 频率名别名映射（统一到 configs/freq/ 下的文件名）
_FREQ_ALIASES: Dict[str, str] = {
    "daily": "daily",
    "day": "daily",
    "1day": "daily",
    "d": "daily",
    "minute": "minute",
    "min": "minute",
    "1min": "minute",
    "m": "minute",
}


def _normalize_model(name: str) -> str:
    """将用户输入的模型名规范化为配置文件名（大小写不敏感 + 常见别名）。"""
    key = name.strip().lower()
    if key not in _MODEL_ALIASES:
        valid = sorted(set(_MODEL_ALIASES.values()))
        raise ValueError(
            f"未知模型名 '{name}'。可选值：{valid}（别名如 LightGBM/lgbm/gbdt 均可）"
        )
    return _MODEL_ALIASES[key]


def _normalize_freq(name: str) -> str:
    """将用户输入的频率名规范化为配置文件名（大小写不敏感 + 常见别名）。"""
    key = name.strip().lower()
    if key not in _FREQ_ALIASES:
        valid = sorted(set(_FREQ_ALIASES.values()))
        raise ValueError(
            f"未知频率名 '{name}'。可选值：{valid}（别名如 daily/day/1day 均可）"
        )
    return _FREQ_ALIASES[key]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """递归合并：override 覆盖 base；list 类型整体替换（不拼接）。"""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = deep_merge(result[key], val)
        else:
            result[key] = copy.deepcopy(val)
    return result


def load_and_merge_configs(
    model_name: Optional[str],
    freq_name: Optional[str],
    full_config: Optional[str],
    strategy_cfg_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    根据参数组合或完整配置路径，返回最终合并的配置字典。

    合并顺序: base.yaml → models/<model>.yaml → freq/<freq>.yaml → strategy/<file>.yaml(可选)

    Args:
        strategy_cfg_path: 可选的策略配置文件路径（相对仓库根），用于
            覆盖 base.yaml 中的 strategy 段。例如 configs/strategy/midfreq_rp.yaml。
    """
    configs_dir = WORKSPACE_ROOT / "configs"

    def _maybe_apply_strategy_override(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """
        如指定 --strategy-cfg，则将其内容应用到主 cfg。

        关键：strategy 段使用"整体替换"而非深度合并，避免不同策略类的 kwargs
        彼此污染（例如把 TopkDropoutStrategy 的 n_drop 误注入到
        QuantMLWeightStrategy 中）。其它顶层段（如 backtest / executor）
        仍用 deep_merge，便于策略文件局部覆盖少量字段。
        """
        if not strategy_cfg_path:
            return cfg
        strat_cfg = _load_yaml(WORKSPACE_ROOT / strategy_cfg_path)
        if not strat_cfg:
            return cfg
        merged = copy.deepcopy(cfg)
        for key, val in strat_cfg.items():
            if key == "strategy":
                merged["strategy"] = copy.deepcopy(val)
            elif key in merged and isinstance(merged[key], dict) and isinstance(val, dict):
                merged[key] = deep_merge(merged[key], val)
            else:
                merged[key] = copy.deepcopy(val)
        return merged

    if full_config:
        # 方式2：直接使用完整配置文件（兼容 mlp_daily_backtest.yaml 等旧配置）
        cfg = _load_yaml(WORKSPACE_ROOT / full_config)
        # 注入必要的 model_meta 默认值（旧文件可能没有）
        cfg.setdefault("model_meta", {})
        cfg["model_meta"].setdefault("dataset_type", "flat")
        cfg["model_meta"].setdefault("freq_key", "1day")
        return _maybe_apply_strategy_override(cfg)

    # 方式1：组合式合并
    if not model_name or not freq_name:
        raise ValueError("必须同时指定 --model 和 --freq，或使用 --config 指定完整配置文件。")

    # 规范化名称（大小写不敏感 + 别名），在此提前报错而非等到文件找不到
    model_name = _normalize_model(model_name)
    freq_name = _normalize_freq(freq_name)

    freq_path = configs_dir / "freq" / f"{freq_name}.yaml"
    freq_cfg = _load_yaml(freq_path)

    # 检查分钟频数据是否就绪
    if freq_cfg.get("freq_meta", {}).get("data_ready") is False:
        raise RuntimeError(
            f"频率配置 '{freq_name}' 的 data_ready=false，"
            "请先确认分钟频数据已就绪并更新 configs/freq/minute.yaml。"
        )

    base_cfg = _load_yaml(configs_dir / "base.yaml")
    model_cfg = _load_yaml(configs_dir / "models" / f"{model_name}.yaml")

    # 合并: base → model → freq
    merged = deep_merge(base_cfg, model_cfg)
    merged = deep_merge(merged, freq_cfg)

    # 将 dataset_segments 注入 dataset.kwargs.segments
    if "dataset_segments" in merged:
        merged.setdefault("dataset", {}).setdefault("kwargs", {})["segments"] = merged.pop(
            "dataset_segments"
        )

    # 注入 run_id（含模型+频率标识）
    merged.setdefault("experiment", {})["run_id"] = f"{model_name}_{freq_name}"

    # 末层：可选的策略配置覆盖（用于 A/B 实验）
    return _maybe_apply_strategy_override(merged)


# ─────────────────────────────────────────────────────────────────
# 数据集构建（自动选择 DatasetH / TSDatasetH）
# ─────────────────────────────────────────────────────────────────

def patch_dataset_class(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据 model_meta.dataset_type 自动选择数据集类型：
      flat → DatasetH（默认，单时间步，MLP/LightGBM）
      ts   → TSDatasetH（时序窗口，LSTM/GRU/Transformer）
    """
    dataset_type = cfg.get("model_meta", {}).get("dataset_type", "flat")
    if dataset_type == "ts":
        step_len = cfg.get("model_meta", {}).get("step_len", 20)
        cfg = copy.deepcopy(cfg)
        ds_kwargs = cfg.setdefault("dataset", {}).setdefault("kwargs", {})
        cfg["dataset"]["class"] = "TSDatasetH"
        cfg["dataset"]["module_path"] = "qlib.data.dataset"
        # TSDatasetH: step_len 在 kwargs 顶层，handler/segments 在 kwargs 内
        ds_kwargs["step_len"] = step_len
    return cfg


def inject_feature_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    从因子库动态生成特征配置并注入 dataset handler，同步更新模型的 input_dim / d_feat。

    读取 model_meta.feature_groups（list[str] | null）决定使用哪些因子分组：
      - null / 未设置 → 使用全部启用因子（含 RDAgent 生成因子）
      - list          → 仅使用指定分组的因子

    若设置 model_meta.combined_factors_json（相对仓库根的路径），仅在此处追加 JSON 中列出的
    额外因子（见 configs/combined_factors_df.json），其它模型配置不设置该键则不受影响。

    注入路径:
      cfg["dataset"]["kwargs"]["handler"]["kwargs"]["data_loader"]["kwargs"]["config"]["feature"]
    同步字段:
      cfg["model"]["kwargs"]["input_dim"]  （MLP 等平铺模型）
      cfg["model"]["kwargs"]["pt_model_kwargs"]["input_dim"]  （DNNModelPytorch）
      cfg["model"]["kwargs"]["d_feat"]     （LSTM/GRU/Transformer 时序模型）
    """
    cfg = copy.deepcopy(cfg)
    feature_groups: Optional[List[str]] = cfg.get("model_meta", {}).get("feature_groups")

    exprs, names = build_feature_config(
        groups=feature_groups,
        include_rdagent=True,
    )
    combined_rel = cfg.get("model_meta", {}).get("combined_factors_json")
    if combined_rel:
        merge_features_from_combined_json(
            exprs,
            names,
            WORKSPACE_ROOT / str(combined_rel),
        )
    n_feat = len(names)

    LOGGER.info(
        "feature_inject: groups=%s, combined_json=%s, n_feat=%d, names=%s",
        feature_groups or "ALL",
        combined_rel or None,
        n_feat,
        names,
        extra={"run_id": "init", "instrument": "ALL", "datetime": "", "signal": 0.0, "version": VERSION},
    )

    # 注入 data_loader.config.feature
    dl_cfg = (
        cfg["dataset"]["kwargs"]["handler"]["kwargs"]["data_loader"]["kwargs"]["config"]
    )
    dl_cfg["feature"] = [exprs, names]

    # 同步模型维度参数（消除 input_dim / d_feat 硬编码）
    model_kwargs: Dict[str, Any] = cfg.get("model", {}).get("kwargs", {})
    if "input_dim" in model_kwargs:
        model_kwargs["input_dim"] = n_feat
    if "d_feat" in model_kwargs:
        model_kwargs["d_feat"] = n_feat
    # DNNModelPytorch 的 input_dim 嵌套在 pt_model_kwargs 内
    pt_kwargs = model_kwargs.get("pt_model_kwargs", {})
    if "input_dim" in pt_kwargs:
        pt_kwargs["input_dim"] = n_feat

    return cfg


def build_dataset(cfg: Dict[str, Any]) -> DatasetH:
    return init_instance_by_config(cfg["dataset"])


# ─────────────────────────────────────────────────────────────────
# 模型训练
# ─────────────────────────────────────────────────────────────────

def train_model(cfg: Dict[str, Any], dataset: DatasetH) -> Model:
    model: Model = init_instance_by_config(cfg["model"])
    model.fit(dataset)
    return model


# ─────────────────────────────────────────────────────────────────
# N-seed ensemble（仅在非 rolling 模式生效）
# ─────────────────────────────────────────────────────────────────

def _override_model_seeds(model_cfg: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """生成把所有随机种子统一为 `seed` 的模型配置副本（用于 ensemble）。

    覆盖字段（LGBM 完整随机源）:
      - seed: tree split / sampling 主种子
      - feature_fraction_seed: 列子采样
      - bagging_seed: 行子采样
      - data_random_seed: 数据 reorder
    其它非随机字段保持不变。
    """
    new_cfg = copy.deepcopy(model_cfg)
    kwargs: Dict[str, Any] = new_cfg.setdefault("kwargs", {})
    for key in ("seed", "feature_fraction_seed", "bagging_seed", "data_random_seed"):
        kwargs[key] = int(seed)
    return new_cfg


def _aggregate_topk_vote(
    pred_seeds: List[pd.Series],
    topk: int,
) -> pd.Series:
    """每个 seed 独立产生当日 TopK 候选，按"多数票"聚合后输出最终 score。

    设计要点:
      - 每个 seed 在每日截面取 top-K（K 取自策略 topk），生成 0/1 矩阵；
      - score_vote: seed 命中数（0~N），保留头部锐度（不被 rank-mean 稀释）；
      - tiebreaker: epsilon * 平均 pct-rank，保证同票数股票仍有稳定排序，
        避免下游 TopkDropoutStrategy 在平票情况下的不可复现行为。
      - 返回的 score 满足 score_a > score_b ⇔ a 应被 TopkDropoutStrategy 优先选中。

    参数:
      pred_seeds: 每个 seed 的预测序列列表（index=(datetime, instrument)）
      topk: 与 strategy.kwargs.topk 一致的头部宽度

    返回:
      ensemble_score: 与单 seed predict 同结构的 pd.Series
    """
    if not pred_seeds:
        raise ValueError("pred_seeds 为空，无法聚合")
    wide = pd.concat(pred_seeds, axis=1)
    rank_per_seed = wide.groupby(level="datetime").rank(pct=True)
    in_topk = wide.groupby(level="datetime").rank(method="first", ascending=False) <= topk
    vote_count = in_topk.sum(axis=1).astype(float)
    mean_rank = rank_per_seed.mean(axis=1, skipna=True)
    epsilon = 1.0 / (len(pred_seeds) + 1)
    ensemble_score = vote_count + epsilon * mean_rank
    ensemble_score.name = "score"
    return ensemble_score


def train_predict_ensemble(
    cfg: Dict[str, Any],
    dataset: DatasetH,
    seeds: List[int],
    run_ctx: "RunContext",
    aggregator: str = "rank_mean",
    test_segment: str = "test",
) -> Tuple[pd.Series, List[float]]:
    """N-seed LGBM 集成：每个 seed 独立训练 & 预测，按指定方式聚合。

    aggregator 取值:
      - "rank_mean": 按日截面 pct-rank 后对所有 seed 取均值（金融 ensemble 标准做法，
        消除不同 seed 间的 magnitude shift）；缺点是头部位置等权平均，对 TopK 离散
        选股有"头部稀释"副作用。
      - "topk_vote": 每个 seed 在每日截面取自身 top-K（K=strategy.topk），按股票
        汇总命中数；保留头部锐度，仅在多数 seed 共识时才入选最终持仓。

    返回:
      ensemble_score: 与单 seed predict 同结构的 pd.Series（index=(datetime, instrument)）
      train_secs:     每个 seed 的训练耗时列表（用于诊断）
    """
    pred_seeds: List[pd.Series] = []
    train_secs: List[float] = []
    test_ic_per_seed: List[float] = []

    label_seed_eval: Optional[pd.Series] = None
    try:
        lbl_df = dataset.prepare(test_segment, col_set="label")
        if isinstance(lbl_df, pd.DataFrame) and not lbl_df.empty:
            label_seed_eval = lbl_df.iloc[:, 0]
    except Exception:  # noqa: BLE001
        label_seed_eval = None

    for i, seed in enumerate(seeds):
        seed_cfg_model = _override_model_seeds(cfg["model"], seed)
        t0 = time.perf_counter()
        model: Model = init_instance_by_config(seed_cfg_model)
        model.fit(dataset)
        elapsed = round(time.perf_counter() - t0, 2)
        train_secs.append(elapsed)

        pred = model.predict(dataset, segment=test_segment)
        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0]
        pred_seeds.append(pred.rename(f"seed_{seed}"))

        if label_seed_eval is not None:
            try:
                joined = pd.concat([pred.rename("p"), label_seed_eval.rename("y")], axis=1).dropna()
                if not joined.empty:
                    daily = joined.groupby(level="datetime").apply(
                        lambda s: float(np.corrcoef(s["p"], s["y"])[0, 1])
                        if s["p"].std() > 1e-12 and s["y"].std() > 1e-12 else np.nan
                    )
                    test_ic_per_seed.append(float(daily.mean()))
                else:
                    test_ic_per_seed.append(float("nan"))
            except Exception:  # noqa: BLE001
                test_ic_per_seed.append(float("nan"))
        else:
            test_ic_per_seed.append(float("nan"))

        _log(
            run_ctx.run_id,
            f"ensemble seed[{i+1}/{len(seeds)}] seed={seed} train_sec={elapsed}s "
            f"n_pred={len(pred)} test_ic={test_ic_per_seed[-1]:+.4f}",
        )

    if aggregator == "rank_mean":
        wide = pd.concat(pred_seeds, axis=1)
        rank_per_seed = wide.groupby(level="datetime").rank(pct=True)
        ensemble_score = rank_per_seed.mean(axis=1, skipna=True)
        ensemble_score.name = "score"
    elif aggregator == "topk_vote":
        topk = int(cfg.get("strategy", {}).get("kwargs", {}).get("topk", 8))
        ensemble_score = _aggregate_topk_vote(pred_seeds, topk=topk)
    else:
        raise ValueError(f"未知 aggregator: {aggregator}（支持 rank_mean / topk_vote）")

    run_ctx.timing["train_sec"] = round(sum(train_secs), 2)
    run_ctx.timing["ensemble_seeds"] = len(seeds)
    run_ctx.timing["ensemble_aggregator"] = aggregator
    run_ctx.timing["ensemble_per_seed_ic"] = [round(x, 4) for x in test_ic_per_seed]
    run_ctx.timing["ensemble_per_seed_train_sec"] = train_secs

    _log(
        run_ctx.run_id,
        f"ensemble_done n_seeds={len(seeds)} aggregator={aggregator} "
        f"per_seed_ic={[round(x, 4) for x in test_ic_per_seed]} "
        f"avg_test_ic={float(np.nanmean(test_ic_per_seed)):+.4f}",
    )
    return ensemble_score, train_secs


# ─────────────────────────────────────────────────────────────────
# 回测（委托 BacktestEngine 分层处理）
# ─────────────────────────────────────────────────────────────────

def run_backtest(
    cfg: Dict[str, Any],
    pred_score: pd.Series,
    output_dir: Optional[Path] = None,
    label: Optional[pd.Series] = None,
    rolling_summary: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    通过 BacktestEngine 执行分层回测，返回:
        (report_df, indicator_df, trades_df, realized_pnl_df, metrics_dict)

    BacktestEngine 内部按层执行：
      SignalLayer → StrategyLayer → ExecutionLayer → MarketLayer
      → qlib.backtest → PortfolioLayer → TradeLayer → AnalysisLayer

    rolling_summary 若非 None，会传给 HTML 报告生成器渲染 fold 间稳定性一节。
    """
    engine = BacktestEngine()
    result = engine.run(pred_score, cfg, output_dir=output_dir, label=label,
                        rolling_summary=rolling_summary)
    metrics_dict = engine.metrics_to_dict(result)
    return (
        result.report_df,
        result.indicator_df,
        result.trades_df,
        result.realized_pnl_df,
        metrics_dict,
    )


# ─────────────────────────────────────────────────────────────────
# 数据校验
# ─────────────────────────────────────────────────────────────────

def validate_provider_uri(provider_uri: str) -> None:
    data_path = Path(provider_uri)
    if not data_path.exists():
        raise FileNotFoundError(f"provider_uri 路径不存在: {provider_uri}")
    for required in ("calendars", "instruments"):
        if not (data_path / required).exists():
            raise FileNotFoundError(f"数据目录缺少子目录: {required}")


def validate_pred_series(pred: pd.Series, run_id: str) -> None:
    if pred.empty:
        raise ValueError("预测信号序列为空，请检查模型与数据集配置。")
    if not isinstance(pred.index, pd.MultiIndex):
        raise TypeError("预测信号索引必须是 MultiIndex(datetime, instrument)。")
    nan_ratio = float(pred.isna().mean())
    if nan_ratio > 0.5:
        LOGGER.warning(
            "pred_quality_warning: NaN 占比 %.1f%%，请检查特征与模型",
            nan_ratio * 100,
            extra={"run_id": run_id, "instrument": "ALL", "datetime": "", "signal": 0.0, "version": VERSION},
        )


def filter_pred_to_universe(
    pred: pd.Series,
    universe: Optional[str],
    start_time: str,
    end_time: str,
    run_id: str,
) -> pd.Series:
    """将 pred_score 限定到指定可投资 universe（如 csi500）。

    若 universe 为空或为 "all"，原样返回；否则只保留 universe 中的股票，
    用于"训练全市场、回测仅限可投资股票池"的解耦设计。

    Args:
        pred:        模型预测打分，MultiIndex(datetime, instrument)
        universe:    qlib 可识别的 universe 名称（如 csi500、csi300）
        start_time:  过滤起始日期（建议传测试集起始）
        end_time:    过滤结束日期
        run_id:      日志标识

    Returns:
        过滤后的 pred 序列
    """
    if not universe or universe.lower() == "all":
        return pred

    from qlib.data import D

    inst_cfg = D.instruments(universe)
    tradable = D.list_instruments(
        inst_cfg, start_time=start_time, end_time=end_time, as_list=True
    )
    if not tradable:
        LOGGER.warning(
            "tradable_universe '%s' 在 [%s, %s] 区间为空，跳过过滤",
            universe, start_time, end_time,
            extra={"run_id": run_id, "instrument": "ALL", "datetime": "", "signal": 0.0, "version": VERSION},
        )
        return pred

    tradable_set = set(tradable)
    inst_level = pred.index.get_level_values(1)
    mask = inst_level.isin(tradable_set)
    filtered = pred[mask]

    LOGGER.info(
        "tradable_universe filter: %s, 保留 %d / %d 条记录（%.1f%%）",
        universe,
        int(mask.sum()),
        len(pred),
        100.0 * float(mask.mean()),
        extra={"run_id": run_id, "instrument": "ALL", "datetime": "", "signal": 0.0, "version": VERSION},
    )
    return filtered


# ─────────────────────────────────────────────────────────────────
# 持久化
# ─────────────────────────────────────────────────────────────────

def persist_artifacts(
    run_ctx: RunContext,
    cfg: Dict[str, Any],
    metrics: Dict[str, float],
    report_df: pd.DataFrame,
    indicator_df: pd.DataFrame,
    pred_score: pd.Series,
    trades_df: Optional[pd.DataFrame] = None,
    realized_pnl_df: Optional[pd.DataFrame] = None,
    rolling_meta: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
) -> None:
    """持久化回测产物：parquet 数据文件 + metrics.json 汇总。

    图表与 HTML 报告已由 BacktestEngine 在 run_backtest 阶段写入同一目录。
    rolling_meta 若非 None，会作为 "rolling" 字段写入 metrics.json（供 audit 读取）。
    model 若非 None，会序列化为 model.pkl 供实盘预测复用，并同步更新
    artifacts/models/latest_<model_name>_<freq>.pkl 软指针（文件拷贝实现）。
    """
    run_ctx.output_dir.mkdir(parents=True, exist_ok=True)
    report_df.to_parquet(run_ctx.output_dir / "backtest_report.parquet")
    pred_score.to_frame("score").to_parquet(run_ctx.output_dir / "signals.parquet")
    if not indicator_df.empty:
        indicator_df.to_parquet(run_ctx.output_dir / "indicator.parquet")
    if trades_df is not None and not trades_df.empty:
        trades_df.to_parquet(run_ctx.output_dir / "trades.parquet")
    if realized_pnl_df is not None and not realized_pnl_df.empty:
        realized_pnl_df.to_parquet(run_ctx.output_dir / "realized_pnl.parquet")

    model_name = cfg.get("model_meta", {}).get("name", "unknown")
    freq_name = cfg.get("freq_meta", {}).get("name", "unknown")

    final_output: Dict[str, Any] = {
        "run_id": run_ctx.run_id,
        "version": VERSION,
        "model": model_name,
        "freq": freq_name,
        "timing_seconds": run_ctx.timing,
        "metrics": metrics,
    }
    if rolling_meta is not None:
        final_output["rolling"] = rolling_meta
    with (run_ctx.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)

    # ── 模型序列化（供实盘预测复用）──────────────────────────────────────
    if model is not None:
        try:
            import joblib
            model_path = run_ctx.output_dir / "model.pkl"
            joblib.dump(model, model_path)

            # 写入 model_meta.json（训练截止日、特征数等，供 predict_live.py 校验）
            train_seg = cfg.get("dataset", {}).get("kwargs", {}).get("segments", {}).get("train", [])
            train_end = train_seg[1] if len(train_seg) >= 2 else "unknown"
            n_features = cfg.get("model", {}).get("kwargs", {}).get("input_dim") or \
                         cfg.get("model", {}).get("kwargs", {}).get("d_feat")
            model_meta = {
                "run_id": run_ctx.run_id,
                "model": model_name,
                "freq": freq_name,
                "train_end": train_end,
                "n_features": n_features,
                "saved_at": pd.Timestamp.now().isoformat(),
                "version": VERSION,
            }
            with (run_ctx.output_dir / "model_meta.json").open("w", encoding="utf-8") as f:
                json.dump(model_meta, f, ensure_ascii=False, indent=2)

            # 更新 artifacts/models/latest_<model>_<freq>.pkl（文件覆盖）
            import shutil
            models_dir = WORKSPACE_ROOT / "artifacts" / "models"
            models_dir.mkdir(parents=True, exist_ok=True)
            latest_path = models_dir / f"latest_{model_name}_{freq_name}.pkl"
            shutil.copy2(model_path, latest_path)
            # 同步写入 latest 的 meta（指向源 run_id 与截止日）
            with (models_dir / f"latest_{model_name}_{freq_name}_meta.json").open("w", encoding="utf-8") as f:
                json.dump(model_meta, f, ensure_ascii=False, indent=2)
            LOGGER.info(
                "model_saved: %s → %s (latest=%s)",
                model_path, model_meta["train_end"], latest_path,
                extra={"run_id": run_ctx.run_id, "instrument": "ALL",
                       "datetime": "", "signal": 0.0, "version": VERSION},
            )
        except Exception as exc:
            LOGGER.warning(
                "model_save_failed: %s（跳过序列化，回测产物不受影响）", exc,
                extra={"run_id": run_ctx.run_id, "instrument": "ALL",
                       "datetime": "", "signal": 0.0, "version": VERSION},
            )


# ─────────────────────────────────────────────────────────────────
# Walk-Forward：单 fold 训练 + 预测
# ─────────────────────────────────────────────────────────────────

def _build_fold_cfg(cfg: Dict[str, Any], fold: FoldSpec) -> Dict[str, Any]:
    """生成注入了该 fold 时间段的配置副本（仅覆盖 segments，handler 保持全段加载）。"""
    fold_cfg = copy.deepcopy(cfg)
    fold_cfg["dataset"]["kwargs"]["segments"] = {
        "train": list(fold.train),
        "valid": list(fold.valid),
        "test":  list(fold.test),
    }
    return fold_cfg


def _build_fold_dataset_with_handler(
    fold: FoldSpec,
    shared_handler: Any,
) -> "DatasetH":
    """用已初始化好的 handler 构建 fold 专属 DatasetH（跨 fold 复用 handler）。

    DatasetH 接受已实例化的 handler 对象（非 dict），不会重新初始化（不重复加载数据）；
    只更新 segments，让 prepare() 时用正确的 train/valid/test 时间切片。
    """
    from qlib.data.dataset import DatasetH
    return DatasetH(
        handler=shared_handler,
        segments={
            "train": list(fold.train),
            "valid": list(fold.valid),
            "test":  list(fold.test),
        },
    )


def _train_fold(
    fold: FoldSpec,
    cfg: Dict[str, Any],
    run_ctx: RunContext,
    seed: int,
    test_segment: str = "test",
    shared_handler: Optional[Any] = None,
) -> Tuple[pd.Series, Dict[str, Any]]:
    """对单个 fold 执行：构建 dataset → 训练 → 预测 OOS → 计算 fold-level IC。

    Args:
        shared_handler: 可选的预初始化 handler 实例（PR2 优化：跨 fold 复用，避免重复加载数据）。
                        若为 None，则每 fold 重新 init dataset（PR1 兜底模式）。

    Returns:
        (pred_score, fold_metrics)  fold_metrics 含 ic_mean/icir 等基础信号指标
    """
    fold_cfg = _build_fold_cfg(cfg, fold)

    _log(run_ctx.run_id, (
        f"[fold {fold.fold_id}] 开始 | "
        f"train=[{fold.train_start},{fold.train_end}] "
        f"valid=[{fold.valid_start},{fold.valid_end}] "
        f"test=[{fold.test_start},{fold.test_end}]"
    ))

    # PR2 优化：复用共享 handler；PR1 兜底：每 fold 重新 init
    t0 = time.perf_counter()
    if shared_handler is not None:
        dataset = _build_fold_dataset_with_handler(fold, shared_handler)
        ds_sec = round(time.perf_counter() - t0, 4)
        _log(run_ctx.run_id, f"[fold {fold.fold_id}] dataset_ready（复用 handler）[{ds_sec}s]")
    else:
        dataset = build_dataset(fold_cfg)
        ds_sec = round(time.perf_counter() - t0, 2)
        _log(run_ctx.run_id, f"[fold {fold.fold_id}] dataset_ready [{ds_sec}s]")

    # 使用 fold 特定 seed 训练（单 seed per fold）
    fold_cfg_seed = copy.deepcopy(fold_cfg)
    kw = fold_cfg_seed.setdefault("model", {}).setdefault("kwargs", {})
    kw.setdefault("seed", seed)
    for k in ("feature_fraction_seed", "bagging_seed", "data_random_seed"):
        kw.setdefault(k, seed)

    t1 = time.perf_counter()
    model: Model = init_instance_by_config(fold_cfg_seed["model"])
    model.fit(dataset)
    train_sec = round(time.perf_counter() - t1, 2)
    _log(run_ctx.run_id, f"[fold {fold.fold_id}] train_done [{train_sec}s] seed={seed}")

    pred: pd.Series = model.predict(dataset, segment=test_segment)
    if isinstance(pred, pd.DataFrame):
        pred = pred.iloc[:, 0]
    pred = pred.rename("score")

    # 计算 fold-level IC（信号端，无需回测）
    fold_metrics: Dict[str, Any] = {
        "fold_id": fold.fold_id,
        "train_start": fold.train_start,
        "train_end": fold.train_end,
        "test_start": fold.test_start,
        "test_end": fold.test_end,
        "dataset_build_sec": ds_sec,
        "train_sec": train_sec,
        "n_pred": int(len(pred)),
    }
    try:
        label_raw = dataset.prepare(test_segment, col_set="label")
        if isinstance(label_raw, pd.DataFrame) and not label_raw.empty:
            lbl = label_raw.iloc[:, 0]
            joined = pd.concat([pred.rename("p"), lbl.rename("y")], axis=1).dropna()
            if not joined.empty:
                daily_ic = joined.groupby(level="datetime").apply(
                    lambda s: float(np.corrcoef(s["p"], s["y"])[0, 1])
                    if s["p"].std() > 1e-12 and s["y"].std() > 1e-12 else np.nan
                )
                ic_mean = float(daily_ic.mean())
                ic_std = float(daily_ic.std())
                icir = ic_mean / ic_std if ic_std > 1e-12 else float("nan")
                fold_metrics["ic_mean"] = round(ic_mean, 6)
                fold_metrics["icir"] = round(icir, 4)
                fold_metrics["n_days"] = int(daily_ic.notna().sum())
    except Exception:
        pass

    _log(
        run_ctx.run_id,
        f"[fold {fold.fold_id}] done | ic_mean={fold_metrics.get('ic_mean', 'n/a')} "
        f"n_pred={fold_metrics['n_pred']}",
    )
    return pred, fold_metrics


def _persist_fold_artifacts(
    fold_dir: Path,
    fold: FoldSpec,
    pred_score: pd.Series,
    fold_metrics: Dict[str, Any],
) -> None:
    """持久化单 fold 产物：pred_score.parquet + metrics.json。"""
    fold_dir.mkdir(parents=True, exist_ok=True)
    pred_score.to_frame("score").to_parquet(fold_dir / "pred_score.parquet")
    with (fold_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────
# Walk-Forward：主流程
# ─────────────────────────────────────────────────────────────────

def run_walk_forward(
    cfg: Dict[str, Any],
    run_ctx: RunContext,
    rolling_params: Dict[str, Any],
) -> None:
    """Walk-Forward 主流程：多 fold 训练 → 拼接 OOS → 一次 BacktestEngine。

    目录结构：
        artifacts/<run_id>/
            fold_1/ pred_score.parquet  metrics.json
            fold_2/ ...
            concatenated/  （完整回测产物）
            rolling_summary.json
    """
    mode = rolling_params.get("mode", "walk_forward")
    if mode != "walk_forward":
        raise ValueError(f"暂不支持 rolling mode={mode!r}，仅支持 walk_forward")

    # 自动推断 purge_days（若配置为 None）
    purge_days = rolling_params.get("purge_days")
    if purge_days is None:
        label_cfg = (
            cfg.get("dataset", {})
            .get("kwargs", {})
            .get("handler", {})
            .get("kwargs", {})
            .get("data_loader", {})
            .get("kwargs", {})
            .get("config", {})
            .get("label", [])
        )
        label_expr = ""
        if label_cfg and isinstance(label_cfg[0], list) and label_cfg[0]:
            label_expr = label_cfg[0][0]
        purge_days = infer_purge_days(label_expr)
        _log(run_ctx.run_id, f"[walk_forward] 自动推断 purge_days={purge_days}（label={label_expr!r}）")

    data_start = cfg.get("experiment", {}).get("start_time", "2020-01-01")
    data_end = cfg.get("experiment", {}).get("end_time", "2026-04-28")

    folds = build_walk_forward_folds(
        data_start=data_start,
        data_end=data_end,
        train_min=str(rolling_params.get("train_min", "4y")),
        valid_size=str(rolling_params.get("valid_size", "6m")),
        test_size=str(rolling_params.get("test_size", "6m")),
        step=str(rolling_params.get("step", "6m")),
        purge_days=int(purge_days),
        embargo_days=int(rolling_params.get("embargo_days", 5)),
    )

    if not folds:
        raise RuntimeError(
            f"Walk-Forward 切分产生 0 个 fold（data_start={data_start}, "
            f"data_end={data_end}, train_min={rolling_params.get('train_min')}），"
            "请检查数据范围与 train_min 配置。"
        )

    _log(run_ctx.run_id, f"[walk_forward] 共 {len(folds)} 个 fold，OOS 范围: "
         f"{folds[0].test_start} ~ {folds[-1].test_end}")

    base_seed = int(cfg.get("model", {}).get("kwargs", {}).get("seed", 42))
    tradable_universe = cfg.get("experiment", {}).get("tradable_universe")

    # ── PR2：预初始化共享 handler（加载一次，跨 fold 复用） ────────────
    shared_handler: Optional[Any] = None
    t_handler = time.perf_counter()
    try:
        full_dataset = build_dataset(cfg)
        shared_handler = full_dataset.handler  # type: ignore[attr-defined]
        handler_sec = round(time.perf_counter() - t_handler, 2)
        _log(run_ctx.run_id,
             f"[walk_forward] handler 预初始化完成 [{handler_sec}s]，后续 fold 复用")
        run_ctx.timing["dataset_build_sec"] = handler_sec
    except Exception as exc:
        handler_sec = round(time.perf_counter() - t_handler, 2)
        _log(run_ctx.run_id,
             f"[walk_forward] handler 预初始化失败（{exc}），回退到每 fold 重 init [{handler_sec}s]")
        shared_handler = None

    fold_preds: List[pd.Series] = []
    fold_metrics_list: List[Dict[str, Any]] = []

    # ── 逐 fold 训练 + 预测 ──────────────────────────────────────
    for fold in folds:
        fold_dir = run_ctx.output_dir / f"fold_{fold.fold_id}"
        pred, fold_metrics = _train_fold(
            fold, cfg, run_ctx, seed=base_seed, shared_handler=shared_handler
        )

        # universe 过滤（与单切分模式保持一致）
        if tradable_universe and tradable_universe.lower() != "all":
            pred = filter_pred_to_universe(
                pred,
                universe=tradable_universe,
                start_time=fold.test_start,
                end_time=fold.test_end,
                run_id=run_ctx.run_id,
            )

        validate_pred_series(pred, run_ctx.run_id)
        fold_preds.append(pred)
        fold_metrics_list.append(fold_metrics)
        _persist_fold_artifacts(fold_dir, fold, pred, fold_metrics)
        _log(run_ctx.run_id, f"[fold {fold.fold_id}] 产物已写入 {fold_dir}")

    # ── 拼接 OOS pred_score ───────────────────────────────────────
    oos_pred = concat_oos_pred_scores(fold_preds)
    _log(
        run_ctx.run_id,
        f"[walk_forward] OOS 拼接完成，共 {len(oos_pred)} 条，"
        f"datetime 范围: {oos_pred.index.get_level_values('datetime').min()} ~ "
        f"{oos_pred.index.get_level_values('datetime').max()}",
    )

    # ── 回测（一次，覆盖完整 OOS 区间）──────────────────────────────
    oos_cfg = copy.deepcopy(cfg)
    oos_cfg["dataset"]["kwargs"]["segments"]["test"] = [
        folds[0].test_start, folds[-1].test_end
    ]
    # 确保 experiment 区间也覆盖完整 OOS
    oos_cfg["experiment"]["start_time"] = folds[0].test_start
    oos_cfg["experiment"]["end_time"] = folds[-1].test_end

    concat_dir = run_ctx.output_dir / "concatenated"
    concat_dir.mkdir(parents=True, exist_ok=True)

    # 获取 OOS 测试集标签（用于 IC/ICIR 计算，取最后一个 fold 的完整 label）
    oos_label: Optional[pd.Series] = None

    # 预生成 rolling_summary（不含 concatenated_metrics，在回测完成后补充）
    preliminary_summary = folds_to_summary(folds, per_fold_metrics=fold_metrics_list)

    t_bt = time.perf_counter()
    report_df, indicator_df, trades_df, realized_pnl_df, metrics = run_backtest(
        oos_cfg,
        oos_pred,
        output_dir=concat_dir,
        label=oos_label,
        rolling_summary=preliminary_summary,
    )
    run_ctx.timing["backtest_sec"] = round(time.perf_counter() - t_bt, 2)
    _log(run_ctx.run_id, f"[walk_forward] 回测完成 [{run_ctx.timing['backtest_sec']}s]")

    # ── 持久化 concatenated/ ───────────────────────────────────────
    concat_ctx = RunContext(run_id=run_ctx.run_id, output_dir=concat_dir)
    concat_ctx.timing = run_ctx.timing.copy()

    rolling_meta = {
        "fold_count": len(folds),
        "mode": "walk_forward",
        "oos_start": folds[0].test_start,
        "oos_end": folds[-1].test_end,
    }
    persist_artifacts(
        concat_ctx, cfg, metrics,
        report_df, indicator_df, oos_pred,
        trades_df=trades_df,
        realized_pnl_df=realized_pnl_df,
        rolling_meta=rolling_meta,
    )
    _log(run_ctx.run_id, f"[walk_forward] concatenated 产物已写入 {concat_dir}")

    # ── rolling_summary.json（补充最终 concatenated_metrics）────────
    rolling_summary = preliminary_summary
    rolling_summary["concatenated_metrics"] = metrics
    with (run_ctx.output_dir / "rolling_summary.json").open("w", encoding="utf-8") as f:
        json.dump(rolling_summary, f, ensure_ascii=False, indent=2)
    _log(run_ctx.run_id, f"[walk_forward] rolling_summary.json 已写入 {run_ctx.output_dir}")

    # ── 汇总打印 ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  Walk-Forward 完成 ({len(folds)} folds)")
    print(f"  OOS 总区间: {folds[0].test_start} ~ {folds[-1].test_end}")
    print(f"  ── per-fold IC ────────────────────────────")
    for fm in fold_metrics_list:
        ic_str = f"{fm.get('ic_mean', float('nan')):.4f}" if fm.get("ic_mean") is not None else "n/a"
        print(f"    fold {fm['fold_id']}: ic_mean={ic_str}  test=[{fm['test_start']},{fm['test_end']}]")
    print(f"  ── 阶段耗时 ───────────────────────────────")
    for k, v in run_ctx.timing.items():
        print(f"    {k:<28}: {v}s")
    print(f"  输出目录: {run_ctx.output_dir}")
    print(f"{'=' * 60}\n")


# ─────────────────────────────────────────────────────────────────
# 日志辅助
# ─────────────────────────────────────────────────────────────────

def _log(run_id: str, msg: str) -> None:
    LOGGER.info(
        msg,
        extra={"run_id": run_id, "instrument": "ALL", "datetime": "", "signal": 0.0, "version": VERSION},
    )


# ─────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Qlib 多模型多频率实验入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 组合式（推荐）—— 模型名和频率名大小写不敏感，支持别名
  python python/run_experiment.py --model mlp --freq daily
  python python/run_experiment.py --model lstm --freq daily
  python python/run_experiment.py --model gru --freq daily
  python python/run_experiment.py --model transformer --freq daily
  python python/run_experiment.py --model lgbm --freq daily

  # 别名同样有效
  python python/run_experiment.py --model LightGBM --freq Daily
  python python/run_experiment.py --model LSTM --freq 1day

  # 完整配置（向下兼容旧配置文件）
  python python/run_experiment.py --config configs/mlp_daily_backtest.yaml

模型别名: mlp/MLP/dnn | lstm/LSTM | gru/GRU | transformer/Transformer | lgbm/LGBM/LightGBM/lightgbm/gbdt
频率别名: daily/Daily/day/1day/d | minute/min/1min/m
        """,
    )
    parser.add_argument("--model", default=None, help="模型名称（mlp/lstm/gru/transformer/lgbm，大小写不敏感）")
    parser.add_argument("--freq", default=None, help="频率名称（daily/minute，别名如 1day/day 均可）")
    parser.add_argument("--config", default=None, help="完整 YAML 配置文件路径（旧接口兼容）")
    parser.add_argument(
        "--ensemble-seeds",
        type=int,
        default=1,
        help=(
            "LGBM 多 seed 集成数量（>=1）。N=1 时行为不变；N>1 时复用同一 dataset，"
            "用 base_seed + i*1000 生成 N 个 seed，每个独立训练，最终按 --aggregator 聚合。"
        ),
    )
    parser.add_argument(
        "--aggregator",
        choices=["rank_mean", "topk_vote"],
        default="rank_mean",
        help=(
            "ensemble 聚合方式：rank_mean=按日截面 pct-rank 取均值（默认，金融标准做法）；"
            "topk_vote=每 seed 各取自身 topK 后按多数票汇总（保留头部锐度）。"
        ),
    )
    parser.add_argument(
        "--feature-fraction",
        type=float,
        default=None,
        help="临时覆盖 lgbm.yaml 的 feature_fraction（不传则保持配置文件 0.7）。降到 0.5 可显著扩大 per-seed 多样性。",
    )
    parser.add_argument(
        "--bagging-fraction",
        type=float,
        default=None,
        help="临时覆盖 lgbm.yaml 的 bagging_fraction（不传则保持配置文件 0.7）。",
    )
    parser.add_argument(
        "--strategy-cfg",
        default=None,
        help=(
            "可选的策略配置 YAML 文件路径（相对仓库根），用于覆盖 base.yaml 中的 strategy 段。"
            "示例：--strategy-cfg configs/strategy/midfreq_rp.yaml 切换到中频风险平价策略。"
            "不传则使用 base.yaml 默认的 TopkDropoutStrategy。"
        ),
    )

    # ── Walk-Forward 参数（rolling 模式） ───────────────────────────
    parser.add_argument(
        "--rolling-mode",
        choices=["none", "walk_forward"],
        default="none",
        help=(
            "滚动训练模式（默认 none = 单切分）。"
            "walk_forward: 扩展窗口，每 step 生成新 fold，OOS pred_score 串接后统一回测。"
        ),
    )
    parser.add_argument(
        "--rolling-cfg",
        default=None,
        help=(
            "Walk-Forward 配置 YAML 路径（相对仓库根），"
            "如 configs/rolling/expanding_halfyearly.yaml。"
            "若指定则从文件读取 rolling.* 参数，CLI 参数可继续覆盖。"
        ),
    )
    parser.add_argument("--rolling-train-min", default=None,
                        help="最小训练窗口（如 '4y'），覆盖 rolling-cfg 中的值")
    parser.add_argument("--rolling-valid", default=None,
                        help="每 fold 验证集大小（如 '6m'）")
    parser.add_argument("--rolling-test", default=None,
                        help="每 fold OOS 测试集大小（如 '6m'）")
    parser.add_argument("--rolling-step", default=None,
                        help="相邻 fold 步长（如 '6m'）")
    parser.add_argument("--rolling-purge-days", type=int, default=None,
                        help="train/valid 间 purge 天数（None 时从 label 表达式自动推导）")
    parser.add_argument("--rolling-embargo-days", type=int, default=None,
                        help="purge 后 embargo 天数（默认 5）")

    args = parser.parse_args()

    wall_start = time.perf_counter()

    # ── 配置加载与合并 ─────────────────────────────────────────────
    cfg = load_and_merge_configs(
        model_name=args.model,
        freq_name=args.freq,
        full_config=args.config,
        strategy_cfg_path=args.strategy_cfg,
    )
    cfg = patch_dataset_class(cfg)
    cfg = inject_feature_config(cfg)

    # ── ensemble 超参临时覆盖（CLI > 配置文件） ─────────────────────
    if args.feature_fraction is not None or args.bagging_fraction is not None:
        kw = cfg.setdefault("model", {}).setdefault("kwargs", {})
        if args.feature_fraction is not None:
            kw["feature_fraction"] = float(args.feature_fraction)
        if args.bagging_fraction is not None:
            kw["bagging_fraction"] = float(args.bagging_fraction)

    # ── qlib 初始化 ────────────────────────────────────────────────
    provider_uri = cfg["qlib_init"]["provider_uri"]
    validate_provider_uri(provider_uri)
    qlib.init(provider_uri=provider_uri, region=cfg["qlib_init"]["region"])

    # ── RunContext ─────────────────────────────────────────────────
    base_run_id = cfg["experiment"]["run_id"]
    unique_run_id = f"{base_run_id}_{uuid.uuid4().hex[:8]}"
    run_ctx = RunContext(
        run_id=unique_run_id,
        output_dir=WORKSPACE_ROOT / "artifacts" / unique_run_id,
    )

    model_name = cfg.get("model_meta", {}).get("name", "unknown")
    freq_name = cfg.get("freq_meta", {}).get("name", "unknown")
    freq_key = cfg.get("model_meta", {}).get("freq_key", "1day")

    experiment_name = f"{model_name}_{freq_name}_backtest"

    n_ensemble = max(1, int(args.ensemble_seeds or 1))
    aggregator = args.aggregator

    # ── 解析 rolling 参数（文件 + CLI 覆盖） ───────────────────────────────
    rolling_mode = args.rolling_mode  # "none" | "walk_forward"
    rolling_params: Dict[str, Any] = {}
    if args.rolling_cfg:
        raw_rolling_cfg = _load_yaml(WORKSPACE_ROOT / args.rolling_cfg)
        rolling_params = raw_rolling_cfg.get("rolling", {})
    # CLI 参数覆盖文件配置
    if args.rolling_train_min is not None:
        rolling_params["train_min"] = args.rolling_train_min
    if args.rolling_valid is not None:
        rolling_params["valid_size"] = args.rolling_valid
    if args.rolling_test is not None:
        rolling_params["test_size"] = args.rolling_test
    if args.rolling_step is not None:
        rolling_params["step"] = args.rolling_step
    if args.rolling_purge_days is not None:
        rolling_params["purge_days"] = args.rolling_purge_days
    if args.rolling_embargo_days is not None:
        rolling_params["embargo_days"] = args.rolling_embargo_days
    # rolling_mode CLI 优先级最高
    if rolling_mode != "none":
        rolling_params.setdefault("mode", rolling_mode)
    elif rolling_params.get("mode") == "walk_forward":
        rolling_mode = "walk_forward"

    with R.start(experiment_name=experiment_name, recorder_name=run_ctx.run_id):
        _log(
            run_ctx.run_id,
            f"run_start (rolling_mode={rolling_mode}, ensemble_seeds={n_ensemble}, aggregator={aggregator})",
        )

        # ── Walk-Forward 模式 ──────────────────────────────────────────────
        if rolling_mode == "walk_forward":
            if n_ensemble > 1:
                _log(run_ctx.run_id,
                     "walk_forward 模式忽略 --ensemble-seeds，每 fold 使用单 seed 训练")
            run_walk_forward(cfg, run_ctx, rolling_params)
            run_ctx.timing["total_sec"] = round(time.perf_counter() - wall_start, 2)
            _log(run_ctx.run_id, "run_end (walk_forward)")
            return

        # ── 单切分模式（原有流程） ─────────────────────────────────────────
        # Step 1: 构建数据集
        t0 = time.perf_counter()
        dataset = build_dataset(cfg)
        run_ctx.timing["dataset_build_sec"] = round(time.perf_counter() - t0, 2)
        _log(run_ctx.run_id, f"dataset_ready [{run_ctx.timing['dataset_build_sec']}s]")

        # Step 2: 训练模型（单 seed 或 N-seed ensemble）
        _trained_model: Optional[Any] = None  # 暂存单 seed 模型供后续序列化
        if n_ensemble == 1:
            t1 = time.perf_counter()
            _trained_model = train_model(cfg, dataset)
            run_ctx.timing["train_sec"] = round(time.perf_counter() - t1, 2)
            _log(run_ctx.run_id, f"train_done [{run_ctx.timing['train_sec']}s]")

            # Step 2.5: 推理并校验
            pred_score: pd.Series = _trained_model.predict(dataset, segment="test")
        else:
            base_seed = int(cfg["model"].get("kwargs", {}).get("seed", 42))
            seeds = [base_seed + i * 1000 for i in range(n_ensemble)]
            _log(run_ctx.run_id, f"ensemble_start n_seeds={n_ensemble} seeds={seeds}")
            t1 = time.perf_counter()
            pred_score, _ = train_predict_ensemble(
                cfg, dataset, seeds, run_ctx, aggregator=aggregator,
            )
            _log(
                run_ctx.run_id,
                f"ensemble_total_train_sec={round(time.perf_counter() - t1, 2)}s",
            )

        validate_pred_series(pred_score, run_ctx.run_id)

        # Step 2.6: 限定可投资 universe（仅在该股票池中选股，避免选到 ST/小盘/退市风险股）
        tradable_universe = cfg.get("experiment", {}).get("tradable_universe")
        if tradable_universe and tradable_universe.lower() != "all":
            test_seg = cfg["dataset"]["kwargs"]["segments"]["test"]
            pred_score = filter_pred_to_universe(
                pred_score,
                universe=tradable_universe,
                start_time=test_seg[0],
                end_time=test_seg[1],
                run_id=run_ctx.run_id,
            )
            validate_pred_series(pred_score, run_ctx.run_id)

        # 尝试获取测试集标签（用于 IC/ICIR 计算）
        test_label: Optional[pd.Series] = None
        try:
            test_data = dataset.prepare("test", col_set="label")
            if isinstance(test_data, pd.DataFrame) and not test_data.empty:
                test_label = test_data.iloc[:, 0]
        except Exception:
            pass

        # Step 3: 回测（分层执行：SignalLayer → ... → AnalysisLayer）
        t2 = time.perf_counter()
        report_df, indicator_df, trades_df, realized_pnl_df, metrics = run_backtest(
            cfg,
            pred_score,
            output_dir=run_ctx.output_dir,
            label=test_label,
        )
        run_ctx.timing["backtest_sec"] = round(time.perf_counter() - t2, 2)
        _log(run_ctx.run_id, f"backtest_done [{run_ctx.timing['backtest_sec']}s]")

        run_ctx.timing["total_sec"] = round(time.perf_counter() - wall_start, 2)

        # Step 4: 持久化（图表/HTML 已由 BacktestEngine 写入，此处写数据文件）
        persist_artifacts(
            run_ctx, cfg, metrics, report_df, indicator_df, pred_score,
            trades_df=trades_df, realized_pnl_df=realized_pnl_df,
            model=_trained_model,  # 单 seed 时序列化模型供实盘复用
        )

        _log(run_ctx.run_id, "run_end")

        # ── 阶段耗时汇总（回测报告由 BacktestEngine 在上方已打印）─────────
        print(f"  ── 阶段耗时 ──────────────────────────────────────────")
        for k, v in run_ctx.timing.items():
            print(f"    {k:<28}: {v}s")
        print(f"  输出目录: {run_ctx.output_dir}")
        print()


if __name__ == "__main__":
    main()
