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
    indicator.parquet        回测成交指标（如有）
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
) -> Dict[str, Any]:
    """
    根据参数组合或完整配置路径，返回最终合并的配置字典。

    合并顺序: base.yaml → models/<model>.yaml → freq/<freq>.yaml
    """
    configs_dir = WORKSPACE_ROOT / "configs"

    if full_config:
        # 方式2：直接使用完整配置文件（兼容 mlp_daily_backtest.yaml 等旧配置）
        cfg = _load_yaml(WORKSPACE_ROOT / full_config)
        # 注入必要的 model_meta 默认值（旧文件可能没有）
        cfg.setdefault("model_meta", {})
        cfg["model_meta"].setdefault("dataset_type", "flat")
        cfg["model_meta"].setdefault("freq_key", "1day")
        return cfg

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

    return merged


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
    n_feat = len(names)

    LOGGER.info(
        "feature_inject: groups=%s, n_feat=%d, names=%s",
        feature_groups or "ALL",
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
# 回测（委托 BacktestEngine 分层处理）
# ─────────────────────────────────────────────────────────────────

def run_backtest(
    cfg: Dict[str, Any],
    pred_score: pd.Series,
    output_dir: Optional[Path] = None,
    label: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    """
    通过 BacktestEngine 执行分层回测，返回 (report_df, indicator_df, metrics_dict)。

    BacktestEngine 内部按层执行：
      SignalLayer → StrategyLayer → ExecutionLayer → MarketLayer
      → qlib.backtest → PortfolioLayer → AnalysisLayer
    """
    engine = BacktestEngine()
    result = engine.run(pred_score, cfg, output_dir=output_dir, label=label)
    metrics_dict = engine.metrics_to_dict(result)
    return result.report_df, result.indicator_df, metrics_dict


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
) -> None:
    """持久化回测产物：parquet 数据文件 + metrics.json 汇总。

    图表与 HTML 报告已由 BacktestEngine 在 run_backtest 阶段写入同一目录。
    """
    run_ctx.output_dir.mkdir(parents=True, exist_ok=True)
    report_df.to_parquet(run_ctx.output_dir / "backtest_report.parquet")
    pred_score.to_frame("score").to_parquet(run_ctx.output_dir / "signals.parquet")
    if not indicator_df.empty:
        indicator_df.to_parquet(run_ctx.output_dir / "indicator.parquet")

    final_output = {
        "run_id": run_ctx.run_id,
        "version": VERSION,
        "model": cfg.get("model_meta", {}).get("name", "unknown"),
        "freq": cfg.get("freq_meta", {}).get("name", "unknown"),
        "timing_seconds": run_ctx.timing,
        "metrics": metrics,
    }
    with (run_ctx.output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(final_output, f, ensure_ascii=False, indent=2)


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
    args = parser.parse_args()

    wall_start = time.perf_counter()

    # ── 配置加载与合并 ─────────────────────────────────────────────
    cfg = load_and_merge_configs(
        model_name=args.model,
        freq_name=args.freq,
        full_config=args.config,
    )
    cfg = patch_dataset_class(cfg)
    cfg = inject_feature_config(cfg)

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

    with R.start(experiment_name=experiment_name, recorder_name=run_ctx.run_id):
        _log(run_ctx.run_id, "run_start")

        # Step 1: 构建数据集
        t0 = time.perf_counter()
        dataset = build_dataset(cfg)
        run_ctx.timing["dataset_build_sec"] = round(time.perf_counter() - t0, 2)
        _log(run_ctx.run_id, f"dataset_ready [{run_ctx.timing['dataset_build_sec']}s]")

        # Step 2: 训练模型
        t1 = time.perf_counter()
        model = train_model(cfg, dataset)
        run_ctx.timing["train_sec"] = round(time.perf_counter() - t1, 2)
        _log(run_ctx.run_id, f"train_done [{run_ctx.timing['train_sec']}s]")

        # Step 2.5: 推理并校验
        pred_score: pd.Series = model.predict(dataset, segment="test")
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
        report_df, indicator_df, metrics = run_backtest(
            cfg,
            pred_score,
            output_dir=run_ctx.output_dir,
            label=test_label,
        )
        run_ctx.timing["backtest_sec"] = round(time.perf_counter() - t2, 2)
        _log(run_ctx.run_id, f"backtest_done [{run_ctx.timing['backtest_sec']}s]")

        run_ctx.timing["total_sec"] = round(time.perf_counter() - wall_start, 2)

        # Step 4: 持久化（图表/HTML 已由 BacktestEngine 写入，此处写数据文件）
        persist_artifacts(run_ctx, cfg, metrics, report_df, indicator_df, pred_score)

        _log(run_ctx.run_id, "run_end")

        # ── 阶段耗时汇总（回测报告由 BacktestEngine 在上方已打印）─────────
        print(f"  ── 阶段耗时 ──────────────────────────────────────────")
        for k, v in run_ctx.timing.items():
            print(f"    {k:<28}: {v}s")
        print(f"  输出目录: {run_ctx.output_dir}")
        print()


if __name__ == "__main__":
    main()
