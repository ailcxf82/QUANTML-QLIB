"""
日频 MLP 训练 + 回测入口脚本。

层级位置: Data -> Feature -> Model -> Signal -> Portfolio -> Backtest -> Evaluation
输入:     configs/mlp_daily_backtest.yaml
输出:     artifacts/<run_id>/
失败处理: 数据校验失败或 backtest 异常均抛出并记录日志，不做静默吞异常。
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml

try:
    import qlib
    from qlib.backtest import backtest
    from qlib.contrib.evaluate import risk_analysis
    from qlib.contrib.strategy import TopkDropoutStrategy
    from qlib.data.dataset import DatasetH
    from qlib.log import get_module_logger
    from qlib.model.base import Model
    from qlib.utils import init_instance_by_config
    from qlib.workflow import R
except ImportError as exc:
    raise ImportError(
        "无法导入 Microsoft Qlib (pyqlib)。请在 qlib_zhengshi 虚拟环境中执行：\n"
        "  pip uninstall -y qlib && pip install pyqlib"
    ) from exc


LOGGER = get_module_logger("run_mlp_daily_backtest")
VERSION = "v1"
FREQ_KEY = "1day"


@dataclass
class RunContext:
    run_id: str
    output_dir: Path
    timing: Dict[str, float] = field(default_factory=dict)


def load_config(config_path: Path) -> Dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_provider_uri(provider_uri: str) -> None:
    data_path = Path(provider_uri)
    if not data_path.exists():
        raise FileNotFoundError(f"provider_uri 路径不存在: {provider_uri}")
    for required in ("calendars", "instruments"):
        if not (data_path / required).exists():
            raise FileNotFoundError(f"数据目录缺少子目录: {required}")


def validate_pred_series(pred: pd.Series, run_id: str) -> None:
    """校验预测分数序列的基本完整性。"""
    if pred.empty:
        raise ValueError("预测信号序列为空，请检查模型与数据集配置。")
    if not isinstance(pred.index, pd.MultiIndex):
        raise TypeError("预测信号索引必须是 MultiIndex(datetime, instrument)。")
    nan_ratio = float(pred.isna().mean())
    if nan_ratio > 0.5:
        LOGGER.warning(
            "pred_quality_warning",
            extra={
                "run_id": run_id,
                "instrument": "ALL",
                "datetime": "",
                "signal": 0.0,
                "version": VERSION,
            },
        )
        LOGGER.warning("预测信号 NaN 占比 %.1f%%，请检查特征与模型", nan_ratio * 100)


def init_qlib(cfg: Dict[str, Any]) -> None:
    provider_uri = cfg["qlib_init"]["provider_uri"]
    validate_provider_uri(provider_uri)
    qlib.init(provider_uri=provider_uri, region=cfg["qlib_init"]["region"])


def create_run_context(cfg: Dict[str, Any], workspace_root: Path) -> RunContext:
    base_run_id = cfg["experiment"]["run_id"]
    unique_run_id = f"{base_run_id}_{uuid.uuid4().hex[:8]}"
    output_dir = workspace_root / "artifacts" / unique_run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    return RunContext(run_id=unique_run_id, output_dir=output_dir)


def build_dataset(cfg: Dict[str, Any]) -> DatasetH:
    dataset: DatasetH = init_instance_by_config(cfg["dataset"])
    return dataset


def train_model(cfg: Dict[str, Any], dataset: DatasetH) -> Model:
    model: Model = init_instance_by_config(cfg["model"])
    model.fit(dataset)
    return model


def run_backtest(
    cfg: Dict[str, Any],
    pred_score: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """执行日频回测，返回 (portfolio_df, indicator_df)。"""
    exp_cfg = cfg["experiment"]
    strategy_cfg = cfg["strategy"]
    executor_cfg = cfg["executor"]
    bt_cfg = cfg["backtest"]["exchange_kwargs"]

    strategy_config = {
        "class": strategy_cfg["class"],
        "module_path": strategy_cfg["module_path"],
        "kwargs": {
            **strategy_cfg["kwargs"],
            "signal": pred_score,
        },
    }

    portfolio_metric_dict, indicator_dict = backtest(
        start_time=exp_cfg["test_start_time"],
        end_time=exp_cfg["end_time"],
        strategy=strategy_config,
        executor=executor_cfg,
        benchmark=exp_cfg["benchmark"],
        account=exp_cfg["account"],
        exchange_kwargs=bt_cfg,
    )

    # PORT_METRIC: Dict[str, Tuple[pd.DataFrame, dict]]，key 通常是 "1day"
    # 取第一个可用 freq，tuple[0] 即 portfolio 日报 DataFrame
    port_tuple = portfolio_metric_dict.get(FREQ_KEY) or next(iter(portfolio_metric_dict.values()))
    report_df = port_tuple[0] if isinstance(port_tuple, tuple) else port_tuple

    # INDICATOR_METRIC: Dict[str, Tuple[pd.DataFrame, Indicator]]
    # tuple[0] 是指标 DataFrame，tuple[1] 是 Indicator 对象
    indicator_df_list = []
    for _freq_key, ind_tuple in indicator_dict.items():
        if isinstance(ind_tuple, tuple) and len(ind_tuple) > 0:
            ind_df = ind_tuple[0]
            if isinstance(ind_df, pd.DataFrame) and not ind_df.empty:
                indicator_df_list.append(ind_df)
    indicator_df = pd.concat(indicator_df_list) if indicator_df_list else pd.DataFrame()

    return report_df, indicator_df


def compute_metrics(report_df: pd.DataFrame) -> Dict[str, float]:
    """计算策略评估所需全部指标。"""
    excess_return = report_df["return"] - report_df["bench"]
    analysis = risk_analysis(excess_return)

    # risk_analysis 返回 DataFrame：行为指标名，列为 "risk"
    def _get(name: str) -> float:
        try:
            return float(analysis.loc[name, "risk"])
        except (KeyError, TypeError):
            return float("nan")

    turnover_mean = float(report_df["turnover"].mean()) if "turnover" in report_df.columns else float("nan")
    win_rate = float((report_df["return"] > 0).mean())

    return {
        "annualized_return": _get("annualized_return"),
        "max_drawdown": _get("max_drawdown"),
        "sharpe": _get("sharpe"),
        "information_ratio": _get("information_ratio"),
        "turnover": turnover_mean,
        "win_rate": win_rate,
    }


def dump_json(data: Dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _log(run_id: str, msg: str, instrument: str = "ALL", dt: str = "", signal: float = 0.0) -> None:
    LOGGER.info(
        msg,
        extra={
            "run_id": run_id,
            "instrument": instrument,
            "datetime": dt,
            "signal": signal,
            "version": VERSION,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Qlib 日频 MLP 训练与回测。")
    parser.add_argument(
        "--config",
        default="configs/mlp_daily_backtest.yaml",
        help="YAML 配置文件路径（相对于项目根目录）",
    )
    args = parser.parse_args()

    wall_start = time.perf_counter()
    workspace_root = Path(__file__).resolve().parents[1]
    config_path = workspace_root / args.config
    cfg = load_config(config_path)

    init_qlib(cfg)
    run_ctx = create_run_context(cfg, workspace_root)

    with R.start(experiment_name="mlp_daily_backtest", recorder_name=run_ctx.run_id):
        _log(run_ctx.run_id, "run_start")

        # ── Step 1: 构建数据集并校验 ──────────────────────────────
        t0 = time.perf_counter()
        dataset = build_dataset(cfg)
        run_ctx.timing["dataset_build_sec"] = round(time.perf_counter() - t0, 2)
        _log(run_ctx.run_id, f"dataset_ready  [{run_ctx.timing['dataset_build_sec']}s]")

        # ── Step 2: 训练 MLP ──────────────────────────────────────
        t1 = time.perf_counter()
        model = train_model(cfg, dataset)
        run_ctx.timing["train_sec"] = round(time.perf_counter() - t1, 2)
        _log(run_ctx.run_id, f"train_done  [{run_ctx.timing['train_sec']}s]")

        # ── Step 2.5: 推理并校验预测质量 ──────────────────────────
        pred_score = model.predict(dataset, segment="test")
        validate_pred_series(pred_score, run_ctx.run_id)

        # ── Step 3: 日频回测 ──────────────────────────────────────
        t2 = time.perf_counter()
        report_df, indicator_df = run_backtest(cfg, pred_score)
        run_ctx.timing["backtest_sec"] = round(time.perf_counter() - t2, 2)
        _log(run_ctx.run_id, f"backtest_done  [{run_ctx.timing['backtest_sec']}s]")

        # ── Step 4: 计算评估指标 ──────────────────────────────────
        metrics = compute_metrics(report_df)
        run_ctx.timing["total_sec"] = round(time.perf_counter() - wall_start, 2)

        # ── Step 5: 持久化输出 ────────────────────────────────────
        report_df.to_parquet(run_ctx.output_dir / "backtest_report.parquet")
        if not indicator_df.empty:
            indicator_df.to_parquet(run_ctx.output_dir / "indicator.parquet")

        final_output = {
            "run_id": run_ctx.run_id,
            "version": VERSION,
            "config": str(config_path),
            "timing_seconds": run_ctx.timing,
            "metrics": metrics,
        }
        dump_json(final_output, run_ctx.output_dir / "metrics.json")

        _log(run_ctx.run_id, "run_end")

        # ── 打印最终结果 ──────────────────────────────────────────
        print("\n" + "=" * 60)
        print(f"  run_id : {run_ctx.run_id}")
        print(f"  输出目录: {run_ctx.output_dir}")
        print("  ── 评估指标 ──")
        for k, v in metrics.items():
            print(f"    {k:<25}: {v:.4f}")
        print("  ── 阶段耗时 ──")
        for k, v in run_ctx.timing.items():
            print(f"    {k:<25}: {v}s")
        print("=" * 60)


if __name__ == "__main__":
    main()
