"""
Optuna 超参搜索：以 Walk-Forward OOS Sharpe 为目标函数优化 LGBM 超参。
=======================================================================

层级位置：训练层工具（Model → Signal 之间的超参优化入口）

核心设计：
  - 目标函数：WF 3-fold OOS 拼接后的年化 Sharpe（非 IC，对齐 TopK 选股实际收益）
  - DatasetH.handler 一次初始化，跨所有 trial 复用（避免每 trial 重加载数据）
  - BacktestEngine.run(output_dir=None) 不写磁盘，只返回指标
  - SQLite 持久化，支持中断后 --resume 续跑
  - 搜索完毕后自动生成 configs/models/lgbm_optuna.yaml

使用方式：
  # 首次启动（50 trials，~250 分钟）
  python python/tuning/optuna_lgbm.py --n-trials 50

  # 中断后续跑
  python python/tuning/optuna_lgbm.py --n-trials 50 --resume

  # 自定义搜索范围
  python python/tuning/optuna_lgbm.py --n-trials 30 --sampler tpe

防前视偏差说明：
  - WF fold 的 test 区间是 OOS，train/valid 区间在 test 之前
  - BacktestEngine 使用 OOS pred_score 做回测，无前视风险
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# 将 python/ 目录加入 sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
_logger = logging.getLogger("optuna_lgbm")

# ─────────────────────────────────────────────────────────────────
# 搜索空间定义（7 维）
# ─────────────────────────────────────────────────────────────────

_SEARCH_SPACE: Dict[str, Any] = {
    "num_leaves":         {"type": "int",   "low": 15,   "high": 63},
    "learning_rate":      {"type": "float", "low": 0.005, "high": 0.05, "log": True},
    "min_child_samples":  {"type": "int",   "low": 50,   "high": 500},
    "feature_fraction":   {"type": "float", "low": 0.50, "high": 0.95},
    "bagging_fraction":   {"type": "float", "low": 0.50, "high": 0.95},
    "reg_lambda":         {"type": "float", "low": 0.5,  "high": 10.0, "log": True},
    "extra_trees":        {"type": "cat",   "choices": [True, False]},
}


def sample_params(trial: Any) -> Dict[str, Any]:
    """从 Optuna trial 中采样超参。"""
    params: Dict[str, Any] = {}
    for name, spec in _SEARCH_SPACE.items():
        t = spec["type"]
        if t == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif t == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif t == "cat":
            params[name] = trial.suggest_categorical(name, spec["choices"])
    return params


def inject_params(cfg: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
    """将超参注入配置副本的 model.kwargs 中。"""
    cfg = copy.deepcopy(cfg)
    model_kwargs = cfg.setdefault("model", {}).setdefault("kwargs", {})
    model_kwargs.update(params)
    return cfg


# ─────────────────────────────────────────────────────────────────
# 配置加载
# ─────────────────────────────────────────────────────────────────

def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """递归深度合并，override 的值覆盖 base（list 整体替换）。"""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_base_cfg(model: str = "lgbm", freq: str = "daily_retrain") -> Dict[str, Any]:
    """加载并合并 base + model + freq 配置，与 run_experiment.py 逻辑一致。"""
    configs_dir = _REPO_ROOT / "configs"
    base = _load_yaml(configs_dir / "base.yaml")
    model_cfg = _load_yaml(configs_dir / "models" / f"{model}.yaml")
    freq_cfg = _load_yaml(configs_dir / "freq" / f"{freq}.yaml")
    merged = _deep_merge(_deep_merge(base, model_cfg), freq_cfg)
    # 将 dataset_segments 注入 dataset.kwargs.segments（与 run_experiment.py 第 263 行一致）
    if "dataset_segments" in merged:
        merged.setdefault("dataset", {}).setdefault("kwargs", {})["segments"] = merged.pop(
            "dataset_segments"
        )
    return merged


# ─────────────────────────────────────────────────────────────────
# Walk-Forward OOS Sharpe（轻量版，不写磁盘）
# ─────────────────────────────────────────────────────────────────

def _wf_oos_sharpe(
    cfg: Dict[str, Any],
    shared_handler: Any,
    trial_id: int,
    output_dir: Optional[Path] = None,
    strategy_override: Optional[Dict[str, Any]] = None,
) -> float:
    """
    执行 WF 3-fold，拼接 OOS pred_score 后用 BacktestEngine 计算 Sharpe。

    Args:
        cfg:               注入了超参的配置字典（已 deep copy，不修改原对象）
        shared_handler:    预初始化的 DatasetH.handler（跨 trial 复用，避免重加载）
        trial_id:          trial 序号（仅用于日志）
        output_dir:        若非 None，将回测产物写到该目录（用于最优 trial 的可视化）
        strategy_override: 若非 None，将覆盖 oos_cfg 中的 strategy 段（对齐部署策略）

    Returns:
        OOS Sharpe（失败返回 -inf）
    """
    try:
        import qlib
        from qlib.data.dataset import DatasetH
        from qlib.model.base import Model
        from qlib.utils import init_instance_by_config
    except ImportError as e:
        _logger.error("qlib 不可用: %s", e)
        return float("-inf")

    from backtest import BacktestEngine
    from backtest.rolling import (
        build_walk_forward_folds,
        concat_oos_pred_scores,
        infer_purge_days,
    )

    # ── WF fold 切分 ──────────────────────────────────────────────
    data_start = cfg.get("experiment", {}).get("start_time", "2020-01-01")
    data_end = cfg.get("experiment", {}).get("end_time", "2026-05-12")

    label_cfg = (
        cfg.get("dataset", {}).get("kwargs", {}).get("handler", {})
        .get("kwargs", {}).get("data_loader", {}).get("kwargs", {})
        .get("config", {}).get("label", [])
    )
    label_expr = ""
    if label_cfg and isinstance(label_cfg[0], list) and label_cfg[0]:
        label_expr = label_cfg[0][0]
    purge_days = infer_purge_days(label_expr)

    folds = build_walk_forward_folds(
        data_start=data_start,
        data_end=data_end,
        train_min="4y",
        valid_size="6m",
        test_size="6m",
        step="6m",
        purge_days=purge_days,
        embargo_days=5,
        oos_start="2025-01-01",
    )

    if not folds:
        _logger.warning("[trial %d] WF 产生 0 个 fold，跳过", trial_id)
        return float("-inf")

    _logger.info("[trial %d] WF %d folds: %s ~ %s",
                 trial_id, len(folds), folds[0].test_start, folds[-1].test_end)

    # ── 逐 fold 训练 + 预测 ───────────────────────────────────────
    base_seed = int(cfg.get("model", {}).get("kwargs", {}).get("seed", 42))
    fold_preds: List[pd.Series] = []

    for fold in folds:
        # 复用共享 handler
        dataset = DatasetH(
            handler=shared_handler,
            segments={
                "train": list(fold.train),
                "valid": list(fold.valid),
                "test":  list(fold.test),
            },
        )

        # 注入 fold seed（与 run_experiment._train_fold 逻辑一致）
        fold_model_cfg = copy.deepcopy(cfg["model"])
        kw = fold_model_cfg.setdefault("kwargs", {})
        for k in ("seed", "feature_fraction_seed", "bagging_seed", "data_random_seed"):
            kw[k] = base_seed

        t0 = time.perf_counter()
        model: Model = init_instance_by_config(fold_model_cfg)
        model.fit(dataset)
        train_sec = round(time.perf_counter() - t0, 1)

        pred: pd.Series = model.predict(dataset, segment="test")
        if isinstance(pred, pd.DataFrame):
            pred = pred.iloc[:, 0]
        pred = pred.rename("score")
        fold_preds.append(pred)
        _logger.info("[trial %d] fold %d 训练完成 [%.1fs] n_pred=%d",
                     trial_id, fold.fold_id, train_sec, len(pred))

    # ── 拼接 OOS pred_score ───────────────────────────────────────
    oos_pred = concat_oos_pred_scores(fold_preds)

    # ── BacktestEngine（不写磁盘，除非 output_dir 指定）──────────
    oos_cfg = copy.deepcopy(cfg)
    oos_cfg["dataset"]["kwargs"]["segments"]["test"] = [
        folds[0].test_start, folds[-1].test_end
    ]
    oos_cfg["experiment"]["start_time"] = folds[0].test_start
    oos_cfg["experiment"]["end_time"] = folds[-1].test_end

    # 部署策略对齐：用调用方指定的 strategy 替换 cfg 中默认的 TopkDropoutStrategy。
    # 这样 Optuna 搜出的超参对应的是真实部署环境下的 Sharpe，而非仅信号质量。
    if strategy_override and "strategy" in strategy_override:
        oos_cfg["strategy"] = copy.deepcopy(strategy_override["strategy"])
        _logger.debug("[trial %d] strategy 已覆盖为: %s",
                      trial_id, strategy_override["strategy"].get("class", "unknown"))

    engine = BacktestEngine()
    result = engine.run(oos_pred, oos_cfg, output_dir=output_dir)
    sharpe = result.metrics.sharpe_ratio
    calmar = result.metrics.calmar_ratio
    ann_ret = result.metrics.annualized_return
    ann_vol = result.metrics.annualized_volatility
    mdd = result.metrics.max_drawdown

    _logger.info(
        "[trial %d] sharpe=%.4f calmar=%.4f ann_ret=%.2f%% ann_vol=%.2f%% mdd=%.2f%%",
        trial_id, sharpe, calmar, ann_ret * 100, ann_vol * 100, mdd * 100,
    )
    return float(sharpe)


# ─────────────────────────────────────────────────────────────────
# Optuna 目标函数
# ─────────────────────────────────────────────────────────────────

class _Objective:
    """Optuna objective 封装，持有跨 trial 共享的 handler、base_cfg、strategy_override。"""

    def __init__(
        self,
        base_cfg: Dict[str, Any],
        shared_handler: Any,
        trial_output_dir: Optional[Path] = None,
        strategy_override: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.base_cfg = base_cfg
        self.shared_handler = shared_handler
        self.trial_output_dir = trial_output_dir
        self.strategy_override = strategy_override
        self._trial_count = 0

    def __call__(self, trial: Any) -> float:
        self._trial_count += 1
        trial_id = trial.number

        params = sample_params(trial)
        _logger.info("[trial %d] params: %s", trial_id, params)

        trial_cfg = inject_params(self.base_cfg, params)

        # 设置 trial 参数为 trial.attrs，方便后续查询
        for k, v in params.items():
            trial.set_user_attr(k, v)

        t0 = time.perf_counter()
        sharpe = _wf_oos_sharpe(
            cfg=trial_cfg,
            shared_handler=self.shared_handler,
            trial_id=trial_id,
            strategy_override=self.strategy_override,
        )
        elapsed = round(time.perf_counter() - t0, 1)
        trial.set_user_attr("elapsed_sec", elapsed)
        trial.set_user_attr("sharpe", sharpe)

        _logger.info("[trial %d] 完成 sharpe=%.4f [%.1fs]", trial_id, sharpe, elapsed)
        return sharpe


# ─────────────────────────────────────────────────────────────────
# 结果输出：生成 lgbm_optuna.yaml
# ─────────────────────────────────────────────────────────────────

def _write_optuna_yaml(best_params: Dict[str, Any], base_lgbm_yaml: Path) -> Path:
    """以 lgbm.yaml 为基础，将 best_params 注入后写出 lgbm_optuna.yaml。"""
    with base_lgbm_yaml.open("r", encoding="utf-8") as f:
        lgbm_cfg = yaml.safe_load(f) or {}

    lgbm_cfg.setdefault("model", {}).setdefault("kwargs", {}).update(best_params)
    # 在 model_meta.name 中标记来源
    lgbm_cfg.setdefault("model_meta", {})["name"] = "lgbm_optuna"

    out_path = base_lgbm_yaml.parent / "lgbm_optuna.yaml"
    with out_path.open("w", encoding="utf-8") as f:
        f.write("# 由 optuna_lgbm.py 自动生成，勿手动修改\n")
        f.write(f"# best WF Sharpe: 见 artifacts/optuna_lgbm_summary.json\n\n")
        yaml.dump(lgbm_cfg, f, allow_unicode=True, default_flow_style=False)
    return out_path


def _write_summary(study: Any, out_path: Path) -> None:
    """将 Optuna study 摘要写到 JSON 文件。"""
    best = study.best_trial
    summary = {
        "best_trial_number": best.number,
        "best_sharpe": best.value,
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "n_trials": len(study.trials),
        "n_complete": len([t for t in study.trials if t.state.name == "COMPLETE"]),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    _logger.info("摘要已写入 %s", out_path)


# ─────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna LGBM 超参搜索（目标函数：WF OOS Sharpe）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 首次启动（推荐：指定部署策略，保证评估-部署一致）
  python python/tuning/optuna_lgbm.py --n-trials 50 \\
      --strategy-cfg configs/strategy/daily_vol_target.yaml

  # 中断后续跑
  python python/tuning/optuna_lgbm.py --n-trials 50 --resume \\
      --strategy-cfg configs/strategy/daily_vol_target.yaml

  # 搜索完成后对比
  python python/run_experiment.py --model lgbm_optuna --freq daily_retrain \\
      --strategy-cfg configs/strategy/daily_vol_target.yaml
        """,
    )
    parser.add_argument("--n-trials", type=int, default=50, help="搜索 trial 数量（默认 50）")
    parser.add_argument(
        "--model", default="lgbm", help="基础模型名（默认 lgbm）"
    )
    parser.add_argument(
        "--freq", default="daily_retrain", help="频率名（默认 daily_retrain）"
    )
    parser.add_argument(
        "--storage",
        default=str(_REPO_ROOT / "artifacts" / "optuna_lgbm.db"),
        help="Optuna SQLite 存储路径（支持中断续跑）",
    )
    parser.add_argument(
        "--study-name", default="lgbm_wf_sharpe", help="Optuna study 名称"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="续跑已有 study（storage 中已存在时自动续跑，此 flag 仅为显式标记）",
    )
    parser.add_argument(
        "--sampler",
        choices=["tpe", "random"],
        default="tpe",
        help="采样器（tpe=贝叶斯，random=随机基线）",
    )
    parser.add_argument(
        "--strategy-cfg",
        default=None,
        help=(
            "部署策略配置 YAML 路径（相对仓库根），用于覆盖目标函数中的回测策略。"
            "例：--strategy-cfg configs/strategy/daily_vol_target.yaml。"
            "不指定时使用 base.yaml 默认的 TopkDropoutStrategy（与 T1 旧行为一致）。"
            "【推荐始终指定】保证 Optuna 评估策略与最终部署策略一致。"
        ),
    )
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        _logger.error("optuna 未安装，请运行：pip install optuna")
        sys.exit(1)

    # 初始化 qlib
    try:
        import qlib
        base_cfg = load_base_cfg(args.model, args.freq)
        qlib_init_cfg = base_cfg.get("qlib_init", {})
        qlib.init(
            provider_uri=qlib_init_cfg.get("provider_uri", "D:/qlib_data/qlib_data"),
            region=qlib_init_cfg.get("region", "cn"),
        )
        _logger.info("qlib 初始化完成，provider_uri=%s", qlib_init_cfg.get("provider_uri"))
    except Exception as e:
        _logger.error("qlib 初始化失败: %s", e)
        sys.exit(1)

    # 注入 feature config（与 run_experiment.inject_feature_config 一致）
    try:
        sys.path.insert(0, str(_REPO_ROOT / "python"))
        from run_experiment import inject_feature_config
        base_cfg = inject_feature_config(base_cfg)
        _logger.info("feature config 注入完成")
    except Exception as e:
        _logger.warning("inject_feature_config 失败（%s），继续运行", e)

    # 构建共享 handler（一次性，跨所有 trial 复用）
    _logger.info("初始化共享 DatasetH.handler（预加载全量数据）...")
    t0 = time.perf_counter()
    try:
        from qlib.utils import init_instance_by_config
        full_dataset = init_instance_by_config(base_cfg["dataset"])
        shared_handler = full_dataset.handler
        _logger.info("共享 handler 初始化完成 [%.1fs]", time.perf_counter() - t0)
    except Exception as e:
        _logger.error("共享 handler 初始化失败: %s", e)
        sys.exit(1)

    # 创建 / 加载 Optuna study
    storage_url = f"sqlite:///{args.storage}"
    Path(args.storage).parent.mkdir(parents=True, exist_ok=True)

    if args.sampler == "tpe":
        sampler = optuna.samplers.TPESampler(seed=42)
    else:
        sampler = optuna.samplers.RandomSampler(seed=42)

    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage_url,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,  # 已存在则续跑
    )

    n_existing = len(study.trials)
    n_remaining = max(0, args.n_trials - n_existing)
    if n_remaining == 0:
        _logger.info("study 已有 %d trials，目标 %d，无需新增", n_existing, args.n_trials)
    else:
        _logger.info(
            "开始搜索：已有 %d trials，新增 %d trials，目标 %d trials",
            n_existing, n_remaining, args.n_trials,
        )

    # 加载部署策略配置（修正 T1 评估-部署不一致问题）
    strategy_override: Optional[Dict[str, Any]] = None
    if args.strategy_cfg:
        strat_path = _REPO_ROOT / args.strategy_cfg
        if strat_path.exists():
            strategy_override = _load_yaml(strat_path)
            strat_class = (
                strategy_override.get("strategy", {}).get("class", "unknown")
            )
            _logger.info(
                "目标函数使用部署策略：%s（来自 %s）",
                strat_class, args.strategy_cfg,
            )
        else:
            _logger.warning("strategy_cfg 文件不存在：%s，使用 base 默认策略", strat_path)

    objective = _Objective(base_cfg, shared_handler, strategy_override=strategy_override)

    try:
        study.optimize(
            objective,
            n_trials=n_remaining,
            show_progress_bar=True,
            gc_after_trial=True,
        )
    except KeyboardInterrupt:
        _logger.info("用户中断，当前已完成 %d trials", len(study.trials))

    # ── 输出结果 ──────────────────────────────────────────────────
    if not study.trials:
        _logger.warning("无完成的 trial，退出")
        return

    best = study.best_trial
    _logger.info(
        "\n最优 trial #%d  WF Sharpe = %.4f\n超参: %s",
        best.number, best.value, json.dumps(best.params, indent=2),
    )

    # 生成 lgbm_optuna.yaml
    base_lgbm_yaml = _REPO_ROOT / "configs" / "models" / f"{args.model}.yaml"
    if base_lgbm_yaml.exists():
        out_yaml = _write_optuna_yaml(best.params, base_lgbm_yaml)
        _logger.info("已生成配置：%s", out_yaml)
        _logger.info(
            "对比基线命令：python python/run_experiment.py --model lgbm_optuna --freq %s",
            args.freq,
        )
    else:
        _logger.warning("基础 yaml 不存在：%s，跳过 lgbm_optuna.yaml 生成", base_lgbm_yaml)

    # 写摘要 JSON
    summary_path = _REPO_ROOT / "artifacts" / "optuna_lgbm_summary.json"
    _write_summary(study, summary_path)

    strat_suffix = f" --strategy-cfg {args.strategy_cfg}" if args.strategy_cfg else ""
    _logger.info(
        "搜索完成。续跑命令：python python/tuning/optuna_lgbm.py --n-trials %d --resume%s",
        args.n_trials, strat_suffix,
    )


if __name__ == "__main__":
    main()
