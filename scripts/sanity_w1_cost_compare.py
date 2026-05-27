"""
W1 撮合修正 sanity 对照工具
================================
不重新训练，直接复用某次 run 的 signals.parquet，分别用 "旧 cost"（W1 前）和
"新 cost"（W1 后）跑两次 BacktestEngine，输出 sharpe / 年化 / MDD / 换手对比。

设计动机（2026-05-26 W1）：
  撮合修正只动了 cost 与 deal_price，理论上影响：
    1) sharpe 略降（双边成本 +20bp → 年换手 800% 时 -1.6pp/年）
    2) MDD 不变（成本不影响最大回撤幅度，只影响速度）
    3) 换手率不变（策略层不消费成本，纯执行层差异）
  本脚本量化这三项，帮助用户判断 W1 修正的"代价"是否符合预期。

用法：
    # 用最近一次 lgbm_optuna run 做对比（自动找最新）
    python scripts/sanity_w1_cost_compare.py --auto

    # 显式指定 run_id
    python scripts/sanity_w1_cost_compare.py \\
        --run-id lgbm_optuna_daily_retrain_c276d396

    # 指定 freq 用于配置加载（默认 daily_retrain）
    python scripts/sanity_w1_cost_compare.py --auto --freq daily_retrain

输出：
    控制台对比表 + 写入 artifacts/_audit/w1_cost_sanity_<timestamp>.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python"))


# ─── W1 修正前的旧成本（用于回放对比）─────────────────────────────────
_OLD_COST: Dict[str, Any] = {
    "deal_price": "close",
    "open_cost": 0.0025,
    "close_cost": 0.0035,
}

# W1 修正后的新成本（与 freq yaml 一致）
_NEW_COST: Dict[str, Any] = {
    "deal_price": "$close_qfq",
    "open_cost": 0.0035,
    "close_cost": 0.0045,
}


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_merge(base: Dict, override: Dict) -> Dict:
    """深度递归合并；与 run_experiment.deep_merge 行为一致（list 整体替换）。"""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_base_cfg(model: str, freq: str) -> Dict[str, Any]:
    """加载并合并 base + model + freq 配置，与 run_experiment 一致。"""
    configs_dir = _REPO_ROOT / "configs"
    base = _load_yaml(configs_dir / "base.yaml")
    model_cfg = _load_yaml(configs_dir / "models" / f"{model}.yaml")
    freq_cfg = _load_yaml(configs_dir / "freq" / f"{freq}.yaml")
    merged = _deep_merge(_deep_merge(base, model_cfg), freq_cfg)
    if "dataset_segments" in merged:
        merged.setdefault("dataset", {}).setdefault("kwargs", {})["segments"] = (
            merged.pop("dataset_segments")
        )
    return merged


def _run_health(run_dir: Path) -> Tuple[bool, str, Dict[str, Any]]:
    """读取 metrics.json 判断 run 是否健康。

    返回 (是否健康, 描述, metrics dict)。"""
    metrics_candidates = [
        run_dir / "concatenated" / "metrics.json",
        run_dir / "metrics.json",
    ]
    raw: Dict[str, Any] = {}
    for mp in metrics_candidates:
        if mp.exists():
            try:
                with mp.open("r", encoding="utf-8") as f:
                    raw = json.load(f)
                break
            except Exception:
                pass
    metrics = raw.get("metrics", raw) if isinstance(raw, dict) and "metrics" in raw else raw
    timing = raw.get("timing_seconds", {}) if isinstance(raw, dict) else {}

    issues: List[str] = []
    ann_ret = metrics.get("annualized_return", 0.0)
    ann_vol = metrics.get("annualized_volatility", 0.0)
    win_rate = metrics.get("win_rate", 0.0)
    sharpe = metrics.get("sharpe_ratio")
    per_seed_ic = timing.get("ensemble_per_seed_ic", []) if isinstance(timing, dict) else []

    if ann_ret == 0.0 and ann_vol == 0.0:
        issues.append("零下单(ann_ret=0)")
    if win_rate == 0.0:
        issues.append("win_rate=0")
    if isinstance(sharpe, float) and sharpe != sharpe:  # NaN
        issues.append("sharpe=NaN")
    if per_seed_ic and all(isinstance(ic, (int, float)) and ic < 0 for ic in per_seed_ic):
        issues.append(f"per_seed_ic 全负={per_seed_ic}")

    return (not issues), ("OK" if not issues else "; ".join(issues)), metrics


def auto_pick_run(model_pattern: str, allow_unhealthy: bool = False) -> Path:
    """自动挑选合适的 run：优先健康 + signals 覆盖天数多 + 最近 mtime。

    历史 bug：之前按 mtime 选最新会挑到失败 run（如 per_seed_ic 全负 / 零下单），
    回测出来 sharpe -15、win_rate=0 等异常值，把 W1 撮合修正的真实代价（≈0）
    掩盖在噪声里。
    """
    artifacts_dir = _REPO_ROOT / "artifacts"
    candidates = [
        p for p in artifacts_dir.iterdir()
        if p.is_dir() and p.name.startswith(model_pattern)
    ]
    if not candidates:
        raise FileNotFoundError(
            f"未找到匹配 '{model_pattern}*' 的 run 目录。"
            f"请显式传 --run-id <full_run_id>。"
        )

    scored = []
    for run_dir in candidates:
        # 优先 concatenated/signals.parquet，否则 signals.parquet
        for sig in (run_dir / "concatenated" / "signals.parquet",
                    run_dir / "signals.parquet"):
            if sig.exists():
                try:
                    df = pd.read_parquet(sig, columns=None)
                    s = df.iloc[:, 0] if hasattr(df, "iloc") else df
                    n_days = s.index.get_level_values("datetime").nunique()
                except Exception:
                    n_days = 0
                break
        else:
            continue

        healthy, reason, _ = _run_health(run_dir)
        scored.append((run_dir, healthy, n_days, run_dir.stat().st_mtime, reason))

    if not scored:
        raise FileNotFoundError(
            f"匹配 '{model_pattern}*' 的 run 目录均无 signals.parquet"
        )

    if not allow_unhealthy:
        healthy_only = [s for s in scored if s[1]]
        if healthy_only:
            scored = healthy_only

    # 排序：健康 > 天数多 > 时间新
    scored.sort(key=lambda x: (x[1], x[2], x[3]), reverse=True)
    chosen = scored[0]
    print(f"[auto] 选中 run: {chosen[0].name}")
    print(f"       healthy={chosen[1]}  n_days={chosen[2]}  reason={chosen[4]}")
    if not chosen[1]:
        print("       警告：仅找到不健康 run，对照结果绝对值不可信；"
              "delta(新-旧 cost) 仍可参考")
    return chosen[0]


def load_signals(run_dir: Path) -> Tuple[pd.Series, Path]:
    """从 run 目录中读取 signals.parquet（兼容 walk_forward 的 concatenated 子目录）。

    返回 (Series, 实际使用的文件路径)。
    """
    candidates = [
        run_dir / "concatenated" / "signals.parquet",
        run_dir / "signals.parquet",
    ]
    for sig_path in candidates:
        if sig_path.exists():
            df = pd.read_parquet(sig_path)
            if isinstance(df, pd.DataFrame):
                series = df.iloc[:, 0]
            else:
                series = df
            series.name = "score"
            return series, sig_path
    raise FileNotFoundError(
        f"signals.parquet 未在 {run_dir} 或其 concatenated/ 子目录中找到"
    )


def align_cfg_to_signals(cfg: Dict[str, Any], pred_score: pd.Series) -> Dict[str, Any]:
    """把 cfg 的 dataset.segments.test / experiment 区间对齐到 signals 的实际范围。

    不动则会出现：cfg.test=[2026-05-01,2026-05-22]、signals 仅覆盖 [2025-01,2026-04]，
    BacktestEngine 跑出空持仓 / 极端负 sharpe 的假象。
    """
    cfg = copy.deepcopy(cfg)
    dates = pred_score.index.get_level_values("datetime")
    start, end = str(dates.min().date()), str(dates.max().date())
    cfg.setdefault("dataset", {}).setdefault("kwargs", {}).setdefault("segments", {})
    cfg["dataset"]["kwargs"]["segments"]["test"] = [start, end]
    cfg.setdefault("experiment", {})["start_time"] = start
    cfg["experiment"]["end_time"] = end
    return cfg


def run_backtest_with_cost(
    pred_score: pd.Series,
    base_cfg: Dict[str, Any],
    cost_override: Dict[str, Any],
    label: str,
) -> Dict[str, float]:
    """用指定 cost 跑一次回测，返回核心指标 dict。"""
    from backtest import BacktestEngine

    cfg = copy.deepcopy(base_cfg)
    ekw = cfg.setdefault("backtest", {}).setdefault("exchange_kwargs", {})
    ekw.update(cost_override)

    print(f"\n[{label}] cost={cost_override} ...")
    t0 = time.perf_counter()
    engine = BacktestEngine()
    result = engine.run(pred_score, cfg, output_dir=None)
    elapsed = round(time.perf_counter() - t0, 1)
    m = result.metrics
    print(f"  完成 [{elapsed}s] sharpe={m.sharpe_ratio:.3f} "
          f"ann_ret={m.annualized_return * 100:.2f}% "
          f"mdd={m.max_drawdown * 100:.2f}% "
          f"turnover={m.avg_turnover * 100:.2f}%")
    return {
        "label": label,
        "elapsed_sec": elapsed,
        "sharpe": m.sharpe_ratio,
        "annualized_return": m.annualized_return,
        "annualized_volatility": m.annualized_volatility,
        "max_drawdown": m.max_drawdown,
        "calmar": m.calmar_ratio,
        "information_ratio": m.information_ratio,
        "annualized_excess_return": m.annualized_excess_return,
        "avg_turnover": m.avg_turnover,
        "win_rate": m.win_rate,
        **{f"cost_{k}": v for k, v in cost_override.items()},
    }


def init_qlib(cfg: Dict[str, Any]) -> None:
    import qlib
    qlib_init_cfg = cfg.get("qlib_init", {})
    qlib.init(
        provider_uri=qlib_init_cfg.get("provider_uri", "D:/qlib_data/qlib_data"),
        region=qlib_init_cfg.get("region", "cn"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="W1 撮合修正 sanity 对照：同 signals.parquet × 新旧两套 cost 对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run-id", default=None,
        help="完整 run_id（对应 artifacts/<run_id>/）。不传则用 --auto。",
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="自动选最近一次 lgbm_optuna_daily_retrain_* 的 run",
    )
    parser.add_argument("--model", default="lgbm_optuna", help="基础模型名（用于配置加载）")
    parser.add_argument("--freq", default="daily_retrain", help="频率名")
    parser.add_argument(
        "--auto-pattern", default="lgbm_optuna_daily_retrain_",
        help="--auto 时匹配 run 目录的前缀",
    )
    parser.add_argument(
        "--allow-unhealthy", action="store_true",
        help="允许 --auto 选择不健康 run（默认跳过零下单 / win_rate=0 / IC 全负的 run）",
    )
    args = parser.parse_args()

    # 1. 定位 run_dir
    if args.run_id:
        run_dir = _REPO_ROOT / "artifacts" / args.run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"run 目录不存在: {run_dir}")
        healthy, reason, _ = _run_health(run_dir)
        if not healthy:
            print(f"[warn] 显式指定的 run 不健康: {reason}")
    elif args.auto:
        run_dir = auto_pick_run(args.auto_pattern, allow_unhealthy=args.allow_unhealthy)
    else:
        raise SystemExit("必须指定 --run-id 或 --auto")

    # 2. 加载 signals + base cfg（对齐区间）
    pred_score, sig_path = load_signals(run_dir)
    print(f"[signals] 来源: {sig_path.relative_to(_REPO_ROOT)}")
    print(f"[signals] 加载 {len(pred_score):,} 条预测 "
          f"({pred_score.index.get_level_values('datetime').min().date()} "
          f"~ {pred_score.index.get_level_values('datetime').max().date()})  "
          f"覆盖 {pred_score.index.get_level_values('datetime').nunique()} 个交易日")

    base_cfg = load_base_cfg(args.model, args.freq)
    base_cfg = align_cfg_to_signals(base_cfg, pred_score)
    init_qlib(base_cfg)

    # 3. 跑两次回测
    metrics_old = run_backtest_with_cost(
        pred_score, base_cfg, _OLD_COST, "OLD (pre-W1)"
    )
    metrics_new = run_backtest_with_cost(
        pred_score, base_cfg, _NEW_COST, "NEW (W1)"
    )

    # 4. 对比表
    print("\n" + "=" * 78)
    print(f"  W1 撮合修正对照 - run_id: {run_dir.name}")
    print("=" * 78)
    cols = [
        ("指标", "old", "new", "delta"),
        ("Sharpe",
         metrics_old["sharpe"], metrics_new["sharpe"],
         metrics_new["sharpe"] - metrics_old["sharpe"]),
        ("年化收益(%)",
         metrics_old["annualized_return"] * 100,
         metrics_new["annualized_return"] * 100,
         (metrics_new["annualized_return"] - metrics_old["annualized_return"]) * 100),
        ("年化波动(%)",
         metrics_old["annualized_volatility"] * 100,
         metrics_new["annualized_volatility"] * 100,
         (metrics_new["annualized_volatility"] - metrics_old["annualized_volatility"]) * 100),
        ("最大回撤(%)",
         metrics_old["max_drawdown"] * 100,
         metrics_new["max_drawdown"] * 100,
         (metrics_new["max_drawdown"] - metrics_old["max_drawdown"]) * 100),
        ("Calmar",
         metrics_old["calmar"], metrics_new["calmar"],
         metrics_new["calmar"] - metrics_old["calmar"]),
        ("信息比率",
         metrics_old["information_ratio"], metrics_new["information_ratio"],
         metrics_new["information_ratio"] - metrics_old["information_ratio"]),
        ("年化超额(%)",
         metrics_old["annualized_excess_return"] * 100,
         metrics_new["annualized_excess_return"] * 100,
         (metrics_new["annualized_excess_return"] - metrics_old["annualized_excess_return"]) * 100),
        ("平均换手(%)",
         metrics_old["avg_turnover"] * 100,
         metrics_new["avg_turnover"] * 100,
         (metrics_new["avg_turnover"] - metrics_old["avg_turnover"]) * 100),
    ]
    header = cols[0]
    print(f"{header[0]:<14} {header[1]:>14} {header[2]:>14} {header[3]:>14}")
    print("-" * 78)
    for row in cols[1:]:
        name, old, new, delta = row
        delta_sign = "+" if delta >= 0 else ""
        print(f"{name:<14} {old:>14.3f} {new:>14.3f} {delta_sign}{delta:>13.3f}")
    print("=" * 78)

    # 5. 解读
    sharpe_delta = metrics_new["sharpe"] - metrics_old["sharpe"]
    turn = metrics_new["avg_turnover"]
    expected_drop = turn * 2 * 0.0020  # 双边 +20bp（10bp×2）× 年化换手
    print("\n解读：")
    print(f"  - 预期 Sharpe 下降 ≈ {expected_drop * 100:.2f}pp/年（双边+20bp × 换手 {turn*100:.1f}%）")
    print(f"  - 实际 Sharpe 下降 = {-sharpe_delta:.3f}")
    if abs(sharpe_delta) < 0.30:
        print(f"  ✓ 符合预期：撮合修正不破坏 alpha 主体")
    else:
        print(f"  ⚠ 偏离预期：差异 {sharpe_delta:.3f} > 0.30，请检查"
              " deal_price/字段缺失/换手率突变 等异常")

    # 6. 持久化
    audit_dir = _REPO_ROOT / "artifacts" / "_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = audit_dir / f"w1_cost_sanity_{ts}.json"
    summary = {
        "ts": ts,
        "run_id": run_dir.name,
        "n_predictions": len(pred_score),
        "metrics_old": metrics_old,
        "metrics_new": metrics_new,
        "delta": {
            "sharpe": sharpe_delta,
            "annualized_return": metrics_new["annualized_return"] - metrics_old["annualized_return"],
            "max_drawdown": metrics_new["max_drawdown"] - metrics_old["max_drawdown"],
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n摘要已写入：{out_path.relative_to(_REPO_ROOT)}")


if __name__ == "__main__":
    main()
