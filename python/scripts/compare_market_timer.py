"""
MarketTimer A/B 对比回测脚本
================================
复用已训练模型的 signals.parquet（无需重新训练），
用同一组预测信号分别运行 null / rolling / garch 三种 MarketTimer，
输出指标对比表。

使用方式：
    python python/scripts/compare_market_timer.py \
        --signals-dir artifacts/lgbm_daily_prod_retrain_b6f3a633

层级位置：脚本工具层（不属于常规训练/回测流水线）
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

# 确保 python/ 目录在 sys.path
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "python"))

import pandas as pd


def _load_signals(signals_dir: Path) -> pd.Series:
    """加载 signals.parquet，返回 pred_score Series。"""
    pq = signals_dir / "signals.parquet"
    if not pq.exists():
        raise FileNotFoundError(f"signals.parquet 不存在: {pq}")
    df = pd.read_parquet(pq)
    # signals 保存格式：MultiIndex(datetime, instrument) 或 columns 含 score
    if isinstance(df.index, pd.MultiIndex):
        if "score" in df.columns:
            return df["score"]
        return df.iloc[:, 0]
    if "score" in df.columns:
        return df["score"]
    return df.iloc[:, 0]


def _inject_timer(cfg: dict, timer_type: str) -> dict:
    """在 strategy.kwargs 中注入（或覆盖）market_timer_cfg。"""
    cfg = copy.deepcopy(cfg)
    strat_kwargs = cfg.get("strategy", {}).get("kwargs", {})
    strat_kwargs["market_timer_cfg"] = {
        "type": timer_type,
        "benchmark": "000905.SZ",  # A 股 Qlib 数据集中 CSI500 代码
        "target_vol": 0.15,
        "min_risk": 0.30,
        "max_risk": 1.0,
        "refit_freq": 5,
        "min_obs": 120,
        "hist_start": "2018-01-01",
    }
    cfg.setdefault("strategy", {})["kwargs"] = strat_kwargs
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="MarketTimer 对比回测")
    parser.add_argument(
        "--signals-dir",
        default="artifacts/lgbm_daily_prod_retrain_b6f3a633",
        help="包含 signals.parquet 和 model_meta.json 的 artifact 目录",
    )
    parser.add_argument(
        "--freq",
        default="daily_prod_retrain",
        help="频率名（对应 configs/freq/<freq>.yaml）",
    )
    parser.add_argument(
        "--model",
        default="lgbm",
        help="模型名（对应 configs/models/<model>.yaml）",
    )
    parser.add_argument(
        "--timers",
        nargs="+",
        default=["null", "rolling", "garch"],
        help="要对比的 MarketTimer 类型列表",
    )
    args = parser.parse_args()

    signals_dir = _REPO_ROOT / args.signals_dir

    # 加载预测信号
    print(f"[信号] 加载: {signals_dir / 'signals.parquet'}")
    pred_score = _load_signals(signals_dir)
    print(f"[信号] 形状: {pred_score.shape}，时间范围: "
          f"{pred_score.index.get_level_values(0).min()} ~ "
          f"{pred_score.index.get_level_values(0).max()}")

    # 加载基础配置（通过 run_experiment.py 的合并函数）
    from run_experiment import load_and_merge_configs  # type: ignore[import]
    base_cfg = load_and_merge_configs(
        model_name=args.model,
        freq_name=args.freq,
        full_config=None,
        strategy_cfg_path="configs/strategy/daily_vol_target.yaml",
    )

    # 初始化 Qlib
    import qlib
    qlib_cfg = base_cfg.get("qlib_init", {})
    provider_uri = qlib_cfg.get("provider_uri", "D:/qlib_data/qlib_data_train_20260516")
    region = qlib_cfg.get("region", "cn")
    print(f"[Qlib] 初始化 provider_uri={provider_uri}, region={region}")
    qlib.init(provider_uri=provider_uri, region=region)

    # 运行对比回测
    from backtest.engine import BacktestEngine  # type: ignore[import]
    engine = BacktestEngine()

    results: dict[str, dict] = {}
    for timer_type in args.timers:
        print(f"\n{'='*60}")
        print(f"[回测] MarketTimer type = {timer_type}")
        print(f"{'='*60}")
        cfg = _inject_timer(base_cfg, timer_type)
        # 生成带 timer 标识的 run_id
        base_run_id = base_cfg.get("experiment", {}).get("run_id", "compare")
        cfg.setdefault("experiment", {})["run_id"] = f"{base_run_id}__timer_{timer_type}"

        result = engine.run(pred_score, cfg, output_dir=None)
        m = result.metrics
        results[timer_type] = {
            "年化收益":   f"{m.annualized_return:.2%}",
            "年化波动":   f"{m.annualized_volatility:.2%}",
            "最大回撤":   f"{m.max_drawdown:.2%}",
            "Sharpe":     f"{m.sharpe_ratio:.3f}",
            "信息比率":   f"{m.information_ratio:.3f}",
            "年化超额":   f"{m.annualized_excess_return:.2%}",
            "平均换手":   f"{m.avg_turnover:.2%}",
        }

    # 打印对比表
    print(f"\n{'='*60}")
    print("MarketTimer 对比结果汇总")
    print(f"{'='*60}")
    metrics_order = ["年化收益", "年化波动", "最大回撤", "Sharpe", "信息比率", "年化超额", "平均换手"]
    col_w = 14
    header = f"{'指标':<12}" + "".join(f"{t:>{col_w}}" for t in args.timers)
    print(header)
    print("-" * (12 + col_w * len(args.timers)))
    for metric in metrics_order:
        row = f"{metric:<12}"
        for timer_type in args.timers:
            row += f"{results[timer_type][metric]:>{col_w}}"
        print(row)
    print()


if __name__ == "__main__":
    main()
