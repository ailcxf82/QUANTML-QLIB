"""诊断 qlib backtest 返回值结构。"""
import yaml
import qlib
import pandas as pd
from pathlib import Path
from qlib.backtest import backtest
from qlib.utils import init_instance_by_config


def main() -> None:
    workspace_root = Path(__file__).resolve().parents[1]
    cfg_path = workspace_root / "configs" / "mlp_daily_backtest.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    qlib.init(provider_uri=cfg["qlib_init"]["provider_uri"], region=cfg["qlib_init"]["region"])

    # 用一个简单的等权策略快速跑一遍，不需要训练
    strategy_config = {
        "class": "TopkDropoutStrategy",
        "module_path": "qlib.contrib.strategy",
        "kwargs": {
            "topk": 10,
            "n_drop": 2,
            "signal": pd.Series(
                [0.0],
                index=pd.MultiIndex.from_tuples(
                    [("2025-01-02", "000001.SZ")], names=["datetime", "instrument"]
                ),
            ),
        },
    }

    port_metrics, indicator_metrics = backtest(
        start_time="2025-01-02",
        end_time="2025-01-10",
        strategy=strategy_config,
        executor=cfg["executor"],
        benchmark=cfg["experiment"]["benchmark"],
        account=cfg["experiment"]["account"],
        exchange_kwargs=cfg["backtest"]["exchange_kwargs"],
    )

    print("=== PORT_METRICS type:", type(port_metrics))
    print("=== PORT_METRICS keys:", list(port_metrics.keys()) if hasattr(port_metrics, "keys") else "N/A")
    for k, v in port_metrics.items():
        print(f"  key={k!r}, type={type(v)}, value={v!r}")

    print("\n=== INDICATOR_METRICS type:", type(indicator_metrics))
    if hasattr(indicator_metrics, "keys"):
        for k, v in indicator_metrics.items():
            print(f"  key={k!r}, type={type(v)}")


if __name__ == "__main__":
    main()
