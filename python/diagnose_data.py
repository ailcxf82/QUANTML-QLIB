"""数据质量诊断脚本，检查实际字段值、NaN 分布、归一化后状态。"""
import numpy as np
import pandas as pd
import qlib
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.utils import init_instance_by_config


def main() -> None:
    qlib.init(provider_uri="D:/qlib_data/qlib_data", region="cn")

    # 1. 检查单只股票原始字段
    fields = [
        "$close_qfq", "$open_qfq", "$high_qfq", "$low_qfq",
        "$volume", "$turnover_rate", "$rsi_qfq_12",
        "$pe_ttm", "$pb", "$total_mv",
    ]
    df_raw = D.features(
        ["000001.SZ"],
        fields,
        start_time="2023-01-01",
        end_time="2023-12-31",
        freq="day",
    )
    df_raw.columns = [f.lstrip("$") for f in fields]
    df_raw.index = df_raw.index.droplevel(1)

    print("=" * 60)
    print("单股 000001.SZ 原始字段 (2023):")
    print("Shape:", df_raw.shape)
    print("NaN counts:\n", df_raw.isna().sum())
    print("\n描述性统计:\n", df_raw.describe().to_string())

    # 2. 检查衍生特征（含 Ref/Mean/Std）
    derived_fields = [
        "$close_qfq/Ref($close_qfq,1)-1",
        "($high_qfq-$low_qfq)/$open_qfq",
        "$volume/Mean($volume,5)",
        "Std($close_qfq/Ref($close_qfq,1)-1,20)",
        "Mean($close_qfq,5)/Mean($close_qfq,20)-1",
        "Ref($close_qfq,-2)/Ref($close_qfq,-1)-1",
    ]
    df_deriv = D.features(
        ["000001.SZ"],
        derived_fields,
        start_time="2022-01-01",
        end_time="2023-06-30",
        freq="day",
    )
    df_deriv.columns = ["RET1", "KLEN", "VOL5", "STD20", "MA5_20", "LABEL0"]
    df_deriv.index = df_deriv.index.droplevel(1)

    print("\n" + "=" * 60)
    print("衍生特征 (2022-2023.6):")
    print("Shape:", df_deriv.shape)
    print("NaN counts:\n", df_deriv.isna().sum())
    nan_ratio = df_deriv.isna().mean()
    print("\nNaN 占比:\n", nan_ratio.to_string())
    print("\n非 NaN 描述:\n", df_deriv.describe().to_string())

    # 3. 检查 CSZScoreNorm 后是否有 NaN 或 Inf
    ret1 = df_deriv["RET1"].dropna()
    if len(ret1) > 0:
        z_fake = (ret1 - ret1.mean()) / (ret1.std() + 1e-9)
        print("\n手动 Z-score 后 NaN:", z_fake.isna().sum(), "Inf:", np.isinf(z_fake).sum())
    else:
        print("\nRET1 全为 NaN！")

    print("\n诊断完成。")


if __name__ == "__main__":
    main()
