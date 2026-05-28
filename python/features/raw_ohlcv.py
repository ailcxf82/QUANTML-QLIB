"""
原始 OHLCV 因子组（专供时序模型 dataset_type=ts 使用）。

设计动机（docs/时序模型优化.md §8.5）：
  LGBM 主力吃浓缩因子（momentum/volume/technical/fundamental），擅长高阶交互；
  时序模型吃原始序列，擅长路径依赖与状态转移。两者输入分离 → ensemble 才有信息增量。

参考：Qlib Alpha360 dataset（60 step × 6 OHLCV 原始序列），benchmark 中
  GRU IC=0.0493 > LightGBM 0.0400（CSI300，20 seed 平均）。

尺度归一：
  价格：除以 Ref($close_qfq, 1) → 接近 1 的 ratio，跨票同分布
  成交量：除以 Mean($volume, 20) → 相对均量，去单位、去市值

启用方式：
  在 model_meta.feature_groups 中显式写 ["raw_ohlcv"]，
  此组不会与其他 group 同时被默认拉取。
"""
from .base import AlphaDef

ALPHAS: list[AlphaDef] = [
    AlphaDef(
        name="OPEN_R",
        expr="$open_qfq/Ref($close_qfq,1)",
        group="raw_ohlcv",
        description="标准化开盘价（相对前收），≈1+跳空幅度",
    ),
    AlphaDef(
        name="HIGH_R",
        expr="$high_qfq/Ref($close_qfq,1)",
        group="raw_ohlcv",
        description="标准化最高价（相对前收）",
    ),
    AlphaDef(
        name="LOW_R",
        expr="$low_qfq/Ref($close_qfq,1)",
        group="raw_ohlcv",
        description="标准化最低价（相对前收）",
    ),
    AlphaDef(
        name="CLOSE_R",
        expr="$close_qfq/Ref($close_qfq,1)",
        group="raw_ohlcv",
        description="标准化收盘价（相对前收），≈1+日收益",
    ),
    AlphaDef(
        name="VOL_R",
        expr="$volume/(Mean($volume,20)+1e-12)",
        group="raw_ohlcv",
        description="标准化成交量（相对 20 日均量），去单位/去市值",
    ),
]
