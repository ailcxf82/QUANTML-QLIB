"""
日内 5min 原始序列因子组（B3 架构 B / Deep Factor，仅供日内时序模型）。

设计动机（docs/时序模型优化.md §8.5 架构 B）：
  承接 B1 负面结论（rank 融合拖累 Sharpe）。改让日内 5min 时序模型不直接输出分数，
  而是把 hidden state 当作 LGBM 的新特征（Deep Factor），由 LGBM 自学权重。
  本因子组定义单日 48 根 5min bar 的 6 维原始输入序列。

频率约束（重要）：
  这些表达式只在 5min 库（configs/freq/fivemin.yaml → provider_uri=qlib_data_5min）
  上有效，字段为 $open/$high/$low/$close/$volume/$amount（dump 时无 _qfq 后缀）。
  绝不能用于日频 LGBM/MLP 主链路 —— 故归入 _OPT_IN_GROUPS（默认排除）。

尺度归一（日内、与标的/价位无关，便于跨票同分布）：
  价格：除以 Ref($close,1)（前一根 5min bar 收盘）→ ≈1 的 bar 收益率
  量/额：除以单日均值（48 根）→ 相对强度，去单位/去市值
  跨日首根 bar 的 Ref 会引用前一日末根 bar，属轻微边界效应，可接受。

启用方式：
  在日内模型 model_meta.feature_groups 中显式写 ["intraday_raw"]。
"""
from .base import AlphaDef

# 单日 5min bar 数（A 股：上午 24 根 + 下午 24 根），供 step_len 对齐参考
BARS_PER_DAY = 48

ALPHAS: list[AlphaDef] = [
    AlphaDef(
        name="I_OPEN_R",
        expr="$open/Ref($close,1)",
        group="intraday_raw",
        description="5min 标准化开盘价（相对前一根收盘），≈1+跳变",
    ),
    AlphaDef(
        name="I_HIGH_R",
        expr="$high/Ref($close,1)",
        group="intraday_raw",
        description="5min 标准化最高价（相对前一根收盘）",
    ),
    AlphaDef(
        name="I_LOW_R",
        expr="$low/Ref($close,1)",
        group="intraday_raw",
        description="5min 标准化最低价（相对前一根收盘）",
    ),
    AlphaDef(
        name="I_CLOSE_R",
        expr="$close/Ref($close,1)",
        group="intraday_raw",
        description="5min 标准化收盘价（相对前一根收盘），≈1+bar 收益",
    ),
    AlphaDef(
        name="I_VOL_R",
        expr="$volume/(Mean($volume,48)+1e-12)",
        group="intraday_raw",
        description="5min 标准化成交量（相对单日均量 48 根），去单位/去市值",
    ),
    AlphaDef(
        name="I_AMT_R",
        expr="$amount/(Mean($amount,48)+1e-12)",
        group="intraday_raw",
        description="5min 标准化成交额（相对单日均额 48 根）",
    ),
]
