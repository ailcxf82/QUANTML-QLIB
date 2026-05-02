"""
技术指标类因子（group: technical）。

包含 K 线形态、波动率、均线系统、RSI/MACD/KDJ 等技术分析衍生因子。
所有含价格的表达式基于复权价格（_qfq 后缀字段）。
"""
from .base import AlphaDef

ALPHAS: list[AlphaDef] = [
    # ── K 线形态 ─────────────────────────────────────────────────────
    AlphaDef(
        name="KMID",
        expr="($close_qfq-$open_qfq)/$open_qfq",
        group="technical",
        description="K线实体方向与大小（收-开）/开，正值为阳线，负值为阴线",
    ),
    AlphaDef(
        name="KLEN",
        expr="($high_qfq-$low_qfq)/$open_qfq",
        group="technical",
        description="日内振幅（高-低）/开，反映当日价格波动范围",
    ),
    AlphaDef(
        name="KSFT",
        expr="(2*$close_qfq-$high_qfq-$low_qfq)/($high_qfq-$low_qfq+1e-12)",
        group="technical",
        description="收盘价在日内区间的相对位置（-1到1），正值靠近高点，负值靠近低点",
    ),
    # ── 波动率 ───────────────────────────────────────────────────────
    AlphaDef(
        name="STD5",
        expr="Std($close_qfq/Ref($close_qfq,1)-1,5)",
        group="technical",
        description="5日收益率标准差，反映近期价格波动风险",
    ),
    AlphaDef(
        name="STD20",
        expr="Std($close_qfq/Ref($close_qfq,1)-1,20)",
        group="technical",
        description="20日收益率标准差，反映中期价格波动风险",
    ),
    # ── 均线系统 ─────────────────────────────────────────────────────
    AlphaDef(
        name="MA5_20",
        expr="Mean($close_qfq,5)/Mean($close_qfq,20)-1",
        group="technical",
        description="5日均线相对20日均线偏离度，正值表示短期均线在上方（短期强势）",
    ),
    AlphaDef(
        name="MA10_60",
        expr="Mean($close_qfq,10)/Mean($close_qfq,60)-1",
        group="technical",
        description="10日均线相对60日均线偏离度，反映中期趋势方向",
    ),
    # ── 技术振荡器 ───────────────────────────────────────────────────
    AlphaDef(
        name="RSI12",
        expr="$rsi_qfq_12/100",
        group="technical",
        description="12日RSI归一化到[0,1]，大于0.7超买，小于0.3超卖",
    ),
    AlphaDef(
        name="MACD_DIF",
        expr="$macd_dif_qfq",
        group="technical",
        description="MACD快慢线差值（DIF），正值表示短期均线在上，动量向上",
    ),
    AlphaDef(
        name="KDJK",
        expr="$kdj_k_qfq/100",
        group="technical",
        description="KDJ指标K值归一化到[0,1]，反映短期超买超卖状态",
    ),
]
