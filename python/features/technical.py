"""
技术指标类因子（group: technical）。

包含 K 线形态、波动率、均线系统、RSI/MACD/KDJ 等技术分析衍生因子。
所有含价格的表达式基于复权价格（_qfq 后缀字段）。

K 线 9 形态对齐 Qlib Alpha158 KBAR 全集：
    KMID, KLEN, KMID2, KUP, KUP2, KLOW, KLOW2, KSFT, KSFT2
源码: https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py
"""
from .base import AlphaDef

ALPHAS: list[AlphaDef] = [
    # ── K 线形态（9 形态对齐 Alpha158 KBAR 全集）────────────────────
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
        name="KMID2",
        expr="($close_qfq-$open_qfq)/($high_qfq-$low_qfq+1e-12)",
        group="technical",
        description="实体占振幅比例（K 实体强度），|值|越大越坚决",
    ),
    AlphaDef(
        name="KUP",
        expr="($high_qfq-Greater($open_qfq,$close_qfq))/$open_qfq",
        group="technical",
        description="上影线相对开盘价（高 - max(开,收) / 开），上影越长可能见顶承压",
    ),
    AlphaDef(
        name="KUP2",
        expr="($high_qfq-Greater($open_qfq,$close_qfq))/($high_qfq-$low_qfq+1e-12)",
        group="technical",
        description="上影线占当日振幅比例，长上影=承压信号",
        # audit_p0_diagnosis.py 实测：train -2.014, valid -4.131, test +0.648，
        # train/valid 强负但 test 正——典型"风格切换"过拟合候选；KUP（绝对值版）
        # 与 KUP2（占振幅比版）信号高度相关，保留 KUP 即可，禁用 KUP2 防止
        # GBDT 在 train 上过度依赖此列。
        enabled=False,
    ),
    AlphaDef(
        name="KLOW",
        expr="(Less($open_qfq,$close_qfq)-$low_qfq)/$open_qfq",
        group="technical",
        description="下影线相对开盘价（min(开,收) - 低 / 开），下影越长可能见底支撑",
    ),
    AlphaDef(
        name="KLOW2",
        expr="(Less($open_qfq,$close_qfq)-$low_qfq)/($high_qfq-$low_qfq+1e-12)",
        group="technical",
        description="下影线占当日振幅比例，长下影=支撑信号",
    ),
    AlphaDef(
        name="KSFT",
        expr="(2*$close_qfq-$high_qfq-$low_qfq)/$open_qfq",
        group="technical",
        description="收盘价偏离日内中位（2 收 - 高 - 低）/开，正值偏强、负值偏弱",
    ),
    AlphaDef(
        name="KSFT2",
        expr="(2*$close_qfq-$high_qfq-$low_qfq)/($high_qfq-$low_qfq+1e-12)",
        group="technical",
        description="收盘价在日内区间的相对位置（-1~1），正值靠近高点，负值靠近低点",
    ),
    # ── 收益率波动率（多窗口 5/10/20/30/60）─────────────────────────
    AlphaDef(
        name="STD5",
        expr="Std($close_qfq/Ref($close_qfq,1)-1,5)",
        group="technical",
        description="5日收益率标准差，反映近期价格波动风险",
    ),
    AlphaDef(
        name="STD10",
        expr="Std($close_qfq/Ref($close_qfq,1)-1,10)",
        group="technical",
        description="10日收益率标准差",
    ),
    AlphaDef(
        name="STD20",
        expr="Std($close_qfq/Ref($close_qfq,1)-1,20)",
        group="technical",
        description="20日收益率标准差，反映中期价格波动风险",
    ),
    AlphaDef(
        name="STD30",
        expr="Std($close_qfq/Ref($close_qfq,1)-1,30)",
        group="technical",
        description="30日收益率标准差",
    ),
    AlphaDef(
        name="STD60",
        expr="Std($close_qfq/Ref($close_qfq,1)-1,60)",
        group="technical",
        description="60日收益率标准差",
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
