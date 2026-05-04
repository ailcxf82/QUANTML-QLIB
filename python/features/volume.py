"""
量能类因子（group: volume）。

包含成交量变化、相对成交量、换手率、资金净流入等反映市场活跃度与资金偏好的因子。
"""
from .base import AlphaDef

ALPHAS: list[AlphaDef] = [
    AlphaDef(
        name="VOL_CHG",
        expr="($volume-Ref($volume,1))/(Ref($volume,1)+1e-12)",
        group="volume",
        description="成交量环比变化率，正值表示放量，负值表示缩量",
    ),
    AlphaDef(
        name="VOL5",
        expr="$volume/Mean($volume,5)",
        group="volume",
        description="当日成交量相对5日均量的比值，大于1表示近期放量",
    ),
    AlphaDef(
        name="VOL20",
        expr="$volume/Mean($volume,20)",
        group="volume",
        description="当日成交量相对20日均量的比值，反映中期量能状态",
    ),
    AlphaDef(
        name="TURN",
        expr="$turnover_rate",
        group="volume",
        description="换手率，反映当日流通股的交易活跃程度",
    ),
    AlphaDef(
        name="NET_AMT",
        expr="$net_amount",
        group="volume",
        description="资金净流入金额，正值为净买入，负值为净卖出",
        # 关于覆盖率（audit_features.py 实测）：
        #   train(2020-01 ~ 2024-06) 段 coverage 仅 18.9%（A 股早期资金流字段缺失），
        #   valid(2024-07~12) / test(2025-01~) 段 coverage 95-99%。
        # 复现 4d43003f 实验：禁用此字段会让年化收益从 87.7% 跌到 65.2%（-22.5pp）。
        # 解释：train 段 Fillna(0) 让 LGBM 把缺失视为"中性截面排名"，但 valid/test
        # 段含真实信号；禁用反而损失了 TopK 头部排序的可用信息。保持启用。
        enabled=True,
    ),
]
