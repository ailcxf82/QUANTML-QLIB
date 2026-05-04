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
        # 禁用原因（audit_features.py 实测）：
        #   train(2020-01 ~ 2024-06) 段 coverage 仅 18.9%，A 股早期资金流字段
        #   缺失严重；valid/test 段已 95-99%。一刀切 Fillna(0) 会把"缺失"和
        #   "净流入为 0"混淆，反而干扰其他股票的截面排名。重新启用前需补完整
        #   历史数据或换为按日 Rank 的鲁棒变换。
        enabled=False,
    ),
]
