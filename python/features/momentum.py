"""
动量类因子（group: momentum）。

包含价格动量、短中期收益率等反映趋势延续或反转的因子。
所有表达式基于复权收盘价 $close_qfq。
"""
from .base import AlphaDef

ALPHAS: list[AlphaDef] = [
    AlphaDef(
        name="RET1",
        expr="$close_qfq/Ref($close_qfq,1)-1",
        group="momentum",
        description="当日收益率（T日相对T-1日涨跌幅），捕捉最近一日动量",
    ),
    AlphaDef(
        name="RET2",
        expr="Ref($close_qfq,1)/Ref($close_qfq,2)-1",
        group="momentum",
        description="T-1日收益率（昨日涨跌幅），用于动量延续或反转判断",
    ),
    AlphaDef(
        name="RET5",
        expr="Ref($close_qfq,2)/Ref($close_qfq,5)-1",
        group="momentum",
        description="T-2至T-5期间收益率，捕捉5日内中短期动量信号",
    ),
]
