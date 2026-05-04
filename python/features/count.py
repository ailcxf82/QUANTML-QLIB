"""涨跌天数统计类因子（group: technical）。

层级位置: Data -> [Feature] -> Model -> Signal -> Portfolio -> Backtest -> Evaluation
来源: Qlib Alpha158 的 CNTP / CNTN / CNTD 系列
    https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py

含义:
  - CNTP%d: 过去 d 日中"上涨天数"占比（值域 [0,1]）
  - CNTN%d: 过去 d 日中"下跌天数"占比
  - CNTD%d: CNTP%d - CNTN%d，反映"上涨胜率"
        正值=多头主导，负值=空头主导

与 STD/MA 等连续型动量指标的区别:
  - 连续型动量受单日大涨/大跌严重影响（一日 +10% 可能盖过其它 4 日的 +1%）；
  - CNTD 是"次数"的尺度，对极端单日鲁棒，与"持续性动量"高度对齐。
"""
from .base import AlphaDef

_WINDOWS: tuple[int, ...] = (5, 10, 20)

# audit_p0_diagnosis.py 实测：CNTN10 / CNTN20 在 train 段 |ICIR| 较强但符号
# 与 valid/test 反向（train -0.526/-1.071 vs valid +2.777/+1.554 vs test
# +0.982/+0.638），属于"过拟合候选 + 风格切换敏感"，预测稳定性差，禁用以
# 避免被 GBDT 在 train 上过度学习后污染头部排名。CNTN5、CNTP*、CNTD* 不在
# 此列表中（符号一致或过弱不致影响），保留启用。
_DISABLED_NAMES: frozenset[str] = frozenset({"CNTN10", "CNTN20"})

ALPHAS: list[AlphaDef] = []

for _w in _WINDOWS:
    ALPHAS.append(
        AlphaDef(
            name=f"CNTP{_w}",
            expr=f"Mean($close_qfq>Ref($close_qfq,1),{_w})",
            group="technical",
            description=f"{_w}日内上涨天数占比，>0.5 表示多头主导",
        )
    )
    ALPHAS.append(
        AlphaDef(
            name=f"CNTN{_w}",
            expr=f"Mean($close_qfq<Ref($close_qfq,1),{_w})",
            group="technical",
            description=f"{_w}日内下跌天数占比，>0.5 表示空头主导",
            enabled=(f"CNTN{_w}" not in _DISABLED_NAMES),
        )
    )
    ALPHAS.append(
        AlphaDef(
            name=f"CNTD{_w}",
            expr=(
                f"Mean($close_qfq>Ref($close_qfq,1),{_w})"
                f"-Mean($close_qfq<Ref($close_qfq,1),{_w})"
            ),
            group="technical",
            description=f"{_w}日上涨天数与下跌天数差占比，反映持续性多空力量对比",
        )
    )
