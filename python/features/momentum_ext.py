"""扩展中长周期动量因子（group: momentum）。

层级位置: Data -> [Feature] -> Model -> Signal -> Portfolio -> Backtest -> Evaluation
来源: Qlib Alpha158 的 ROC 系列（5/10/20/30/60）补全
    https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py

设计:
  - 现有 momentum.py 只覆盖 RET1/RET2/RET5（≤ 5 日），缺中长期；
  - selected-20 中明确包含 ROC60，其它窗口虽不在 20 强但在 LGBM importance 中
    经常进入 Top 30；
  - 此处用"过去 d 日累计收益率"形式（$close_qfq/Ref($close_qfq,d)-1），
    与 Alpha158 的 ROC%d 等价但以 0 为中心，可读性更好。
"""
from .base import AlphaDef

_WINDOWS: tuple[int, ...] = (10, 20, 30, 60)

ALPHAS: list[AlphaDef] = [
    AlphaDef(
        name=f"RET{_w}",
        expr=f"$close_qfq/Ref($close_qfq,{_w})-1",
        group="momentum",
        description=f"过去{_w}日累计收益率，捕捉中长期动量/反转信号",
    )
    for _w in _WINDOWS
]
