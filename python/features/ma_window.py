"""标准 Alpha158 风格的均线 / 波动率多窗口因子（group: technical）。

层级位置: Data -> [Feature] -> Model -> Signal -> Portfolio -> Backtest -> Evaluation
来源: Qlib Alpha158 的 MA / STD / ROC 系列（用于补全 5/10/20/30/60 五档窗口）
    https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py

与现有 technical.py 的差异:
  - 现有 STD5/STD20 是"收益率标准差"（Std($close/Ref($close,1)-1, d)）
  - 此处 STD%d 是 Alpha158 标准"价格标准差/价格"（Std($close,d)/$close）
  - 两者都是无量纲、可截面比较的，但口径不同；同时纳入可让 GBDT 自由选择
  - 名称用 STDP%d 区分（"P"= price-based）

  - 现有 MA5_20 / MA10_60 是"均线之比 - 1"
  - 此处 MA%d 是 Alpha158 标准"d 日均线/当前价"（< 1 → 当前价高于均线 → 多头）
  - 名称用 MA 字母不冲突，加窗口后缀
"""
from .base import AlphaDef

_WINDOWS: tuple[int, ...] = (5, 10, 20, 30, 60)

ALPHAS: list[AlphaDef] = []

for _w in _WINDOWS:
    ALPHAS.append(
        AlphaDef(
            name=f"MA{_w}",
            expr=f"Mean($close_qfq,{_w})/$close_qfq",
            group="technical",
            description=f"{_w}日均线/当前价，<1 表示当前价高于均线（多头）",
        )
    )
    ALPHAS.append(
        AlphaDef(
            name=f"STDP{_w}",
            expr=f"Std($close_qfq,{_w})/$close_qfq",
            group="technical",
            description=f"{_w}日收盘价标准差/当前价（Alpha158 标准），相对波动率",
        )
    )
