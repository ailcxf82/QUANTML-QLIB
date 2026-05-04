"""量加权波动 / 量稳定性类因子（group: volume）。

层级位置: Data -> [Feature] -> Model -> Signal -> Portfolio -> Backtest -> Evaluation
来源: Qlib Alpha158 的 WVMA / VSTD 系列
    https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py
    https://github.com/vnpy/vnpy/blob/master/vnpy/alpha/dataset/datasets/alpha_158.py

含义:
  - WVMA%d:  Std(|ret|·vol, d) / Mean(|ret|·vol, d)
             —— 量加权"绝对收益"序列的变异系数；越大表示放量震荡越剧烈，
                通常对应做多情绪不稳定（selected-20 中包含 WVMA5/60）
  - VSTD%d:  Std($volume, d) / ($volume + 1e-12)
             —— 当日成交量相对最近 d 日量序列波动率的相对值；
                Alpha158 标准定义（除以当日量），不是除以 d 日均量

注: $volume 字段在 qlib provider 中是原始（未复权）成交量；与 $close_qfq 配合
   计算 |ret|·vol 没有口径冲突（收益率本就基于复权价计算）。
"""
from .base import AlphaDef

_WINDOWS: tuple[int, ...] = (5, 10, 20, 60)

ALPHAS: list[AlphaDef] = []

for _w in _WINDOWS:
    ALPHAS.append(
        AlphaDef(
            name=f"WVMA{_w}",
            expr=(
                f"Std(Abs($close_qfq/Ref($close_qfq,1)-1)*$volume,{_w})"
                f"/(Mean(Abs($close_qfq/Ref($close_qfq,1)-1)*$volume,{_w})+1e-12)"
            ),
            group="volume",
            description=f"{_w}日量加权波动变异系数（Std/Mean），刻画放量震荡强度",
        )
    )
    ALPHAS.append(
        AlphaDef(
            name=f"VSTD{_w}",
            expr=f"Std($volume,{_w})/($volume+1e-12)",
            group="volume",
            description=f"{_w}日成交量波动相对值（Alpha158 标准），反映量能稳定性",
        )
    )
