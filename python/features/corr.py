"""价量相关性类因子（group: technical）。

层级位置: Data -> [Feature] -> Model -> Signal -> Portfolio -> Backtest -> Evaluation
来源: Qlib Alpha158 的 CORR / CORD 系列
    https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py

为何重要:
  - Qlib 官方 README 列出的 "selected 20 features"（由 LGBM feature importance 筛出）
    中价量相关性占 7 个：CORR5, CORR10, CORR20, CORR60, CORD5, CORD10, CORD60
  - GBDT 直接受益于这类"协同/背离"信号

字段适配:
  - 价格使用复权字段 $close_qfq（与本工程其它模块一致）
  - 成交量使用 $volume（无复权后缀，原始量；Alpha158 同口径）
"""
from .base import AlphaDef

_WINDOWS_CORR: tuple[int, ...] = (5, 10, 20, 60)
_WINDOWS_CORD: tuple[int, ...] = (5, 10, 60)

ALPHAS: list[AlphaDef] = []

for _w in _WINDOWS_CORR:
    ALPHAS.append(
        AlphaDef(
            name=f"CORR{_w}",
            expr=f"Corr($close_qfq,Log($volume+1),{_w})",
            group="technical",
            description=f"{_w}日收盘价与对数成交量相关性，正值=价升量升（趋势型），负值=价升量缩（背离）",
        )
    )

for _w in _WINDOWS_CORD:
    ALPHAS.append(
        AlphaDef(
            name=f"CORD{_w}",
            expr=(
                f"Corr($close_qfq/Ref($close_qfq,1),"
                f"Log($volume/Ref($volume,1)+1),{_w})"
            ),
            group="technical",
            description=f"{_w}日收益率与量变率相关性，反映短期资金跟风/反向行为",
        )
    )
