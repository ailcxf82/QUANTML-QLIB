"""时间序列线性回归类因子（group: technical）。

层级位置: Data -> [Feature] -> Model -> Signal -> Portfolio -> Backtest -> Evaluation
来源: Qlib Alpha158 的 BETA / RSQR / RESI 系列
    https://github.com/microsoft/qlib/blob/main/qlib/contrib/data/loader.py

含义:
  - BETA%d: Slope($close,d)/$close —— 过去 d 日"时间-价格"线性回归斜率，
            做了 1/$close 无量纲化；正值=趋势上行
  - RSQR%d: Rsquare($close,d)        —— 上述线性回归的 R-square，越接近 1 趋势越线性、
                                        越接近 0 越震荡
  - RESI%d: Resi($close,d)/$close    —— 当前价相对线性趋势线的残差（无量纲），
                                        正值=超出趋势上方（短期超买信号）

为何重要:
  - selected-20 中包含 RSQR5/10/20/60、RESI5/10、BETA 系列在 LGBM feature importance
    中也常进 Top 30；它们刻画的是"趋势性 vs 震荡性"，与 momentum/MA 类信号互补。
"""
from .base import AlphaDef

_WINDOWS: tuple[int, ...] = (5, 10, 20, 60)

# audit_p0_diagnosis.py 实测：RSQR5 / RSQR60 在 train 与 test 段 ICIR 符号
# 反转（train +0.503/+1.109 vs test -2.422/-1.007），属于"风格切换敏感"
# 候选噪声因子，禁用以稳定 OOS 表现。RSQR10、RSQR20 与训练同向，保留启用。
_DISABLED_NAMES: frozenset[str] = frozenset({"RSQR5", "RSQR60"})

ALPHAS: list[AlphaDef] = []

for _w in _WINDOWS:
    ALPHAS.append(
        AlphaDef(
            name=f"BETA{_w}",
            expr=f"Slope($close_qfq,{_w})/$close_qfq",
            group="technical",
            description=f"{_w}日时间-价格线性回归斜率（无量纲），趋势强度与方向",
        )
    )
    ALPHAS.append(
        AlphaDef(
            name=f"RSQR{_w}",
            expr=f"Rsquare($close_qfq,{_w})",
            group="technical",
            description=f"{_w}日线性回归 R-square，趋势线性度（越大越纯粹趋势）",
            enabled=(f"RSQR{_w}" not in _DISABLED_NAMES),
        )
    )
    ALPHAS.append(
        AlphaDef(
            name=f"RESI{_w}",
            expr=f"Resi($close_qfq,{_w})/$close_qfq",
            group="technical",
            description=f"{_w}日线性回归残差（无量纲），正值=当前价偏离趋势线上方",
        )
    )
