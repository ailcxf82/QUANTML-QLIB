"""
基本面类因子（group: fundamental）。

包含估值、规模等截面基本面因子。
适合树模型（LightGBM）和平铺神经网络（MLP）；
时序模型（LSTM/GRU/Transformer）默认不使用此组，因为基本面因子时序变化慢，
在序列建模中对模型贡献有限，且会稀释时序信号密度。
"""
from .base import AlphaDef

ALPHAS: list[AlphaDef] = [
    AlphaDef(
        name="PB",
        expr="$pb",
        group="fundamental",
        description="市净率（Price-to-Book），低PB通常对应价值风格，截面比较有效",
    ),
    AlphaDef(
        name="MV",
        expr="$total_mv",
        group="fundamental",
        description="总市值（元），用于规模因子暴露控制与市值中性化",
    ),
]
