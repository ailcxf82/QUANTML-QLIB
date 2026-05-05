"""
策略子包（Strategy Sub-package）
================================
将原本"配置透传"的 StrategyLayer 升级为可插拔的五段式管线：

    pred_score
        ↓
    SignalProcessor      （预留：截面 zscore / EWMA 平滑）
        ↓
    MarketTimer          （预留：vol target / drawdown 熔断）
        ↓
    Selector             ✓ 第一阶段实现：截面 topk + 分位过滤
        ↓
    Weighter             ✓ 第一阶段实现：等权 / Score / InverseVol / RiskParity
        ↓
    RiskConstraints      ✓ 第一阶段实现：单票 + 板块 + 换手 三层约束
        ↓
    QuantMLWeightStrategy   ✓ 第一阶段实现：继承 WeightStrategyBase，承接 Qlib 撮合

设计原则：
  - 每个子模块单一职责，可单测、可替换
  - 默认参数对齐"中频机构化"目标：双周调仓 / 单票 ≤ 8% / 年换手 ≤ 300%
  - 完全兼容 Qlib backtest 撮合引擎（通过 generate_target_weight_position 接口）
  - 旧的 TopkDropoutStrategy 路径不动，由配置 strategy.class 切换
"""

from .selector import TopKSelector
from .weighter import (
    Weighter,
    EqualWeighter,
    ScoreWeighter,
    InverseVolWeighter,
    RiskParityWeighter,
    build_weighter,
)

# constraints / quantml_strategy 在后续 todo 中实现，按需 import
try:
    from .constraints import RiskConstraints
except ImportError:
    RiskConstraints = None  # type: ignore[assignment]

try:
    from .quantml_strategy import QuantMLWeightStrategy
except ImportError:
    QuantMLWeightStrategy = None  # type: ignore[assignment]

__all__ = [
    "TopKSelector",
    "Weighter",
    "EqualWeighter",
    "ScoreWeighter",
    "InverseVolWeighter",
    "RiskParityWeighter",
    "build_weighter",
    "RiskConstraints",
    "QuantMLWeightStrategy",
]
