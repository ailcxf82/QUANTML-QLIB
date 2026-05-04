"""
因子库入口模块。

层级位置: Data -> [Feature] -> Model -> Signal -> Portfolio -> Backtest -> Evaluation
职责:
  1. 聚合内置因子（momentum / volume / technical / fundamental）
  2. 自动发现 rdagent/ 目录下 RDAgent 生成的因子文件
  3. 提供 build_feature_config() 供 run_experiment.py 动态注入 QlibDataLoader 配置
  4. 提供 feature_count() 供 run_experiment.py 自动设置 input_dim / d_feat

调用方式:
    from features import build_feature_config, feature_count

    # 获取所有启用因子的 Qlib 配置格式
    exprs, names = build_feature_config()

    # 只获取适合时序模型的因子
    exprs, names = build_feature_config(groups=["momentum", "volume", "technical"])

    # 包含 RDAgent 生成的新因子
    exprs, names = build_feature_config(include_rdagent=True)
"""
from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from typing import Optional

from .base import AlphaDef, VALID_GROUPS
from . import (
    corr,
    count,
    fundamental,
    ma_window,
    momentum,
    momentum_ext,
    regression,
    technical,
    volatility,
    volume,
)

# ── 因子集开关 ───────────────────────────────────────────────────────
# D3 阶段以"D1 基线"作为生产线：仅启用原 26 列原子因子；
#   - audit_p0_diagnosis.py 实测：26 列方案在 TopK=8 / csi500 上 IR=+0.473，
#     78 列扩展集 IC/ICIR 略胜但 IR 跌至 +0.105，TopK 离散选股下不占优；
#   - 78 列代码完整保留作"信号种子"，未来做信号融合 / 大 TopK / 行业中性化
#     时一键启用即可。
# 启用方式: 把常量改为 True，或在调试时通过 monkey-patch 模块属性切换。
USE_EXTENDED_ALPHAS: bool = False

# ── 原 26 列基线（D1 / 4d43003f 配置）──────────────────────────────
# 顺序与 4d43003f 完全一致，确保特征列顺序可复现。
#
# ⚠ 注意 KSFT 表达式漂移修复：
#   - 4d43003f（commit 0599d82f）时期的 `KSFT` 表达式为
#       (2*$close_qfq-$high_qfq-$low_qfq)/($high_qfq-$low_qfq+1e-12)   # ← 相对振幅
#   - 后续重构对齐 Alpha158 时把 `KSFT` 改成 (2*close-high-low)/open（相对开盘价），
#     原表达式被命名为 `KSFT2`。
#   - 因此恢复 D1 基线时要选 `KSFT2`，而不是当前同名的 `KSFT`，否则因子语义会偏。
_TECHNICAL_BASELINE_NAMES: frozenset[str] = frozenset({
    "KMID", "KLEN", "KSFT2",
    "STD5", "STD20",
    "MA5_20", "MA10_60",
    "RSI12", "MACD_DIF", "KDJK",
})

_ORIGINAL_26: list[AlphaDef] = (
    momentum.ALPHAS                                # 3: RET1, RET2, RET5
    + volume.ALPHAS                                # 5: VOL_CHG, VOL5, VOL20, TURN, NET_AMT(disabled)
    + [a for a in technical.ALPHAS if a.name in _TECHNICAL_BASELINE_NAMES]  # 10
    + fundamental.ALPHAS                           # 2: PB, MV
)

# ── 扩展 57 列（P0 引入，D3 默认禁用）────────────────────────────────
_EXTENDED_57: list[AlphaDef] = (
    momentum_ext.ALPHAS                            # +4: RET10/20/30/60
    + volatility.ALPHAS                            # +8: WVMA + VSTD ×4
    + [a for a in technical.ALPHAS if a.name not in _TECHNICAL_BASELINE_NAMES]  # +9: K线/STD扩档
    + ma_window.ALPHAS                             # +10: MA + STDP ×5
    + corr.ALPHAS                                  # +7: CORR ×4 + CORD ×3
    + regression.ALPHAS                            # +12: BETA/RSQR/RESI ×4
    + count.ALPHAS                                 # +9: CNTP/CNTN/CNTD ×3
)

# ── 内置因子注册表（顺序即 QlibDataLoader 中的列顺序）────────────────
_BUILTIN_ALPHAS: list[AlphaDef] = (
    _ORIGINAL_26 + _EXTENDED_57 if USE_EXTENDED_ALPHAS else list(_ORIGINAL_26)
)


def _discover_rdagent_alphas() -> list[AlphaDef]:
    """
    自动扫描 python/features/rdagent/ 目录，加载所有 RDAgent 生成的因子文件。

    约定: 每个 .py 文件必须暴露 ALPHAS: list[AlphaDef] 变量。
    失败时打印警告并跳过，不中断主流程。
    """
    rdagent_pkg = Path(__file__).parent / "rdagent"
    discovered: list[AlphaDef] = []

    for module_info in pkgutil.iter_modules([str(rdagent_pkg)]):
        mod_name = f"features.rdagent.{module_info.name}"
        try:
            mod = importlib.import_module(mod_name)
            alphas: list[AlphaDef] = getattr(mod, "ALPHAS", [])
            discovered.extend(alphas)
        except Exception as exc:  # noqa: BLE001
            print(f"[features] 警告：加载 RDAgent 因子 '{mod_name}' 失败：{exc}")

    return discovered


def get_registry(
    groups: Optional[list[str]] = None,
    include_rdagent: bool = True,
) -> list[AlphaDef]:
    """
    返回满足过滤条件的 AlphaDef 列表（仅 enabled=True 的因子）。

    参数:
        groups         因子组过滤列表；None 表示全部组
        include_rdagent 是否包含 rdagent/ 目录中的因子

    异常:
        ValueError: groups 包含非法组名时抛出
    """
    if groups is not None:
        invalid = set(groups) - VALID_GROUPS
        if invalid:
            raise ValueError(
                f"无效的 feature_groups：{sorted(invalid)}，合法值：{sorted(VALID_GROUPS)}"
            )

    all_alphas = _BUILTIN_ALPHAS.copy()
    if include_rdagent:
        all_alphas += _discover_rdagent_alphas()

    active = [a for a in all_alphas if a.enabled]
    if groups is not None:
        active = [a for a in active if a.group in groups]

    return active


def build_feature_config(
    groups: Optional[list[str]] = None,
    include_rdagent: bool = True,
) -> tuple[list[str], list[str]]:
    """
    生成 QlibDataLoader 所需的 [[expr...], [name...]] 配置格式。

    参数:
        groups         因子组过滤；None 表示全部启用因子
        include_rdagent 是否包含 RDAgent 生成的因子

    返回:
        (exprs, names) 两个等长列表，可直接赋值给
        dataset.kwargs.handler.kwargs.data_loader.kwargs.config.feature

    异常:
        ValueError: 过滤后因子列表为空时抛出
    """
    alphas = get_registry(groups=groups, include_rdagent=include_rdagent)
    if not alphas:
        raise ValueError(
            f"过滤后因子列表为空（groups={groups}），请检查因子注册表或 enabled 状态。"
        )
    exprs = [a.expr for a in alphas]
    names = [a.name for a in alphas]
    return exprs, names


def feature_count(
    groups: Optional[list[str]] = None,
    include_rdagent: bool = True,
) -> int:
    """
    返回启用因子的数量，供模型自动设置 input_dim / d_feat。

    参数与 build_feature_config() 相同。
    """
    return len(get_registry(groups=groups, include_rdagent=include_rdagent))


def print_registry(
    groups: Optional[list[str]] = None,
    include_rdagent: bool = True,
) -> None:
    """打印当前因子注册表，便于调试与审计。"""
    alphas = get_registry(groups=groups, include_rdagent=include_rdagent)
    header = f"{'#':<4} {'Name':<20} {'Group':<14} {'Description'}"
    print(header)
    print("-" * len(header))
    for i, a in enumerate(alphas, 1):
        rdagent_tag = " [RDAgent]" if a not in _BUILTIN_ALPHAS else ""
        print(f"{i:<4} {a.name:<20} {a.group:<14} {a.description}{rdagent_tag}")
    print(f"\n共 {len(alphas)} 个因子")
