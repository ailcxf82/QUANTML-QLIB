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

# ── 内置因子注册表（顺序即 QlibDataLoader 中的列顺序）────────────────
# 按业务组别聚合：momentum → volume → technical → fundamental，
# 每组内先现有原子因子、后新增的多窗口因子。
_BUILTIN_ALPHAS: list[AlphaDef] = (
    momentum.ALPHAS
    + momentum_ext.ALPHAS
    + volume.ALPHAS
    + volatility.ALPHAS
    + technical.ALPHAS
    + ma_window.ALPHAS
    + corr.ALPHAS
    + regression.ALPHAS
    + count.ALPHAS
    + fundamental.ALPHAS
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
