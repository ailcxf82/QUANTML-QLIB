"""
与 configs/combined_factors_df.json 中 factors[].name 对齐的 Qlib 表达式。

说明:
  - JSON 仅包含筛选后的因子名与 IC 等元数据，不含表达式；表达式在此维护并与名称一一对应。
  - 仅当 model_meta.combined_factors_json 指定时由 run_experiment 追加到特征列表，不影响其他模型。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# JSON name -> Qlib 表达式（与 refresh_rdagent / 因子研究口径语义对齐的可实现版本）
_EXPR_BY_JSON_NAME: dict[str, str] = {
    # PB 相对 20 日均值的低估程度（价值均值回归，PB 低于近期均值时为正）
    "ValueMR_20D": "(Mean($pb,20)-$pb)/(Std($pb,20)+1e-12)",
    "volume_zscore_20d": "($volume-Mean($volume,20))/(Std($volume,20)+1e-12)",
    "turnover_acceleration_5d": (
        "$turnover_rate/(Mean($turnover_rate,5)+1e-12)"
        "-Ref($turnover_rate/(Mean($turnover_rate,5)+1e-12),1)"
    ),
    "volume_ratio_ma_15d": "$volume/(Mean($volume,15)+1e-12)",
    "roe_momentum_60d": "$roe/(Ref($roe,60)+1e-12)-1",
    # A 股日线常用财报字段：基本每股收益（季度披露，qlib 已向前填充）
    "quarterly_eps": "$eps",
}

# 已知在当前 qlib provider 上覆盖率严重不足、必须在合并阶段直接跳过的因子名。
#   - quarterly_eps: provider 中 $eps 字段全 NaN（audit_features.py 实测 train/valid/test 三段
#     coverage=0.000）。重新启用前需补完整 EPS 历史数据，并通过 IC 筛选脚本重新生成 JSON。
_DISABLED_NAMES: frozenset[str] = frozenset({
    "quarterly_eps",
})


def merge_features_from_combined_json(
    exprs: list[str],
    names: list[str],
    json_path: Path,
) -> tuple[list[str], list[str]]:
    """
    读取 combined_factors_df.json，按其中 factors[].name 顺序追加尚未存在的特征列。

    Args:
        exprs: 已有表达式列表（会被原地追加）
        names: 已有列名列表（会被原地追加）
        json_path: JSON 文件路径（通常为 configs/combined_factors_df.json）

    Returns:
        (exprs, names) 同一对象引用，便于调用方链式使用。

    Raises:
        FileNotFoundError: json_path 不存在
        ValueError: JSON 结构非法或解析后无有效因子名
    """
    if not json_path.is_file():
        raise FileNotFoundError(f"combined_factors_json 路径不存在: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        payload: dict[str, Any] = json.load(f)

    factor_entries = payload.get("factors")
    if not isinstance(factor_entries, list):
        raise ValueError("combined_factors JSON 缺少有效的 'factors' 列表")

    existing = set(names)
    for entry in factor_entries:
        if not isinstance(entry, dict):
            continue
        raw_name = entry.get("name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            continue
        name = raw_name.strip()
        if name in existing:
            continue
        if name in _DISABLED_NAMES:
            # 已知数据覆盖率极低/字段缺失的因子静默跳过，不写入特征列表。
            continue
        expr = _EXPR_BY_JSON_NAME.get(name)
        if expr is None:
            raise ValueError(
                f"combined_factors JSON 中的因子名 '{name}' 尚无内置 Qlib 表达式映射，"
                f"请在 python/features/combined_json_factors.py 的 _EXPR_BY_JSON_NAME 中补充。"
            )
        exprs.append(expr)
        names.append(name)
        existing.add(name)

    return exprs, names
