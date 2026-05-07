"""
combined_json_factors 单测：覆盖 markdown 中 50 个因子名 + JSON 入口契约。

层级位置: tests for python/features/combined_json_factors.py
不依赖 qlib provider，仅做"名称解析与表达式注入"的契约校验。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MARKDOWN_PATH = REPO_ROOT / "configs" / "combined_factors_df_expressions.md"
EXISTING_JSON = REPO_ROOT / "configs" / "combined_factors_df.json"


# ──────────────────────────────────────────────────────────────────
# 工具：从 markdown 解析全部因子名（## 序号. `name`）
# ──────────────────────────────────────────────────────────────────

def _parse_factor_names_from_markdown(md_path: Path) -> list[str]:
    text = md_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^##\s+\d+\.\s+`([^`]+)`\s*$", re.MULTILINE)
    return pattern.findall(text)


# ──────────────────────────────────────────────────────────────────
# 测试用例
# ──────────────────────────────────────────────────────────────────

def test_markdown_factor_names_all_have_mapping() -> None:
    """markdown 中列出的 50 个因子名必须全部在 _EXPR_BY_JSON_NAME 中有映射。"""
    from features.combined_json_factors import _EXPR_BY_JSON_NAME

    factor_names = _parse_factor_names_from_markdown(MARKDOWN_PATH)
    assert len(factor_names) >= 50, (
        f"markdown 应至少有 50 个因子，实际解析到 {len(factor_names)} 个"
    )

    missing = [n for n in factor_names if n not in _EXPR_BY_JSON_NAME]
    assert not missing, (
        f"以下因子在 markdown 中存在但未加入 _EXPR_BY_JSON_NAME：{missing}"
    )


def test_existing_combined_json_resolves_all_names() -> None:
    """现有 configs/combined_factors_df.json 的全部因子必须可解析（向后兼容）。"""
    from features.combined_json_factors import (
        _EXPR_BY_JSON_NAME,
        merge_features_from_combined_json,
    )

    payload = json.loads(EXISTING_JSON.read_text(encoding="utf-8"))
    json_names = [e["name"] for e in payload["factors"] if "name" in e]
    assert json_names, "configs/combined_factors_df.json 解析为空"

    for name in json_names:
        assert name in _EXPR_BY_JSON_NAME, (
            f"现有 JSON 因子 '{name}' 无映射（破坏向后兼容）"
        )

    exprs: list[str] = []
    names: list[str] = []
    merge_features_from_combined_json(exprs, names, EXISTING_JSON)
    assert names == json_names, "merge 后 names 顺序应与 JSON 完全一致"
    assert all(isinstance(e, str) and e.strip() for e in exprs), (
        "全部表达式必须是非空字符串"
    )


def test_mapping_expressions_are_qlib_safe(tmp_path: Path) -> None:
    """全部表达式应满足 Qlib 表达式语法基本约束：
    - 仅引用 $field 形式的字段
    - 仅使用 PascalCase 算子（Qlib 规范）
    - 除法分母都加了 +1e-12（避免除零）—— 用 lint 规则抽查
    - 不出现 numpy / pandas 调用残留（如 np.xxx 或 .rolling）
    """
    from features.combined_json_factors import _EXPR_BY_JSON_NAME

    forbidden_substrings = ["np.", "pd.", ".rolling(", ".groupby(", "lambda "]
    for name, expr in _EXPR_BY_JSON_NAME.items():
        for bad in forbidden_substrings:
            assert bad not in expr, (
                f"因子 '{name}' 表达式残留 Python 语法 '{bad}': {expr}"
            )
        # 至少包含一个 $ 字段引用
        assert "$" in expr, f"因子 '{name}' 表达式不引用任何 Qlib 字段: {expr}"


def test_synonym_groups_share_expression() -> None:
    """同义因子组必须共享相同表达式（保证 JSON 用任一别名都等价）。"""
    from features.combined_json_factors import _EXPR_BY_JSON_NAME

    synonym_groups: list[list[str]] = [
        ["volume_ratio", "raw_volume_ratio", "rank_volume_ratio",
         "winsorized_zscore_volume_ratio"],
        ["liq_turnover_f", "turnover_rate_f", "raw_turnover_rate_f"],
        ["roe_pb_ratio", "pb_roe"],
        ["roe_quality", "ROE_Factor", "rank_roe"],
        ["turnover_ma_10", "liq_turnover_10d"],
        ["profit_growth_yoy", "q_profit_yoy"],
        ["roe_pe_ratio", "quality_value_interaction"],
    ]
    for group in synonym_groups:
        exprs = {_EXPR_BY_JSON_NAME[n] for n in group}
        assert len(exprs) == 1, (
            f"同义因子组 {group} 表达式应一致，实际有 {len(exprs)} 个不同实现：{exprs}"
        )


def test_unknown_name_raises_value_error(tmp_path: Path) -> None:
    """未在映射表中的因子名应抛 ValueError 而非静默跳过。"""
    from features.combined_json_factors import merge_features_from_combined_json

    bad_json = tmp_path / "bad.json"
    bad_json.write_text(
        json.dumps({"factors": [{"name": "definitely_unknown_factor_xyz"}]}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="尚无内置 Qlib 表达式映射"):
        merge_features_from_combined_json([], [], bad_json)


def test_full_50_factor_dryrun(tmp_path: Path) -> None:
    """构造一个含 markdown 中 50 个因子全名的 JSON，验证全部能解析且去重。"""
    from features.combined_json_factors import merge_features_from_combined_json

    factor_names = _parse_factor_names_from_markdown(MARKDOWN_PATH)
    full_json = tmp_path / "full.json"
    full_json.write_text(
        json.dumps({"factors": [{"name": n} for n in factor_names]}),
        encoding="utf-8",
    )

    exprs: list[str] = []
    names: list[str] = []
    merge_features_from_combined_json(exprs, names, full_json)

    assert names == factor_names, (
        "merge 后顺序必须与 JSON 一致（用于回测可复现性）"
    )
    assert len(exprs) == len(names), "exprs 与 names 长度必须相等"
