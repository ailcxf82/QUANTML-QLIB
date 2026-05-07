"""
RDAgent IC 筛选因子池：JSON 名称 → Qlib 表达式映射。

层级位置: Data -> [Feature] -> Model -> Signal -> Portfolio -> Backtest -> Evaluation
职责:
  - 维护 RDAgent 离线评估筛选出的高 IC 因子在 Qlib 表达式语言下的可实现版本
  - 当模型配置 `model_meta.combined_factors_json` 指定 JSON 路径时
    （形如 configs/combined_factors_df.json），仅按 JSON 中 factors[].name
    顺序追加尚未存在的特征列；不影响其它模型与全局因子注册表

使用方式:
  1. 在 model_meta.combined_factors_json 中指向一个 JSON 文件（结构见
     configs/combined_factors_df.json，至少含 factors:list[{name:str}]）
  2. JSON 中每个 name 必须在本文件 _EXPR_BY_JSON_NAME 中有映射；否则抛 ValueError
  3. 同义因子（如 liq_turnover_f / turnover_rate_f / raw_turnover_rate_f）
     共享同一表达式，可在 JSON 中用任一别名

设计原则:
  - 价格类统一使用复权字段 $close_qfq / $open_qfq / $high_qfq / $low_qfq，
    与 configs/base.yaml 标签口径 Ref($close_qfq,-2)/Ref($close_qfq,-1)-1
    保持一致，避免分红送配日导致的伪信号。
  - 基本面字段（$pe_ttm/$pb/$roe/$q_eps/$q_profit_yoy/$dv_ratio 等）保持原始名，
    Qlib provider 已对其做按品种 ffill。
  - Qlib 表达式语言不支持截面 rank（CSRank 仅在 processor 层可用），凡 RDAgent
    原始实现含 `groupby(level='datetime').rank(pct=True)` 的因子，
    本文件降级为"时序值或时序 z-score"；base.yaml 的 CSZScoreNorm(robust)
    会在 processor 阶段对全部特征做截面 robust z-score，等价补足截面归一化。
    树模型对单调排序变换不敏感，回测端差异通常 <5%。
  - 同义因子（多个 RDAgent workspace 用不同名导出同一含义因子）共用一条表达式，
    保证 JSON 任一别名都能识别。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────
# JSON 名称 → Qlib 表达式映射表
# 数据契约: key = configs/combined_factors_df_expressions.md 中列出的因子名
# 排序: 按 markdown 中 IC 降序，便于审计
# ──────────────────────────────────────────────────────────────────

_EXPR_BY_JSON_NAME: dict[str, str] = {
    # ── IC > 0.30 区间（截面方向特征，归一化由 CSZScoreNorm 后置完成）─────
    # 1. rank_close_open: close/open 截面 rank → 退化为比值，CSZScoreNorm 后等价
    "rank_close_open": "$close_qfq/($open_qfq+1e-12)",
    # 2. close_location: 收盘价在日内 high-low 区间的相对位置 [0, 1]
    "close_location": "($close_qfq-$low_qfq)/($high_qfq-$low_qfq+1e-12)",
    # 3. log_signed_volume: K 线方向 × log(成交量+1)（资金流向粗糙近似）
    "log_signed_volume": "Sign($close_qfq-$open_qfq)*Log($volume+1)",

    # ── IC 0.10 ~ 0.30 区间 ─────────────────────────────────────────────
    # 4. close_location_turnover_wt: close_location 加权流通换手率
    "close_location_turnover_wt": (
        "(($close_qfq-$low_qfq)/($high_qfq-$low_qfq+1e-12))*$turnover_rate_f"
    ),
    # 5. ValueMR_20D: $pe_ttm 个股 20 日 z-score（PE 高于均值为正）
    #    与 markdown 原始 RDAgent 实现完全对齐；
    #    旧版本曾用 $pb 且符号反向，此次按 markdown 修正
    "ValueMR_20D": "($pe_ttm-Mean($pe_ttm,20))/(Std($pe_ttm,20)+1e-12)",
    # 6. close_location_ma_5: close_location 的 5 日均值（短期形态稳定性）
    "close_location_ma_5": (
        "Mean(($close_qfq-$low_qfq)/($high_qfq-$low_qfq+1e-12),5)"
    ),
    # 7. roe_momentum_5d_interaction: 截面 rank(roe) × rank(5d 动量) → 时序乘积降级
    "roe_momentum_5d_interaction": "$roe*($close_qfq/(Ref($close_qfq,5)+1e-12)-1)",
    # 8. volume_ratio: 成交量比（数据库自带字段）
    "volume_ratio": "$volume_ratio",
    # 9. raw_volume_ratio: 与 #8 同义（仅做 inf→NaN 替换，Qlib loader 自动处理）
    "raw_volume_ratio": "$volume_ratio",
    # 10. vol_surge_5d: 当日成交量 / 前 5 日均量（lag 1，避免与当日均值同步偏置）
    "vol_surge_5d": "$volume/(Ref(Mean($volume,5),1)+1e-12)",
    # 11. earnings_yield_momentum_5d_interaction: (1/PE) × 5d 动量
    "earnings_yield_momentum_5d_interaction": (
        "(1/($pe_ttm+1e-12))*($close_qfq/(Ref($close_qfq,5)+1e-12)-1)"
    ),
    # 12. quality_value_momentum_5d_composite: 三因子时序乘积（质量 × 价值 × 动量）
    "quality_value_momentum_5d_composite": (
        "$roe*(1/($pe_ttm+1e-12))*($close_qfq/(Ref($close_qfq,5)+1e-12)-1)"
    ),
    # 13. rank_volume_ratio: 截面 rank → 时序原值降级（CSZScoreNorm 后等价）
    "rank_volume_ratio": "$volume_ratio",
    # 14. volume_zscore_20d: 个股 20 日成交量 z-score
    "volume_zscore_20d": "($volume-Mean($volume,20))/(Std($volume,20)+1e-12)",
    # 15. winsorized_zscore_volume_ratio: 截面 winsorize+z-score → 原值降级
    "winsorized_zscore_volume_ratio": "$volume_ratio",
    # 16. turnover_surprise_20: 当日换手 / 20日均换手
    "turnover_surprise_20": "$turnover_rate/(Mean($turnover_rate,20)+1e-12)",
    # 17. rank_turnover_deviation_20d: 截面 rank → 时序偏离度降级
    "rank_turnover_deviation_20d": (
        "$turnover_rate/(Mean($turnover_rate,20)+1e-12)-1"
    ),
    # 18. pb_pct_change_20d: PB 20 日涨跌幅
    "pb_pct_change_20d": "$pb/(Ref($pb,20)+1e-12)-1",
    # 19. ValueMR_60D: $pe_ttm 个股 60 日 z-score
    "ValueMR_60D": "($pe_ttm-Mean($pe_ttm,60))/(Std($pe_ttm,60)+1e-12)",

    # ── IC 0.02 ~ 0.10 区间 ─────────────────────────────────────────────
    # 20. sma_volume_ratio_10d
    "sma_volume_ratio_10d": "Mean($volume_ratio,10)",
    # 21. sma_volume_ratio_5d
    "sma_volume_ratio_5d": "Mean($volume_ratio,5)",
    # 22. profit_growth_yoy: 季度净利润同比增速（Qlib 自带字段）
    "profit_growth_yoy": "$q_profit_yoy",
    # 23. q_profit_yoy: 与 #22 同义
    "q_profit_yoy": "$q_profit_yoy",
    # 24. earnings_growth_smooth: 季度净利润增速 5 日均值（平滑）
    "earnings_growth_smooth": "Mean($q_profit_yoy,5)",
    # 25. roe_momentum_60d: ROE 60 日差分
    #    旧版本用 ratio (Ref/x-1)，此次按 markdown diff(60) 修正：
    #    diff 在 ROE 接近 0 时数值稳定，更适合长周期改善信号
    "roe_momentum_60d": "$roe-Ref($roe,60)",
    # 26. rank_pb_roe: 截面 rank(roe)-rank(pb) → 时序 roe-pb 降级
    "rank_pb_roe": "$roe-$pb",
    # 27. intraday_range: (high-low)/close 日内振幅（与 base 因子 KLEN 分母不同）
    "intraday_range": "($high_qfq-$low_qfq)/($close_qfq+1e-12)",
    # 28. vol_ratio_5d_20d: 5 日 / 20 日均量比
    "vol_ratio_5d_20d": "Mean($volume,5)/(Mean($volume,20)+1e-12)",
    # 29. turnover_accel_5d: 当前 5 日均换手 / 前 5 日均换手
    "turnover_accel_5d": (
        "Mean($turnover_rate,5)/(Ref(Mean($turnover_rate,5),5)+1e-12)"
    ),
    # 30. turnover_acceleration_5d: 5 日 / 10 日均换手 - 1
    "turnover_acceleration_5d": (
        "Mean($turnover_rate,5)/(Mean($turnover_rate,10)+1e-12)-1"
    ),
    # 31. roe_pb_momentum_divergence_60d: ROE 改善 - PB 涨幅
    "roe_pb_momentum_divergence_60d": (
        "($roe-Ref($roe,60))-($pb/(Ref($pb,60)+1e-12)-1)"
    ),
    # 32. liq_turnover_f: 流通换手率
    "liq_turnover_f": "$turnover_rate_f",
    # 33. turnover_rate_f: 与 #32 同义
    "turnover_rate_f": "$turnover_rate_f",
    # 34. raw_turnover_rate_f: 与 #32 同义
    "raw_turnover_rate_f": "$turnover_rate_f",
    # 35. roe_pb_ratio: ROE / PB（GP 风格的"质量价值比"）
    "roe_pb_ratio": "$roe/($pb+1e-12)",
    # 36. pb_roe: 与 #35 同义（RDAgent 中两个 workspace 实现一致）
    "pb_roe": "$roe/($pb+1e-12)",
    # 37. volume_ratio_ma_15d
    "volume_ratio_ma_15d": "Mean($volume_ratio,15)",
    # 38. rank_high_low: 截面 rank(high/low) → 时序值降级
    "rank_high_low": "$high_qfq/($low_qfq+1e-12)",
    # 39. rank_roe: 截面 rank → 时序值降级
    "rank_roe": "$roe",
    # 40. quarterly_eps: 季度 EPS
    #    旧版本曾用 $eps（年度），此次按 markdown 修正为 $q_eps（季度）
    "quarterly_eps": "$q_eps",
    # 41. turnover_rate_f_ma_15d: 注意 markdown 中此因子 IC 为负（短期高换手反指）
    "turnover_rate_f_ma_15d": "Mean($turnover_rate_f,15)",
    # 42. roe_quality: 直接用 ROE
    "roe_quality": "$roe",
    # 43. ROE_Factor: 与 #42 同义
    "ROE_Factor": "$roe",
    # 44. TURN_Factor: 当日换手率
    "TURN_Factor": "$turnover_rate",
    # 45. roe_pe_ratio: ROE / PE（与 #35 不同分母，独立信号）
    "roe_pe_ratio": "$roe/($pe_ttm+1e-12)",
    # 46. quality_value_interaction: 截面 rank(roe)×rank(1/pe) → 与 #45 等价降级
    "quality_value_interaction": "$roe/($pe_ttm+1e-12)",
    # 47. turnover_ma_10
    "turnover_ma_10": "Mean($turnover_rate,10)",
    # 48. liq_turnover_10d: 与 #47 同义
    "liq_turnover_10d": "Mean($turnover_rate,10)",
    # 49. sma_dv_ratio_10d: 股息率 10 日均值
    "sma_dv_ratio_10d": "Mean($dv_ratio,10)",
    # 50. sma_dv_ratio_5d: 股息率 5 日均值
    "sma_dv_ratio_5d": "Mean($dv_ratio,5)",
}


def _list_supported_names() -> list[str]:
    """返回所有支持的 JSON 因子名（用于报错提示与文档生成）。"""
    return sorted(_EXPR_BY_JSON_NAME.keys())


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
        ValueError: JSON 结构非法、解析后无有效因子名，
                    或某因子名在 _EXPR_BY_JSON_NAME 中无映射
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
        expr = _EXPR_BY_JSON_NAME.get(name)
        if expr is None:
            raise ValueError(
                f"combined_factors JSON 中的因子名 '{name}' 尚无内置 Qlib 表达式映射，"
                f"请在 python/features/combined_json_factors.py 的 _EXPR_BY_JSON_NAME "
                f"中补充。当前已支持 {len(_EXPR_BY_JSON_NAME)} 个名称："
                f"{_list_supported_names()}"
            )
        exprs.append(expr)
        names.append(name)
        existing.add(name)

    return exprs, names
