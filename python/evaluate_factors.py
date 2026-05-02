"""
单因子质量评估工具：IC / RankIC / ICIR。

层级位置: Data -> [Feature] -> Model -> Signal -> Portfolio -> Backtest -> Evaluation
职责: 对因子库中的每个因子（或指定子集）计算截面 IC 时序，输出评估报告。
      结果写入 artifacts/factor_eval/ 并在终端打印摘要表格。

RDAgent 兼容:
  本脚本可被 RDAgent 调用以评估新生成的因子，调用约定:
    python python/evaluate_factors.py --groups momentum --start 2022-01-01 --end 2024-12-31

输入:
  --groups    因子组过滤（逗号分隔，默认全部）
  --factors   具体因子名过滤（逗号分隔，与 groups 互斥）
  --start     评估起始日期（YYYY-MM-DD）
  --end       评估结束日期（YYYY-MM-DD）
  --instruments 股票池（默认 all）
  --ic-method pearson | spearman（默认 spearman，即 RankIC）
  --output-dir 结果输出目录（默认 artifacts/factor_eval）

输出:
  artifacts/factor_eval/ic_series.parquet   每因子每日 IC 时序
  artifacts/factor_eval/ic_summary.csv      汇总表（ICIR/均值/标准差/胜率）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# 将 python/ 目录加入 sys.path，确保 from features import ... 可正常解析
sys.path.insert(0, str(Path(__file__).resolve().parent))

from features import build_feature_config, get_registry, print_registry  # noqa: E402

try:
    import qlib
    from qlib.data import D
    from qlib.log import get_module_logger
except ImportError as exc:
    raise ImportError(
        "无法导入 qlib。请在 qlib_zhengshi 虚拟环境中执行：pip install pyqlib"
    ) from exc

LOGGER = get_module_logger("evaluate_factors")
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
VERSION = "v1"

_LOG_EXTRA = {"run_id": "factor_eval", "instrument": "ALL", "datetime": "", "signal": 0.0, "version": VERSION}


# ─────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────

def _load_provider_uri() -> str:
    cfg_path = WORKSPACE_ROOT / "configs" / "base.yaml"
    with cfg_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    return base_cfg["qlib_init"]["provider_uri"]


def load_factor_data(
    exprs: list[str],
    names: list[str],
    instruments: str,
    start_time: str,
    end_time: str,
) -> pd.DataFrame:
    """
    使用 Qlib D.features() 加载因子数据。

    返回:
        DataFrame，MultiIndex (datetime, instrument)，列为因子名
    """
    LOGGER.info("加载因子数据: %d 个因子, %s ~ %s", len(names), start_time, end_time, extra=_LOG_EXTRA)
    instrument_list = D.instruments(instruments)
    df = D.features(
        instruments=instrument_list,
        fields=exprs,
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )
    df.columns = names
    return df


def load_label_data(
    instruments: str,
    start_time: str,
    end_time: str,
) -> pd.Series:
    """
    加载标签（未来收益率）：Ref($close_qfq,-2)/Ref($close_qfq,-1)-1。

    返回:
        Series，MultiIndex (datetime, instrument)，名称 LABEL0
    """
    label_expr = "Ref($close_qfq,-2)/Ref($close_qfq,-1)-1"
    LOGGER.info("加载标签数据: %s ~ %s", start_time, end_time, extra=_LOG_EXTRA)
    instrument_list = D.instruments(instruments)
    df = D.features(
        instruments=instrument_list,
        fields=[label_expr],
        start_time=start_time,
        end_time=end_time,
        freq="day",
    )
    df.columns = ["LABEL0"]
    return df["LABEL0"]


# ─────────────────────────────────────────────────────────────────
# IC 计算
# ─────────────────────────────────────────────────────────────────

def compute_ic_series(
    factor_df: pd.DataFrame,
    label_s: pd.Series,
    method: str = "spearman",
) -> pd.DataFrame:
    """
    计算每个因子在每个截面日期的 IC（截面相关系数）。

    参数:
        factor_df  MultiIndex (datetime, instrument) 的因子 DataFrame
        label_s    MultiIndex (datetime, instrument) 的标签 Series
        method     'spearman'（RankIC，默认）或 'pearson'（IC）

    返回:
        DataFrame，index=datetime，columns=因子名，值为每日截面 IC
    """
    combined = factor_df.copy()
    combined["__label__"] = label_s
    combined = combined.dropna(subset=["__label__"])

    def _cross_section_ic(grp: pd.DataFrame) -> pd.Series:
        label = grp["__label__"]
        factors = grp.drop(columns=["__label__"])
        return factors.corrwith(label, method=method)

    ic_df = (
        combined.groupby(level="datetime")
        .apply(_cross_section_ic)
    )
    return ic_df


def compute_ic_summary(ic_df: pd.DataFrame) -> pd.DataFrame:
    """
    从 IC 时序计算汇总指标。

    返回列:
        ic_mean   IC 均值
        ic_std    IC 标准差
        icir      IC 信息比率（均值/标准差）
        ic_gt0    IC>0 的比率（胜率）
        obs_days  有效观测天数
    """
    summary = pd.DataFrame({
        "ic_mean": ic_df.mean(),
        "ic_std": ic_df.std(),
        "icir": ic_df.mean() / (ic_df.std() + 1e-12),
        "ic_gt0": (ic_df > 0).mean(),
        "obs_days": ic_df.count(),
    })
    summary = summary.sort_values("icir", ascending=False)
    return summary


# ─────────────────────────────────────────────────────────────────
# 输出
# ─────────────────────────────────────────────────────────────────

def print_summary(summary: pd.DataFrame, method: str) -> None:
    method_label = "RankIC (Spearman)" if method == "spearman" else "IC (Pearson)"
    print(f"\n{'='*72}")
    print(f"  单因子评估报告 — {method_label}")
    print(f"{'='*72}")
    header = f"{'因子名':<20} {'IC均值':>9} {'IC标准差':>9} {'ICIR':>9} {'IC>0%':>8} {'观测天数':>8}"
    print(header)
    print("-" * 72)
    for name, row in summary.iterrows():
        icir_flag = " ★" if abs(row["icir"]) >= 0.3 else ""
        print(
            f"{str(name):<20} {row['ic_mean']:>9.4f} {row['ic_std']:>9.4f} "
            f"{row['icir']:>9.4f} {row['ic_gt0']:>8.1%} {int(row['obs_days']):>8}{icir_flag}"
        )
    print(f"{'='*72}")
    print("  ★ ICIR ≥ 0.3 为推荐保留因子")
    print()


def persist_results(
    ic_df: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ic_df.to_parquet(output_dir / "ic_series.parquet")
    summary.to_csv(output_dir / "ic_summary.csv", encoding="utf-8-sig")
    LOGGER.info("评估结果已写入: %s", output_dir, extra=_LOG_EXTRA)


# ─────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="单因子 IC/ICIR/RankIC 评估工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估全部因子
  python python/evaluate_factors.py

  # 只评估动量和量能组
  python python/evaluate_factors.py --groups momentum,volume

  # 评估指定因子
  python python/evaluate_factors.py --factors RET1,RSI12,MA5_20

  # 指定时间区间与方法
  python python/evaluate_factors.py --start 2022-01-01 --end 2024-12-31 --ic-method pearson
        """,
    )
    parser.add_argument("--groups", default=None, help="因子组过滤，逗号分隔（momentum,volume,technical,fundamental）")
    parser.add_argument("--factors", default=None, help="具体因子名过滤，逗号分隔（与 --groups 互斥）")
    parser.add_argument("--start", default="2021-01-01", help="评估起始日期（默认 2021-01-01）")
    parser.add_argument("--end", default="2025-12-31", help="评估结束日期（默认 2025-12-31）")
    parser.add_argument("--instruments", default="all", help="股票池（默认 all）")
    parser.add_argument("--ic-method", choices=["spearman", "pearson"], default="spearman",
                        help="IC 计算方法：spearman（RankIC，默认）/ pearson（IC）")
    parser.add_argument("--output-dir", default=None, help="输出目录（默认 artifacts/factor_eval）")
    parser.add_argument("--list-factors", action="store_true", help="仅打印因子注册表，不执行评估")
    args = parser.parse_args()

    if args.list_factors:
        print_registry()
        return

    output_dir = Path(args.output_dir) if args.output_dir else WORKSPACE_ROOT / "artifacts" / "factor_eval"

    # ── qlib 初始化 ──────────────────────────────────────────────
    provider_uri = _load_provider_uri()
    qlib.init(provider_uri=provider_uri, region="cn")

    # ── 解析因子过滤条件 ──────────────────────────────────────────
    if args.factors and args.groups:
        parser.error("--factors 与 --groups 不能同时使用")

    groups: list[str] | None = None
    filter_names: set[str] | None = None

    if args.groups:
        groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    if args.factors:
        filter_names = {f.strip() for f in args.factors.split(",") if f.strip()}

    # 获取完整列表再过滤
    exprs, names = build_feature_config(groups=groups)
    if filter_names:
        pairs = [(e, n) for e, n in zip(exprs, names) if n in filter_names]
        if not pairs:
            print(f"[错误] 未找到因子：{filter_names}，请用 --list-factors 查看可用因子")
            sys.exit(1)
        exprs, names = zip(*pairs)  # type: ignore[assignment]
        exprs, names = list(exprs), list(names)

    print(f"\n评估 {len(names)} 个因子：{names}")

    # ── 加载数据 ──────────────────────────────────────────────────
    factor_df = load_factor_data(exprs, names, args.instruments, args.start, args.end)
    label_s = load_label_data(args.instruments, args.start, args.end)

    # ── 计算 IC ───────────────────────────────────────────────────
    LOGGER.info("计算 %s IC...", args.ic_method, extra=_LOG_EXTRA)
    ic_df = compute_ic_series(factor_df, label_s, method=args.ic_method)
    summary = compute_ic_summary(ic_df)

    # ── 输出 ──────────────────────────────────────────────────────
    print_summary(summary, method=args.ic_method)
    persist_results(ic_df, summary, output_dir)

    print(f"  IC 时序已保存至: {output_dir / 'ic_series.parquet'}")
    print(f"  汇总表已保存至: {output_dir / 'ic_summary.csv'}\n")


if __name__ == "__main__":
    main()
