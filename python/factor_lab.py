"""
因子实验台对比工具（factor_lab.py）

层级位置: Data -> Feature -> Model -> Signal -> Portfolio -> Backtest -> [Evaluation]
职责:
  1. 列出 configs/factor_lab/ 下的因子白名单 json 与对应实验是否跑过（list）
  2. 扫描 lgbm_lab 实验产物（latest meta + rolling_summary）+ 静态锚点（v18-v2 / lgbm_ext），
     生成统一对比表（compare）

子命令:
  list                              列出所有 lab 的状态（已跑/未跑/最近修改时间）
  compare [--out PATH]              生成 markdown 对比表（默认输出到控制台）
                                    若 --out 指定路径，同时写入文件

设计原则:
  - 锚点行（v18-v2 / lgbm_ext）从 artifacts/models/latest_lgbm_daily_meta.json
    已记录的人工标定字段读，不再扫 100+ 历史 run 目录
  - lab 实验的单切分指标从 latest_lgbm_lab_<name>_daily_meta.json 的 metrics 段读
    （persist_artifacts 已透传），找不到时降级为 None
  - lab 实验的 WF 指标从最新 artifacts/lgbm_lab_<name>_daily_*/rolling_summary.json 读
  - 不调用任何外网 / 不修改任何文件（除 --out 指定的对比表）
  - JSON 中可能含 NaN，统一用 fmt() 转成 "—"
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = WORKSPACE_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
FACTOR_LAB_DIR = WORKSPACE_ROOT / "configs" / "factor_lab"

LAB_PREFIX = "lgbm_lab_"


@dataclass
class LabRow:
    """统一对比行：锚点 / lab 实验都用同一结构。"""

    name: str  # 显示名（v18-v2 / lgbm_ext / base_only / new_md_full15 / ...）
    role: str  # anchor / lab / sentinel
    n_features: Optional[int] = None
    combined_json: Optional[str] = None
    single_run_id: Optional[str] = None
    wf_run_id: Optional[str] = None
    # 单切分指标
    s_ic: Optional[float] = None
    s_ir: Optional[float] = None
    s_annret: Optional[float] = None
    s_sharpe: Optional[float] = None
    s_mdd: Optional[float] = None
    s_turn: Optional[float] = None
    # WF 指标
    wf_ic: Optional[float] = None
    wf_ir: Optional[float] = None
    wf_annret: Optional[float] = None
    wf_sharpe: Optional[float] = None
    wf_mdd: Optional[float] = None
    wf_turn: Optional[float] = None
    note: str = ""


# ─────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────


def _safe_load_json(path: Path) -> Optional[dict]:
    """容忍 NaN 的 JSON 加载（标准库 json 可读 NaN/Infinity）。"""
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # pragma: no cover - 防御式
        print(f"[factor_lab] 警告：解析 {path} 失败：{exc}", file=sys.stderr)
        return None


def _f(value: Any) -> Optional[float]:
    """提取 float，遇到 NaN / None / 非数值返回 None。"""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _fmt(value: Optional[float], digits: int = 4) -> str:
    """格式化对比表单元；None / NaN 显示 '—'。"""
    if value is None:
        return "—"
    return f"{value:.{digits}f}"


def _fmt_pct(value: Optional[float], digits: int = 2) -> str:
    """百分号格式化（输入 0.123 → '12.30%'）。"""
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


# ─────────────────────────────────────────────────────────────────
# 静态锚点（v18-v2 / lgbm_ext）
# ─────────────────────────────────────────────────────────────────


def _build_anchor_v18_v2() -> Optional[LabRow]:
    """从 artifacts/models/latest_lgbm_daily_meta.json 读 v18-v2 生产基线锚点。

    依赖该 meta 文件中由人工维护的 production_wf_baseline 字段。
    若该字段缺失或文件不存在，返回 None（对比表会自动隐藏该行）。
    """
    meta = _safe_load_json(MODELS_DIR / "latest_lgbm_daily_meta.json")
    if not meta:
        return None
    wf = meta.get("production_wf_baseline") or {}
    # ic_mean 优先（3 fold 平均 IC，最具代表性）；缺失则降级到 fold3_ic（最近一期）
    wf_ic = _f(wf.get("ic_mean"))
    if wf_ic is None:
        wf_ic = _f(wf.get("fold3_ic"))
    return LabRow(
        name="v18_v2 (production)",
        role="anchor",
        n_features=meta.get("n_features"),
        combined_json="configs/combined_factors_df.json",
        single_run_id=meta.get("run_id"),
        wf_run_id=meta.get("production_wf_run_id"),
        wf_ic=wf_ic,
        wf_ir=_f(wf.get("ir")),
        wf_annret=_f(wf.get("annret")),
        wf_sharpe=_f(wf.get("sharpe")),
        wf_mdd=_f(wf.get("mdd")),
        note="生产线（不动）；WF IR 为人工标定基线",
    )


def _build_anchor_lgbm_ext() -> Optional[LabRow]:
    """从 latest_lgbm_daily_meta.json 的 rejected_experiments[0] 读 lgbm_ext 已拒绝实验锚点。

    若没有这条记录则尝试 latest_lgbm_ext_daily_meta.json（旧格式，无 metrics）。
    """
    meta = _safe_load_json(MODELS_DIR / "latest_lgbm_daily_meta.json")
    if meta:
        for rec in meta.get("rejected_experiments", []) or []:
            if rec.get("version") == "v20_lgbm_ext":
                ss = rec.get("single_split_metrics", {}) or {}
                wf = rec.get("wf_metrics", {}) or {}
                return LabRow(
                    name="lgbm_ext (rejected)",
                    role="anchor",
                    n_features=45,
                    combined_json="configs/combined_factors_df_extension.json",
                    single_run_id=rec.get("run_id_single"),
                    wf_run_id=rec.get("run_id_wf"),
                    s_ic=_f(ss.get("ic_mean")),
                    s_ir=_f(ss.get("ir")),
                    s_annret=_f(ss.get("annret")),
                    s_sharpe=_f(ss.get("sharpe")),
                    s_mdd=_f(ss.get("mdd")),
                    wf_ic=_f(wf.get("fold3_ic")),
                    wf_ir=_f(wf.get("ir")),
                    wf_annret=_f(wf.get("annret")),
                    wf_sharpe=_f(wf.get("sharpe")),
                    wf_mdd=_f(wf.get("mdd")),
                    note="历史增量实验（v18-v2 + new.md 7 个候选），WF IR -35.4% 已拒绝",
                )
    # 回退：纯 latest_lgbm_ext_daily_meta.json（早期产物，无 metrics）
    ext_meta = _safe_load_json(MODELS_DIR / "latest_lgbm_ext_daily_meta.json")
    if ext_meta:
        return LabRow(
            name="lgbm_ext (legacy)",
            role="anchor",
            n_features=ext_meta.get("n_features"),
            combined_json="configs/combined_factors_df_extension.json",
            single_run_id=ext_meta.get("run_id"),
            note="legacy meta（无 metrics 详情），建议在生产 meta 标注后再读",
        )
    return None


# ─────────────────────────────────────────────────────────────────
# Lab 实验扫描
# ─────────────────────────────────────────────────────────────────


def _scan_wf_for_lab(lab_name: str) -> Optional[Path]:
    """扫描 artifacts/lgbm_lab_<name>_daily_*/rolling_summary.json，按 mtime 取最新。

    Returns:
        最新 rolling_summary.json 路径；找不到则 None。
    """
    pattern = f"lgbm_lab_{lab_name}_daily_*"
    candidates = sorted(
        ARTIFACTS_DIR.glob(f"{pattern}/rolling_summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _build_lab_row(latest_meta_path: Path) -> Optional[LabRow]:
    """从 artifacts/models/latest_lgbm_lab_<name>_daily_meta.json 构建一行。"""
    meta = _safe_load_json(latest_meta_path)
    if not meta:
        return None

    # 解析 lab_name（优先 meta.lab_name，回退到文件名解析）
    lab_name = meta.get("lab_name")
    if not lab_name:
        # 文件名形如 latest_lgbm_lab_<lab>_daily_meta.json
        stem = latest_meta_path.stem  # latest_lgbm_lab_<lab>_daily_meta
        if stem.startswith("latest_lgbm_lab_") and stem.endswith("_daily_meta"):
            lab_name = stem[len("latest_lgbm_lab_"):-len("_daily_meta")]
        else:
            lab_name = stem

    row = LabRow(
        name=lab_name,
        role="lab",
        n_features=meta.get("n_features"),
        combined_json=meta.get("combined_factors_json") or (
            "(none, base_only)" if lab_name == "base_only" else None
        ),
        single_run_id=meta.get("run_id"),
    )

    # 单切分指标：persist_artifacts 已透传到 meta.metrics
    metrics = meta.get("metrics") or {}
    if metrics:
        row.s_ic = _f(metrics.get("ic_mean"))
        row.s_ir = _f(metrics.get("information_ratio"))
        row.s_annret = _f(metrics.get("annualized_return"))
        row.s_sharpe = _f(metrics.get("sharpe_ratio"))
        row.s_mdd = _f(metrics.get("max_drawdown"))
        row.s_turn = _f(metrics.get("avg_turnover"))

    # WF 指标：扫最近一次 rolling_summary.json
    wf_path = _scan_wf_for_lab(lab_name)
    if wf_path is not None:
        wf_summary = _safe_load_json(wf_path) or {}
        fold_stats = wf_summary.get("fold_stats", {}) or {}
        cm = wf_summary.get("concatenated_metrics", {}) or {}
        ic_block = fold_stats.get("ic_mean", {}) if isinstance(fold_stats, dict) else {}
        row.wf_ic = _f(ic_block.get("mean")) if isinstance(ic_block, dict) else None
        row.wf_ir = _f(cm.get("information_ratio"))
        row.wf_annret = _f(cm.get("annualized_return"))
        row.wf_sharpe = _f(cm.get("sharpe_ratio"))
        row.wf_mdd = _f(cm.get("max_drawdown"))
        row.wf_turn = _f(cm.get("avg_turnover"))
        row.wf_run_id = wf_path.parent.name

    return row


def _list_all_labs() -> list[tuple[str, Path, bool, bool]]:
    """扫描 configs/factor_lab/*.json，返回 [(name, json_path, has_single, has_wf)]。

    has_single/has_wf 通过对应 latest meta 与 rolling_summary 推断。
    """
    if not FACTOR_LAB_DIR.exists():
        return []
    result: list[tuple[str, Path, bool, bool]] = []
    for jp in sorted(FACTOR_LAB_DIR.glob("*.json")):
        name = jp.stem
        latest_meta = MODELS_DIR / f"latest_lgbm_lab_{name}_daily_meta.json"
        has_single = latest_meta.exists()
        has_wf = _scan_wf_for_lab(name) is not None
        result.append((name, jp, has_single, has_wf))
    return result


# ─────────────────────────────────────────────────────────────────
# 子命令实现
# ─────────────────────────────────────────────────────────────────


def cmd_list() -> int:
    """列出所有 lab 的状态。"""
    rows = _list_all_labs()
    if not rows:
        print(f"configs/factor_lab/ 为空或不存在：{FACTOR_LAB_DIR}")
        return 0

    print(f"=== Factor Lab 状态总览 ({len(rows)} 个 lab) ===")
    print(f"{'lab_name':<28} {'json_size':>9} {'single?':>9} {'wf?':>5}  {'last_modified'}")
    print("-" * 90)
    for name, jp, has_single, has_wf in rows:
        size_kb = f"{jp.stat().st_size / 1024:.1f}KB"
        mtime = datetime.fromtimestamp(jp.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        s_mark = "[OK]" if has_single else "[--]"
        w_mark = "[OK]" if has_wf else "[--]"
        print(f"{name:<28} {size_kb:>9} {s_mark:>9} {w_mark:>5}  {mtime}")
    print()
    print("提示：")
    print("  - [OK] = 已跑过该模式；[--] = 尚未跑过")
    print("  - base_only 是哨兵文件（运行时不读 json），单切分仅当 --lab base_only 跑过才会显示 OK")
    print("  - 跑实验：python python/run_experiment.py --model lgbm_lab --freq daily --lab <name>")
    print("  - 拉对比表：python python/factor_lab.py compare")
    return 0


def _collect_compare_rows() -> list[LabRow]:
    """汇总锚点 + 所有 lab 实验行。"""
    rows: list[LabRow] = []

    # 锚点置顶（顺序：v18-v2 → lgbm_ext）
    anchor_v18 = _build_anchor_v18_v2()
    if anchor_v18 is not None:
        rows.append(anchor_v18)
    anchor_ext = _build_anchor_lgbm_ext()
    if anchor_ext is not None:
        rows.append(anchor_ext)

    # 扫所有 lab 实验
    if MODELS_DIR.exists():
        lab_metas = sorted(MODELS_DIR.glob("latest_lgbm_lab_*_daily_meta.json"))
        for meta_path in lab_metas:
            row = _build_lab_row(meta_path)
            if row is not None:
                rows.append(row)

    return rows


def _render_markdown(rows: list[LabRow]) -> str:
    """渲染 markdown 对比表。"""
    headers = [
        "exp", "role", "n_feat", "combined_json",
        "IC(s)", "IR(s)", "AnnRet(s)", "Sharpe(s)", "MDD(s)", "Turn(s)",
        "IC(WF)", "IR(WF)", "AnnRet(WF)", "Sharpe(WF)", "MDD(WF)", "Turn(WF)",
        "note",
    ]
    lines = []
    lines.append("# Factor Lab 对比表")
    lines.append("")
    lines.append(f"_生成时间: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("- 后缀 (s) = 单切分 single-split；后缀 (WF) = walk-forward。")
    lines.append("- IR 是核心指标；AnnRet/MDD/Sharpe/Turn 提供风险与稳定性参考。")
    lines.append("- 锚点（v18_v2 / lgbm_ext）来自人工标定的 latest_lgbm_daily_meta.json，与历史结论保持一致。")
    lines.append("")

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for r in rows:
        cells = [
            r.name,
            r.role,
            str(r.n_features) if r.n_features is not None else "—",
            r.combined_json or "—",
            _fmt(r.s_ic), _fmt(r.s_ir), _fmt_pct(r.s_annret), _fmt(r.s_sharpe),
            _fmt_pct(r.s_mdd), _fmt_pct(r.s_turn),
            _fmt(r.wf_ic), _fmt(r.wf_ir), _fmt_pct(r.wf_annret), _fmt(r.wf_sharpe),
            _fmt_pct(r.wf_mdd), _fmt_pct(r.wf_turn),
            r.note,
        ]
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append("## 解读建议")
    lines.append("")
    lines.append("- **base_only vs <lab_x>**：直接看 IR(WF) 差。base_only 是 base-only 下界，任何 lab 实验若 IR(WF) ≤ base_only 则该套因子整体无正向贡献。")
    lines.append("- **<lab_x> vs v18_v2**：若 IR(WF) ≥ v18_v2 (生产)，可考虑替换；否则只作为研究存档。")
    lines.append("- **IR(s) 与 IR(WF) 一致性**：若 IR(s) 高但 IR(WF) 低，常见原因是风格漂移或单切分过拟合（参考 v19 / lgbm_ext 历史教训）。")
    lines.append("- **Turnover 警戒线**：相比 v18_v2 (~0.40) 提升 >20% 但 IR 未同比上升，多半是噪声因子叠加。")

    return "\n".join(lines) + "\n"


def cmd_compare(out_path: Optional[Path]) -> int:
    rows = _collect_compare_rows()
    if not rows:
        print("[factor_lab] 没有任何可对比的实验数据：未找到锚点 meta，也没有 lgbm_lab_* 实验。", file=sys.stderr)
        return 1

    md = _render_markdown(rows)

    # 控制台渲染（直接打印 markdown）
    print(md)

    # 文件输出（可选）
    if out_path:
        out_path = (out_path if out_path.is_absolute() else WORKSPACE_ROOT / out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md, encoding="utf-8")
        print(f"[factor_lab] 对比表已写入：{out_path}")
    return 0


# ─────────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="factor_lab",
        description="因子实验台对比工具：列出 lab / 生成 markdown 对比表",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出 configs/factor_lab/ 下所有 lab 与跑过状态")

    p_compare = sub.add_parser(
        "compare",
        help="生成 markdown 对比表（含锚点 v18_v2 / lgbm_ext + 所有 lab 实验）",
    )
    p_compare.add_argument(
        "--out",
        default=None,
        type=Path,
        help="可选：把对比表写入到该路径（相对仓库根，如 artifacts/factor_lab_compare.md）",
    )

    args = parser.parse_args()
    if args.cmd == "list":
        return cmd_list()
    if args.cmd == "compare":
        return cmd_compare(args.out)
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
