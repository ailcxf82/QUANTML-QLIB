"""
实盘账户对账脚本（Reconcile Live Account）
=============================================

层级位置：诊断脚本（只读 selection + 用户提供的持仓快照，零侧效）

用途：
  当模拟净值（diagnose_recent_predictions.py 输出）与实盘账户感受不一致时，
  用本脚本逐日定位差异来源：漏单 / 多仓 / 数量不一致 / 不在推荐池的持仓。

工作流（两步）：
  1) 生成模板：
       python scripts/reconcile_live_account.py --gen-template --since 2026-05-13
     输出 reports/live_positions_template.csv，预填 expected_shares。
     拷贝为 reports/live_positions.csv，在 actual_shares 一列填你的真实持仓。
     若有"selection 没推荐但你也持有"的票，可直接在文件末尾新增行。

  2) 跑对账：
       python scripts/reconcile_live_account.py --positions reports/live_positions.csv
     输出每日差异表（match / missing / extra / short / excess）+ 累计漏单金额。

文件契约（CSV 列）：
  date,instrument,expected_shares,actual_shares[,close_price,note]
  - date           : YYYY-MM-DD（建议 = predictions/<date>/ 的目录名）
  - instrument     : qlib 编码（如 600906.SH / 000050.SZ）
  - expected_shares: 推荐持仓股数（脚本预填，可留空表示"不在推荐中"）
  - actual_shares  : 你的真实持仓股数（必填；不持有写 0）
  - close_price    : 当日收盘价（可选，留空时脚本会从 qlib 拉）
  - note           : 备注（可选）

约束：
  - actual_shares 必须为整数（A 股以 100 股为单位，建议是 100 的整数倍）
  - 单元格空白当作"无数据"；actual_shares 为空整行跳过
  - 一日内同一股票多行 → 报错（请合并）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = WORKSPACE_ROOT / "predictions"
REPORTS_DIR = WORKSPACE_ROOT / "reports"
LIVE_PROVIDER = "D:/qlib_data/qlib_data"
DEFAULT_ACCOUNT = 200000.0  # 与 configs/live/daily_live.yaml 一致
DEFAULT_TRADE_UNIT = 100


# ─────────────────────────────────────────────────────────────────────────────
# qlib 价格拉取（懒加载，仅模板生成时需要）
# ─────────────────────────────────────────────────────────────────────────────

_QLIB_INITED = False


def _ensure_qlib() -> None:
    global _QLIB_INITED
    if _QLIB_INITED:
        return
    import qlib
    qlib.init(
        provider_uri=LIVE_PROVIDER,
        region="cn",
        expression_cache=None,
        dataset_cache=None,
    )
    _QLIB_INITED = True


def _get_close(instruments: List[str], date_str: str) -> Dict[str, float]:
    """拉指定日期收盘价（qfq）。失败返回 0.0。"""
    if not instruments:
        return {}
    _ensure_qlib()
    from qlib.data import D
    try:
        df = D.features(
            instruments, ["$close_qfq"],
            start_time=date_str, end_time=date_str, freq="day",
        )
    except Exception as e:
        print(f"  [WARN] qlib 拉价失败 ({date_str}): {e}", file=sys.stderr)
        return {}
    if df.empty:
        return {}
    df = df.reset_index().rename(columns={"$close_qfq": "close"})
    out: Dict[str, float] = {}
    for inst, sub in df.groupby("instrument"):
        c = sub["close"].dropna()
        if not c.empty:
            out[str(inst)] = float(c.iloc[-1])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 读 selection.csv，归一化为 (date, instrument) → expected_shares 映射
# ─────────────────────────────────────────────────────────────────────────────

def _parse_selection(
    csv_path: Path,
    account: float,
    trade_unit: int,
) -> List[dict]:
    """
    将 selection.csv 转换为标准化"应持仓"行列表。
    返回元素 dict：
      date, instrument, expected_shares, close_price, source
        source = "target_shares" (5/15 起) | "weight_est" (5/13 之前)
    """
    df = pd.read_csv(csv_path, dtype={"instrument": str})
    if df.empty:
        return []

    df["rank_int"] = pd.to_numeric(df["rank"], errors="coerce").fillna(9999).astype(int)
    date_str = str(df["predict_date"].iloc[0])

    rows: List[dict] = []

    # rank == 999 即"应清仓"
    drop_rows = df[df["rank_int"] >= 999].copy()
    for _, r in drop_rows.iterrows():
        inst = str(r["instrument"])
        price = float(r.get("close_price", 0) or 0)
        rows.append({
            "date": date_str,
            "instrument": inst,
            "expected_shares": 0,
            "close_price": price,
            "source": "drop_out",
        })

    held_rows = df[df["rank_int"] < 999].copy()
    if held_rows.empty:
        return rows

    if "target_shares" in held_rows.columns and held_rows["target_shares"].notna().any():
        for _, r in held_rows.iterrows():
            t = pd.to_numeric(r.get("target_shares"), errors="coerce")
            price = float(r.get("close_price", 0) or 0)
            rows.append({
                "date": date_str,
                "instrument": str(r["instrument"]),
                "expected_shares": int(t) if pd.notna(t) else 0,
                "close_price": price,
                "source": "target_shares",
            })
    else:
        # 早期 csv（5/13 之前）：用 suggested_weight × account / close_price 估
        prices = _get_close(held_rows["instrument"].astype(str).tolist(), date_str)
        for _, r in held_rows.iterrows():
            inst = str(r["instrument"])
            w = pd.to_numeric(r.get("suggested_weight"), errors="coerce")
            if pd.isna(w) or w <= 0:
                continue
            price = prices.get(inst, 0.0)
            if price <= 0:
                exp_shares = 0
            else:
                budget = float(w) * account
                lots = int(budget // (price * trade_unit))
                exp_shares = lots * trade_unit
            rows.append({
                "date": date_str,
                "instrument": inst,
                "expected_shares": exp_shares,
                "close_price": price,
                "source": "weight_est",
            })

    return rows


def _collect_expected(since: Optional[str], account: float, trade_unit: int) -> pd.DataFrame:
    rows: List[dict] = []
    for sub in sorted(PRED_DIR.iterdir()):
        if not sub.is_dir():
            continue
        if since and sub.name < since:
            continue
        csv = sub / "selection.csv"
        if not csv.exists():
            continue
        rows.extend(_parse_selection(csv, account, trade_unit))
    if not rows:
        return pd.DataFrame(
            columns=["date", "instrument", "expected_shares", "close_price", "source"]
        )
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 子命令：生成模板
# ─────────────────────────────────────────────────────────────────────────────

def cmd_gen_template(args: argparse.Namespace) -> int:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    exp = _collect_expected(args.since, args.account, args.trade_unit)
    if exp.empty:
        print(f"[ERROR] 在 {PRED_DIR} 找不到 since>={args.since} 的 selection.csv",
              file=sys.stderr)
        return 2

    # 只保留 expected_shares > 0 的"应持仓"行 + drop_out 的清仓行（后者不在模板）
    held = exp[exp["expected_shares"] > 0].copy()
    held = held.sort_values(["date", "instrument"]).reset_index(drop=True)
    held["actual_shares"] = ""  # 用户填
    held["note"] = ""  # 用户备注

    out_path = REPORTS_DIR / "live_positions_template.csv"
    out = held[["date", "instrument", "expected_shares", "actual_shares",
                "close_price", "note"]]
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  [生成] 模板 → {out_path}")
    print(f"  共 {len(out)} 行，覆盖 {held['date'].nunique()} 个交易日")
    print(f"  日期范围: {held['date'].min()} ~ {held['date'].max()}")
    print(f"  涉及 {held['instrument'].nunique()} 只不同股票\n")
    print("  接下来：")
    print(f"    1) 把 {out_path.name} 拷贝为 live_positions.csv（同目录）")
    print( "    2) 在 actual_shares 一列填入你的真实持仓股数（不持有=0）")
    print( "    3) 若有 selection 没推荐但你也持有的票，在文件末尾追加行")
    print( "    4) 运行 python scripts/reconcile_live_account.py "
           "--positions reports/live_positions.csv")
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# 子命令：跑对账
# ─────────────────────────────────────────────────────────────────────────────

def _classify(exp: int, act: int) -> str:
    if act == 0 and exp == 0:
        return "match"
    if act == 0 and exp > 0:
        return "missing"   # 漏单（应买未买）
    if exp == 0 and act > 0:
        return "extra"     # 多仓（应清未清，或额外加仓不在推荐池）
    if act == exp:
        return "match"
    if act < exp:
        return "short"     # 少仓（数量偏少）
    return "excess"        # 超仓（数量偏多）


def cmd_reconcile(args: argparse.Namespace) -> int:
    pos_path = Path(args.positions)
    if not pos_path.is_absolute():
        pos_path = WORKSPACE_ROOT / pos_path
    if not pos_path.exists():
        print(f"[ERROR] 未找到 {pos_path}", file=sys.stderr)
        return 3

    actual_df = pd.read_csv(pos_path, dtype={"instrument": str})
    needed = {"date", "instrument", "actual_shares"}
    missing = needed - set(actual_df.columns)
    if missing:
        print(f"[ERROR] {pos_path.name} 缺少必填列: {sorted(missing)}", file=sys.stderr)
        return 3

    # 清洗 actual_shares
    actual_df["actual_shares"] = pd.to_numeric(
        actual_df["actual_shares"], errors="coerce"
    )
    # 整行 actual 为空 → 视作未填，跳过
    actual_df = actual_df.dropna(subset=["actual_shares"]).copy()
    if actual_df.empty:
        print(f"[ERROR] {pos_path.name} 中 actual_shares 全部为空，请先填入实盘持仓",
              file=sys.stderr)
        return 3
    actual_df["actual_shares"] = actual_df["actual_shares"].astype(int)

    # 同日同标的检查
    dup = actual_df.duplicated(subset=["date", "instrument"], keep=False)
    if dup.any():
        print("[ERROR] 检测到同日同标的多行（请合并）:", file=sys.stderr)
        print(actual_df.loc[dup, ["date", "instrument", "actual_shares"]].to_string(
            index=False), file=sys.stderr)
        return 3

    # 拉 expected
    since = actual_df["date"].min()
    until = actual_df["date"].max()
    exp_df = _collect_expected(since, args.account, args.trade_unit)
    exp_df = exp_df[(exp_df["date"] >= since) & (exp_df["date"] <= until)].copy()

    # 全外连接
    merged = pd.merge(
        exp_df, actual_df,
        on=["date", "instrument"], how="outer", suffixes=("_e", "_a"),
    )
    # 兼容 close_price 字段：merge 时 actual 文件中也允许有该列
    if "close_price_a" in merged.columns and "close_price_e" in merged.columns:
        merged["close_price"] = merged["close_price_a"].combine_first(
            merged["close_price_e"]
        )
    elif "close_price_e" in merged.columns:
        merged.rename(columns={"close_price_e": "close_price"}, inplace=True)
    elif "close_price_a" in merged.columns:
        merged.rename(columns={"close_price_a": "close_price"}, inplace=True)

    # 缺失补 0
    merged["expected_shares"] = (
        pd.to_numeric(merged["expected_shares"], errors="coerce").fillna(0).astype(int)
    )
    merged["actual_shares"] = (
        pd.to_numeric(merged["actual_shares"], errors="coerce").fillna(0).astype(int)
    )
    merged["close_price"] = pd.to_numeric(
        merged["close_price"], errors="coerce"
    )

    # 对未提供 close_price 的行，按日期批量从 qlib 拉
    need_price_mask = merged["close_price"].isna() | (merged["close_price"] <= 0)
    if need_price_mask.any():
        for d, sub in merged[need_price_mask].groupby("date"):
            insts = sorted(sub["instrument"].unique().tolist())
            prices = _get_close(insts, str(d))
            for inst, p in prices.items():
                cond = (merged["date"] == d) & (merged["instrument"] == inst)
                merged.loc[cond, "close_price"] = p
    merged["close_price"] = merged["close_price"].fillna(0.0)

    merged["diff_shares"] = merged["actual_shares"] - merged["expected_shares"]
    merged["expected_value"] = merged["expected_shares"] * merged["close_price"]
    merged["actual_value"] = merged["actual_shares"] * merged["close_price"]
    merged["diff_value"] = merged["actual_value"] - merged["expected_value"]
    merged["status"] = merged.apply(
        lambda r: _classify(int(r["expected_shares"]), int(r["actual_shares"])),
        axis=1,
    )

    # 输出：每日差异表
    sort_key = {"missing": 0, "extra": 1, "short": 2, "excess": 3, "match": 4}
    merged["_sk"] = merged["status"].map(sort_key)
    merged = merged.sort_values(["date", "_sk", "instrument"]).drop(columns="_sk")

    print("\n" + "=" * 90)
    print("逐日持仓对账（match=一致, missing=漏单, extra=多仓, short=少仓, excess=超仓）")
    print("=" * 90)
    if args.hide_match:
        view = merged[merged["status"] != "match"].copy()
    else:
        view = merged.copy()

    if view.empty:
        print("  全部一致，零差异。")
    else:
        fmt = view.copy()
        fmt["close_price"] = fmt["close_price"].apply(
            lambda v: f"{v:.2f}" if pd.notna(v) and v > 0 else "—"
        )
        fmt["expected_value"] = fmt["expected_value"].apply(
            lambda v: f"{v:,.0f}" if pd.notna(v) else "—"
        )
        fmt["actual_value"] = fmt["actual_value"].apply(
            lambda v: f"{v:,.0f}" if pd.notna(v) else "—"
        )
        fmt["diff_value"] = fmt["diff_value"].apply(
            lambda v: f"{v:+,.0f}" if pd.notna(v) else "—"
        )
        cols = ["date", "instrument", "status", "expected_shares", "actual_shares",
                "diff_shares", "close_price", "expected_value", "actual_value",
                "diff_value"]
        print(fmt[cols].to_string(index=False))

    # 按日汇总
    print("\n" + "=" * 90)
    print("逐日汇总（按差异类型分桶，金额为绝对值）")
    print("=" * 90)
    summary = merged.groupby(["date", "status"]).agg(
        n=("instrument", "count"),
        diff_value_abs=("diff_value", lambda s: float(s.abs().sum())),
    ).reset_index()
    if summary.empty:
        print("  无数据")
    else:
        piv_n = summary.pivot(index="date", columns="status", values="n").fillna(0).astype(int)
        piv_v = summary.pivot(
            index="date", columns="status", values="diff_value_abs"
        ).fillna(0)

        for status in ["missing", "extra", "short", "excess", "match"]:
            if status not in piv_n.columns:
                piv_n[status] = 0
                piv_v[status] = 0.0

        piv_n = piv_n[["missing", "extra", "short", "excess", "match"]]
        piv_v = piv_v[["missing", "extra", "short", "excess"]]
        piv_v_fmt = piv_v.applymap(lambda v: f"{v:,.0f}" if v > 0 else "—")

        print("  [行数]")
        print(piv_n.to_string())
        print("\n  [偏离金额（元）]")
        print(piv_v_fmt.to_string())

    # 总累计
    print("\n" + "=" * 90)
    print("全程累计偏离")
    print("=" * 90)
    by_status = merged.groupby("status").agg(
        n=("instrument", "count"),
        diff_value_abs=("diff_value", lambda s: float(s.abs().sum())),
    ).reset_index()
    by_status["diff_value_abs"] = by_status["diff_value_abs"].apply(
        lambda v: f"{v:,.0f}"
    )
    print(by_status.to_string(index=False))

    miss_total = float(
        merged.loc[merged["status"] == "missing", "diff_value"].abs().sum()
    )
    extra_total = float(
        merged.loc[merged["status"] == "extra", "diff_value"].abs().sum()
    )
    short_total = float(
        merged.loc[merged["status"] == "short", "diff_value"].abs().sum()
    )
    excess_total = float(
        merged.loc[merged["status"] == "excess", "diff_value"].abs().sum()
    )
    print(f"\n  漏单累计金额:   {miss_total:>12,.0f} 元  (应买未买)")
    print(f"  多仓累计金额:   {extra_total:>12,.0f} 元  (应清未清 / 推荐外加仓)")
    print(f"  少仓累计金额:   {short_total:>12,.0f} 元  (数量偏少)")
    print(f"  超仓累计金额:   {excess_total:>12,.0f} 元  (数量偏多)")

    # 写出明细 CSV
    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = WORKSPACE_ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"\n  明细已保存: {out_path}")

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(
        description="实盘账户与 selection.csv 对账，定位漏单/多仓",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "用法：\n"
            "  生成模板：\n"
            "    python scripts/reconcile_live_account.py --gen-template --since 2026-05-13\n"
            "  填好 reports/live_positions.csv 后跑对账：\n"
            "    python scripts/reconcile_live_account.py "
            "--positions reports/live_positions.csv\n"
        ),
    )
    p.add_argument(
        "--gen-template", action="store_true",
        help="生成对账模板（reports/live_positions_template.csv），预填 expected_shares"
    )
    p.add_argument(
        "--positions", type=str, default=None,
        help="你填好的实盘持仓 CSV 路径（与 --gen-template 互斥）"
    )
    p.add_argument(
        "--since", type=str, default="2026-05-13",
        help="只看 date >= 此日期的 selection（默认 2026-05-13）"
    )
    p.add_argument(
        "--account", type=float, default=DEFAULT_ACCOUNT,
        help=f"账户总值（元，仅 5/13 之前 selection 估 expected_shares 用，默认 {DEFAULT_ACCOUNT:.0f}）"
    )
    p.add_argument(
        "--trade-unit", type=int, default=DEFAULT_TRADE_UNIT,
        help=f"A 股交易单位（默认 {DEFAULT_TRADE_UNIT}）"
    )
    p.add_argument(
        "--hide-match", action="store_true",
        help="对账输出中隐藏 status=match 的行，只看差异"
    )
    p.add_argument(
        "--out", type=str, default="reports/reconcile_detail.csv",
        help="对账明细输出 CSV 路径（默认 reports/reconcile_detail.csv，传空字符串关闭）"
    )
    args = p.parse_args()

    if args.gen_template and args.positions:
        print("[ERROR] --gen-template 与 --positions 不能同时使用", file=sys.stderr)
        return 2

    if args.gen_template:
        return cmd_gen_template(args)

    if not args.positions:
        print("[ERROR] 必须指定 --gen-template 或 --positions（用 --help 看用法）",
              file=sys.stderr)
        return 2

    return cmd_reconcile(args)


if __name__ == "__main__":
    raise SystemExit(main())
