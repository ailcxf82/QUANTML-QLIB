"""
诊断脚本：对最近的实盘 predictions 做 T+1 / T+5 后验，定位回撤根因。

层级位置：Data / Signal 诊断（只读，零侧效）
用途：
  1) 逐日 T+1 真实收益 vs CSI500，判断模型选股是否真的不赚钱
  2) 按 score 分位看头部 / 中部 / 尾部命中差异
  3) 输出每日持仓个股的真实涨跌幅，便于人工核对
  4) 拉新模型（train_end=2025-10-24）今天的 pred_score 截面分布

用法（qlib_zhengshi 环境）：
  python scripts/diagnose_recent_predictions.py
  python scripts/diagnose_recent_predictions.py --since 2026-05-13
  python scripts/diagnose_recent_predictions.py --topn 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = WORKSPACE_ROOT / "predictions"
LIVE_PROVIDER = "D:/qlib_data/qlib_data"


def _init_qlib(provider_uri: str = LIVE_PROVIDER) -> None:
    import qlib
    qlib.init(
        provider_uri=provider_uri,
        region="cn",
        expression_cache=None,
        dataset_cache=None,
    )


def _load_selections(since: str | None) -> pd.DataFrame:
    rows = []
    for sub in sorted(PRED_DIR.iterdir()):
        if not sub.is_dir():
            continue
        if since and sub.name < since:
            continue
        csv = sub / "selection.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv, dtype={"instrument": str})
        df["pred_date"] = pd.Timestamp(sub.name).date()
        rows.append(df)
    if not rows:
        raise SystemExit("no selections found")
    return pd.concat(rows, ignore_index=True)


def _next_n_trade_days(d, n: int, cal_map, cal) -> list:
    i = cal_map.get(d)
    if i is None:
        return []
    return [pd.Timestamp(cal[j]).date() for j in range(i + 1, min(i + 1 + n, len(cal)))]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--since", default="2026-05-13", help="只看 pred_date >= 此日期")
    p.add_argument("--topn", type=int, default=5, help="只看 rank <= topn 的持仓")
    p.add_argument("--horizons", default="1,3,5", help="后验 horizon 列表，逗号分隔")
    args = p.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]

    _init_qlib()
    from qlib.data import D

    cal = list(D.calendar(freq="day"))
    cal_map = {pd.Timestamp(d).date(): i for i, d in enumerate(cal)}
    print(f"qlib 日历末日: {pd.Timestamp(cal[-1]).date()}")

    raw = _load_selections(args.since)
    print(f"已加载 selection: {len(raw)} 行，{raw['pred_date'].nunique()} 个交易日")

    # 标识"持仓行"（rank != 999）
    raw["rank_int"] = pd.to_numeric(raw["rank"], errors="coerce")
    held = raw[raw["rank_int"] <= args.topn].copy()
    print(f"  其中 rank <= topn={args.topn} 持仓行: {len(held)}")

    # 拉全部相关股票 + benchmark 收盘价
    # CSI500 在 qlib_data 里以 "000905.SZ" 编码（与 base.yaml 一致）
    bench = "000905.SZ"
    insts = sorted(held["instrument"].unique().tolist()) + [bench]
    dmin = pd.Timestamp(held["pred_date"].min()) - pd.Timedelta(days=10)
    dmax = pd.Timestamp(held["pred_date"].max()) + pd.Timedelta(days=20)
    print(f"  D.features: {len(insts)} 标的, {dmin.date()} ~ {dmax.date()}")
    px = D.features(insts, ["$close_qfq"], start_time=dmin, end_time=dmax, freq="day")
    if px.empty:
        print("[ERROR] D.features 返回空")
        return 2
    px = px.reset_index().rename(columns={"$close_qfq": "close"})
    px["datetime"] = pd.to_datetime(px["datetime"]).dt.date
    pivot = px.pivot(index="datetime", columns="instrument", values="close").sort_index()

    # 计算各 horizon 后验收益
    def fwd_ret(inst, d0, n):
        ds = _next_n_trade_days(d0, n, cal_map, cal)
        if len(ds) < n:
            return None
        d_end = ds[-1]
        if inst not in pivot.columns:
            return None
        if d0 not in pivot.index or d_end not in pivot.index:
            return None
        c0 = pivot.at[d0, inst]
        c1 = pivot.at[d_end, inst]
        if pd.isna(c0) or pd.isna(c1) or c0 <= 0:
            return None
        return float(c1 / c0 - 1.0)

    for h in horizons:
        held[f"ret_{h}d"] = held.apply(
            lambda r: fwd_ret(r["instrument"], r["pred_date"], h), axis=1,
        )
        held[f"bench_{h}d"] = held["pred_date"].apply(
            lambda d: fwd_ret(bench, d, h),
        )
        held[f"excess_{h}d"] = held[f"ret_{h}d"] - held[f"bench_{h}d"]

    # 逐日聚合
    print("\n" + "=" * 78)
    print("逐日持仓 T+H 等权组合收益 (rank<={})".format(args.topn))
    print("=" * 78)
    daily = held.groupby("pred_date").agg(
        n_held=("instrument", "count"),
        avg_score=("score", "mean"),
        **{f"port_{h}d": (f"ret_{h}d", "mean") for h in horizons},
        **{f"bench_{h}d": (f"bench_{h}d", "first") for h in horizons},
        **{f"excess_{h}d": (f"excess_{h}d", "mean") for h in horizons},
    ).reset_index()

    pct_cols = [c for c in daily.columns if c.startswith(("port_", "bench_", "excess_"))]
    for c in pct_cols:
        daily[c] = daily[c].apply(lambda v: f"{v:+.2%}" if pd.notna(v) else "  NA  ")
    daily["avg_score"] = daily["avg_score"].apply(lambda v: f"{v:.4f}")
    print(daily.to_string(index=False))

    # 累计
    print("\n" + "=" * 78)
    print("累计后验（按交易日合成日 PnL，等权）")
    print("=" * 78)
    summary_rows = []
    for h in horizons:
        col = f"ret_{h}d"
        bcol = f"bench_{h}d"
        # 仅 H=1 时直接累乘（其他 H 重叠不能简单累乘，只算"次次独立"近似）
        valid = held.dropna(subset=[col, bcol])
        if valid.empty:
            continue
        if h == 1:
            daily_v = valid.groupby("pred_date").agg(p=(col, "mean"), b=(bcol, "first"))
            cum_p = (1 + daily_v["p"]).prod() - 1
            cum_b = (1 + daily_v["b"]).prod() - 1
            cum_e = (1 + cum_p) / (1 + cum_b) - 1
            wr_b = (daily_v["p"] > daily_v["b"]).mean()
            wr_0 = (daily_v["p"] > 0).mean()
            n_days = len(daily_v)
            summary_rows.append([
                f"T+{h}", n_days,
                f"{cum_p:+.2%}", f"{cum_b:+.2%}", f"{cum_e:+.2%}",
                f"{wr_b:.1%}", f"{wr_0:.1%}",
            ])
        else:
            # 横截面平均（每只单独 H 日收益的平均），便于看趋势
            mean_p = valid[col].mean()
            mean_b = valid[bcol].mean()
            mean_e = valid[f"excess_{h}d"].mean()
            wr_b = (valid[col] > valid[bcol]).mean()
            wr_0 = (valid[col] > 0).mean()
            summary_rows.append([
                f"T+{h}(avg)", len(valid),
                f"{mean_p:+.2%}", f"{mean_b:+.2%}", f"{mean_e:+.2%}",
                f"{wr_b:.1%}", f"{wr_0:.1%}",
            ])
    print(pd.DataFrame(
        summary_rows,
        columns=["horizon", "n", "port", "bench", "excess", "win>bench", "win>0"],
    ).to_string(index=False))

    # ── 实盘净值近似（考虑 MarketTimer + 换手成本） ─────────────────────────
    print("\n" + "=" * 78)
    print("实盘账户净值模拟（每日 selection 严格执行，考虑 risk_factor + 双边成本）")
    print("=" * 78)
    # 重新加载完整 selection（含 actual_weight / risk_factor / change）
    full = _load_selections(args.since)
    # 注：actual_weight 只在 5/15 起出现；之前默认 suggested_weight × 1.0
    full["rank_int"] = pd.to_numeric(full["rank"], errors="coerce")
    full["actual_weight"] = pd.to_numeric(full.get("actual_weight", 0), errors="coerce")
    full["suggested_weight"] = pd.to_numeric(full.get("suggested_weight", 0), errors="coerce")
    full["risk_factor"] = pd.to_numeric(full.get("risk_factor", 1.0), errors="coerce").fillna(1.0)
    # 若无 actual_weight 列，退化为 suggested_weight
    full["w_actual"] = full["actual_weight"].where(
        full["actual_weight"] > 0, full["suggested_weight"],
    )

    cost_one_side = 0.0030  # 单边 0.30%（5BP 佣金 + 10BP 印花税(单卖) + 10BP 滑点近似）
    # 简化按双边 0.30% × 2 = 0.60% 算（卖出 0.35% + 买入 0.25% ≈ 0.60%）
    rows_net = []
    for d, g in full.groupby("pred_date"):
        held_d = g[g["rank_int"] <= args.topn].copy()
        if held_d.empty:
            continue
        # 拉这一日所有持仓的 T+1 收益
        def _r(inst):
            return fwd_ret(inst, d, 1)
        held_d["r1"] = held_d["instrument"].apply(_r)
        held_d = held_d.dropna(subset=["r1"])
        if held_d.empty:
            continue
        # 持仓段毛收益 = sum(w_i * r_i) ；剩余 (1 - sum_w) 视为现金（0 收益）
        gross = float((held_d["w_actual"] * held_d["r1"]).sum())
        invested = float(held_d["w_actual"].sum())
        rf = float(held_d["risk_factor"].iloc[0]) if "risk_factor" in held_d else 1.0
        # 换手：reduce 行的 sum |w_change|（用 |trade_value|/total_value 估计）
        if "trade_value" in g.columns:
            turn_abs = pd.to_numeric(g["trade_value"], errors="coerce").abs().fillna(0).sum()
            total_v = (pd.to_numeric(g["target_value"], errors="coerce").fillna(0).sum() +
                       1e-9)
            # 用账户资金估，account=20 万（与 daily_live.yaml 一致）
            account = 200000.0
            turnover = float(turn_abs / account)
        else:
            turnover = invested  # 退化
        cost = turnover * cost_one_side  # 双边均摊给两侧，已含买卖
        net = gross - cost
        rows_net.append({
            "pred_date": d, "n_held": len(held_d),
            "risk_factor": rf, "invested_w": invested,
            "gross_ret": gross, "turnover": turnover, "cost": cost, "net_ret": net,
        })

    if rows_net:
        net_df = pd.DataFrame(rows_net)
        net_df["cum_gross"] = (1 + net_df["gross_ret"]).cumprod() - 1
        net_df["cum_net"] = (1 + net_df["net_ret"]).cumprod() - 1
        fmt_cols = ["risk_factor", "invested_w", "gross_ret", "turnover",
                    "cost", "net_ret", "cum_gross", "cum_net"]
        for c in fmt_cols:
            net_df[c] = net_df[c].apply(lambda v: f"{v:+.2%}" if pd.notna(v) else "NA")
        print(net_df.to_string(index=False))

        # 全程汇总
        raw = pd.DataFrame(rows_net)
        gross_cum = (1 + raw["gross_ret"]).prod() - 1
        net_cum = (1 + raw["net_ret"]).prod() - 1
        cost_cum = (1 + raw["cost"]).prod() - 1
        print(f"\n  累计毛收益: {gross_cum:+.2%}")
        print(f"  累计净收益: {net_cum:+.2%}  (扣除单边 {cost_one_side*100:.2f}% × turnover)")
        print(f"  累计成本损耗: ≈ {cost_cum:.2%}")
        print(f"  期间 CSI500: ~ {(daily['bench_1d'].apply(lambda s: float(s.strip('%+'))/100 if isinstance(s,str) and '%' in s else 0).add(1).prod()-1):+.2%}")

    # 个股明细（最近 5 天）
    print("\n" + "=" * 78)
    print("个股 T+1 明细（最近 5 个 pred_date）")
    print("=" * 78)
    last5 = sorted(held["pred_date"].unique())[-5:]
    detail = held[held["pred_date"].isin(last5)].copy()
    show_cols = ["pred_date", "instrument", "score", "rank_int", "ret_1d", "bench_1d", "excess_1d"]
    if all(c in detail.columns for c in show_cols):
        detail = detail.sort_values(["pred_date", "rank_int"])
        for c in ["ret_1d", "bench_1d", "excess_1d"]:
            detail[c] = detail[c].apply(lambda v: f"{v:+.2%}" if pd.notna(v) else "  NA  ")
        detail["score"] = detail["score"].apply(lambda v: f"{v:.4f}" if pd.notna(v) else "NA")
        print(detail[show_cols].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
