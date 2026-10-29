"""Recalculate costs from saved daily results only; never train, predict or backtest.

Legacy exports omitted Qlib's cost rate and account columns. Recover the account
scale from turnover = daily traded value / previous account value, then use
account[t] = account[t-1] * (1 + gross_return[t]) - cash_cost[t].
Reconcile turnover on EVERY date before accepting the reconstructed cost rates.
This is an accounting reconciliation of existing trades, not a new simulation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import html
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from backtest.analysis_layer import AnalysisLayer
from backtest.portfolio_layer import PortfolioLayer


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct_costs(report: pd.DataFrame, trades: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Restore legacy daily cost rates, rejecting incomplete/inconsistent trades."""
    if report.empty or not isinstance(report.index, pd.DatetimeIndex):
        raise ValueError("A nonempty daily DatetimeIndex report is required")
    if report.index.has_duplicates or not report.index.is_monotonic_increasing:
        raise ValueError("Report dates must be unique and ordered")
    if not report.index.equals(report.index.normalize()):
        raise ValueError("Only daily reports are supported")
    for column in ("return", "turnover", "bench"):
        if not np.isfinite(report[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid report {column}")
    if (report.turnover < 0).any():
        raise ValueError("Negative turnover")
    if trades.empty:
        raise ValueError("No trade evidence; cannot assume costs are zero")
    trades = trades.copy()
    trades["datetime"] = pd.to_datetime(trades["datetime"], errors="raise")
    if trades.datetime.isna().any() or not trades.datetime.isin(report.index).all():
        raise ValueError("Trade dates must all belong to the report")
    for column in ("value", "cost"):
        values = trades[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid trade {column}")
    daily = trades.groupby("datetime")[["value", "cost"]].sum().reindex(report.index, fill_value=0)
    active = report.index[report.turnover > 0]
    if active.empty:
        raise ValueError("No positive turnover to recover account scale")
    first = active[0]
    before = report.index < first
    if (daily.loc[before] != 0).any().any():
        raise ValueError("Trades or costs before the first positive turnover")
    growth_before_first = float((1 + report.loc[before, "return"]).prod())
    initial = float(daily.loc[first, "value"] / report.loc[first, "turnover"] / growth_before_first)
    if not np.isfinite(initial) or initial <= 0:
        raise ValueError("Invalid reconstructed initial account")
    equity = initial
    rows = []
    for date, row in report.iterrows():
        value, cash_cost = daily.loc[date, ["value", "cost"]]
        implied_turnover = value / equity
        if not np.isclose(implied_turnover, row.turnover, rtol=1e-8, atol=1e-10):
            raise ValueError(f"Trade/account turnover reconciliation failed on {date.date()}")
        cost_rate = float(cash_cost / equity)
        end_equity = equity * (1 + row['return']) - cash_cost
        if not np.isfinite(end_equity) or end_equity <= 0:
            raise ValueError(f"Invalid reconstructed account on {date.date()}")
        rows.append((equity, end_equity, cash_cost, cost_rate, abs(implied_turnover - row.turnover)))
        equity = end_equity
    audit = pd.DataFrame(rows, index=report.index, columns=[
        "previous_account", "account", "cash_cost", "cost", "turnover_abs_error",
    ])
    # If a raw Qlib report is available, check both independent fields too.
    for column in ("cost", "account"):
        if column in report and not np.allclose(report[column], audit[column], rtol=1e-8, atol=1e-8):
            raise ValueError(f"Reconstruction disagrees with stored {column}")
    restored = report.copy()
    for column in audit:
        restored[column] = audit[column]
    restored = PortfolioLayer().to_dataframe(PortfolioLayer().parse({"1day": restored}))
    return restored, {
        "initial_account_inferred": initial,
        "ending_account_reconstructed": equity,
        "cash_cost_sum": float(audit.cash_cost.sum()),
        "max_turnover_abs_error": float(audit.turnover_abs_error.max()),
        "turnover_reconciled_days": len(report),
        "method": "saved_gross_return_and_trade_costs_reconciled_against_daily_turnover",
    }


def metrics(report: pd.DataFrame, *, gross: bool = False) -> dict:
    frame = report.assign(cost=0.0) if gross else report
    state = PortfolioLayer().parse({"1day": frame})
    return asdict(AnalysisLayer().compute_metrics(state))


def inventory(artifacts: Path) -> list[dict]:
    rows = []
    for path in sorted(artifacts.glob("**/backtest_report.parquet")):
        frame = pd.read_parquet(path)
        rows.append({
            "path": str(path.resolve()), "sha256": sha256(path), "rows": len(frame),
            "start": str(frame.index.min()), "end": str(frame.index.max()),
            "has_cost": "cost" in frame,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True)
    parser.add_argument("--requested-end", required=True)
    parser.add_argument("--calendar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    requested_end = pd.Timestamp(args.requested_end)
    calendar = pd.DatetimeIndex(pd.to_datetime(args.calendar.read_text().splitlines()))
    runs = [Path(p).resolve() for p in args.run_dir]
    output = args.output.resolve()
    if output.exists() or any(output == p or p in output.parents for p in runs):
        raise ValueError("Use a fresh output directory outside the source runs")
    sources = [args.calendar.resolve()]
    for run in runs:
        sources.extend(p for p in run.iterdir() if p.is_file())
    # Verify all live model pointers, too; this audit must not promote a model.
    sources.extend(p for p in (ROOT / "artifacts/models").glob("latest_*") if p.is_file())
    hashes = {str(p): sha256(p) for p in sources}
    coverage = inventory(ROOT / "artifacts")
    results, comparisons, daily_results = [], [], []
    for run in runs:
        original = pd.read_parquet(run / "backtest_report.parquet")
        trades = pd.read_parquet(run / "trades.parquet")
        restored, reconciliation = reconstruct_costs(original, trades)
        # Restore using the complete source first; never invent dates or trades.
        restored = restored.loc[restored.index <= requested_end]
        if restored.empty:
            raise ValueError(f"No evidence at or before requested end for {run.name}")
        gross, net = metrics(restored, gross=True), metrics(restored)
        saved = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        # Detect unexpected historical metric definitions instead of silently comparing.
        full_gross = metrics(original.assign(cost=0.0), gross=True)
        for key in ("total_return", "annualized_return", "sharpe_ratio", "max_drawdown"):
            if not np.isclose(full_gross[key], saved["metrics"][key], rtol=1e-8, atol=1e-10):
                raise ValueError(f"Saved gross metric mismatch: {run.name}/{key}")
        expected = calendar[(calendar >= restored.index.min()) & (calendar <= requested_end)]
        missing = expected.difference(restored.index)
        result = {
            "run_id": run.name, "source_run": str(run),
            "requested_end": str(requested_end.date()),
            "actual_start": str(restored.index.min().date()),
            "actual_end": str(restored.index.max().date()), "n_days": len(restored),
            "coverage_status": "complete" if missing.empty and restored.index.max() >= requested_end else "partial",
            "missing_sessions": [str(d.date()) for d in missing],
            "missing_within_covered_range": [str(d.date()) for d in missing if d <= restored.index.max()],
            "missing_after_end": [str(d.date()) for d in missing if d > restored.index.max()],
            "saved_dates_absent_from_calendar": [str(d.date()) for d in restored.index.difference(calendar)],
            "gross": gross, "net": net, "reconciliation": reconciliation,
            "source_signal_metrics": {k: v for k, v in saved['metrics'].items() if k.startswith('ic')},
        }
        results.append(result)
        daily_results.append((run.name, restored))
        for basis, values in (("gross", gross), ("net", net)):
            comparisons.append({"run_id": run.name, "basis": basis, **values})
    changed = [p for p, digest in hashes.items() if sha256(Path(p)) != digest]
    changed += [row['path'] for row in coverage if sha256(Path(row['path'])) != row['sha256']]
    if changed:
        raise RuntimeError(f"Source files changed during audit: {changed}")
    summary = {
        "operation": "saved_results_accounting_only", "training_runs": 0, "backtest_runs": 0,
        "requested_end": str(requested_end.date()),
        "calendar_end": str(calendar.max().date()),
        "source_hashes_unchanged": True, "source_sha256": hashes,
        "report_inventory": coverage, "results": results,
        "notes": [
            "Gross figures are the existing Qlib gross-return series, not a new zero-cost simulation.",
            "Costs come from saved trades, not today's fee configuration; no additional slippage is invented.",
            "Missing dates are not padded with zero returns and are excluded from all metrics.",
            "Original artifacts, models and daily predictions are preserved.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    for name, frame in daily_results:
        frame.to_parquet(output / f"{name}_daily_cost_audit.parquet")
    (output / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame(comparisons).to_csv(output / "comparison.csv", index=False, encoding="utf-8-sig")
    table_rows = []
    for result in results:
        for basis in ("gross", "net"):
            values = result[basis]
            table_rows.append(
                f"<tr><td>{html.escape(result['run_id'])}</td><td>{'扣费前' if basis == 'gross' else '扣费后'}</td>"
                f"<td>{values['total_return']:.2%}</td><td>{values['annualized_return']:.2%}</td>"
                f"<td>{values['sharpe_ratio']:.3f}</td><td>{values['max_drawdown']:.2%}</td></tr>"
            )
    ranges = "；".join(f"{html.escape(r['run_id'])}: {r['actual_start']} 至 {r['actual_end']}（{r['n_days']}日，缺{len(r['missing_sessions'])}日）" for r in results)
    gaps = "；".join(
        f"{html.escape(r['run_id'])}: 已有区间内与当前日历相比缺 {', '.join(r['missing_within_covered_range']) or '无'}；"
        f"已有末日之后至目标日还缺 {len(r['missing_after_end'])} 个交易日"
        for r in results
    )
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>已有回测扣费核对</title>
<style>body{{font:16px system-ui;max-width:1250px;margin:40px auto;padding:20px;color:#182736}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;text-align:right;border-bottom:1px solid #ddd}}td:first-child{{text-align:left}}p{{line-height:1.7}}.note{{background:#fff2d5;padding:18px}}</style>
<h1>已有回测扣费核对</h1><p class="note">目标截至 {requested_end.date()}；已有结果覆盖不足，不补造收益。训练0次，回测0次。</p>
<p>{ranges}</p><p>{gaps}。指标均按各实验现存日记录统计，缺失日期没有补值。</p><table><thead><tr><th>实验</th><th>口径</th><th>累计收益</th><th>年化收益</th><th>夏普</th><th>最大回撤</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table>
<p>扣费后日收益 = 原毛收益 − 当日成交费用 / 前日账户余额。账户余额由保存的成交金额与换手率确定，并逐日递推核对。费用已采用原实验记录，不再额外添加假设滑点。</p>
<p>“扣费前”是原报告口径，不是重新运行的零成本策略。费用也会影响原实验持仓规模，因此不能用“毛收益金额减累计费用”代替逐日扣费计算。</p>
<p>原始结果与模型哈希已核对未变。细节：<a href="audit.json">审计记录</a> · <a href="comparison.csv">指标表</a></p></html>"""
    (output / "report.html").write_text(page, encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": len(results), "reports_scanned": len(coverage),
                      "latest_saved_report_end": max(r['end'] for r in coverage),
                      "net": {r['run_id']: r['net'] for r in results}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
