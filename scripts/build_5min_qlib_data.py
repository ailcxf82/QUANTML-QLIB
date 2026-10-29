"""
5min 行情入库与校验（B3 分钟 Deep Factor 数据前置）
==============================================================
将 csv_data_5min/ 下的 per-symbol CSV 用 qlib-main 的 dump_bin.py 转为 qlib bin
（freq=5min），并做数据完整性校验（每日 ~48 bar、时间戳单调、价量非负）。

依赖 qlib-main/scripts/dump_bin.py（pip 版 qlib 不含该模块）。通过 --qlib-scripts
指定其所在目录（默认 D:/quant_project/qlib-main/scripts），脚本会将其加入 sys.path。

典型流程（在可联网机器采集后，或在本机已有 CSV 时）：
  # 1) 采集（联网机器）
  python python/data/baostock_5min_updater.py --start 2020-01-01 --end 2026-05-22
  # 2) 入库（本步骤）
  python scripts/build_5min_qlib_data.py --csv-dir D:/qlib_data/csv_data_5min \
      --qlib-dir D:/qlib_data/qlib_data_5min
  # 3) 仅校验已入库数据
  python scripts/build_5min_qlib_data.py --qlib-dir D:/qlib_data/qlib_data_5min --validate-only

注意：dump 全量 csi500 5min（数千万行）耗时较长，建议先用 baostock_5min_updater.py
的 --limit 做 1 个月冒烟，跑通 dump+校验后再全量。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import pandas as pd

EXPECTED_BARS_PER_DAY = 48


def _import_dump_bin(qlib_scripts: Path):
    """将 qlib-main/scripts 加入 sys.path 并导入 DumpDataAll。"""
    if not (qlib_scripts / "dump_bin.py").exists():
        raise FileNotFoundError(
            f"未找到 dump_bin.py: {qlib_scripts}。请用 --qlib-scripts 指定 qlib-main/scripts 目录。"
        )
    sys.path.insert(0, str(qlib_scripts))
    from dump_bin import DumpDataAll  # type: ignore[import-not-found]

    return DumpDataAll


def dump_csv_to_qlib(
    csv_dir: Path,
    qlib_dir: Path,
    qlib_scripts: Path,
    max_workers: int = 16,
    limit_nums: Optional[int] = None,
) -> None:
    """调用 qlib dump_bin DumpDataAll(freq=5min) 入库。"""
    DumpDataAll = _import_dump_bin(qlib_scripts)
    if not csv_dir.is_dir() or not any(csv_dir.glob("*.csv")):
        raise FileNotFoundError(f"CSV 目录为空或不存在: {csv_dir}")
    print(f"[dump] csv_dir={csv_dir} → qlib_dir={qlib_dir} (freq=5min)")
    DumpDataAll(
        data_path=str(csv_dir),
        qlib_dir=str(qlib_dir),
        freq="5min",
        max_workers=max_workers,
        date_field_name="date",
        symbol_field_name="symbol",
        exclude_fields="date,symbol",
        limit_nums=limit_nums,
    ).dump()
    print("[dump] done")


def validate(qlib_dir: Path, sample: int = 20) -> dict:
    """抽样校验 5min 库：日历存在、每日 bar 数接近 48、价量非负、时间单调。

    返回校验摘要 dict；发现严重问题时在 issues 中标注。
    """
    import qlib
    from qlib.data import D

    issues: List[str] = []

    cal_path = qlib_dir / "calendars" / "5min.txt"
    if not cal_path.exists():
        issues.append(f"缺少日历 {cal_path}")
        return {"ok": False, "issues": issues}

    cal = pd.read_csv(cal_path, header=None)[0]
    n_cal = len(cal)
    cal_ts = pd.to_datetime(cal)
    n_days = cal_ts.normalize().nunique()
    bars_per_day = round(n_cal / max(1, n_days), 2)

    qlib.init(provider_uri=str(qlib_dir), region="cn")
    inst_path = qlib_dir / "instruments" / "all.txt"
    insts = [ln.split("\t")[0] for ln in inst_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    sample_insts = insts[:sample]

    df = D.features(
        sample_insts,
        ["$open", "$high", "$low", "$close", "$volume"],
        freq="5min",
    )
    neg_price = bool((df[["$open", "$high", "$low", "$close"]] < 0).any().any())
    neg_vol = bool((df["$volume"] < 0).any())
    if neg_price:
        issues.append("存在负价格")
    if neg_vol:
        issues.append("存在负成交量")
    if not (40 <= bars_per_day <= 50):
        issues.append(f"每日 bar 数异常: {bars_per_day}（期望≈{EXPECTED_BARS_PER_DAY}）")

    summary = {
        "ok": len(issues) == 0,
        "n_calendar_points": n_cal,
        "n_trading_days": int(n_days),
        "bars_per_day": bars_per_day,
        "n_instruments": len(insts),
        "sampled": len(sample_insts),
        "neg_price": neg_price,
        "neg_volume": neg_vol,
        "date_range": f"{cal_ts.min()}~{cal_ts.max()}",
        "issues": issues,
    }
    return summary


def main() -> None:
    import json

    parser = argparse.ArgumentParser(description="5min CSV → qlib bin 入库与校验")
    parser.add_argument("--csv-dir", default="D:/qlib_data/csv_data_5min")
    parser.add_argument("--qlib-dir", default="D:/qlib_data/qlib_data_5min")
    parser.add_argument("--qlib-scripts", default="D:/quant_project/qlib-main/scripts")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--limit-nums", type=int, default=None, help="只 dump 前 N 个 CSV（冒烟）")
    parser.add_argument("--validate-only", action="store_true", help="跳过 dump，只校验已入库数据")
    parser.add_argument("--sample", type=int, default=20, help="校验抽样标的数")
    parser.add_argument(
        "--audit-out",
        default=None,
        help="可选：将校验摘要写入 JSON，例如 artifacts/_audit/5min_validate_dev.json",
    )
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    qlib_dir = Path(args.qlib_dir)
    qlib_scripts = Path(args.qlib_scripts)

    if not args.validate_only:
        dump_csv_to_qlib(
            csv_dir, qlib_dir, qlib_scripts,
            max_workers=args.max_workers, limit_nums=args.limit_nums,
        )

    summary = validate(qlib_dir, sample=args.sample)
    if args.audit_out:
        audit_path = Path(args.audit_out)
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"[audit] wrote {audit_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    if not summary["ok"]:
        print("[validate] 发现问题，请检查上面的 issues")
        sys.exit(2)
    print("[validate] PASS")


if __name__ == "__main__":
    main()
