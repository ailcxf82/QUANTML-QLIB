"""
qlib 数据指纹快照与对比工具
============================
层级位置: Data → 数据治理（与 Feature / Model 完全解耦）
用途:
  在每次 run_experiment.py 前后快照 qlib provider_uri 的关键文件状态，
  用于事后定位 "sharpe 在同 commit 下反复抖动" 是否由数据 dump 漂移引起。

设计原则:
  - 不依赖 qlib 库（避免环境耦合）；纯文件系统 + hashlib 实现
  - O(常数) 时间：只对 calendars / instruments / 采样若干股票做 md5
  - 不读取/修改 qlib 数据本身

子命令:
  snapshot                     生成当前快照
  diff <old.json> <new.json>   对比两份快照
  list                         列出已有快照

输出位置:
  artifacts/_data_fingerprints/<UTC时间戳>[_<label>].json

典型用法:
  python scripts/check_qlib_data_fingerprint.py snapshot --label before_daily_prod
  python python/run_experiment.py --model lgbm --freq daily_prod
  python scripts/check_qlib_data_fingerprint.py snapshot --label after_daily_prod
  python scripts/check_qlib_data_fingerprint.py diff \\
      artifacts/_data_fingerprints/<old>.json \\
      artifacts/_data_fingerprints/<new>.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]

# Windows PowerShell 默认编码为 cp936，强制 stdout/stderr 走 utf-8 以保证中文摘要可读
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
FINGERPRINTS_DIR = WORKSPACE_ROOT / "artifacts" / "_data_fingerprints"
DEFAULT_PROVIDER_URI = "D:/qlib_data/qlib_data"
KEY_FEATURE_FIELDS: Tuple[str, ...] = (
    "close_qfq.day.bin",
    "volume.day.bin",
    "turnover_rate_f.day.bin",
)
KEY_INSTRUMENTS_FILES: Tuple[str, ...] = ("all.txt", "csi500.txt", "csi300.txt")


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _md5_of_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """计算文件的 md5 摘要（流式读取，避免大文件占内存）。"""
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _file_stat(path: Path) -> Dict[str, Any]:
    """读取文件大小、mtime、md5。文件不存在时返回 exists=False。"""
    if not path.exists():
        return {"exists": False, "path": str(path)}
    stat = path.stat()
    return {
        "exists": True,
        "size": int(stat.st_size),
        "mtime_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%d %H:%M:%S"),
        "mtime_epoch": float(stat.st_mtime),
        "md5": _md5_of_file(path),
    }


def _read_lines(path: Path) -> List[str]:
    """读取文本文件并去除空行（utf-8）。"""
    if not path.exists():
        return []
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if ln.strip()
    ]


def _resolve_provider_uri(cli_value: Optional[str]) -> str:
    """provider_uri 解析优先级:CLI 参数 > configs/base.yaml > 默认常量。"""
    if cli_value:
        return cli_value
    base_yaml = WORKSPACE_ROOT / "configs" / "base.yaml"
    if yaml is not None and base_yaml.exists():
        try:
            cfg = yaml.safe_load(base_yaml.read_text(encoding="utf-8")) or {}
            uri = cfg.get("qlib_init", {}).get("provider_uri")
            if uri:
                return str(uri)
        except Exception:
            pass
    return DEFAULT_PROVIDER_URI


# ─────────────────────────────────────────────────────────────────────────────
# 快照采集
# ─────────────────────────────────────────────────────────────────────────────

def _collect_calendars(provider_root: Path) -> Dict[str, Any]:
    """采集 calendars/day.txt 的全部关键属性。"""
    day_path = provider_root / "calendars" / "day.txt"
    info = _file_stat(day_path)
    if not info.get("exists"):
        return info
    lines = _read_lines(day_path)
    info["n_days"] = len(lines)
    info["first_day"] = lines[0] if lines else None
    info["last_day"] = lines[-1] if lines else None
    info["tail_5"] = lines[-5:] if lines else []
    return info


def _collect_instruments(provider_root: Path) -> Dict[str, Any]:
    """采集 instruments/ 下关键 universe 文件。"""
    inst_dir = provider_root / "instruments"
    result: Dict[str, Any] = {"_dir_exists": inst_dir.is_dir()}
    if not inst_dir.is_dir():
        return result
    for name in KEY_INSTRUMENTS_FILES:
        path = inst_dir / name
        info = _file_stat(path)
        if info.get("exists"):
            lines = _read_lines(path)
            info["n_lines"] = len(lines)
            # 抽取唯一 instrument 数（lines 可能含起止日期列）
            insts = {ln.split()[0] for ln in lines if ln}
            info["n_unique_instruments"] = len(insts)
        result[name] = info
    return result


def _collect_features_summary(
    provider_root: Path,
    sample_size: int,
    csi500_instruments: List[str],
) -> Dict[str, Any]:
    """采集 features/ 目录元数据：总文件数、mtime 直方图、采样股票指纹。

    采样策略：取 csi500_instruments 字典序前 sample_size 只（确定性，便于复现对比）。
    若 csi500 列表为空，回退为按 features/ 目录字典序前 sample_size。
    """
    feat_dir = provider_root / "features"
    if not feat_dir.is_dir():
        return {"_dir_exists": False}

    # 全局统计：文件数 + mtime 范围
    mtime_buckets: Counter = Counter()
    n_files = 0
    mtime_min: Optional[float] = None
    mtime_max: Optional[float] = None
    for sub in feat_dir.iterdir():
        if not sub.is_dir():
            continue
        try:
            for f in sub.iterdir():
                if not f.is_file():
                    continue
                n_files += 1
                m = f.stat().st_mtime
                if mtime_min is None or m < mtime_min:
                    mtime_min = m
                if mtime_max is None or m > mtime_max:
                    mtime_max = m
                bucket = datetime.fromtimestamp(m, tz=timezone.utc).astimezone().strftime(
                    "%Y-%m-%d %H"
                )
                mtime_buckets[bucket] += 1
        except PermissionError:
            continue

    # 采样股票指纹（取 csi500 头部 N 只）
    sampled: List[str]
    if csi500_instruments:
        sampled = sorted(csi500_instruments)[:sample_size]
    else:
        sampled = sorted(p.name for p in feat_dir.iterdir() if p.is_dir())[:sample_size]

    sample_records: Dict[str, Dict[str, Any]] = {}
    for inst in sampled:
        stock_dir = feat_dir / inst.lower()
        if not stock_dir.is_dir():
            sample_records[inst] = {"exists": False}
            continue
        files: Dict[str, Any] = {}
        for fname in KEY_FEATURE_FIELDS:
            files[fname] = _file_stat(stock_dir / fname)
        sample_records[inst] = {"exists": True, "fields": files}

    top_buckets = [
        {"hour_utc_local": bucket, "n_files": cnt}
        for bucket, cnt in mtime_buckets.most_common(10)
    ]

    def _fmt_ts(m: Optional[float]) -> Optional[str]:
        if m is None:
            return None
        return (
            datetime.fromtimestamp(m, tz=timezone.utc)
            .astimezone()
            .strftime("%Y-%m-%d %H:%M:%S")
        )

    return {
        "_dir_exists": True,
        "n_files": n_files,
        "mtime_min": _fmt_ts(mtime_min),
        "mtime_max": _fmt_ts(mtime_max),
        "mtime_top_buckets": top_buckets,
        "n_stock_dirs": sum(1 for p in feat_dir.iterdir() if p.is_dir()),
        "sample_size": len(sampled),
        "sampled_instruments": list(sampled),
        "sample_records": sample_records,
    }


def collect_snapshot(provider_uri: str, label: Optional[str], sample_size: int) -> Dict[str, Any]:
    """生成完整快照字典。"""
    provider_root = Path(provider_uri)
    if not provider_root.exists():
        raise FileNotFoundError(f"provider_uri 不存在: {provider_root}")

    calendars = _collect_calendars(provider_root)
    instruments = _collect_instruments(provider_root)

    # 提取 csi500 instrument 列表用于稳定采样
    csi500_block = instruments.get("csi500.txt", {})
    csi500_path = provider_root / "instruments" / "csi500.txt"
    csi500_insts: List[str] = []
    if csi500_block.get("exists") and csi500_path.exists():
        csi500_insts = sorted({ln.split()[0] for ln in _read_lines(csi500_path) if ln})

    features = _collect_features_summary(
        provider_root, sample_size=sample_size, csi500_instruments=csi500_insts
    )

    return {
        "meta": {
            "version": "v1",
            "captured_at_iso": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %z"),
            "captured_at_epoch": datetime.now().timestamp(),
            "provider_uri": str(provider_root),
            "workspace_root": str(WORKSPACE_ROOT),
            "label": label or "",
        },
        "calendars": calendars,
        "instruments": instruments,
        "features": features,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 持久化
# ─────────────────────────────────────────────────────────────────────────────

def save_snapshot(snapshot: Dict[str, Any], label: Optional[str]) -> Path:
    """落盘到 artifacts/_data_fingerprints/<UTC时间戳>[_<label>].json。"""
    FINGERPRINTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    suffix = f"_{label}" if label else ""
    out_path = FINGERPRINTS_DIR / f"{ts}{suffix}.json"
    out_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def list_snapshots() -> List[Path]:
    """按 mtime 升序返回所有快照文件。"""
    if not FINGERPRINTS_DIR.exists():
        return []
    return sorted(FINGERPRINTS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)


# ─────────────────────────────────────────────────────────────────────────────
# 对比
# ─────────────────────────────────────────────────────────────────────────────

def _changed(a: Any, b: Any) -> bool:
    """安全等值比较（NaN / None / dict 都按值比较）。"""
    return a != b


def diff_snapshots(old: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """两份快照之间的差异摘要，按维度组织。"""
    report: Dict[str, Any] = {
        "old_label": old.get("meta", {}).get("label", ""),
        "new_label": new.get("meta", {}).get("label", ""),
        "old_captured_at": old.get("meta", {}).get("captured_at_iso"),
        "new_captured_at": new.get("meta", {}).get("captured_at_iso"),
        "changes": [],
    }
    changes: List[Dict[str, Any]] = report["changes"]

    # 1) calendars
    oc = old.get("calendars", {})
    nc = new.get("calendars", {})
    for key in ("md5", "size", "n_days", "last_day", "mtime_iso"):
        if _changed(oc.get(key), nc.get(key)):
            changes.append({
                "section": "calendars/day.txt",
                "field": key,
                "old": oc.get(key),
                "new": nc.get(key),
            })

    # 2) instruments
    oi = old.get("instruments", {})
    ni = new.get("instruments", {})
    for fname in KEY_INSTRUMENTS_FILES:
        of = oi.get(fname, {})
        nf = ni.get(fname, {})
        for key in ("md5", "size", "n_unique_instruments", "mtime_iso"):
            if _changed(of.get(key), nf.get(key)):
                changes.append({
                    "section": f"instruments/{fname}",
                    "field": key,
                    "old": of.get(key),
                    "new": nf.get(key),
                })

    # 3) features 总体
    of_ = old.get("features", {})
    nf_ = new.get("features", {})
    for key in ("n_files", "n_stock_dirs", "mtime_min", "mtime_max"):
        if _changed(of_.get(key), nf_.get(key)):
            changes.append({
                "section": "features (summary)",
                "field": key,
                "old": of_.get(key),
                "new": nf_.get(key),
            })

    # 4) features 采样股票级
    o_samples: Dict[str, Any] = (of_ or {}).get("sample_records", {})
    n_samples: Dict[str, Any] = (nf_ or {}).get("sample_records", {})
    all_insts = sorted(set(o_samples.keys()) | set(n_samples.keys()))
    for inst in all_insts:
        o_rec = o_samples.get(inst, {})
        n_rec = n_samples.get(inst, {})
        o_fields = o_rec.get("fields", {})
        n_fields = n_rec.get("fields", {})
        for fname in KEY_FEATURE_FIELDS:
            o_info = o_fields.get(fname, {})
            n_info = n_fields.get(fname, {})
            if _changed(o_info.get("md5"), n_info.get("md5")) or _changed(
                o_info.get("size"), n_info.get("size")
            ):
                changes.append({
                    "section": f"features/{inst}/{fname}",
                    "field": "md5/size",
                    "old": {"md5": o_info.get("md5"), "size": o_info.get("size")},
                    "new": {"md5": n_info.get("md5"), "size": n_info.get("size")},
                })

    report["n_changes"] = len(changes)
    return report


# ─────────────────────────────────────────────────────────────────────────────
# 打印辅助
# ─────────────────────────────────────────────────────────────────────────────

def _print_snapshot_summary(snapshot: Dict[str, Any], out_path: Optional[Path] = None) -> None:
    """快照核心字段一行摘要。"""
    meta = snapshot.get("meta", {})
    cal = snapshot.get("calendars", {})
    feats = snapshot.get("features", {})
    instruments = snapshot.get("instruments", {})
    csi500 = instruments.get("csi500.txt", {})
    print(f"\n{'='*68}")
    print(f"  qlib 数据指纹快照")
    print(f"{'='*68}")
    print(f"  采集时间       : {meta.get('captured_at_iso')}")
    print(f"  provider_uri   : {meta.get('provider_uri')}")
    if meta.get("label"):
        print(f"  label          : {meta.get('label')}")
    print(f"  ── calendars ─────────────────────────────────────────────")
    print(f"  day.txt md5    : {cal.get('md5')}")
    print(f"  日历范围       : {cal.get('first_day')} ~ {cal.get('last_day')}  ({cal.get('n_days')} 日)")
    print(f"  day.txt mtime  : {cal.get('mtime_iso')}")
    print(f"  ── instruments ──────────────────────────────────────────")
    print(f"  csi500 md5     : {csi500.get('md5')}")
    print(f"  csi500 unique  : {csi500.get('n_unique_instruments')}")
    print(f"  ── features ─────────────────────────────────────────────")
    print(f"  特征文件总数   : {feats.get('n_files')}")
    print(f"  股票目录总数   : {feats.get('n_stock_dirs')}")
    print(f"  mtime 最早     : {feats.get('mtime_min')}")
    print(f"  mtime 最新     : {feats.get('mtime_max')}")
    top = feats.get("mtime_top_buckets", [])[:5]
    if top:
        print(f"  mtime top 5 桶 :")
        for b in top:
            print(f"      {b['hour_utc_local']}  +{b['n_files']} 文件")
    if out_path is not None:
        print(f"\n  快照已写入     : {out_path}")
    print(f"{'='*68}\n")


def _print_diff_report(report: Dict[str, Any]) -> None:
    """对比报告人可读输出。"""
    print(f"\n{'='*68}")
    print(f"  qlib 数据指纹对比")
    print(f"{'='*68}")
    print(f"  old   : {report.get('old_captured_at')}  label={report.get('old_label') or '(无)'}")
    print(f"  new   : {report.get('new_captured_at')}  label={report.get('new_label') or '(无)'}")
    print(f"  变化数: {report.get('n_changes')}")
    print(f"{'─'*68}")
    if report.get("n_changes", 0) == 0:
        print("  ✓ 两份快照完全一致（数据无漂移）")
    else:
        for ch in report["changes"]:
            print(f"  [{ch['section']}] {ch['field']}")
            print(f"      old: {ch['old']}")
            print(f"      new: {ch['new']}")
    print(f"{'='*68}\n")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cmd_snapshot(args: argparse.Namespace) -> int:
    provider_uri = _resolve_provider_uri(args.provider_uri)
    snapshot = collect_snapshot(
        provider_uri=provider_uri,
        label=args.label,
        sample_size=int(args.sample_size),
    )
    out_path = save_snapshot(snapshot, label=args.label)
    _print_snapshot_summary(snapshot, out_path=out_path)
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    old_path = Path(args.old)
    new_path = Path(args.new)
    if not old_path.exists():
        print(f"ERROR: 快照文件不存在: {old_path}", file=sys.stderr)
        return 2
    if not new_path.exists():
        print(f"ERROR: 快照文件不存在: {new_path}", file=sys.stderr)
        return 2
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    report = diff_snapshots(old, new)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_diff_report(report)
    return 0 if report.get("n_changes", 0) == 0 else 1


def _cmd_list(_: argparse.Namespace) -> int:
    snaps = list_snapshots()
    if not snaps:
        print("(尚无快照)")
        return 0
    print(f"\n{'='*68}")
    print(f"  已有快照（{len(snaps)} 份） @ {FINGERPRINTS_DIR}")
    print(f"{'='*68}")
    for p in snaps:
        try:
            meta = json.loads(p.read_text(encoding="utf-8")).get("meta", {})
            print(
                f"  {p.name:<48} "
                f"[{meta.get('captured_at_iso', '?'):<25}]  "
                f"label={meta.get('label') or '-'}"
            )
        except Exception:
            print(f"  {p.name:<48} [读取失败]")
    print()
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_qlib_data_fingerprint",
        description="qlib 数据指纹快照与对比工具（独立运行，不依赖 qlib 库）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot", help="生成当前快照")
    s.add_argument("--provider-uri", default=None,
                   help="qlib 数据根目录（默认从 configs/base.yaml 读取）")
    s.add_argument("--label", default=None,
                   help="可选标签，将作为文件名后缀（如 before_daily_prod）")
    s.add_argument("--sample-size", type=int, default=20,
                   help="采样股票数量（取 csi500 字典序前 N 只，默认 20）")
    s.set_defaults(func=_cmd_snapshot)

    d = sub.add_parser("diff", help="对比两份快照")
    d.add_argument("old", help="较早的快照路径")
    d.add_argument("new", help="较新的快照路径")
    d.add_argument("--json", action="store_true", help="按 JSON 输出（便于程序消费）")
    d.set_defaults(func=_cmd_diff)

    l = sub.add_parser("list", help="列出已有快照")
    l.set_defaults(func=_cmd_list)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
