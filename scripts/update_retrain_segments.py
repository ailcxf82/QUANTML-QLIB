"""
按当前日期推进训练/验证/测试切分（dataset_segments）的助手脚本。
==================================================================

层级位置：脚本工具（与 Data/Feature/Model 解耦），用于 monthly_retrain 链路。

背景：
  原 ``configs/freq/daily_retrain.yaml`` 中 ``dataset_segments`` 三段日期硬编码
  为 ``train=[2020-01-01,2023-12-31]``，导致 ``monthly_retrain.ps1`` 每次"重训"
  实际只是用同一段历史重新拟合（换种子），训练数据不会随时间推进。

本脚本目的：
  在 monthly_retrain 链路中，按"当前数据末日"自动重写 ``daily_retrain.yaml``
  的 train/valid/test 三段日期，并同步 ``base.yaml`` 的两处 ``end_time``，
  让重训后的 ``latest_lgbm_optuna_daily.pkl`` 真正用上最新数据。

切分算法（默认）：
  data_end:    --data-end 参数（默认今天）
  test_end:    data_end
  test_start:  data_end 所在月的第 1 天
  valid_end:   test_start - 1 天
  valid_start: valid_end 往前推 5 个月，再取月初（共 6 个月 valid）
  train_end:   valid_start - 8 天（purge 2 + embargo 5 + 1 缓冲，防 label 泄漏）
  train_start: 默认沿用 ``daily_retrain.yaml`` 原值（一般 "2020-01-01"）

示例（data_end = 2026-05-25）：
  train: [2020-01-01, 2025-10-24]
  valid: [2025-11-01, 2026-04-30]
  test:  [2026-05-01, 2026-05-25]

设计要点：
  - 使用纯文本正则替换（不走 PyYAML），完整保留中文注释、键顺序、缩进
  - 修改前自动备份原文件到 ``configs/_archive/<name>_pre_<run_ts>.yaml``
  - 支持 ``--dry-run`` 仅打印计划，不写文件
  - 失败处理：找不到匹配行时抛 ``RuntimeError``，monthly_retrain 链路会终止

CLI 用法：
  python scripts/update_retrain_segments.py                  # 用今天作 data_end
  python scripts/update_retrain_segments.py --data-end 2026-05-25
  python scripts/update_retrain_segments.py --dry-run        # 只打印不写
  python scripts/update_retrain_segments.py --no-update-base # 跳过 base.yaml
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

# Windows PowerShell 默认控制台用 GBK 解码，强制 stdout/stderr 走 UTF-8 避免中文乱码
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass

try:
    from dateutil.relativedelta import relativedelta
except ImportError as exc:  # pragma: no cover - 防御式
    raise ImportError(
        "需要 python-dateutil（pandas 依赖，应已存在）。"
        "请在 qlib_zhengshi 环境中确认安装。"
    ) from exc


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DAILY_RETRAIN_YAML = WORKSPACE_ROOT / "configs" / "freq" / "daily_retrain.yaml"
BASE_YAML = WORKSPACE_ROOT / "configs" / "base.yaml"
ARCHIVE_DIR = WORKSPACE_ROOT / "configs" / "_archive"


# ─────────────────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Segments:
    """train/valid/test 三段时间切分。"""
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str

    def summary_lines(self) -> List[str]:
        return [
            f"train: [{self.train_start}, {self.train_end}]",
            f"valid: [{self.valid_start}, {self.valid_end}]",
            f"test:  [{self.test_start}, {self.test_end}]",
        ]


# ─────────────────────────────────────────────────────────────────────────────
# 切分算法
# ─────────────────────────────────────────────────────────────────────────────

def compute_segments(data_end: date, train_start: str = "2020-01-01") -> Segments:
    """
    按数据末日反向推算 train/valid/test 三段。

    Args:
        data_end:    数据可用截止日（含），通常为今天
        train_start: 训练集起始日（默认 "2020-01-01"，沿用原配置）

    Returns:
        Segments（六个 "YYYY-MM-DD" 字符串）

    Raises:
        ValueError: 推算出的 train_end 早于 train_start，说明 data_end 太早不足以训练
    """
    # test：data_end 所在月（保证至少有 1 天 test 数据用于 sanity 评估）
    test_end = data_end
    test_start = data_end.replace(day=1)

    # valid：6 个月，月初对齐，与 test 段不重叠
    valid_end = test_start - timedelta(days=1)
    valid_start = (valid_end - relativedelta(months=5)).replace(day=1)

    # train：[train_start, valid_start - 8 天]
    # 8 天 = purge 2 + embargo 5 + 1 缓冲（与 expanding_halfyearly.yaml 一致）
    train_end_dt = valid_start - timedelta(days=8)

    train_start_dt = datetime.strptime(train_start, "%Y-%m-%d").date()
    if train_end_dt <= train_start_dt:
        raise ValueError(
            f"data_end={data_end} 太早：推算 train_end={train_end_dt} "
            f"≤ train_start={train_start_dt}，无法构成有效训练集"
        )

    return Segments(
        train_start=train_start,
        train_end=train_end_dt.isoformat(),
        valid_start=valid_start.isoformat(),
        valid_end=valid_end.isoformat(),
        test_start=test_start.isoformat(),
        test_end=test_end.isoformat(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# YAML 行级替换（保留注释）
# ─────────────────────────────────────────────────────────────────────────────

_SEG_LINE_RE_TMPL = (
    r'^(?P<prefix>\s*{key}:\s*)\[\s*"[^"]+"\s*,\s*"[^"]+"\s*\]'
    r'(?P<suffix>.*)$'
)


def _replace_segment_line(text: str, key: str, start: str, end: str) -> Tuple[str, str]:
    """
    替换 yaml 中形如 ``  train: ["2020-01-01", "2023-12-31"]`` 的行。

    Returns:
        (new_text, old_line)。匹配不到时抛 RuntimeError。
    """
    pattern = re.compile(_SEG_LINE_RE_TMPL.format(key=key), re.MULTILINE)
    matches = list(pattern.finditer(text))
    if not matches:
        raise RuntimeError(f'未找到 "{key}:" 段（带 ["...", "..."]）')
    if len(matches) > 1:
        raise RuntimeError(
            f'找到多处 "{key}:" 段（{len(matches)} 处），无法精确替换。'
            "请检查 yaml 是否结构异常。"
        )
    m = matches[0]
    old_line = text[m.start(): m.end()]
    new_line = f'{m.group("prefix")}["{start}", "{end}"]{m.group("suffix")}'
    new_text = text[: m.start()] + new_line + text[m.end():]
    return new_text, old_line


def update_retrain_yaml(
    yaml_path: Path,
    seg: Segments,
    dry_run: bool = False,
    keep_original_train_start: bool = True,
) -> List[str]:
    """改写 daily_retrain.yaml 的 dataset_segments。

    Args:
        yaml_path:                  daily_retrain.yaml 路径
        seg:                        compute_segments 输出
        dry_run:                    True = 仅返回变更摘要，不写文件
        keep_original_train_start:  True 时若原 yaml 中 train_start 与 seg 不同，
                                    以原 yaml 为准（保护使用者手动设定的训练起点）

    Returns:
        变更摘要（每行为可读字符串）。
    """
    text = yaml_path.read_text(encoding="utf-8")
    diffs: List[str] = []

    # 如保留原 train_start，则先嗅探出原 yaml 里的 train_start
    final_train_start = seg.train_start
    if keep_original_train_start:
        m = re.search(
            r'^\s*train:\s*\[\s*"(?P<ts>[^"]+)"\s*,',
            text,
            re.MULTILINE,
        )
        if m:
            existing_ts = m.group("ts")
            if existing_ts != seg.train_start:
                diffs.append(
                    f"  [INFO] 保留原 train_start={existing_ts}（脚本计算值 "
                    f"{seg.train_start} 被覆盖）"
                )
            final_train_start = existing_ts

    new_text, train_old = _replace_segment_line(
        text, "train", final_train_start, seg.train_end
    )
    new_text, valid_old = _replace_segment_line(
        new_text, "valid", seg.valid_start, seg.valid_end
    )
    new_text, test_old = _replace_segment_line(
        new_text, "test", seg.test_start, seg.test_end
    )

    diffs.append(f"  - {train_old.strip()}")
    diffs.append(f"  + train: [\"{final_train_start}\", \"{seg.train_end}\"]")
    diffs.append(f"  - {valid_old.strip()}")
    diffs.append(f"  + valid: [\"{seg.valid_start}\", \"{seg.valid_end}\"]")
    diffs.append(f"  - {test_old.strip()}")
    diffs.append(f"  + test:  [\"{seg.test_start}\", \"{seg.test_end}\"]")

    if not dry_run:
        yaml_path.write_text(new_text, encoding="utf-8")

    return diffs


def update_base_end_time(
    yaml_path: Path,
    new_end: str,
    dry_run: bool = False,
) -> List[str]:
    """把 base.yaml 中所有 ``end_time: "..."`` 行替换为新值。

    base.yaml 通常包含两处：experiment.end_time 与 dataset.kwargs.handler.kwargs.end_time，
    两处需保持一致（否则数据加载会截断在更小者）。
    """
    text = yaml_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r'^(?P<prefix>\s*end_time:\s*)"(?P<old>[^"]+)"(?P<suffix>.*)$',
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        raise RuntimeError(
            "未在 base.yaml 中找到任何 end_time 字段，配置结构可能已变化"
        )

    diffs: List[str] = []
    for m in matches:
        if m.group("old") != new_end:
            diffs.append(
                f"  - end_time: \"{m.group('old')}\" (line {text.count(chr(10), 0, m.start()) + 1})"
            )
            diffs.append(f"  + end_time: \"{new_end}\"")

    if not diffs:
        diffs.append(f"  [INFO] base.yaml 所有 end_time 已为 {new_end}，无改动")
    elif not dry_run:
        new_text = pattern.sub(
            lambda mm: f'{mm.group("prefix")}"{new_end}"{mm.group("suffix")}',
            text,
        )
        yaml_path.write_text(new_text, encoding="utf-8")

    return diffs


# ─────────────────────────────────────────────────────────────────────────────
# 备份
# ─────────────────────────────────────────────────────────────────────────────

def _read_provider_uri(yaml_path: Path) -> Optional[str]:
    """从 yaml 文本中解析 qlib_init.provider_uri（保留注释的文本读取方式）。"""
    text = yaml_path.read_text(encoding="utf-8")
    m = re.search(
        r'provider_uri:\s*["\'](?P<uri>[^"\']+)["\']',
        text,
    )
    return m.group("uri") if m else None


def _calendar_last_day(provider_uri: str) -> date:
    """查询 qlib 日历最后一个交易日。"""
    import qlib
    from qlib.data import D

    qlib.init(
        provider_uri=provider_uri,
        region="cn",
        expression_cache=None,
        dataset_cache=None,
    )
    cal = D.calendar(freq="day")
    if len(cal) == 0:
        raise RuntimeError(f"日历为空: {provider_uri}")
    return pd.Timestamp(cal[-1]).date()


def _backup_file(src: Path, archive_dir: Path, run_ts: str) -> Path:
    """把 src 备份到 archive_dir/<stem>_pre_retrain_<run_ts>.<suffix>。"""
    archive_dir.mkdir(parents=True, exist_ok=True)
    dest = archive_dir / f"{src.stem}_pre_retrain_{run_ts}{src.suffix}"
    shutil.copy2(src, dest)
    return dest


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "按当前日期推进 configs/freq/daily_retrain.yaml 的 dataset_segments，"
            "并同步 configs/base.yaml 的 end_time。"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  python scripts/update_retrain_segments.py\n"
            "  python scripts/update_retrain_segments.py --data-end 2026-05-25\n"
            "  python scripts/update_retrain_segments.py --dry-run\n"
            "  python scripts/update_retrain_segments.py --no-update-base\n"
        ),
    )
    p.add_argument(
        "--data-end",
        type=str,
        default=None,
        help='数据末日（YYYY-MM-DD）；未指定且未用 --from-retrain-provider 时默认今天',
    )
    p.add_argument(
        "--from-retrain-provider",
        action="store_true",
        help=(
            "从 configs/freq/daily_retrain.yaml 的 qlib_init.provider_uri "
            "读取日历末日作为 data_end（monthly_retrain 推荐，避免 test 超出冻结快照）"
        ),
    )
    p.add_argument(
        "--train-start",
        type=str,
        default="2020-01-01",
        help='训练起始日（默认 "2020-01-01"，仅在原 yaml 无 train_start 时生效）',
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印切分计划，不写入文件",
    )
    p.add_argument(
        "--no-update-base",
        action="store_true",
        help="跳过 base.yaml 的 end_time 更新（默认会同步）",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="跳过备份（默认会在 configs/_archive/ 保留原文件副本）",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    if args.data_end and args.from_retrain_provider:
        print("[ERROR] --data-end 与 --from-retrain-provider 不能同时使用", file=sys.stderr)
        return 2

    if args.from_retrain_provider:
        if not DAILY_RETRAIN_YAML.exists():
            print(f"[ERROR] 不存在: {DAILY_RETRAIN_YAML}", file=sys.stderr)
            return 3
        uri = _read_provider_uri(DAILY_RETRAIN_YAML)
        if not uri:
            print("[ERROR] daily_retrain.yaml 中未找到 qlib_init.provider_uri", file=sys.stderr)
            return 3
        try:
            data_end_dt = _calendar_last_day(uri)
            print(f"  [from-retrain-provider] {uri}")
            print(f"  [from-retrain-provider] 日历末日 = {data_end_dt}")
        except Exception as exc:
            print(f"[ERROR] 读取日历失败: {exc}", file=sys.stderr)
            return 4
    elif args.data_end:
        try:
            data_end_dt = datetime.strptime(args.data_end, "%Y-%m-%d").date()
        except ValueError:
            print(f"[ERROR] --data-end 必须是 YYYY-MM-DD 格式，得到: {args.data_end}",
                  file=sys.stderr)
            return 2
    else:
        data_end_dt = date.today()

    print(f"\n{'='*60}")
    print(f"  update_retrain_segments  data_end={data_end_dt}  "
          f"{'(dry-run)' if args.dry_run else ''}")
    print(f"{'='*60}")

    # 计算切分
    try:
        seg = compute_segments(data_end_dt, train_start=args.train_start)
    except ValueError as e:
        print(f"[ERROR] 切分推算失败: {e}", file=sys.stderr)
        return 2

    print("\n  [计算] 新切分:")
    for line in seg.summary_lines():
        print(f"    {line}")

    # 备份（dry-run 时跳过）
    if not args.dry_run and not args.no_backup:
        run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_paths = []
        if DAILY_RETRAIN_YAML.exists():
            backup_paths.append(_backup_file(DAILY_RETRAIN_YAML, ARCHIVE_DIR, run_ts))
        if not args.no_update_base and BASE_YAML.exists():
            backup_paths.append(_backup_file(BASE_YAML, ARCHIVE_DIR, run_ts))
        if backup_paths:
            print("\n  [备份]")
            for p in backup_paths:
                print(f"    {p.relative_to(WORKSPACE_ROOT)}")

    # 更新 daily_retrain.yaml
    if not DAILY_RETRAIN_YAML.exists():
        print(f"[ERROR] 不存在: {DAILY_RETRAIN_YAML}", file=sys.stderr)
        return 3
    print(f"\n  [改写] {DAILY_RETRAIN_YAML.relative_to(WORKSPACE_ROOT)}")
    try:
        for line in update_retrain_yaml(DAILY_RETRAIN_YAML, seg, dry_run=args.dry_run):
            print(line)
    except RuntimeError as e:
        print(f"[ERROR] 改写失败: {e}", file=sys.stderr)
        return 4

    # 更新 base.yaml
    if not args.no_update_base:
        if not BASE_YAML.exists():
            print(f"[ERROR] 不存在: {BASE_YAML}", file=sys.stderr)
            return 3
        # base.yaml 用 LIVE 数据源的日历末日（predict_live / 日常实验），
        # 不用冻结快照的 test_end，避免 handler 被截断到快照日期。
        live_uri = _read_provider_uri(BASE_YAML) or "D:/qlib_data/qlib_data"
        try:
            base_end = _calendar_last_day(live_uri).isoformat()
            print(
                f"\n  [改写] {BASE_YAML.relative_to(WORKSPACE_ROOT)}  "
                f"(end_time → {base_end}, live={live_uri})"
            )
            for line in update_base_end_time(
                BASE_YAML, base_end, dry_run=args.dry_run,
            ):
                print(line)
        except RuntimeError as e:
            print(f"[ERROR] 改写 base.yaml 失败: {e}", file=sys.stderr)
            return 4
        except Exception as exc:
            print(f"[ERROR] 读取 live 日历失败: {exc}", file=sys.stderr)
            return 4

    if args.dry_run:
        print("\n  [dry-run] 未写入任何文件。去掉 --dry-run 执行实际改写。")
    else:
        print("\n  [完成] 切分已推进。下一步运行 run_experiment.py 进行重训。")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
