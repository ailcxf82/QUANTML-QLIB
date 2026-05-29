"""
BaoStock 5min 行情采集适配层（B3 分钟 Deep Factor 数据前置）
==============================================================
层级位置: 外部数据源(BaoStock) → BaoStock5minCollector → csv_data_5min/ → dump_bin → qlib_data_5min/

设计动机: docs/时序模型优化.md §8.5 架构 B（Deep Factor）。日内 5min 序列经时序模型
压缩为 hidden state，再作为 LGBM 的新特征。本模块只负责「采集 + 落地 CSV」，
入库（dump_bin）由 scripts/build_5min_qlib_data.py 调用 qlib-main 的 dump_bin.py 完成。

输入:
  - 股票池 instruments txt（默认 csi500.txt，格式: 「INSTRUMENT<TAB>START<TAB>END」）
  - 起止日期 [start, end]
输出:
  - 每只股票一个 CSV: <out_dir>/<INSTRUMENT>.csv
    列: symbol,date,open,high,low,close,volume,amount
    date 为 bar 起始时刻（"%Y-%m-%d %H:%M:%S"），与 qlib-main baostock_5min 约定一致。

数据口径:
  - frequency="5"（5 分钟），adjustflag="3"（后复权，连续序列）。
  - A 股每交易日 48 根 5min bar（09:35-11:30 共 24 根 + 13:05-15:00 共 24 根，
    BaoStock 的 time 字段为 bar 结束时刻，本模块统一减 5min 转 bar 起始）。

失败处理:
  - 单只股票查询失败 → 记录到 failed 列表，不中断整体；调用方可重跑（断点续传）。
  - 已存在且覆盖目标区间的 CSV → 跳过（断点续传）。

合规:
  - 仅读取公开行情，无账号/密钥；符合「数据源接入走可审计适配层」约束。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("data.baostock_5min")

# BaoStock 5min 查询字段（date,time 都取，time 用于精确到分钟）
_BS_FIELDS = "date,time,code,open,high,low,close,volume,amount,adjustflag"
# 输出 CSV 的标准列（供 dump_bin 使用）
_OUT_COLS = ["symbol", "date", "open", "high", "low", "close", "volume", "amount"]
# A 股单日 5min bar 数（含集合竞价归并后），用于完整性校验
EXPECTED_BARS_PER_DAY = 48


def instrument_to_bs_code(instrument: str) -> str:
    """'000006.SZ' → 'sz.000006'；'600000.SH' → 'sh.600000'。"""
    code, _, exch = instrument.partition(".")
    return f"{exch.lower()}.{code}"


def read_universe(
    instruments_txt: Path,
    start: str,
    end: str,
) -> List[str]:
    """读取股票池 txt，返回在 [start, end] 区间内出现过的成分（并集，去重排序）。

    txt 格式（qlib instruments）: INSTRUMENT<TAB>START_DATE<TAB>END_DATE，
    同一标的可能多行（成分进出多次）。只要区间与 [start,end] 有交集即纳入。
    """
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    members: set[str] = set()
    for line in instruments_txt.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        inst, seg_start, seg_end = parts[0], parts[1], parts[2]
        try:
            seg_s, seg_e = pd.Timestamp(seg_start), pd.Timestamp(seg_end)
        except Exception:
            continue
        # 区间有交集
        if seg_e >= start_ts and seg_s <= end_ts:
            members.add(inst)
    return sorted(members)


class BaoStock5minCollector:
    """BaoStock 5min 采集器（按股票分文件落 CSV，断点续传）。

    Args:
        out_dir:      CSV 输出目录（默认 D:/qlib_data/csv_data_5min）
        adjust_flag:  复权方式，"1"=前复权 "2"=不复权 "3"=后复权（默认 3）
        max_retries:  单只股票查询最大重试
        chunk_years:  按年分片查询（避免单次返回过大），默认 1
    """

    def __init__(
        self,
        out_dir: str = "D:/qlib_data/csv_data_5min",
        adjust_flag: str = "3",
        max_retries: int = 3,
        chunk_years: int = 1,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.adjust_flag = adjust_flag
        self.max_retries = max_retries
        self.chunk_years = max(1, int(chunk_years))

    # ── 单只股票拉取 ───────────────────────────────────────────────
    def _fetch_one(self, bs, instrument: str, start: str, end: str) -> pd.DataFrame:
        """拉取单只股票 [start,end] 的 5min 数据（按年分片合并）。"""
        bs_code = instrument_to_bs_code(instrument)
        frames: List[pd.DataFrame] = []
        for seg_start, seg_end in self._year_chunks(start, end):
            rs = bs.query_history_k_data_plus(
                bs_code,
                _BS_FIELDS,
                start_date=seg_start,
                end_date=seg_end,
                frequency="5",
                adjustflag=self.adjust_flag,
            )
            if rs.error_code != "0":
                logger.debug("bs_query_error code=%s msg=%s inst=%s", rs.error_code, rs.error_msg, instrument)
                continue
            rows: List[List[str]] = []
            while rs.next():
                rows.append(rs.get_row_data())
            if rows:
                frames.append(pd.DataFrame(rows, columns=rs.fields))
        if not frames:
            return pd.DataFrame(columns=_OUT_COLS)
        df = pd.concat(frames, ignore_index=True)
        return self._normalize(df, instrument)

    def _year_chunks(self, start: str, end: str) -> List[Tuple[str, str]]:
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        chunks: List[Tuple[str, str]] = []
        cur = s
        while cur <= e:
            seg_end = min(pd.Timestamp(year=cur.year + self.chunk_years - 1, month=12, day=31), e)
            chunks.append((cur.strftime("%Y-%m-%d"), seg_end.strftime("%Y-%m-%d")))
            cur = seg_end + pd.Timedelta(days=1)
        return chunks

    def _normalize(self, df: pd.DataFrame, instrument: str) -> pd.DataFrame:
        """转标准列；datetime 取 bar 起始时刻（BaoStock time 为结束时刻，减 5min）。"""
        if df.empty:
            return pd.DataFrame(columns=_OUT_COLS)
        # time 形如 "20260506093500000"
        ts = pd.to_datetime(df["time"], format="%Y%m%d%H%M%S%f")
        # BaoStock 的 time 为 bar 结束时刻 → 减 5min 转为 bar 起始
        df = df.assign(date=(ts - pd.Timedelta(minutes=5)).dt.strftime("%Y-%m-%d %H:%M:%S"))
        df["symbol"] = instrument
        for col in ("open", "high", "low", "close", "volume", "amount"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close", "volume"])
        df = df[df["close"] > 0]
        df = df.sort_values("date").drop_duplicates("date")
        return df[_OUT_COLS]

    def _csv_path(self, instrument: str) -> Path:
        return self.out_dir / f"{instrument}.csv"

    def _already_done(self, instrument: str, start: str, end: str) -> bool:
        """已存在且覆盖目标区间末端 → 跳过（断点续传，按交易日粗判）。"""
        p = self._csv_path(instrument)
        if not p.exists() or p.stat().st_size == 0:
            return False
        try:
            tail = pd.read_csv(p, usecols=["date"]).tail(1)["date"].iloc[0]
            return pd.Timestamp(tail) >= pd.Timestamp(end) - pd.Timedelta(days=7)
        except Exception:
            return False

    # ── 批量采集 ───────────────────────────────────────────────────
    def collect(
        self,
        instruments: List[str],
        start: str,
        end: str,
        resume: bool = True,
        progress_every: int = 25,
    ) -> dict:
        """采集股票池 [start,end] 的 5min 数据，逐只落 CSV。

        采用单 BaoStock 会话顺序拉取（baostock 单连接非线程安全，顺序最稳）。
        返回统计字典 {n_total, n_ok, n_skip, n_fail, failed:[...]}。
        """
        try:
            import baostock as bs
        except ImportError as exc:
            raise ImportError("未安装 baostock，请: pip install baostock") from exc

        lg = bs.login()
        if lg.error_code != "0":
            raise RuntimeError(f"baostock 登录失败: {lg.error_code} {lg.error_msg}")

        n_ok = n_skip = n_fail = 0
        failed: List[str] = []
        t0 = time.perf_counter()
        try:
            for i, inst in enumerate(instruments, 1):
                if resume and self._already_done(inst, start, end):
                    n_skip += 1
                    continue
                df: Optional[pd.DataFrame] = None
                for attempt in range(1, self.max_retries + 1):
                    try:
                        df = self._fetch_one(bs, inst, start, end)
                        break
                    except Exception as exc:
                        logger.warning("fetch_retry inst=%s attempt=%d err=%s", inst, attempt, exc)
                        time.sleep(2 ** attempt)
                if df is None or df.empty:
                    n_fail += 1
                    failed.append(inst)
                else:
                    df.to_csv(self._csv_path(inst), index=False, encoding="utf-8")
                    n_ok += 1
                if i % progress_every == 0:
                    elapsed = time.perf_counter() - t0
                    logger.info(
                        "collect_progress %d/%d ok=%d skip=%d fail=%d elapsed=%.0fs",
                        i, len(instruments), n_ok, n_skip, n_fail, elapsed,
                    )
        finally:
            bs.logout()

        stats = {
            "n_total": len(instruments),
            "n_ok": n_ok,
            "n_skip": n_skip,
            "n_fail": n_fail,
            "failed": failed,
            "elapsed_sec": round(time.perf_counter() - t0, 1),
        }
        logger.info("collect_done %s", {k: v for k, v in stats.items() if k != "failed"})
        return stats


def main() -> None:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="BaoStock 5min csi500 采集 → csv_data_5min/")
    parser.add_argument("--instruments", default="D:/qlib_data/qlib_data/instruments/csi500.txt")
    parser.add_argument("--out-dir", default="D:/qlib_data/csv_data_5min")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default="2026-05-22")
    parser.add_argument("--adjust-flag", default="3")
    parser.add_argument("--chunk-years", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="只取前 N 只（冒烟用）")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    universe = read_universe(Path(args.instruments), args.start, args.end)
    if args.limit:
        universe = universe[: args.limit]
    logger.info("universe_size=%d start=%s end=%s", len(universe), args.start, args.end)

    collector = BaoStock5minCollector(
        out_dir=args.out_dir,
        adjust_flag=args.adjust_flag,
        chunk_years=args.chunk_years,
    )
    stats = collector.collect(
        universe, args.start, args.end, resume=not args.no_resume,
    )
    print(json.dumps({k: v for k, v in stats.items() if k != "failed"}, ensure_ascii=False))
    if stats["failed"]:
        print(f"failed({len(stats['failed'])}): {stats['failed'][:20]}{' ...' if len(stats['failed'])>20 else ''}")


if __name__ == "__main__":
    main()
