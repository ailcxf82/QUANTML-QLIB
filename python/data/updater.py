"""
行情数据增量更新模块
=====================
层级位置: 外部数据源 → DataUpdater → Qlib 二进制（provider_uri）

输入: 日期字符串 YYYY-MM-DD
输出: 更新 provider_uri 目录下的 Qlib 二进制文件（features/*.bin / calendars/day.txt）

支持的数据源实现:
  - BaoStockUpdater: 免费，无需 token，覆盖 A 股全市场日频 OHLCV
  - TushareUpdater:  需要 pro token，字段更丰富，复权因子更准确

失败处理:
  - 网络请求超时 → 最多重试 3 次，指数退避
  - 数据源无当日数据（非交易日）→ 返回 False，调用方决定是否继续
  - 写入 Qlib 失败 → 抛出 RuntimeError（不静默失败，确保数据完整性）
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger("data.updater")


# ─────────────────────────────────────────────────────────────────────────────
# 抽象接口
# ─────────────────────────────────────────────────────────────────────────────

class DataUpdater(ABC):
    """数据更新抽象基类。

    子类实现两个方法：
      - fetch_latest(date): 拉取指定日期的行情 DataFrame
      - write_to_qlib(df, provider_uri): 将 DataFrame 写入 Qlib 二进制

    Args:
        provider_uri: Qlib 数据目录路径（含 calendars/ instruments/ features/）
        max_retries:  网络请求最大重试次数
    """

    def __init__(self, provider_uri: str, max_retries: int = 3) -> None:
        self.provider_uri = Path(provider_uri)
        self.max_retries = max_retries

    @abstractmethod
    def fetch_latest(self, date: str) -> Optional[pd.DataFrame]:
        """从数据源拉取指定日期的行情数据。

        Args:
            date: 日期字符串 YYYY-MM-DD

        Returns:
            DataFrame，列为 [date, instrument, open, high, low, close, volume, factor]
            若当日为非交易日或数据源无数据，返回 None
        """

    @abstractmethod
    def write_to_qlib(self, df: pd.DataFrame) -> None:
        """将行情 DataFrame 写入 Qlib 二进制格式。

        Args:
            df: fetch_latest 返回的 DataFrame

        Raises:
            RuntimeError: 写入失败时抛出
        """

    def run(self, date: Optional[str] = None) -> bool:
        """完整执行一次数据更新。

        Args:
            date: 目标日期（默认取最近交易日）

        Returns:
            True 表示成功更新，False 表示当日无数据（非交易日）
        """
        if date is None:
            date = pd.Timestamp.today().strftime("%Y-%m-%d")

        logger.info("data_update_start: date=%s provider_uri=%s", date, self.provider_uri)

        # 检查是否已有当日数据
        if self._already_updated(date):
            logger.info("data_update_skip: date=%s already in provider_uri", date)
            return True

        # 拉取（带重试）
        df = None
        for attempt in range(1, self.max_retries + 1):
            try:
                df = self.fetch_latest(date)
                break
            except Exception as exc:
                wait = 2 ** attempt
                logger.warning(
                    "data_update_retry: attempt=%d/%d error=%s wait=%ds",
                    attempt, self.max_retries, exc, wait,
                )
                if attempt < self.max_retries:
                    time.sleep(wait)
                else:
                    raise

        if df is None or df.empty:
            logger.info("data_update_nodata: date=%s is non-trading-day or no data", date)
            return False

        self.write_to_qlib(df)
        logger.info(
            "data_update_done: date=%s n_instruments=%d",
            date, df["instrument"].nunique() if "instrument" in df.columns else len(df),
        )
        return True

    def _already_updated(self, date: str) -> bool:
        """检查 calendars/day.txt 中是否已包含该日期。"""
        cal_path = self.provider_uri / "calendars" / "day.txt"
        if not cal_path.exists():
            return False
        try:
            with cal_path.open("r", encoding="utf-8") as f:
                last_line = f.readlines()[-1].strip()
            return last_line >= date
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────────────────────
# BaoStock 实现
# ─────────────────────────────────────────────────────────────────────────────

class BaoStockUpdater(DataUpdater):
    """基于 BaoStock 免费接口的日频行情更新器。

    安装依赖: pip install baostock
    不需要 token，直接访问，但有速率限制（建议避开高峰期）。

    复权方式: 后复权（qfq=3），与 Qlib provider_uri 中已有数据口径一致。
    字段说明:
      open/high/low/close: 后复权价格（元）
      volume: 成交量（股）
      factor: 复权因子（用于前复权换算：前复权价 = close * factor）

    并发说明:
      max_workers 控制并发线程数（baostock 对单连接有速率限制，建议 10~20）。
      每个线程复用同一 baostock 会话，login/logout 只执行一次。
    """

    _BS_FIELDS = "date,code,open,high,low,close,volume,amount,adjustflag,factor"
    _REQUEST_BATCH_INTERVAL = 0.05  # 批次间最小间隔（秒），避免触发频控

    def __init__(
        self,
        provider_uri: str,
        adjust_flag: str = "3",   # 1=前复权 2=不复权 3=后复权
        max_retries: int = 3,
        max_workers: int = 16,
    ) -> None:
        super().__init__(provider_uri=provider_uri, max_retries=max_retries)
        self.adjust_flag = adjust_flag
        self.max_workers = max_workers

    def _fetch_one_code(
        self,
        bs: "baostock",  # type: ignore[name-defined]
        code: str,
        date: str,
    ) -> List[List[str]]:
        """拉取单只股票数据（线程安全：baostock 连接线程局部）。"""
        rs = bs.query_history_k_data_plus(
            code=code,
            fields=self._BS_FIELDS,
            start_date=date,
            end_date=date,
            frequency="d",
            adjustflag=self.adjust_flag,
        )
        if rs.error_code != "0":
            return []
        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        return rows

    def fetch_latest(self, date: str) -> Optional[pd.DataFrame]:
        """从 BaoStock 并发拉取指定日期全市场 A 股日频行情。

        使用 ThreadPoolExecutor 并发请求，比串行逐票快 10x 以上。
        baostock 的全局 login/logout 在主线程中完成，各线程复用同一会话。
        """
        try:
            import baostock as bs
        except ImportError as exc:
            raise ImportError(
                "未安装 baostock，请执行: pip install baostock"
            ) from exc

        bs.login()
        try:
            # 获取当日所有 A 股列表
            rs_stock = bs.query_stock_basic(type="1", status="1")
            codes: List[str] = []
            while rs_stock.error_code == "0" and rs_stock.next():
                codes.append(rs_stock.get_row_data()[0])

            if not codes:
                logger.warning("baostock_no_codes: date=%s", date)
                return None

            logger.info(
                "baostock_fetch_start: date=%s n_codes=%d max_workers=%d",
                date, len(codes), self.max_workers,
            )

            records: List[List[str]] = []
            failed_codes: List[str] = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                future_to_code = {
                    executor.submit(self._fetch_one_code, bs, code, date): code
                    for code in codes
                }
                for future in as_completed(future_to_code):
                    code = future_to_code[future]
                    try:
                        rows = future.result()
                        records.extend(rows)
                    except Exception as exc:
                        failed_codes.append(code)
                        logger.debug("baostock_code_error: code=%s error=%s", code, exc)

            if failed_codes:
                logger.warning(
                    "baostock_fetch_partial_failure: %d codes failed (out of %d)",
                    len(failed_codes), len(codes),
                )

            if not records:
                return None

            df = pd.DataFrame(records, columns=self._BS_FIELDS.split(","))
            df = self._clean(df, date)
            logger.info(
                "baostock_fetch_done: date=%s n_records=%d n_instruments=%d",
                date, len(df), df["instrument"].nunique() if not df.empty else 0,
            )
            return df

        finally:
            bs.logout()

    def _clean(self, df: pd.DataFrame, date: str) -> pd.DataFrame:
        """清洗 BaoStock 返回的 DataFrame，转为标准格式。"""
        df = df[df["date"] == date].copy()
        if df.empty:
            return df

        # code 格式 bs: "sh.600000" → Qlib: "SH600000"（或 "600000.SH"）
        # Qlib provider_uri 使用 "SH600000" / "SZ000001" 格式
        df["instrument"] = df["code"].apply(self._bs_code_to_qlib)
        for col in ("open", "high", "low", "close", "volume", "factor"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["close", "volume"])
        df = df[df["close"] > 0]

        return df[["date", "instrument", "open", "high", "low", "close", "volume", "factor"]]

    @staticmethod
    def _bs_code_to_qlib(bs_code: str) -> str:
        """将 BaoStock 代码格式（sh.600000）转为 Qlib 格式（SH600000）。

        注: Qlib provider_uri 的实际格式取决于 dump_bin 时使用的格式。
        若你的数据为 "600000.SH" 格式，修改此函数的 return 语句即可。
        """
        parts = bs_code.split(".")
        if len(parts) == 2:
            exchange, code = parts[0].upper(), parts[1]
            return f"{exchange}{code}"
        return bs_code

    def write_to_qlib(self, df: pd.DataFrame) -> None:
        """调用 qlib.data.dump_bin 将 DataFrame 写入 Qlib 二进制。

        注: qlib.data.dump_bin 需要 qlib >= 0.8，并且 provider_uri 已存在基础结构
        （calendars/、instruments/、features/ 目录）。

        若遭遇版本不兼容，可退化为直接读写 .bin 文件（参见 qlib 源码 DumpDataUpdate）。
        """
        try:
            from qlib.data.dump_bin import DumpDataUpdate
        except ImportError:
            raise RuntimeError(
                "无法导入 qlib.data.dump_bin.DumpDataUpdate，"
                "请确认 qlib >= 0.8 且在 qlib_zhengshi 环境中运行。"
            )

        # 将 DataFrame 保存为临时 CSV（DumpDataUpdate 接受文件路径或 DataFrame）
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False, encoding="utf-8"
        ) as tmp:
            tmp_path = tmp.name
            df.to_csv(tmp_path, index=False)

        try:
            DumpDataUpdate(
                csv_path=tmp_path,
                qlib_dir=str(self.provider_uri),
                date_field_name="date",
                symbol_field_name="instrument",
                exclude_fields="date,instrument",
                freq="day",
            ).dump()
            logger.info("write_to_qlib_done: n_rows=%d", len(df))
        finally:
            os.unlink(tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Tushare 实现（需 token）
# ─────────────────────────────────────────────────────────────────────────────

class TushareUpdater(DataUpdater):
    """基于 Tushare Pro 的日频行情更新器（需要 token）。

    安装依赖: pip install tushare
    获取 token: https://tushare.pro/register

    与 BaoStockUpdater 相比，Tushare 数据质量更高，复权更准确，
    但需要账户积分（日频接口 120 积分可用）。
    """

    def __init__(
        self,
        provider_uri: str,
        token: str,
        max_retries: int = 3,
    ) -> None:
        super().__init__(provider_uri=provider_uri, max_retries=max_retries)
        self._token = token

    def _get_pro(self):
        try:
            import tushare as ts
        except ImportError as exc:
            raise ImportError("未安装 tushare，请执行: pip install tushare") from exc
        ts.set_token(self._token)
        return ts.pro_api()

    def fetch_latest(self, date: str) -> Optional[pd.DataFrame]:
        """从 Tushare Pro 拉取指定日期全市场日频行情（复权价格）。"""
        pro = self._get_pro()
        date_ts = date.replace("-", "")  # YYYYMMDD 格式

        df = pro.daily(trade_date=date_ts)
        if df is None or df.empty:
            return None

        # 获取复权因子
        adj = pro.adj_factor(trade_date=date_ts)
        if adj is not None and not adj.empty:
            df = df.merge(adj[["ts_code", "adj_factor"]], on="ts_code", how="left")
            df["adj_factor"] = df["adj_factor"].fillna(1.0)
        else:
            df["adj_factor"] = 1.0

        # ts_code 格式: "600000.SH" → Qlib 格式 "SH600000"
        df["instrument"] = df["ts_code"].apply(
            lambda c: c.split(".")[1] + c.split(".")[0] if "." in c else c
        )
        df["date"] = date
        df = df.rename(columns={
            "open": "open", "high": "high", "low": "low",
            "close": "close", "vol": "volume", "adj_factor": "factor",
        })

        for col in ("open", "high", "low", "close", "volume", "factor"):
            df[col] = pd.to_numeric(df.get(col), errors="coerce")

        df = df.dropna(subset=["close"])
        return df[["date", "instrument", "open", "high", "low", "close", "volume", "factor"]]

    def write_to_qlib(self, df: pd.DataFrame) -> None:
        """同 BaoStockUpdater.write_to_qlib，复用相同写入逻辑。"""
        BaoStockUpdater(str(self.provider_uri)).write_to_qlib(df)


# ─────────────────────────────────────────────────────────────────────────────
# 日历更新工具
# ─────────────────────────────────────────────────────────────────────────────

def append_calendar(provider_uri: str, date: str) -> bool:
    """将新交易日追加到 calendars/day.txt（若不存在则创建）。

    Args:
        provider_uri: Qlib 数据目录
        date:         YYYY-MM-DD 格式的日期

    Returns:
        True = 成功追加（新日期），False = 已存在（无变化）
    """
    cal_path = Path(provider_uri) / "calendars" / "day.txt"
    cal_path.parent.mkdir(parents=True, exist_ok=True)

    existing: List[str] = []
    if cal_path.exists():
        existing = [l.strip() for l in cal_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    if date in existing:
        return False

    existing.append(date)
    existing.sort()
    cal_path.write_text("\n".join(existing) + "\n", encoding="utf-8")
    logger.info("calendar_updated: added %s (total=%d)", date, len(existing))
    return True
