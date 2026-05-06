"""
StockFilter — ST 与连续涨停过滤器
=====================================
层级位置：Signal 层 → Portfolio 层之间的预筛选屏障。

职责：
  1. 读取 is_st.csv，按日期快速查询 ST 类股票（O(1) 字典查找）
  2. 通过 Qlib D.features 检测近期连续涨停股票（结果缓存复用）
  3. 提供统一接口 get_excluded_with_reasons()，供回测策略与实盘预测共同调用

设计约束：
  - CSV 仅在 __init__ 时加载一次，不重复 IO
  - 涨停缓存上限 500 条，防止回测全量缓存撑满内存
  - 所有异常静默处理并记录日志，过滤失败不中断主流程
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

_logger = logging.getLogger("StockFilter")


class StockFilter:
    """
    ST + 连续涨停过滤器，回测策略与实盘预测共用。

    Args:
        st_csv_path:            is_st.csv 路径（绝对路径或相对工作目录）。
                                None 或路径不存在时跳过 ST 过滤。
        consecutive_limit_days: 连续涨停天数阈值（0 = 不过滤涨停）。
                                建议值 2~4。
    """

    def __init__(
        self,
        st_csv_path: Optional[str] = None,
        consecutive_limit_days: int = 3,
    ) -> None:
        # {date_str_YYYYMMDD: Set[UPPER_INSTRUMENT_CODE]}
        self._st_index: Dict[str, Set[str]] = {}
        self._consecutive_limit_days = int(consecutive_limit_days)
        # 涨停缓存：{cache_key: Set[instrument]}
        self._limit_up_cache: Dict[str, Set[str]] = {}

        if st_csv_path:
            self._load_st_csv(Path(st_csv_path))
        else:
            _logger.info("未指定 st_csv_path，跳过 ST 过滤")

    # ─────────────────────────────────────────────────────────────────
    # 初始化
    # ─────────────────────────────────────────────────────────────────

    def _load_st_csv(self, path: Path) -> None:
        """加载 is_st.csv，构建 date → instruments 索引。

        CSV 格式（tushare 标准）：
            ts_code (str)  trade_date (int, YYYYMMDD)  type  type_name
        """
        if not path.exists():
            _logger.warning("ST CSV 不存在: %s，跳过 ST 过滤", path)
            return
        try:
            df = pd.read_csv(
                path,
                usecols=["ts_code", "trade_date"],
                dtype={"ts_code": str, "trade_date": str},
            )
            df["trade_date"] = df["trade_date"].astype(str).str.strip().str.zfill(8)
            df["ts_code"] = df["ts_code"].str.strip().str.upper()

            for date_str, grp in df.groupby("trade_date"):
                self._st_index[str(date_str)] = set(grp["ts_code"].tolist())

            _logger.info(
                "ST CSV 已加载: %d 个交易日，共 %d 条 ST 记录",
                len(self._st_index),
                len(df),
            )
        except Exception as exc:
            _logger.error("ST CSV 加载失败: %s", exc)

    # ─────────────────────────────────────────────────────────────────
    # 公开查询接口
    # ─────────────────────────────────────────────────────────────────

    def get_st_set(self, date: Any) -> Set[str]:
        """返回指定日期的 ST 股票代码集合（空集表示无数据或未加载）。

        Args:
            date: 日期，支持 "2026-05-06"、"20260506"、pd.Timestamp 等格式。
        """
        date_str = _to_date8(date)
        return self._st_index.get(date_str, set())

    def get_limit_up_set(
        self,
        instruments: List[str],
        date: Any,
    ) -> Set[str]:
        """检测截至 date 前一个交易日已连续涨停的股票。

        Args:
            instruments: 候选股票代码列表
            date:        当前决策日期（T），检测 T-1 及之前的状态

        Returns:
            Set[instrument] 需要过滤的连续涨停股集合
        """
        if not instruments or self._consecutive_limit_days <= 0:
            return set()

        date_str = _to_date8(date)
        # 用日期 + 前 N 只代码作缓存 key（平衡精度与内存）
        cache_key = f"{date_str}|{','.join(sorted(str(i) for i in instruments[:20]))}"
        if cache_key in self._limit_up_cache:
            return self._limit_up_cache[cache_key]

        result: Set[str] = set()
        try:
            from qlib.data import D

            lookback_days = self._consecutive_limit_days * 4 + 15
            end_dt = pd.Timestamp(date) - pd.Timedelta(days=1)
            start_dt = end_dt - pd.Timedelta(days=lookback_days)

            features = D.features(
                list(instruments),
                fields=["$close"],
                start_time=start_dt.strftime("%Y-%m-%d"),
                end_time=end_dt.strftime("%Y-%m-%d"),
                freq="day",
            )
            if features is not None and not features.empty:
                for inst in features.index.get_level_values("instrument").unique():
                    try:
                        close = (
                            features.xs(inst, level="instrument")["$close"]
                            .sort_index()
                            .dropna()
                        )
                        if len(close) < self._consecutive_limit_days + 1:
                            continue
                        rets = close.pct_change().dropna()
                        recent = rets.iloc[-self._consecutive_limit_days :]
                        if (recent >= 0.095).all():
                            result.add(str(inst))
                    except Exception:
                        continue
        except Exception as exc:
            _logger.debug("连续涨停检测跳过 (%s)", exc)

        # 写入缓存，超限时删除最旧条目
        self._limit_up_cache[cache_key] = result
        if len(self._limit_up_cache) > 500:
            del self._limit_up_cache[next(iter(self._limit_up_cache))]

        return result

    def get_excluded_with_reasons(
        self,
        instruments: List[str],
        date: Any,
        check_st: bool = True,
        check_limit_up: bool = True,
    ) -> Dict[str, str]:
        """返回需要过滤的股票及其原因。

        Args:
            instruments:    候选股票代码列表
            date:           决策日期
            check_st:       是否检测 ST
            check_limit_up: 是否检测连续涨停

        Returns:
            {instrument: reason_str}，未被过滤的股票不出现在字典中
        """
        if not instruments:
            return {}

        excluded: Dict[str, str] = {}

        if check_st and self._st_index:
            st_today = self.get_st_set(date)
            for inst in instruments:
                if str(inst).upper() in st_today:
                    excluded[str(inst)] = "ST 类股票（日限幅 ±5%，财务风险高）"
            if excluded:
                _logger.info(
                    "[StockFilter] %s ST 过滤 %d 支: %s",
                    _to_date8(date),
                    len(excluded),
                    sorted(excluded.keys()),
                )

        if check_limit_up and self._consecutive_limit_days > 0:
            remaining = [i for i in instruments if str(i) not in excluded]
            if remaining:
                limit_set = self.get_limit_up_set(remaining, date)
                for inst in limit_set:
                    if str(inst) not in excluded:
                        excluded[str(inst)] = (
                            f"连续涨停 {self._consecutive_limit_days} 天"
                            f"（封单风险，T+1 可能无法买入）"
                        )
                if limit_set:
                    _logger.info(
                        "[StockFilter] %s 涨停过滤 %d 支: %s",
                        _to_date8(date),
                        len(limit_set),
                        sorted(limit_set),
                    )

        return excluded

    def filter_candidates(
        self,
        candidates: List[str],
        date: Any,
        check_st: bool = True,
        check_limit_up: bool = True,
    ) -> Tuple[List[str], Dict[str, str]]:
        """过滤候选列表，返回 (干净候选列表, 被排除字典)。

        顺序保持不变（仅删除被排除项）。
        """
        excluded = self.get_excluded_with_reasons(
            candidates, date, check_st=check_st, check_limit_up=check_limit_up
        )
        clean = [c for c in candidates if str(c) not in excluded]
        return clean, excluded

    @property
    def has_st_data(self) -> bool:
        """是否已成功加载 ST CSV 数据。"""
        return bool(self._st_index)

    def describe(self) -> str:
        return (
            f"StockFilter(st_dates={len(self._st_index)}, "
            f"limit_up_days={self._consecutive_limit_days})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def _to_date8(date: Any) -> str:
    """将各种日期格式统一转为 'YYYYMMDD' 字符串。"""
    if isinstance(date, str):
        return date.replace("-", "").strip()[:8]
    if isinstance(date, pd.Timestamp):
        return date.strftime("%Y%m%d")
    try:
        return pd.Timestamp(date).strftime("%Y%m%d")
    except Exception:
        return str(date).replace("-", "")[:8]


def build_stock_filter(cfg: Optional[Dict[str, Any]]) -> Optional["StockFilter"]:
    """从配置字典构建 StockFilter（供策略层调用）。

    cfg 结构：
        st_csv_path:            "D:/qlib_data/qlib_data/is_st.csv"
        consecutive_limit_days: 3
        exclude_st:             true
        exclude_limit_up:       true

    Returns:
        StockFilter 实例，或 None（cfg 为空时）
    """
    if not cfg:
        return None
    return StockFilter(
        st_csv_path=cfg.get("st_csv_path"),
        consecutive_limit_days=int(cfg.get("consecutive_limit_days", 3)),
    )
