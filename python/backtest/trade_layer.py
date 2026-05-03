"""
交易层（Trade Layer）
======================
职责：
  - 从 Qlib `indicator_dict` 中提取每笔订单明细（含价格、数量、手续费、方向）
  - 按 FIFO 法配对买卖，计算每笔已实现盈亏（含持仓天数、净收益率）
  - 附带股票基本信息（市场板块分类）

输入:
  - indicator_dict: qlib.backtest() 第二个返回值（dict[freq -> (df, Indicator)]）

输出:
  - TradeRecords:
      trades: 每笔订单明细 DataFrame（长格式）
      realized_pnl: FIFO 配对后的已实现盈亏 DataFrame

设计权衡:
  - 绕过 qlib `SingleData.to_series()` 在新 pandas 下的兼容性 bug，
    直接读 `SingleData.data` + `SingleData.indices[0].idx_list`
  - FIFO 仅按股票内部配对，不考虑除权除息（qlib 已用 adjusted price，
    买卖价已是后复权价，PnL 直接相减即可，无需手工对齐）
  - 卖出量超过持仓队列时（一般不应发生），多余部分忽略并打 warning
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TradeRecords:
    """订单 + 已实现盈亏汇总。"""
    trades: pd.DataFrame
    realized_pnl: pd.DataFrame


# ─────────────────────────────────────────────────────────────────────────────
# 市场分类（按 A 股代码前缀）
# ─────────────────────────────────────────────────────────────────────────────

def classify_market(stock_id: str) -> str:
    """根据股票代码推断市场板块。

    规则:
      - 6 开头 + .SH  → 上证主板
      - 688 开头 + .SH → 科创板
      - 9 开头 + .SH   → 沪 B 股
      - 0 开头 + .SZ   → 深证主板
      - 3 开头 + .SZ   → 创业板
      - 2 开头 + .SZ   → 深 B 股
      - 8/4 开头 + .BJ → 北交所
      - 其他           → 未知
    """
    if not isinstance(stock_id, str) or "." not in stock_id:
        return "未知"
    code, _, suffix = stock_id.partition(".")
    suffix = suffix.upper()
    if suffix == "SH":
        if code.startswith("688"):
            return "科创板"
        if code.startswith("9"):
            return "沪B股"
        if code.startswith("6"):
            return "上证主板"
    elif suffix == "SZ":
        if code.startswith("3"):
            return "创业板"
        if code.startswith("2"):
            return "深B股"
        if code.startswith("0"):
            return "深证主板"
    elif suffix == "BJ":
        if code.startswith(("8", "4")):
            return "北交所"
    return "未知"


# ─────────────────────────────────────────────────────────────────────────────
# 主体逻辑
# ─────────────────────────────────────────────────────────────────────────────

class TradeLayer:
    """从 indicator_dict 提取订单明细并计算 FIFO 已实现盈亏。"""

    # 输出 DataFrame 字段顺序
    _TRADE_COLUMNS = [
        "datetime", "stock_id", "market", "direction",
        "amount", "price", "value", "cost", "fill_ratio",
    ]
    _PNL_COLUMNS = [
        "close_date", "open_date", "stock_id", "market",
        "holding_days", "amount", "buy_price", "sell_price",
        "gross_pnl", "trade_cost", "net_pnl", "return_pct",
    ]

    def extract(self, indicator_dict: Dict[str, Any]) -> TradeRecords:
        """主入口：从 indicator_dict 提取订单与已实现盈亏。"""
        trades_df = self._extract_orders(indicator_dict)
        if trades_df.empty:
            return TradeRecords(
                trades=pd.DataFrame(columns=self._TRADE_COLUMNS),
                realized_pnl=pd.DataFrame(columns=self._PNL_COLUMNS),
            )
        realized_df = self._compute_fifo_pnl(trades_df)
        return TradeRecords(trades=trades_df, realized_pnl=realized_df)

    @staticmethod
    def _extract_orders(indicator_dict: Dict[str, Any]) -> pd.DataFrame:
        """从 qlib `Indicator.order_indicator_his` 解析每笔订单明细。"""
        rows: List[pd.DataFrame] = []
        for _fk, payload in indicator_dict.items():
            if not isinstance(payload, (tuple, list)) or len(payload) < 2:
                continue
            indicator_obj = payload[1]
            order_his = getattr(indicator_obj, "order_indicator_his", None)
            if not isinstance(order_his, dict):
                continue

            for dt, order_ind in order_his.items():
                df = TradeLayer._extract_one_step(dt, order_ind)
                if df is not None and not df.empty:
                    rows.append(df)

        if not rows:
            return pd.DataFrame(columns=TradeLayer._TRADE_COLUMNS)

        merged = pd.concat(rows, ignore_index=True)
        # 仅保留实际成交（deal_amount != 0）
        merged = merged[merged["amount"].abs() > 1e-9].copy()
        if merged.empty:
            return pd.DataFrame(columns=TradeLayer._TRADE_COLUMNS)

        merged["datetime"] = pd.to_datetime(merged["datetime"], errors="coerce")
        merged = merged.dropna(subset=["datetime"])
        merged = merged.sort_values(["datetime", "stock_id"]).reset_index(drop=True)
        return merged[TradeLayer._TRADE_COLUMNS]

    @staticmethod
    def _extract_one_step(
        dt: pd.Timestamp,
        order_ind: Any,
    ) -> Optional[pd.DataFrame]:
        """从单步 BaseOrderIndicator 中读取所有 stock 的订单数据。"""
        data = getattr(order_ind, "data", None)
        if not isinstance(data, dict) or not data:
            return None

        cols: Dict[str, np.ndarray] = {}
        stock_ids: Optional[List[str]] = None
        for metric, sd in data.items():
            arr = getattr(sd, "data", None)
            if arr is None:
                continue
            if stock_ids is None:
                indices = getattr(sd, "indices", None)
                if indices and len(indices) > 0:
                    idx_list = getattr(indices[0], "idx_list", None)
                    if idx_list is not None:
                        stock_ids = [str(s) for s in idx_list]
            cols[metric] = np.asarray(arr)

        if not cols or not stock_ids:
            return None

        # 标准化字段。qlib 字段含义参考 report.py docstring：
        #   amount/deal_amount: 带方向（买正卖负）
        #   trade_value: 已乘 order.sign（买正卖负），这里转绝对值
        #   trade_cost:  绝对值
        #   trade_dir:   1=BUY, 0=SELL
        deal_amount = cols.get("deal_amount", np.zeros(len(stock_ids)))
        trade_price = cols.get("trade_price", np.full(len(stock_ids), np.nan))
        trade_value = cols.get("trade_value", np.zeros(len(stock_ids)))
        trade_cost = cols.get("trade_cost", np.zeros(len(stock_ids)))
        trade_dir = cols.get("trade_dir", np.full(len(stock_ids), np.nan))
        ffr = cols.get("ffr", np.ones(len(stock_ids)))

        directions = np.where(
            np.isfinite(trade_dir),
            np.where(trade_dir > 0.5, "BUY", "SELL"),
            np.where(deal_amount >= 0, "BUY", "SELL"),
        )

        df = pd.DataFrame({
            "datetime": [dt] * len(stock_ids),
            "stock_id": stock_ids,
            "market": [classify_market(s) for s in stock_ids],
            "direction": directions,
            "amount": np.abs(deal_amount),
            "price": trade_price,
            "value": np.abs(trade_value),
            "cost": np.abs(trade_cost),
            "fill_ratio": ffr,
        })
        return df

    @staticmethod
    def _compute_fifo_pnl(trades_df: pd.DataFrame) -> pd.DataFrame:
        """按股票 FIFO 配对买卖，输出每笔已实现盈亏。

        费用分摊:
            买入手续费按本次成交量占该笔买入总量的比例分摊；
            卖出手续费按本次成交量占该笔卖出总量的比例分摊。
        """
        records: List[Dict[str, Any]] = []

        for stock_id, grp in trades_df.groupby("stock_id", sort=False):
            grp = grp.sort_values("datetime")
            buy_lots: deque = deque()  # 队列元素 dict(date, amount_left, price, total_amount, total_cost)

            for _, row in grp.iterrows():
                amount = float(row["amount"])
                if amount <= 1e-9:
                    continue
                if row["direction"] == "BUY":
                    buy_lots.append({
                        "date": row["datetime"],
                        "amount_left": amount,
                        "price": float(row["price"]),
                        "total_amount": amount,
                        "total_cost": float(row["cost"]),
                    })
                    continue

                # SELL：从队头消耗买入批次
                sell_amount_left = amount
                sell_price = float(row["price"])
                sell_total_cost = float(row["cost"])

                while sell_amount_left > 1e-9 and buy_lots:
                    lot = buy_lots[0]
                    matched = min(lot["amount_left"], sell_amount_left)

                    buy_cost_share = (
                        lot["total_cost"] * matched / lot["total_amount"]
                        if lot["total_amount"] > 1e-9 else 0.0
                    )
                    sell_cost_share = (
                        sell_total_cost * matched / amount if amount > 1e-9 else 0.0
                    )
                    gross_pnl = (sell_price - lot["price"]) * matched
                    total_cost = buy_cost_share + sell_cost_share
                    net_pnl = gross_pnl - total_cost
                    cost_basis = lot["price"] * matched
                    return_pct = net_pnl / cost_basis if cost_basis > 1e-9 else 0.0

                    records.append({
                        "close_date": row["datetime"],
                        "open_date": lot["date"],
                        "stock_id": stock_id,
                        "market": row["market"],
                        "holding_days": int((row["datetime"] - lot["date"]).days),
                        "amount": matched,
                        "buy_price": lot["price"],
                        "sell_price": sell_price,
                        "gross_pnl": gross_pnl,
                        "trade_cost": total_cost,
                        "net_pnl": net_pnl,
                        "return_pct": return_pct,
                    })

                    lot["amount_left"] -= matched
                    sell_amount_left -= matched
                    if lot["amount_left"] <= 1e-9:
                        buy_lots.popleft()
                # 卖空场景被忽略（A 股无个股做空，此处不应出现），
                # 若 buy_lots 已空但 sell_amount_left > 0，剩余视为无配对成本。

        if not records:
            return pd.DataFrame(columns=TradeLayer._PNL_COLUMNS)
        df = pd.DataFrame(records)
        return df.sort_values("close_date").reset_index(drop=True)[TradeLayer._PNL_COLUMNS]
