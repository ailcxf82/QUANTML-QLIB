"""
市场状态因子（market state features）。

层级位置: Data -> [Feature] -> Model -> Signal -> Portfolio -> Backtest -> Evaluation

设计动机
--------
个股因子（momentum / volume / technical / fundamental）描述单只股票的横截面差异，
但在风格切换、系统性风险事件（如 2025-Q4）下，全市场单边运动会让所有个股因子同时
失效，IC 标准差被异常拉大。

市场状态因子刻画"全市场宏观状态"，作为额外特征注入后，LightGBM 等模型可以
学到分裂规则"当大盘高波动 / 大盘连续下跌时，降低对动量因子的依赖"，从而显著
压低风险事件期的 IC 噪声。

注入方式
--------
QLib 表达式系统不支持跨 instrument 引用，因此本模块：
1. 基于 ``$close`` / ``$volume`` 计算指数（默认 ``000905.SH`` 中证 500）的市场状态序列；
2. 通过 :class:`MarketStateDataLoader` 把同一序列广播到每只股票同日同值；
3. 由 :func:`run_experiment.py` 在 ``inject_feature_config`` 之后将其包装进
   ``NestedDataLoader``，与个股因子横向 merge。

为何选 000905.SH
----------------
回测 ``tradable_universe = "csi500"``，使用同一指数计算市场状态可以与基准
强相关，使模型学到的"市场状态"和回测口径一致；若未来切换 universe，可同步
修改 ``MARKET_INDEX``。
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import pandas as pd

from qlib.data import D
from qlib.data.dataset.loader import DataLoader
from qlib.data.dataset.processor import Processor

# ── 市场状态指数（与 base.yaml 中 benchmark 对齐）────────────────────
# 注：当前 qlib 数据集中中证 500 指数的代码为 ``000905.SZ``（实测可读到行情），
#     ``.SH`` / ``SHxxx`` 等变体在该数据集中均无数据；与 base.yaml benchmark 一致。
MARKET_INDEX: str = "000905.SZ"

# ── 表达式定义（针对指数日线）─────────────────────────────────────
# 命名前缀统一加 MKT_，避免与个股因子重名
_MARKET_EXPRS: "dict[str, str]" = {
    "MKT_RET1":   "$close/Ref($close,1)-1",
    "MKT_RET5":   "$close/Ref($close,5)-1",
    "MKT_RET20":  "$close/Ref($close,20)-1",
    "MKT_VOL5":   "Std($close/Ref($close,1)-1, 5)",
    "MKT_VOL20":  "Std($close/Ref($close,1)-1, 20)",
    # 当前点位距 20 日高点的相对回撤（负值，越小=越靠近高点；越大=离高点越远）
    "MKT_DD20":   "$close/Max($close,20)-1",
    # 大盘量比（当日 / 20 日均量），>1 表示放量
    "MKT_TURN":   "$volume/(Mean($volume,20)+1e-12)",
}

MARKET_STATE_NAMES: List[str] = list(_MARKET_EXPRS.keys())
MARKET_STATE_EXPRS: List[str] = list(_MARKET_EXPRS.values())


def compute_market_state_series(
    start_time: str,
    end_time: str,
    index_code: str = MARKET_INDEX,
    freq: str = "day",
) -> pd.DataFrame:
    """计算指数级市场状态序列（去掉 instrument 维度，仅保留 datetime）。

    Args:
        start_time: 起始日期（含），格式同 qlib expr 接口
        end_time:   结束日期（含）
        index_code: 指数代码，默认 ``000905.SH``
        freq:       数据频率，默认日线

    Returns:
        DataFrame，``index = DatetimeIndex``，``columns = MARKET_STATE_NAMES``。

    Raises:
        RuntimeError: 指数在该区间无数据时抛出。
    """
    df = D.features(
        [index_code],
        MARKET_STATE_EXPRS,
        start_time=start_time,
        end_time=end_time,
        freq=freq,
    )
    if df is None or df.empty:
        raise RuntimeError(
            f"market_state: 指数 {index_code} 在 [{start_time}, {end_time}] 区间无可用数据"
        )
    df.columns = MARKET_STATE_NAMES
    # qlib 返回的 index 是 (instrument, datetime)，去掉 instrument 维
    df = df.reset_index(level=0, drop=True)
    df.sort_index(inplace=True)
    return df


def _resolve_instrument_list(
    instruments,
    start_time: Optional[str],
    end_time: Optional[str],
) -> List[str]:
    """把 qlib instruments 参数（dict / 字符串 / 列表）解析为具体股票列表。

    与 :class:`qlib.data.dataset.loader.QlibDataLoader` 的解析逻辑等价，
    避免 StaticDataLoader 不支持字符串 instruments 的问题。
    """
    if instruments is None:
        # None 在上层一般意味着"全部"，回退到 all
        instruments = "all"
    if isinstance(instruments, str):
        inst_cfg = D.instruments(instruments)
    elif isinstance(instruments, dict):
        inst_cfg = instruments
    elif isinstance(instruments, Iterable):
        return list(instruments)
    else:
        raise TypeError(f"无法解析 instruments={instruments!r}")
    return D.list_instruments(
        inst_cfg,
        start_time=start_time,
        end_time=end_time,
        as_list=True,
    )


class MarketStateDataLoader(DataLoader):
    """把指数级市场状态广播到全市场每只股票的 DataLoader。

    用于 :class:`qlib.data.dataset.loader.NestedDataLoader` 的子加载器：
    与个股因子的 :class:`QlibDataLoader` 横向合并后，所有股票在同一日的市场状态
    特征值相同，从而给树模型提供"市场环境"上下文。

    注：本 Loader 不缓存指数序列；每次 load() 会按传入区间重新计算，避免在
    跨频率（日/分钟）回测中复用陈旧缓存。
    """

    def __init__(
        self,
        index_code: str = MARKET_INDEX,
        freq: str = "day",
        fields_group: str = "feature_mkt",
    ) -> None:
        super().__init__()
        self.index_code = index_code
        self.freq = freq
        self.fields_group = fields_group

    def load(
        self,
        instruments=None,
        start_time=None,
        end_time=None,
    ) -> pd.DataFrame:
        if start_time is None or end_time is None:
            raise ValueError(
                "MarketStateDataLoader.load 需要明确的 start_time / end_time"
            )

        market_df = compute_market_state_series(
            start_time=start_time,
            end_time=end_time,
            index_code=self.index_code,
            freq=self.freq,
        )

        inst_list = _resolve_instrument_list(instruments, start_time, end_time)
        if not inst_list:
            raise RuntimeError(
                f"MarketStateDataLoader: instruments 解析为空（输入={instruments}）"
            )

        # 广播：把单 datetime 序列复制到每只 instrument
        broadcasted = pd.concat(
            {inst: market_df for inst in inst_list},
            names=["instrument"],
        )
        # 当前 index 顺序: (instrument, datetime)，调整为 (datetime, instrument)
        broadcasted = broadcasted.swaplevel().sort_index()
        broadcasted.index.set_names(["datetime", "instrument"], inplace=True)

        # 加 fields_group 列层（与 QlibDataLoader 输出格式一致）
        broadcasted.columns = pd.MultiIndex.from_product(
            [[self.fields_group], MARKET_STATE_NAMES]
        )
        return broadcasted


class MarketStateMergeProcessor(Processor):
    """把 ``feature_mkt`` 组合并到 ``feature`` 组的处理器。

    背景：截面归一化（``CSZScoreNorm``）作用于 ``feature`` 组时会把每日全市场同值
    的市场状态因子清零（``(x - median) / mad = 0/0 → NaN``），使其完全失效。
    本 Processor 设计为放在所有 CSZScoreNorm/Fillna 之后，把仍保留原始值的
    ``feature_mkt`` 组重命名并合并到 ``feature`` 组，让下游模型（LGBM 等）通过
    ``df["feature"]`` 自动取到市场状态因子。

    注：本 Processor 不修改数值，只修改列结构。市场状态因子保留原始量纲，由
    LGBM 等树模型自适应；线性 / 神经网络模型可在此之后再追加 ``RobustZScoreNorm``。
    """

    def __init__(
        self,
        src_group: str = "feature_mkt",
        dst_group: str = "feature",
    ) -> None:
        self.src_group = src_group
        self.dst_group = dst_group

    def fit(self, df: pd.DataFrame) -> None:
        return

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.columns, pd.MultiIndex):
            return df
        top_level = set(df.columns.get_level_values(0))
        if self.src_group not in top_level:
            return df

        src_block = df[self.src_group]
        # 重命名 fields_group level 为 dst_group
        renamed = src_block.copy()
        renamed.columns = pd.MultiIndex.from_product(
            [[self.dst_group], src_block.columns]
        )
        merged = pd.concat(
            [df.drop(columns=self.src_group, level=0), renamed],
            axis=1,
        )
        # 列排序保持稳定（feature 组放一起，原顺序优先）
        merged = merged.sort_index(axis=1)
        return merged

    def is_for_infer(self) -> bool:
        return True
