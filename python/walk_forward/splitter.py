"""
Walk-Forward 时间窗口切分器

层级位置：Feature → Model 的时间序列分割前置逻辑
输入：全局回测区间 [global_start, global_end] + 窗口参数
输出：List[WFSplit]，每个 WFSplit 描述一折的 train/valid/test 边界

设计原则：
- 使用交易日历（A股）计数，避免自然日对节假日的误差
- 严格防止未来泄漏：test_start > valid_end > train_end
- 支持 rolling（固定训练窗口）与 expanding（扩张训练窗口）两种模式
- step_days 等于 test_window_days 时 OOS 区间恰好不重叠（推荐）
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence


@dataclass(frozen=True)
class WFSplit:
    """单折 Walk-Forward 时间边界。

    所有日期字符串格式：'YYYY-MM-DD'。
    """

    fold_idx: int
    train_start: str
    train_end: str
    valid_start: str
    valid_end: str
    test_start: str
    test_end: str

    def __str__(self) -> str:
        return (
            f"Fold[{self.fold_idx:02d}] "
            f"train=[{self.train_start}, {self.train_end}] "
            f"valid=[{self.valid_start}, {self.valid_end}] "
            f"test=[{self.test_start}, {self.test_end}]"
        )


class WalkForwardSplitter:
    """Walk-Forward 交易日窗口生成器。

    Args:
        calendar: A 股交易日序列（list/ndarray of datetime-like），由 qlib D.calendar() 获取。
        global_start: 数据可用起始日（训练集最早日期），格式 'YYYY-MM-DD'。
        global_end:   数据可用截止日（测试集最晚日期），格式 'YYYY-MM-DD'。
        train_window: 训练窗口交易日数量（rolling 模式为固定长度；expanding 模式为最短长度）。
        valid_window: 验证窗口交易日数量（用于早停/超参选择）。
        test_window:  OOS 测试窗口交易日数量（每折向前预测的长度）。
        step:         每折向前滑动的交易日数量（建议等于 test_window，确保 OOS 不重叠）。
        mode:         'rolling' 固定训练窗口 | 'expanding' 扩张训练窗口（起始固定）。
        embargo_days: valid_end 与 test_start 之间的额外间隔交易日（防止自相关泄漏）。
                      日频策略一般设 0；高频策略建议 1-5。

    示例（A 股日频，3 年训练 + 6 月验证 + 3 月 OOS，步长 = OOS，rolling）:
        splitter = WalkForwardSplitter(
            calendar=qlib_calendar,
            global_start="2018-01-02",
            global_end="2026-04-28",
            train_window=756,   # ≈ 3 × 252
            valid_window=126,   # ≈ 0.5 × 252
            test_window=63,     # ≈ 0.25 × 252
            step=63,
            mode="rolling",
        )
        splits = splitter.generate()
    """

    def __init__(
        self,
        calendar: Sequence,
        global_start: str,
        global_end: str,
        train_window: int,
        valid_window: int,
        test_window: int,
        step: int,
        mode: str = "rolling",
        embargo_days: int = 0,
    ) -> None:
        if mode not in ("rolling", "expanding"):
            raise ValueError(f"mode 必须为 'rolling' 或 'expanding'，收到: {mode}")
        if train_window <= 0 or valid_window < 0 or test_window <= 0 or step <= 0:
            raise ValueError("train_window/test_window/step 必须 > 0，valid_window 必须 >= 0")
        if step > test_window:
            import warnings
            warnings.warn(
                f"step={step} > test_window={test_window}，OOS 区间之间将存在间隙，"
                "部分市场日期不会被回测覆盖。",
                stacklevel=2,
            )

        # 将 calendar 转为 'YYYY-MM-DD' 字符串列表
        self._cal: List[str] = [
            str(d)[:10] if not isinstance(d, str) else d[:10]
            for d in calendar
        ]
        self._global_start = global_start[:10]
        self._global_end = global_end[:10]
        self.train_window = train_window
        self.valid_window = valid_window
        self.test_window = test_window
        self.step = step
        self.mode = mode
        self.embargo_days = embargo_days

        # 裁剪日历到全局区间
        self._dates: List[str] = [
            d for d in self._cal if self._global_start <= d <= self._global_end
        ]
        if len(self._dates) == 0:
            raise ValueError(
                f"日历在 [{global_start}, {global_end}] 内无交易日，请检查 global_start/global_end。"
            )

    # ──────────────────────────────────────────────────────────
    # 公共接口
    # ──────────────────────────────────────────────────────────

    def generate(self) -> List[WFSplit]:
        """生成所有折叠切片。

        Returns:
            按时间顺序排列的 WFSplit 列表；若全局区间不足以生成一个完整折叠，返回空列表。
        """
        dates = self._dates
        n = len(dates)
        min_fold_len = self.train_window + self.valid_window + self.embargo_days + self.test_window

        if n < min_fold_len:
            raise ValueError(
                f"全局交易日数量 {n} 不足以构成一个完整折叠（需 {min_fold_len} 日）。"
                f"请缩小窗口参数或扩大全局区间。"
            )

        splits: List[WFSplit] = []
        fold_idx = 0
        # 第一折：训练从第 0 根交易日开始
        cursor = 0  # train_start 索引

        while True:
            # ── 计算本折各段索引 ──────────────────────────────
            if self.mode == "rolling":
                train_start_idx = cursor
                train_end_idx = cursor + self.train_window - 1
            else:  # expanding：train 起始固定，尾端随折次前移
                train_start_idx = 0
                train_end_idx = cursor + self.train_window - 1

            valid_start_idx: Optional[int]
            valid_end_idx: Optional[int]
            if self.valid_window > 0:
                valid_start_idx = train_end_idx + 1
                valid_end_idx = valid_start_idx + self.valid_window - 1
                test_start_idx = valid_end_idx + 1 + self.embargo_days
            else:
                valid_start_idx = None
                valid_end_idx = None
                test_start_idx = train_end_idx + 1 + self.embargo_days

            test_end_idx = test_start_idx + self.test_window - 1

            # ── 越界检查 ─────────────────────────────────────
            if test_end_idx >= n:
                break

            # ── valid 边界处理（valid_window=0 时用 train_end 作占位）─
            vs = dates[valid_start_idx] if valid_start_idx is not None else dates[train_end_idx]
            ve = dates[valid_end_idx] if valid_end_idx is not None else dates[train_end_idx]

            splits.append(
                WFSplit(
                    fold_idx=fold_idx,
                    train_start=dates[train_start_idx],
                    train_end=dates[train_end_idx],
                    valid_start=vs,
                    valid_end=ve,
                    test_start=dates[test_start_idx],
                    test_end=dates[test_end_idx],
                )
            )
            fold_idx += 1
            cursor += self.step

        return splits

    def summary(self) -> str:
        """打印所有折叠的日期摘要。"""
        splits = self.generate()
        lines = [
            f"WalkForwardSplitter [{self.mode}] "
            f"train={self.train_window}d valid={self.valid_window}d "
            f"test={self.test_window}d step={self.step}d "
            f"embargo={self.embargo_days}d | "
            f"总折数={len(splits)}"
        ]
        for s in splits:
            lines.append(f"  {s}")
        return "\n".join(lines)
