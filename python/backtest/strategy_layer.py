"""
策略层（Strategy Layer）
=========================
职责：
  - 接收预测信号，结合策略参数，构造 Qlib 策略配置字典
  - 对策略参数做基本合法性检查（topk、n_drop 范围等）
  - 提供策略描述供日志与报告使用

输入: strategy_cfg（dict，来自 YAML）+ pred_score（pd.Series）
输出: Qlib 可直接使用的策略配置 dict

支持策略:
  - TopkDropoutStrategy: 选取预测分前 topk 只，每期换出 n_drop 只
  （扩展时实现 build_custom 即可，不影响现有流程）
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd


class StrategyLayer:
    """
    策略层：将预测信号与策略参数合并，构造 Qlib 策略配置。

    设计原则:
      - 信号注入在此层完成，下游（ExecutionLayer）只看执行配置
      - 参数校验在此层报错，保证错误信息友好
    """

    def build(
        self,
        strategy_cfg: Dict[str, Any],
        signal: pd.Series,
    ) -> Dict[str, Any]:
        """
        构造 Qlib strategy 配置字典。

        Args:
            strategy_cfg: 来自 YAML 的策略配置（含 class/module_path/kwargs）
            signal:       模型预测打分序列（MultiIndex datetime×instrument）

        Returns:
            注入了 signal 的策略配置 dict，可直接传入 qlib.backtest.backtest()
        """
        class_name = strategy_cfg["class"]
        self._validate_params(class_name, strategy_cfg.get("kwargs", {}))

        return {
            "class": class_name,
            "module_path": strategy_cfg["module_path"],
            "kwargs": {
                **strategy_cfg.get("kwargs", {}),
                "signal": signal,
            },
        }

    def describe(self, strategy_cfg: Dict[str, Any]) -> str:
        """返回策略的可读描述字符串，用于日志和报告。"""
        class_name = strategy_cfg.get("class", "Unknown")
        kwargs = strategy_cfg.get("kwargs", {})

        if class_name == "TopkDropoutStrategy":
            topk = kwargs.get("topk", "?")
            n_drop = kwargs.get("n_drop", "?")
            hold_thresh = kwargs.get("hold_thresh", 0.5)
            return (
                f"TopkDropoutStrategy("
                f"持仓={topk}只, 每期换出最多={n_drop}只, "
                f"持有阈值={hold_thresh})"
            )
        return f"{class_name}({kwargs})"

    def _validate_params(self, class_name: str, kwargs: Dict[str, Any]) -> None:
        """对已知策略类型做参数范围校验。"""
        if class_name == "TopkDropoutStrategy":
            topk = kwargs.get("topk", 50)
            n_drop = kwargs.get("n_drop", 5)
            if not isinstance(topk, int) or topk <= 0:
                raise ValueError(f"TopkDropoutStrategy.topk 必须为正整数，当前: {topk}")
            if not isinstance(n_drop, int) or n_drop < 0:
                raise ValueError(f"TopkDropoutStrategy.n_drop 必须为非负整数，当前: {n_drop}")
            if n_drop >= topk:
                raise ValueError(
                    f"TopkDropoutStrategy.n_drop({n_drop}) 不应 >= topk({topk})，"
                    "否则每期会清空全部持仓"
                )
