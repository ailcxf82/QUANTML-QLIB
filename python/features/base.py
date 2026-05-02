"""
因子基类定义。

层级位置: Data -> Feature -> Model -> Signal -> Portfolio -> Backtest -> Evaluation
职责: 定义单因子的数据契约，兼容 Qlib QlibDataLoader 与 RDAgent 输出接口。

RDAgent 兼容约定:
  - 生成的因子文件放在 python/features/rdagent/ 目录
  - 每个文件必须暴露 ALPHAS: list[AlphaDef] 变量
  - expr 必须是合法的 Qlib 表达式字符串
"""
from __future__ import annotations

from dataclasses import dataclass, field


# 合法的因子分组，用于 feature_groups 过滤
VALID_GROUPS = frozenset({"momentum", "volume", "technical", "fundamental"})


@dataclass
class AlphaDef:
    """单因子定义。

    属性:
        name        因子名（QlibDataLoader config 中的 name，全大写下划线风格）
        expr        Qlib 表达式字符串（引用 $field 或 Qlib 算子）
        group       因子组别，必须是 VALID_GROUPS 之一
        description 中文描述，说明因子含义与预期方向
        enabled     是否启用（设为 False 可临时关闭，不需删除代码）

    示例（兼容 RDAgent 输出格式）:
        AlphaDef(
            name="MY_ALPHA",
            expr="Mean($close_qfq,5)/Mean($close_qfq,20)-1",
            group="momentum",
            description="5日均线相对20日均线偏离，正值表示短期强势",
        )
    """

    name: str
    expr: str
    group: str
    description: str
    enabled: bool = field(default=True)

    def __post_init__(self) -> None:
        if self.group not in VALID_GROUPS:
            raise ValueError(
                f"AlphaDef '{self.name}' 的 group='{self.group}' 无效，"
                f"合法值：{sorted(VALID_GROUPS)}"
            )
        if not self.name:
            raise ValueError("AlphaDef.name 不能为空")
        if not self.expr:
            raise ValueError(f"AlphaDef '{self.name}' 的 expr 不能为空")
