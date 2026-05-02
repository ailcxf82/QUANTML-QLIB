"""
RDAgent 生成因子的插件目录。

使用方式:
  RDAgent 自动生成的因子文件应放在此目录下，命名格式: alpha_<id>.py
  每个文件必须暴露 ALPHAS: list[AlphaDef] 变量。

文件模板:
    from features.base import AlphaDef

    ALPHAS: list[AlphaDef] = [
        AlphaDef(
            name="RDAGENT_ALPHA_001",
            expr="...",       # 合法的 Qlib 表达式
            group="momentum", # momentum | volume | technical | fundamental
            description="由 RDAgent 生成：...",
        ),
    ]

注意:
  - expr 必须是 Qlib QlibDataLoader 支持的表达式，引用 $field 或内置算子
  - 新增因子后无需修改任何已有代码，run_experiment.py 会自动发现并加载
  - 若某因子 IC 不达标，将 enabled=False 即可关闭，不需删除文件
"""
