# 因子实验台 / Factor Lab

本目录是"随时测新因子并自动入对比表"的核心配置区。每个 `<exp_name>.json` 描述一组待测因子白名单，运行时会被注入到 `configs/models/lgbm_lab.yaml` 的 `model_meta.combined_factors_json`，与生产线 v18-v2 完全隔离。

## 1. 工作流（3 步走）

```
新建 <exp>.json  →  python python/run_experiment.py --model lgbm_lab --freq daily --lab <exp>
                →  python python/factor_lab.py compare
```

- 第一步：在本目录新建 `<exp>.json`（结构见下方"JSON 规范"）。
- 第二步：单切分 `--lab <exp>`，可选追加 `--rolling-mode walk_forward --rolling-cfg configs/rolling/expanding_halfyearly.yaml` 跑 WF。
- 第三步：用 `python/factor_lab.py compare` 一键拉对比表（自动收录所有 lab + v18-v2 / lgbm_ext 锚点）。

## 2. 命名约定

- 文件名：`<exp_name>.json`，`<exp_name>` 须满足 `^[a-zA-Z0-9_]+$`，作为 `--lab` 参数传入。
- 产物自动落到 `artifacts/models/latest_lgbm_lab_<exp_name>_daily.pkl` 与同名 `_meta.json`。
- 特殊保留名：
  - `base_only`：哨兵值，**不读 json**，跑纯 20 维 base 因子（基线锚点）。
  - 其它名字：必须有对应 `.json`。

## 3. JSON 规范

最小必填字段：

```json
{
  "factors": [
    { "name": "log_signed_volume" },
    { "name": "close_location" }
  ]
}
```

可选 metadata（强烈推荐填写，便于审计）：

| 字段 | 说明 |
| --- | --- |
| `created_at` | ISO 时间戳 |
| `purpose` | 一行话讲清这次实验想验证什么假设 |
| `expected_total_features` | 预期总维度，例如 "20 base + 15 = 35" |
| `compare_to` | 主对比基线（建议 `base_only` 或 `v18_v2`） |
| `n_factors` | factors 数组长度（自检字段） |
| `rationale` | 入选这些因子的理由（list[str]） |
| `warning` | 已知风险或反例证据（list[str]） |

约束：
- `factors[].name` 必须在 `[python/features/combined_json_factors.py](../../python/features/combined_json_factors.py)` 的 `_EXPR_BY_JSON_NAME` 中有表达式映射；否则训练阶段会报 `ValueError`。
- 若需要新增因子，先在 `_EXPR_BY_JSON_NAME` 加映射，再在 json 里引用。

## 4. 与生产线的边界

- **不影响**：`configs/combined_factors_df.json`、`configs/models/lgbm.yaml`、`latest_lgbm_daily.pkl`、`predict_live.py`。
- **不影响**：`configs/combined_factors_df_extension.json`、`latest_lgbm_ext_daily.pkl`（lgbm_ext 旧实验保留）。
- 本目录所有实验跑 `--model lgbm_lab`，模型 pkl 与 meta 自带 `lgbm_lab_<exp>_` 前缀，互不污染。

## 5. 已有实验

| 文件 | 因子数 | 总维度 | 说明 |
| --- | --- | --- | --- |
| `base_only.json` | 0 | 20 | 哨兵：base-only 基线锚点；运行时不读文件 |
| `new_md_full15.json` | 15 | 35 | new.md 全量 15 因子（含与 v18-v2 重叠的 8 个）；用于看 new.md 整套相对纯 base 的边际 |

## 6. 常用命令速记

```bash
# 单切分（约 2 分钟）
python python/run_experiment.py --model lgbm_lab --freq daily --lab base_only
python python/run_experiment.py --model lgbm_lab --freq daily --lab new_md_full15

# 完整 WF（约 14 分钟）
python python/run_experiment.py --model lgbm_lab --freq daily --lab new_md_full15 \
    --rolling-mode walk_forward --rolling-cfg configs/rolling/expanding_halfyearly.yaml

# 拉对比表（写入 artifacts/factor_lab_compare.md）
python python/factor_lab.py compare --out artifacts/factor_lab_compare.md

# 列出已有 lab 与跑过/未跑过状态
python python/factor_lab.py list
```
