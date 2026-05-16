# QuantML-Qlib 工程概览与命令速查

> 版本：v1 · 2026-05-16  
> 详细重训/数据治理流程见 [retrain_playbook.md](./retrain_playbook.md)

---

## 工程定位

**QuantML-Qlib** 是基于 Qlib 的 A 股多模型日频选股框架，核心链路：

```
数据 → 因子 → 训练 → 回测 → 信号 → 实盘预测
```

分层架构（与代码目录对应）：


| 层级                   | 模块位置                              | 职责              |
| -------------------- | --------------------------------- | --------------- |
| Data                 | `python/data/`                    | 数据更新、校验         |
| Feature              | `python/features/`                | 因子注册与 JSON 因子合并 |
| Model                | `python/run_experiment.py`        | 训练、MLflow 记录    |
| Signal               | 实验产物 `signals.parquet`            | 截面预测分数          |
| Portfolio / Backtest | `python/backtest/`                | 策略、执行、组合、分析     |
| Evaluation           | `artifacts/<run_id>/metrics.json` | Sharpe、IC、回撤等   |
| Live                 | `python/predict_live.py`          | 每日 TopK 选股      |


---

## 目录结构（精简）

```
configs/
  base.yaml                 公共配置（qlib、特征、策略）
  models/<model>.yaml       各模型超参
  freq/<freq>.yaml          日频 / 重训 / 生产等频率变体
  combined_factors_df.json  主力因子集
  factor_lab/*.json         因子实验台

python/
  run_experiment.py         训练 + 回测主入口
  predict_live.py           实盘/准实盘预测
  factor_lab.py             因子实验对比
  evaluate_factors.py       单因子 IC 评估
  backtest/                 分层回测引擎
  features/                 因子工程

artifacts/<run_id>/         单次实验产物（按 run_id 隔离）
artifacts/models/           上线模型 latest_*.pkl
predictions/<date>/         每日选股结果

scripts/
  check_qlib_data_fingerprint.py   数据指纹 snapshot / diff / list
```

配置合并顺序：`**base.yaml` → `models/<model>.yaml` → `freq/<freq>.yaml**`

---

## 模型支持


| 模型           | 命令参数          | dataset | 特点             |
| ------------ | ------------- | ------- | -------------- |
| LightGBM     | `lgbm`        | flat    | 最快，无 GPU，推荐首选  |
| MLP          | `mlp`         | flat    | 神经网络快速基线       |
| LSTM         | `lstm`        | ts      | 时序建模           |
| GRU          | `gru`         | ts      | 比 LSTM 轻量      |
| Transformer  | `transformer` | ts      | 注意力机制，batch 较小 |
| LightGBM Lab | `lgbm_lab`    | flat    | 因子实验台，不影响主力模型  |


---

## 运行环境

在 `**qlib_zhengshi**` 虚拟环境中执行：

```powershell
pip uninstall -y qlib
pip install -r requirements.txt
```

- 数据目录默认：`D:/qlib_data/qlib_data`（可在 `configs/base.yaml` 的 `qlib_init.provider_uri` 修改）
- Python 建议 3.10 / 3.11（3.13 下 pyqlib 兼容性较差）

---

## 核心命令速查

### 1. 实验训练 + 回测

```powershell
# 最快验证环境（推荐先跑）
python python/run_experiment.py --model lgbm --freq daily

# 其他模型
python python/run_experiment.py --model mlp --freq daily
python python/run_experiment.py --model lstm --freq daily
python python/run_experiment.py --model gru --freq daily
python python/run_experiment.py --model transformer --freq daily

# 旧接口：直接指定完整配置文件
python python/run_experiment.py --config configs/mlp_daily_backtest.yaml

# 冻结数据训练（可复现，月度重训用）
python python/run_experiment.py --model lgbm --freq daily_retrain

# 上线前 sanity（短窗口，冻结路径）
python python/run_experiment.py --model lgbm --freq daily_prod_retrain

# Walk-forward 走步进验证
python python/run_experiment.py --model lgbm --freq daily_retrain `
  --rolling-mode walk_forward `
  --rolling-cfg configs/rolling/expanding_halfyearly.yaml
```

### 2. 实盘预测（每日，不重训）

```powershell
# 用上线模型生成今日 TopK
python python/predict_live.py --model lgbm --freq daily --mode predict-only

# 指定日期
python python/predict_live.py --model lgbm --freq daily --mode predict-only --date 2026-05-16

# 指定 TopK
python python/predict_live.py --model lgbm --freq daily --mode predict-only --topk 10
```

产物：`predictions/<YYYY-MM-DD>/selection.csv`、`report.html`、`pred_score.parquet`

### 3. 数据指纹

```powershell
# 拍摄快照
python scripts/check_qlib_data_fingerprint.py snapshot --label my_label

# 列出历史快照
python scripts/check_qlib_data_fingerprint.py list

# 对比两份快照（训练区间是否被污染）
python scripts/check_qlib_data_fingerprint.py diff <early>.json <late>.json

# 指定冻结路径
python scripts/check_qlib_data_fingerprint.py snapshot `
  --provider-uri D:/qlib_data/qlib_data_train_202605 `
  --label train_baseline_202605
```

### 4. 因子评估与实验台

```powershell
# 单因子 IC / RankIC / ICIR
python python/evaluate_factors.py

# 实验台：训练某一 lab
python python/run_experiment.py --model lgbm_lab --lab <lab_name> --freq daily_retrain

# 实验台：多 run 指标对比
python python/factor_lab.py compare
```

### 5. 诊断与审计

```powershell
python python/diagnose_data.py              # 数据质量
python python/diagnose_backtest_result.py   # 回测返回结构
python python/audit_compare_runs.py         # 多 run metrics 横向对比
```

### 6. 测试

```powershell
pytest tests/python/
```

---

## 频率别名说明


| `--freq`             | 用途             | 数据路径                           |
| -------------------- | -------------- | ------------------------------ |
| `daily`              | 研究 / 环境验证      | 默认 LIVE 或 base 配置              |
| `daily_retrain`      | 月度可复现训练 + 回测   | 冻结快照 `qlib_data_train_<MONTH>` |
| `daily_prod`         | 短窗口 sanity     | LIVE                           |
| `daily_prod_retrain` | 短窗口 sanity（冻结） | 冻结快照                           |
| `minute`             | 分钟频（需数据就绪）     | 见 `configs/freq/minute.yaml`   |


---

## 四条运维流程（摘要）


| 流程         | 触发时机                   | 主要动作                                            |
| ---------- | ---------------------- | ----------------------------------------------- |
| **A 日常预测** | 每日 16:00~16:30（dump 后） | `predict_live.py --mode predict-only`           |
| **B 重训**   | 每月 / 因子改动 / IC 退化      | 冻结快照 → `daily_retrain` → 门禁 → 替换 `latest_*.pkl` |
| **C 回滚**   | 新模型上线 3 天内表现下滑         | 从 `artifacts/models/_archive/` 恢复 pkl           |
| **D 每周巡检** | 周一开盘前                  | walk_forward 看最近 fold IC                        |


流程 0（每日 dump）：**仅 `DumpDataUpdate` 增量**，禁止 `DumpDataAll` 全量重写。详见 [retrain_playbook.md](./retrain_playbook.md)。

---

## 关键路径

```
# 配置
configs/base.yaml
configs/models/<model>.yaml
configs/freq/daily_retrain.yaml          # 冻结 provider_uri（月度改）

# 产物
artifacts/<run_id>/metrics.json
artifacts/<run_id>/signals.parquet
artifacts/<run_id>/report.html
artifacts/models/latest_lgbm_daily.pkl   # 当前上线模型
artifacts/models/_archive/               # 历史模型备份

# 预测
predictions/<date>/selection.csv

# 数据（双轨）
D:/qlib_data/qlib_data/                  # LIVE：推理用
D:/qlib_data/qlib_data_train_<MONTH>/    # 冻结：训练用
```

---

## 设计原则（简）


| 原则                 | 说明                                                           |
| ------------------ | ------------------------------------------------------------ |
| 训练用冻结、推理用 LIVE     | 训练求复现；推理只用当日截面，CSZScoreNorm 安全                               |
| 不每日重训              | 每日只做 predict-only；重训约 2~4 周或因子/数据变更时                         |
| universe 限定 csi500 | `predict_live` 支持成分股表滞后时回退到最近调整日                             |
| 可复现                | 固定种子、配置快照、run_id、数据指纹                                        |
| 上线门禁               | sharpe / ic_mean / icir / max_drawdown 人工 review 后再替换 latest |


---

## 实验产物说明

每次 `run_experiment.py` 在 `artifacts/<run_id>/` 典型输出：


| 文件                              | 说明                 |
| ------------------------------- | ------------------ |
| `metrics.json`                  | Sharpe、IC、回撤、阶段耗时等 |
| `signals.parquet`               | 测试区间预测分数           |
| `backtest_report.parquet`       | 逐日组合收益 / 基准 / 换手   |
| `model.pkl` / `model_meta.json` | 模型与元信息             |
| `report.html`                   | 可视化报告              |


---

## 相关文档

- [retrain_playbook.md](./retrain_playbook.md) — 数据治理、双轨路径、重训与回滚完整流程
- [README.md](../README.md) — 环境与多模型快速入门
- [configs/factor_lab/README.md](../configs/factor_lab/README.md) — 因子实验台说明
- [docs/lessons/](./lessons/) — 策略反思与踩坑记录

