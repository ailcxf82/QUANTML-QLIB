# 数据治理与重训流程手册

> 版本：v1 · 2026-05-13
> 适用范围：每日更新 qlib 数据 + 日频 LightGBM 模型 + csi500 选股

---

## 一、背景：为什么需要这份手册

2026-05-07 ~ 5-12 期间，`python python/run_experiment.py --model lgbm --freq daily_prod` 在同一 git commit、同一命令、同一测试段下，sharpe 在 **−2.06 ~ +4.78** 之间剧烈抖动。深入排查发现两个根因：

1. **qlib 数据被每日全量 dump 重写**（5/12 20:22 一次性重写 206,226 个特征文件），训练样本的特征值随之漂移，模型输出虽然 deterministic 但输入不稳定；
2. **csi500 成分股表只更新到 2026-04-30**，5 月以来 `predict_live.py` 的 universe 过滤静默回退到全市场，实盘推荐里混入了北交所、ST、科创板等非 csi500 标的。

本手册沉淀本次治理的方案，确保后续：

- ✅ 训练结果可复现（同输入 → 同输出，14 位精度一致）
- ✅ 每日预测的 universe 严格限定在最新一次 csi500 调整快照
- ✅ qlib 数据漂移有指纹可追溯
- ✅ 模型上线 / 回滚有备份机制

---

## 二、核心架构：双轨数据路径

```
D:/qlib_data/
  qlib_data/                    ← LIVE 路径（每日 dump 写入，回填会覆盖）
                                  用途：predict_live.py 实盘推理
                                  特点：包含最新交易日数据，但历史会被回填修改

  qlib_data_train_<YYYYMM>/     ← 冻结快照（月度 robocopy 同步）
                                  用途：run_experiment.py 训练 + 回测
                                  特点：位级稳定，复现性 100%
```


| 路径                                     | 用途      | 谁读它                                      | 更新周期              |
| -------------------------------------- | ------- | ---------------------------------------- | ----------------- |
| `D:/qlib_data/qlib_data/`              | 每日推理    | `predict_live.py`                        | 每天（你的 dump 脚本）    |
| `D:/qlib_data/qlib_data_train_202605/` | 训练 + 回测 | `run_experiment.py --freq daily_retrain` | **月初手动 robocopy** |


---

## 三、本次改动清单

### 3.1 新增文件


| 文件                                       | 用途                                                 |
| ---------------------------------------- | -------------------------------------------------- |
| `scripts/check_qlib_data_fingerprint.py` | qlib 数据指纹快照与对比工具（snapshot / diff / list 三子命令）      |
| `configs/freq/daily_retrain.yaml`        | 在冻结快照上训练的日频配置（继承 daily.yaml 切分，仅覆盖 `provider_uri`） |
| `configs/freq/daily_prod_retrain.yaml`   | 上线前 sanity 检查配置（继承 daily_prod.yaml 切分 + 冻结快照）      |
| `docs/retrain_playbook.md`               | 本手册                                                |


### 3.2 修改文件


| 文件                         | 改动内容                                                                                                                  |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| `python/run_experiment.py` | `_FREQ_ALIASES` 新增 `daily_retrain` / `daily_prod_retrain` 两个频率别名（5 行）                                                 |
| `python/predict_live.py`   | 新增 `_find_latest_universe_anchor()` 辅助函数；`filter_to_universe()` 增加"当日 universe 为空时自动回退到最近成分股调整日"的 fallback 逻辑（约 60 行） |


### 3.3 物理产物（不入 git，已在 `.gitignore` 内）


| 路径                                                                | 用途                                |
| ----------------------------------------------------------------- | --------------------------------- |
| `D:/qlib_data/qlib_data_train_202605/`                            | 5/12 冻结的训练数据快照（1.04 GB，206230 文件） |
| `artifacts/models/_archive/lgbm_daily_202605_pre_freeze.pkl`      | 旧模型备份（含 meta），可回滚                 |
| `artifacts/models/_archive/lgbm_daily_prod_202605_pre_freeze.pkl` | daily_prod 旧模型备份                  |
| `artifacts/_data_fingerprints/<时间戳>_<label>.json`                 | 历次数据指纹快照                          |


---

## 四、两条核心流程（必读）

整个系统只有两条流程：**A. 日常预测**（每天 1 次）和 **B. 重训**（数据/因子有变更时）。它们的关系如下：

```
        数据源 (BaoStock / Tushare)
                │
       【流程 0】每日 dump（增量）
        15:30 ~ 16:00
                │ ↓ DumpDataUpdate
       LIVE 数据 (D:/qlib_data/qlib_data)        冻结 (D:/qlib_data/qlib_data_train_<MONTH>)
                │                                      │
                │ ↓ 指纹比对训练区间                     │ read-only
                │   （无漂移才允许进流程 A）              │
                │                                      │
        【流程 A】每日                          【流程 B】数据/因子变更触发
        16:00 ~ 16:30                          （月度 / 因子改动）
                │                                      │
        predict_live.py                        robocopy 同步 → 改 retrain.yaml →
        (predict-only)                         run_experiment.py
                │                                      │
        predictions/<date>/                    artifacts/lgbm_daily_retrain_*/
        (今日 TopK 选股)                        (新模型 + 回测)
                │                                      │
        人工 review                            上线门禁 → 替换 latest_*.pkl
                                                       │
                                                       ▼
                                                下一天进入流程 0/A
```

> **关键：流程 0 的 dump 方式决定了模型训练能否复现**。
> 用错 `DumpDataAll`（全量重写）会让训练区间的历史特征被改写，sharpe 直接抖动 ±5 个单位；
> 用对 `DumpDataUpdate`（增量追加）则历史不变，模型 100% deterministic。

---

### 流程 0：每日 dump（最容易出错，最严格管控）

#### 0.1 三条铁律

> 1. **只走增量**：`DumpDataUpdate`（仅追加 `calendars/day.txt` + 给每只票的 .bin 末尾 append 一日），**禁止 `DumpDataAll`**。
> 2. **只写 LIVE**：dump 目标永远是 `D:/qlib_data/qlib_data/`，**冻结快照 `D:/qlib_data/qlib_data_train_<MONTH>/` 任何脚本不得写入**（建议 `attrib +R /S /D` 设为只读）。
> 3. **dump 后必须验证**：完成 dump 立即跑指纹快照 + diff 训练区间，**有任何 hash 变化即视为故障，跳过当日预测**。

#### 0.2 三种 dump 方式对比


| 方式   | qlib 类           | 行为                        | 对训练影响            | 何时用                       |
| ---- | ---------------- | ------------------------- | ---------------- | ------------------------- |
| 全量重写 | `DumpDataAll`    | 把 CSV 重新转 .bin，**全部历史覆盖** | ❌ **致命**：训练特征被改写 | **永远不用**（除非首次建库）          |
| 历史修复 | `DumpDataFix`    | 改指定日期段（除权除息日）             | ⚠️ 取决于修复范围       | 仅复权因子真实修正时，且 fix 范围 ≤ 5 天 |
| 增量追加 | `DumpDataUpdate` | 只追加新日期，历史 .bin 不动         | ✅ 完全无影响          | **每日例行 dump 唯一选择**        |


#### 0.3 推荐的 dump 命令（二选一）

**方案 A：用项目内 `python/data/updater.py`（推荐，已是增量）**

```powershell
# 后台脚本 scripts/daily_dump.py（如果还没有，建议新增）：
python -c "from python.data.updater import BaoStockUpdater; BaoStockUpdater('D:/qlib_data/qlib_data').run()"

# 或用 tushare（需 token）
python -c "from python.data.updater import TushareUpdater; TushareUpdater('D:/qlib_data/qlib_data', token='YOUR_TOKEN').run()"
```

**方案 B：用 qlib 官方仓库的 dump_bin.py（必须用 `update_data_to_bin` 子命令）**

```powershell
# ✅ 正确：增量追加
python <qlib_repo>/scripts/dump_bin.py update_data_to_bin `
  --csv_path D:/qlib_data/daily_csv `
  --qlib_data_1d_dir D:/qlib_data/qlib_data `
  --date_field_name date `
  --symbol_field_name instrument

# ❌ 错误：dump_all 会重写所有历史 .bin，直接破坏训练数据
# python <qlib_repo>/scripts/dump_bin.py dump_all ...
```

#### 0.4 复权 / 基本面字段的特殊处理

A 股两类字段在"增量"语义下仍可能"回填"，需特别小心：


| 字段                             | 为何会回填                        | 应对                                |
| ------------------------------ | ---------------------------- | --------------------------------- |
| **复权因子 `factor`**              | 除权除息发生时，整条 factor 序列重新计算     | 训练用冻结快照（已落地）；OR 锁定历史 factor 文件为只读 |
| **后复权价 `open/high/low/close`** | 等于原始价 × 累计 factor，factor 变即变 | 同上                                |
| **基本面 PE/PB/ROE**              | 财报披露后 T+30~90 日补录            | 财报字段不参与每日 dump；月度重训时再统一刷新         |


#### 0.5 dump 完成后的强制校验

```powershell
# Step 1：记录 dump 后的 LIVE 指纹
python scripts/check_qlib_data_fingerprint.py snapshot `
  --label daily_$(Get-Date -Format yyyyMMdd)_after_dump

# Step 2：与冻结快照基线对比（核心：训练区间不能有任何变化）
$LATEST_BASELINE = Get-ChildItem artifacts/_data_fingerprints/*_train_baseline_*.json `
  | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$LATEST_LIVE = Get-ChildItem artifacts/_data_fingerprints/*_after_dump.json `
  | Sort-Object LastWriteTime -Descending | Select-Object -First 1

python scripts/check_qlib_data_fingerprint.py diff $LATEST_BASELINE.FullName $LATEST_LIVE.FullName

# 预期输出："✅ 训练区间内 features 无变化（只有 calendars/day.txt 多一行 + 部分新增交易日 bin 增长）"
# 异常输出："❌ N 个 features/<code>/*.bin 的 md5 发生变化（涉及训练日期）"
#         → 立即停止预测，排查是否误跑了 dump_all
```

#### 0.6 冻结快照只读保护（一次性设置，强烈推荐）

```powershell
# 在 PowerShell 管理员模式下运行（每次新建冻结快照后执行一次）
attrib +R "D:\qlib_data\qlib_data_train_202605\*" /S /D
# 之后即使脚本 bug 写到错路径，也会立即报错而不是静默污染数据
```

---

### 流程 A：日常预测（每天 1 次，**不重训**）





cd D:\quant_project\qlibQuantData

# 第一步：下载今天的 CSV（先确认 API 通了再跑）

.\incremental_update.ps1 -Universe Csi500 -StartDate 20260513 -EndDate 20260513 -CsvDir D:\qlib_data\csv_data\csi500_update -QlibDir D:\qlib_data\qlib_data -Stage download

# 第二步：确认 CSV 有数据后，转 bin（DumpDataUpdate 增量）

.\incremental_update.ps1 -Universe Csi500 -StartDate 20260509 -EndDate 20260513 -CsvDir D:\qlib_data\csv_data\csi500_update -QlibDir D:\qlib_data\qlib_data -Stage convert





#### A.0 触发时机


| 时机                                  | 操作         |
| ----------------------------------- | ---------- |
| 每日 **16:00 ~ 16:30**（收盘 + dump 完成后） | 必须         |
| 周末 / 节假日                            | 跳过（A股无新数据） |


#### A.1 前置条件检查（30 秒）

```powershell
# 1. 确认今日 dump 已完成（calendars/day.txt 最后一行是今日）
Get-Content D:\qlib_data\qlib_data\calendars\day.txt -Tail 1

# 2. 确认上线模型存在
Get-Item artifacts\models\latest_lgbm_daily.pkl
Get-Item artifacts\models\latest_lgbm_daily_meta.json

# 3. 确认 csi500.txt 存在（即使 end_date 落后于今日也 OK，predict_live 会自动回退到最近调整日）
Get-Item D:\qlib_data\qlib_data\instruments\csi500.txt
```

#### A.2 执行流程

```powershell
# Step 1（可选但推荐）：先做 LIVE 数据指纹快照，便于事后追溯
python scripts/check_qlib_data_fingerprint.py snapshot `
  --label daily_$(Get-Date -Format yyyyMMdd)

# Step 2：用上线模型 + LIVE 数据生成今日 TopK 选股
python python/predict_live.py --model lgbm --freq daily --mode predict-only

# （可选 1）指定预测日期（默认自动取 calendars/day.txt 最后一行）
python python/predict_live.py --model lgbm --freq daily --mode predict-only --date 2026-05-12

# （可选 2）指定 TopK（默认读 configs/live/daily_live.yaml 的 live.topk）
python python/predict_live.py --model lgbm --freq daily --mode predict-only --topk 10
```

#### A.3 验证（1 分钟人工 review）

打开 `predictions/<today>/report.html`，检查 3 个指标：


| 指标              | 健康范围                             | 异常处理                                    |
| --------------- | -------------------------------- | --------------------------------------- |
| **universe 行数** | csi500 应保留 300 ~ 500 条（不应 < 100） | < 100 → 检查 instruments/csi500.txt 是否被截断 |
| **与昨日 top5 重叠** | ≥ 3/5（换手率 ≤ 40%）                 | < 2/5 → 信号剧烈跳动，谨慎执行                     |
| **top1 截面百分位**  | > 90%                            | < 80% → 信号衰减，考虑提前重训                     |


控制台日志必须出现这两行之一：

```
universe 过滤: csi500@2026-05-12 保留 489 / 3196 条        ← 当日 csi500 表有效
universe 过滤: csi500@2026-04-30 保留 347 / 3196 条        ← 自动回退到最近调整日
```

如果出现 `csi500 当日为空且 120 天内无锚点，跳过 universe 过滤` 警告 ⇒ **不要执行交易**，先排查 instruments 表。

#### A.4 产物路径

```
predictions/<YYYY-MM-DD>/
  selection.csv                  TopK 选股清单（直接交易用）
  report.html                    可视化报告
  pred_score.parquet             全市场预测分数（debug 用）
```

---

### 流程 B：重训（数据/因子变更时触发）

#### B.0 何时该重训？

**3 个独立触发分支，满足任一即可启动**：


| 分支             | 触发条件                                                                  | 影响范围            |
| -------------- | --------------------------------------------------------------------- | --------------- |
| **B-1 仅数据更新**  | 距上次冻结 ≥ 30 天 / 数据指纹显示训练区间被回填 / 最近 walk_forward fold ic_mean < 0.025   | 只换冻结快照，配置/因子不动  |
| **B-2 因子更新**   | 改了 `configs/combined_factors_df.json` 或新增 `configs/factor_lab/*.json` | 配置变更触发，数据快照不必换  |
| **B-3 实盘表现退化** | 累计 5 个交易日 top5 平均回报相对 csi500 落后 > 2%                                  | 通常 B-1 + B-2 一起 |


判断当前是哪种触发：

```powershell
# 查最近一次冻结日期
Get-Item D:\qlib_data\qlib_data_train_202605 | Select-Object LastWriteTime

# 查最近一次数据漂移
python scripts/check_qlib_data_fingerprint.py list

# 查最近因子改动（git）
git log --oneline -10 -- configs/combined_factors_df.json configs/factor_lab/
```

---

#### B-1：仅数据更新重训（最常见，每月 1 次）

```powershell
# ─── Step 1：冻结新一份数据快照 ───────────────────────────
$MONTH = Get-Date -Format 20260513
$DEST = "D:\qlib_data\qlib_data_train_$MONTH"

robocopy D:\qlib_data\qlib_data $DEST /MIR /MT:16 /NFL /NDL /NJH /NJS /R:1 /W:1
# robocopy exit code 0/1/2/3 都算成功；≥8 才是失败

# ─── Step 2：记录冻结基线指纹（事后比对用）───────────────
python scripts/check_qlib_data_fingerprint.py snapshot `
  --provider-uri D:/qlib_data/qlib_data_train_$MONTH `
  --label train_baseline_$MONTH

# ─── Step 3：改 configs/freq/daily_retrain.yaml 指向新快照 ──
# 把 qlib_init.provider_uri 从 "D:/qlib_data/qlib_data_train_202605"
# 改成 "D:/qlib_data/qlib_data_train_<MONTH>"
# （同样改 configs/freq/daily_prod_retrain.yaml）

# ─── Step 4：备份当前上线模型 ────────────────────────────
$ARCHIVE = "artifacts/models/_archive"
New-Item -ItemType Directory -Path $ARCHIVE -Force | Out-Null
Copy-Item "artifacts/models/latest_lgbm_daily.pkl" `
          "$ARCHIVE/lgbm_daily_${MONTH}_pre_freeze.pkl"
Copy-Item "artifacts/models/latest_lgbm_daily_meta.json" `
          "$ARCHIVE/lgbm_daily_${MONTH}_pre_freeze_meta.json"

# ─── Step 5：重训 + 回测 ────────────────────────────────
python python/run_experiment.py --model lgbm --freq daily_retrain

# ─── Step 6：上线门禁（人工 review）─────────────────────
# 打开 artifacts/lgbm_daily_retrain_<run_id>/metrics.json，要求：
#   ✓ sharpe        ≥ 上一版 - 0.3      （允许小幅退化）
#   ✓ ic_mean       ≥ 0.025
#   ✓ icir          ≥ 0.13
#   ✓ max_drawdown  ≥ -0.30
# 通过 → 替换 latest 上线模型：
Copy-Item -Force "artifacts/models/latest_lgbm_daily_retrain.pkl" `
                 "artifacts/models/latest_lgbm_daily.pkl"
Copy-Item -Force "artifacts/models/latest_lgbm_daily_retrain_meta.json" `
                 "artifacts/models/latest_lgbm_daily_meta.json"

# ─── Step 7（可选）：daily_prod 短窗口 sanity 检查 ───────
python python/run_experiment.py --model lgbm --freq daily_prod_retrain
# 看 information_ratio > 1.0 即合格；绝对 sharpe 受市场 beta 干扰，不作为单独依据
```

---

#### B-2：因子更新重训（改 combined_factors / factor_lab 时）

**子情况 1：改 `configs/combined_factors_df.json`（主力因子集，影响 latest_lgbm_daily.pkl）**

```powershell
# ─── Step 1：备份当前因子集（防回滚）───────────────────
git diff configs/combined_factors_df.json | Out-File `
  "artifacts/_factor_diffs/diff_$(Get-Date -Format yyyyMMdd_HHmmss).txt"

# ─── Step 2：（可选）先在 walk_forward 上验证因子改动是否有效 ─
python python/run_experiment.py `
  --model lgbm `
  --freq daily_retrain `
  --rolling-mode walk_forward `
  --rolling-cfg configs/rolling/expanding_halfyearly.yaml
# 看 rolling_summary.json：
#   concatenated.ic_mean   是否优于上一版
#   concatenated.sharpe    是否优于上一版
#   fold_stats.icir.std    < 0.12 才算稳定

# ─── Step 3：备份当前上线模型 ────────────────────────────
Copy-Item "artifacts/models/latest_lgbm_daily.pkl" `
          "artifacts/models/_archive/lgbm_daily_pre_factor_$(Get-Date -Format yyyyMMdd).pkl"
Copy-Item "artifacts/models/latest_lgbm_daily_meta.json" `
          "artifacts/models/_archive/lgbm_daily_pre_factor_$(Get-Date -Format yyyyMMdd)_meta.json"

# ─── Step 4：单切分重训（产出上线候选）───────────────────
python python/run_experiment.py --model lgbm --freq daily_retrain

# ─── Step 5：上线门禁（同 B-1 Step 6）─────────────────────
# 通过 → 替换 latest 模型（同 B-1 Step 6 后半段）
```

**子情况 2：改 `configs/factor_lab/<lab>.json`（因子实验台，产出 `latest_lgbm_lab_<lab>_daily.pkl`，不影响主力）**

```powershell
# 实验台不动主力模型；命名隔离到 latest_lgbm_lab_<lab>_*.pkl
python python/run_experiment.py --model lgbm_lab --lab <your_lab_name> --freq daily_retrain

# 对照基线（纯 base 因子）
python python/run_experiment.py --model lgbm_lab --lab base_only --freq daily_retrain

# 看 metrics.json 是否优于基线，决定是否合并到 combined_factors_df.json
```

---

#### B-3：B-1 + B-2 联合重训（实盘表现退化时）

按 **B-1 Step 1~3**（冻结新数据 + 改 retrain yaml） → 然后做完 **B-2** 的因子改动 → 最后跑：

```powershell
python python/run_experiment.py --model lgbm --freq daily_retrain
```

一次拿到"新数据 + 新因子"的复合改进版本。门禁通过后按 B-1 Step 6 替换 latest。

---

### 流程 C：故障回滚（新模型上线后 3 天内表现下滑）

```powershell
# 1. 找到要回滚的备份
Get-ChildItem artifacts/models/_archive/lgbm_daily_*.pkl

# 2. 恢复
$ROLLBACK = "lgbm_daily_202605_pre_freeze.pkl"  # 改成你要的版本
Copy-Item -Force "artifacts/models/_archive/$ROLLBACK" `
                 "artifacts/models/latest_lgbm_daily.pkl"
$META = $ROLLBACK -replace '\.pkl$', '_meta.json'
Copy-Item -Force "artifacts/models/_archive/$META" `
                 "artifacts/models/latest_lgbm_daily_meta.json"

# 3. 验证回滚
python python/predict_live.py --model lgbm --freq daily --mode predict-only
```

### 流程 D：每周巡检（每周一开盘前 8:30，不重训）

```powershell
# 跑 walk_forward 看最近 fold 的 IC 稳定性
python python/run_experiment.py `
  --model lgbm `
  --freq daily_retrain `
  --rolling-mode walk_forward `
  --rolling-cfg configs/rolling/expanding_halfyearly.yaml

# 查看：artifacts/lgbm_daily_retrain_<id>/rolling_summary.json
# 健康阈值：
#   最近 fold ic_mean      > 0.025
#   最近 fold icir         > 0.15
#   fold_stats.icir.std    < 0.12
#   concatenated sharpe    > 1.5
# 任一不达标 → 触发流程 B
```

### 流程 E：数据指纹工具用法

```powershell
# 列出所有历史快照
python scripts/check_qlib_data_fingerprint.py list

# 对比两份快照
python scripts/check_qlib_data_fingerprint.py diff `
  artifacts/_data_fingerprints/<early>.json `
  artifacts/_data_fingerprints/<late>.json

# 指定 provider_uri（默认从 configs/base.yaml 读）
python scripts/check_qlib_data_fingerprint.py snapshot `
  --provider-uri D:/qlib_data/qlib_data_train_202605 `
  --label train_baseline_202605 `
  --sample-size 20
```

---

## 五、验证：本次治理后的实测效果

### 5.1 训练 deterministic 验证

`daily_retrain` 同命令连续跑 3 次：

| run_id   | sharpe             | ic_mean              | signals 行数 | 与第 1 次的 max|diff| |
| -------- | ------------------ | -------------------- | ---------- | ----------------- |
| c63db9f8 | 2.7293608205007740 | 0.029734671062959718 | 190,483    | —                 |
| ba4ca33e | 2.7293608205007733 | 0.029734671062959718 | 190,483    | **0**             |
| ff731ed7 | 2.7293608205007733 | 0.029734671062959718 | 190,483    | **0**             |

`max |diff| = 0, corr = 1.000`，**14 位精度完全一致**。

### 5.2 daily_prod 短窗口 sharpe 对比


| run                    | 路径     | sharpe    | ic_mean    | 复现性       |
| ---------------------- | ------ | --------- | ---------- | --------- |
| 5/7 LIVE (314ec0c8)    | LIVE   | +4.78     | 0.0453     | ❌ 不可复现    |
| 5/11 LIVE (5239532f)   | LIVE   | −2.06     | 0.0350     | ❌ 不可复现    |
| 5/12 LIVE (7f8bcc53)   | LIVE   | +0.66     | 0.0200     | ❌ 不可复现    |
| 5/12 FROZEN (08ff570c) | **冻结** | **+0.73** | **0.0200** | ✅ **可复现** |


冻结后 daily_prod ic_mean、icir 与最新 LIVE 完全一致（信号端逐行 maxdiff=0），剩余微小 sharpe 差异来自 test_end 多覆盖 5/11、5/12 两个交易日。

### 5.3 universe 修复验证

修复前 (`2026-05-12` selection.csv 一版)：

```
top1  603580.SH  非 csi500  ❌
top2  600381.SH  非 csi500  ❌
...
top9  920000.BJ  北交所  ❌❌
```

修复后（同日重跑）：

```
注意: csi500 在 2026-05-12 当日为空，按月调整规则回退到最近调整日 2026-04-30
universe 过滤: csi500@2026-04-30 保留 347 / 3196 条

top1  600535.SH  ✅ csi500
top2  600764.SH  ✅ csi500
top3  600566.SH  ✅ csi500
top4  688100.SH  ✅ csi500
top5  601666.SH  ✅ csi500
... 全部 12 行均在 csi500@2026-04-30 内
```

---

## 六、关键设计原则


| 原则                   | 说明                                                                   |
| -------------------- | -------------------------------------------------------------------- |
| **不每日重训**            | 重训对短期数据回填高度敏感，业界经验是 2~4 周一次。每日只做推理。                                  |
| **训练用冻结、推理用 LIVE**   | 训练求复现性走冻结快照；推理求新鲜度走 LIVE（推理只用当日截面，CSZScoreNorm 不依赖历史，安全）。            |
| **universe 按月规则回退**  | csi500 真实业务规则就是按月调整，月内沿用上次快照。fallback 严格控制在 120 天内（超过视为数据故障）。        |
| **看 IR 不看绝对 sharpe** | daily_prod 测试期太短 + CSI500 大跌时，绝对 sharpe 受市场 beta 干扰大；IR > 1.0 才是真信号。 |
| **指纹监控 + 备份**        | 每次重训前后 snapshot；每次上线前备份旧模型，确保 3 天内可回滚。                               |


---

## 七、FAQ

**Q1: 为什么不直接修复 csi500.txt，把 end_date 改成 9999-12-31？**

A: 每次 qlib dump 都会覆盖 instruments/csi500.txt，手工修改无法持久化。代码层 fallback 才是一次到位的方案。

**Q2: 修复后 sharpe 反而比 LIVE 训练版本（5/12 21:54）的 2.53 高（变成 2.73），为什么？**

A: 冻结快照对应的是某次 dump 后的完整数据状态（含所有回填）；而 LIVE 训练那次的 dump 是更早的状态，那时部分基本面数据还未回填。冻结后 IC 反而略降（0.0386 → 0.0297），但回测的头部选股恰好命中了几个高收益日，所以 sharpe 提升。这是数据快照差异，不是模型改进。

**Q3: 冻结路径 vs LIVE 路径数据完全一样吗？**

A: 冻结当下完全相同（md5 全部一致，已验证）。但每次 LIVE 被 dump 之后，两者就会出现差异，这正是为什么要冻结：让训练数据稳定。

**Q4: predict_live.py 加载的模型是冻结路径训练的，但推理用 LIVE 数据，会不会出问题？**

A: 不会。LGBM 的 `model.pkl` 是 deterministic 的函数 `f(features) → score`，参数固定。推理只依赖**今天的特征截面**，CSZScoreNorm 也是当日截面归一化，不依赖历史完整性。

**Q5: 如何确认 walk_forward 的 fold 3（2026Q1）IC 衰减是真实的，不是数据问题？**

A: 跑 `--freq daily_retrain --rolling-mode walk_forward`，看 fold 3 的 ic_mean。如果在冻结快照上仍 ≈ 0.02，那是真实的市场风格切换；如果显著回升，说明之前 LIVE 的衰减来自数据漂移。

---

## 八、附录：本手册涉及的文件路径速查表

```
# 工具脚本
scripts/check_qlib_data_fingerprint.py

# 配置
configs/freq/daily.yaml                  原始日频（不动）
configs/freq/daily_prod.yaml             原始 sanity 检查（不动）
configs/freq/daily_retrain.yaml          ★ 新增：冻结路径主线训练
configs/freq/daily_prod_retrain.yaml     ★ 新增：冻结路径 sanity 检查
configs/base.yaml                        公共基础（不动）

# 入口脚本
python/run_experiment.py                 训练 + 回测（_FREQ_ALIASES 增 2 行）
python/predict_live.py                   实盘预测（新增 universe fallback）

# 产物目录
artifacts/<run_id>/                      每次 run 产物
artifacts/models/latest_lgbm_daily.pkl   ★ 上线模型（指向最新冻结训练）
artifacts/models/_archive/               ★ 历史模型备份（手动 commit-message 风格命名）
artifacts/_data_fingerprints/            ★ 历次数据指纹快照
predictions/<date>/                      每日选股结果

# 数据
D:/qlib_data/qlib_data/                  LIVE 路径
D:/qlib_data/qlib_data_train_202605/     ★ 冻结快照（月度更新）
```

---

## 九、未完成项 / 可选改进


| 优先级 | 项目                                                                    | 说明                                                             |
| --- | --------------------------------------------------------------------- | -------------------------------------------------------------- |
| P2  | 把 `latest_lgbm_daily_retrain.pkl` 自动 rename 给 `latest_lgbm_daily.pkl` | 当前需手动 Copy-Item，可在 persist_artifacts 加一个 `--upgrade-latest` 开关 |
| P2  | `run_experiment.py` 的 `filter_pred_to_universe` 也加 fallback           | 当前只在区间完全失效时才触发空，回测影响小，但补完更稳健                                   |
| P3  | 把 universe fallback 抽到 `python/backtest/universe.py` 公共模块             | 减少 predict_live 和 run_experiment 两处重复逻辑                        |
| P3  | 把每月冻结流程写成 PowerShell 脚本 `scripts/freeze_qlib_data.ps1`                | 一键执行第 4.3 节的 Step 1+2+4                                        |
| P3  | 在 `predict_live.py` 中输出"换手率 / drop_out 数量"的告警                         | 自动判断当日推荐健康度，超阈值打印 ⚠️                                           |


