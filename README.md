# QuantML-Qlib

基于 Qlib 的多模型 × 多频率量化选股实验框架（MLP / LSTM / GRU / Transformer / LightGBM × 日频 / 分钟频）。

## 目录说明

```
configs/
  base.yaml                  公共配置（qlib_init / 特征 / 策略）
  models/
    mlp.yaml                 MLP（DNNModelPytorch，flat 特征）
    lstm.yaml                LSTM（pytorch_lstm_ts，时序）
    gru.yaml                 GRU（pytorch_gru_ts，时序）
    transformer.yaml         Transformer（pytorch_transformer_ts，时序）
    lgbm.yaml                LightGBM（LGBModel，flat 特征）
  freq/
    daily.yaml               日频配置（时间切分 / executor / exchange）
    minute.yaml              分钟频模板（数据就绪后启用）
  mlp_daily_backtest.yaml    历史兼容配置（保留，可直接使用）

python/
  run_experiment.py          通用入口（支持 --model / --freq 组合）
  run_mlp_daily_backtest.py  旧版 MLP 入口（保留）
  diagnose_data.py           数据诊断工具

artifacts/<run_id>/
  signals.parquet            测试区间预测分数
  backtest_report.parquet    逐日组合收益 / 基准 / 换手
  indicator.parquet          回测成交指标（如有）
  metrics.json               核心评估指标 + 阶段耗时
```

## 运行环境

请在 `qlib_zhengshi` 虚拟环境中执行：

```powershell
pip uninstall -y qlib
pip install -r requirements.txt
```

若本机是 Python 3.13，建议切换到 Python 3.10/3.11 后再安装 `pyqlib`，兼容性更稳定。

数据目录默认为 `D:/qlib_data/qlib_data`，可在 `configs/base.yaml` 的 `qlib_init.provider_uri` 中修改。

## 快速运行（多模型多频率）

```powershell
# LightGBM 日频（最快，推荐验证环境时先跑此命令）
python python/run_experiment.py --model lgbm --freq daily

# MLP 日频
python python/run_experiment.py --model mlp --freq daily

# LSTM 日频（时序模型，需要更长训练时间）
python python/run_experiment.py --model lstm --freq daily

# GRU 日频
python python/run_experiment.py --model gru --freq daily

# Transformer 日频
python python/run_experiment.py --model transformer --freq daily

# 旧接口（向下兼容，直接指定完整配置文件）
python python/run_experiment.py --config configs/mlp_daily_backtest.yaml
```

## 可用模型对比

| 模型        | dataset_type | 模块                               | 特点                          |
|-------------|--------------|------------------------------------|-----------------------------|
| mlp         | flat         | pytorch_nn.DNNModelPytorch         | 快速基线，无时序依赖          |
| lgbm        | flat         | gbdt.LGBModel                      | 无 GPU 需求，训练最快         |
| lstm        | ts           | pytorch_lstm_ts.LSTM               | 时序建模，需 TSDatasetH       |
| gru         | ts           | pytorch_gru_ts.GRU                 | 时序建模，结构比 LSTM 轻量    |
| transformer | ts           | pytorch_transformer_ts.TransformerModel | 注意力机制，batch 较小   |

## 扩展分钟频（数据就绪后）

1. 确认 `D:/qlib_data/qlib_data/calendars/1min.txt` 存在
2. 在 `configs/freq/minute.yaml` 将 `data_ready: false` 改为 `true`，并补充分钟级特征表达式
3. 执行：`python python/run_experiment.py --model mlp --freq minute`

## 可调参数

### 回测收益口径与已有结果核对

绩效指标、净值和回撤使用扣费后日收益 `return - cost`。导出的
`backtest_report.parquet` 保留 Qlib 原始 `return`（毛收益）、`cost`（费用率）
以及账户字段，并新增 `net_return`；`cum_return` 是扣费后累计净值。
缺少或无效的费用数据会报错，不能默认当成零费用。

旧产物保持原样。可用 `scripts/audit_saved_costs.py` 从已有日收益、成交金额、
逐笔费用和换手率核对扣费前后表现；该脚本不训练、不预测、不执行撮合回测。
输出目录必须是新目录，审计会记录源文件哈希并核对逐日账户与换手率。

```powershell
python scripts/audit_saved_costs.py `
  --run-dir artifacts/lgbm_optuna_daily_f2c798bb `
  --requested-end 2026-10-15 `
  --calendar D:/qlib_data/qlib_data/calendars/day.txt `
  --output artifacts/cost_audit/example_new_directory
```

默认 `daily` 配置的测试截止日现为 `2026-10-15`，训练与验证窗口不变。
这是配置目标，不代表已有结果覆盖到该日；核对脚本列出缺失交易日，
不补零收益、不延长旧持仓、不生成新的交易。

需要用已保存的39特征 `lgbm_optuna` 模型生成新预测并回测时，使用
`scripts/replay_frozen_lgbm.py --output <全新目录> --factor-workspace <原始因子目录>`。
它验证外部因子历史输出的来源，重算当前数据上的因子与预测，仅执行一次回测，
不训练模型、不更新 `latest`。保存配置、模型副本、输入特征、预测、扣费报告及哈希核对。
默认区间为2025年初至2026-10-15；当前数据版本的历史预测可能与旧报告不同。
为保持原实验口径，该入口沿用区间成分股集合与 `000905.SZ` 基准配置；
该代码不是中证500指数代码，不能将其基准比较解读为相对中证500的超额收益。

若财务与资金流字段断档，`scripts/repair_replay_data.py --output <全新目录>`
使用已有 Tushare 环境凭据下载 `fina_indicator_vip` 与原口径 `moneyflow_dc`，
将原始响应、核对记录和 Qlib 补充字段保存在独立目录。仅填充财务字段
2026-03-31 后、资金流 2026-07-09 后的缺失值，已有非空值不改动。
新补财务值从公告后下一交易日起生效；歧义记录不选取，历史原库与数据商
修订版本不因此获得完整 PIT 保证。补齐成功后，重放入口增加
`--data-repair <补齐目录>`，由只读的双数据源提供器读取补充字段。
下载中断时可加 `--resume` 复用已保存的 API 页；已写完的补充数据不覆盖。

日频撮合通过 `backtest.quote_contract: tushare_qfq_v1` 适配全字段行情库：
涨跌幅取 `$pct_chg / 100`，成交及日末估值统一取 `$close_qfq`。
适配仅作用于 Exchange 的内存报价，不重写共享字段，也不改变模型特征。
缺失价格或涨跌幅时禁止成交，原9.5%统一阈值和费用保持原配置；
该阈值仍不是按证券类别和日期区分的完整交易所涨跌停规则。

仅复用既有预测重跑撮合（不加载模型、不训练、不生成预测）：

```powershell
python scripts/replay_saved_predictions.py `
  --source artifacts/frozen_model_replay/20261029_lgbm_optuna_to_20261015_v2 `
  --output artifacts/execution_replay/example_new_directory
```

该入口要求全新目录，保存相同日期的扣费后对比、逐笔报价核对和持仓估值重建。
预测文件逐字节复用；验证每笔成交的涨跌幅、成交价、每日账户、现金费用和源文件不变。

- **特征与处理器**：`configs/base.yaml` → `dataset.kwargs.handler`
- **时间切分**：`configs/freq/daily.yaml` → `dataset_segments`
- **模型超参**：`configs/models/<model>.yaml` → `model.kwargs`
- **交易成本**：`configs/freq/daily.yaml` → `backtest.fee_model`；历史配置仍可使用 `exchange_kwargs` 中的合并费率。
- **策略参数**：`configs/base.yaml` → `strategy.kwargs`（`topk` / `n_drop`）

2026-10-29 日频费用修正：佣金双边万3.5（按含规费假设），每笔最低5元仅作用于佣金；
印花税卖出0.05%，过户费双边0.001%。税费参照 https://one.sse.com.cn/onething/gptz/ 。
当前日频滑点按用户指定单边0.05%假设，以现金等价成本另列，计入净收益，不改变基准成交价，
不再附加未定义的缓冲。新分项配置不能与历史合并费率同时使用，以免重复计费。

用已修复报价的结果作为基线，仅修改费用并复用预测：

```powershell
python scripts/replay_saved_predictions.py `
  --source artifacts/execution_replay/20261029_lgbm_to_20261015_v1 `
  --fee-config configs/freq/daily.yaml `
  --output artifacts/fee_replay/example_new_directory
```

输出包含 `trade_cost_components.csv`（逐笔）、`daily_cost_components.csv`（逐日）、
`cost_components.json`（合计）、`comparison_net.csv` 和同区间扣费后对比报告。
新费用会影响可用资金、持仓数量和复利路径，不应把旧费用简单加回当作新回测。

2026-10-29 的 5bp 重跑结果保存于 `artifacts/fee_replay/20261029_lgbm_slip5bp_to_20261015_v1/`，
与此前独立保留的 10bp 版本比较。`investment_details/index.html` 提供可筛选的逐笔成交、
费用、FIFO已实现盈亏、个股汇总、每日持仓及期末持仓，附中文表头CSV。
用 `scripts/build_investment_details.py --run <已完成回测目录>` 可从保存结果生成明细，
不再运行回测。数量为复权数量，价格为前复权价，不能直接当作实盘下单股数。

仅比较排除创业板和科创板的影响时，运行
`scripts/replay_saved_predictions.py --source <已修复报价的结果目录> --output <全新目录> --exclude-growth-boards`。
它在Topk选股前过滤创业板（深市3开头）与科创板（沪市688/689开头），保留其他股票的既有分数，
继承基线的全部策略与成本参数；不再标准化、生成预测或训练，也不能同时传入 `--fee-config`。
输出保存原始预测副本、过滤后预测、排除代码清单与原候选池对比。

每周一次的低换手候选配置位于 `configs/strategy/weekly_top5_drop1.yaml`：
首次有信号时建仓，此后每个ISO周首个交易日读取前一交易日分数，持有5只、最多替换1只；
周内不重试、不新增止盈止损交易。保留全部原候选板块，沿用日频报价和5bp滑点。
使用 `scripts/replay_saved_predictions.py --source <全板块基线> --output <全新目录> --strategy-config configs/strategy/weekly_top5_drop1.yaml`
可独立复现；不得同时修改费用或过滤板块。现有日频基线配置未被候选策略替换。
三组固定周调仓试验及分段比较在 `artifacts/weekly_strategy/20261029_all_boards_slip5bp_v1/report.html`。
该候选是在已观察历史区间上选择，尚未通过新的独立样本或实际执行验证。
