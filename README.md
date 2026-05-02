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

- **特征与处理器**：`configs/base.yaml` → `dataset.kwargs.handler`
- **时间切分**：`configs/freq/daily.yaml` → `dataset_segments`
- **模型超参**：`configs/models/<model>.yaml` → `model.kwargs`
- **交易成本**：`configs/freq/daily.yaml` → `backtest.exchange_kwargs`
- **策略参数**：`configs/base.yaml` → `strategy.kwargs`（`topk` / `n_drop`）
