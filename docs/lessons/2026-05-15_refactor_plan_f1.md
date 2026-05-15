# F1 重构计划：mega-file 拆分

**日期**: 2026-05-15  
**状态**: 待实施（后续分批进行）

## 现状

`run_experiment.py`（~1430 行）和 `predict_live.py`（~1470 行）各自承担完整链路，违反单文件职责原则。两者共同问题：

- `inject_feature_config` 独立实现了两遍，一旦因子注入逻辑变化，必须双向同步
- 配置加载（`_load_yaml` / `deep_merge`）也有两套微小差异的实现
- `main()` 函数膨胀，三条大分支（单切分 / walk-forward / ensemble）耦合在一处

## 目标目录结构

```
python/
  config/
    loader.py          # _load_yaml, deep_merge, load_and_merge_configs（统一版）
    feature_injector.py # inject_feature_config（唯一实现，run_experiment + predict_live 共用）
  training/
    trainer.py         # train_model, _override_model_seeds, train_predict_ensemble
    walk_forward_runner.py  # run_walk_forward, _train_fold, _persist_fold_artifacts
  inference/
    predictor.py       # generate_pred_score, load_latest_model, train_and_save_model
    stock_selector.py  # select_topk, filter_and_refill, filter_to_universe（live 侧）
  persistence/
    artifact_writer.py # persist_artifacts
  run_experiment.py    # 瘦化为：CLI 解析 + 模块组合调用（~100 行）
  predict_live.py      # 瘦化为：CLI 解析 + 模块组合调用（~150 行）
```

## 实施步骤

1. 新建 `python/config/loader.py`，合并两份 `_load_yaml / deep_merge`
2. 新建 `python/config/feature_injector.py`，移入 `inject_feature_config`；  
   两个入口文件改为 `from config.feature_injector import inject_feature_config`
3. 新建 `python/training/trainer.py`，移入 `train_model / train_predict_ensemble`
4. 新建 `python/training/walk_forward_runner.py`，移入 walk-forward 相关函数
5. 新建 `python/inference/predictor.py`，移入推理相关函数
6. 新建 `python/inference/stock_selector.py`，移入选股过滤逻辑
7. 新建 `python/persistence/artifact_writer.py`，移入 `persist_artifacts`
8. 精简 `run_experiment.py` 和 `predict_live.py` 为薄 CLI 层
9. 同步更新 `tests/` 中对应的 import 路径

## 注意事项

- 所有移动须保证 `sys.path` 设置一致（以 `python/` 为根）
- 先建立测试 smoke test 基线，再做拆分，确保拆前拆后输出一致
- `run_mlp_daily_backtest.py` 在拆分完成后可标注废弃并最终删除
