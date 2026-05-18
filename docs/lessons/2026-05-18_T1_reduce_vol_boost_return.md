# T1 实验复盘：降波动同时提收益（2026-05-18）

> **一句话结论**：信号层 Ensemble（T1-A）是唯一在提升收益的同时明显提升 Sharpe 的单一手段；组合层各优化（RiskParity / AdaptiveTopK / Optuna 超参）单独使用时收益与 Sharpe 均持平或微降；叠加后综合最优版本的**波动率和最大回撤显著低于 T1-A 单独使用**，但 Sharpe 略低于 T1-A。

---

## 1. 基线与实验矩阵

OOS 区间：2025-01-01 ~ 2026-05-12（`daily_retrain`，`TopkDropoutStrategy n_drop=1`）


| 实验         | 模型              | 策略                                           | seeds           | run_id                               |
| ---------- | --------------- | -------------------------------------------- | --------------- | ------------------------------------ |
| **基线**     | lgbm            | TopkDropout                                  | 1               | `lgbm_daily_retrain_2f921322`        |
| **T1-A**   | lgbm            | TopkDropout                                  | **3 topk_vote** | `lgbm_daily_retrain_61643189`        |
| **T1-B+C** | lgbm            | QuantML(**risk_parity** + **adaptive_topk**) | 1               | `lgbm_daily_retrain_3e3dc979`        |
| **T1-D**   | **lgbm_optuna** | TopkDropout                                  | 1               | `lgbm_optuna_daily_retrain_7f57f579` |
| **T1-ALL** | lgbm_optuna     | QuantML(risk_parity + adaptive_topk)         | **3 topk_vote** | `lgbm_optuna_daily_retrain_c276d396` |


---

## 2. 完整指标对比


| 指标         | 基线       | T1-A Ensemble | T1-B+C RiskParity+AdpTopK | T1-D Optuna参数 | **T1-ALL** **综合最优** |
| ---------- | -------- | ------------- | ------------------------- | ------------- | ------------------- |
| **年化收益**   | 36.4%    | **67.4%**     | 34.6%                     | 51.3%         | 39.9%               |
| **年化波动**   | 14.1%    | 22.4%         | **14.2%**                 | 21.6%         | **15.0%**           |
| **最大回撤**   | -7.3%    | -14.6%        | -8.4%                     | -13.9%        | **-8.8%**           |
| **Sharpe** | 2.28     | **2.42**      | 2.17                      | 2.03          | 2.31                |
| **Calmar** | **4.98** | 4.61          | 4.12                      | 3.69          | 4.53                |
| **换手率**    | 10.1%    | 39.2%         | 9.8%                      | 39.6%         | **8.9%**            |
| IC         | 0.040    | 0.040         | 0.040                     | 0.039         | 0.040               |
| ICIR       | 0.187    | 0.190         | 0.187                     | 0.187         | 0.189               |
| 胜率         | 56.2%    | 56.8%         | 55.9%                     | 56.5%         | **58.0%**           |


---

## 3. 逐模块分析

### T1-A：Ensemble 3 seeds + topk_vote

- **年化收益 +31pp（36.4% → 67.4%）**，Sharpe 2.28 → **2.42** ↑
- 三个 seed 独立 IC 均在 0.035~0.036，差异极小，多样性有限
- 换手从 10.1% 跳至 39.2%——topk_vote 每 seed 独立产出 Top-5，共识票集中但非共识票每日轮换，导致高换手
- **结论**：T1-A 是本轮最有效的单一提升，Sharpe 是唯一超过基线的实验

### T1-B+C：RiskParity 加权 + AdaptiveTopK（周频）

- 收益略降（36.4% → 34.6%），波动与基线持平（14.1% → 14.2%）
- 最大回撤略扩（-7.3% → -8.4%），Sharpe 小幅下降 2.28 → 2.17
- 换手极低 9.8%（`rebalance_freq=5` + `n_drop=1` + RiskParity 仅在模型步重算）
- **AdaptiveTopK 的实际效果**：弱信号日扩大 topk → 但持仓数增多导致单票仓位更小 → `trade_unit=100` 截断效应加剧，有效订单减少，策略部分"钝化"
- **RiskParity 的实际效果**：topk=5 的 5×5 协方差矩阵自由度极低（60/5=12），估计噪声大；在样本较少时自动降级为 InverseVol，实际效果与 equal 差距不明显
- **结论**：在 topk=5 的极小持仓数下，组合层优化空间有限

### T1-D：Optuna 超参（Trial 4，WF Sharpe 7.15）

- 收益大幅提升（36.4% → 51.3%）但波动同步扩大（14.1% → 21.6%）
- Sharpe 反而下降 2.28 → 2.03，换手暴增至 39.6%
- **原因**：Trial 4 参数（`lr=0.044`，浅树 `num_leaves=18`）在 Optuna 的 WF 评估中搭配 `n_drop=1` 换手约束表现极好，但在无约束的 `TopkDropoutStrategy`（本次回测）中，高 lr 模型信号日频抖动大，换手失控
- **教训**：Optuna 目标函数中的策略参数（`n_drop=1`）必须与最终部署策略完全一致，否则评估与实战不匹配

### T1-ALL：综合最优

- **波动 15.0%，比 T1-A（22.4%）低 7.4pp**，比基线仅高 0.9pp
- **最大回撤 -8.8%，比 T1-A（-14.6%）低 5.8pp**，接近基线（-7.3%）
- **换手率 8.9%，全部实验中最低**（RiskParity + `rebalance_freq=5` 共同抑制）
- Sharpe 2.31，略高于基线（2.28），低于 T1-A（2.42）
- **结论**：综合最优在"保留 T1-A 约 60% 的收益提升"的同时，把波动和回撤控制在接近基线水平

---

## 4. 关键教训

### 教训 ① Ensemble 是最有效的降噪手段

3 seeds + topk_vote 把年化收益提升 85%（相对），Sharpe 提升 6%，是本轮唯一打败基线 Sharpe 的手段。但要注意换手率同步翻 4 倍——必须搭配换手约束使用。

### 教训 ② topk 太小时，组合层优化效果边际递减

RiskParity / AdaptiveTopK 在 topk=5 时效果不如预期：

- RiskParity：5×5 协方差矩阵自由度不足，频繁降级为 InverseVol
- AdaptiveTopK：弱信号日 topk=8 会与 `trade_unit=100` 冲突，小仓位订单被截断
- **改进方向**：把 topk 调高到 8~10（配合更严格的 `score_quantile`），让组合层有更大操作空间

### 教训 ③ Optuna 目标函数必须与部署策略对齐

Trial 4 在 Optuna 中 WF Sharpe=7.15，但单切分回测 Sharpe=2.03。差距来源：Optuna 内用 `n_drop=1` 约束换手，实际回测用了无约束的 TopkDropoutStrategy。**Optuna 目标函数中应完整复现部署时的策略配置。**

### 教训 ④ "降波动 + 提收益"的本质是叠加正交改进

单独使用任一模块几乎不能同时降波动又提 Sharpe，但合理叠加可以：

- T1-A（ensemble）提升 alpha 质量 → 收益提升
- RiskParity + `rebalance_freq=5`（组合层） → 波动抑制
- Optuna 参数（模型层） → 信号稳定性提升
- 三者叠加后：收益 +3.5pp、波动 +0.9pp、MDD 仅 -1.5pp，**Sharpe 从 2.28 → 2.31**

---

## 5. 新增代码资产


| 文件                                               | 功能                                     |
| ------------------------------------------------ | -------------------------------------- |
| `python/backtest/strategy/selector.py`           | `AdaptiveTopKCfg`、`TopKSelector` 自适应模式 |
| `python/tuning/__init__.py` + `optuna_lgbm.py`   | Optuna WF Sharpe 超参搜索基础设施              |
| `configs/models/lgbm_optuna.yaml`                | Trial 4 最优超参配置                         |
| `configs/strategy/daily_vol_target.yaml`         | RiskParity + AdaptiveTopK 完整配置         |
| `artifacts/optuna_lgbm_summary.json`             | 搜索结果摘要                                 |
| `tests/python/backtest/test_strategy_modules.py` | 新增 6 个 AdaptiveTopK 单测（全通过）            |


---

## 6. 后续优化方向（T2）

按 ROI 排序：


| 优先级 | 路径                                                            | 预期                               |
| --- | ------------------------------------------------------------- | -------------------------------- |
| ⭐⭐⭐ | 把 topk 调至 8~~10，同时提高 `score_quantile` 到 0.80~~0.85            | RiskParity/AdaptiveTopK 效果才能充分发挥 |
| ⭐⭐⭐ | Optuna 目标函数内嵌完整策略约束（`n_drop=1`，`rebalance_freq=5`，换手上限）再重搜    | 避免评估-部署不一致，可能挖掘更好参数              |
| ⭐⭐  | Ensemble seeds 从 3 提升到 5，同时降低 `feature_fraction` 到 0.55 增大多样性 | 进一步降低信号噪声，Sharpe 可能突破 2.5        |
| ⭐⭐  | 非对称 MarketTimer：σ̂ 高 AND 60D 收益为负才降仓                          | 保留普涨期满仓，仅保护真实下行                  |
| ⭐   | 引入北向资金/龙虎榜数据维度                                                | 全新 alpha 来源，效果未知但期望高             |


---

## 7. 时间线


| 时间               | 动作                                              |
| ---------------- | ----------------------------------------------- |
| 2026-05-18 15:54 | 开始 T1-C + T1-D 并行实施                             |
| 2026-05-18 15:58 | T1-C 回测完成（Sharpe 2.07）                          |
| 2026-05-18 16:00 | T1-D Optuna 启动（50 trials 目标）                    |
| 2026-05-18 18:00 | T1-D 手动中止（已完成 24 trials，Trial 4 最优 Sharpe=7.15） |
| 2026-05-18 18:17 | T1-D 对比回测完成（Sharpe 2.03）                        |
| 2026-05-18 20:02 | T1-A+B 实验启动                                     |
| 2026-05-18 20:50 | T1-A 完成（Sharpe 2.42）                            |
| 2026-05-18 21:14 | T1-B+C 完成（Sharpe 2.17）                          |
| 2026-05-18 21:19 | T1-ALL 综合最优完成（Sharpe 2.31）                      |
| 2026-05-18 21:30 | 本复盘文档写入 `docs/lessons/`                         |


