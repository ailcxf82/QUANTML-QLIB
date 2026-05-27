# W1 撮合修正 + Optuna v2（2026-05-26）

> **一句话结论**：把回测撮合从"裸佣金 + close 字段"修正为"含 10bp 滑点 + $close_qfq 字段"，
> 缩小实盘 vs 回测 sharpe gap（理论缩小约 0.3）；同时把 Optuna 默认搜参
> 升级到 9 维 / 5 seeds / 强制 strategy_cfg 对齐，规避 T1-D 评估-部署不一致退化。

---

## 1. 背景

T1 复盘（`2026-05-18_T1_reduce_vol_boost_return.md`）发现：

| 实验 | WF Optuna 内 sharpe | 单切分实测 sharpe | gap |
|------|---------------------|------------------|-----|
| T1-D（Trial 4） | 7.15 | 2.03 | **-5.12** |
| T1-A（3 seeds） | n/a | 2.42 | n/a |

差距来源（按贡献排序）：

1. Optuna 目标函数与部署策略不一致（最大）—— T1-D 教训③
2. 回测撮合用 `close` + 0.25%/0.35% cost 偷看了实盘不可得的 alpha
3. ensemble seeds 太少，多样性不足（T1-A 教训①）

本次 W1 修正同时解决上述 #2 / #3 / #1（第 1 项通过 Optuna 默认 strategy_cfg 解决）。

---

## 2. 撮合修正：cost + deal_price

### 2.1 修正前后对照

| 字段 | W1 前 | W1 后 | 依据 |
|------|--------|--------|------|
| `deal_price` | `close` / `$close_qfq` 混用 | **统一 `$close_qfq`** | lessons 2026-05-17：消除 qlib adjusted_price 模式 trade unit 100 警告，与训练 label 字段对齐 |
| `open_cost` | 0.0025 | **0.0035** | 双边 +10bp 滑点估计（csi500 中小盘 5min VWAP 90 分位） |
| `close_cost` | 0.0035 | **0.0045** | 同上 |
| `impact_cost` | 0 | 0（保持） | qlib 公式 `$volume` 存在 9 个数量级 bug，启用会冻结策略 |

### 2.2 关键决策：为什么不改成 `$open_qfq`？

初版方案曾建议把 `deal_price` 改成 `$open_qfq` 模拟 T+1 开盘买入实盘，但：

- 当前 label = `Ref($close_qfq,-2)/Ref($close_qfq,-1)-1` 是 close-to-close 回报
- 只改撮合不改 label → 回测会"偷看 T+1 开-收上涨段"，**反而引入新 bias**

正确做法：**主链路维持 close-to-close 一致**（cost 调整），新增独立配置
`configs/freq/daily_t1_open.yaml`（label 同步改 open-to-open）作为 A/B 对照，
未来若验证 sharpe 不显著退化再切到主链路。

### 2.3 预期影响（按 T1-ALL 状态推算）

- 双边 cost +20bp × 当前换手 ~9% × 252 / 0.9 ≈ **-0.04% 年化绝对收益**
- 因 cost 是常数项，对 sharpe / MDD 几乎无影响（**< 0.05 sharpe 下降**）
- 因 `deal_price` 改 `$close_qfq` 是字段口径修正而非价格变动，**不影响指标**，
  只消除 qlib 警告

### 2.4 涉及文件

| 文件 | 改动 |
|------|------|
| `configs/freq/daily.yaml` | deal_price + cost 修正 + 中文注释翻译（原乱码） |
| `configs/freq/daily_retrain.yaml` | 同上 |
| `configs/freq/daily_prod.yaml` | cost 修正（deal_price 已是 `$close_qfq`） |
| `configs/freq/daily_prod_retrain.yaml` | 同上 |
| `configs/freq/daily_5d_label.yaml` | 同 daily.yaml |
| `configs/mlp_daily_backtest.yaml` | 同步主链路（虽计划废弃但 cost 原值偏低） |
| `configs/freq/daily_t1_open.yaml` | **新增**：T+1 开盘买入对照实验 |
| `python/backtest/market_layer.py` | CostModel 默认值 + 新增 `slippage_estimate` 属性 + describe 增强 |

---

## 3. Optuna v2：搜索空间 + 默认行为升级

### 3.1 搜索空间扩展（7 → 9 维）

| 维度 | W1 前 | W1 后 |
|------|--------|--------|
| `num_leaves` | [15, 63] | [15, 63] |
| `learning_rate` | [0.005, 0.05] log | [0.005, 0.05] log |
| `min_child_samples` | [50, 500] | **[50, 700]**（允许更保守叶节点） |
| `feature_fraction` | [0.50, 0.95] | **[0.40, 0.95]**（强化列采样多样性） |
| `bagging_fraction` | [0.50, 0.95] | **[0.45, 0.95]**（同上） |
| `reg_lambda` | [0.5, 10.0] log | [0.5, 10.0] log |
| `reg_alpha` | 固定 0.3 | **[0.0, 2.0]**（开放 L1） |
| `path_smooth` | 固定 2.0 | **[0.0, 5.0]**（lgbm.yaml 注释强调跨周期稳定性关键） |
| `extra_trees` | {True, False} | {True, False} |

### 3.2 默认参数升级

| CLI 参数 | W1 前默认 | W1 后默认 | 依据 |
|---------|------------|------------|------|
| `--strategy-cfg` | None | `configs/strategy/daily_vol_target.yaml` | **T1-D 教训③**：必须评估-部署一致 |
| `--ensemble-seeds` | 1 | **5** | T1-A 教训①：3 seeds 多样性不足 |
| `--n-trials` | 50 | 50 | 不变 |

### 3.3 单 trial 计算量估算

- 5 seeds × 3 folds × ~5min/seed ≈ **75 min/trial**
- 50 trials ≈ **62 小时**（建议夜间 + 周末跑，配合 SQLite 中断续跑）
- 快速冒烟：`--n-trials 3 --ensemble-seeds 1` ≈ 30 min

### 3.4 涉及文件

- `python/tuning/optuna_lgbm.py`：搜索空间 + 默认值 + epilog 更新

---

## 4. WF 切分扩展（季度 5-fold）

新增 `configs/rolling/expanding_quarterly.yaml`：

- `valid_size: 3m, test_size: 3m, step: 3m`
- OOS 切成 5 个 fold（vs halfyearly 的 4 个）
- 对季度风格切换更敏感（2025 小盘风格年每季度持续 ~3m）

未替换原 `expanding_halfyearly.yaml`，两者并存供选择。

---

## 5. Sanity 工具：撮合修正前后 A/B

新增 `scripts/sanity_w1_cost_compare.py`：

- 不重训，只用已有 signals.parquet 跑两次回测（W1 前 / 后 cost）
- 输出 sharpe / 年化 / MDD / 换手对照
- 单次运行 ~1-2 分钟

用法示例：

```powershell
# 用最近一次 lgbm_optuna_daily_retrain_* run 自动对比
python scripts/sanity_w1_cost_compare.py --auto

# 显式指定 run_id
python scripts/sanity_w1_cost_compare.py --run-id lgbm_optuna_daily_retrain_c276d396
```

判定准则：

- 实际 sharpe 下降 < 0.30 → 符合预期（cost 修正不破坏 alpha 主体）
- 实际 sharpe 下降 ≥ 0.30 → 异常（检查 deal_price 字段缺失、换手突变等）

---

## 6. 红线检查（lessons 累积）

| 红线 | 本次执行情况 |
|------|---------------|
| 不动 `base.yaml` 的 processor 链 | ✓ 未碰 |
| 不批量加因子 | ✓ 未加 |
| 不默认开 CSNeutralize | ✓ 未开 |
| Optuna 评估必须 = 部署策略 | ✓ 默认强制 `daily_vol_target.yaml` |
| label winsor < [1%, 99%] | ✓ 无 winsor 改动 |

---

## 7. 上线节奏建议

1. **立即（commit 后）**：跑 `sanity_w1_cost_compare.py --auto`，确认 sharpe 下降 < 0.30
2. **本周内**：用 `--ensemble-seeds 5` 重跑一次 `lgbm_optuna --freq daily_retrain` 单切分，
   对比 T1-A 历史 sharpe 2.42（验证 ensemble seeds 5 是否优于 3）
3. **下周内**：启动 `optuna_lgbm.py --n-trials 50`（夜间 + 周末，约 62 小时），
   完成后用新 `lgbm_optuna.yaml` 跑 sanity + 走 `monthly_retrain.ps1` gate
4. **W2**：跑 `daily_t1_open` A/B 测试，验证撮合字段切换的影响幅度

---

## 8. 相关文档

- [T1降波动同时提收益](./2026-05-18_T1_reduce_vol_boost_return.md) — Optuna / Ensemble / RiskParity 三件套
- [MarketTimer实盘工程](./2026-05-17_market_timer_live_predict.md) — deal_price 字段历史
- [P1P2中性化陷阱](./2026-05-11_p1p2_neutralize_pitfall.md) — 中性化红线 + 实验 yaml 隔离原则
