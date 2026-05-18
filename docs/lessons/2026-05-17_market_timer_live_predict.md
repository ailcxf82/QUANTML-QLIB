# MarketTimer 实盘预测与工程经验（2026-05-17）

> **一句话教训**：GARCH 降仓能压回撤，但 `min_risk` 过低 + 日频等权微调 + `trade_unit=100` 会导致策略「冻结」；实盘必须输出**具体买卖手数**，不能只给权重比例。

---

## 1. 背景与目标

- **MarketTimer**：用 CSI500 指数 GARCH(1,1) 预测市场波动，动态缩放总仓位（`risk_factor`）。
- **实盘命令**：`python python/predict_live.py --model lgbm --freq daily --mode predict-only`
- **配置**：`configs/live/daily_live.yaml`（账户资金、MarketTimer）、`configs/strategy/daily_vol_target.yaml`（回测对照）

---

## 2. 实测效果（回测区间 2025-01 ~ 2026-04）


| 指标      | 无 MarketTimer（基线 TopkDropout） | GARCH + 等权 + min_change |
| ------- | ----------------------------- | ----------------------- |
| 年化收益    | 71.64%                        | 33.55%                  |
| 年化波动    | 18.85%                        | 10.11%                  |
| 最大回撤    | -11.29%                       | **-5.54%**              |
| Sharpe  | 2.965                         | 2.913                   |
| 平均换手    | 80.48%                        | 22.45%                  |
| 订单笔数（约） | ~640                          | ~350（修复后）               |


**结论**：GARCH 显著压低波动与回撤，但收益下降；需配合 `min_risk≥0.50` 与周频调仓，避免订单爆炸与冻结。

---

## 3. 根因与修复（按主题）

### 3.1 回测极慢（43 分钟 → ~2 分钟）


| 原因                                     | 修复                                |
| -------------------------------------- | --------------------------------- |
| 每个交易日重复 `D.features` 拉收盘价              | `preload_instruments()` 一次加载全市场标的 |
| `StockFilter` 每日查涨停                    | 按标的缓存 `close.day.bin`             |
| `MarketTimer` 每日重载 benchmark           | 一次性加载全历史，按日切片                     |
| 无 `factor.day.bin` → adjusted_price 模式 | 批量生成 `factor = close_qfq/close`   |


### 3.2 订单 3400+ vs 基线 643


| 原因                         | 机制                            | 修复                                     |
| -------------------------- | ----------------------------- | -------------------------------------- |
| `InverseVolWeighter`       | 每日重算权重 → 每日微调单                | 改为 `equal`                             |
| `exit_rules` + renormalize | 止损后权重重分配 → 5 只票各 1 单          | 实盘去掉 exit_rules；回测用 `daily_sl_tp.yaml` |
| GARCH 每日微变 `risk_factor`   | 每日 5 只票微调                     | `min_change=0.10` + `rebalance_freq=5` |
| `min_risk=0.30` 过小         | 单票仓位 < 18k → 微调 < 100 股被截断为 0 | `min_risk=0.50`                        |


### 3.3 4 月后「无交易」冻结

- 非突然停止，而是 2025-10 起逐月衰减。
- 机制：`risk_factor` 下降 → 单票金额小 → 漂移修正不足 100 股 → `trade_unit` 丢弃 → 零订单。
- 修复：`min_risk: 0.50`，`rebalance_freq: 5`。

### 3.4 交易单位与合规

- Qlib `amount` 字段为**股**（非手）。
- `TopkDropoutStrategy` 按金额分配，**不保证** 100 股整数倍（回测研究可用，实盘需合规）。
- 生成 `factor.day.bin` 后消除 `trade unit 100 is not supported` 警告。

### 3.5 数据层

- `deal_price: "close"` 在数据集中无 `factor.day.bin` 时退回 adjusted_price，且不支持 100 股整数倍。
- 已改为 `deal_price: "$close_qfq"`（回测配置）并批量生成 factor 文件。

---

## 4. 实盘输出契约

`selection.csv` 关键字段：


| 字段              | 含义              |
| --------------- | --------------- |
| `actual_weight` | 目标权重比例（含 GARCH） |
| `close_price`   | 参考收盘价           |
| `prev_shares`   | 上期推荐持仓（估算）      |
| `target_shares` | 目标股数（100 股整数倍）  |
| `trade_shares`  | 正买负卖（股）         |
| `action_cn`     | 中文操作说明          |


控制台与 `report.html` 含「明日 T+1 调仓操作计划」表。

**账户资金**：`configs/live/daily_live.yaml` → `account.total_value`（默认 30 万）。

---

## 5. 配置速查

```yaml
# configs/live/daily_live.yaml
account:
  total_value: 300000
  trade_unit: 100
market_timer:
  enabled: true
  type: garch
  target_vol: 0.25
  min_risk: 0.50      # 勿低于 0.50
  min_change: 0.10
  benchmark: "000905.SZ"
```

回测 MarketTimer 参数见 `configs/strategy/daily_vol_target.yaml`（`rebalance_freq: 5`、`equal` 权重、`min_risk: 0.50`）。

---

## 6. 与 P1/P2 中性化教训的对比（勿混用）


| 维度  | P1/P2 行业中性化            | MarketTimer |
| --- | ---------------------- | ----------- |
| 目标  | 提升截面 RankIC            | 控制组合波动率     |
| 风险  | 可能毁掉小盘 alpha（2025 风格年） | 低波动期收益偏低    |
| 验证  | 必须看 TopK 超额与绝对收益       | 看回撤、换手、订单笔数 |


**不要**因 GARCH 降波动就默认 `target_vol=0.15` 且 `min_risk=0.30` 而不调参。

---

## 7. 后续建议

1. 实盘前用 `min_risk=0.50` 跑一段样本，确认无长期「冻结」。
2. 以实际持仓校准 `prev_shares`，勿仅信估算值。
3. 回测对比时固定 `account.total_value` 与 `trade_unit`，与实盘一致。
4. 若需止损，用 `daily_sl_tp.yaml` 回测；实盘 GARCH 与 exit_rules 叠加会重复降仓，需明确分工。
5. `docs/live_predict_guide.md` 已写操作说明，新同事可直接参考。

---

## 8. 相关文件


| 路径                                                   | 说明                         |
| ---------------------------------------------------- | -------------------------- |
| `python/backtest/strategy/market_timer.py`           | GARCH / Rolling / Null     |
| `python/predict_live.py`                             | `compute_trade_plan`、控制台输出 |
| `configs/live/daily_live.yaml`                       | 实盘参数                       |
| `configs/strategy/daily_vol_target.yaml`             | 回测 MarketTimer             |
| `docs/live_predict_guide.md`                         | 实盘操作手册                     |
| `docs/lessons/2026-05-11_p1p2_neutralize_pitfall.md` | IC≠Alpha、中性化陷阱             |


---

## 9. 相关提交（功能分支，非 main 合并说明）

- MarketTimer 模块、`daily_vol_target.yaml`、`predict_live` 调仓计划、`factor.day.bin` 批量生成、`preload_instruments`、配置 `deal_price` 修复等分布在当日功能提交中，合并前需跑通 `pytest` 与一次完整 `predict-only` smoke test。

