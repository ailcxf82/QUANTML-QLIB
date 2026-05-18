# 实盘预测与调仓说明

> 适用命令：`python python/predict_live.py --model lgbm --freq daily --mode predict-only`  
> 配置文件：`configs/live/daily_live.yaml`、`configs/strategy/daily_vol_target.yaml`（回测 MarketTimer）

---

## 1. 命令与输出

```powershell
# 日常（收盘后）
python python/predict_live.py --model lgbm --freq daily --mode predict-only

# 指定日期
python python/predict_live.py --model lgbm --freq daily --mode predict-only --date 2026-05-15

# 指定账户资金（在 daily_live.yaml 中修改 account.total_value）
```


| 输出路径                                  | 内容              |
| ------------------------------------- | --------------- |
| `predictions/<日期>/selection.csv`      | 推荐列表 + **调仓明细** |
| `predictions/<日期>/report.html`        | 可视化报告（含操作计划区块）  |
| `predictions/<日期>/pred_score.parquet` | 全市场打分（调试用）      |
| `predictions/<日期>/model_meta.json`    | 模型版本信息          |


---

## 2. `selection.csv` 字段说明


| 字段                 | 含义                                           |
| ------------------ | -------------------------------------------- |
| `instrument`       | 股票代码                                         |
| `score`            | 模型打分（越高越看好）                                  |
| `rank`             | 截面排名                                         |
| `suggested_weight` | 等权权重（= 1/topk）                               |
| `actual_weight`    | **实际目标权重**（= 等权 × MarketTimer 的 risk_factor） |
| `percentile`       | 在 CSI500 内的打分百分位                             |
| `change`           | `new_in` / `hold` / `reduce` / `drop_out`    |
| `risk_factor`      | 当日仓位系数（GARCH 输出，如 0.84 表示约 84% 仓位）           |
| `close_price`      | T 日收盘价（元/股）                                  |
| `prev_shares`      | **估算**的上期持仓股数（假设上期按推荐执行）                     |
| `target_shares`    | 目标持仓股数（已取整到 100 股整数倍）                        |
| `trade_shares`     | 需交易股数（正=买，负=卖，0=不动）                          |
| `trade_value`      | 预计成交金额（元）                                    |
| `action_cn`        | 中文操作说明，如「买入 78 手（7800股，约5.0万）」               |


---

## 3. MarketTimer（GARCH 目标波动率）

根据 CSI500 指数波动率动态调节总仓位：

```
risk_factor = clip(σ_target / (√252 · σ̂_{t+1|GARCH}), min_risk, max_risk)
实际权重 = 等权 × risk_factor
```


| 参数           | 实盘默认        | 说明                                     |
| ------------ | ----------- | -------------------------------------- |
| `type`       | `garch`     | GARCH(1,1) + Student-t                 |
| `benchmark`  | `000905.SZ` | CSI500                                 |
| `target_vol` | `0.25`      | 年化目标波动率 25%                            |
| `min_risk`   | `0.50`      | 最低仓位 50%（勿低于此值，否则单票仓位过小导致微调单被 100 股截断） |
| `max_risk`   | `1.0`       | 不加杠杆                                   |
| `min_change` | `0.10`      | risk_factor 变化超过 10% 才触发调仓             |
| `refit_freq` | `5`         | 每 5 个交易日重拟合 GARCH                      |


**实盘配置**（`configs/live/daily_live.yaml`）：

```yaml
market_timer:
  enabled: true
  type: "garch"
  benchmark: "000905.SZ"
  target_vol: 0.25
  min_risk: 0.50
  min_change: 0.10
```

---

## 4. 调仓手数计算逻辑

```
目标金额 = account_total × actual_weight
目标股数 = floor(目标金额 / 收盘价 / 100) × 100   # 取整到 1 手（100 股）
交易量   = 目标股数 - prev_shares（估算）
```

- **账户资金**：`configs/live/daily_live.yaml` → `account.total_value`（默认 30 万）
- **最小手数**：100 股（1 手），不足 100 股的调整单会被丢弃
- **上期持仓**：由上一日 `actual_weight` 与收盘价估算，**请以实际持仓核对**

控制台与 HTML 报告中的「明日 T+1 调仓操作计划」表即为此逻辑的输出。

---

## 5. 与回测策略的差异


| 项目       | 基线 `TopkDropoutStrategy` | `daily_vol_target` + MarketTimer        |
| -------- | ------------------------ | --------------------------------------- |
| 策略类      | Qlib 内置                  | `QuantMLWeightStrategy`                 |
| 订单量级     | ~640 笔/年                 | 可配置（见下）                                 |
| 调仓明细     | 无                        | 有（手数、金额）                                |
| 100 股整数倍 | 否（按金额分配）                 | 是（`trade_unit=100`）                     |
| 止损止盈     | 无                        | 回测可用 `daily_sl_tp.yaml`；实盘 GARCH 承担下行保护 |


**回测订单量参考**（`daily_vol_target.yaml`）：

- `weighter_cfg.type: equal`（等权，避免每日权重漂移产生大量微调单）
- `rebalance_freq: 5`（周频模型调仓）
- `min_change: 0.10`
- `min_risk: 0.50`
- 无 `exit_rules`（避免 renormalize 触发额外订单）

---

## 6. 数据与工程修复（摘要）


| 问题                                | 处理                                               |
| --------------------------------- | ------------------------------------------------ |
| 回测极慢（43 分钟）                       | 预加载全市场收盘价 + `factor.day.bin` 批量生成                |
| 订单 3400+ vs 基线 643                | 等权 + `min_change` + 无 exit_rules；`min_risk≥0.50` |
| 4 月后无交易（冻结）                       | `min_risk` 从 0.30 提到 0.50；`rebalance_freq=5`     |
| `trade unit 100 is not supported` | 生成 `factor.day.bin`（`close_qfq/close`）           |
| 实盘无具体手数                           | 新增 `compute_trade_plan` + `account` 配置           |


---

## 7. 配置速查


| 文件                                       | 用途                              |
| ---------------------------------------- | ------------------------------- |
| `configs/live/daily_live.yaml`           | 实盘：账户资金、选股参数、MarketTimer        |
| `configs/strategy/daily_vol_target.yaml` | 回测：GARCH + QuantML 策略           |
| `configs/strategy/daily_sl_tp.yaml`      | 回测：止盈止损 + MarketTimer           |
| `configs/base.yaml`                      | 默认 `TopkDropoutStrategy`（基线对比用） |


修改账户资金：

```yaml
# configs/live/daily_live.yaml
account:
  total_value: 300000   # 改为实际资金（元）
  trade_unit: 100
```

---

## 8. 常见问题

**Q：为什么 risk_factor 经常低于 1？**  
A 股 2025–2026 年波动偏高，GARCH 预测高于目标_vol=25% 时会减仓；`min_risk=0.50` 保证单票仍有足够金额做 100 股整数倍调整。

**Q：prev_shares 不准怎么办？**  
以券商/实际持仓为准，手动计算：`目标_shares - 实际持仓 = trade_shares`。

**Q：回测和实盘 MarketTimer 参数要一致吗？**  
实盘用 `configs/live/daily_live.yaml` 的 `market_timer`；回测用 `configs/strategy/daily_vol_target.yaml`。逻辑相同，参数可分别调。

**Q：如何关闭 MarketTimer？**  
`configs/live/daily_live.yaml` 中设置 `market_timer.enabled: false`，或 `type: null`。

---

## 9. 相关文档

- 回测对比脚本：`python/scripts/compare_market_timer.py`（null / rolling / garch）
- 重训手册：`docs/retrain_playbook.md`
- 项目总则：`.cursor/rules/00-项目总则-QLib高频选股.md`

