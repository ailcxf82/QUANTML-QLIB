# LGBM 深度优化失败复盘（2026-05-11）

> **一句话教训**：截面排名 IC（Spearman RankIC）大幅上升 ≠ TopK 选股实际赚钱。  
> 在 A 股 2025 风格分化年份，强行做"行业 + 市值中性化"会把最关键的 alpha 来源残差化掉。

---

## 1. 实验范围

在不引入新模型族的前提下做 P0-P6 全套深度优化：


| 阶段  | 改动                                                                            |
| --- | ----------------------------------------------------------------------------- |
| P0  | 评估口径：fold IC 统一 Spearman、修复 `concatenated_metrics.ic_`* NaN、walk-forward 默认开启 |
| P1  | 标签清洗：CSWinsor(1%, 99%) + 停牌/涨跌停 mask（`If(...)+0/0` Qlib 表达式 trick）            |
| P2  | 行业市值中性化：`CSNeutralize(log_mv + industry_one_hot)` 对每个截面做 OLS 残差化              |
| P3  | 共线性自动剔除：feature_select 按 RankIC +                                             |
| P4  | 目标函数：huber 单头 + regression/lambdarank 双头                                      |
| P5  | 样本加权：`InvVolReweighter`（按 20 日波动率倒数）                                          |
| P6  | Optuna 7 维超参搜索                                                                |


---

## 2. 实测对比（同 OOS 区间 2025-01 ~ 2026-04-28）


| 指标                       | V18-V2 基线（WF 标定） | P1+P2 全开（WF）                          | 差异             |
| ------------------------ | ---------------- | ------------------------------------- | -------------- |
| **fold 信号 IC（Spearman）** | 0.040            | **0.073**（fold 0.092 / 0.063 / 0.064） | **+82%** ↑     |
| **fold ICIR**            | 0.187            | **0.55**（fold 0.73 / 0.48 / 0.45）     | **+194%** ↑    |
| **年化收益（绝对）**             | **+85.79%**      | +26.65%                               | **−69%** ↓     |
| **年化超额（vs 中证500）**       | **+9.28%**       | **−24.72%**                           | **−34pp** ↓    |
| **信息比率**                 | +0.438           | **−0.290**                            | 由正转负           |
| Sharpe                   | 2.78             | 1.48                                  | −47%           |
| 最大回撤                     | −13.5%           | −8.5%                                 | "改善"但无意义（收益太低） |
| 换手率                      | 0.41             | 0.39                                  | 持平             |
| 基准（中证500）年化              | +31.8%           | +30.0%                                | 持平             |


**核心悖论**：模型在信号端横截面排名能力大幅提升，但 TopK 5 票实际跑输基准 24.7 pp。

参考产物（已被 revert，可通过 `git checkout experiment-p0-p6-2026-05-11` 找回）：

- V18-V2：`artifacts/models/latest_lgbm_daily_meta.json`（run_id `lgbm_daily_0d030abf`）
- P1+P2：`artifacts/lgbm_daily_5763ffc7/rolling_summary.json`

---

## 3. 根因分析（按影响排序）

### 根因 1：CSNeutralize 杀掉 A 股 2025 最强 alpha 来源（影响 ≈70%）

- 2025 年 A 股是典型的"小盘 + 科创"风格年；V18-V2 +85% 年化主要来自选中"小市值 + 高成长"票。
- `CSNeutralize(feature, log_mv+industry)` 把所有 feature 与 `log(MV)` 和行业 dummies 强制残差化为零相关。
- **后果**：模型不再能识别"小市值 + 高动量"信号，TopK 选出的票退化为"行业中位数附近"。
- 信号端 IC 反而升高的原因：label 也是截面 z-score，模型学到的"纯横截面排序"在 IC 上更准，**但 A 股 2025 的 alpha 主要来自风格，不在这里**。

### 根因 2：5 类粗分行业 → 残差化过度（影响 ≈20%）

- 行业映射缺真实申万一级数据时回退到 `classify_market` 5 类粗分（沪主板/深主板/创业板/科创板/北交所）。
- 创业板内电子/医药/新能源/消费 alpha 差异巨大，被强制视为同质 → 残差化过度。
- 在没有真行业数据前提下做了"超粒度"中性化，把同板块内 alpha 都磨平。

### 根因 3：CSWinsor(label, [1%, 99%]) 截断了涨停股监督信号（影响 ≈10%）

- A 股涨跌停 ±10%，截面 1% 分位 ≈ −7%，99% 分位 ≈ +7%。
- winsor 后涨停股 label 被压平到 +7%，LGBM 训练时**看不到"涨停级别强势"的明确监督信号**，TopK 头部敏感性下降。

---

## 4. 关键教训（务必内化）

### 教训 ① IC ≠ Alpha（**最重要**）

> **"信号 IC 高"只能说明截面排名准，不代表 TopK 5 票能赚钱。**

回测时务必同时看：

- **alpha 端**：`annualized_excess_return`、`information_ratio`、`win_rate`
- **信号端**：`ic_mean`、`icir`、`ic_positive_ratio`

任何"IC 升但超额跌"的实验都是**红色警报**，不能仅凭 IC 决策。

### 教训 ② 单切分表现 ≠ Walk-Forward 表现

- `lgbm_ext` 单切分 IR 0.49（看似改善），但 WF IR 0.56 < V18-V2 WF 标定 0.87
- 上线决策**必须以 WF 为准**；单切分仅供快速 sanity check

### 教训 ③ "加因子"在本工程已被两次 WF 否决


| 实验                       | 加的因子                | WF IR              | 结论     |
| ------------------------ | ------------------- | ------------------ | ------ |
| `lgbm_ext`               | +7 个（v18+extension） | 0.560 vs V18 0.868 | 退化 35% |
| `lgbm_lab/new_md_full15` | +15 个（new.md 全套）    | 0.014 vs V18 0.868 | 退化 98% |


**不要轻易"批量加因子"**——markdown 池里已穷尽 50 个表达式，剩余独立新维度只剩约 6 个，且多数 `real_icir_eval` 为负。

### 教训 ④ 中性化前必须问三个问题

引入任何"中性化 / 残差化"前，先确认：

1. **真实风格暴露有没有 alpha**？
  A 股市值+行业是历史强 alpha 来源，盲目残差化等于自废武功。  
   决策依据：先看基准（如中证500）当年是否被"小盘/创业/科创"显著跑赢，若是→**禁用 size/industry 中性化**。
2. **行业映射粒度够细吗**？
  5 类粗分会让残差化"过度"。**必须用申万一级 28 类或更细粒度**才有意义。
3. **可否做 partial neutralize（β<1）而非全去除**？
  学术界主流是 `feature_resid = feature − β · style_factor`，β 通常取 0.3~0.7，保留部分风格暴露。

### 教训 ⑤ Label Winsor 在 A 股要极谨慎

- A 股 ±10% 涨跌停是**最重要的监督信号源**（涨停股 = 模型最该选的票）
- 截面 winsor 哪怕只截到 [1%, 99%]，也会把多数涨停股 label 压平
- 替代方案：要么不 winsor，要么放宽到 [0.5%, 99.5%] 或 [0.2%, 99.8%]

### 教训 ⑥ 默认链路必须用 V18-V2 兼容

- `configs/base.yaml` 是所有 model yaml 的根；改 base.yaml 的 processor 链会**自动污染**所有继承它的模型（lgbm/lgbm_ext/lgbm_lab/lgbm_huber/lgbm_dual/lgbm_v2/...）
- **任何"实验性 processor"都应通过独立 model yaml overlay 注入**（如 `lgbm_p1.yaml`），不要动 base.yaml

---

## 5. 决策准则（之后实验请先对照）

### 必须做（红线）

- 任何新 processor / 新 loss / 新加因子，**先用独立 model yaml 隔离**，不动 `base.yaml`
- 验收**必须**走 walk-forward，单切分仅 sanity check
- 同时看**信号端 IC** 与 **alpha 端超额收益**；只看一个一定踩坑
- 实验失败时**立即写复盘文档**到 `docs/lessons/`，下次能避免

### 禁止做（红线）

- ❌ 在没有申万一级 28 类真行业数据时启用 `CSNeutralize`
- ❌ 对 label 做 [1%, 99%] 以上力度的截面 winsor
- ❌ 把 P1/P2 类破坏性 processor 默认加进 `base.yaml`
- ❌ 仅凭"信号 IC 上升"就认为优化成功
- ❌ 一次性批量加因子（markdown 池已穷尽，剩余独立维度 < 6 个）

---

## 6. 未来重启 P2 中性化的前置条件

如果未来想再尝试行业/市值中性化，必须**先满足以下全部**：

1. **接入真实申万一级 28 类行业映射**到 `configs/industry_map.csv`（不要再用 5 类粗分）
2. `**CSNeutralize` 增加 `beta_scale` 参数**（默认 1.0 全去除，建议 0.3~0.5），保留部分风格暴露
3. **先用独立 yaml 跑 WF 3 fold 对照**，确认 IC + 超额 + 换手 + 容量四指标都不退化，再考虑合入
4. **基准期非"小盘风格主导"**时再启用（如 2024 上半年中证 500 跑赢沪深 300 的"大盘价值"区间）

---

## 7. 真正值得做的稳妥优化路径（取代"加因子"）

按 ROI 排序：


| 优先级 | 路径                                 | 预期 WF IR 提升       | 风险  |
| --- | ---------------------------------- | ----------------- | --- |
| ⭐⭐⭐ | Optuna 7 维超参搜索（无新因子、无新模型）          | +3~5%             | 低   |
| ⭐⭐⭐ | sample_weight (`InvVolReweighter`) | +1~3%             | 低   |
| ⭐⭐⭐ | huber loss 单头对照                    | +2~5%             | 低   |
| ⭐⭐  | Ensemble seed 数量与 aggregator 调优    | +2~5% 稳定性         | 低   |
| ⭐⭐  | 引入 **新数据维度**（北向资金/龙虎榜/分钟频/Wind 行业） | +10~30% 未知        | 中   |
| ⭐   | 引入异构模型族（GRU/HIST）                  | +10~15%           | 中   |
| ❌   | 批量加 markdown 因子                    | **历史已两次否决**       | 高   |
| ❌   | 在 base.yaml 默认开启 CSNeutralize      | **本次已验证退化 −34pp** | 极高  |


---

## 8. 如何找回完整实验代码

实验快照已用 git tag 永久存档：

```powershell
# 查看 tag
git tag -l "experiment-*"

# 检出整个实验快照到独立分支（不影响当前 main）
git checkout -b lgbm-experiment experiment-p0-p6-2026-05-11

# 或只摘出某个工具
git checkout experiment-p0-p6-2026-05-11 -- python/tuning/optuna_lgbm.py
git checkout experiment-p0-p6-2026-05-11 -- python/processors/cs_winsor.py
git checkout experiment-p0-p6-2026-05-11 -- python/models/lgbm_dual_head.py

# 重新看本次完整退化诊断
git show experiment-p0-p6-2026-05-11:docs/experiments/lgbm_v2.md
```

实验快照 commit：`99fd6d9`（已 tag 为 `experiment-p0-p6-2026-05-11`）

---

## 9. 时间线参考


| 时间               | 动作                                                           |
| ---------------- | ------------------------------------------------------------ |
| 2026-05-10       | V18-V2 基线运行良好（年化超额 +9.28%）                                   |
| 2026-05-11 上午    | 实施 P0-P6 全套改动，base.yaml 默认开启 P1+P2                           |
| 2026-05-11 下午    | 发现 OOS 年化超额 −24.72%，IR −0.29                                 |
| 2026-05-11 14:42 | 定位根因为 CSNeutralize 残差化 size+industry alpha                   |
| 2026-05-11 14:46 | 回滚 base.yaml，把 P1/P2 改为独立 yaml 对照实验                          |
| 2026-05-11 19:48 | 实验快照 commit + revert，存 git tag `experiment-p0-p6-2026-05-11` |
| 2026-05-11 21:00 | `git reset --hard c061ae7`，正式回到 V18-V2 节点                    |
| 2026-05-11 21:55 | 本复盘文档写入 `docs/lessons/`                                      |


---

## 10. 用一张图记住这次教训

```
基准（中证500）涨 30%
                       ↓
        V18-V2 模型选小盘 → 涨 85% → 超额 +9%   ← 成功
                       ↓
        P1+P2 中性化 → 选行业中位数 → 涨 27% → 超额 −25%   ← 失败
                       ↓
        但 IC 反而 +82% ↑ ← 这是"信号 IC ≠ TopK Alpha"的经典陷阱
```

**当一个"看起来更科学"的改动让 IC 升但超额跌时，立即停止，先做根因分析再决定。**