# combined_factors_df 完整因子清单（19 个）—— v21

> 生成时间: 2026-05-24 09:43:50
> 时间窗: 2020-01-01 ~ 2026-05-23; 池: csi500; mode: strict; shape: [1432556, 19]
> manifest 当前版本: **v21**; 对比版本: v20

## 组合模型性能概况（最新会话）


| 指标                 | 值                             |
| ------------------ | ----------------------------- |
| 最新运行时间             | 2026-05-24 21:37              |
| 最新 IC              | nan                           |
| 最新 IC_IR           | **4.8184**                    |
| 最新 Composite       | 9.8602                        |
| 最新 Enhanced        | **10.1849**                   |
| 最新 Diversity Bonus | 0.3247                        |
| 本轮最佳 IC_IR         | **4.8461** (2026-05-24 17:16) |
| 本轮最佳 Composite     | 9.6923                        |
| 本轮最佳 Enhanced      | **9.8953**                    |


## 版本变更：v20 → v21


| 变更类型    | 因子                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ✅ 新增    | `ROEAccel_60D`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| 🔄 重新加入 | `volume_ratio_ma_15d`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ❌ 移除    | `ValueMR_60D`, `Volume_Surprise_20D`, `avg_vol_ratio_5d`, `close_location_turnover_wt`, `earnings_growth_smooth`, `eps_yield_momentum_20d`, `free_float_turnover`, `intraday_return`, `liq_turnover_f`, `log_amihud_20d`, `pb_pct_change_20d`, `profit_growth_yoy`, `profit_yoy`, `q_profit_yoy`, `quality_value_momentum_5d_composite`, `rank_pb_roe`, `rank_turnover_deviation_20d`, `rank_volume_ratio`, `raw_turnover_rate_f`, `raw_volume_ratio`, `roe_pb_momentum_divergence_60d`, `sma_volume_ratio_5d`, `turnover_accel_5d`, `turnover_acceleration_20d`, `turnover_rate_f`, `turnover_surprise_20`, `vol_ratio_5d_20d`, `vol_surge_5d`, `volume_ratio`, `volume_ratio_rank`, `volume_ratio_signal`, `volume_ratio_unusual`, `volume_zscore_20d` |


## 因子性能汇总（按 IC 降序）


| #   | 因子名                                      | IC      | IC_IR(全期) | IC(样本内) | IC(样本外) | nan比例  | 最大相关特征                                                      |
| --- | ---------------------------------------- | ------- | --------- | ------- | ------- | ------ | ----------------------------------------------------------- |
| 1   | `rank_close_open`                        | +0.3519 | +11.4885  | +0.3541 | +0.3538 | 0.0012 | 0.023 (Ref($close_qfq, 2)/Ref($close_qfq, 1) - 1)           |
| 2   | `close_location`                         | +0.3451 | +16.9919  | +0.3479 | +0.3294 | 0.0012 | 0.037 (Ref($close_qfq, 6)/Ref($close_qfq, 1) - 1)           |
| 3   | `log_signed_volume`                      | +0.3330 | +27.7288  | +0.3334 | +0.3337 | 0.0012 | 0.026 (Ref($close_qfq, 6)/Ref($close_qfq, 1) - 1)           |
| 4   | `ValueMR_20D`                            | +0.1566 | +7.1405   | +0.1548 | +0.1473 | 0.1330 | 0.685 (Ref($rsi_qfq_12, 1))                                 |
| 5   | `close_location_ma_5`                    | +0.1345 | +4.7995   | +0.1343 | +0.1187 | 0.0049 | 0.606 (Ref($close_qfq, 5)/Ref($close_qfq, 1) - 1)           |
| 6   | `roe_momentum_5d_interaction`            | +0.1278 | +5.6139   | +0.1304 | +0.1255 | 0.0034 | 0.594 (Ref($roe, 1))                                        |
| 7   | `earnings_yield_momentum_5d_interaction` | +0.1113 | +4.5170   | +0.1147 | +0.1011 | 0.0245 | 0.648 (Ref($pe_ttm, 1))                                     |
| 8   | `turnover_deviation_20d`                 | +0.1027 | +4.6547   | +0.1050 | +0.1055 | 0.0188 | 0.361 (Ref($volume_ratio, 1))                               |
| 9   | `winsorized_zscore_volume_ratio`         | +0.0991 | +5.7512   | +0.0948 | +0.1189 | 0.0000 | 0.369 (Ref($amount, 1) / Mean(Ref($amount, 1), 5) - 1)      |
| 10  | `turnover_acceleration_5d`               | +0.0860 | +4.3456   | +0.0873 | +0.0868 | 0.0054 | 0.513 (Ref($volume_ratio, 1))                               |
| 11  | `overnight_return_5d_avg`                | +0.0675 | +1.3247   | +0.0596 | +0.0881 | 0.0059 | 0.247 (Ref($close_qfq, 5)/Ref($close_qfq, 1) - 1)           |
| 12  | `sma_volume_ratio_10d`                   | +0.0482 | +2.1169   | +0.0482 | +0.0434 | 0.0103 | 0.376 (Ref($close_qfq, 1)/Mean(Ref($close_qfq, 1), 10) - 1) |
| 13  | `eps_yield`                              | +0.0393 | +2.5795   | +0.0427 | +0.0316 | 0.0012 | 0.586 (Ref($pe_ttm, 1))                                     |
| 14  | `amihud_illiquidity_20d`                 | +0.0380 | +2.1552   | +0.0594 | +0.0172 | 0.0194 | 0.557 (Ref($total_mv, 1))                                   |
| 15  | `roe_momentum_60d`                       | +0.0361 | +0.6836   | +0.0344 | +0.0377 | 0.0405 | 0.509 (Ref($roe, 1))                                        |
| 16  | `intraday_range`                         | +0.0333 | +0.8424   | +0.0342 | +0.0296 | 0.0012 | 0.493 (Std(Ref($close_qfq, 1), 5))                          |
| 17  | `Turnover_Acceleration_10D`              | +0.0316 | +2.2881   | +0.0369 | +0.0121 | 0.0096 | 0.427 (Ref($volume_ratio, 1))                               |
| 18  | `volume_ratio_ma_15d` 🔄                 | +0.0290 | +0.9482   | +0.0306 | +0.0155 | 0.0149 | 0.384 (Ref($macd_qfq, 1))                                   |
| 19  | `ROEAccel_60D` ⭐                         | +0.0258 | +0.3485   | +0.0253 | +0.0483 | 0.2878 | 0.417 (Ref($roe, 1))                                        |


---

## 各因子详情

### 1. `rank_close_open`

- IC = **+0.351881**, nan_ratio = 0.0012
- IC(in) = 0.354113, IC(oos) = 0.353833, IC_IR = 11.4885
- Max abs corr vs LGB = 0.0232 (`Ref($close_qfq, 2)/Ref($close_qfq, 1) - 1`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/cbb76a222a154bd399964e1bc0736433/factor.py`

```python
import pandas as pd

def calculate_rank_close_open():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    ratio = df['$close'] / df['$open']
    series = ratio.groupby(level='datetime').rank(pct=True)
    
    result = pd.DataFrame({'rank_close_open': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_rank_close_open()
```

---

### 2. `close_location`

- IC = **+0.345070**, nan_ratio = 0.0012
- IC(in) = 0.347950, IC(oos) = 0.329433, IC_IR = 16.9919
- Max abs corr vs LGB = 0.0370 (`Ref($close_qfq, 6)/Ref($close_qfq, 1) - 1`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/ebf2554a4bf54b8fba4dd958ab289486/factor.py`

```python
import pandas as pd
import numpy as np

def calculate_close_location():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    close = df['$close']
    high = df['$high']
    low = df['$low']
    
    denom = high - low
    series = np.where(denom != 0, (close - low) / denom, 0.5)
    series = pd.Series(series, index=df.index, dtype='float64')
    
    result = pd.DataFrame({'close_location': series}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_close_location()
```

---

### 3. `log_signed_volume`

- IC = **+0.333016**, nan_ratio = 0.0012
- IC(in) = 0.333363, IC(oos) = 0.333736, IC_IR = 27.7288
- Max abs corr vs LGB = 0.0260 (`Ref($close_qfq, 6)/Ref($close_qfq, 1) - 1`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/02efb81eff3e414ea32e4d697957855d/factor.py`

```python
import pandas as pd
import numpy as np

def calculate_log_signed_volume():
    df = pd.read_hdf('daily_pv.h5')

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Forward-fill fundamental columns
    fundamental_cols = ['$pe_ttm', '$pb', '$roe', '$q_profit_yoy', '$q_eps', '$ps_ttm', '$dv_ratio']
    for col in fundamental_cols:
        if col in df.columns:
            df[col] = df[col].groupby(level='instrument').transform(lambda s: s.ffill())

    # Calculate log_signed_volume = sign(Close - Open) * ln(Volume + 1)
    # Using $close and $open as they represent the same day's prices
    sign_diff = np.sign(df['$close'] - df['$open'])
    log_vol = np.log1p(df['$volume'])
    
    series = sign_diff * log_vol

    result = pd.DataFrame({'log_signed_volume': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_log_signed_volume()
```

---

### 4. `ValueMR_20D`

- IC = **+0.156560**, nan_ratio = 0.133
- IC(in) = 0.154760, IC(oos) = 0.147268, IC_IR = 7.1405
- Max abs corr vs LGB = 0.6846 (`Ref($rsi_qfq_12, 1)`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/7414735840764942a3bc424a8e107213/factor.py`

```python
import pandas as pd
import numpy as np

def calculate_ValueMR_20D():
    # Load data
    df = pd.read_hdf('daily_pv.h5')
    
    # Coerce all columns to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Forward-fill fundamental column $pe_ttm within each instrument
    df['$pe_ttm'] = df['$pe_ttm'].groupby(level='instrument').transform(lambda s: s.ffill())
    
    # Compute 20-day rolling mean and std of $pe_ttm per instrument
    rolling_mean = df['$pe_ttm'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=20, min_periods=1).mean()
    )
    rolling_std = df['$pe_ttm'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=20, min_periods=1).std()
    )
    
    # Compute z-score
    z_score = (df['$pe_ttm'] - rolling_mean) / rolling_std
    
    # Clip z-score to [-3, 3]
    series = z_score.clip(-3, 3)
    
    # Save result
    result = pd.DataFrame({'ValueMR_20D': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_ValueMR_20D()
```

---

### 5. `close_location_ma_5`

- IC = **+0.134528**, nan_ratio = 0.0049
- IC(in) = 0.134273, IC(oos) = 0.118697, IC_IR = 4.7995
- Max abs corr vs LGB = 0.6063 (`Ref($close_qfq, 5)/Ref($close_qfq, 1) - 1`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/6186b3f986b34c8d823bb06a32ebe74c/factor.py`

```python
import pandas as pd
import numpy as np

def calculate_close_location_ma_5():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Calculate close_location: (Close - Low) / (High - Low), with 0.5 when High == Low
    df['close_location'] = np.where(
        df['$high'] == df['$low'], 
        0.5, 
        (df['$close'] - df['$low']) / (df['$high'] - df['$low'])
    )
    
    series = df['close_location'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=5, min_periods=5).mean()
    )
    
    result = pd.DataFrame({'close_location_ma_5': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_close_location_ma_5()
```

---

### 6. `roe_momentum_5d_interaction`

- IC = **+0.127850**, nan_ratio = 0.0034
- IC(in) = 0.130436, IC(oos) = 0.125505, IC_IR = 5.6139
- Max abs corr vs LGB = 0.5937 (`Ref($roe, 1)`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/c7cdaa39d15d493c869a61c1ad3a6a6b/factor.py`

```python
import pandas as pd

def calculate_roe_momentum_5d_interaction():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Forward-fill fundamental data
    df['$roe'] = df['$roe'].groupby(level='instrument').transform(lambda s: s.ffill())
    
    # Calculate 5-day price momentum using $close column as specified in variables
    df['mom_5d'] = df['$close'].groupby(level='instrument').transform(lambda s: s.pct_change(periods=5))
    
    # Calculate cross-sectional rank as rank / count
    df['roe_rank'] = df['$roe'].groupby(level='datetime').rank() / df['$roe'].groupby(level='datetime').transform('count')
    df['mom_rank'] = df['mom_5d'].groupby(level='datetime').rank() / df['mom_5d'].groupby(level='datetime').transform('count')
    
    # Compute interaction factor
    series = df['roe_rank'] * df['mom_rank']
    
    result = pd.DataFrame({'roe_momentum_5d_interaction': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_roe_momentum_5d_interaction()
```

---

### 7. `earnings_yield_momentum_5d_interaction`

- IC = **+0.111274**, nan_ratio = 0.0245
- IC(in) = 0.114712, IC(oos) = 0.101118, IC_IR = 4.5170
- Max abs corr vs LGB = 0.6481 (`Ref($pe_ttm, 1)`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/b8fcc774b95d406e8fe0b8085fa1f92d/factor.py`

```python
import pandas as pd

def calculate_earnings_yield_momentum_5d_interaction():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Forward-fill fundamental columns
    df['$pe_ttm'] = df['$pe_ttm'].groupby(level='instrument').transform(lambda s: s.ffill())
    
    # Calculate earnings yield
    df['earnings_yield'] = 1.0 / df['$pe_ttm']
    
    # Calculate 5-day momentum using raw close price as per variable definition
    df['momentum_5d'] = df['$close'].groupby(level='instrument').transform(lambda s: s.pct_change(periods=5))
    
    # Cross-sectional rank (rank / count)
    df['ey_rank'] = df.groupby(level='datetime')['earnings_yield'].rank(method='average') / df.groupby(level='datetime')['earnings_yield'].transform('count')
    df['mom_rank'] = df.groupby(level='datetime')['momentum_5d'].rank(method='average') / df.groupby(level='datetime')['momentum_5d'].transform('count')
    
    # Interaction factor
    series = df['ey_rank'] * df['mom_rank']
    
    result = pd.DataFrame({'earnings_yield_momentum_5d_interaction': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_earnings_yield_momentum_5d_interaction()
```

---

### 8. `turnover_deviation_20d`

- IC = **+0.102656**, nan_ratio = 0.0188
- IC(in) = 0.104968, IC(oos) = 0.105465, IC_IR = 4.6547
- Max abs corr vs LGB = 0.3608 (`Ref($volume_ratio, 1)`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/4441c31cd4694ae2b0f18e10430f47fc/factor.py`

```python
import pandas as pd
import numpy as np

def calculate_turnover_deviation_20d():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    series = df['$turnover_rate'].groupby(level='instrument').transform(
        lambda s: s - s.rolling(window=20, min_periods=20).mean()
    )

    result = pd.DataFrame({'turnover_deviation_20d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_turnover_deviation_20d()
```

---

### 9. `winsorized_zscore_volume_ratio`

- IC = **+0.099123**, nan_ratio = 0.0
- IC(in) = 0.094753, IC(oos) = 0.118856, IC_IR = 5.7512
- Max abs corr vs LGB = 0.3692 (`Ref($amount, 1) / Mean(Ref($amount, 1), 5) - 1`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/1ea9fe1a506644209fb61c6476bb19cd/factor.py`

```python
import pandas as pd
import numpy as np

def calculate_winsorized_zscore_volume_ratio():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    vr = df['$volume_ratio']

    # Step 1: Compute cross-sectional mean and std for winsorization bounds
    cs_mean = vr.groupby(level='datetime').transform('mean')
    cs_std = vr.groupby(level='datetime').transform('std')

    # Step 2: Winsorize at cross-sectional mean +/- 3 standard deviations
    lower = cs_mean - 3 * cs_std
    upper = cs_mean + 3 * cs_std
    vr_win = vr.clip(lower=lower, upper=upper)

    # Step 3: Compute cross-sectional mean and std of winsorized values
    cs_mean_win = vr_win.groupby(level='datetime').transform('mean')
    cs_std_win = vr_win.groupby(level='datetime').transform('std')

    # Step 4: Z-score with epsilon for numerical stability
    z_score = (vr_win - cs_mean_win) / (cs_std_win + 1e-8)

    # Step 5: Fill remaining NaN with cross-sectional median
    cs_median = z_score.groupby(level='datetime').transform('median')
    z_score_filled = z_score.fillna(cs_median)

    result = pd.DataFrame({'winsorized_zscore_volume_ratio': z_score_filled.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_winsorized_zscore_volume_ratio()
```

---

### 10. `turnover_acceleration_5d`

- IC = **+0.085952**, nan_ratio = 0.0054
- IC(in) = 0.087267, IC(oos) = 0.086824, IC_IR = 4.3456
- Max abs corr vs LGB = 0.5129 (`Ref($volume_ratio, 1)`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/48d092bb42c14aba9eac39dc6c9b9453/factor.py`
*(同名实现 3 个，已选最新修改版本)*

```python
import pandas as pd

def calculate_turnover_acceleration_5d():
    df = pd.read_hdf('daily_pv.h5')
    
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    fundamental_cols = ['$pe_ttm', '$pb', '$roe', '$q_profit_yoy', '$q_eps', '$assets_turn', '$profit_to_gr']
    for col in fundamental_cols:
        if col in df.columns:
            df[col] = df[col].groupby(level='instrument').transform(lambda s: s.ffill())
            
    series = df['$turnover_rate'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=5, min_periods=5).mean() / s.rolling(window=10, min_periods=10).mean() - 1
    )
    
    result = pd.DataFrame({'turnover_acceleration_5d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_turnover_acceleration_5d()
```

---

### 11. `overnight_return_5d_avg`

- IC = **+0.067451**, nan_ratio = 0.0059
- IC(in) = 0.059630, IC(oos) = 0.088055, IC_IR = 1.3247
- Max abs corr vs LGB = 0.2469 (`Ref($close_qfq, 5)/Ref($close_qfq, 1) - 1`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/1e7af3eb2de04a05a59b10a0830d77d3/factor.py`

```python
import pandas as pd

def calculate_overnight_return_5d_avg():
    df = pd.read_hdf('daily_pv.h5')
    
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    fundamental_cols = ['$pe_ttm', '$pb', '$ps_ttm', '$roe', '$q_profit_yoy', '$q_eps', '$dv_ratio']
    for col in fundamental_cols:
        if col in df.columns:
            df[col] = df[col].groupby(level='instrument').transform(lambda s: s.ffill())
            
    prev_close = df['$close'].groupby(level='instrument').shift(1)
    overnight_ret = (df['$open'] - prev_close) / prev_close
    
    series = overnight_ret.groupby(level='instrument').transform(
        lambda s: s.rolling(window=5, min_periods=5).mean()
    )
    
    result = pd.DataFrame({'overnight_return_5d_avg': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_overnight_return_5d_avg()
```

---

### 12. `sma_volume_ratio_10d`

- IC = **+0.048159**, nan_ratio = 0.0103
- IC(in) = 0.048249, IC(oos) = 0.043409, IC_IR = 2.1169
- Max abs corr vs LGB = 0.3764 (`Ref($close_qfq, 1)/Mean(Ref($close_qfq, 1), 10) - 1`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/70fdc44abd624bb995a2f0b95bbb8383/factor.py`

```python
import pandas as pd
import numpy as np

def calculate_sma_volume_ratio_10d():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    series = df['$volume_ratio'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=10, min_periods=10).mean()
    )
    result = pd.DataFrame({'sma_volume_ratio_10d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_sma_volume_ratio_10d()
```

---

### 13. `eps_yield`

- IC = **+0.039334**, nan_ratio = 0.0012
- IC(in) = 0.042666, IC(oos) = 0.031594, IC_IR = 2.5795
- Max abs corr vs LGB = 0.5864 (`Ref($pe_ttm, 1)`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/35f64f7b13f640ff9418c9b3b32beac9/factor.py`

```python
import pandas as pd
import numpy as np

def calculate_eps_yield():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['$q_eps'] = df['$q_eps'].groupby(level='instrument').transform(lambda s: s.ffill())
    series = df['$q_eps'] / df['$close']
    result = pd.DataFrame({'eps_yield': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_eps_yield()
```

---

### 14. `amihud_illiquidity_20d`

- IC = **+0.038011**, nan_ratio = 0.0194
- IC(in) = 0.059419, IC(oos) = 0.017217, IC_IR = 2.1552
- Max abs corr vs LGB = 0.5566 (`Ref($total_mv, 1)`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/b3ae6589f756424383d79b088c63a57f/factor.py`

```python
import pandas as pd

def calculate_amihud_illiquidity_20d():
    df = pd.read_hdf('daily_pv.h5')
    
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    fundamental_cols = ['$pe_ttm', '$pb', '$ps_ttm', '$roe', '$q_profit_yoy', '$q_eps', '$dv_ratio']
    for col in fundamental_cols:
        if col in df.columns:
            df[col] = df[col].groupby(level='instrument').transform(lambda s: s.ffill())
            
    # Calculate daily return using $close
    ret = df['$close'].groupby(level='instrument').transform(lambda s: s.pct_change())
    
    # Calculate daily Amihud illiquidity: |return| / (close * volume)
    daily_amihud = ret.abs() / (df['$close'] * df['$volume'])
    
    # Calculate 20-day rolling mean
    series = daily_amihud.groupby(level='instrument').transform(
        lambda s: s.rolling(window=20, min_periods=20).mean()
    )
    
    result = pd.DataFrame({'amihud_illiquidity_20d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_amihud_illiquidity_20d()
```

---

### 15. `roe_momentum_60d`

- IC = **+0.036138**, nan_ratio = 0.0405
- IC(in) = 0.034411, IC(oos) = 0.037685, IC_IR = 0.6836
- Max abs corr vs LGB = 0.5091 (`Ref($roe, 1)`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/57c03556f09b4fd8bb2ee21f49216ce0/factor.py`

```python
import pandas as pd
import numpy as np

def calculate_roe_momentum_60d():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['$roe'] = df['$roe'].groupby(level='instrument').transform(lambda s: s.ffill())
    series = df['$roe'].groupby(level='instrument').transform(lambda s: s.diff(60))
    result = pd.DataFrame({'roe_momentum_60d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_roe_momentum_60d()
```

---

### 16. `intraday_range`

- IC = **+0.033306**, nan_ratio = 0.0012
- IC(in) = 0.034238, IC(oos) = 0.029639, IC_IR = 0.8424
- Max abs corr vs LGB = 0.4933 (`Std(Ref($close_qfq, 1), 5)`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/b640623c6bd34e2188d15d0cbb96b4db/factor.py`
*(同名实现 3 个，已选最新修改版本)*

```python
import pandas as pd
import numpy as np

def calculate_intraday_range():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    series = (df['$high'] - df['$low']) / df['$close']

    result = pd.DataFrame({'intraday_range': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_intraday_range()
```

---

### 17. `Turnover_Acceleration_10D`

- IC = **+0.031560**, nan_ratio = 0.0096
- IC(in) = 0.036878, IC(oos) = 0.012078, IC_IR = 2.2881
- Max abs corr vs LGB = 0.4273 (`Ref($volume_ratio, 1)`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/77ccd25106bd4b07aace89d3ada76e3e/factor.py`

```python
import pandas as pd
import numpy as np

def calculate_Turnover_Acceleration_10D():
    # Load data
    df = pd.read_hdf('daily_pv.h5')

    # Coerce all columns to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Forward-fill fundamental columns
    fundamental_cols = ['$pe_ttm', '$pb', '$ps_ttm', '$roe', '$q_profit_yoy', '$q_eps', '$dv_ratio']
    for col in fundamental_cols:
        if col in df.columns:
            df[col] = df[col].groupby(level='instrument').transform(lambda s: s.ffill())

    # Compute 5-day moving average of turnover rate
    ma5 = df['$turnover_rate'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=5, min_periods=5).mean()
    )

    # Compute 5-day moving average of turnover rate lagged by 5 days
    ma5_lag5 = ma5.groupby(level='instrument').transform(
        lambda s: s.shift(5)
    )

    # Compute Turnover Acceleration 10D
    series = ma5 / ma5_lag5 - 1

    # Save result
    result = pd.DataFrame({'Turnover_Acceleration_10D': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_Turnover_Acceleration_10D()
```

---

### 18. `volume_ratio_ma_15d` 🔄 重新加入

- IC = **+0.028983**, nan_ratio = 0.0149
- IC(in) = 0.030621, IC(oos) = 0.015477, IC_IR = 0.9482
- Max abs corr vs LGB = 0.3835 (`Ref($macd_qfq, 1)`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/e84f0ea7d4784c9cb0b21b26370ffd53/factor.py`

```python
import pandas as pd
import numpy as np

def calculate_volume_ratio_ma_15d():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Forward-fill fundamental columns
    fundamental_cols = ['$pe_ttm', '$pb', '$ps_ttm', '$roe', '$q_profit_yoy', '$q_eps', '$dv_ratio', '$total_mv']
    for col in fundamental_cols:
        if col in df.columns:
            df[col] = df[col].groupby(level='instrument').transform(lambda s: s.ffill())

    # Calculate 15-day rolling mean of volume_ratio
    series = df['$volume_ratio'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=15).mean()
    )

    result = pd.DataFrame({'volume_ratio_ma_15d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_volume_ratio_ma_15d()
```

---

### 19. `ROEAccel_60D` ⭐ 新增

- IC = **+0.025799**, nan_ratio = 0.2878
- IC(in) = 0.025298, IC(oos) = 0.048349, IC_IR = 0.3485
- Max abs corr vs LGB = 0.4172 (`Ref($roe, 1)`)
- 源文件: `git_ignore_folder/RD-Agent_workspace/58cf8f15f4934a219313c838c586380a/factor.py`
*(同名实现 2 个，已选最新修改版本)*

```python
import pandas as pd
import numpy as np

def calculate_ROEAccel_60D():
    df = pd.read_hdf('daily_pv.h5')

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Forward-fill ROE within each instrument (fundamental/quarterly data)
    df['$roe'] = df['$roe'].groupby(level='instrument').transform(lambda s: s.ffill())

    # Calculate 60-day change in ROE per instrument
    roe_delta = df['$roe'].groupby(level='instrument').transform(lambda s: s - s.shift(60))

    # Calculate 60-day rolling standard deviation of the delta per instrument
    roe_delta_std = roe_delta.groupby(level='instrument').transform(
        lambda s: s.rolling(window=60, min_periods=2).std()
    )

    # Compute the factor: ROE acceleration normalized by its rolling std
    series = roe_delta / roe_delta_std
    series = series.replace([np.inf, -np.inf], np.nan)

    result = pd.DataFrame({'ROEAccel_60D': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_ROEAccel_60D()
```

---

## 组合模型历史性能（debug 日志解析）


| 时间               | IC     | IC_IR      | Composite | Enhanced | Div.Bonus | 来源日志             |
| ---------------- | ------ | ---------- | --------- | -------- | --------- | ---------------- |
| 2026-05-05 11:21 | 0.4459 | **4.6623** | 9.3246    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-05 11:57 | 0.4397 | **4.5901** | 9.1801    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-05 12:16 | 0.4453 | **4.5986** | 9.1971    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-05 12:42 | 0.4503 | **4.7272** | 9.4544    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-08 18:02 | 0.4474 | **4.7272** | 9.4543    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-08 18:32 | 0.4505 | **4.8189** | 9.6377    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-08 19:01 | 0.4415 | **4.5552** | 9.1105    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-08 19:24 | 0.4197 | **4.2502** | 8.5003    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-08 19:43 | 0.4463 | **4.6556** | 9.3112    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-08 20:11 | 0.4502 | **4.7853** | 9.5706    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-08 20:37 | 0.3596 | **3.9938** | 7.9875    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-08 21:01 | 0.4492 | **4.7629** | 9.5259    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-08 21:29 | 0.4464 | **4.7289** | 9.4578    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-09 13:10 | 0.4491 | **4.8225** | 9.6450    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-09 13:22 | 0.4467 | **4.7081** | 9.4162    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-09 13:35 | 0.4447 | **4.6702** | 9.3404    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-09 13:48 | 0.4417 | **4.6028** | 9.2057    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-09 14:02 | 0.4450 | **4.6676** | 9.3353    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-09 14:32 | 0.4051 | **3.9483** | 7.8967    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-09 14:46 | 0.4457 | **4.6724** | 9.3447    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-09 14:59 | 0.4456 | **4.6840** | 9.3679    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-09 15:16 | 0.4444 | **4.6583** | 9.3166    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-09 15:34 | 0.4461 | **4.6986** | 9.3971    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-13 14:21 | 0.4466 | **4.7306** | 9.4612    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-13 14:40 | 0.4481 | **4.7172** | 9.4344    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-13 15:02 | 0.4455 | **4.7015** | 9.4030    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-13 15:15 | 0.4475 | **4.7193** | 9.4386    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-13 15:34 | 0.4476 | **4.7427** | 9.4853    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-13 17:07 | 0.4505 | **4.8011** | 9.6021    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-13 17:31 | 0.4380 | **4.5321** | 9.0642    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-13 17:57 | 0.4366 | **4.6406** | 9.2812    | N/A      | N/A       | debug-fc2594.log |
| 2026-05-24 10:01 | nan    | **4.7821** | 9.5642    | 9.7665   | 0.2023    | debug-fc2594.log |
| 2026-05-24 10:17 | nan    | **4.8026** | 9.6051    | 9.9205   | 0.3154    | debug-fc2594.log |
| 2026-05-24 10:37 | nan    | **4.8189** | 9.6379    | 9.9077   | 0.2698    | debug-fc2594.log |
| 2026-05-24 10:58 | nan    | **4.8149** | 9.6298    | 9.9235   | 0.2938    | debug-fc2594.log |
| 2026-05-24 11:19 | nan    | **4.8248** | 9.6497    | 9.8520   | 0.2023    | debug-fc2594.log |
| 2026-05-24 11:41 | nan    | **4.8099** | 9.6199    | 9.9448   | 0.3249    | debug-fc2594.log |
| 2026-05-24 12:03 | nan    | **4.8075** | 9.6150    | 9.9519   | 0.3369    | debug-fc2594.log |
| 2026-05-24 12:38 | nan    | **4.7994** | 9.5988    | 9.9010   | 0.3022    | debug-fc2594.log |
| 2026-05-24 13:11 | nan    | **4.8154** | 9.6308    | 9.9565   | 0.3256    | debug-fc2594.log |
| 2026-05-24 13:30 | nan    | **4.8046** | 9.6092    | 9.9637   | 0.3545    | debug-fc2594.log |
| 2026-05-24 16:57 | nan    | **4.7160** | 9.4320    | 9.7018   | 0.2698    | debug-fc2594.log |
| 2026-05-24 17:16 | nan    | **4.8461** | 9.6923    | 9.8953   | 0.2031    | debug-fc2594.log |
| 2026-05-24 17:40 | nan    | **4.8149** | 9.6298    | 9.9452   | 0.3154    | debug-fc2594.log |
| 2026-05-24 19:05 | nan    | **4.7776** | 9.5551    | 9.7574   | 0.2023    | debug-fc2594.log |
| 2026-05-24 19:24 | nan    | **4.7861** | 9.5722    | 9.8420   | 0.2698    | debug-fc2594.log |
| 2026-05-24 19:46 | nan    | **4.8012** | 9.6024    | 9.9046   | 0.3022    | debug-fc2594.log |
| 2026-05-24 20:57 | nan    | **4.7201** | 9.6087    | 9.8117   | 0.2031    | debug-fc2594.log |
| 2026-05-24 21:17 | nan    | **4.8312** | 10.1325   | 10.3888  | 0.2563    | debug-fc2594.log |
| 2026-05-24 21:37 | nan    | **4.8184** | 9.8602    | 10.1849  | 0.3247    | debug-fc2594.log |


## 本次会话候选因子（未采纳）


| 因子名                         | 工作空间 ID                            | 修改时间             | result.h5 |
| --------------------------- | ---------------------------------- | ---------------- | --------- |
| `risk_adj_mom_20`           | `516ae16bf7fa439ea57e403d5dfde1f8` | 2026-05-22 22:10 | ✅         |
| `quality_value`             | `d6802c6ec17e4547995bb1ad61130ccb` | 2026-05-22 22:11 | ✅         |
| `net_amount_ratio_10`       | `711310fbba1e49adbb841f3b2fa20325` | 2026-05-22 22:12 | ✅         |
| `rsi_12`                    | `e7dfe3b82dea482ca486e04fff07703a` | 2026-05-22 22:33 | ✅         |
| `kdj_qfq`                   | `55438e07ebfd43c0a614d0105211bfe1` | 2026-05-22 22:34 | ✅         |
| `log_mv`                    | `6515e2f0a8bc41619910f2c9545dc081` | 2026-05-22 22:35 | ✅         |
| `OvernightRet_20D`          | `cce6c16e7e66422dbff0b1bd129cb47e` | 2026-05-23 21:15 | ✅         |
| `ROEMom_60D`                | `5df0e4599de0425d87ea7217cb4b6417` | 2026-05-23 21:38 | ✅         |
| `OverNight_60D`             | `f5b8709570eb4f0096422bc3078bf757` | 2026-05-23 21:39 | ✅         |
| `EarnRev_60D`               | `f0c1aee9de2e4845be5e4dd06bd4c3b7` | 2026-05-23 21:55 | ✅         |
| `ResidMom_90D`              | `c2cf516d5f9b4cffb3cfecd4b8c477b8` | 2026-05-23 21:56 | ✅         |
| `OvernightRet_20D`          | `c9a724f457d249ef8600f93e605059e3` | 2026-05-24 09:51 | ✅         |
| `ROE_Momentum_60D`          | `76888ef368f64ab0b8eb5b0780c91787` | 2026-05-24 10:08 | ✅         |
| `EarningsRevision_60D`      | `bd759695921d491ebf23de6e863c57f6` | 2026-05-24 10:08 | ✅         |
| `ResidualMomentum_90D`      | `0aba07215b224e84981adaca52e27ac9` | 2026-05-24 10:27 | ✅         |
| `ROE_RankDelta_120D`        | `9f6b15bf38a44046a66547225e834c2f` | 2026-05-24 10:27 | ✅         |
| `PEG_Signal_60D`            | `3722cd0377db4c5097a2a15314da51c5` | 2026-05-24 10:48 | ✅         |
| `OvernightIntradayDiff_60D` | `fc5ece299e094ea5917c479954b4a46a` | 2026-05-24 10:48 | ✅         |
| `ValMeanRev_60D`            | `8494b9ecc6af459f90519ad95a38f34e` | 2026-05-24 11:08 | ✅         |
| `OvernightPersist_20D`      | `40f2071ffc0c413abf15fcae8634cd1a` | 2026-05-24 11:09 | ✅         |
| `ShortPressure_20D`         | `29f1a9d5041a4e3ca8f25a861ac080bd` | 2026-05-24 11:31 | ✅         |
| `RankValMeanRev_60D`        | `b198081a73f641e7ac3c50dfca589d59` | 2026-05-24 11:31 | ✅         |
| `AmihudTrend_60D`           | `13eb642a26a54657a64f45b3b2f37b00` | 2026-05-24 11:53 | ✅         |
| `ResidMom_90D`              | `c6ada104f0f84943a225661d314e5d92` | 2026-05-24 11:53 | ✅         |
| `EarnYieldQual_60D`         | `7573f5b531314117b730226e1fc47832` | 2026-05-24 12:28 | ✅         |
| `SkipMonthMom_120D`         | `1ec2b20c9e934535b229a2ceae3332e0` | 2026-05-24 12:29 | ✅         |
| `TurnoverCV_60D`            | `543a3d80bf52469d83884fe4af36415f` | 2026-05-24 13:01 | ✅         |
| `NetFlowNorm_20D`           | `636d95851aa04e7a804197e1fff1c4a3` | 2026-05-24 13:01 | ✅         |
| `MarginNetTrend_60D`        | `640fed772a7d48a9bceb2c4bd81d061d` | 2026-05-24 13:20 | ✅         |
| `RetSkew_90D`               | `42624fa48bc848e2825db2fd8e50bf2f` | 2026-05-24 13:21 | ✅         |
| `ResidMom_90D`              | `acd8951d18bd402c8adf1c5841c06ad5` | 2026-05-24 16:47 | ✅         |
| `ROE_Accel_60D`             | `c06e3f7fa58a48aab7c37eaf07145d78` | 2026-05-24 16:47 | ✅         |
| `ROE_Delta_Rank_120D`       | `f1177f2b6f5b4d39a338ebadbcd0334f` | 2026-05-24 17:05 | ✅         |
| `OvernightReturn_20D`       | `bd235165a55641c4bdc0d33628ff0a11` | 2026-05-24 17:06 | ✅         |
| `EarnRev_60D`               | `c4c134dd28be4c6899ec7e475762983e` | 2026-05-24 17:29 | ✅         |
| `OvernightGap_VolAdj_20D`   | `24316eb490cc4cef96d8e93a9aa045ba` | 2026-05-24 17:30 | ✅         |
| `EarnAccel_60D`             | `fcb408af223f44a7bb889acbf39de7ab` | 2026-05-24 18:54 | ✅         |
| `OvernightInfo_20D`         | `a0f2d45be28341baa18b2a9c0db3e552` | 2026-05-24 18:55 | ✅         |
| `ResidMom_90D`              | `d8619a09afcf42e1899c960d97b68010` | 2026-05-24 19:14 | ✅         |
| `EarnMom_60D`               | `ff3a661c6466407299edc0cf14cb7c73` | 2026-05-24 19:14 | ✅         |
| `ProfitMargin_Quality_60D`  | `61724bbd3b8b43ed873487e7461bf08f` | 2026-05-24 19:35 | ✅         |
| `SkipMom_120D`              | `a9d4421dc98f4633b4e4b9af2fa43669` | 2026-05-24 19:35 | ✅         |
| `ROE_Accel_60D`             | `c5e03252e1ec46e5b107159f2225100a` | 2026-05-24 20:46 | ✅         |
| `OvernightRet_20D`          | `b1aaf3aafb914219b8b41217180e16ad` | 2026-05-24 20:47 | ✅         |
| `ROE_Momentum_120D`         | `b8a83a7702dc4d89b9a47f8d63c89872` | 2026-05-24 21:06 | ✅         |
| `ResidMom_120D`             | `7fa6a0241a59438d9b6b99478a6962de` | 2026-05-24 21:07 | ✅         |
| `ShortPressure_20D`         | `94a85d12cabe46fd9ed5c74b29482822` | 2026-05-24 21:27 | ✅         |
| `EarnRev_60D`               | `b1e359cac71c496ea04144edf7c4db87` | 2026-05-24 21:27 | ✅         |


## manifest 版本历史摘要


| 版本  | 因子数 | 状态      | 注册时间       | 退役时间       |
| --- | --- | ------- | ---------- | ---------- |
| v1  | 5   | retired | 2026-04-20 | 2026-04-26 |
| v2  | 8   | retired | 2026-04-26 | 2026-04-27 |
| v3  | 8   | retired | 2026-04-27 | 2026-04-27 |
| v4  | 8   | retired | 2026-04-27 | 2026-04-27 |
| v5  | 2   | retired | 2026-04-27 | 2026-04-28 |
| v6  | 2   | retired | 2026-04-28 | 2026-04-28 |
| v7  | 5   | retired | 2026-04-28 | 2026-04-29 |
| v8  | 8   | retired | 2026-04-29 | 2026-04-29 |
| v9  | 6   | retired | 2026-04-29 | 2026-04-29 |
| v10 | 6   | retired | 2026-04-29 | 2026-04-30 |
| v11 | 4   | retired | 2026-04-30 | 2026-04-30 |
| v12 | 6   | retired | 2026-04-30 | 2026-04-30 |
| v13 | 4   | retired | 2026-04-30 | 2026-04-30 |
| v14 | 6   | retired | 2026-04-30 | 2026-05-05 |
| v15 | 11  | retired | 2026-05-05 | 2026-05-05 |
| v16 | 4   | retired | 2026-05-05 | 2026-05-08 |
| v17 | 15  | retired | 2026-05-08 | 2026-05-08 |
| v18 | 15  | retired | 2026-05-08 | 2026-05-13 |
| v19 | 15  | retired | 2026-05-13 | 2026-05-22 |
| v20 | 50  | retired | 2026-05-22 | 2026-05-24 |
| v21 | 19  | active  | 2026-05-24 | —          |


