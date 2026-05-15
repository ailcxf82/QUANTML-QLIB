# combined_factors_df 完整因子清单（15 个）—— v19

> 生成时间: 2026-05-13 13:58:17
> 时间窗: 2020-01-01 ~ 2026-05-12; 池: csi500; mode: strict; shape: [1426945, 15]
> manifest 当前版本: **v19**; 对比版本: v18

## 组合模型性能概况（最新会话）

| 指标 | 值 |
|---|---|
| 最新运行时间 | 2026-05-13 17:57 |
| 最新 IC | 0.4366 |
| 最新 IC_IR | **4.6406** |
| 最新 Composite | 9.2812 |
| 本轮最佳 IC_IR | **4.8225** (2026-05-09 13:10) |
| 本轮最佳 Composite | 9.6450 |

## 版本变更：v18 → v19

| 变更类型 | 因子 |
|---|---|
| ✅ 新增 | `amihud_illiquidity_20d`, `overnight_return_5d_avg`, `turnover_accel_5d` |
| 🔄 重新加入 | `turnover_acceleration_5d` |
| ❌ 移除 | `roe_momentum_5d_interaction`, `roe_momentum_60d`, `sma_dv_ratio_10d`, `sma_volume_ratio_5d` |

## 因子性能汇总（按 IC 降序）

| # | 因子名 | IC | IC_IR(全期) | IC(样本内) | IC(样本外) | nan比例 | 最大相关特征 |
|---|---|---|---|---|---|---|---|
| 1 | `rank_close_open` | +0.3520 | +11.4878 | +0.3541 | +0.3551 | 0.0012 | 0.023 (Ref($net_amount, 1)) |
| 2 | `close_location` | +0.3453 | +16.9892 | +0.3479 | +0.3301 | 0.0012 | 0.034 (Ref($close_qfq, 6)/Ref($close_qfq, 1) - 1) |
| 3 | `log_signed_volume` | +0.3333 | +27.7272 | +0.3334 | +0.3349 | 0.0012 | 0.024 (Ref($close_qfq, 1)/Mean(Ref($close_qfq, 1), 10) - 1) |
| 4 | `ValueMR_20D` | +0.1564 | +7.1402 | +0.1548 | +0.1453 | 0.1330 | 0.684 (Ref($rsi_qfq_12, 1)) |
| 5 | `close_location_ma_5` | +0.1342 | +4.7989 | +0.1343 | +0.1161 | 0.0050 | 0.608 (Ref($close_qfq, 5)/Ref($close_qfq, 1) - 1) |
| 6 | `earnings_yield_momentum_5d_interaction` | +0.1110 | +4.5170 | +0.1147 | +0.1000 | 0.0245 | 0.649 (Ref($pe_ttm, 1)) |
| 7 | `winsorized_zscore_volume_ratio` | +0.0990 | +5.7508 | +0.0948 | +0.1187 | 0.0000 | 0.369 (Ref($amount, 1) / Mean(Ref($amount, 1), 5) - 1) |
| 8 | `turnover_surprise_20` | +0.0972 | +4.4856 | +0.0957 | +0.1018 | 0.0188 | 0.390 (Ref($volume_ratio, 1)) |
| 9 | `turnover_acceleration_5d` 🔄 | +0.0860 | +4.3453 | +0.0873 | +0.0869 | 0.0054 | 0.512 (Ref($volume_ratio, 1)) |
| 10 | `overnight_return_5d_avg` ⭐ | +0.0675 | +1.3246 | +0.0596 | +0.0891 | 0.0059 | 0.247 (Ref($close_qfq, 5)/Ref($close_qfq, 1) - 1) |
| 11 | `sma_volume_ratio_10d` | +0.0492 | +2.1168 | +0.0482 | +0.0485 | 0.0102 | 0.376 (Ref($close_qfq, 1)/Mean(Ref($close_qfq, 1), 10) - 1) |
| 12 | `amihud_illiquidity_20d` ⭐ | +0.0370 | +2.1546 | +0.0594 | +0.0182 | 0.0195 | 0.555 (Ref($total_mv, 1)) |
| 13 | `intraday_range` | +0.0332 | +0.8425 | +0.0342 | +0.0314 | 0.0012 | 0.493 (Std(Ref($close_qfq, 1), 5)) |
| 14 | `turnover_accel_5d` ⭐ | +0.0323 | +2.2877 | +0.0369 | +0.0149 | 0.0096 | 0.429 (Ref($volume_ratio, 1)) |
| 15 | `volume_ratio_ma_15d` | +0.0299 | +0.9482 | +0.0306 | +0.0201 | 0.0148 | 0.390 (Ref($macd_qfq, 1)) |

---

## 各因子详情

### 1. `rank_close_open`
- IC = **+0.352046**, nan_ratio = 0.0012
- IC(in) = 0.354113, IC(oos) = 0.355095, IC_IR = 11.4878
- Max abs corr vs LGB = 0.0226 (`Ref($net_amount, 1)`)

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
- IC = **+0.345282**, nan_ratio = 0.0012
- IC(in) = 0.347946, IC(oos) = 0.330103, IC_IR = 16.9892
- Max abs corr vs LGB = 0.0335 (`Ref($close_qfq, 6)/Ref($close_qfq, 1) - 1`)

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
- IC = **+0.333254**, nan_ratio = 0.0012
- IC(in) = 0.333362, IC(oos) = 0.334911, IC_IR = 27.7272
- Max abs corr vs LGB = 0.0244 (`Ref($close_qfq, 1)/Mean(Ref($close_qfq, 1), 10) - 1`)

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
- IC = **+0.156396**, nan_ratio = 0.133
- IC(in) = 0.154753, IC(oos) = 0.145342, IC_IR = 7.1402
- Max abs corr vs LGB = 0.6842 (`Ref($rsi_qfq_12, 1)`)

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
- IC = **+0.134175**, nan_ratio = 0.005
- IC(in) = 0.134263, IC(oos) = 0.116064, IC_IR = 4.7989
- Max abs corr vs LGB = 0.6078 (`Ref($close_qfq, 5)/Ref($close_qfq, 1) - 1`)

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

### 6. `earnings_yield_momentum_5d_interaction`
- IC = **+0.111017**, nan_ratio = 0.0245
- IC(in) = 0.114707, IC(oos) = 0.100035, IC_IR = 4.5170
- Max abs corr vs LGB = 0.6495 (`Ref($pe_ttm, 1)`)

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

### 7. `winsorized_zscore_volume_ratio`
- IC = **+0.098984**, nan_ratio = 0.0
- IC(in) = 0.094754, IC(oos) = 0.118670, IC_IR = 5.7508
- Max abs corr vs LGB = 0.3691 (`Ref($amount, 1) / Mean(Ref($amount, 1), 5) - 1`)

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

### 8. `turnover_surprise_20`
- IC = **+0.097248**, nan_ratio = 0.0188
- IC(in) = 0.095683, IC(oos) = 0.101814, IC_IR = 4.4856
- Max abs corr vs LGB = 0.3898 (`Ref($volume_ratio, 1)`)

- 源文件: `git_ignore_folder/RD-Agent_workspace/97184b5f014f48f999575a0691e91ec4/factor.py`

```python
import pandas as pd
import numpy as np

def calculate_turnover_surprise_20():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    series = df['$turnover_rate'].groupby(level='instrument').transform(
        lambda s: s / s.rolling(window=20, min_periods=20).mean()
    )
    result = pd.DataFrame({'turnover_surprise_20': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_turnover_surprise_20()
```


---

### 9. `turnover_acceleration_5d` 🔄 重新加入
- IC = **+0.085966**, nan_ratio = 0.0054
- IC(in) = 0.087266, IC(oos) = 0.086919, IC_IR = 4.3453
- Max abs corr vs LGB = 0.5119 (`Ref($volume_ratio, 1)`)

- 源文件: `git_ignore_folder/RD-Agent_workspace/25b4f50ceb444cd8bf2d60fd55d6373b/factor.py`
  _(同名实现 2 个，已选最新修改版本)_

```python
import pandas as pd
import numpy as np

def calculate_turnover_acceleration_5d():
    # Load data
    df = pd.read_hdf('daily_pv.h5')

    # Coerce all columns to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Compute 5-day turnover acceleration
    series = df['$turnover_rate_f'].groupby(level='instrument').transform(
        lambda s: s - s.shift(5)
    )

    # Save result
    result = pd.DataFrame({'turnover_acceleration_5d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_turnover_acceleration_5d()
```


---

### 10. `overnight_return_5d_avg` ⭐ 新增
- IC = **+0.067497**, nan_ratio = 0.0059
- IC(in) = 0.059612, IC(oos) = 0.089051, IC_IR = 1.3246
- Max abs corr vs LGB = 0.2475 (`Ref($close_qfq, 5)/Ref($close_qfq, 1) - 1`)

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

### 11. `sma_volume_ratio_10d`
- IC = **+0.049211**, nan_ratio = 0.0102
- IC(in) = 0.048246, IC(oos) = 0.048533, IC_IR = 2.1168
- Max abs corr vs LGB = 0.3756 (`Ref($close_qfq, 1)/Mean(Ref($close_qfq, 1), 10) - 1`)

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

### 12. `amihud_illiquidity_20d` ⭐ 新增
- IC = **+0.036989**, nan_ratio = 0.0195
- IC(in) = 0.059403, IC(oos) = 0.018234, IC_IR = 2.1546
- Max abs corr vs LGB = 0.5549 (`Ref($total_mv, 1)`)

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

### 13. `intraday_range`
- IC = **+0.033247**, nan_ratio = 0.0012
- IC(in) = 0.034241, IC(oos) = 0.031395, IC_IR = 0.8425
- Max abs corr vs LGB = 0.4935 (`Std(Ref($close_qfq, 1), 5)`)

- 源文件: `git_ignore_folder/RD-Agent_workspace/b640623c6bd34e2188d15d0cbb96b4db/factor.py`
  _(同名实现 3 个，已选最新修改版本)_

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

### 14. `turnover_accel_5d` ⭐ 新增
- IC = **+0.032261**, nan_ratio = 0.0096
- IC(in) = 0.036879, IC(oos) = 0.014914, IC_IR = 2.2877
- Max abs corr vs LGB = 0.4292 (`Ref($volume_ratio, 1)`)

- 源文件: `git_ignore_folder/RD-Agent_workspace/eddbf82ec26a4431b25606040b516a93/factor.py`

```python
import pandas as pd
import numpy as np

def calculate_turnover_accel_5d():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    series = df['$turnover_rate'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=5, min_periods=5).mean() / s.shift(5).rolling(window=5, min_periods=5).mean()
    )
    
    result = pd.DataFrame({'turnover_accel_5d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')

if __name__ == '__main__':
    calculate_turnover_accel_5d()
```


---

### 15. `volume_ratio_ma_15d`
- IC = **+0.029925**, nan_ratio = 0.0148
- IC(in) = 0.030621, IC(oos) = 0.020137, IC_IR = 0.9482
- Max abs corr vs LGB = 0.3897 (`Ref($macd_qfq, 1)`)

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

## 组合模型历史性能（debug 日志解析）

| 时间 | IC | IC_IR | Composite | 来源日志 |
|---|---|---|---|---|
| 2026-04-29 11:27 | nan | **nan** | N/A | debug-fc2594.log |
| 2026-04-29 11:45 | nan | **nan** | N/A | debug-fc2594.log |
| 2026-04-29 19:11 | 0.4440 | **4.5976** | 9.1952 | debug-fc2594.log |
| 2026-04-29 19:42 | 0.4465 | **4.6386** | 9.2772 | debug-fc2594.log |
| 2026-04-29 19:59 | 0.4436 | **4.5544** | 9.1087 | debug-fc2594.log |
| 2026-04-29 20:19 | 0.4466 | **4.6312** | 9.2625 | debug-fc2594.log |
| 2026-04-29 20:37 | 0.4360 | **4.3810** | 8.7620 | debug-fc2594.log |
| 2026-04-29 20:58 | 0.4433 | **4.5705** | 9.1411 | debug-fc2594.log |
| 2026-04-29 21:26 | 0.4371 | **4.4267** | 8.8533 | debug-fc2594.log |
| 2026-04-29 21:47 | 0.4467 | **4.6940** | 9.3879 | debug-fc2594.log |
| 2026-04-29 22:12 | 0.4276 | **4.4926** | 8.9852 | debug-fc2594.log |
| 2026-04-29 22:50 | 0.4481 | **4.6659** | 9.3317 | debug-fc2594.log |
| 2026-04-29 23:28 | 0.4427 | **4.5447** | 9.0894 | debug-fc2594.log |
| 2026-05-05 09:30 | 0.4488 | **4.7370** | 9.4740 | debug-fc2594.log |
| 2026-05-05 09:51 | 0.4472 | **4.6755** | 9.3510 | debug-fc2594.log |
| 2026-05-05 10:12 | 0.4476 | **4.6723** | 9.3447 | debug-fc2594.log |
| 2026-05-05 10:29 | 0.4404 | **4.5308** | 9.0615 | debug-fc2594.log |
| 2026-05-05 10:45 | 0.4458 | **4.6353** | 9.2706 | debug-fc2594.log |
| 2026-05-05 11:07 | 0.4486 | **4.7245** | 9.4491 | debug-fc2594.log |
| 2026-05-05 11:21 | 0.4459 | **4.6623** | 9.3246 | debug-fc2594.log |
| 2026-05-05 11:57 | 0.4397 | **4.5901** | 9.1801 | debug-fc2594.log |
| 2026-05-05 12:16 | 0.4453 | **4.5986** | 9.1971 | debug-fc2594.log |
| 2026-05-05 12:42 | 0.4503 | **4.7272** | 9.4544 | debug-fc2594.log |
| 2026-05-08 18:02 | 0.4474 | **4.7272** | 9.4543 | debug-fc2594.log |
| 2026-05-08 18:32 | 0.4505 | **4.8189** | 9.6377 | debug-fc2594.log |
| 2026-05-08 19:01 | 0.4415 | **4.5552** | 9.1105 | debug-fc2594.log |
| 2026-05-08 19:24 | 0.4197 | **4.2502** | 8.5003 | debug-fc2594.log |
| 2026-05-08 19:43 | 0.4463 | **4.6556** | 9.3112 | debug-fc2594.log |
| 2026-05-08 20:11 | 0.4502 | **4.7853** | 9.5706 | debug-fc2594.log |
| 2026-05-08 20:37 | 0.3596 | **3.9938** | 7.9875 | debug-fc2594.log |
| 2026-05-08 21:01 | 0.4492 | **4.7629** | 9.5259 | debug-fc2594.log |
| 2026-05-08 21:29 | 0.4464 | **4.7289** | 9.4578 | debug-fc2594.log |
| 2026-05-09 13:10 | 0.4491 | **4.8225** | 9.6450 | debug-fc2594.log |
| 2026-05-09 13:22 | 0.4467 | **4.7081** | 9.4162 | debug-fc2594.log |
| 2026-05-09 13:35 | 0.4447 | **4.6702** | 9.3404 | debug-fc2594.log |
| 2026-05-09 13:48 | 0.4417 | **4.6028** | 9.2057 | debug-fc2594.log |
| 2026-05-09 14:02 | 0.4450 | **4.6676** | 9.3353 | debug-fc2594.log |
| 2026-05-09 14:32 | 0.4051 | **3.9483** | 7.8967 | debug-fc2594.log |
| 2026-05-09 14:46 | 0.4457 | **4.6724** | 9.3447 | debug-fc2594.log |
| 2026-05-09 14:59 | 0.4456 | **4.6840** | 9.3679 | debug-fc2594.log |
| 2026-05-09 15:16 | 0.4444 | **4.6583** | 9.3166 | debug-fc2594.log |
| 2026-05-09 15:34 | 0.4461 | **4.6986** | 9.3971 | debug-fc2594.log |
| 2026-05-13 14:21 | 0.4466 | **4.7306** | 9.4612 | debug-fc2594.log |
| 2026-05-13 14:40 | 0.4481 | **4.7172** | 9.4344 | debug-fc2594.log |
| 2026-05-13 15:02 | 0.4455 | **4.7015** | 9.4030 | debug-fc2594.log |
| 2026-05-13 15:15 | 0.4475 | **4.7193** | 9.4386 | debug-fc2594.log |
| 2026-05-13 15:34 | 0.4476 | **4.7427** | 9.4853 | debug-fc2594.log |
| 2026-05-13 17:07 | 0.4505 | **4.8011** | 9.6021 | debug-fc2594.log |
| 2026-05-13 17:31 | 0.4380 | **4.5321** | 9.0642 | debug-fc2594.log |
| 2026-05-13 17:57 | 0.4366 | **4.6406** | 9.2812 | debug-fc2594.log |

## 本次会话候选因子（未采纳）

| 因子名 | 工作空间 ID | 修改时间 | result.h5 |
|---|---|---|---|
| `momentum_20d` | `53c01310673e47c2a77c5fedfb58f5cc` | 2026-05-13 14:11 | ✅ |
| `earnings_yield` | `9da4982133f645d0bd450ec7ee22c7d1` | 2026-05-13 14:11 | ✅ |
| `turnover_volatility_20d` | `e92ac7d608764b718aa84e33957891b5` | 2026-05-13 14:12 | ✅ |
| `quality_adjusted_value` | `a4c57b5f5cdb4e099c7cac646531bf52` | 2026-05-13 14:30 | ✅ |
| `risk_adjusted_momentum_20d` | `05466f8a30914a52b42b4accf9a9b6d7` | 2026-05-13 14:31 | ✅ |
| `turnover_acceleration_20d` | `d5b07f55c58342d99c92d58628ac3fe2` | 2026-05-13 14:32 | ✅ |
| `momentum_5d` | `90c77b7e8957491b8ac87932a7f43aa7` | 2026-05-13 14:53 | ✅ |
| `log_pb` | `1229fbd8298a4e8b985b39b43993739b` | 2026-05-13 14:54 | ✅ |
| `turnover_deviation_20d` | `4441c31cd4694ae2b0f18e10430f47fc` | 2026-05-13 14:54 | ✅ |
| `momentum_5d` | `a4a9ed2061314b4484b71b70fe65a72f` | 2026-05-13 15:07 | ✅ |
| `log_roe_quality` | `e3f92fc605db4eaca91d6eab4759b601` | 2026-05-13 15:07 | ✅ |
| `rsi_mean_reversion` | `9a7d6693b33c4dfca2a5fce5f9ec49bb` | 2026-05-13 15:08 | ✅ |
| `dividend_yield` | `3039278396fc4f5a9eac0c9a8e89bc2d` | 2026-05-13 15:26 | ✅ |
| `log_market_cap` | `aa111e076e3144caaa31a44fad3950ef` | 2026-05-13 15:26 | ✅ |
| `log_price_to_sales` | `cd565521c16d4c95951cf1e87836bfaf` | 2026-05-13 15:26 | ✅ |
| `intraday_return` | `55cc57b83f4e4fec94bd740cc68cce47` | 2026-05-13 15:43 | ✅ |
| `overnight_return` | `f162d29652a546f58fd3889c15f842e8` | 2026-05-13 15:45 | ✅ |
| `volume_ratio_signal` | `043ed1483aa540fc93eff8f4bcb3c11a` | 2026-05-13 15:47 | ✅ |
| `macd_signal` | `e30f08259e9b4a0eb27a3147b4f49d48` | 2026-05-13 15:48 | ✅ |
| `Momentum_20D` | `494e9bdcca8045b7bb2d923851f5567e` | 2026-05-13 16:57 | ✅ |
| `Volatility_20D` | `c4bc52d404274157a99e39f48766f4df` | 2026-05-13 16:58 | ✅ |
| `Avg_Turnover_20D` | `eac10eacd7db4560adc2e6fe4c53af69` | 2026-05-13 16:58 | ✅ |
| `PB_Ratio` | `74d4a91d5bfb4098a618a6bac36cdb77` | 2026-05-13 16:59 | ✅ |
| `RSI_12` | `ac7bef4299034190b4a644d60df53322` | 2026-05-13 16:59 | ✅ |
| `Risk_Adj_Momentum_20D` | `04abf87235e84e6884b56770122ea7fb` | 2026-05-13 17:22 | ✅ |
| `Turnover_Weighted_Momentum_20D` | `1dcae7ba3e2b42879b1ca01c6efc4816` | 2026-05-13 17:22 | ✅ |
| `CS_Zscore_Momentum_20D` | `25000deb05c54154a664fcf822b3ac10` | 2026-05-13 17:22 | ✅ |
| `CS_Rank_PB` | `38c15901fa2a41e084e81b55edd5b2df` | 2026-05-13 17:23 | ✅ |
| `RSI_Deviation_20D` | `5a601e0bc3144ffab3ac72235e064314` | 2026-05-13 17:23 | ✅ |
| `Volume_Surprise_20D` | `047c200344974c698f79e46660e692d8` | 2026-05-13 17:48 | ✅ |
| `Intraday_Position_1D` | `946eb739c0cd4094b197b9d8df59e363` | 2026-05-13 17:48 | ✅ |
| `Overnight_Return_1D` | `bbb6ad60d97a4c848eb101945fb7a1f1` | 2026-05-13 17:48 | ✅ |
| `Intraday_Return_1D` | `52b9c0db701549d6bc35757454cee567` | 2026-05-13 17:49 | ✅ |
| `Turnover_Acceleration_10D` | `77ccd25106bd4b07aace89d3ada76e3e` | 2026-05-13 17:49 | ✅ |

## manifest 版本历史摘要

| 版本 | 因子数 | 状态 | 注册时间 | 退役时间 |
|---|---|---|---|---|
| v1 | 5 | retired | 2026-04-20 | 2026-04-26 |
| v2 | 8 | retired | 2026-04-26 | 2026-04-27 |
| v3 | 8 | retired | 2026-04-27 | 2026-04-27 |
| v4 | 8 | retired | 2026-04-27 | 2026-04-27 |
| v5 | 2 | retired | 2026-04-27 | 2026-04-28 |
| v6 | 2 | retired | 2026-04-28 | 2026-04-28 |
| v7 | 5 | retired | 2026-04-28 | 2026-04-29 |
| v8 | 8 | retired | 2026-04-29 | 2026-04-29 |
| v9 | 6 | retired | 2026-04-29 | 2026-04-29 |
| v10 | 6 | retired | 2026-04-29 | 2026-04-30 |
| v11 | 4 | retired | 2026-04-30 | 2026-04-30 |
| v12 | 6 | retired | 2026-04-30 | 2026-04-30 |
| v13 | 4 | retired | 2026-04-30 | 2026-04-30 |
| v14 | 6 | retired | 2026-04-30 | 2026-05-05 |
| v15 | 11 | retired | 2026-05-05 | 2026-05-05 |
| v16 | 4 | retired | 2026-05-05 | 2026-05-08 |
| v17 | 15 | retired | 2026-05-08 | 2026-05-08 |
| v18 | 15 | retired | 2026-05-08 | 2026-05-13 |
| v19 | 15 | active | 2026-05-13 | — |
