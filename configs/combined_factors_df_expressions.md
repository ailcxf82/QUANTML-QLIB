# combined_factors_df 完整因子清单（50 个）

> 生成时间: 2026-05-05 14:04:14
> 时间窗: 2020-01-01 ~ 2026-04-30; 池: csi500; mode: loose; shape: [1422429, 50]

---

## 1. `rank_close_open`

- IC = **+0.351996**, nan_ratio = 0.0012
- 源文件: `git_ignore_folder/RD-Agent_workspace/cbb76a222a154bd399964e1bc0736433/factor.py`

```python
def calculate_rank_close_open():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    ratio = df['$close'] / df['$open']
    series = ratio.groupby(level='datetime').rank(pct=True)
    
    result = pd.DataFrame({'rank_close_open': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 2. `close_location`

- IC = **+0.345314**, nan_ratio = 0.0012
- 源文件: `git_ignore_folder/RD-Agent_workspace/ebf2554a4bf54b8fba4dd958ab289486/factor.py`

```python
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
```

---

## 3. `log_signed_volume`

- IC = **+0.333128**, nan_ratio = 0.0012
- 源文件: `git_ignore_folder/RD-Agent_workspace/02efb81eff3e414ea32e4d697957855d/factor.py`

```python
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
```

---

## 4. `close_location_turnover_wt`

- IC = **+0.264197**, nan_ratio = 0.0013
- 源文件: `git_ignore_folder/RD-Agent_workspace/a2d246c0f16f43889539ae51665dc19d/factor.py`

```python
def calculate_close_location_turnover_wt():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    high = df['$high']
    low = df['$low']
    close = df['$close']
    turnover_f = df['$turnover_rate_f']
    
    close_location = np.where(high != low, (close - low) / (high - low), 0.5)
    series = pd.Series(close_location, index=df.index) * turnover_f
    
    result = pd.DataFrame({'close_location_turnover_wt': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 5. `ValueMR_20D`

- IC = **+0.156577**, nan_ratio = 0.1329
- 源文件: `git_ignore_folder/RD-Agent_workspace/7414735840764942a3bc424a8e107213/factor.py`

```python
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
```

---

## 6. `close_location_ma_5`

- IC = **+0.134236**, nan_ratio = 0.005
- 源文件: `git_ignore_folder/RD-Agent_workspace/6186b3f986b34c8d823bb06a32ebe74c/factor.py`

```python
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
```

---

## 7. `roe_momentum_5d_interaction`

- IC = **+0.127711**, nan_ratio = 0.0034
- 源文件: `git_ignore_folder/RD-Agent_workspace/c7cdaa39d15d493c869a61c1ad3a6a6b/factor.py`

```python
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
```

---

## 8. `volume_ratio`

- IC = **+0.114528**, nan_ratio = 0.0018
- 源文件: `git_ignore_folder/RD-Agent_workspace/7f0dfba4fea04271a9383469b29c7a41/factor.py`

```python
def calculate_volume_ratio():
    df = pd.read_hdf('daily_pv.h5')
    
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    series = df['$volume_ratio']
    
    result = pd.DataFrame({'volume_ratio': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 9. `raw_volume_ratio`

- IC = **+0.114528**, nan_ratio = 0.0018
- 源文件: `git_ignore_folder/RD-Agent_workspace/e4aeda541f724fa881a92f26b6bdedae/factor.py`

```python
def calculate_raw_volume_ratio():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Replace inf and -inf with NaN directly from the pre-computed column
    series = df['$volume_ratio'].replace([np.inf, -np.inf], np.nan)

    result = pd.DataFrame({'raw_volume_ratio': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 10. `vol_surge_5d`

- IC = **+0.114230**, nan_ratio = 0.0059
- 源文件: `git_ignore_folder/RD-Agent_workspace/61dbe889ecc843a28e693117d6a47ffe/factor.py`

```python
def calculate_vol_surge_5d():
    df = pd.read_hdf('daily_pv.h5')
    
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    fundamental_cols = ['$pe_ttm', '$pb', '$roe', '$q_profit_yoy', '$q_eps', '$ps_ttm', '$dv_ratio']
    for col in fundamental_cols:
        if col in df.columns:
            df[col] = df[col].groupby(level='instrument').transform(lambda s: s.ffill())
            
    series = df['$volume'].groupby(level='instrument').transform(lambda s: s / s.rolling(window=5).mean().shift(1))
    
    result = pd.DataFrame({'vol_surge_5d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 11. `earnings_yield_momentum_5d_interaction`

- IC = **+0.111687**, nan_ratio = 0.0246
- 源文件: `git_ignore_folder/RD-Agent_workspace/b8fcc774b95d406e8fe0b8085fa1f92d/factor.py`

```python
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
```

---

## 12. `quality_value_momentum_5d_composite`

- IC = **+0.101037**, nan_ratio = 0.0246
- 源文件: `git_ignore_folder/RD-Agent_workspace/346857f203c4412282f1cae2f879176f/factor.py`

```python
def calculate_quality_value_momentum_5d_composite():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Forward-fill fundamental columns
    df['$roe'] = df['$roe'].groupby(level='instrument').transform(lambda s: s.ffill())
    df['$pe_ttm'] = df['$pe_ttm'].groupby(level='instrument').transform(lambda s: s.ffill())
    
    # Calculate 5-day momentum using $close
    mom_5d = df['$close'].groupby(level='instrument').transform(lambda s: s.pct_change(periods=5))
    
    # Calculate 1 / PE_TTM
    inv_pe = 1.0 / df['$pe_ttm']
    
    # Calculate cross-sectional ranks as rank / count
    roe_rank = df['$roe'].groupby(level='datetime').transform(lambda s: s.rank() / s.count())
    inv_pe_rank = inv_pe.groupby(level='datetime').transform(lambda s: s.rank() / s.count())
    mom_5d_rank = mom_5d.groupby(level='datetime').transform(lambda s: s.rank() / s.count())
    
    # Triple product
    series = roe_rank * inv_pe_rank * mom_5d_rank
    
    result = pd.DataFrame({'quality_value_momentum_5d_composite': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 13. `rank_volume_ratio`

- IC = **+0.099408**, nan_ratio = 0.0018
- 源文件: `git_ignore_folder/RD-Agent_workspace/b68b17c0d74643f3b424c67c6763b662/factor.py`

```python
def calculate_rank_volume_ratio():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    series = df['$volume_ratio'].groupby(level='datetime').rank(pct=True)

    result = pd.DataFrame({'rank_volume_ratio': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 14. `volume_zscore_20d`

- IC = **+0.099231**, nan_ratio = 0.0188
- 源文件: `git_ignore_folder/RD-Agent_workspace/e4a89e4557d149b189e95c0676d966da/factor.py`

```python
def calculate_volume_zscore_20d():
    df = pd.read_hdf('daily_pv.h5')

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Forward-fill fundamental columns
    for col in ['$pe_ttm', '$pb', '$roe', '$q_profit_yoy', '$q_eps', '$ps_ttm', '$dv_ratio']:
        if col in df.columns:
            df[col] = df[col].groupby(level='instrument').transform(lambda s: s.ffill())

    # Calculate 20-day rolling z-score of trading volume and clip to [-5, 5]
    series = df['$volume'].groupby(level='instrument').transform(
        lambda s: ((s - s.rolling(window=20, min_periods=20).mean()) / s.rolling(window=20, min_periods=20).std()).clip(-5, 5)
    )

    result = pd.DataFrame({'volume_zscore_20d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 15. `winsorized_zscore_volume_ratio`

- IC = **+0.099134**, nan_ratio = 0.0
- 源文件: `git_ignore_folder/RD-Agent_workspace/1ea9fe1a506644209fb61c6476bb19cd/factor.py`

```python
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
```

---

## 16. `turnover_surprise_20`

- IC = **+0.096589**, nan_ratio = 0.0188
- 源文件: `git_ignore_folder/RD-Agent_workspace/97184b5f014f48f999575a0691e91ec4/factor.py`

```python
def calculate_turnover_surprise_20():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    series = df['$turnover_rate'].groupby(level='instrument').transform(
        lambda s: s / s.rolling(window=20, min_periods=20).mean()
    )
    result = pd.DataFrame({'turnover_surprise_20': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 17. `rank_turnover_deviation_20d`

- IC = **+0.079115**, nan_ratio = 0.0188
- 源文件: `git_ignore_folder/RD-Agent_workspace/a64bb578f8e44f3d87563882b2e9fa24/factor.py`

```python
def calculate_rank_turnover_deviation_20d():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    turnover_20d_mean = df['$turnover_rate'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=20, min_periods=20).mean()
    )

    ratio = df['$turnover_rate'] / turnover_20d_mean

    ranked = ratio.groupby(level='datetime').rank(pct=True)

    series = ranked - 0.5

    result = pd.DataFrame({'rank_turnover_deviation_20d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 18. `pb_pct_change_20d`

- IC = **+0.074312**, nan_ratio = 0.0136
- 源文件: `git_ignore_folder/RD-Agent_workspace/a150300283cd40c5826a630387e8300c/factor.py`

```python
def calculate_pb_pct_change_20d():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['$pb'] = df['$pb'].groupby(level='instrument').transform(lambda s: s.ffill())
    series = df['$pb'].groupby(level='instrument').transform(lambda s: s.pct_change(20))
    result = pd.DataFrame({'pb_pct_change_20d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 19. `ValueMR_60D`

- IC = **+0.066704**, nan_ratio = 0.1188
- 源文件: `git_ignore_folder/RD-Agent_workspace/8a4297499ed740f4b4646aeea42349d6/factor.py`

```python
def calculate_ValueMR_60D():
    # Load data
    df = pd.read_hdf('daily_pv.h5')
    
    # Coerce all columns to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Forward-fill fundamental column $pe_ttm within each instrument
    df['$pe_ttm'] = df['$pe_ttm'].groupby(level='instrument').transform(lambda s: s.ffill())
    
    # Compute 60-day rolling mean and std of $pe_ttm per instrument
    rolling_mean = df['$pe_ttm'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=60, min_periods=1).mean()
    )
    rolling_std = df['$pe_ttm'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=60, min_periods=1).std()
    )
    
    # Compute z-score
    series = (df['$pe_ttm'] - rolling_mean) / rolling_std
    
    # Save result
    result = pd.DataFrame({'ValueMR_60D': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 20. `sma_volume_ratio_10d`

- IC = **+0.048864**, nan_ratio = 0.0102
- 源文件: `git_ignore_folder/RD-Agent_workspace/70fdc44abd624bb995a2f0b95bbb8383/factor.py`

```python
def calculate_sma_volume_ratio_10d():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    series = df['$volume_ratio'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=10, min_periods=10).mean()
    )
    result = pd.DataFrame({'sma_volume_ratio_10d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 21. `sma_volume_ratio_5d`

- IC = **+0.046474**, nan_ratio = 0.0055
- 源文件: `git_ignore_folder/RD-Agent_workspace/02b7fea660f94534a0893f1a9aeffc77/factor.py`

```python
def calculate_sma_volume_ratio_5d():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    series = df['$volume_ratio'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=5, min_periods=5).mean()
    )
    result = pd.DataFrame({'sma_volume_ratio_5d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 22. `profit_growth_yoy`

- IC = **+0.037325**, nan_ratio = 0.0
- 源文件: `git_ignore_folder/RD-Agent_workspace/14d710ec5ac24a0bab3fa56445c62358/factor.py`

```python
def calculate_profit_growth_yoy():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['$q_profit_yoy'] = df['$q_profit_yoy'].groupby(level='instrument').transform(lambda s: s.ffill())
    series = df['$q_profit_yoy']
    result = pd.DataFrame({'profit_growth_yoy': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 23. `q_profit_yoy`

- IC = **+0.037325**, nan_ratio = 0.0
- 源文件: `git_ignore_folder/RD-Agent_workspace/90461124d2c248cb87b06a63a152bd4b/factor.py`

```python
def calculate_q_profit_yoy():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['$q_profit_yoy'] = df['$q_profit_yoy'].groupby(level='instrument').transform(lambda s: s.ffill())
    series = df['$q_profit_yoy']
    result = pd.DataFrame({'q_profit_yoy': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 24. `earnings_growth_smooth`

- IC = **+0.036822**, nan_ratio = 0.0027
- 源文件: `git_ignore_folder/RD-Agent_workspace/b711f70bd640473c9f5bb27339221be1/factor.py`

```python
def calculate_earnings_growth_smooth():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Forward-fill fundamental column $q_profit_yoy within each instrument
    df['$q_profit_yoy'] = df['$q_profit_yoy'].groupby(level='instrument').transform(lambda s: s.ffill())
    
    # Compute 5-day moving average of q_profit_yoy
    series = df['$q_profit_yoy'].groupby(level='instrument').transform(lambda s: s.rolling(window=5, min_periods=5).mean())
    
    series = series.astype('float64')
    result = pd.DataFrame({'earnings_growth_smooth': series}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 25. `roe_momentum_60d`

- IC = **+0.036040**, nan_ratio = 0.0408
- 源文件: `git_ignore_folder/RD-Agent_workspace/57c03556f09b4fd8bb2ee21f49216ce0/factor.py`

```python
def calculate_roe_momentum_60d():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['$roe'] = df['$roe'].groupby(level='instrument').transform(lambda s: s.ffill())
    series = df['$roe'].groupby(level='instrument').transform(lambda s: s.diff(60))
    result = pd.DataFrame({'roe_momentum_60d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 26. `rank_pb_roe`

- IC = **+0.032673**, nan_ratio = 0.0
- 源文件: `git_ignore_folder/RD-Agent_workspace/3669a8903e4640c9967c2e461525ec8c/factor.py`

```python
def calculate_rank_pb_roe():
    df = pd.read_hdf('daily_pv.h5')

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df['$roe'] = df['$roe'].groupby(level='instrument').transform(lambda s: s.ffill())
    df['$pb'] = df['$pb'].groupby(level='instrument').transform(lambda s: s.ffill())

    # Cross-sectional percentile rank: rank across all instruments on the same day, returning values in [0, 1]
    roe_rank = df['$roe'].groupby(level='datetime').rank(pct=True)
    pb_rank = df['$pb'].groupby(level='datetime').rank(pct=True)

    series = roe_rank - pb_rank

    result = pd.DataFrame({'rank_pb_roe': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 27. `intraday_range`

- IC = **+0.032529**, nan_ratio = 0.0012
- 出现于 2 个 workspace: `ccfc68d1770c46eda2400461f1782b1d, f0c672e5bb624be482112c81b2a3d40a`
- 源文件: `git_ignore_folder/RD-Agent_workspace/ccfc68d1770c46eda2400461f1782b1d/factor.py`

```python
def calculate_intraday_range():
    df = pd.read_hdf('daily_pv.h5')

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Forward-fill fundamental columns
    fundamental_cols = ['$pe_ttm', '$pb', '$roe', '$q_profit_yoy', '$q_eps', '$ps_ttm', '$dv_ratio']
    for col in fundamental_cols:
        if col in df.columns:
            df[col] = df[col].groupby(level='instrument').transform(lambda s: s.ffill())

    # Calculate intraday range: (High - Low) / Close
    intraday_range = (df['$high'] - df['$low']) / df['$close']

    result = pd.DataFrame({'intraday_range': intraday_range.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 28. `vol_ratio_5d_20d`

- IC = **+0.032167**, nan_ratio = 0.0188
- 源文件: `git_ignore_folder/RD-Agent_workspace/63ce3db89d194870b6fa8d71881078b8/factor.py`

```python
def calculate_vol_ratio_5d_20d():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    vol_5d_mean = df['$volume'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=5, min_periods=5).mean()
    )
    vol_20d_mean = df['$volume'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=20, min_periods=20).mean()
    )

    series = vol_5d_mean / vol_20d_mean

    result = pd.DataFrame({'vol_ratio_5d_20d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 29. `turnover_accel_5d`

- IC = **+0.032019**, nan_ratio = 0.0096
- 源文件: `git_ignore_folder/RD-Agent_workspace/eddbf82ec26a4431b25606040b516a93/factor.py`

```python
def calculate_turnover_accel_5d():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    series = df['$turnover_rate'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=5, min_periods=5).mean() / s.shift(5).rolling(window=5, min_periods=5).mean()
    )
    
    result = pd.DataFrame({'turnover_accel_5d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 30. `turnover_acceleration_5d`

- IC = **+0.032019**, nan_ratio = 0.0096
- 源文件: `git_ignore_folder/RD-Agent_workspace/53620d7dc2f240d19bd43b1b34b49934/factor.py`

```python
def calculate_turnover_acceleration_5d():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Forward-fill fundamental columns
    fundamental_cols = ['$pe_ttm', '$pb', '$ps_ttm', '$roe', '$q_profit_yoy', '$q_eps', '$dv_ratio', '$total_mv']
    for col in fundamental_cols:
        if col in df.columns:
            df[col] = df[col].groupby(level='instrument').transform(lambda s: s.ffill())

    # Compute turnover acceleration: 5-day rolling mean / 10-day rolling mean - 1
    series = df['$turnover_rate'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=5).mean() / s.rolling(window=10).mean() - 1
    )

    result = pd.DataFrame({'turnover_acceleration_5d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 31. `roe_pb_momentum_divergence_60d`

- IC = **+0.031973**, nan_ratio = 0.0408
- 源文件: `git_ignore_folder/RD-Agent_workspace/254a3020737f4b8fbd1c8a99c605910c/factor.py`

```python
def calculate_roe_pb_momentum_divergence_60d():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['$roe'] = df['$roe'].groupby(level='instrument').transform(lambda s: s.ffill())
    df['$pb'] = df['$pb'].groupby(level='instrument').transform(lambda s: s.ffill())
    
    roe_change = df['$roe'].groupby(level='instrument').transform(lambda s: s - s.shift(60))
    pb_pct_change = df['$pb'].groupby(level='instrument').transform(lambda s: (s - s.shift(60)) / s.shift(60))
    
    series = roe_change - pb_pct_change
    
    result = pd.DataFrame({'roe_pb_momentum_divergence_60d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 32. `liq_turnover_f`

- IC = **+0.031650**, nan_ratio = 0.0013
- 源文件: `git_ignore_folder/RD-Agent_workspace/343e65f9a81a49fd8245067aa9adcf7d/factor.py`

```python
def calculate_liq_turnover_f():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    series = df['$turnover_rate_f']
    result = pd.DataFrame({'liq_turnover_f': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 33. `turnover_rate_f`

- IC = **+0.031650**, nan_ratio = 0.0013
- 源文件: `git_ignore_folder/RD-Agent_workspace/8df140ea61f943b68af1739cd706cbca/factor.py`

```python
def calculate_turnover_rate_f():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    series = df['$turnover_rate_f']

    result = pd.DataFrame({'turnover_rate_f': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 34. `raw_turnover_rate_f`

- IC = **+0.031650**, nan_ratio = 0.0013
- 源文件: `git_ignore_folder/RD-Agent_workspace/93c1386db67041cfbe7fe11a53d48e6e/factor.py`

```python
def calculate_raw_turnover_rate_f():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    series = df['$turnover_rate_f'].replace([np.inf, -np.inf], np.nan)

    result = pd.DataFrame({'raw_turnover_rate_f': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 35. `roe_pb_ratio`

- IC = **+0.030990**, nan_ratio = 0.0
- 出现于 2 个 workspace: `bb8f5917a82d476f9461156ec27a5e3f, db77ede5e66240c4a8c3f2be29152596`
- 源文件: `git_ignore_folder/RD-Agent_workspace/bb8f5917a82d476f9461156ec27a5e3f/factor.py`

```python
def calculate_roe_pb_ratio():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['$roe'] = df['$roe'].groupby(level='instrument').transform(lambda s: s.ffill())
    df['$pb'] = df['$pb'].groupby(level='instrument').transform(lambda s: s.ffill())
    series = df['$roe'] / df['$pb'].where(df['$pb'] > 0.01)
    series = series.astype('float64')
    result = pd.DataFrame({'roe_pb_ratio': series}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 36. `pb_roe`

- IC = **+0.030990**, nan_ratio = 0.0
- 源文件: `git_ignore_folder/RD-Agent_workspace/dfbc90d86ce547389be9af4bb53c7873/factor.py`

```python
def calculate_pb_roe():
    df = pd.read_hdf('daily_pv.h5')

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    df['$roe'] = df['$roe'].groupby(level='instrument').transform(lambda s: s.ffill())
    df['$pb'] = df['$pb'].groupby(level='instrument').transform(lambda s: s.ffill())

    series = df['$roe'] / df['$pb']

    result = pd.DataFrame({'pb_roe': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 37. `volume_ratio_ma_15d`

- IC = **+0.029234**, nan_ratio = 0.0148
- 源文件: `git_ignore_folder/RD-Agent_workspace/e84f0ea7d4784c9cb0b21b26370ffd53/factor.py`

```python
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
```

---

## 38. `rank_high_low`

- IC = **+0.028632**, nan_ratio = 0.0012
- 源文件: `git_ignore_folder/RD-Agent_workspace/14949823ca6647cd9b97d55d7c7a5e9b/factor.py`

```python
def calculate_rank_high_low():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Compute intra-day high/low ratio
    high_low_ratio = df['$high'] / df['$low']

    # Cross-sectional percentile rank
    series = high_low_ratio.groupby(level='datetime').rank(pct=True)

    result = pd.DataFrame({'rank_high_low': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 39. `rank_roe`

- IC = **+0.026781**, nan_ratio = 0.0
- 源文件: `git_ignore_folder/RD-Agent_workspace/0845646959824b1f91191d5798a32c51/factor.py`

```python
def calculate_rank_roe():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['$roe'] = df['$roe'].groupby(level='instrument').transform(lambda s: s.ffill())
    series = df['$roe'].groupby(level='datetime').rank(pct=True)
    result = pd.DataFrame({'rank_roe': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 40. `quarterly_eps`

- IC = **+0.026569**, nan_ratio = 0.0
- 源文件: `git_ignore_folder/RD-Agent_workspace/48b98f86eb4847e2a30e5ae5980e079d/factor.py`

```python
def calculate_quarterly_eps():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['$q_eps'] = df['$q_eps'].groupby(level='instrument').transform(lambda s: s.ffill())
    series = df['$q_eps']
    result = pd.DataFrame({'quarterly_eps': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 41. `turnover_rate_f_ma_15d`

- IC = **-0.023847**, nan_ratio = 0.0143
- 源文件: `git_ignore_folder/RD-Agent_workspace/a73095d140cb44d4ba0d117c61fff3af/factor.py`

```python
def calculate_turnover_rate_f_ma_15d():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Forward-fill fundamental columns
    fundamental_cols = ['$pe_ttm', '$pb', '$ps_ttm', '$roe', '$q_profit_yoy', '$q_eps', '$dv_ratio', '$total_mv']
    for col in fundamental_cols:
        if col in df.columns:
            df[col] = df[col].groupby(level='instrument').transform(lambda s: s.ffill())

    # Compute 15-day rolling mean of free-float turnover rate per instrument
    series = df['$turnover_rate_f'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=15).mean()
    )

    result = pd.DataFrame({'turnover_rate_f_ma_15d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 42. `roe_quality`

- IC = **+0.023675**, nan_ratio = 0.0
- 源文件: `git_ignore_folder/RD-Agent_workspace/b7313e10a95d4a9e830c0218d193f65e/factor.py`

```python
def calculate_roe_quality():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['$roe'] = df['$roe'].groupby(level='instrument').transform(lambda s: s.ffill())
    series = df['$roe']
    result = pd.DataFrame({'roe_quality': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 43. `ROE_Factor`

- IC = **+0.023675**, nan_ratio = 0.0
- 源文件: `git_ignore_folder/RD-Agent_workspace/cefafa7a2d6f46728442d890ccf2b99a/factor.py`

```python
def calculate_ROE_Factor():
    # Load data
    df = pd.read_hdf('daily_pv.h5')
    
    # Coerce all columns to numeric
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Forward-fill fundamental column $roe within each instrument
    df['$roe'] = df['$roe'].groupby(level='instrument').transform(lambda s: s.ffill())
    
    # The factor is simply $roe
    series = df['$roe']
    
    # Save result
    result = pd.DataFrame({'ROE_Factor': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 44. `TURN_Factor`

- IC = **+0.022838**, nan_ratio = 0.0012
- 源文件: `git_ignore_folder/RD-Agent_workspace/f88753e451c94819a8380f5ae9744f1a/factor.py`

```python
def calculate_TURN_Factor():
    df = pd.read_hdf('daily_pv.h5')
    
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        
    # Forward-fill fundamental columns as per rules
    fundamental_cols = ['$pe_ttm', '$pb', '$roe', '$ps_ttm', '$dv_ratio', '$q_profit_yoy', '$q_eps']
    for col in fundamental_cols:
        if col in df.columns:
            df[col] = df[col].groupby(level='instrument').transform(lambda s: s.ffill())
            
    # Compute factor: TURN_Factor is simply the $turnover_rate
    series = df['$turnover_rate'].groupby(level='instrument').transform(lambda s: s)
    
    # Save result
    result = pd.DataFrame({'TURN_Factor': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 45. `roe_pe_ratio`

- IC = **+0.022002**, nan_ratio = 0.0215
- 源文件: `git_ignore_folder/RD-Agent_workspace/3385beb7a30c489490d448290a32b5b1/factor.py`

```python
def calculate_roe_pe_ratio():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    df['$pe_ttm'] = df['$pe_ttm'].groupby(level='instrument').transform(lambda s: s.ffill())
    df['$roe'] = df['$roe'].groupby(level='instrument').transform(lambda s: s.ffill())
    
    series = df['$roe'] / df['$pe_ttm']
    series = series.astype('float64')
    
    result = pd.DataFrame({'roe_pe_ratio': series}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 46. `quality_value_interaction`

- IC = **+0.021376**, nan_ratio = 0.0215
- 源文件: `git_ignore_folder/RD-Agent_workspace/3a99c87b530d4900805add78ded76f1a/factor.py`

```python
def calculate_quality_value_interaction():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Forward-fill fundamental columns
    df['$roe'] = df['$roe'].groupby(level='instrument').transform(lambda s: s.ffill())
    df['$pe_ttm'] = df['$pe_ttm'].groupby(level='instrument').transform(lambda s: s.ffill())
    
    # Compute earnings yield
    earnings_yield = 1.0 / df['$pe_ttm']
    
    # Compute cross-sectional percentile rank [0, 1] for each day
    rank_roe = df['$roe'].groupby(level='datetime').rank(pct=True)
    rank_ey = earnings_yield.groupby(level='datetime').rank(pct=True)
    
    # Interaction term
    series = rank_roe * rank_ey
    
    result = pd.DataFrame({'quality_value_interaction': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 47. `turnover_ma_10`

- IC = **-0.021209**, nan_ratio = 0.0096
- 源文件: `git_ignore_folder/RD-Agent_workspace/9abc0d6559bd4077a41d8776514022f7/factor.py`

```python
def calculate_turnover_ma_10():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    series = df['$turnover_rate'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=10, min_periods=10).mean()
    )
    result = pd.DataFrame({'turnover_ma_10': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 48. `liq_turnover_10d`

- IC = **-0.021209**, nan_ratio = 0.0096
- 源文件: `git_ignore_folder/RD-Agent_workspace/b15817094b2a47bf867acbc119427078/factor.py`

```python
def calculate_liq_turnover_10d():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    series = df['$turnover_rate'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=10, min_periods=10).mean()
    )
    result = pd.DataFrame({'liq_turnover_10d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 49. `sma_dv_ratio_10d`

- IC = **+0.020613**, nan_ratio = 0.0935
- 源文件: `git_ignore_folder/RD-Agent_workspace/ea1853e9668648bb827df4acb1f1cbd5/factor.py`

```python
def calculate_sma_dv_ratio_10d():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Forward-fill $dv_ratio as it is a valuation/fundamental column
    df['$dv_ratio'] = df['$dv_ratio'].groupby(level='instrument').transform(lambda s: s.ffill())
    
    # Calculate 10-day simple moving average
    series = df['$dv_ratio'].groupby(level='instrument').transform(lambda s: s.rolling(window=10, min_periods=10).mean())
    
    result = pd.DataFrame({'sma_dv_ratio_10d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

---

## 50. `sma_dv_ratio_5d`

- IC = **+0.020535**, nan_ratio = 0.0903
- 源文件: `git_ignore_folder/RD-Agent_workspace/fda41e683c1749059f2c2f9769397ac7/factor.py`

```python
def calculate_sma_dv_ratio_5d():
    df = pd.read_hdf('daily_pv.h5')
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    
    # Forward fill fundamental/valuation columns within each instrument
    df['$dv_ratio'] = df['$dv_ratio'].groupby(level='instrument').transform(lambda s: s.ffill())
    
    series = df['$dv_ratio'].groupby(level='instrument').transform(
        lambda s: s.rolling(window=5, min_periods=5).mean()
    )
    result = pd.DataFrame({'sma_dv_ratio_5d': series.astype('float64')}, index=df.index)
    result.to_hdf('result.h5', key='data', mode='w', format='table')
```

