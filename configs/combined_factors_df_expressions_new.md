# combined_factors_df Full Factor List (15)

> Generated at: 2026-05-09 17:48:25
> Window: 2020-01-01 ~ 2026-05-07; instruments: csi500; mode: strict; shape: [1424365, 15]

---

## 1. `log_signed_volume`

- IC = **+0.333121**, nan_ratio = 0.0012
- IC(in) = 0.333363, IC(oos) = 0.334348, IC_IR = 27.7288
- Max abs corr vs LGB = 0.0318 (`Ref($close_qfq, 6)/Ref($close_qfq, 1) - 1`)
- Source file: `git_ignore_folder/RD-Agent_workspace/02efb81eff3e414ea32e4d697957855d/factor.py`

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

if __name__ == '__main__':
    calculate_log_signed_volume()
```

---

## 2. `close_location`

- IC = **+0.345281**, nan_ratio = 0.0012
- IC(in) = 0.34795, IC(oos) = 0.330053, IC_IR = 16.9919
- Max abs corr vs LGB = 0.035 (`Ref($close_qfq, 6)/Ref($close_qfq, 1) - 1`)
- Source file: `git_ignore_folder/RD-Agent_workspace/ebf2554a4bf54b8fba4dd958ab289486/factor.py`

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

if __name__ == '__main__':
    calculate_close_location()
```

---

## 3. `rank_close_open`

- IC = **+0.351901**, nan_ratio = 0.0012
- IC(in) = 0.354113, IC(oos) = 0.354346, IC_IR = 11.4885
- Max abs corr vs LGB = 0.021 (`Ref($close_qfq, 2)/Ref($close_qfq, 1) - 1`)
- Source file: `git_ignore_folder/RD-Agent_workspace/cbb76a222a154bd399964e1bc0736433/factor.py`

```python
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

## 4. `ValueMR_20D`

- IC = **+0.156373**, nan_ratio = 0.1329
- IC(in) = 0.15476, IC(oos) = 0.145317, IC_IR = 7.1405
- Max abs corr vs LGB = 0.686 (`Ref($rsi_qfq_12, 1)`)
- Source file: `git_ignore_folder/RD-Agent_workspace/7414735840764942a3bc424a8e107213/factor.py`

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

if __name__ == '__main__':
    calculate_ValueMR_20D()
```

---

## 5. `winsorized_zscore_volume_ratio`

- IC = **+0.099009**, nan_ratio = 0.0
- IC(in) = 0.094753, IC(oos) = 0.118984, IC_IR = 5.7512
- Max abs corr vs LGB = 0.3709 (`Ref($amount, 1) / Mean(Ref($amount, 1), 5) - 1`)
- Source file: `git_ignore_folder/RD-Agent_workspace/1ea9fe1a506644209fb61c6476bb19cd/factor.py`

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

if __name__ == '__main__':
    calculate_winsorized_zscore_volume_ratio()
```

---

## 6. `roe_momentum_5d_interaction`

- IC = **+0.127643**, nan_ratio = 0.0034
- IC(in) = 0.130436, IC(oos) = 0.124567, IC_IR = 5.6139
- Max abs corr vs LGB = 0.5945 (`Ref($roe, 1)`)
- Source file: `git_ignore_folder/RD-Agent_workspace/c7cdaa39d15d493c869a61c1ad3a6a6b/factor.py`

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

if __name__ == '__main__':
    calculate_roe_momentum_5d_interaction()
```

---

## 7. `close_location_ma_5`

- IC = **+0.134094**, nan_ratio = 0.005
- IC(in) = 0.134273, IC(oos) = 0.115763, IC_IR = 4.7995
- Max abs corr vs LGB = 0.607 (`Ref($close_qfq, 5)/Ref($close_qfq, 1) - 1`)
- Source file: `git_ignore_folder/RD-Agent_workspace/6186b3f986b34c8d823bb06a32ebe74c/factor.py`

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

if __name__ == '__main__':
    calculate_close_location_ma_5()
```

---

## 8. `earnings_yield_momentum_5d_interaction`

- IC = **+0.111393**, nan_ratio = 0.0245
- IC(in) = 0.114712, IC(oos) = 0.101663, IC_IR = 4.517
- Max abs corr vs LGB = 0.647 (`Ref($pe_ttm, 1)`)
- Source file: `git_ignore_folder/RD-Agent_workspace/b8fcc774b95d406e8fe0b8085fa1f92d/factor.py`

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

if __name__ == '__main__':
    calculate_earnings_yield_momentum_5d_interaction()
```

---

## 9. `turnover_surprise_20`

- IC = **+0.096807**, nan_ratio = 0.0188
- IC(in) = 0.095683, IC(oos) = 0.100225, IC_IR = 4.4858
- Max abs corr vs LGB = 0.3917 (`Ref($volume_ratio, 1)`)
- Source file: `git_ignore_folder/RD-Agent_workspace/97184b5f014f48f999575a0691e91ec4/factor.py`

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

if __name__ == '__main__':
    calculate_turnover_surprise_20()
```

---

## 10. `sma_volume_ratio_5d`

- IC = **+0.046597**, nan_ratio = 0.0055
- IC(in) = 0.05068, IC(oos) = 0.030979, IC_IR = 3.4185
- Max abs corr vs LGB = 0.5807 (`Ref($volume_ratio, 1)`)
- Source file: `git_ignore_folder/RD-Agent_workspace/02b7fea660f94534a0893f1a9aeffc77/factor.py`

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

if __name__ == '__main__':
    calculate_sma_volume_ratio_5d()
```

---

## 11. `sma_volume_ratio_10d`

- IC = **+0.048960**, nan_ratio = 0.0102
- IC(in) = 0.048249, IC(oos) = 0.047692, IC_IR = 2.1169
- Max abs corr vs LGB = 0.3729 (`Ref($close_qfq, 1)/Mean(Ref($close_qfq, 1), 10) - 1`)
- Source file: `git_ignore_folder/RD-Agent_workspace/70fdc44abd624bb995a2f0b95bbb8383/factor.py`

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

if __name__ == '__main__':
    calculate_sma_volume_ratio_10d()
```

---

## 12. `volume_ratio_ma_15d`

- IC = **+0.029586**, nan_ratio = 0.0148
- IC(in) = 0.030621, IC(oos) = 0.018779, IC_IR = 0.9482
- Max abs corr vs LGB = 0.3834 (`Ref($macd_qfq, 1)`)
- Source file: `git_ignore_folder/RD-Agent_workspace/e84f0ea7d4784c9cb0b21b26370ffd53/factor.py`

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

if __name__ == '__main__':
    calculate_volume_ratio_ma_15d()
```

---

## 13. `sma_dv_ratio_10d`

- IC = **+0.020286**, nan_ratio = 0.0935
- IC(in) = 0.021992, IC(oos) = 0.009479, IC_IR = 0.9065
- Max abs corr vs LGB = 0.6462 (`Ref($pe, 1)`)
- Source file: `git_ignore_folder/RD-Agent_workspace/ea1853e9668648bb827df4acb1f1cbd5/factor.py`

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

if __name__ == '__main__':
    calculate_sma_dv_ratio_10d()
```

---

## 14. `intraday_range`

- IC = **+0.032902**, nan_ratio = 0.0012
- IC(in) = 0.034238, IC(oos) = 0.029234, IC_IR = 0.8424
- Max abs corr vs LGB = 0.4935 (`Std(Ref($close_qfq, 1), 5)`)
- Found in 2 workspaces: `ccfc68d1770c46eda2400461f1782b1d, f0c672e5bb624be482112c81b2a3d40a`
- Source file: `git_ignore_folder/RD-Agent_workspace/ccfc68d1770c46eda2400461f1782b1d/factor.py`

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

if __name__ == '__main__':
    calculate_intraday_range()
```

---

## 15. `roe_momentum_60d`

- IC = **+0.035738**, nan_ratio = 0.0407
- IC(in) = 0.034411, IC(oos) = 0.036515, IC_IR = 0.6836
- Max abs corr vs LGB = 0.5093 (`Ref($roe, 1)`)
- Source file: `git_ignore_folder/RD-Agent_workspace/57c03556f09b4fd8bb2ee21f49216ce0/factor.py`

```python
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

