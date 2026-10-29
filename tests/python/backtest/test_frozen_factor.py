import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location('frozen_replay', Path(__file__).resolve().parents[3]/'scripts/replay_frozen_lgbm.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def prices():
    random = np.random.default_rng(42)
    return pd.DataFrame(10*np.exp(np.cumsum(random.normal(0, .02, (220, 4)), axis=0)),
                        index=pd.date_range('2024-01-01', periods=220, name='datetime'),
                        columns=pd.Index(['A', 'B', 'C', 'D'], name='instrument'))


def test_factor_matches_matrix_equations():
    close = prices()
    returns = close.pct_change(fill_method=None)
    residual = returns.sub(returns.mean(axis=1), axis=0)
    ratio = returns.rolling(20).std()/returns.rolling(120).std()
    expected = (residual.rolling(90).sum()*(1-ratio.rank(axis=1, pct=True))).rolling(20).mean().shift(1)
    actual = runner.calculate_external(close.stack(future_stack=True)).unstack()
    np.testing.assert_allclose(actual, expected, atol=1e-12, equal_nan=True)


def test_later_prices_cannot_change_prior_factor_values():
    close = prices()
    original = runner.calculate_external(close.stack(future_stack=True))
    close.iloc[190:] *= np.arange(1, 31)[:, None]
    changed = runner.calculate_external(close.stack(future_stack=True))
    pd.testing.assert_series_equal(original.loc[:close.index[190]], changed.loc[:close.index[190]])
