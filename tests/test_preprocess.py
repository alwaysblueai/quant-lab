import pandas as pd

from alpha_lab.preprocess import winsorize_series, zscore_series


def test_winsorize_series_basic():
    s = pd.Series([1, 2, 3, 100])
    out = winsorize_series(s, 0.0, 0.75)
    assert len(out) == 4
    assert out.max() <= s.quantile(0.75)


def test_zscore_series_basic():
    s = pd.Series([1.0, 2.0, 3.0])
    out = zscore_series(s)
    assert round(float(out.mean()), 10) == 0.0


def test_zscore_series_constant():
    s = pd.Series([5.0, 5.0, 5.0])
    out = zscore_series(s)
    assert (out == 0).all()
