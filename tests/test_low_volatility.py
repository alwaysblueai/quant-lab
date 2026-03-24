from __future__ import annotations

import math

import pandas as pd
import pytest

from alpha_lab.factors.low_volatility import low_volatility
from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS, validate_factor_output


def _make_df(prices: dict[str, list[float]], start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=max(len(v) for v in prices.values()), freq="B")
    rows = []
    for asset, path in prices.items():
        for i, price in enumerate(path):
            rows.append({"date": dates[i], "asset": asset, "close": price})
    return pd.DataFrame(rows)


def test_low_volatility_is_negative_realized_volatility() -> None:
    prices = [100.0, 101.0, 103.0, 102.0, 104.0]
    result = low_volatility(_make_df({"A": prices}), window=3)
    assert tuple(result.columns) == FACTOR_OUTPUT_COLUMNS
    assert set(result["factor"]) == {"low_volatility_3d"}

    rets = pd.Series(prices).pct_change()
    expected = -rets.iloc[1:4].std(ddof=1)
    assert result["value"].iloc[:3].isna().all()
    assert math.isclose(float(result["value"].iloc[3]), float(expected), rel_tol=1e-12)


def test_low_volatility_no_future_data() -> None:
    prices = [100.0, 99.0, 101.0, 100.0, 102.0, 101.0]
    full = low_volatility(_make_df({"A": prices}), window=3)["value"]
    prefix = low_volatility(_make_df({"A": prices[:5]}), window=3)["value"]
    assert math.isclose(float(full.iloc[4]), float(prefix.iloc[4]), rel_tol=1e-12)


def test_low_volatility_validates_canonical_output() -> None:
    result = low_volatility(_make_df({"A": [100.0 + i for i in range(10)]}), window=3)
    validate_factor_output(result)


def test_low_volatility_rejects_invalid_min_periods() -> None:
    df = _make_df({"A": [100.0 + i for i in range(10)]})
    with pytest.raises(ValueError, match="cannot exceed"):
        low_volatility(df, window=3, min_periods=4)
