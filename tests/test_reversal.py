from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.factors.reversal import reversal
from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS, validate_factor_output


def _make_df(prices: dict[str, list[float]], start: str = "2024-01-01") -> pd.DataFrame:
    dates = pd.date_range(start, periods=max(len(v) for v in prices.values()), freq="B")
    rows = []
    for asset, path in prices.items():
        for i, price in enumerate(path):
            rows.append({"date": dates[i], "asset": asset, "close": price})
    return pd.DataFrame(rows)


def test_reversal_matches_negative_trailing_return() -> None:
    prices = [100.0, 101.0, 103.0, 102.0, 99.0, 98.0, 97.0]
    result = reversal(_make_df({"A": prices}), window=3)
    assert tuple(result.columns) == FACTOR_OUTPUT_COLUMNS
    assert set(result["factor"]) == {"reversal_3d"}

    values = result["value"].reset_index(drop=True)
    expected = -(prices[3] / prices[0] - 1.0)
    assert values.iloc[:3].isna().all()
    assert abs(values.iloc[3] - expected) < 1e-12


def test_reversal_no_future_data() -> None:
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    full = reversal(_make_df({"A": prices}), window=3)["value"]
    prefix = reversal(_make_df({"A": prices[:5]}), window=3)["value"]
    assert abs(full.iloc[4] - prefix.iloc[4]) < 1e-12


def test_reversal_validates_canonical_output() -> None:
    result = reversal(_make_df({"A": [100.0 + i for i in range(10)]}), window=3)
    validate_factor_output(result)


def test_reversal_duplicate_date_asset_raises() -> None:
    df = _make_df({"A": [1.0, 2.0, 3.0, 4.0]})
    df_bad = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        reversal(df_bad, window=2)
