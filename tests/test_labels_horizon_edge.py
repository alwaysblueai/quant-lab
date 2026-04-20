"""Horizon edge cases for :func:`alpha_lab.labels.forward_return`.

These tests pin the behavior on degenerate horizons so future refactors can't
silently start emitting partial or lookahead labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.exceptions import AlphaLabConfigError
from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS
from alpha_lab.labels import forward_return


def _asset_frame(prices: list[float], asset: str = "A") -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=len(prices), freq="B")
    return pd.DataFrame({"date": dates, "asset": [asset] * len(prices), "close": prices})


def test_horizon_equal_to_length_yields_all_nan():
    df = _asset_frame([100.0, 101.0, 102.0])
    result = forward_return(df, horizon=3)
    assert result["value"].isna().all()


def test_horizon_greater_than_length_yields_all_nan():
    df = _asset_frame([100.0, 101.0])
    result = forward_return(df, horizon=5)
    assert result["value"].isna().all()


def test_horizon_zero_rejected():
    df = _asset_frame([100.0, 101.0])
    with pytest.raises(AlphaLabConfigError):
        forward_return(df, horizon=0)


def test_horizon_negative_rejected():
    df = _asset_frame([100.0, 101.0])
    with pytest.raises(AlphaLabConfigError):
        forward_return(df, horizon=-1)


def test_multi_asset_asymmetric_lengths_short_asset_is_all_nan_at_horizon():
    long_df = _asset_frame([100.0, 101.0, 102.0, 103.0, 104.0], asset="L")
    short_df = _asset_frame([200.0, 201.0], asset="S")
    df = pd.concat([long_df, short_df], ignore_index=True)

    result = forward_return(df, horizon=3)
    short = result[result["asset"] == "S"]["value"]
    long = result[result["asset"] == "L"]["value"]

    assert short.isna().all(), "short asset has fewer rows than horizon → all NaN"
    # Long asset: first two labels finite, last three NaN (tail).
    assert long.iloc[:2].notna().all()
    assert long.iloc[-3:].isna().all()


def test_empty_input_returns_canonical_empty_frame():
    df = pd.DataFrame(columns=["date", "asset", "close"])
    result = forward_return(df, horizon=1)
    assert list(result.columns) == list(FACTOR_OUTPUT_COLUMNS)
    assert result.empty


def test_horizon_one_single_row_is_nan():
    df = _asset_frame([100.0])
    result = forward_return(df, horizon=1)
    assert len(result) == 1
    assert np.isnan(result["value"].iloc[0])
