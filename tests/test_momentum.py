"""Tests for alpha_lab.factors.momentum."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.factors.momentum import momentum
from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS

# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #


def _make_df(prices: dict[str, list[float]], start: str = "2020-01-02") -> pd.DataFrame:
    """Build a long-form [date, asset, close] DataFrame from a price dict."""
    dates = pd.date_range(start, periods=max(len(v) for v in prices.values()), freq="B")
    rows = []
    for asset, px in prices.items():
        for i, p in enumerate(px):
            rows.append({"date": dates[i], "asset": asset, "close": p})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# T1 – correct 20-day return value                                             #
# --------------------------------------------------------------------------- #


def test_correct_return_value():
    n = 25
    prices = [100.0 * (1.01**i) for i in range(n)]
    df = _make_df({"A": prices})
    result = momentum(df)
    factor = result.set_index("date")["value"]

    assert tuple(result.columns) == FACTOR_OUTPUT_COLUMNS
    assert set(result["factor"]) == {"momentum_20d"}

    # First 20 rows must be NaN
    assert factor.iloc[:20].isna().all()

    # Row 20 (index 20): close[20] / close[0] - 1
    for i in range(20, n):
        expected = prices[i] / prices[i - 20] - 1.0
        assert abs(factor.iloc[i] - expected) < 1e-10, (
            f"Mismatch at row {i}: got {factor.iloc[i]:.8f}, expected {expected:.8f}"
        )


# --------------------------------------------------------------------------- #
# T2 – no lookahead bias                                                       #
# --------------------------------------------------------------------------- #


def test_no_lookahead_bias():
    """Factor computed on a truncated series must equal the full-series factor."""
    n = 30
    prices = [float(100 + i) for i in range(n)]
    df_full = _make_df({"A": prices})
    result_full = momentum(df_full).set_index("date")["value"]

    # Re-compute on a prefix of 22 rows and compare the last value
    df_prefix = _make_df({"A": prices[:22]})
    result_prefix = momentum(df_prefix).set_index("date")["value"]

    # The value at row 21 (first non-NaN) must be identical
    assert abs(result_full.iloc[21] - result_prefix.iloc[21]) < 1e-10


# --------------------------------------------------------------------------- #
# T3 – insufficient history → all NaN                                          #
# --------------------------------------------------------------------------- #


def test_insufficient_history_returns_nan():
    df = _make_df({"A": [float(i) for i in range(1, 16)]})  # only 15 rows
    result = momentum(df)  # window=20
    assert result["value"].isna().all()


# --------------------------------------------------------------------------- #
# T4 – duplicate (date, asset) raises ValueError                               #
# --------------------------------------------------------------------------- #


def test_duplicate_raises():
    df = _make_df({"A": [1.0, 2.0, 3.0]})
    dupe = df.iloc[[0]].copy()
    df_bad = pd.concat([df, dupe], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate"):
        momentum(df_bad)


# --------------------------------------------------------------------------- #
# T5 – zero close price → NaN (not inf)                                        #
# --------------------------------------------------------------------------- #


def test_zero_close_produces_nan():
    prices = [0.0] + [float(i) for i in range(1, 25)]
    df = _make_df({"A": prices})
    result = momentum(df)
    # Row 20 uses close[0]=0 as denominator; result must be NaN, not inf
    assert not result["value"].isin([np.inf, -np.inf]).any()
    # The specific row that divides by zero is NaN
    assert result["value"].iloc[20] != result["value"].iloc[20]  # NaN check


# --------------------------------------------------------------------------- #
# T6 – all-NaN close → all-NaN factor, no exception                           #
# --------------------------------------------------------------------------- #


def test_all_nan_close():
    df = _make_df({"A": [float("nan")] * 25})
    result = momentum(df)
    assert result["value"].isna().all()


# --------------------------------------------------------------------------- #
# T7 – constant prices → factor == 0.0                                         #
# --------------------------------------------------------------------------- #


def test_constant_prices_zero_return():
    df = _make_df({"A": [50.0] * 25})
    result = momentum(df)
    non_nan = result["value"].dropna()
    assert len(non_nan) == 5  # rows 20-24
    assert (non_nan == 0.0).all()


# --------------------------------------------------------------------------- #
# T8 – missing required column raises KeyError                                 #
# --------------------------------------------------------------------------- #


def test_missing_column_raises():
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=5), "close": range(5)})
    with pytest.raises(KeyError, match="asset"):
        momentum(df)


# --------------------------------------------------------------------------- #
# T9 – unsorted input == sorted input                                          #
# --------------------------------------------------------------------------- #


def test_unsorted_input_same_result():
    prices = [float(100 + i) for i in range(25)]
    df_sorted = _make_df({"A": prices})
    df_shuffled = df_sorted.sample(frac=1, random_state=42).reset_index(drop=True)

    r_sorted = momentum(df_sorted).set_index(["date", "asset"]).sort_index()
    r_shuffled = momentum(df_shuffled).set_index(["date", "asset"]).sort_index()

    pd.testing.assert_frame_equal(r_sorted, r_shuffled)


# --------------------------------------------------------------------------- #
# T10 – min_periods: tolerate NaN gaps within the window                       #
# --------------------------------------------------------------------------- #


def test_min_periods_tolerates_nan_gaps():
    """With NaN gaps in the middle of a window, strict mode masks but relaxed
    mode does not, as long as both endpoints are present."""
    # 25 prices with NaN holes in the middle (rows 5-9)
    prices = [float(100 + i) for i in range(25)]
    prices_with_gaps = prices.copy()
    for i in range(5, 10):
        prices_with_gaps[i] = float("nan")

    df = _make_df({"A": prices_with_gaps})

    # Strict (default min_periods=21): rows 20-24 should be NaN because the
    # rolling window [0..20] contains only 16 valid values (5 are NaN)
    result_strict = momentum(df, window=20)
    assert result_strict["value"].iloc[20] != result_strict["value"].iloc[20]

    # Relaxed (min_periods=16): rolling window has exactly 16 valid values →
    # value should be emitted
    result_relaxed = momentum(df, window=20, min_periods=16)
    val = result_relaxed["value"].iloc[20]
    assert not np.isnan(val)
    # Value is close[20] / close[0] - 1 (pct_change uses positional shift)
    expected = prices[20] / prices[0] - 1.0
    assert abs(val - expected) < 1e-10


# --------------------------------------------------------------------------- #
# T11 – multiple assets are computed independently                             #
# --------------------------------------------------------------------------- #


def test_multiple_assets():
    df = _make_df(
        {
            "A": [float(100 + i) for i in range(25)],
            "B": [float(200 + i * 2) for i in range(25)],
        }
    )
    result = momentum(df)
    assert set(result["asset"].unique()) == {"A", "B"}

    for asset in ["A", "B"]:
        vals = result[result["asset"] == asset]["value"]
        assert vals.iloc[:20].isna().all()
        assert vals.iloc[20:].notna().all()


# --------------------------------------------------------------------------- #
# T12 – empty input returns empty DataFrame                                    #
# --------------------------------------------------------------------------- #


def test_empty_input_returns_empty_df():
    df = pd.DataFrame(columns=["date", "asset", "close"])
    result = momentum(df)
    assert isinstance(result, pd.DataFrame)
    assert result.empty
    assert list(result.columns) == list(FACTOR_OUTPUT_COLUMNS)


# --------------------------------------------------------------------------- #
# T13 – NaN in date or asset raises ValueError                                 #
# --------------------------------------------------------------------------- #


def test_nan_date_raises():
    df = _make_df({"A": [10.0, 11.0, 12.0]})
    df.loc[1, "date"] = np.nan
    with pytest.raises(ValueError, match="date"):
        momentum(df)


def test_nan_asset_raises():
    df = _make_df({"A": [10.0, 11.0, 12.0]})
    df.loc[1, "asset"] = np.nan
    with pytest.raises(ValueError, match="asset"):
        momentum(df)


# --------------------------------------------------------------------------- #
# T14 – non-positive prices produce NaN                                        #
# --------------------------------------------------------------------------- #


def test_non_positive_prices_produce_nan():
    # Prices include 0 and negative value
    prices = [10.0, 0.0, -5.0, 12.0] + [10.0] * 20
    df = _make_df({"A": prices})
    # window=1
    result = momentum(df, window=1)
    mom = result.set_index("date")["value"]

    # row 0: NaN (no history)
    # row 1: 0.0 -> non-positive -> NaN
    # row 2: -5.0 -> non-positive -> NaN
    # row 3: 12.0 / -5.0 is blocked because -5.0 is masked as NaN
    assert mom.iloc[1] != mom.iloc[1]
    assert mom.iloc[2] != mom.iloc[2]
    assert mom.iloc[3] != mom.iloc[3]


def test_sparse_calendar_uses_per_asset_history():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-06",
                    "2020-01-07",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-06",
                    "2020-01-07",
                ]
            ),
            "asset": ["A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "close": [100.0, 101.0, 102.0, 103.0, 200.0, 201.0, 202.0, 203.0, 204.0],
        }
    )

    result = momentum(df, window=1)

    asset_a = result[result["asset"] == "A"].set_index("date")["value"]
    assert np.isnan(asset_a.loc[pd.Timestamp("2020-01-01")])
    assert asset_a.loc[pd.Timestamp("2020-01-02")] == pytest.approx(101.0 / 100.0 - 1.0)
    assert asset_a.loc[pd.Timestamp("2020-01-06")] == pytest.approx(102.0 / 101.0 - 1.0)
    assert asset_a.loc[pd.Timestamp("2020-01-07")] == pytest.approx(103.0 / 102.0 - 1.0)


def test_invalid_window_raises():
    df = _make_df({"A": [10.0, 11.0, 12.0]})

    with pytest.raises(ValueError, match="window"):
        momentum(df, window=0)


def test_invalid_min_periods_raises():
    df = _make_df({"A": [10.0, 11.0, 12.0]})

    with pytest.raises(ValueError, match="min_periods"):
        momentum(df, window=2, min_periods=0)

    with pytest.raises(ValueError, match="min_periods"):
        momentum(df, window=2, min_periods=4)
