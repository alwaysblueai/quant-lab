from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.data_quality.corporate_actions import adjust_for_dividends


def _legacy_adjust_for_dividends(
    prices_df: pd.DataFrame,
    dividend_df: pd.DataFrame,
) -> pd.DataFrame:
    """Reference implementation kept for numerical parity checks."""
    df = prices_df.copy()
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    for _, row in dividend_df.iterrows():
        asset = row["asset"]
        ex_date = row["date"]
        div = row["dividend_per_share"]

        mask = (df["asset"] == asset) & (df["date"] < ex_date)
        pre_ex = df[(df["asset"] == asset) & (df["date"] < ex_date)]
        if pre_ex.empty:
            continue
        prev_close = pre_ex.sort_values("date").iloc[-1]["close"]
        if prev_close <= 0:
            continue
        ratio = 1.0 - div / prev_close
        if ratio <= 0:
            continue
        df.loc[mask, "close"] = df.loc[mask, "close"] * ratio
    return df


def test_adjust_for_dividends_matches_legacy_on_same_day_multi_events() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-03",
                    "2024-01-04",
                    "2024-01-05",
                ]
            ),
            "asset": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "close": [10.0, 10.2, 10.5, 10.8, 11.0, 20.0, 20.2, 20.3, 20.5, 20.8],
        }
    )
    dividends = pd.DataFrame(
        {
            "asset": ["A", "B", "A", "A"],
            "date": pd.to_datetime(["2024-01-04", "2024-01-03", "2024-01-04", "2024-01-06"]),
            "dividend_per_share": [0.2, 0.05, 0.1, 0.3],
        }
    )

    expected = _legacy_adjust_for_dividends(prices, dividends)
    actual = adjust_for_dividends(prices, dividends)

    pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-12, atol=1e-12)


def test_adjust_for_dividends_ex_date_boundary_behavior() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "asset": ["A", "A", "A"],
            "close": [10.0, 20.0, 30.0],
        }
    )
    dividends = pd.DataFrame(
        {
            "asset": ["A", "A"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "dividend_per_share": [1.0, 2.0],
        }
    )

    actual = adjust_for_dividends(prices, dividends)
    day1 = actual.loc[actual["date"] == pd.Timestamp("2024-01-01"), "close"].iloc[0]
    day2 = actual.loc[actual["date"] == pd.Timestamp("2024-01-02"), "close"].iloc[0]
    day3 = actual.loc[actual["date"] == pd.Timestamp("2024-01-03"), "close"].iloc[0]
    assert day1 == pytest.approx(8.0)
    assert day2 == pytest.approx(20.0)
    assert day3 == pytest.approx(30.0)


def test_adjust_for_dividends_skips_missing_event_rows() -> None:
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "asset": ["A", "A", "A"],
            "close": [10.0, 11.0, 12.0],
        }
    )
    dividends = pd.DataFrame(
        {
            "asset": ["A", "A", None, "ZZZ", "A"],
            "date": pd.to_datetime(["NaT", "2024-01-03", "2024-01-03", "2024-01-03", "2024-01-03"]),
            "dividend_per_share": [0.1, np.nan, 0.1, 0.2, 0.55],
        }
    )

    actual = adjust_for_dividends(prices, dividends)
    assert actual["close"].isna().sum() == 0
    day1 = actual.loc[actual["date"] == pd.Timestamp("2024-01-01"), "close"].iloc[0]
    day2 = actual.loc[actual["date"] == pd.Timestamp("2024-01-02"), "close"].iloc[0]
    day3 = actual.loc[actual["date"] == pd.Timestamp("2024-01-03"), "close"].iloc[0]
    assert day1 == pytest.approx(9.5)
    assert day2 == pytest.approx(10.45)
    assert day3 == pytest.approx(12.0)
