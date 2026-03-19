from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS
from alpha_lab.labels import forward_return


def _make_df(prices: dict[str, list[float]], start: str = "2020-01-02") -> pd.DataFrame:
    dates = pd.date_range(start, periods=max(len(v) for v in prices.values()), freq="B")
    rows = []
    for asset, px in prices.items():
        for i, price in enumerate(px):
            rows.append({"date": dates[i], "asset": asset, "close": price})
    return pd.DataFrame(rows)


def test_forward_return_basic_correctness():
    df = _make_df({"A": [100.0, 110.0, 121.0]})
    result = forward_return(df, horizon=1)

    assert tuple(result.columns) == FACTOR_OUTPUT_COLUMNS
    assert set(result["factor"]) == {"forward_return_1"}
    assert result["value"].iloc[0] == pytest.approx(110.0 / 100.0 - 1.0)
    assert result["value"].iloc[1] == pytest.approx(121.0 / 110.0 - 1.0)
    assert np.isnan(result["value"].iloc[2])


def test_forward_return_bad_horizon_raises():
    df = _make_df({"A": [100.0, 101.0]})

    with pytest.raises(ValueError, match="horizon"):
        forward_return(df, horizon=0)


def test_forward_return_unsorted_input_matches_sorted():
    df_sorted = _make_df({"A": [100.0, 101.0, 102.0], "B": [200.0, 198.0, 201.0]})
    df_unsorted = df_sorted.sample(frac=1, random_state=7).reset_index(drop=True)

    result_sorted = forward_return(df_sorted, horizon=1).set_index(["date", "asset"]).sort_index()
    result_unsorted = (
        forward_return(df_unsorted, horizon=1).set_index(["date", "asset"]).sort_index()
    )

    pd.testing.assert_frame_equal(result_sorted, result_unsorted)


def test_forward_return_sparse_per_asset_history_uses_row_count():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-03",
                    "2020-01-07",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-06",
                ]
            ),
            "asset": ["A", "A", "A", "B", "B", "B", "B"],
            "close": [100.0, 105.0, 110.0, 200.0, 210.0, 220.0, 230.0],
        }
    )

    result = forward_return(df, horizon=1)
    asset_a = result[result["asset"] == "A"].set_index("date")["value"]

    assert asset_a.loc[pd.Timestamp("2020-01-01")] == pytest.approx(105.0 / 100.0 - 1.0)
    assert asset_a.loc[pd.Timestamp("2020-01-03")] == pytest.approx(110.0 / 105.0 - 1.0)
    assert np.isnan(asset_a.loc[pd.Timestamp("2020-01-07")])


def test_forward_return_duplicate_rows_raise():
    df = _make_df({"A": [100.0, 101.0, 102.0]})
    df_bad = pd.concat([df, df.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="Duplicate"):
        forward_return(df_bad, horizon=1)


def test_forward_return_non_positive_prices_produce_nan():
    df = _make_df({"A": [100.0, 0.0, 110.0, -5.0]})
    result = forward_return(df, horizon=1)

    assert result["value"].iloc[0] != result["value"].iloc[0]
    assert result["value"].iloc[1] != result["value"].iloc[1]
    assert result["value"].iloc[2] != result["value"].iloc[2]


def test_forward_return_no_lookahead_on_prefix():
    prices = [100.0, 102.0, 104.0, 106.0]
    full = forward_return(_make_df({"A": prices}), horizon=1)
    prefix = forward_return(_make_df({"A": prices[:3]}), horizon=1)

    pd.testing.assert_series_equal(
        full["value"].iloc[:2], prefix["value"].iloc[:2], check_names=False
    )
