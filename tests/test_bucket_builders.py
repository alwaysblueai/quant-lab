from __future__ import annotations

import pandas as pd

from alpha_lab.bucket_builders import build_size_bucket, build_trailing_return_bucket
from alpha_lab.research.bucket_builders import (
    build_past_ret_lookback_bucket,
    build_two_dim_bucket,
)


def test_build_size_bucket_returns_date_asset_bucket_frame() -> None:
    date = pd.Timestamp("2024-01-01")
    frame = pd.DataFrame(
        {
            "date": [date] * 6,
            "asset": [f"A{i}" for i in range(6)],
            "circ_mv": [10, 20, 30, 40, 50, 60],
        }
    )

    out = build_size_bucket(
        frame,
        n_buckets=3,
        bucket_labels=("small", "mid", "large"),
    )

    assert list(out.columns) == ["date", "asset", "bucket"]
    assert out["bucket"].tolist() == ["small", "small", "mid", "mid", "large", "large"]


def test_build_trailing_return_bucket_can_compute_from_close() -> None:
    rows = []
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    for asset, base in [("A", 10.0), ("B", 20.0), ("C", 30.0)]:
        for idx, date in enumerate(dates):
            rows.append({"date": date, "asset": asset, "close": base + idx})
    prices = pd.DataFrame(rows)

    out = build_trailing_return_bucket(prices, horizon=1, n_buckets=3)

    assert list(out.columns) == ["date", "asset", "bucket"]
    assert out["date"].nunique() == 2
    assert set(out["bucket"]) == {"Q1", "Q2", "Q3"}


def test_research_past_return_and_two_dim_buckets() -> None:
    rows = []
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    for asset, base in [("A", 10.0), ("B", 20.0), ("C", 30.0)]:
        for idx, date in enumerate(dates):
            rows.append({"date": date, "asset": asset, "close": base + idx})
    prices = pd.DataFrame(rows)

    past_1 = build_past_ret_lookback_bucket(prices, lookback=1, n_buckets=3)
    past_2 = build_past_ret_lookback_bucket(prices, lookback=2, n_buckets=3)
    crossed = build_two_dim_bucket(
        past_1,
        past_2,
        left_name="past1",
        right_name="past2",
    )

    assert list(crossed.columns) == ["date", "asset", "bucket"]
    assert crossed["bucket"].str.contains("past1=").all()
    assert crossed["bucket"].str.contains("past2=").all()
