"""Smoke coverage for :mod:`alpha_lab.sorted_panel`."""

from __future__ import annotations

import pandas as pd

from alpha_lab.sorted_panel import SORTED_ATTR_KEY, ensure_sorted, mark_sorted


def _unsorted_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-04", "2024-01-01"]),
            "asset": ["B", "A", "A", "B"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_mark_sorted_records_attribute() -> None:
    frame = _unsorted_frame()
    out = mark_sorted(frame, by=("asset", "date"))
    assert out.attrs[SORTED_ATTR_KEY] == ("asset", "date")


def test_ensure_sorted_reorders_and_marks() -> None:
    frame = _unsorted_frame()
    sorted_frame = ensure_sorted(frame, by=("asset", "date"))

    assert sorted_frame.attrs.get(SORTED_ATTR_KEY) == ("asset", "date")
    expected_order = [("A", "2024-01-02"), ("A", "2024-01-04"), ("B", "2024-01-01"), ("B", "2024-01-03")]
    actual_order = [
        (row["asset"], row["date"].strftime("%Y-%m-%d"))
        for _, row in sorted_frame.iterrows()
    ]
    assert actual_order == expected_order


def test_ensure_sorted_is_idempotent_when_marked() -> None:
    frame = _unsorted_frame()
    first = ensure_sorted(frame, by=("asset", "date"))
    second = ensure_sorted(first, by=("asset", "date"))

    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
