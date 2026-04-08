from __future__ import annotations

import pandas as pd

from alpha_lab.research_integrity.asof import (
    asof_join_frame,
    validate_forward_fill_lag,
)


def test_asof_join_frame_valid_alignment_matches_without_future_leakage():
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
            "asset": ["AAA", "AAA"],
        }
    )
    aux = pd.DataFrame(
        {
            "asset": ["AAA", "AAA"],
            "effective_date": pd.to_datetime(["2024-01-01", "2024-01-04"]),
            "available_at": pd.to_datetime(["2024-01-02", "2024-01-04"]),
            "aux_value": [1.0, 2.0],
        }
    )

    aligned, summary = asof_join_frame(
        signals,
        aux,
        by="asset",
        left_on="date",
        right_effective_col="effective_date",
        right_known_at_col="available_at",
    )

    assert summary.matched_count == 2
    assert summary.future_blocked_count == 0
    assert aligned["aux_value"].tolist() == [1.0, 2.0]


def test_asof_join_frame_blocks_rows_known_after_signal_date():
    signals = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-04"]),
            "asset": ["AAA", "AAA"],
        }
    )
    aux = pd.DataFrame(
        {
            "asset": ["AAA", "AAA"],
            "effective_date": pd.to_datetime(["2024-01-01", "2024-01-04"]),
            "available_at": pd.to_datetime(["2024-01-02", "2024-01-05"]),
            "aux_value": [1.0, 2.0],
        }
    )

    aligned, summary = asof_join_frame(
        signals,
        aux,
        by="asset",
        left_on="date",
        right_effective_col="effective_date",
        right_known_at_col="available_at",
        strict_known_at=True,
    )

    assert summary.future_blocked_count == 1
    assert summary.matched_count == 1
    assert aligned["aux_value"].isna().tolist() == [False, True]


def test_validate_forward_fill_lag_fails_when_lag_exceeds_boundary():
    aligned = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-10", "2024-01-11"]),
            "effective_date": pd.to_datetime(["2024-01-01", "2024-01-10"]),
        }
    )

    summary = validate_forward_fill_lag(
        aligned,
        left_date_col="date",
        matched_effective_col="effective_date",
        max_lag="3D",
        object_name="test_asof_join",
    )

    assert summary.result.status == "fail"
    assert summary.violation_count == 1
