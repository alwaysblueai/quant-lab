from __future__ import annotations

import pandas as pd

from alpha_lab.research_integrity.leakage_checks import (
    check_asof_inputs_not_after_signal_date,
    check_cross_section_transform_scope,
    check_factor_label_temporal_order,
    check_no_future_dates_in_input,
)


def test_check_factor_label_temporal_order_fails_when_factor_after_label():
    factor_df = pd.DataFrame(
        {
            "row_id": [1, 2],
            "asset": ["AAA", "BBB"],
            "factor_date": pd.to_datetime(["2024-01-05", "2024-01-05"]),
        }
    )
    label_df = pd.DataFrame(
        {
            "row_id": [1, 2],
            "asset": ["AAA", "BBB"],
            "label_date": pd.to_datetime(["2024-01-04", "2024-01-05"]),
        }
    )

    result = check_factor_label_temporal_order(
        factor_df,
        label_df,
        join_keys=("row_id", "asset"),
        factor_date_col="factor_date",
        label_date_col="label_date",
        object_name="factor_vs_label",
    )

    assert result.status == "fail"
    assert result.severity == "error"


def test_check_no_future_dates_in_input_fails_on_future_rows():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-03", "2024-01-05"]),
            "asset": ["AAA", "AAA", "AAA"],
        }
    )

    result = check_no_future_dates_in_input(
        frame,
        max_allowed_date="2024-01-04",
        object_name="aux_input",
    )

    assert result.status == "fail"
    assert result.metrics["future_rows"] == 1


def test_check_asof_inputs_not_after_signal_date_fails_when_known_at_is_future():
    signal_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-04"]),
            "asset": ["AAA"],
        }
    )
    aux_df = pd.DataFrame(
        {
            "asset": ["AAA"],
            "effective_date": pd.to_datetime(["2024-01-04"]),
            "available_at": pd.to_datetime(["2024-01-05"]),
            "value": [1.0],
        }
    )

    result = check_asof_inputs_not_after_signal_date(
        signal_df,
        aux_df,
        by=("asset",),
        signal_date_col="date",
        aux_effective_date_col="effective_date",
        aux_known_at_col="available_at",
        object_name="pit_aux",
    )

    assert result.status == "fail"
    assert result.severity == "error"


def test_check_cross_section_transform_scope_fails_when_output_emits_new_pairs():
    raw_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "asset": ["AAA", "BBB"],
            "value": [1.0, 2.0],
        }
    )
    transformed_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "asset": ["AAA", "AAA"],
            "value": [10.0, 11.0],
        }
    )

    result = check_cross_section_transform_scope(
        raw_df,
        transformed_df,
        date_col="date",
        asset_col="asset",
        object_name="rank_transform",
    )

    assert result.status == "fail"
    assert result.severity == "error"
