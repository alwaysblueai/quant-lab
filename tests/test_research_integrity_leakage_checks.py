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


def test_check_no_future_dates_in_input_passes_datetime_success_path():
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "asset": ["AAA"] * 5,
        }
    )

    result = check_no_future_dates_in_input(
        frame,
        max_allowed_date=pd.Timestamp("2024-01-05"),
        object_name="datetime_input",
    )

    assert result.status == "pass"
    assert result.metrics["rows_checked"] == 5


def test_check_no_future_dates_in_input_still_fails_invalid_values():
    frame = pd.DataFrame({"date": ["2024-01-01", "not-a-date"], "asset": ["AAA", "BBB"]})

    result = check_no_future_dates_in_input(
        frame,
        max_allowed_date="2024-01-05",
        object_name="invalid_input",
    )

    assert result.status == "fail"
    assert "invalid timestamp" in result.message


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


def test_check_factor_label_temporal_order_passes_when_dates_equal():
    """Timestamp equality is the canonical PIT case: factor observed at ``t``,
    label *stored at* ``t`` (its value is computed from strictly future
    prices via ``forward_return``). The temporal-order check must not
    flag this — leakage discipline is enforced upstream by the label
    construction, not by the date comparison.
    """

    factor_df = pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "asset": ["AAA", "BBB", "CCC"],
            "factor_date": pd.to_datetime(["2024-01-05"] * 3),
        }
    )
    label_df = pd.DataFrame(
        {
            "row_id": [1, 2, 3],
            "asset": ["AAA", "BBB", "CCC"],
            "label_date": pd.to_datetime(["2024-01-05"] * 3),
        }
    )

    result = check_factor_label_temporal_order(
        factor_df,
        label_df,
        join_keys=("row_id", "asset"),
        factor_date_col="factor_date",
        label_date_col="label_date",
        object_name="factor_vs_label_equal",
    )

    assert result.status == "pass"
    assert result.metrics["label_after_factor_rows"] == 0


def test_check_factor_label_temporal_order_warns_when_label_after_factor():
    """Label dated *after* factor is unusual (often signal-date vs.
    realization-date semantics) — it's not strict lookahead, so the check
    should warn rather than fail. Pin this so a future tightening surfaces.
    """

    factor_df = pd.DataFrame(
        {
            "row_id": [1, 2],
            "asset": ["AAA", "BBB"],
            "factor_date": pd.to_datetime(["2024-01-04", "2024-01-04"]),
        }
    )
    label_df = pd.DataFrame(
        {
            "row_id": [1, 2],
            "asset": ["AAA", "BBB"],
            "label_date": pd.to_datetime(["2024-01-05", "2024-01-04"]),
        }
    )

    result = check_factor_label_temporal_order(
        factor_df,
        label_df,
        join_keys=("row_id", "asset"),
        factor_date_col="factor_date",
        label_date_col="label_date",
        object_name="factor_vs_label_lagged",
    )

    assert result.status == "warn"
    assert result.metrics["label_after_factor_rows"] == 1


def test_check_cross_section_transform_scope_passes_when_within_date_reordered():
    """A cross-section transform may legitimately reorder rows within a
    date (e.g. a stable rank reshuffles row positions). Scope check is
    over the (date, asset) **set**, not row order — pin the pass.
    """

    raw_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "asset": ["AAA", "BBB", "CCC"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    transformed_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "asset": ["CCC", "AAA", "BBB"],  # within-date reorder
            "value": [3.0, 1.0, 2.0],
        }
    )

    result = check_cross_section_transform_scope(
        raw_df,
        transformed_df,
        date_col="date",
        asset_col="asset",
        object_name="reorder_only",
    )

    assert result.status == "pass"


def test_check_cross_section_transform_scope_warns_or_passes_on_dropped_pairs():
    """Dropping a (date, asset) pair from the transformed output is not a
    scope expansion — it's a coverage shrinkage. Pin the current status
    so future tightening surfaces explicitly.
    """

    raw_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 3),
            "asset": ["AAA", "BBB", "CCC"],
            "value": [1.0, 2.0, 3.0],
        }
    )
    transformed_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"] * 2),
            "asset": ["AAA", "BBB"],  # CCC dropped
            "value": [1.0, 2.0],
        }
    )

    result = check_cross_section_transform_scope(
        raw_df,
        transformed_df,
        date_col="date",
        asset_col="asset",
        object_name="drop_only",
    )

    assert result.status in {"pass", "warn"}, (
        f"dropped-pair behavior changed: status={result.status} message={result.message}"
    )
