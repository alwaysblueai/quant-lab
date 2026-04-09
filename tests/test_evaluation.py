from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.evaluation import compute_ic, compute_rank_ic


def _canonical(
    *,
    dates: list[str],
    assets: list[str],
    factor_name: str,
    values: list[float],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates),
            "asset": assets,
            "factor": [factor_name] * len(values),
            "value": values,
        }
    )


def test_compute_ic_basic_correctness():
    factors = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-02"],
        assets=["A", "B", "C"],
        factor_name="momentum_20d",
        values=[1.0, 2.0, 3.0],
    )
    labels = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-02"],
        assets=["A", "B", "C"],
        factor_name="forward_return_1",
        values=[2.0, 4.0, 6.0],
    )

    result = compute_ic(factors, labels)

    assert list(result.columns) == ["date", "factor", "label", "ic"]
    assert result.loc[0, "factor"] == "momentum_20d"
    assert result.loc[0, "label"] == "forward_return_1"
    assert result.loc[0, "ic"] == pytest.approx(1.0)


def test_compute_rank_ic_basic_correctness():
    factors = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-02"],
        assets=["A", "B", "C"],
        factor_name="momentum_20d",
        values=[1.0, 2.0, 3.0],
    )
    labels = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-02"],
        assets=["A", "B", "C"],
        factor_name="forward_return_1",
        values=[30.0, 10.0, 20.0],
    )

    result = compute_rank_ic(factors, labels)

    assert list(result.columns) == ["date", "factor", "label", "rank_ic"]
    assert result.loc[0, "rank_ic"] == pytest.approx(-0.5)


def test_compute_ic_merges_only_on_date_and_asset():
    factors = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
        assets=["A", "B", "A", "B"],
        factor_name="momentum_20d",
        values=[1.0, 2.0, 1.0, 2.0],
    )
    labels = _canonical(
        dates=["2024-01-02", "2024-01-03", "2024-01-03"],
        assets=["A", "A", "B"],
        factor_name="forward_return_1",
        values=[10.0, 20.0, 40.0],
    )

    result = compute_ic(factors, labels)

    assert len(result) == 2
    assert np.isnan(
        result.loc[result["date"] == pd.Timestamp("2024-01-02"), "ic"]
    ).all()
    assert result.loc[result["date"] == pd.Timestamp("2024-01-03"), "ic"].iloc[
        0
    ] == pytest.approx(1.0)


def test_compute_ic_drops_nan_pairs_within_date():
    factors = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-02"],
        assets=["A", "B", "C"],
        factor_name="momentum_20d",
        values=[1.0, np.nan, 3.0],
    )
    labels = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-02"],
        assets=["A", "B", "C"],
        factor_name="forward_return_1",
        values=[4.0, 5.0, 6.0],
    )

    result = compute_ic(factors, labels)
    assert result.loc[0, "ic"] == pytest.approx(1.0)


def test_compute_ic_returns_nan_for_insufficient_cross_section():
    factors = _canonical(
        dates=["2024-01-02"],
        assets=["A"],
        factor_name="momentum_20d",
        values=[1.0],
    )
    labels = _canonical(
        dates=["2024-01-02"],
        assets=["A"],
        factor_name="forward_return_1",
        values=[2.0],
    )

    result = compute_ic(factors, labels)
    assert np.isnan(result.loc[0, "ic"])


def test_compute_ic_returns_nan_for_degenerate_cross_section():
    factors = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-02"],
        assets=["A", "B", "C"],
        factor_name="momentum_20d",
        values=[1.0, 1.0, 1.0],
    )
    labels = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-02"],
        assets=["A", "B", "C"],
        factor_name="forward_return_1",
        values=[2.0, 3.0, 4.0],
    )

    result = compute_ic(factors, labels)
    assert np.isnan(result.loc[0, "ic"])


def test_compute_ic_rejects_multiple_factor_names():
    factors = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "asset": ["A", "B"],
            "factor": ["momentum_20d", "value_5d"],
            "value": [1.0, 2.0],
        }
    )
    labels = _canonical(
        dates=["2024-01-02", "2024-01-02"],
        assets=["A", "B"],
        factor_name="forward_return_1",
        values=[2.0, 3.0],
    )

    with pytest.raises(ValueError, match="exactly one factor name"):
        compute_ic(factors, labels)


def test_compute_ic_rejects_duplicate_rows():
    factors = _canonical(
        dates=["2024-01-02", "2024-01-02"],
        assets=["A", "A"],
        factor_name="momentum_20d",
        values=[1.0, 2.0],
    )
    labels = _canonical(
        dates=["2024-01-02", "2024-01-02"],
        assets=["A", "B"],
        factor_name="forward_return_1",
        values=[2.0, 3.0],
    )

    with pytest.raises(ValueError, match="duplicate"):
        compute_ic(factors, labels)


def test_compute_ic_empty_input_returns_empty_result():
    factors = pd.DataFrame(columns=["date", "asset", "factor", "value"])
    labels = _canonical(
        dates=["2024-01-02"],
        assets=["A"],
        factor_name="forward_return_1",
        values=[2.0],
    )

    result = compute_ic(factors, labels)
    assert result.empty
    assert list(result.columns) == ["date", "factor", "label", "ic"]


def test_compute_ic_raises_on_all_nan_factor_values():
    # validate_factor_output raises when the entire value column is NaN.
    # AlphaLabDataError is a ValueError subclass so ValueError is caught.
    factors = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-02"],
        assets=["A", "B", "C"],
        factor_name="momentum_20d",
        values=[float("nan"), float("nan"), float("nan")],
    )
    labels = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-02"],
        assets=["A", "B", "C"],
        factor_name="forward_return_1",
        values=[1.0, 2.0, 3.0],
    )
    with pytest.raises(ValueError):
        compute_ic(factors, labels)


def test_compute_ic_omits_dates_where_all_merged_pairs_are_nan():
    # When every factor value for a date is NaN, the inner merge produces
    # no clean pairs for that date.  The vectorised path omits the date
    # entirely (no NaN row) rather than emitting a NaN row as the old
    # per-group path did.  A valid date is included to confirm the output
    # frame is non-empty and correctly shaped.
    factors = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-03"]
            ),
            "asset": ["A", "B", "C", "D", "E"],
            "factor": ["momentum_20d"] * 5,
            "value": [float("nan"), float("nan"), 1.0, 2.0, 3.0],
        }
    )
    labels = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03", "2024-01-03"],
        assets=["A", "B", "C", "D", "E"],
        factor_name="forward_return_1",
        values=[1.0, 2.0, 3.0, 5.0, 7.0],
    )
    result = compute_ic(factors, labels)
    assert list(result.columns) == ["date", "factor", "label", "ic"]
    assert len(result) == 1
    assert pd.Timestamp("2024-01-02") not in result["date"].values
    assert result.loc[result["date"] == pd.Timestamp("2024-01-03"), "ic"].iloc[0] == pytest.approx(1.0)
