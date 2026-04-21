from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.evaluation import (
    compute_ic,
    compute_mean_rank_ic_permutation_null,
    compute_mutual_information,
    compute_rank_ic,
)


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


def test_compute_mean_rank_ic_permutation_null_is_deterministic_for_seed() -> None:
    factors = _canonical(
        dates=[
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
            "2024-01-03",
            "2024-01-03",
            "2024-01-03",
            "2024-01-03",
        ],
        assets=["A", "B", "C", "D", "A", "B", "C", "D"],
        factor_name="f",
        values=[1.0, 2.0, 4.0, 3.0, 2.0, 1.0, 3.0, 4.0],
    )
    labels = _canonical(
        dates=[
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
            "2024-01-03",
            "2024-01-03",
            "2024-01-03",
            "2024-01-03",
        ],
        assets=["A", "B", "C", "D", "A", "B", "C", "D"],
        factor_name="y",
        values=[10.0, 11.0, 14.0, 12.0, 7.0, 6.0, 8.0, 9.0],
    )

    observed_1, null_1 = compute_mean_rank_ic_permutation_null(
        factors,
        labels,
        n_permutations=25,
        seed=123,
    )
    observed_2, null_2 = compute_mean_rank_ic_permutation_null(
        factors,
        labels,
        n_permutations=25,
        seed=123,
    )

    assert np.isfinite(observed_1)
    assert observed_1 == pytest.approx(observed_2)
    assert len(null_1) == 25
    assert np.all(np.isfinite(null_1))
    np.testing.assert_allclose(null_1, null_2, rtol=0.0, atol=0.0)


def test_compute_mean_rank_ic_permutation_null_is_invariant_to_batch_size() -> None:
    factors = _canonical(
        dates=[
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
            "2024-01-03",
            "2024-01-03",
            "2024-01-03",
            "2024-01-03",
        ],
        assets=["A", "B", "C", "D", "A", "B", "C", "D"],
        factor_name="f",
        values=[1.0, 2.0, 4.0, 3.0, 2.0, 1.0, 3.0, 4.0],
    )
    labels = _canonical(
        dates=[
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
            "2024-01-02",
            "2024-01-03",
            "2024-01-03",
            "2024-01-03",
            "2024-01-03",
        ],
        assets=["A", "B", "C", "D", "A", "B", "C", "D"],
        factor_name="y",
        values=[10.0, 11.0, 14.0, 12.0, 7.0, 6.0, 8.0, 9.0],
    )

    observed_1, null_1 = compute_mean_rank_ic_permutation_null(
        factors,
        labels,
        n_permutations=25,
        seed=123,
        batch_size=3,
    )
    observed_2, null_2 = compute_mean_rank_ic_permutation_null(
        factors,
        labels,
        n_permutations=25,
        seed=123,
        batch_size=17,
    )

    assert observed_1 == pytest.approx(observed_2)
    np.testing.assert_allclose(null_1, null_2, rtol=0.0, atol=0.0)


def test_compute_mean_rank_ic_permutation_null_returns_empty_when_cross_section_too_small() -> None:
    factors = _canonical(
        dates=["2024-01-02", "2024-01-02"],
        assets=["A", "B"],
        factor_name="f",
        values=[1.0, 2.0],
    )
    labels = _canonical(
        dates=["2024-01-02", "2024-01-02"],
        assets=["A", "B"],
        factor_name="y",
        values=[10.0, 11.0],
    )

    observed, null_samples = compute_mean_rank_ic_permutation_null(
        factors,
        labels,
        n_permutations=10,
        seed=1,
        min_assets_per_date=3,
    )

    assert np.isnan(observed)
    assert null_samples.size == 0


def test_compute_mean_rank_ic_permutation_null_rejects_non_positive_batch_size() -> None:
    factors = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-02"],
        assets=["A", "B", "C"],
        factor_name="f",
        values=[1.0, 2.0, 3.0],
    )
    labels = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-02"],
        assets=["A", "B", "C"],
        factor_name="y",
        values=[10.0, 11.0, 12.0],
    )

    with pytest.raises(ValueError, match="batch_size must be > 0"):
        compute_mean_rank_ic_permutation_null(
            factors,
            labels,
            n_permutations=10,
            seed=1,
            batch_size=0,
        )


def test_compute_mutual_information_detects_nonlinear_dependence() -> None:
    factors = _canonical(
        dates=["2024-01-02"] * 9,
        assets=[f"A{i}" for i in range(9)],
        factor_name="f",
        values=[-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
    )
    labels = _canonical(
        dates=["2024-01-02"] * 9,
        assets=[f"A{i}" for i in range(9)],
        factor_name="y",
        values=[16.0, 9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0, 16.0],
    )
    result = compute_mutual_information(factors, labels)
    assert list(result.columns) == ["date", "factor", "label", "mutual_information"]
    mi = float(result.loc[0, "mutual_information"])
    assert mi > 0.0


def test_compute_mutual_information_returns_nan_for_degenerate_cross_section() -> None:
    factors = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-02"],
        assets=["A", "B", "C"],
        factor_name="f",
        values=[1.0, 1.0, 1.0],
    )
    labels = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-02"],
        assets=["A", "B", "C"],
        factor_name="y",
        values=[0.5, 0.7, 0.9],
    )
    result = compute_mutual_information(factors, labels)
    assert np.isnan(result.loc[0, "mutual_information"])


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
    assert np.isnan(result.loc[result["date"] == pd.Timestamp("2024-01-02"), "ic"]).all()
    assert result.loc[result["date"] == pd.Timestamp("2024-01-03"), "ic"].iloc[0] == pytest.approx(
        1.0
    )


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


def test_compute_ic_keeps_dates_where_all_merged_pairs_are_nan_as_nan_rows():
    # When every factor value for a date is NaN, the date must still appear
    # in IC output as NaN so valid-ratio denominators remain auditable.
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
    assert len(result) == 2
    assert pd.Timestamp("2024-01-02") in result["date"].values
    assert np.isnan(result.loc[result["date"] == pd.Timestamp("2024-01-02"), "ic"].iloc[0])
    assert result.loc[result["date"] == pd.Timestamp("2024-01-03"), "ic"].iloc[0] == pytest.approx(
        1.0
    )


def test_merged_pairs_path_matches_default_for_ic_rankic_and_mi():
    factors = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
        assets=["A", "B", "A", "B"],
        factor_name="f",
        values=[1.0, 2.0, 3.0, 4.0],
    )
    labels = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
        assets=["A", "B", "A", "B"],
        factor_name="y",
        values=[2.0, 1.0, 4.0, 3.0],
    )
    merged_pairs = (
        factors[["date", "asset", "value"]]
        .rename(columns={"value": "value_factor"})
        .merge(
            labels[["date", "asset", "value"]].rename(columns={"value": "value_label"}),
            on=["date", "asset"],
            how="inner",
            validate="one_to_one",
        )
    )

    ic_default = compute_ic(factors, labels)
    ic_merged = compute_ic(factors, labels, merged_pairs=merged_pairs)
    pd.testing.assert_frame_equal(ic_default, ic_merged)

    rank_default = compute_rank_ic(factors, labels)
    rank_merged = compute_rank_ic(factors, labels, merged_pairs=merged_pairs)
    pd.testing.assert_frame_equal(rank_default, rank_merged)

    mi_default = compute_mutual_information(factors, labels)
    mi_merged = compute_mutual_information(factors, labels, merged_pairs=merged_pairs)
    pd.testing.assert_frame_equal(mi_default, mi_merged)
