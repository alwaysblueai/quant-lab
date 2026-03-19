from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.quantile import long_short_return, quantile_returns


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


# ---------------------------------------------------------------------------
# quantile assignment
# ---------------------------------------------------------------------------


def test_quantile_assignment_covers_all_buckets():
    """With 5 assets and n_quantiles=5 every bucket should appear."""
    factors = _canonical(
        dates=["2024-01-02"] * 5,
        assets=["A", "B", "C", "D", "E"],
        factor_name="f",
        values=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    labels = _canonical(
        dates=["2024-01-02"] * 5,
        assets=["A", "B", "C", "D", "E"],
        factor_name="fwd",
        values=[0.0, 0.0, 0.0, 0.0, 0.0],
    )
    result = quantile_returns(factors, labels, n_quantiles=5)
    assert set(result["quantile"].tolist()) == {1, 2, 3, 4, 5}


def test_quantile_bottom_has_lowest_factor():
    """Lowest factor value must land in quantile 1."""
    factors = _canonical(
        dates=["2024-01-02"] * 5,
        assets=["A", "B", "C", "D", "E"],
        factor_name="f",
        values=[10.0, 20.0, 30.0, 40.0, 50.0],
    )
    labels = _canonical(
        dates=["2024-01-02"] * 5,
        assets=["A", "B", "C", "D", "E"],
        factor_name="fwd",
        values=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    result = quantile_returns(factors, labels, n_quantiles=5)
    bottom = float(result.loc[result["quantile"] == 1, "mean_return"].iloc[0])
    top = float(result.loc[result["quantile"] == 5, "mean_return"].iloc[0])
    assert bottom < top


# ---------------------------------------------------------------------------
# long-short correctness
# ---------------------------------------------------------------------------


def test_long_short_return_value():
    """long_short_return = top quantile mean_return - bottom quantile mean_return."""
    factors = _canonical(
        dates=["2024-01-02"] * 5,
        assets=["A", "B", "C", "D", "E"],
        factor_name="f",
        values=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    labels = _canonical(
        dates=["2024-01-02"] * 5,
        assets=["A", "B", "C", "D", "E"],
        factor_name="fwd",
        values=[0.01, 0.02, 0.03, 0.04, 0.10],
    )
    qr = quantile_returns(factors, labels, n_quantiles=5)
    ls = long_short_return(qr)

    assert list(ls.columns) == ["date", "factor", "long_short_return"]
    top = float(qr.loc[qr["quantile"] == 5, "mean_return"].iloc[0])
    bottom = float(qr.loc[qr["quantile"] == 1, "mean_return"].iloc[0])
    assert ls["long_short_return"].iloc[0] == pytest.approx(top - bottom)


def test_long_short_columns():
    """Output columns are exactly [date, factor, long_short_return]."""
    factors = _canonical(
        dates=["2024-01-02"] * 4,
        assets=["A", "B", "C", "D"],
        factor_name="f",
        values=[1.0, 2.0, 3.0, 4.0],
    )
    labels = _canonical(
        dates=["2024-01-02"] * 4,
        assets=["A", "B", "C", "D"],
        factor_name="fwd",
        values=[0.1, 0.2, 0.3, 0.4],
    )
    qr = quantile_returns(factors, labels, n_quantiles=4)
    ls = long_short_return(qr)
    assert list(ls.columns) == ["date", "factor", "long_short_return"]


# ---------------------------------------------------------------------------
# monotonicity sanity check
# ---------------------------------------------------------------------------


def test_monotonicity_perfectly_ordered_factor():
    """When factor rank == return rank exactly, mean_return must be strictly
    increasing across quantiles."""
    n = 20
    dates = ["2024-01-02"] * n
    assets = [f"A{i}" for i in range(n)]
    values = list(range(1, n + 1))
    returns = list(range(1, n + 1))

    factors = _canonical(dates=dates, assets=assets, factor_name="f", values=values)
    labels = _canonical(dates=dates, assets=assets, factor_name="fwd", values=returns)

    qr = quantile_returns(factors, labels, n_quantiles=5)
    qr_sorted = qr.sort_values("quantile").reset_index(drop=True)
    mean_returns = qr_sorted["mean_return"].tolist()
    assert mean_returns == sorted(mean_returns), "mean_return must increase with quantile"


# ---------------------------------------------------------------------------
# NaN handling
# ---------------------------------------------------------------------------


def test_nan_factor_row_excluded():
    """Asset with NaN factor value must be excluded from all quantile buckets."""
    factors = _canonical(
        dates=["2024-01-02"] * 4,
        assets=["A", "B", "C", "D"],
        factor_name="f",
        values=[1.0, np.nan, 3.0, 4.0],
    )
    labels = _canonical(
        dates=["2024-01-02"] * 4,
        assets=["A", "B", "C", "D"],
        factor_name="fwd",
        values=[0.1, 0.2, 0.3, 0.4],
    )
    qr = quantile_returns(factors, labels, n_quantiles=4)
    assert qr["mean_return"].notna().all()
    # Three valid assets → at most 3 quantile rows
    assert len(qr) <= 3


def test_nan_label_row_excluded():
    """Asset with NaN label must not contaminate any quantile's mean_return."""
    factors = _canonical(
        dates=["2024-01-02"] * 4,
        assets=["A", "B", "C", "D"],
        factor_name="f",
        values=[1.0, 2.0, 3.0, 4.0],
    )
    labels = _canonical(
        dates=["2024-01-02"] * 4,
        assets=["A", "B", "C", "D"],
        factor_name="fwd",
        values=[0.1, np.nan, 0.3, 0.4],
    )
    qr = quantile_returns(factors, labels, n_quantiles=4)
    assert qr["mean_return"].notna().all()


# ---------------------------------------------------------------------------
# small cross-section
# ---------------------------------------------------------------------------


def test_small_cross_section_fewer_than_n_quantiles():
    """Cross-section smaller than n_quantiles should still produce valid output."""
    factors = _canonical(
        dates=["2024-01-02"] * 2,
        assets=["A", "B"],
        factor_name="f",
        values=[1.0, 2.0],
    )
    labels = _canonical(
        dates=["2024-01-02"] * 2,
        assets=["A", "B"],
        factor_name="fwd",
        values=[0.01, 0.05],
    )
    qr = quantile_returns(factors, labels, n_quantiles=5)
    assert len(qr) > 0
    assert qr["mean_return"].notna().all()


def test_single_asset_per_date_excluded():
    """Dates with only one non-NaN asset cannot form a valid cross-section."""
    factors = _canonical(
        dates=["2024-01-02"],
        assets=["A"],
        factor_name="f",
        values=[1.0],
    )
    labels = _canonical(
        dates=["2024-01-02"],
        assets=["A"],
        factor_name="fwd",
        values=[0.05],
    )
    qr = quantile_returns(factors, labels, n_quantiles=5)
    assert qr.empty


# ---------------------------------------------------------------------------
# empty input
# ---------------------------------------------------------------------------


def test_quantile_returns_empty_factors_returns_empty():
    factors = pd.DataFrame(columns=["date", "asset", "factor", "value"])
    labels = _canonical(
        dates=["2024-01-02"] * 3,
        assets=["A", "B", "C"],
        factor_name="fwd",
        values=[0.1, 0.2, 0.3],
    )
    result = quantile_returns(factors, labels)
    assert result.empty
    assert list(result.columns) == ["date", "factor", "quantile", "mean_return"]


def test_long_short_empty_returns_empty():
    qr = pd.DataFrame(columns=["date", "factor", "quantile", "mean_return"])
    result = long_short_return(qr)
    assert result.empty
    assert list(result.columns) == ["date", "factor", "long_short_return"]


def test_long_short_single_quantile_is_nan():
    """If only one quantile bucket exists on a date, long-short is NaN."""
    qr = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"]),
            "factor": ["f"],
            "quantile": [3],
            "mean_return": [0.05],
        }
    )
    result = long_short_return(qr)
    assert np.isnan(result["long_short_return"].iloc[0])


# ---------------------------------------------------------------------------
# merge key discipline
# ---------------------------------------------------------------------------


def test_merges_only_on_date_and_asset():
    """Factor and label for different dates must not be mixed."""
    factors = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
        assets=["A", "B", "A", "B"],
        factor_name="f",
        values=[1.0, 4.0, 2.0, 3.0],
    )
    labels = _canonical(
        dates=["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
        assets=["A", "B", "A", "B"],
        factor_name="fwd",
        values=[0.01, 0.04, 0.02, 0.03],
    )
    qr = quantile_returns(factors, labels, n_quantiles=2)
    assert pd.Timestamp("2024-01-02") in qr["date"].values
    assert pd.Timestamp("2024-01-03") in qr["date"].values


# ---------------------------------------------------------------------------
# Regression: tie handling in quantile assignment (critical fix)
# ---------------------------------------------------------------------------


def test_tied_factor_values_bottom_bucket_always_exists():
    """With ties at the bottom, the bottom bucket (quantile=1) must always be
    occupied and must contain the tied assets."""
    factors = _canonical(
        dates=["2024-01-02"] * 3,
        assets=["A", "B", "C"],
        factor_name="f",
        values=[1.0, 1.0, 3.0],
    )
    labels = _canonical(
        dates=["2024-01-02"] * 3,
        assets=["A", "B", "C"],
        factor_name="fwd",
        values=[0.01, 0.02, 0.09],
    )
    qr = quantile_returns(factors, labels, n_quantiles=3)
    occupied = set(qr["quantile"].tolist())
    assert 1 in occupied, "bottom bucket must be occupied even with ties"
    assert max(occupied) in occupied, "top bucket must be occupied"


def test_top_tied_values_land_in_top_bucket():
    """With ties at the top, the highest-numbered bucket must contain the
    highest-valued assets.  Previously, rank('min') + ceil mapped top-tied
    assets to a middle bucket number, so long_short_return used an incorrect
    'long' leg.  The fix uses rank('dense') + linear map which pins the
    highest distinct value to bucket effective_q.
    """
    # Three assets tie at the top (value 3.0); n_quantiles=5 requested.
    factors = _canonical(
        dates=["2024-01-02"] * 5,
        assets=["A", "B", "C", "D", "E"],
        factor_name="f",
        values=[1.0, 2.0, 3.0, 3.0, 3.0],
    )
    labels = _canonical(
        dates=["2024-01-02"] * 5,
        assets=["A", "B", "C", "D", "E"],
        factor_name="fwd",
        values=[0.01, 0.02, 0.09, 0.10, 0.08],
    )
    qr = quantile_returns(factors, labels, n_quantiles=5)
    top_bucket = int(qr["quantile"].max())
    # With 3 distinct values and effective_q=5, the linear map pins rank 3
    # (highest) to bucket 5.
    assert top_bucket == 5, (
        f"top-tied assets should land in bucket 5 (effective_q), got {top_bucket}"
    )
    # Bottom bucket must also be correctly populated.
    assert 1 in set(qr["quantile"].tolist()), "bottom bucket must be occupied"
    # long_short must use the true top-vs-bottom.
    ls = long_short_return(qr)
    assert np.isfinite(ls["long_short_return"].iloc[0])
    assert ls["long_short_return"].iloc[0] > 0


def test_intermediate_gap_bucket_pattern():
    """When n_distinct < n_quantiles, intermediate bucket numbers are empty.

    For [1, 2, 3, 3, 3] with n_quantiles=5 and n_distinct=3:
      dense rank 1 → bucket 1  (linear map: (0/2)*4+1 = 1)
      dense rank 2 → bucket 3  (linear map: (1/2)*4+1 = 3)
      dense rank 3 → bucket 5  (linear map: (2/2)*4+1 = 5)
    Occupied buckets are exactly {1, 3, 5}; buckets 2 and 4 are absent.
    This locks in the semantics of the dense-rank + linear-map policy.
    """
    factors = _canonical(
        dates=["2024-01-02"] * 5,
        assets=["A", "B", "C", "D", "E"],
        factor_name="f",
        values=[1.0, 2.0, 3.0, 3.0, 3.0],
    )
    labels = _canonical(
        dates=["2024-01-02"] * 5,
        assets=["A", "B", "C", "D", "E"],
        factor_name="fwd",
        values=[0.01, 0.02, 0.09, 0.10, 0.08],
    )
    qr = quantile_returns(factors, labels, n_quantiles=5)
    occupied = set(qr["quantile"].tolist())
    assert occupied == {1, 3, 5}, (
        f"expected buckets {{1, 3, 5}} for 3 distinct values with n_quantiles=5, "
        f"got {occupied}"
    )


def test_midpoint_rounding_uses_half_up():
    """Locks in the round-half-up convention for the exact .5 midpoint case.

    With 3 distinct values and n_quantiles=4 the middle dense rank maps to
      q = (1/2) * (4-1) + 1 = 2.5
    Round-half-up gives bucket 3 (int(2.5 + 0.5) = 3).
    Banker's rounding (Python's built-in round()) would give 2 instead.
    Occupied buckets must be exactly {1, 3, 4}.
    """
    factors = _canonical(
        dates=["2024-01-02"] * 4,
        assets=["A", "B", "C", "D"],
        factor_name="f",
        values=[1.0, 2.0, 2.0, 3.0],  # 3 distinct values, middle ties at 2.0
    )
    labels = _canonical(
        dates=["2024-01-02"] * 4,
        assets=["A", "B", "C", "D"],
        factor_name="fwd",
        values=[0.01, 0.02, 0.03, 0.04],
    )
    qr = quantile_returns(factors, labels, n_quantiles=4)
    occupied = set(qr["quantile"].tolist())
    assert occupied == {1, 3, 4}, (
        f"expected buckets {{1, 3, 4}} (half-up rounding), got {occupied}"
    )


def test_constant_factor_long_short_is_nan():
    """When all assets share the same factor value the cross-section is
    uninformative.  rank('dense') gives all assets dense rank 1 (one distinct
    value), the special-case path assigns every asset to bucket 1, so
    q_min == q_max and long_short_return must be NaN.
    A finite spread here would be manufactured from row order, not factor signal.
    """
    factors = _canonical(
        dates=["2024-01-02"] * 4,
        assets=["A", "B", "C", "D"],
        factor_name="f",
        values=[5.0, 5.0, 5.0, 5.0],
    )
    labels = _canonical(
        dates=["2024-01-02"] * 4,
        assets=["A", "B", "C", "D"],
        factor_name="fwd",
        values=[0.1, 0.2, 0.3, 0.4],
    )
    qr = quantile_returns(factors, labels, n_quantiles=4)
    ls = long_short_return(qr)
    assert np.isnan(ls["long_short_return"].iloc[0]), (
        "constant factor must produce NaN long-short, not an artificial spread"
    )


# ---------------------------------------------------------------------------
# Regression: row-order invariance (critical fix)
# ---------------------------------------------------------------------------


def test_quantile_stable_under_row_reordering():
    """Shuffling input rows must not change quantile assignments or mean_return.

    Tied assets must always map to the same bucket regardless of which row
    they appear on.  This verifies rank('dense') + linear map is row-order invariant.
    """
    # Ties at bottom (A, B share 1.0) and top (D, E share 3.0).
    factors_orig = _canonical(
        dates=["2024-01-02"] * 5,
        assets=["A", "B", "C", "D", "E"],
        factor_name="f",
        values=[1.0, 1.0, 2.0, 3.0, 3.0],
    )
    labels_orig = _canonical(
        dates=["2024-01-02"] * 5,
        assets=["A", "B", "C", "D", "E"],
        factor_name="fwd",
        values=[0.01, 0.02, 0.03, 0.04, 0.05],
    )
    # Shuffle rows consistently (same asset stays with same label via merge key)
    order = [4, 0, 2, 1, 3]
    factors_shuffled = factors_orig.iloc[order].reset_index(drop=True)
    labels_shuffled = labels_orig.iloc[order].reset_index(drop=True)

    qr_orig = quantile_returns(factors_orig, labels_orig, n_quantiles=5)
    qr_shuf = quantile_returns(factors_shuffled, labels_shuffled, n_quantiles=5)

    # Sort by quantile so comparison is position-independent
    qr_orig_s = qr_orig.sort_values("quantile").reset_index(drop=True)
    qr_shuf_s = qr_shuf.sort_values("quantile").reset_index(drop=True)

    assert list(qr_orig_s["quantile"]) == list(qr_shuf_s["quantile"]), (
        "quantile bucket labels changed after row reordering"
    )
    assert qr_orig_s["mean_return"].tolist() == pytest.approx(
        qr_shuf_s["mean_return"].tolist()
    ), "mean_return changed after row reordering"


# ---------------------------------------------------------------------------
# Regression: stacked labels (critical fix — must error, not silently corrupt)
# ---------------------------------------------------------------------------


def test_stacked_labels_multiple_factor_names_rejected():
    """labels with more than one factor name must raise ValueError.

    Without this guard, the merge becomes one-to-many and contaminates
    mean_return silently.
    """
    factors = _canonical(
        dates=["2024-01-02"] * 3,
        assets=["A", "B", "C"],
        factor_name="f",
        values=[1.0, 2.0, 3.0],
    )
    # Two label variants for the same (date, asset) — stacked horizons
    labels = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02"] * 6),
            "asset": ["A", "B", "C", "A", "B", "C"],
            "factor": ["fwd_1", "fwd_1", "fwd_1", "fwd_5", "fwd_5", "fwd_5"],
            "value": [0.01, 0.02, 0.03, 0.05, 0.06, 0.07],
        }
    )
    with pytest.raises(ValueError, match="exactly one factor name"):
        quantile_returns(factors, labels, n_quantiles=3)


# ---------------------------------------------------------------------------
# Regression: n_quantiles validation
# ---------------------------------------------------------------------------


def test_invalid_n_quantiles_raises():
    factors = _canonical(
        dates=["2024-01-02"] * 3,
        assets=["A", "B", "C"],
        factor_name="f",
        values=[1.0, 2.0, 3.0],
    )
    labels = _canonical(
        dates=["2024-01-02"] * 3,
        assets=["A", "B", "C"],
        factor_name="fwd",
        values=[0.1, 0.2, 0.3],
    )
    with pytest.raises(ValueError, match="n_quantiles"):
        quantile_returns(factors, labels, n_quantiles=1)
    with pytest.raises(ValueError, match="n_quantiles"):
        quantile_returns(factors, labels, n_quantiles=0)
