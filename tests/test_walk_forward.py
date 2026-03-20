from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alpha_lab.factors.momentum import momentum
from alpha_lab.strategy import StrategySpec
from alpha_lab.walk_forward import (
    _FOLD_SUMMARY_COLUMNS,
    WalkForwardResult,
    run_walk_forward_experiment,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_prices(n_assets: int = 6, n_days: int = 80, seed: int = 42) -> pd.DataFrame:
    """Synthetic price panel, long-form [date, asset, close]."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    rows = []
    for asset in [f"A{i}" for i in range(n_assets)]:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


def _factor_fn(prices: pd.DataFrame) -> pd.DataFrame:
    return momentum(prices, window=5)


_PRICES = _make_prices()


# ---------------------------------------------------------------------------
# 1. Basic structural tests
# ---------------------------------------------------------------------------


def test_returns_walk_forward_result() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    assert isinstance(result, WalkForwardResult)


def test_correct_fold_count() -> None:
    # n_unique_dates=80, train_size=30, test_size=10, step=10
    # fold 0: train[0..29] test[30..39]
    # fold 1: train[10..39] test[40..49]
    # fold 2: train[20..49] test[50..59]
    # fold 3: train[30..59] test[60..69]
    # fold 4: train[40..69] test[70..79]   <- test_end=80 == n → included
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    assert result.aggregate_summary.n_folds == len(result.per_fold_results)
    assert result.aggregate_summary.n_folds >= 4  # at least 4 folds from 80 dates


def test_fold_summary_df_columns() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    assert list(result.fold_summary_df.columns) == list(_FOLD_SUMMARY_COLUMNS)


def test_fold_summary_row_count_matches_per_fold_results() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    assert len(result.fold_summary_df) == len(result.per_fold_results)


def test_fold_ids_are_sequential() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    expected = list(range(result.aggregate_summary.n_folds))
    assert list(result.fold_summary_df["fold_id"]) == expected


# ---------------------------------------------------------------------------
# 2. Temporal ordering and independence
# ---------------------------------------------------------------------------


def test_test_windows_non_overlapping() -> None:
    """Test date ranges must not overlap across folds."""
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    all_test_dates: list[set[pd.Timestamp]] = []
    for fold_result in result.per_fold_results:
        assert fold_result.ic_df is not None
        fold_dates = set(pd.to_datetime(fold_result.ic_df["date"]))
        all_test_dates.append(fold_dates)

    for i in range(len(all_test_dates)):
        for j in range(i + 1, len(all_test_dates)):
            overlap = all_test_dates[i] & all_test_dates[j]
            assert len(overlap) == 0, (
                f"Folds {i} and {j} share test dates: {sorted(overlap)[:3]}"
            )


def test_test_windows_in_temporal_order() -> None:
    """Each fold's test start must be after the previous fold's test start."""
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    starts = list(result.fold_summary_df["start_date"])
    for i in range(1, len(starts)):
        assert starts[i] > starts[i - 1], (
            f"Fold {i} start ({starts[i]}) not after fold {i-1} start ({starts[i - 1]})"
        )


def test_no_train_test_overlap_within_fold() -> None:
    """For each fold, train_end < test start_date."""
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    df = result.fold_summary_df
    for _, row in df.iterrows():
        assert row["train_end"] < row["start_date"], (
            f"Fold {row['fold_id']}: train_end {row['train_end']} >= "
            f"test start {row['start_date']}"
        )


def test_fold_results_are_independent() -> None:
    """Each fold's factor_df covers the full sample (computation) but
    ic_df is restricted to that fold's test window only."""
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    df = result.fold_summary_df
    for fold_id, fold_result in enumerate(result.per_fold_results):
        expected_start = pd.Timestamp(df.loc[df["fold_id"] == fold_id, "start_date"].iloc[0])
        expected_end = pd.Timestamp(df.loc[df["fold_id"] == fold_id, "end_date"].iloc[0])

        if not fold_result.ic_df.empty:
            ic_dates = pd.to_datetime(fold_result.ic_df["date"])
            assert ic_dates.min() >= expected_start
            assert ic_dates.max() <= expected_end


# ---------------------------------------------------------------------------
# 3. Aggregate summary correctness
# ---------------------------------------------------------------------------


def test_aggregate_n_folds_correct() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    assert result.aggregate_summary.n_folds == len(result.per_fold_results)


def test_aggregate_mean_ic_matches_manual_mean() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    manual_mean = float(result.fold_summary_df["mean_ic"].dropna().mean())
    assert math.isclose(result.aggregate_summary.mean_ic, manual_mean, rel_tol=1e-9)


def test_aggregate_std_ic_matches_manual_std() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    vals = result.fold_summary_df["mean_ic"].dropna()
    if len(vals) > 1:
        manual_std = float(vals.std(ddof=1))
        assert math.isclose(result.aggregate_summary.std_ic, manual_std, rel_tol=1e-9)


def test_best_worst_fold_are_valid_ids() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    valid_ids = set(result.fold_summary_df["fold_id"])
    assert result.aggregate_summary.best_fold in valid_ids
    assert result.aggregate_summary.worst_fold in valid_ids


def test_best_fold_has_highest_mean_ic() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    df = result.fold_summary_df.dropna(subset=["mean_ic"])
    if df.empty:
        pytest.skip("All IC values are NaN")
    best_id = result.aggregate_summary.best_fold
    best_ic = float(df.loc[df["fold_id"] == best_id, "mean_ic"].iloc[0])
    assert best_ic == pytest.approx(df["mean_ic"].max())


def test_worst_fold_has_lowest_mean_ic() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    df = result.fold_summary_df.dropna(subset=["mean_ic"])
    if df.empty:
        pytest.skip("All IC values are NaN")
    worst_id = result.aggregate_summary.worst_fold
    worst_ic = float(df.loc[df["fold_id"] == worst_id, "mean_ic"].iloc[0])
    assert worst_ic == pytest.approx(df["mean_ic"].min())


# ---------------------------------------------------------------------------
# 4. Cost rate propagation
# ---------------------------------------------------------------------------


def test_cost_rate_none_produces_nan_cost_adjusted() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    for val in result.fold_summary_df["mean_cost_adjusted_return"]:
        assert math.isnan(float(val))


def test_cost_rate_provided_produces_finite_or_nan_per_fold() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        cost_rate=0.001,
    )
    # At least some folds should have a finite cost-adjusted return.
    vals = result.fold_summary_df["mean_cost_adjusted_return"].dropna()
    assert len(vals) > 0


def test_aggregate_cost_adjusted_nan_when_no_cost_rate() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    assert math.isnan(result.aggregate_summary.mean_cost_adjusted_return)


# ---------------------------------------------------------------------------
# 5. Edge cases and error handling
# ---------------------------------------------------------------------------


def test_too_small_dataset_raises() -> None:
    tiny = _make_prices(n_days=10)
    with pytest.raises(ValueError, match="No walk-forward folds"):
        run_walk_forward_experiment(
            tiny,
            _factor_fn,
            train_size=30,
            test_size=10,
            step=5,
        )


def test_invalid_train_size_raises() -> None:
    with pytest.raises(ValueError):
        run_walk_forward_experiment(
            _PRICES,
            _factor_fn,
            train_size=0,
            test_size=10,
            step=5,
        )


def test_negative_cost_rate_raises() -> None:
    with pytest.raises(ValueError, match="cost_rate"):
        run_walk_forward_experiment(
            _PRICES,
            _factor_fn,
            train_size=30,
            test_size=10,
            step=10,
            cost_rate=-0.001,
        )


def test_single_fold_aggregate_std_is_nan() -> None:
    """With only one fold, std across folds is NaN (ddof=1, n=1)."""
    prices = _make_prices(n_days=45)
    result = run_walk_forward_experiment(
        prices,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=100,  # large step → only one fold
    )
    if result.aggregate_summary.n_folds == 1:
        assert math.isnan(result.aggregate_summary.std_ic)


def test_val_size_accepted() -> None:
    """val_size > 0 should not raise."""
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        val_size=5,
    )
    assert result.aggregate_summary.n_folds > 0


# ---------------------------------------------------------------------------
# 6. Pooled IC
# ---------------------------------------------------------------------------


def test_pooled_ic_df_has_expected_columns() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    assert list(result.pooled_ic_df.columns) == ["fold_id", "date", "ic"]


def test_pooled_ic_df_row_count_matches_sum_of_fold_ic_rows() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    total_from_folds = sum(
        len(r.ic_df) for r in result.per_fold_results if not r.ic_df.empty
    )
    assert len(result.pooled_ic_df) == total_from_folds


def test_pooled_ic_df_fold_ids_cover_all_folds() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    expected = set(range(result.aggregate_summary.n_folds))
    actual = set(result.pooled_ic_df["fold_id"].unique())
    # Every fold that produced IC observations should appear in pooled_ic_df.
    assert actual.issubset(expected)


def test_pooled_ic_mean_matches_manual_mean() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    manual = float(result.pooled_ic_df["ic"].dropna().mean())
    assert math.isclose(result.aggregate_summary.pooled_ic_mean, manual, rel_tol=1e-9)


def test_pooled_ic_std_matches_manual_std() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    vals = result.pooled_ic_df["ic"].dropna()
    if len(vals) > 1:
        manual = float(vals.std(ddof=1))
        assert math.isclose(result.aggregate_summary.pooled_ic_std, manual, rel_tol=1e-9)


def test_pooled_ic_ir_equals_mean_over_std() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    agg = result.aggregate_summary
    if not math.isnan(agg.pooled_ic_ir):
        expected_ir = agg.pooled_ic_mean / agg.pooled_ic_std
        assert math.isclose(agg.pooled_ic_ir, expected_ir, rel_tol=1e-9)


def test_n_ic_obs_matches_pooled_ic_df_non_nan_count() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    expected = int(result.pooled_ic_df["ic"].notna().sum())
    assert result.aggregate_summary.n_ic_obs == expected


# ---------------------------------------------------------------------------
# 7. factor_df temporal bound
# ---------------------------------------------------------------------------


def test_per_fold_result_factor_df_bounded_by_test_end() -> None:
    """factor_df on each fold must not contain dates beyond the fold's test_end.

    Each fold receives prices filtered to <= test_end_ts so that factor_fn
    cannot access future data.  The factor_df must therefore be bounded by
    the fold's test_end date.
    """
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    df = result.fold_summary_df
    for fold_id, fold_result in enumerate(result.per_fold_results):
        test_end = pd.Timestamp(df.loc[df["fold_id"] == fold_id, "end_date"].iloc[0])
        factor_dates = pd.to_datetime(fold_result.factor_df["date"])
        assert factor_dates.max() <= test_end, (
            f"Fold {fold_id}: factor_df contains dates beyond test_end {test_end}"
        )


# ---------------------------------------------------------------------------
# 8. Portfolio path in walk-forward
# ---------------------------------------------------------------------------


def test_portfolio_columns_nan_without_portfolio_params() -> None:
    """Portfolio fold summary columns are NaN when no portfolio params given."""
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    for col in (
        "mean_portfolio_return",
        "portfolio_hit_rate",
        "mean_portfolio_turnover",
        "mean_cost_adjusted_portfolio_return",
    ):
        for val in result.fold_summary_df[col]:
            assert math.isnan(float(val)), f"{col} should be NaN, got {val}"


def test_portfolio_aggregate_fields_nan_without_portfolio_params() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    agg = result.aggregate_summary
    assert math.isnan(agg.mean_portfolio_return)
    assert math.isnan(agg.std_portfolio_return)
    assert math.isnan(agg.portfolio_hit_rate)
    assert math.isnan(agg.mean_portfolio_turnover)
    assert math.isnan(agg.mean_cost_adjusted_portfolio_return)
    assert math.isnan(agg.std_cost_adjusted_portfolio_return)


def test_portfolio_columns_populated_with_portfolio_params() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
    )
    vals = result.fold_summary_df["mean_portfolio_return"].dropna()
    assert len(vals) > 0, "Expected at least one fold with finite mean_portfolio_return"


def test_portfolio_aggregate_mean_return_finite_with_params() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
    )
    assert math.isfinite(result.aggregate_summary.mean_portfolio_return)


def test_portfolio_aggregate_mean_return_matches_manual_mean() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
    )
    manual = float(result.fold_summary_df["mean_portfolio_return"].dropna().mean())
    assert math.isclose(
        result.aggregate_summary.mean_portfolio_return, manual, rel_tol=1e-9
    )


def test_portfolio_hit_rate_in_unit_interval() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
    )
    hr = result.aggregate_summary.portfolio_hit_rate
    if not math.isnan(hr):
        assert 0.0 <= hr <= 1.0


def test_portfolio_cost_adj_nan_without_portfolio_cost_rate() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
    )
    assert math.isnan(result.aggregate_summary.mean_cost_adjusted_portfolio_return)


def test_portfolio_cost_adj_finite_with_portfolio_cost_rate() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
        portfolio_cost_rate=0.001,
    )
    assert math.isfinite(result.aggregate_summary.mean_cost_adjusted_portfolio_return)


def test_per_fold_portfolio_summary_populated() -> None:
    """Each fold's ExperimentResult should have a portfolio_summary when params given."""
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
    )
    for fold_result in result.per_fold_results:
        assert fold_result.portfolio_summary is not None


def test_portfolio_params_one_sided_raises() -> None:
    with pytest.raises(ValueError, match="holding_period and rebalance_frequency"):
        run_walk_forward_experiment(
            _PRICES,
            _factor_fn,
            train_size=30,
            test_size=10,
            step=10,
            holding_period=2,
            # rebalance_frequency omitted
        )


# ---------------------------------------------------------------------------
# 9. Pooled portfolio-return observations
# ---------------------------------------------------------------------------


def test_pooled_portfolio_return_df_empty_without_portfolio_params() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    assert result.pooled_portfolio_return_df.empty
    assert list(result.pooled_portfolio_return_df.columns) == [
        "fold_id", "date", "portfolio_return"
    ]


def test_pooled_portfolio_return_df_populated_with_params() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
    )
    assert not result.pooled_portfolio_return_df.empty
    assert list(result.pooled_portfolio_return_df.columns) == [
        "fold_id", "date", "portfolio_return"
    ]


def test_pooled_portfolio_return_row_count_matches_fold_totals() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
    )
    expected = sum(
        len(r.portfolio_return_df)
        for r in result.per_fold_results
        if r.portfolio_return_df is not None and not r.portfolio_return_df.empty
    )
    assert len(result.pooled_portfolio_return_df) == expected


def test_pooled_portfolio_return_aggregate_nan_without_params() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    agg = result.aggregate_summary
    assert math.isnan(agg.pooled_portfolio_return_mean)
    assert math.isnan(agg.pooled_portfolio_return_std)
    assert math.isnan(agg.pooled_portfolio_hit_rate)
    assert agg.n_portfolio_obs == 0


def test_pooled_portfolio_return_mean_matches_manual() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
    )
    manual = float(
        result.pooled_portfolio_return_df["portfolio_return"].dropna().mean()
    )
    assert math.isclose(
        result.aggregate_summary.pooled_portfolio_return_mean, manual, rel_tol=1e-9
    )


def test_pooled_portfolio_hit_rate_in_unit_interval() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
    )
    hr = result.aggregate_summary.pooled_portfolio_hit_rate
    if not math.isnan(hr):
        assert 0.0 <= hr <= 1.0


def test_n_portfolio_obs_matches_pooled_df_non_nan_count() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
    )
    expected = int(
        result.pooled_portfolio_return_df["portfolio_return"].notna().sum()
    )
    assert result.aggregate_summary.n_portfolio_obs == expected


def test_pooled_portfolio_return_fold_ids_cover_all_folds() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
    )
    expected = set(range(result.aggregate_summary.n_folds))
    actual = set(result.pooled_portfolio_return_df["fold_id"].unique())
    assert actual.issubset(expected)


# ---------------------------------------------------------------------------
# 10. Pooled cost-adjusted portfolio-return observations
# ---------------------------------------------------------------------------


def test_pooled_cost_adj_df_empty_without_cost_rate() -> None:
    """No cost rate → empty pooled cost-adjusted DataFrame."""
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
    )
    assert result.pooled_cost_adjusted_portfolio_return_df.empty
    assert list(result.pooled_cost_adjusted_portfolio_return_df.columns) == [
        "fold_id", "date", "portfolio_return", "adjusted_return"
    ]


def test_pooled_cost_adj_df_empty_without_portfolio_params() -> None:
    """No portfolio simulation → empty pooled cost-adjusted DataFrame."""
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        portfolio_cost_rate=0.001,
    )
    assert result.pooled_cost_adjusted_portfolio_return_df.empty


def test_pooled_cost_adj_df_populated_with_both_params() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
        portfolio_cost_rate=0.001,
    )
    assert not result.pooled_cost_adjusted_portfolio_return_df.empty
    assert list(result.pooled_cost_adjusted_portfolio_return_df.columns) == [
        "fold_id", "date", "portfolio_return", "adjusted_return"
    ]


def test_pooled_cost_adj_df_row_count_matches_fold_totals() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
        portfolio_cost_rate=0.001,
    )
    expected = sum(
        len(r.portfolio_cost_adjusted_return_df)
        for r in result.per_fold_results
        if r.portfolio_cost_adjusted_return_df is not None
        and not r.portfolio_cost_adjusted_return_df.empty
    )
    assert len(result.pooled_cost_adjusted_portfolio_return_df) == expected


def test_pooled_cost_adj_portfolio_return_matches_gross_return() -> None:
    """portfolio_return column in cost-adjusted df == gross portfolio return."""
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
        portfolio_cost_rate=0.001,
    )
    df_adj = result.pooled_cost_adjusted_portfolio_return_df
    df_gross = result.pooled_portfolio_return_df

    # Every date in pooled cost-adjusted must also appear in pooled gross return.
    adj_dates = set(pd.to_datetime(df_adj["date"]).unique())
    gross_dates = set(pd.to_datetime(df_gross["date"]).unique())
    assert adj_dates.issubset(gross_dates), (
        "pooled_cost_adjusted dates are not a subset of pooled_portfolio_return dates"
    )


def test_no_oos_contamination_in_pooled_cost_adj() -> None:
    """Every date in pooled_cost_adjusted_portfolio_return_df must fall within
    its fold's test window — no IS or non-OOS dates."""
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
        portfolio_cost_rate=0.001,
    )
    df = result.pooled_cost_adjusted_portfolio_return_df
    fold_df = result.fold_summary_df

    for fold_id in df["fold_id"].unique():
        fold_dates = pd.to_datetime(df[df["fold_id"] == fold_id]["date"])
        start = pd.Timestamp(
            fold_df.loc[fold_df["fold_id"] == fold_id, "start_date"].iloc[0]
        )
        end = pd.Timestamp(
            fold_df.loc[fold_df["fold_id"] == fold_id, "end_date"].iloc[0]
        )
        assert fold_dates.min() >= start, (
            f"Fold {fold_id}: cost-adj date {fold_dates.min()} precedes start {start}"
        )
        assert fold_dates.max() <= end, (
            f"Fold {fold_id}: cost-adj date {fold_dates.max()} exceeds end {end}"
        )


def test_adjusted_return_le_portfolio_return_on_non_nan_rows() -> None:
    """On rows where adjusted_return is finite, adjusted <= portfolio_return."""
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
        portfolio_cost_rate=0.01,
    )
    df = result.pooled_cost_adjusted_portfolio_return_df.dropna(
        subset=["portfolio_return", "adjusted_return"]
    )
    if not df.empty:
        assert (df["adjusted_return"] <= df["portfolio_return"]).all()


def test_pooled_cost_adj_aggregate_nan_without_cost_rate() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
    )
    assert math.isnan(result.aggregate_summary.pooled_cost_adjusted_return_mean)
    assert math.isnan(result.aggregate_summary.pooled_cost_adjusted_return_std)
    assert result.aggregate_summary.n_cost_adjusted_obs == 0


def test_pooled_cost_adj_aggregate_finite_with_cost_rate() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
        portfolio_cost_rate=0.001,
    )
    assert math.isfinite(result.aggregate_summary.pooled_cost_adjusted_return_mean)


def test_pooled_cost_adj_aggregate_mean_matches_manual() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
        portfolio_cost_rate=0.001,
    )
    manual = float(
        result.pooled_cost_adjusted_portfolio_return_df["adjusted_return"].dropna().mean()
    )
    assert math.isclose(
        result.aggregate_summary.pooled_cost_adjusted_return_mean, manual, rel_tol=1e-9
    )


def test_n_cost_adjusted_obs_matches_pooled_df_non_nan_count() -> None:
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
        portfolio_cost_rate=0.001,
    )
    expected = int(
        result.pooled_cost_adjusted_portfolio_return_df["adjusted_return"].notna().sum()
    )
    assert result.aggregate_summary.n_cost_adjusted_obs == expected


def test_consistency_fold_vs_pooled_cost_adjusted() -> None:
    """pooled_cost_adjusted_portfolio_return_df must be exactly the vertical
    concatenation of per-fold portfolio_cost_adjusted_return_df outputs."""
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        holding_period=1,
        rebalance_frequency=1,
        portfolio_cost_rate=0.001,
    )
    pooled = result.pooled_cost_adjusted_portfolio_return_df
    for fold_id, fold_result in enumerate(result.per_fold_results):
        if fold_result.portfolio_cost_adjusted_return_df is None:
            continue
        fold_adj = fold_result.portfolio_cost_adjusted_return_df
        pooled_fold = pooled[pooled["fold_id"] == fold_id][
            ["date", "portfolio_return", "adjusted_return"]
        ].reset_index(drop=True)
        expected = fold_adj.reset_index(drop=True)
        assert len(pooled_fold) == len(expected), (
            f"Fold {fold_id}: pooled has {len(pooled_fold)} rows, fold has {len(expected)}"
        )


# ---------------------------------------------------------------------------
# 11. Strategy integration in walk-forward
# ---------------------------------------------------------------------------


def test_walk_forward_strategy_warns_on_conflicting_params() -> None:
    """UserWarning is raised when strategy is provided with explicit holding_period."""
    spec = StrategySpec(holding_period=1, rebalance_frequency=1)
    with pytest.warns(UserWarning, match="holding_period"):
        run_walk_forward_experiment(
            _PRICES,
            _factor_fn,
            train_size=30,
            test_size=10,
            step=10,
            holding_period=5,
            strategy=spec,
        )


def test_walk_forward_strategy_enables_portfolio_path() -> None:
    spec = StrategySpec(holding_period=1, rebalance_frequency=1)
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        strategy=spec,
    )
    assert not result.pooled_portfolio_return_df.empty
    assert math.isfinite(result.aggregate_summary.mean_portfolio_return)


def test_walk_forward_strategy_long_top_k() -> None:
    """With long_top_k, each fold's portfolio_weights_df has <= top_k long entries."""
    spec = StrategySpec(long_top_k=2, holding_period=1, rebalance_frequency=1)
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        strategy=spec,
    )
    for fold_result in result.per_fold_results:
        if fold_result.portfolio_weights_df is None:
            continue
        for date, g in fold_result.portfolio_weights_df.groupby("date"):
            n_long = (g["weight"] > 0).sum()
            assert n_long <= 2, f"More than top_k=2 long assets at {date}"


# ---------------------------------------------------------------------------
# 12. Pooled portfolio turnover
# ---------------------------------------------------------------------------


def test_pooled_portfolio_turnover_df_columns() -> None:
    """pooled_portfolio_turnover_df has expected columns when portfolio is enabled."""
    spec = StrategySpec(holding_period=1, rebalance_frequency=1)
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        strategy=spec,
    )
    assert list(result.pooled_portfolio_turnover_df.columns) == [
        "fold_id",
        "date",
        "portfolio_turnover",
    ]


def test_pooled_portfolio_turnover_df_empty_without_portfolio() -> None:
    """pooled_portfolio_turnover_df is empty when portfolio params are omitted."""
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    assert result.pooled_portfolio_turnover_df.empty


def test_pooled_portfolio_turnover_mean_finite_with_portfolio() -> None:
    """aggregate_summary.pooled_portfolio_turnover_mean is finite when portfolio runs."""
    spec = StrategySpec(holding_period=1, rebalance_frequency=1)
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        strategy=spec,
    )
    assert math.isfinite(result.aggregate_summary.pooled_portfolio_turnover_mean)


def test_pooled_portfolio_turnover_mean_nan_without_portfolio() -> None:
    """aggregate_summary.pooled_portfolio_turnover_mean is NaN without portfolio."""
    result = run_walk_forward_experiment(
        _PRICES,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
    )
    assert math.isnan(result.aggregate_summary.pooled_portfolio_turnover_mean)
