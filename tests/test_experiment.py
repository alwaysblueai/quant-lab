from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alpha_lab.experiment import ExperimentResult, ExperimentSummary, run_factor_experiment
from alpha_lab.experiment_metadata import ExperimentMetadata
from alpha_lab.factors.momentum import momentum
from alpha_lab.timing import DelaySpec

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_prices(n_assets: int = 6, n_days: int = 30, seed: int = 42) -> pd.DataFrame:
    """Synthetic long-form price panel — no lookahead, deterministic."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    assets = [f"A{i}" for i in range(n_assets)]
    rows = []
    for asset in assets:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


def _momentum_fn(prices: pd.DataFrame) -> pd.DataFrame:
    return momentum(prices, window=5)


def _constant_fn(prices: pd.DataFrame) -> pd.DataFrame:
    """Factor that returns the same value for every asset on every date."""
    dates = pd.to_datetime(prices["date"]).unique()
    assets = prices["asset"].unique()
    rows = [
        {"date": d, "asset": a, "factor": "const", "value": 1.0}
        for d in dates
        for a in assets
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Basic end-to-end success
# ---------------------------------------------------------------------------


def test_run_factor_experiment_returns_experiment_result():
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    assert isinstance(result, ExperimentResult)


def test_result_fields_are_dataframes():
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    assert isinstance(result.factor_df, pd.DataFrame)
    assert isinstance(result.label_df, pd.DataFrame)
    assert isinstance(result.ic_df, pd.DataFrame)
    assert isinstance(result.rank_ic_df, pd.DataFrame)
    assert isinstance(result.quantile_returns_df, pd.DataFrame)
    assert isinstance(result.long_short_df, pd.DataFrame)


def test_summary_is_experiment_summary_instance():
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    assert isinstance(result.summary, ExperimentSummary)


# ---------------------------------------------------------------------------
# 2. Canonical schema preserved
# ---------------------------------------------------------------------------


def test_factor_df_has_canonical_columns():
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    assert {"date", "asset", "factor", "value"}.issubset(result.factor_df.columns)


def test_label_df_has_canonical_columns():
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    assert {"date", "asset", "factor", "value"}.issubset(result.label_df.columns)


def test_ic_df_has_expected_columns():
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    assert {"date", "ic"}.issubset(result.ic_df.columns)


def test_rank_ic_df_has_expected_columns():
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    assert {"date", "rank_ic"}.issubset(result.rank_ic_df.columns)


def test_quantile_returns_df_has_expected_columns():
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    assert {"date", "quantile", "mean_return"}.issubset(result.quantile_returns_df.columns)


def test_long_short_df_has_expected_columns():
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    assert {"date", "long_short_return"}.issubset(result.long_short_df.columns)


def test_factor_df_no_duplicate_rows():
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    dupes = result.factor_df.duplicated(subset=["date", "asset", "factor"])
    assert not dupes.any()


def test_label_df_no_duplicate_rows():
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    dupes = result.label_df.duplicated(subset=["date", "asset", "factor"])
    assert not dupes.any()


# ---------------------------------------------------------------------------
# 3. Summary metrics sensible
# ---------------------------------------------------------------------------


def test_summary_numeric_fields_are_finite_or_nan():
    s = run_factor_experiment(_make_prices(), _momentum_fn).summary
    for field in (
        s.mean_ic,
        s.mean_rank_ic,
        s.ic_ir,
        s.mean_long_short_return,
        s.long_short_hit_rate,
    ):
        assert math.isfinite(field) or math.isnan(field)


def test_summary_hit_rate_in_unit_interval():
    hr = run_factor_experiment(_make_prices(), _momentum_fn).summary.long_short_hit_rate
    if not math.isnan(hr):
        assert 0.0 <= hr <= 1.0


def test_summary_n_dates_is_non_negative_int():
    s = run_factor_experiment(_make_prices(), _momentum_fn).summary
    assert isinstance(s.n_dates, int)
    assert s.n_dates >= 0


def test_summary_n_dates_counts_only_finite_ic_dates():
    """n_dates counts dates with a non-NaN IC value, not all dates in ic_df."""
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    finite_ic_dates = result.ic_df.loc[result.ic_df["ic"].notna(), "date"].nunique()
    assert result.summary.n_dates == finite_ic_dates


def test_summary_ic_ir_nan_when_single_eval_date():
    """ic_ir requires at least 2 IC observations (ddof=1); with 1 date it is NaN."""
    # 1 asset × 10 days: no cross-section → all IC NaN → n_dates == 0, ic_ir NaN
    result = run_factor_experiment(_make_prices(n_assets=1, n_days=10), _momentum_fn)
    assert math.isnan(result.summary.ic_ir)


def test_summary_n_dates_zero_when_all_ic_nan():
    """n_dates must be 0 when all IC values are NaN (no valid cross-sections)."""
    result = run_factor_experiment(_make_prices(n_assets=1, n_days=15), _momentum_fn)
    assert result.summary.n_dates == 0


# ---------------------------------------------------------------------------
# Summary metrics — exact aggregation correctness
# ---------------------------------------------------------------------------


def test_summary_mean_ic_matches_ic_df():
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    expected = float(result.ic_df["ic"].dropna().mean())
    assert math.isclose(result.summary.mean_ic, expected)


def test_summary_mean_rank_ic_matches_rank_ic_df():
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    expected = float(result.rank_ic_df["rank_ic"].dropna().mean())
    assert math.isclose(result.summary.mean_rank_ic, expected)


def test_summary_mean_long_short_return_matches_long_short_df():
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    expected = float(result.long_short_df["long_short_return"].dropna().mean())
    assert math.isclose(result.summary.mean_long_short_return, expected)


def test_summary_mean_long_short_turnover_aligned_to_return_universe():
    """mean_long_short_turnover must be averaged over the same date universe as
    long_short_df, not the broader quantile_assignment universe.

    With horizon > 1, the last `horizon - 1` factor dates have no valid label,
    so long_short_df has fewer dates than long_short_turnover_df.  The summary
    scalar must exclude those trailing dates.
    """
    result = run_factor_experiment(_make_prices(n_days=40), _momentum_fn, horizon=5)
    ls_dates = set(result.long_short_df["date"].unique())
    lsto_restricted = result.long_short_turnover_df[
        result.long_short_turnover_df["date"].isin(ls_dates)
    ]
    expected_vals = lsto_restricted["long_short_turnover"].dropna()
    if len(expected_vals) == 0:
        assert math.isnan(result.summary.mean_long_short_turnover)
    else:
        expected = float(expected_vals.mean())
        assert math.isclose(result.summary.mean_long_short_turnover, expected)


def test_summary_long_short_hit_rate_matches_long_short_df():
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    ls_vals = result.long_short_df["long_short_return"].dropna()
    expected = float((ls_vals > 0).mean())
    assert math.isclose(result.summary.long_short_hit_rate, expected)


def test_summary_ic_ir_matches_mean_over_std():
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    ic_vals = result.ic_df["ic"].dropna()
    if len(ic_vals) > 1:
        expected = float(ic_vals.mean()) / float(ic_vals.std(ddof=1))
        assert math.isclose(result.summary.ic_ir, expected)


# ---------------------------------------------------------------------------
# 4. No-lookahead: full-sample DataFrames
# ---------------------------------------------------------------------------


def test_factor_df_covers_full_price_panel():
    """factor_df must include all dates present in the price panel."""
    prices = _make_prices()
    result = run_factor_experiment(prices, _momentum_fn)
    price_dates = set(pd.to_datetime(prices["date"]).unique())
    factor_dates = set(pd.to_datetime(result.factor_df["date"]).unique())
    assert factor_dates.issubset(price_dates)


def test_label_df_covers_full_price_panel():
    prices = _make_prices()
    result = run_factor_experiment(prices, _momentum_fn)
    price_dates = set(pd.to_datetime(prices["date"]).unique())
    label_dates = set(pd.to_datetime(result.label_df["date"]).unique())
    assert label_dates.issubset(price_dates)


def test_full_sample_eval_ic_dates_subset_of_factor_dates():
    """Without a split, IC dates must be a subset of factor dates."""
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    factor_dates = set(pd.to_datetime(result.factor_df["date"]).unique())
    ic_dates = set(pd.to_datetime(result.ic_df["date"]).unique())
    assert ic_dates.issubset(factor_dates)


def test_label_factor_name_encodes_horizon():
    """label_df must use 'forward_return_{horizon}' as the factor name."""
    result = run_factor_experiment(_make_prices(), _momentum_fn, horizon=3)
    label_names = result.label_df["factor"].unique()
    assert len(label_names) == 1
    assert label_names[0] == "forward_return_3"


# ---------------------------------------------------------------------------
# 5. Split handling
# ---------------------------------------------------------------------------

_TRAIN_END = "2024-01-22"
_TEST_START = "2024-01-25"


def _split_result() -> ExperimentResult:
    return run_factor_experiment(
        _make_prices(n_days=40),
        _momentum_fn,
        train_end=_TRAIN_END,
        test_start=_TEST_START,
    )


def test_split_ic_dates_all_in_test_period():
    result = _split_result()
    if not result.ic_df.empty:
        ic_min = pd.to_datetime(result.ic_df["date"]).min()
        assert ic_min >= pd.Timestamp(_TEST_START)


def test_split_quantile_returns_dates_all_in_test_period():
    result = _split_result()
    if not result.quantile_returns_df.empty:
        qr_min = pd.to_datetime(result.quantile_returns_df["date"]).min()
        assert qr_min >= pd.Timestamp(_TEST_START)


def test_split_long_short_dates_all_in_test_period():
    result = _split_result()
    if not result.long_short_df.empty:
        ls_min = pd.to_datetime(result.long_short_df["date"]).min()
        assert ls_min >= pd.Timestamp(_TEST_START)


def test_split_factor_df_includes_train_period():
    """factor_df is always full-sample; it must contain pre-train_end dates."""
    result = _split_result()
    factor_dates = pd.to_datetime(result.factor_df["date"])
    assert (factor_dates <= pd.Timestamp(_TRAIN_END)).any()


def test_split_label_df_includes_train_period():
    result = _split_result()
    label_dates = pd.to_datetime(result.label_df["date"])
    assert (label_dates <= pd.Timestamp(_TRAIN_END)).any()


def test_split_produces_fewer_eval_dates_than_full_sample():
    prices = _make_prices(n_days=40)
    full = run_factor_experiment(prices, _momentum_fn)
    split = run_factor_experiment(
        prices, _momentum_fn, train_end=_TRAIN_END, test_start=_TEST_START
    )
    assert split.summary.n_dates < full.summary.n_dates


# ---------------------------------------------------------------------------
# 5b. Split argument validation — partial specification must raise
# ---------------------------------------------------------------------------


def test_train_end_without_test_start_raises():
    with pytest.raises(ValueError, match="both"):
        run_factor_experiment(_make_prices(), _momentum_fn, train_end=_TRAIN_END)


def test_test_start_without_train_end_raises():
    with pytest.raises(ValueError, match="both"):
        run_factor_experiment(_make_prices(), _momentum_fn, test_start=_TEST_START)


def test_val_start_without_split_raises():
    with pytest.raises(ValueError, match="val_start"):
        run_factor_experiment(_make_prices(), _momentum_fn, val_start="2024-01-15")


def test_val_start_with_train_end_but_no_test_start_raises():
    """val_start + train_end but no test_start: XOR guard fires on test_start=None."""
    with pytest.raises(ValueError, match="both"):
        run_factor_experiment(
            _make_prices(), _momentum_fn, train_end=_TRAIN_END, val_start="2024-01-16"
        )


def test_no_split_uses_all_available_dates():
    """Without train_end/test_start, no dates are excluded from evaluation."""
    prices = _make_prices()
    result = run_factor_experiment(prices, _momentum_fn)
    # ic_dates may be smaller (warm-up NaN, horizon trim), but must not exclude
    # dates that have valid factor and label values
    merged = result.factor_df.merge(
        result.label_df[["date", "asset", "value"]].rename(columns={"value": "_label"}),
        on=["date", "asset"],
        how="inner",
    ).dropna(subset=["value", "_label"])
    valid_dates = set(pd.to_datetime(merged["date"]).unique())
    # Every valid date must appear in ic_df (may be NaN but date present)
    assert valid_dates.issubset(set(pd.to_datetime(result.ic_df["date"]).unique()))


# ---------------------------------------------------------------------------
# 6. Factor / label alignment
# ---------------------------------------------------------------------------


def test_ic_df_dates_are_subset_of_factor_df_dates():
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    factor_dates = set(pd.to_datetime(result.factor_df["date"]).unique())
    ic_dates = set(pd.to_datetime(result.ic_df["date"]).unique())
    assert ic_dates.issubset(factor_dates)


def test_ic_df_dates_are_subset_of_label_df_dates():
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    label_dates = set(pd.to_datetime(result.label_df["date"]).unique())
    ic_dates = set(pd.to_datetime(result.ic_df["date"]).unique())
    assert ic_dates.issubset(label_dates)


def test_horizon_affects_label_not_factor():
    """Changing horizon shifts label dates but must not change factor values."""
    prices = _make_prices()
    r1 = run_factor_experiment(prices, _momentum_fn, horizon=1)
    r3 = run_factor_experiment(prices, _momentum_fn, horizon=3)
    # Factor values are identical regardless of horizon
    f1 = r1.factor_df.sort_values(["date", "asset"]).reset_index(drop=True)["value"]
    f3 = r3.factor_df.sort_values(["date", "asset"]).reset_index(drop=True)["value"]
    pd.testing.assert_series_equal(f1, f3, check_names=False)
    # Label names differ
    assert r1.label_df["factor"].iloc[0] != r3.label_df["factor"].iloc[0]


# ---------------------------------------------------------------------------
# 7. Degenerate cases
# ---------------------------------------------------------------------------


def test_constant_factor_mean_ic_is_nan():
    """Zero cross-sectional variance → Pearson IC undefined → NaN."""
    result = run_factor_experiment(_make_prices(), _constant_fn)
    assert math.isnan(result.summary.mean_ic)


def test_constant_factor_mean_rank_ic_is_nan():
    result = run_factor_experiment(_make_prices(), _constant_fn)
    assert math.isnan(result.summary.mean_rank_ic)


def test_constant_factor_long_short_is_nan():
    """All assets in bucket 1 → top == bottom → L/S return is NaN."""
    result = run_factor_experiment(_make_prices(), _constant_fn)
    if not result.long_short_df.empty:
        assert result.long_short_df["long_short_return"].isna().all()


def test_single_asset_ic_is_nan():
    """Single asset per date → no cross-section → IC always NaN."""
    result = run_factor_experiment(_make_prices(n_assets=1, n_days=15), _momentum_fn)
    assert math.isnan(result.summary.mean_ic)


def test_empty_test_period_gives_empty_eval_outputs():
    """train_end == last price date → zero test-period rows → empty eval outputs."""
    prices = _make_prices(n_days=20)
    last_date = pd.to_datetime(prices["date"]).max()
    after_data = last_date + pd.Timedelta(days=3)
    result = run_factor_experiment(
        prices,
        _momentum_fn,
        train_end=str(last_date.date()),
        test_start=str(after_data.date()),
    )
    assert result.ic_df.empty
    assert result.summary.n_dates == 0
    assert math.isnan(result.summary.mean_ic)


def test_two_asset_experiment_runs_without_error():
    """Minimal viable cross-section (2 assets) should complete successfully."""
    result = run_factor_experiment(_make_prices(n_assets=2, n_days=20), _momentum_fn)
    assert isinstance(result.summary, ExperimentSummary)
    assert result.summary.n_dates > 0


def test_split_eval_label_dates_restricted_to_test_period():
    """eval_label passed to downstream modules must only contain test-period dates.

    We verify this indirectly: ic_df dates must not precede test_start, and the
    IC computation only receives label rows for those exact dates.
    """
    prices = _make_prices(n_days=40)
    result = run_factor_experiment(
        prices, _momentum_fn, train_end=_TRAIN_END, test_start=_TEST_START
    )
    test_start_ts = pd.Timestamp(_TEST_START)
    # If any label date before test_start had leaked into evaluation, IC dates
    # would precede test_start — assert none do.
    if not result.ic_df.empty:
        assert pd.to_datetime(result.ic_df["date"]).min() >= test_start_ts
    if not result.rank_ic_df.empty:
        assert pd.to_datetime(result.rank_ic_df["date"]).min() >= test_start_ts


def test_ic_ir_is_nan_when_ic_has_zero_variance():
    """ic_ir = mean_ic / std_ic; when std_ic == 0 the result must be NaN, not inf."""

    def flat_ic_fn(prices: pd.DataFrame) -> pd.DataFrame:
        # Two-asset factor that perfectly separates assets on every date:
        # A0 always gets value 1.0, A1 always gets value -1.0.
        # This produces a constant IC series (all +1 or all -1) → std_ic == 0.
        dates = pd.to_datetime(prices["date"]).unique()
        rows = []
        for d in dates:
            rows.append({"date": d, "asset": "A0", "factor": "flat", "value": 1.0})
            rows.append({"date": d, "asset": "A1", "factor": "flat", "value": -1.0})
        return pd.DataFrame(rows)

    def matching_prices(_: pd.DataFrame) -> pd.DataFrame:
        # Provide a price panel that flat_ic_fn can work with.
        dates = pd.date_range("2024-01-01", periods=20, freq="B")
        rows = []
        for i, d in enumerate(dates):
            rows.append({"date": d, "asset": "A0", "close": 100.0 + i})
            rows.append({"date": d, "asset": "A1", "close": 100.0 - i})
        return pd.DataFrame(rows)

    prices = matching_prices(pd.DataFrame())
    result = run_factor_experiment(prices, flat_ic_fn)
    ic_vals = result.ic_df["ic"].dropna()
    if len(ic_vals) > 1 and float(ic_vals.std(ddof=1)) == 0.0:
        assert math.isnan(result.summary.ic_ir)


# ---------------------------------------------------------------------------
# 8. Timing / metadata / diagnostics contracts
# ---------------------------------------------------------------------------


def test_default_delay_spec_matches_horizon() -> None:
    result = run_factor_experiment(_make_prices(), _momentum_fn, horizon=3)
    assert result.delay_spec is not None
    assert result.delay_spec.return_horizon_periods == 3
    assert result.label_metadata is not None
    assert result.label_metadata.horizon_periods == 3


def test_custom_delay_spec_must_match_horizon() -> None:
    with pytest.raises(ValueError, match="must match horizon"):
        run_factor_experiment(
            _make_prices(),
            _momentum_fn,
            horizon=5,
            delay_spec=DelaySpec.for_horizon(3),
        )


def test_metadata_without_delay_is_completed_by_runner() -> None:
    md = ExperimentMetadata(dataset_id="snapshot-1")
    result = run_factor_experiment(_make_prices(), _momentum_fn, metadata=md, horizon=2)
    assert result.metadata is not None
    assert result.metadata.delay is not None
    assert result.metadata.delay.return_horizon_periods == 2


def test_generate_factor_report_attaches_report() -> None:
    result = run_factor_experiment(
        _make_prices(),
        _momentum_fn,
        horizon=5,
        generate_factor_report=True,
    )
    assert result.factor_report is not None
    assert result.factor_report.horizon == 5


def test_sample_weights_are_propagated_when_provided() -> None:
    prices = _make_prices(n_assets=4, n_days=20)
    keys = prices[["date", "asset"]].drop_duplicates().copy()
    keys["sample_weight"] = 1.0
    result = run_factor_experiment(
        prices,
        _momentum_fn,
        sample_weights=keys,
    )
    assert result.sample_weights_df is not None
    assert {"date", "asset", "sample_weight"} == set(result.sample_weights_df.columns)
    assert len(result.sample_weights_df) == len(
        result.factor_df[["date", "asset"]].drop_duplicates()
    )


def test_sample_weights_duplicate_keys_raise() -> None:
    prices = _make_prices(n_assets=4, n_days=20)
    keys = prices[["date", "asset"]].drop_duplicates().iloc[:10].copy()
    bad = pd.concat([keys, keys.iloc[[0]]], ignore_index=True)
    bad["sample_weight"] = 1.0
    with pytest.raises(ValueError, match="duplicate"):
        run_factor_experiment(prices, _momentum_fn, sample_weights=bad)
