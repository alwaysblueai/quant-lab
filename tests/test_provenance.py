"""Tests for ExperimentProvenance, diagnostics fields, and no-op parameter warnings."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.experiment import ExperimentProvenance, run_factor_experiment
from alpha_lab.factors.momentum import momentum
from alpha_lab.strategy import StrategySpec
from alpha_lab.walk_forward import run_walk_forward_experiment

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_prices(n_assets: int = 6, n_days: int = 30, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows = []
    for asset in [f"A{i}" for i in range(n_assets)]:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


_PRICES = _make_prices()


def _momentum_fn(prices: pd.DataFrame) -> pd.DataFrame:
    return momentum(prices, window=5)


# ---------------------------------------------------------------------------
# 1. ExperimentProvenance fields
# ---------------------------------------------------------------------------


def test_provenance_is_attached_to_result() -> None:
    result = run_factor_experiment(_PRICES, _momentum_fn)
    assert isinstance(result.provenance, ExperimentProvenance)


def test_provenance_factor_name_matches_factor_df() -> None:
    result = run_factor_experiment(_PRICES, _momentum_fn)
    assert result.provenance.factor_name == result.factor_df["factor"].iloc[0]


def test_provenance_horizon_matches_argument() -> None:
    result = run_factor_experiment(_PRICES, _momentum_fn, horizon=3)
    assert result.provenance.horizon == 3


def test_provenance_n_quantiles_matches_argument() -> None:
    result = run_factor_experiment(_PRICES, _momentum_fn, n_quantiles=7)
    assert result.provenance.n_quantiles == 7


def test_provenance_run_timestamp_utc_is_iso_string() -> None:
    result = run_factor_experiment(_PRICES, _momentum_fn)
    ts = result.provenance.run_timestamp_utc
    # Should be an ISO-8601 string ending in +00:00 (UTC offset)
    assert isinstance(ts, str)
    assert "+00:00" in ts or "Z" in ts or len(ts) >= 19  # at minimum YYYY-MM-DDTHH:MM:SS


def test_provenance_git_commit_is_string_or_none() -> None:
    result = run_factor_experiment(_PRICES, _momentum_fn)
    gc = result.provenance.git_commit
    assert gc is None or isinstance(gc, str)


def test_provenance_portfolio_cost_rate_none_when_not_supplied() -> None:
    result = run_factor_experiment(_PRICES, _momentum_fn)
    assert result.provenance.portfolio_cost_rate is None


def test_provenance_portfolio_cost_rate_captured() -> None:
    result = run_factor_experiment(
        _PRICES, _momentum_fn, holding_period=1, rebalance_frequency=1, portfolio_cost_rate=0.001
    )
    assert result.provenance.portfolio_cost_rate == 0.001


def test_provenance_strategy_repr_none_without_strategy() -> None:
    result = run_factor_experiment(_PRICES, _momentum_fn)
    assert result.provenance.strategy_repr is None


def test_provenance_strategy_repr_populated_with_strategy() -> None:
    spec = StrategySpec(holding_period=1, rebalance_frequency=1)
    result = run_factor_experiment(_PRICES, _momentum_fn, strategy=spec)
    assert result.provenance.strategy_repr == repr(spec)


def test_provenance_is_frozen() -> None:
    result = run_factor_experiment(_PRICES, _momentum_fn)
    with pytest.raises(AttributeError):
        result.provenance.horizon = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. Diagnostics fields
# ---------------------------------------------------------------------------


def test_n_eval_dates_full_sample() -> None:
    result = run_factor_experiment(_PRICES, _momentum_fn)
    n_unique = int(result.factor_df["date"].nunique())
    # eval_factor = factor_df for full-sample run
    assert result.n_eval_dates == n_unique


def test_n_eval_dates_test_period() -> None:
    result = run_factor_experiment(
        _PRICES,
        _momentum_fn,
        train_end="2024-01-19",
        test_start="2024-01-22",
    )
    # n_eval_dates must be less than the full sample count
    full_result = run_factor_experiment(_PRICES, _momentum_fn)
    assert result.n_eval_dates < full_result.n_eval_dates


def test_n_eval_assets_matches_universe() -> None:
    result = run_factor_experiment(_PRICES, _momentum_fn)
    assert result.n_eval_assets == 6  # _make_prices has 6 assets


def test_n_label_nan_dates_is_nonnegative() -> None:
    result = run_factor_experiment(_PRICES, _momentum_fn, horizon=1)
    assert result.n_label_nan_dates >= 0


def test_n_label_nan_dates_matches_horizon() -> None:
    """For a full-sample run, n_label_nan_dates should equal the horizon
    (the last `horizon` dates have no forward return)."""
    horizon = 3
    result = run_factor_experiment(_PRICES, _momentum_fn, horizon=horizon)
    # The number of terminal dates without labels should equal the horizon.
    assert result.n_label_nan_dates == horizon


def test_diagnostics_are_integers() -> None:
    result = run_factor_experiment(_PRICES, _momentum_fn)
    assert isinstance(result.n_eval_dates, int)
    assert isinstance(result.n_eval_assets, int)
    assert isinstance(result.n_label_nan_dates, int)


# ---------------------------------------------------------------------------
# 3. portfolio_cost_rate no-op warning
# ---------------------------------------------------------------------------


def test_cost_rate_without_portfolio_mode_warns_experiment() -> None:
    """UserWarning fired when portfolio_cost_rate is set but no portfolio mode."""
    with pytest.warns(UserWarning, match="portfolio_cost_rate is ignored"):
        run_factor_experiment(_PRICES, _momentum_fn, portfolio_cost_rate=0.001)


def test_cost_rate_with_portfolio_mode_no_warning() -> None:
    """No warning when portfolio_cost_rate is paired with active portfolio mode."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        run_factor_experiment(
            _PRICES,
            _momentum_fn,
            holding_period=1,
            rebalance_frequency=1,
            portfolio_cost_rate=0.001,
        )


def test_cost_rate_without_portfolio_mode_warns_walk_forward() -> None:
    """UserWarning fired in walk_forward when portfolio_cost_rate set but no portfolio mode."""
    prices = _make_prices(n_days=80)
    with pytest.warns(UserWarning, match="portfolio_cost_rate is ignored"):
        run_walk_forward_experiment(
            prices,
            _momentum_fn,
            train_size=30,
            test_size=10,
            step=10,
            portfolio_cost_rate=0.001,
        )


def test_cost_rate_with_strategy_no_warning_walk_forward() -> None:
    """No warning when portfolio_cost_rate paired with strategy in walk_forward."""
    import warnings

    prices = _make_prices(n_days=80)
    spec = StrategySpec(holding_period=1, rebalance_frequency=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        run_walk_forward_experiment(
            prices,
            _momentum_fn,
            train_size=30,
            test_size=10,
            step=10,
            strategy=spec,
            portfolio_cost_rate=0.001,
        )


# ---------------------------------------------------------------------------
# 4. validate_price_panel called from run_factor_experiment
# ---------------------------------------------------------------------------


def test_experiment_rejects_bad_prices_nan_close() -> None:
    df = _make_prices()
    df.iloc[0, df.columns.get_loc("close")] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        run_factor_experiment(df, _momentum_fn)


def test_experiment_rejects_bad_prices_duplicate_date_asset() -> None:
    df = _make_prices()
    dupe = df.iloc[[0]].copy()
    df = pd.concat([df, dupe], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        run_factor_experiment(df, _momentum_fn)
