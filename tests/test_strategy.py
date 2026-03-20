from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alpha_lab.experiment import run_factor_experiment
from alpha_lab.factors.momentum import momentum
from alpha_lab.portfolio_research import _WEIGHT_COLUMNS
from alpha_lab.strategy import StrategySpec, portfolio_weights_from_strategy

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_prices(n_assets: int = 6, n_days: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows = []
    for asset in [f"A{i}" for i in range(n_assets)]:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


def _make_factor_df(n_assets: int = 6, n_days: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows = [
        {"date": d, "asset": f"A{i}", "factor": "f", "value": float(rng.standard_normal())}
        for d in dates
        for i in range(n_assets)
    ]
    return pd.DataFrame(rows)


_PRICES = _make_prices()


# ---------------------------------------------------------------------------
# 1. StrategySpec construction and defaults
# ---------------------------------------------------------------------------


def test_default_spec_is_long_only() -> None:
    spec = StrategySpec()
    assert not spec.is_long_short


def test_spec_with_short_bottom_k_is_long_short() -> None:
    spec = StrategySpec(short_bottom_k=2)
    assert spec.is_long_short


def test_default_values() -> None:
    spec = StrategySpec()
    assert spec.long_top_k is None
    assert spec.short_bottom_k is None
    assert spec.weighting_method == "equal"
    assert spec.holding_period == 1
    assert spec.rebalance_frequency == 1


def test_spec_is_immutable() -> None:
    spec = StrategySpec()
    with pytest.raises(AttributeError):
        spec.weighting_method = "rank"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. StrategySpec validation
# ---------------------------------------------------------------------------


def test_long_top_k_zero_raises() -> None:
    with pytest.raises(ValueError, match="long_top_k"):
        StrategySpec(long_top_k=0)


def test_short_bottom_k_zero_raises() -> None:
    with pytest.raises(ValueError, match="short_bottom_k"):
        StrategySpec(short_bottom_k=0)


def test_invalid_weighting_method_raises() -> None:
    with pytest.raises(ValueError, match="weighting_method"):
        StrategySpec(weighting_method="invalid")


def test_holding_period_zero_raises() -> None:
    with pytest.raises(ValueError, match="holding_period"):
        StrategySpec(holding_period=0)


def test_rebalance_frequency_zero_raises() -> None:
    with pytest.raises(ValueError, match="rebalance_frequency"):
        StrategySpec(rebalance_frequency=0)


def test_valid_spec_constructs_without_error() -> None:
    spec = StrategySpec(
        long_top_k=5,
        short_bottom_k=3,
        weighting_method="rank",
        holding_period=3,
        rebalance_frequency=2,
    )
    assert spec.is_long_short


# ---------------------------------------------------------------------------
# 3. portfolio_weights_from_strategy
# ---------------------------------------------------------------------------


def test_portfolio_weights_from_strategy_returns_weight_columns() -> None:
    spec = StrategySpec()
    factor_df = _make_factor_df()
    result = portfolio_weights_from_strategy(factor_df, spec)
    assert list(result.columns) == list(_WEIGHT_COLUMNS)


def test_portfolio_weights_from_strategy_long_only_all_positive() -> None:
    spec = StrategySpec(short_bottom_k=None)
    factor_df = _make_factor_df()
    result = portfolio_weights_from_strategy(factor_df, spec)
    assert (result["weight"] >= 0).all()


def test_portfolio_weights_from_strategy_long_short_net_zero() -> None:
    spec = StrategySpec(long_top_k=2, short_bottom_k=2, weighting_method="equal")
    factor_df = _make_factor_df()
    result = portfolio_weights_from_strategy(factor_df, spec)
    for date, g in result.groupby("date"):
        net = g["weight"].sum()
        assert math.isclose(net, 0.0, abs_tol=1e-9), f"Net weight at {date} is {net}"


def test_portfolio_weights_from_strategy_top_k_limits_long_leg() -> None:
    spec = StrategySpec(long_top_k=2)
    factor_df = _make_factor_df(n_assets=6, n_days=1)
    result = portfolio_weights_from_strategy(factor_df, spec)
    # Exactly 2 assets in the long leg.
    assert len(result) == 2
    assert (result["weight"] > 0).all()


def test_portfolio_weights_from_strategy_rank_weights_sum_to_1() -> None:
    spec = StrategySpec(weighting_method="rank")
    factor_df = _make_factor_df(n_assets=4, n_days=5)
    result = portfolio_weights_from_strategy(factor_df, spec)
    for date, g in result.groupby("date"):
        total = g["weight"].sum()
        assert math.isclose(total, 1.0, rel_tol=1e-9), f"Sum at {date}: {total}"


# ---------------------------------------------------------------------------
# 4. Integration with run_factor_experiment
# ---------------------------------------------------------------------------


def test_experiment_strategy_warns_on_conflicting_holding_period() -> None:
    """A UserWarning is raised when strategy is provided with explicit holding_period."""
    prices = _make_prices()
    spec = StrategySpec(holding_period=1, rebalance_frequency=1)
    with pytest.warns(UserWarning, match="holding_period"):
        run_factor_experiment(
            prices,
            lambda p: momentum(p, window=5),
            strategy=spec,
            holding_period=2,
        )


def test_experiment_strategy_enables_portfolio_simulation() -> None:
    prices = _make_prices()
    spec = StrategySpec(holding_period=1, rebalance_frequency=1)
    result = run_factor_experiment(
        prices, lambda p: momentum(p, window=5), strategy=spec
    )
    assert result.portfolio_weights_df is not None
    assert result.portfolio_return_df is not None
    assert result.portfolio_summary is not None


def test_experiment_strategy_none_leaves_no_portfolio() -> None:
    """Without strategy or explicit portfolio params, portfolio fields are None."""
    prices = _make_prices()
    result = run_factor_experiment(prices, lambda p: momentum(p, window=5))
    assert result.portfolio_weights_df is None
    assert result.portfolio_summary is None


def test_experiment_strategy_long_top_k_limits_weights() -> None:
    prices = _make_prices(n_assets=6)
    spec = StrategySpec(long_top_k=3, holding_period=1, rebalance_frequency=1)
    result = run_factor_experiment(
        prices, lambda p: momentum(p, window=5), strategy=spec
    )
    assert result.portfolio_weights_df is not None
    for date, g in result.portfolio_weights_df.groupby("date"):
        n_long = (g["weight"] > 0).sum()
        assert n_long <= 3, f"More than top_k=3 long assets at {date}"


def test_experiment_strategy_long_short_net_zero() -> None:
    prices = _make_prices(n_assets=6)
    spec = StrategySpec(long_top_k=2, short_bottom_k=2, holding_period=1, rebalance_frequency=1)
    result = run_factor_experiment(
        prices, lambda p: momentum(p, window=5), strategy=spec
    )
    assert result.portfolio_weights_df is not None
    for date, g in result.portfolio_weights_df.groupby("date"):
        net = g["weight"].sum()
        assert math.isclose(net, 0.0, abs_tol=1e-9), f"Net weight at {date}: {net}"


def test_experiment_strategy_holding_period_wins_over_explicit() -> None:
    """strategy.holding_period wins over the explicit holding_period argument."""
    prices = _make_prices()
    spec = StrategySpec(holding_period=1, rebalance_frequency=1)
    with pytest.warns(UserWarning):
        result = run_factor_experiment(
            prices,
            lambda p: momentum(p, window=5),
            holding_period=99,  # ignored — spec wins
            rebalance_frequency=99,  # ignored — spec wins
            strategy=spec,
        )
    # Portfolio simulation runs with spec's holding_period=1, not 99.
    assert result.portfolio_weights_df is not None


def test_experiment_strategy_cost_rate_separate_from_spec() -> None:
    """portfolio_cost_rate is not part of StrategySpec; it is a separate param."""
    prices = _make_prices()
    spec = StrategySpec(holding_period=1, rebalance_frequency=1)
    result_no_cost = run_factor_experiment(
        prices, lambda p: momentum(p, window=5), strategy=spec
    )
    result_with_cost = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        strategy=spec,
        portfolio_cost_rate=0.001,
    )
    assert result_no_cost.portfolio_cost_adjusted_return_df is None
    assert result_with_cost.portfolio_cost_adjusted_return_df is not None
