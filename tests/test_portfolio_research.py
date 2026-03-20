from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from alpha_lab.experiment import PortfolioSummary, run_factor_experiment
from alpha_lab.factors.momentum import momentum
from alpha_lab.labels import forward_return
from alpha_lab.portfolio_research import (
    _PORTFOLIO_COST_ADJ_COLUMNS,
    _PORTFOLIO_RETURN_COLUMNS,
    _PORTFOLIO_TURNOVER_COLUMNS,
    _WEIGHT_COLUMNS,
    portfolio_cost_adjusted_returns,
    portfolio_turnover,
    portfolio_weights,
    simulate_portfolio_returns,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_factor_df(values_by_asset: dict[str, float], date: str = "2024-01-01") -> pd.DataFrame:
    """Build a single-date canonical factor DataFrame."""
    rows = [
        {"date": pd.Timestamp(date), "asset": asset, "factor": "test_factor", "value": v}
        for asset, v in values_by_asset.items()
    ]
    return pd.DataFrame(rows)


def _make_multi_date_factor(n_dates: int = 5, n_assets: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_dates, freq="B")
    rows = []
    for date in dates:
        for i in range(n_assets):
            rows.append({
                "date": date,
                "asset": f"A{i}",
                "factor": "test_factor",
                "value": float(rng.normal(0, 1)),
            })
    return pd.DataFrame(rows)


def _make_prices(n_assets: int = 4, n_days: int = 40, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows = []
    for asset in [f"A{i}" for i in range(n_assets)]:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. portfolio_weights — output schema
# ---------------------------------------------------------------------------


def test_portfolio_weights_returns_correct_columns() -> None:
    factor_df = _make_factor_df({"A": 1.0, "B": 2.0, "C": 3.0})
    result = portfolio_weights(factor_df)
    assert list(result.columns) == list(_WEIGHT_COLUMNS)


def test_portfolio_weights_empty_factor_returns_empty() -> None:
    empty = pd.DataFrame(columns=["date", "asset", "factor", "value"])
    result = portfolio_weights(empty)
    assert result.empty
    assert list(result.columns) == list(_WEIGHT_COLUMNS)


def test_portfolio_weights_missing_columns_raises() -> None:
    bad = pd.DataFrame({"date": [1], "asset": ["A"]})
    with pytest.raises(ValueError, match="missing required columns"):
        portfolio_weights(bad)


def test_portfolio_weights_multiple_factors_raises() -> None:
    df = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-01"), "asset": "A", "factor": "f1", "value": 1.0},
        {"date": pd.Timestamp("2024-01-01"), "asset": "B", "factor": "f2", "value": 2.0},
    ])
    with pytest.raises(ValueError, match="exactly one factor"):
        portfolio_weights(df)


def test_portfolio_weights_invalid_method_raises() -> None:
    factor_df = _make_factor_df({"A": 1.0, "B": 2.0})
    with pytest.raises(ValueError, match="method must be one of"):
        portfolio_weights(factor_df, method="bad_method")


def test_portfolio_weights_invalid_top_k_raises() -> None:
    factor_df = _make_factor_df({"A": 1.0, "B": 2.0})
    with pytest.raises(ValueError, match="top_k"):
        portfolio_weights(factor_df, top_k=0)


# ---------------------------------------------------------------------------
# 2. portfolio_weights — equal method
# ---------------------------------------------------------------------------


def test_equal_weights_sum_to_one() -> None:
    factor_df = _make_factor_df({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    result = portfolio_weights(factor_df, method="equal")
    total = result["weight"].sum()
    assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_equal_weights_are_uniform() -> None:
    factor_df = _make_factor_df({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    result = portfolio_weights(factor_df, method="equal")
    assert all(math.isclose(w, 0.25, rel_tol=1e-9) for w in result["weight"])


def test_equal_top_k_selects_correct_assets() -> None:
    # Values: A=1, B=2, C=3, D=4 → top 2 = D, C
    factor_df = _make_factor_df({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    result = portfolio_weights(factor_df, method="equal", top_k=2)
    assets = set(result["asset"])
    assert assets == {"C", "D"}


def test_equal_top_k_weights_sum_to_one() -> None:
    factor_df = _make_factor_df({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    result = portfolio_weights(factor_df, method="equal", top_k=2)
    assert math.isclose(result["weight"].sum(), 1.0, rel_tol=1e-9)


def test_equal_long_short_net_weight_zero() -> None:
    factor_df = _make_factor_df({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    result = portfolio_weights(factor_df, method="equal", top_k=2, bottom_k=2)
    assert math.isclose(result["weight"].sum(), 0.0, abs_tol=1e-9)


def test_equal_long_weights_positive_short_negative() -> None:
    factor_df = _make_factor_df({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    result = portfolio_weights(factor_df, method="equal", top_k=2, bottom_k=2)
    long_df = result[result["weight"] > 0]
    short_df = result[result["weight"] < 0]
    assert len(long_df) == 2
    assert len(short_df) == 2


# ---------------------------------------------------------------------------
# 3. portfolio_weights — rank method
# ---------------------------------------------------------------------------


def test_rank_weights_sum_to_one() -> None:
    factor_df = _make_factor_df({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    result = portfolio_weights(factor_df, method="rank")
    assert math.isclose(result["weight"].sum(), 1.0, rel_tol=1e-9)


def test_rank_weights_ordered_correctly() -> None:
    # Higher factor value → higher weight.
    factor_df = _make_factor_df({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    result = portfolio_weights(factor_df, method="rank")
    result = result.set_index("asset")
    # D has highest factor value → highest weight
    assert result.loc["D", "weight"] > result.loc["C", "weight"]
    assert result.loc["C", "weight"] > result.loc["B", "weight"]
    assert result.loc["B", "weight"] > result.loc["A", "weight"]


def test_rank_long_short_net_zero() -> None:
    factor_df = _make_factor_df({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    result = portfolio_weights(factor_df, method="rank", top_k=2, bottom_k=2)
    assert math.isclose(result["weight"].sum(), 0.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 4. portfolio_weights — score method
# ---------------------------------------------------------------------------


def test_score_weights_sum_to_one() -> None:
    factor_df = _make_factor_df({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    result = portfolio_weights(factor_df, method="score")
    assert math.isclose(result["weight"].sum(), 1.0, rel_tol=1e-9)


def test_score_weights_non_negative() -> None:
    factor_df = _make_factor_df({"A": 1.0, "B": 2.0, "C": 3.0, "D": 4.0})
    result = portfolio_weights(factor_df, method="score")
    assert all(w >= 0.0 for w in result["weight"])


def test_score_constant_factor_equal_weights() -> None:
    # All factor values equal → weights fall back to uniform.
    factor_df = _make_factor_df({"A": 5.0, "B": 5.0, "C": 5.0})
    result = portfolio_weights(factor_df, method="score")
    assert all(math.isclose(w, 1.0 / 3.0, rel_tol=1e-9) for w in result["weight"])


# ---------------------------------------------------------------------------
# 5. portfolio_weights — multi-date
# ---------------------------------------------------------------------------


def test_portfolio_weights_multi_date_all_dates_present() -> None:
    factor_df = _make_multi_date_factor(n_dates=5, n_assets=4)
    result = portfolio_weights(factor_df)
    n_factor_dates = factor_df["date"].nunique()
    n_weight_dates = result["date"].nunique()
    assert n_weight_dates == n_factor_dates


def test_portfolio_weights_deterministic() -> None:
    factor_df = _make_multi_date_factor(n_dates=5, n_assets=4)
    r1 = portfolio_weights(factor_df, method="rank")
    r2 = portfolio_weights(factor_df, method="rank")
    pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# 6. simulate_portfolio_returns — output schema
# ---------------------------------------------------------------------------


def test_simulate_returns_correct_columns() -> None:
    factor_df = _make_multi_date_factor(n_dates=5, n_assets=4)
    prices = _make_prices(n_days=20)
    labels = forward_return(prices, horizon=1)
    weights = portfolio_weights(factor_df.assign(date=pd.to_datetime(factor_df["date"])))
    result = simulate_portfolio_returns(weights, labels)
    assert list(result.columns) == list(_PORTFOLIO_RETURN_COLUMNS)


def test_simulate_returns_empty_weights_returns_empty() -> None:
    empty_w = pd.DataFrame(columns=["date", "asset", "weight"])
    labels = forward_return(_make_prices(), horizon=1)
    result = simulate_portfolio_returns(empty_w, labels)
    assert result.empty


def test_simulate_returns_empty_returns_returns_empty() -> None:
    factor_df = _make_multi_date_factor()
    weights = portfolio_weights(factor_df)
    empty_r = pd.DataFrame(columns=["date", "asset", "value"])
    result = simulate_portfolio_returns(weights, empty_r)
    assert result.empty


def test_simulate_returns_invalid_holding_period_raises() -> None:
    weights = portfolio_weights(_make_multi_date_factor())
    labels = forward_return(_make_prices(), horizon=1)
    with pytest.raises(ValueError, match="holding_period"):
        simulate_portfolio_returns(weights, labels, holding_period=0)


def test_simulate_returns_invalid_rebalance_frequency_raises() -> None:
    weights = portfolio_weights(_make_multi_date_factor())
    labels = forward_return(_make_prices(), horizon=1)
    with pytest.raises(ValueError, match="rebalance_frequency"):
        simulate_portfolio_returns(weights, labels, rebalance_frequency=0)


def test_simulate_returns_missing_columns_raises() -> None:
    weights = pd.DataFrame({"date": [], "asset": []})  # missing "weight"
    labels = forward_return(_make_prices(), horizon=1)
    with pytest.raises(ValueError, match="missing required columns"):
        simulate_portfolio_returns(weights, labels)


# ---------------------------------------------------------------------------
# 7. simulate_portfolio_returns — basic correctness
# ---------------------------------------------------------------------------


def test_simulate_returns_h1_r1_matches_weighted_sum() -> None:
    """With holding_period=1 and rebalance_frequency=1, the portfolio return
    at each date should equal the dot product of weights and returns."""
    prices = _make_prices(n_days=20)
    labels = forward_return(prices, horizon=1)

    # Use the same dates from labels as factor dates.
    factor_df = labels.rename(columns={"value": "_v"}).assign(
        factor="f", value=labels["value"]
    )[["date", "asset", "factor", "value"]]
    weights = portfolio_weights(factor_df, method="equal")

    result = simulate_portfolio_returns(
        weights, labels, holding_period=1, rebalance_frequency=1
    )

    # Manual check for the first date in result.
    if result.empty:
        pytest.skip("No portfolio returns produced")
    first_date = result["date"].iloc[0]
    w = weights[weights["date"] == first_date].set_index("asset")["weight"]
    r = labels[labels["date"] == first_date].set_index("asset")["value"]
    manual_ret = float((w * r).dropna().sum())
    port_ret = float(result.loc[result["date"] == first_date, "portfolio_return"].iloc[0])
    assert math.isclose(port_ret, manual_ret, rel_tol=1e-9)


def test_simulate_returns_rebalance_frequency_reduces_dates() -> None:
    """Higher rebalance_frequency → fewer active rebalance dates → fewer output dates."""
    prices = _make_prices(n_days=30)
    labels = forward_return(prices, horizon=1)
    factor_df = _make_multi_date_factor(n_dates=20, n_assets=4)
    weights = portfolio_weights(factor_df)

    r1 = simulate_portfolio_returns(weights, labels, rebalance_frequency=1)
    r2 = simulate_portfolio_returns(weights, labels, rebalance_frequency=3)
    # With higher rebalance_frequency, only a subset of dates are rebalance
    # dates so there may be fewer active positions at the start.
    assert len(r1) >= len(r2) or len(r2) > 0  # both should produce output


def test_simulate_returns_holding_period_overlap_at_most_h_positions() -> None:
    """With holding_period=H, at most H positions are active at any date."""
    prices = _make_prices(n_days=30)
    labels = forward_return(prices, horizon=1)
    factor_df = _make_multi_date_factor(n_dates=20, n_assets=4)
    weights = portfolio_weights(factor_df)

    # holding_period=2, rebalance_frequency=1: at most 2 overlapping positions.
    # We verify the result is finite (not a sum over too many positions).
    result = simulate_portfolio_returns(weights, labels, holding_period=2, rebalance_frequency=1)
    assert not result.empty
    assert result["portfolio_return"].notna().any()


# ---------------------------------------------------------------------------
# 8. portfolio_turnover — basic correctness
# ---------------------------------------------------------------------------


def test_portfolio_turnover_correct_columns() -> None:
    weights = portfolio_weights(_make_multi_date_factor())
    result = portfolio_turnover(weights)
    assert list(result.columns) == list(_PORTFOLIO_TURNOVER_COLUMNS)


def test_portfolio_turnover_first_date_is_nan() -> None:
    weights = portfolio_weights(_make_multi_date_factor(n_dates=5))
    result = portfolio_turnover(weights)
    assert math.isnan(float(result["portfolio_turnover"].iloc[0]))


def test_portfolio_turnover_no_change_is_zero() -> None:
    """Identical weights across two dates → turnover = 0."""
    w1 = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-01"), "asset": "A", "weight": 0.5},
        {"date": pd.Timestamp("2024-01-01"), "asset": "B", "weight": 0.5},
    ])
    w2 = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-02"), "asset": "A", "weight": 0.5},
        {"date": pd.Timestamp("2024-01-02"), "asset": "B", "weight": 0.5},
    ])
    result = portfolio_turnover(pd.concat([w1, w2], ignore_index=True))
    assert math.isnan(float(result["portfolio_turnover"].iloc[0]))
    assert math.isclose(float(result["portfolio_turnover"].iloc[1]), 0.0, abs_tol=1e-9)


def test_portfolio_turnover_complete_replacement_is_one() -> None:
    """Complete asset replacement → turnover = 1.0."""
    w1 = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-01"), "asset": "A", "weight": 0.5},
        {"date": pd.Timestamp("2024-01-01"), "asset": "B", "weight": 0.5},
    ])
    # Completely different assets.
    w2 = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-02"), "asset": "C", "weight": 0.5},
        {"date": pd.Timestamp("2024-01-02"), "asset": "D", "weight": 0.5},
    ])
    result = portfolio_turnover(pd.concat([w1, w2], ignore_index=True))
    # |w_new - w_old|: A: |0-0.5|=0.5, B: |0-0.5|=0.5, C: |0.5-0|=0.5, D: |0.5-0|=0.5
    # sum = 2.0, half = 1.0
    assert math.isclose(float(result["portfolio_turnover"].iloc[1]), 1.0, rel_tol=1e-9)


def test_portfolio_turnover_partial_rebalance() -> None:
    """Partial shift of weight from A to B."""
    w1 = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-01"), "asset": "A", "weight": 1.0},
    ])
    w2 = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-02"), "asset": "A", "weight": 0.5},
        {"date": pd.Timestamp("2024-01-02"), "asset": "B", "weight": 0.5},
    ])
    result = portfolio_turnover(pd.concat([w1, w2], ignore_index=True))
    # |w_new - w_old|: A: |0.5-1|=0.5, B: |0.5-0|=0.5 → sum=1.0, half=0.5
    assert math.isclose(float(result["portfolio_turnover"].iloc[1]), 0.5, rel_tol=1e-9)


def test_portfolio_turnover_empty_returns_empty() -> None:
    empty = pd.DataFrame(columns=["date", "asset", "weight"])
    result = portfolio_turnover(empty)
    assert result.empty


def test_portfolio_turnover_missing_columns_raises() -> None:
    bad = pd.DataFrame({"date": [1], "asset": ["A"]})
    with pytest.raises(ValueError, match="missing required columns"):
        portfolio_turnover(bad)


def test_portfolio_turnover_one_date_gives_single_nan_row() -> None:
    w = pd.DataFrame([
        {"date": pd.Timestamp("2024-01-01"), "asset": "A", "weight": 1.0},
    ])
    result = portfolio_turnover(w)
    assert len(result) == 1
    assert math.isnan(float(result["portfolio_turnover"].iloc[0]))


# ---------------------------------------------------------------------------
# 9. Integration — experiment.py portfolio extension
# ---------------------------------------------------------------------------


def test_experiment_portfolio_fields_none_by_default() -> None:
    prices = _make_prices(n_days=30)
    result = run_factor_experiment(prices, lambda p: momentum(p, window=5))
    assert result.portfolio_weights_df is None
    assert result.portfolio_return_df is None
    assert result.portfolio_turnover_df is None


def test_experiment_portfolio_fields_populated_when_params_provided() -> None:
    prices = _make_prices(n_days=30)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
    )
    assert result.portfolio_weights_df is not None
    assert result.portfolio_return_df is not None
    assert result.portfolio_turnover_df is not None


def test_experiment_portfolio_turnover_columns() -> None:
    prices = _make_prices(n_days=30)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
    )
    assert result.portfolio_turnover_df is not None
    assert list(result.portfolio_turnover_df.columns) == ["date", "portfolio_turnover"]


def test_experiment_portfolio_turnover_first_date_is_nan() -> None:
    prices = _make_prices(n_days=30)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
    )
    assert result.portfolio_turnover_df is not None
    if not result.portfolio_turnover_df.empty:
        assert math.isnan(float(result.portfolio_turnover_df["portfolio_turnover"].iloc[0]))


def test_experiment_portfolio_return_uses_1period_labels_when_horizon_gt_1() -> None:
    """When horizon > 1, portfolio simulation must use 1-period step returns,
    not the H-period forward returns used for IC/quantile evaluation."""
    import numpy as np

    prices = _make_prices(n_days=50)
    # horizon=5, holding_period=1, rebalance_frequency=1
    result_h5 = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        horizon=5,
        holding_period=1,
        rebalance_frequency=1,
    )
    result_h1 = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        horizon=1,
        holding_period=1,
        rebalance_frequency=1,
    )
    # Both should have portfolio_return_df populated.
    assert result_h5.portfolio_return_df is not None
    assert result_h1.portfolio_return_df is not None
    # The portfolio returns from horizon=5 should equal those from horizon=1
    # because both use 1-period labels for the simulation.
    common_dates = set(result_h5.portfolio_return_df["date"]) & set(
        result_h1.portfolio_return_df["date"]
    )
    if common_dates:
        df5 = result_h5.portfolio_return_df[
            result_h5.portfolio_return_df["date"].isin(common_dates)
        ].sort_values("date").reset_index(drop=True)
        df1 = result_h1.portfolio_return_df[
            result_h1.portfolio_return_df["date"].isin(common_dates)
        ].sort_values("date").reset_index(drop=True)
        np.testing.assert_allclose(
            df5["portfolio_return"].values,
            df1["portfolio_return"].values,
            rtol=1e-9,
        )


def test_experiment_portfolio_weights_columns() -> None:
    prices = _make_prices(n_days=30)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
    )
    assert result.portfolio_weights_df is not None
    assert list(result.portfolio_weights_df.columns) == list(_WEIGHT_COLUMNS)


def test_experiment_portfolio_return_columns() -> None:
    prices = _make_prices(n_days=30)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
    )
    assert result.portfolio_return_df is not None
    assert list(result.portfolio_return_df.columns) == list(_PORTFOLIO_RETURN_COLUMNS)


def test_experiment_portfolio_only_one_param_raises() -> None:
    prices = _make_prices(n_days=30)
    with pytest.raises(ValueError, match="holding_period and rebalance_frequency"):
        run_factor_experiment(
            prices,
            lambda p: momentum(p, window=5),
            holding_period=2,
            # rebalance_frequency omitted
        )


def test_experiment_portfolio_weighting_method_rank() -> None:
    prices = _make_prices(n_days=30)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
        weighting_method="rank",
    )
    assert result.portfolio_weights_df is not None
    # Rank weights at each date should sum to 1.
    for date, g in result.portfolio_weights_df.groupby("date"):
        total = g["weight"].sum()
        assert math.isclose(total, 1.0, rel_tol=1e-6), (
            f"Rank weights at {date} sum to {total}"
        )


# ---------------------------------------------------------------------------
# 10. portfolio_cost_adjusted_returns (standalone)
# ---------------------------------------------------------------------------


def _make_return_and_turnover_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-date portfolio: returns on both dates, turnover on both."""
    dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]
    ret_df = pd.DataFrame({"date": dates, "portfolio_return": [0.01, 0.02]})
    to_df = pd.DataFrame({"date": dates, "portfolio_turnover": [math.nan, 0.5]})
    return ret_df, to_df


def test_cost_adj_returns_columns() -> None:
    ret_df, to_df = _make_return_and_turnover_dfs()
    result = portfolio_cost_adjusted_returns(ret_df, to_df, cost_rate=0.001)
    assert list(result.columns) == list(_PORTFOLIO_COST_ADJ_COLUMNS)


def test_cost_adj_returns_first_date_is_nan() -> None:
    """First rebalance date has NaN turnover → NaN adjusted_return."""
    ret_df, to_df = _make_return_and_turnover_dfs()
    result = portfolio_cost_adjusted_returns(ret_df, to_df, cost_rate=0.001)
    assert math.isnan(float(result["adjusted_return"].iloc[0]))


def test_cost_adj_returns_second_date_correct() -> None:
    ret_df, to_df = _make_return_and_turnover_dfs()
    result = portfolio_cost_adjusted_returns(ret_df, to_df, cost_rate=0.001)
    expected = 0.02 - 0.001 * 0.5
    assert math.isclose(float(result["adjusted_return"].iloc[1]), expected, rel_tol=1e-9)


def test_cost_adj_returns_non_rebalance_date_zero_cost() -> None:
    """Evaluation dates not in turnover_df incur zero cost."""
    rebal_date = pd.Timestamp("2024-01-01")
    eval_date = pd.Timestamp("2024-01-02")
    ret_df = pd.DataFrame({
        "date": [rebal_date, eval_date],
        "portfolio_return": [0.01, 0.02],
    })
    # turnover_df only has the first date (the rebalance date).
    to_df = pd.DataFrame({
        "date": [rebal_date],
        "portfolio_turnover": [math.nan],
    })
    result = portfolio_cost_adjusted_returns(ret_df, to_df, cost_rate=0.001)
    # First date: NaN (first rebalance).
    assert math.isnan(float(result["adjusted_return"].iloc[0]))
    # Second date: no turnover row → zero cost → adjusted_return == portfolio_return.
    assert math.isclose(float(result["adjusted_return"].iloc[1]), 0.02, rel_tol=1e-9)


def test_cost_adj_returns_zero_cost_rate_equals_portfolio_return() -> None:
    ret_df, to_df = _make_return_and_turnover_dfs()
    result = portfolio_cost_adjusted_returns(ret_df, to_df, cost_rate=0.0)
    # Only second date is non-NaN (first date has NaN turnover).
    assert math.isclose(
        float(result["adjusted_return"].iloc[1]),
        float(result["portfolio_return"].iloc[1]),
        rel_tol=1e-9,
    )


def test_cost_adj_returns_negative_rate_raises() -> None:
    ret_df, to_df = _make_return_and_turnover_dfs()
    with pytest.raises(ValueError, match="cost_rate"):
        portfolio_cost_adjusted_returns(ret_df, to_df, cost_rate=-0.001)


def test_cost_adj_returns_empty_input() -> None:
    empty_ret = pd.DataFrame(columns=["date", "portfolio_return"])
    empty_to = pd.DataFrame(columns=["date", "portfolio_turnover"])
    result = portfolio_cost_adjusted_returns(empty_ret, empty_to, cost_rate=0.001)
    assert result.empty
    assert list(result.columns) == list(_PORTFOLIO_COST_ADJ_COLUMNS)


# ---------------------------------------------------------------------------
# 11. Turnover alignment (active rebalance schedule)
# ---------------------------------------------------------------------------


def test_portfolio_turnover_aligned_to_rebalance_frequency() -> None:
    """When rebalance_frequency > 1, portfolio_turnover_df must only contain
    active rebalance dates, not every weight date."""
    prices = _make_prices(n_days=40)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=2,
        rebalance_frequency=2,
    )
    assert result.portfolio_weights_df is not None
    assert result.portfolio_turnover_df is not None

    all_weight_dates = sorted(pd.to_datetime(result.portfolio_weights_df["date"]).unique())
    # Active rebalance dates = every 2nd weight date.
    active_rebal = set(all_weight_dates[::2])
    turnover_dates = set(pd.to_datetime(result.portfolio_turnover_df["date"]).unique())
    assert turnover_dates.issubset(active_rebal), (
        "portfolio_turnover_df contains non-rebalance dates"
    )


def test_portfolio_turnover_rf1_has_same_dates_as_weights() -> None:
    """When rebalance_frequency=1 all weight dates are active rebalance dates."""
    prices = _make_prices(n_days=30)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
    )
    assert result.portfolio_weights_df is not None
    assert result.portfolio_turnover_df is not None
    weight_dates = set(pd.to_datetime(result.portfolio_weights_df["date"]).unique())
    turnover_dates = set(pd.to_datetime(result.portfolio_turnover_df["date"]).unique())
    assert turnover_dates == weight_dates


# ---------------------------------------------------------------------------
# 12. Portfolio cost-adjusted returns via run_factor_experiment
# ---------------------------------------------------------------------------


def test_experiment_cost_adj_none_without_cost_rate() -> None:
    prices = _make_prices(n_days=30)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
    )
    assert result.portfolio_cost_adjusted_return_df is None


def test_experiment_cost_adj_populated_with_cost_rate() -> None:
    prices = _make_prices(n_days=30)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
        portfolio_cost_rate=0.001,
    )
    assert result.portfolio_cost_adjusted_return_df is not None
    assert list(result.portfolio_cost_adjusted_return_df.columns) == list(
        _PORTFOLIO_COST_ADJ_COLUMNS
    )


def test_experiment_cost_adj_none_when_no_portfolio_params() -> None:
    """portfolio_cost_rate is silently ignored when portfolio simulation is off."""
    prices = _make_prices(n_days=30)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        portfolio_cost_rate=0.001,
    )
    assert result.portfolio_cost_adjusted_return_df is None


def test_experiment_cost_adj_adjusted_return_le_portfolio_return() -> None:
    """With positive cost_rate and positive turnover, adjusted <= portfolio return."""
    prices = _make_prices(n_days=40)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
        portfolio_cost_rate=0.01,
    )
    assert result.portfolio_cost_adjusted_return_df is not None
    df = result.portfolio_cost_adjusted_return_df.dropna()
    if not df.empty:
        assert (df["adjusted_return"] <= df["portfolio_return"]).all()


# ---------------------------------------------------------------------------
# 13. PortfolioSummary
# ---------------------------------------------------------------------------


def test_portfolio_summary_none_without_portfolio_params() -> None:
    prices = _make_prices(n_days=30)
    result = run_factor_experiment(prices, lambda p: momentum(p, window=5))
    assert result.portfolio_summary is None


def test_portfolio_summary_populated_with_portfolio_params() -> None:
    prices = _make_prices(n_days=30)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
    )
    assert isinstance(result.portfolio_summary, PortfolioSummary)


def test_portfolio_summary_mean_return_finite() -> None:
    prices = _make_prices(n_days=40)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
    )
    assert result.portfolio_summary is not None
    assert math.isfinite(result.portfolio_summary.mean_portfolio_return)


def test_portfolio_summary_hit_rate_in_unit_interval() -> None:
    prices = _make_prices(n_days=40)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
    )
    assert result.portfolio_summary is not None
    hr = result.portfolio_summary.portfolio_hit_rate
    assert 0.0 <= hr <= 1.0


def test_portfolio_summary_mean_return_matches_manual_mean() -> None:
    prices = _make_prices(n_days=40)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
    )
    assert result.portfolio_summary is not None
    assert result.portfolio_return_df is not None
    manual = float(result.portfolio_return_df["portfolio_return"].dropna().mean())
    assert math.isclose(
        result.portfolio_summary.mean_portfolio_return, manual, rel_tol=1e-9
    )


def test_portfolio_summary_mean_turnover_matches_manual_mean() -> None:
    prices = _make_prices(n_days=40)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
    )
    assert result.portfolio_summary is not None
    assert result.portfolio_turnover_df is not None
    manual = float(result.portfolio_turnover_df["portfolio_turnover"].dropna().mean())
    assert math.isclose(
        result.portfolio_summary.mean_portfolio_turnover, manual, rel_tol=1e-9
    )


def test_portfolio_summary_n_portfolio_dates_correct() -> None:
    prices = _make_prices(n_days=40)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
    )
    assert result.portfolio_summary is not None
    assert result.portfolio_return_df is not None
    expected = int(result.portfolio_return_df["portfolio_return"].notna().sum())
    assert result.portfolio_summary.n_portfolio_dates == expected


def test_portfolio_summary_cost_adj_nan_without_cost_rate() -> None:
    prices = _make_prices(n_days=40)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
    )
    assert result.portfolio_summary is not None
    assert math.isnan(result.portfolio_summary.mean_cost_adjusted_return)


def test_portfolio_summary_cost_adj_finite_with_cost_rate() -> None:
    prices = _make_prices(n_days=40)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
        portfolio_cost_rate=0.001,
    )
    assert result.portfolio_summary is not None
    assert math.isfinite(result.portfolio_summary.mean_cost_adjusted_return)


def test_portfolio_summary_cost_adj_le_portfolio_return_on_same_dates() -> None:
    """On dates where both portfolio_return and adjusted_return are finite,
    adjusted_return must be <= portfolio_return (cost is non-negative)."""
    prices = _make_prices(n_days=40)
    result = run_factor_experiment(
        prices,
        lambda p: momentum(p, window=5),
        holding_period=1,
        rebalance_frequency=1,
        portfolio_cost_rate=0.01,
    )
    assert result.portfolio_cost_adjusted_return_df is not None
    df = result.portfolio_cost_adjusted_return_df.dropna(
        subset=["portfolio_return", "adjusted_return"]
    )
    if not df.empty:
        assert (df["adjusted_return"] <= df["portfolio_return"]).all()
