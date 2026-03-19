from __future__ import annotations

import math

import pandas as pd
import pytest

from alpha_lab.costs import apply_linear_cost, cost_adjusted_long_short

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_ls_df(returns: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(returns), freq="B")
    return pd.DataFrame({"date": dates, "factor": "f", "long_short_return": returns})


def _make_ls_turnover_df(turnovers: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(turnovers), freq="B")
    return pd.DataFrame(
        {"date": dates, "factor": "f", "long_short_turnover": turnovers}
    )


# ---------------------------------------------------------------------------
# 1. apply_linear_cost — basic correctness
# ---------------------------------------------------------------------------


def test_apply_linear_cost_basic():
    returns = pd.Series([0.02, 0.01, -0.005], index=range(3))
    turnover = pd.Series([0.5, 0.3, 0.8], index=range(3))
    adjusted = apply_linear_cost(returns, turnover, cost_rate=0.001)
    expected = pd.Series(
        [0.02 - 0.001 * 0.5, 0.01 - 0.001 * 0.3, -0.005 - 0.001 * 0.8],
        index=range(3),
    )
    pd.testing.assert_series_equal(adjusted, expected, rtol=1e-12)


def test_apply_linear_cost_zero_rate_unchanged():
    returns = pd.Series([0.05, -0.02, 0.01], index=range(3))
    turnover = pd.Series([0.4, 0.6, 0.2], index=range(3))
    adjusted = apply_linear_cost(returns, turnover, cost_rate=0.0)
    pd.testing.assert_series_equal(adjusted, returns)


def test_apply_linear_cost_full_turnover():
    """Turnover = 1.0 → cost = cost_rate × 1.0 = cost_rate."""
    returns = pd.Series([0.02], index=[0])
    turnover = pd.Series([1.0], index=[0])
    adjusted = apply_linear_cost(returns, turnover, cost_rate=0.005)
    assert float(adjusted.iloc[0]) == pytest.approx(0.02 - 0.005)


def test_apply_linear_cost_nan_return_propagates():
    returns = pd.Series([float("nan"), 0.01], index=range(2))
    turnover = pd.Series([0.5, 0.3], index=range(2))
    adjusted = apply_linear_cost(returns, turnover, cost_rate=0.001)
    assert math.isnan(float(adjusted.iloc[0]))
    assert not math.isnan(float(adjusted.iloc[1]))


def test_apply_linear_cost_nan_turnover_propagates():
    """NaN turnover (first date, no prior state) → NaN adjusted return."""
    returns = pd.Series([0.02, 0.01], index=range(2))
    turnover = pd.Series([float("nan"), 0.5], index=range(2))
    adjusted = apply_linear_cost(returns, turnover, cost_rate=0.001)
    assert math.isnan(float(adjusted.iloc[0]))
    assert not math.isnan(float(adjusted.iloc[1]))


def test_apply_linear_cost_returns_series():
    returns = pd.Series([0.01], index=[0])
    turnover = pd.Series([0.5], index=[0])
    result = apply_linear_cost(returns, turnover, cost_rate=0.001)
    assert isinstance(result, pd.Series)


# ---------------------------------------------------------------------------
# 2. apply_linear_cost — input validation
# ---------------------------------------------------------------------------


def test_apply_linear_cost_rejects_negative_cost_rate():
    returns = pd.Series([0.01], index=[0])
    turnover = pd.Series([0.5], index=[0])
    with pytest.raises(ValueError, match="cost_rate must be >= 0"):
        apply_linear_cost(returns, turnover, cost_rate=-0.001)


def test_apply_linear_cost_rejects_mismatched_index():
    returns = pd.Series([0.01, 0.02], index=[0, 1])
    turnover = pd.Series([0.5, 0.3], index=[1, 2])  # different index
    with pytest.raises(ValueError, match="same index"):
        apply_linear_cost(returns, turnover, cost_rate=0.001)


def test_apply_linear_cost_rejects_different_length():
    returns = pd.Series([0.01, 0.02], index=[0, 1])
    turnover = pd.Series([0.5], index=[0])
    with pytest.raises(ValueError, match="same index"):
        apply_linear_cost(returns, turnover, cost_rate=0.001)


# ---------------------------------------------------------------------------
# 3. cost_adjusted_long_short — basic correctness
# ---------------------------------------------------------------------------


def test_cost_adjusted_long_short_returns_dataframe():
    ls_df = _make_ls_df([0.02, 0.01, -0.005])
    ls_turn_df = _make_ls_turnover_df([float("nan"), 0.5, 0.3])
    result = cost_adjusted_long_short(ls_df, ls_turn_df, cost_rate=0.001)
    assert isinstance(result, pd.DataFrame)


def test_cost_adjusted_long_short_has_expected_columns():
    ls_df = _make_ls_df([0.02, 0.01])
    ls_turn_df = _make_ls_turnover_df([float("nan"), 0.5])
    result = cost_adjusted_long_short(ls_df, ls_turn_df, cost_rate=0.001)
    assert {"date", "factor", "long_short_return", "turnover", "adjusted_return"}.issubset(
        result.columns
    )


def test_cost_adjusted_long_short_correct_values():
    ls_df = _make_ls_df([0.02, 0.01, -0.005])
    ls_turn_df = _make_ls_turnover_df([float("nan"), 0.5, 0.3])
    result = cost_adjusted_long_short(ls_df, ls_turn_df, cost_rate=0.001)
    # row 0: turnover NaN → adjusted NaN
    assert math.isnan(float(result["adjusted_return"].iloc[0]))
    # row 1: 0.01 - 0.001 * 0.5 = 0.0095
    assert float(result["adjusted_return"].iloc[1]) == pytest.approx(0.0095)
    # row 2: -0.005 - 0.001 * 0.3 = -0.0053
    assert float(result["adjusted_return"].iloc[2]) == pytest.approx(-0.0053)


def test_cost_adjusted_long_short_zero_cost_equals_raw_return():
    ls_df = _make_ls_df([0.02, 0.01, -0.005])
    ls_turn_df = _make_ls_turnover_df([0.0, 0.5, 0.3])
    result = cost_adjusted_long_short(ls_df, ls_turn_df, cost_rate=0.0)
    pd.testing.assert_series_equal(
        result["adjusted_return"],
        result["long_short_return"],
        check_names=False,
    )


def test_cost_adjusted_long_short_preserves_raw_return():
    ls_df = _make_ls_df([0.02, 0.01])
    ls_turn_df = _make_ls_turnover_df([float("nan"), 0.5])
    result = cost_adjusted_long_short(ls_df, ls_turn_df, cost_rate=0.001)
    assert float(result["long_short_return"].iloc[1]) == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# 4. cost_adjusted_long_short — input validation
# ---------------------------------------------------------------------------


def test_cost_adjusted_long_short_rejects_negative_cost_rate():
    ls_df = _make_ls_df([0.01])
    ls_turn_df = _make_ls_turnover_df([0.5])
    with pytest.raises(ValueError, match="cost_rate must be >= 0"):
        cost_adjusted_long_short(ls_df, ls_turn_df, cost_rate=-0.001)


def test_cost_adjusted_long_short_empty_ls_df_returns_empty():
    ls_df = pd.DataFrame(columns=["date", "factor", "long_short_return"])
    ls_turn_df = _make_ls_turnover_df([0.5])
    result = cost_adjusted_long_short(ls_df, ls_turn_df, cost_rate=0.001)
    assert result.empty


def test_cost_adjusted_long_short_empty_turnover_df_returns_empty():
    ls_df = _make_ls_df([0.01])
    ls_turn_df = pd.DataFrame(columns=["date", "factor", "long_short_turnover"])
    result = cost_adjusted_long_short(ls_df, ls_turn_df, cost_rate=0.001)
    assert result.empty


def test_cost_adjusted_long_short_sorted_by_date():
    ls_df = _make_ls_df([0.03, 0.02, 0.01])
    ls_turn_df = _make_ls_turnover_df([float("nan"), 0.5, 0.3])
    result = cost_adjusted_long_short(ls_df, ls_turn_df, cost_rate=0.001)
    dates = list(result["date"])
    assert dates == sorted(dates)


# ---------------------------------------------------------------------------
# 5. cost_adjusted_long_short — integration with experiment runner
# ---------------------------------------------------------------------------


def test_cost_adjusted_from_experiment_result():
    """End-to-end: run an experiment then compute cost-adjusted returns."""
    import numpy as np

    from alpha_lab.experiment import run_factor_experiment
    from alpha_lab.factors.momentum import momentum

    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=25, freq="B")
    assets = ["A0", "A1", "A2", "A3", "A4", "A5"]
    rows = []
    for asset in assets:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": price})
    prices = pd.DataFrame(rows)

    result = run_factor_experiment(prices, lambda p: momentum(p, window=5))
    adj = cost_adjusted_long_short(
        result.long_short_df,
        result.long_short_turnover_df,
        cost_rate=0.001,
    )
    assert not adj.empty
    # Adjusted return must always be <= raw return when cost_rate > 0 and turnover >= 0
    valid = adj.dropna(subset=["adjusted_return", "long_short_return"])
    assert (valid["adjusted_return"] <= valid["long_short_return"] + 1e-12).all()
