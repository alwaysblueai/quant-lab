"""Pin the rate-convention asymmetry between the two cost helpers.

- :func:`apply_linear_cost` takes a **per-period** cost_rate and subtracts
  ``cost_rate × turnover(t)`` from the return at ``t`` with no calendar
  scaling. If returns are daily, ``cost_rate`` is per-day. If returns are
  weekly, ``cost_rate`` is per-week. No ``/252`` anywhere.
- :func:`apply_short_borrow_cost` takes an **annual** rate and internally
  divides by ``252`` (daily convention).

Mixing the two with inconsistent cadences silently double- or half-scales
the cost. These tests lock the convention so a future refactor can't
quietly change it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.costs import apply_linear_cost, apply_short_borrow_cost


def _series(values: list[float], start: str = "2024-01-02") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


def test_linear_cost_is_per_period_no_annualization():
    returns = _series([0.01, 0.02, -0.01])
    turnover = _series([0.5, 1.0, 0.25])
    cost_rate = 0.001  # 10 bps per one-way turnover, per period

    adjusted = apply_linear_cost(returns, turnover, cost_rate=cost_rate)
    expected = returns - cost_rate * turnover

    pd.testing.assert_series_equal(adjusted, expected, check_names=False)
    # Explicit: there is NO /252 scaling.
    wrong_if_annualized = returns - (cost_rate / 252.0) * turnover
    assert not np.allclose(adjusted, wrong_if_annualized)


def test_short_borrow_cost_annualized_divides_by_252():
    returns = _series([0.0, 0.0, 0.0])
    short_weights = _series([-0.5, -0.5, -0.5])  # 50% short book
    annual_rate = 0.08

    adjusted = apply_short_borrow_cost(returns, short_weights, annual_rate=annual_rate)
    expected_daily_cost = 0.5 * annual_rate / 252.0
    expected = returns - expected_daily_cost

    pd.testing.assert_series_equal(adjusted, expected, check_names=False)


def test_short_borrow_only_charges_short_leg():
    returns = _series([0.0, 0.0])
    weights_long_only = _series([0.5, 0.5])  # positive → no borrow cost

    adjusted = apply_short_borrow_cost(returns, weights_long_only, annual_rate=0.08)
    pd.testing.assert_series_equal(adjusted, returns, check_names=False)


def test_linear_cost_nan_turnover_nans_adjusted_return():
    returns = _series([0.01, 0.02])
    turnover = pd.Series([float("nan"), 0.5], index=returns.index, dtype=float)
    adjusted = apply_linear_cost(returns, turnover, cost_rate=0.001)
    assert np.isnan(adjusted.iloc[0])
    assert np.isclose(adjusted.iloc[1], 0.02 - 0.001 * 0.5)


def test_scaling_mismatch_is_visibly_large():
    """A cost_rate accidentally passed as an annual (e.g. 0.08) vs per-period
    (e.g. 0.0003 daily ~ 8% / 252) should produce adjusted returns that
    differ by orders of magnitude. This test protects downstream consumers
    from a silent unit mix-up."""

    returns = _series([0.0] * 10)
    turnover = _series([1.0] * 10)

    per_period = apply_linear_cost(returns, turnover, cost_rate=0.0003)
    annual_like = apply_linear_cost(returns, turnover, cost_rate=0.08)

    # Annual-like rate is ~250x larger than a reasonable per-period rate.
    ratio = float(annual_like.abs().mean() / per_period.abs().mean())
    assert ratio > 100.0
