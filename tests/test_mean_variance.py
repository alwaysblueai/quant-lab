from __future__ import annotations

import numpy as np
import pytest

from alpha_lab.optimization.mean_variance import (
    PortfolioConstraints,
    optimize_portfolio,
)

pytest.importorskip("cvxpy")


def test_mean_variance_equal_alpha_diagonal_cov_produces_near_equal_weights() -> None:
    alpha = np.ones(4, dtype=float)
    cov = np.eye(4, dtype=float)
    constraints = PortfolioConstraints(
        max_weight=1.0,
        min_weight=0.0,
        max_gross=1.0,
        max_net=1.0,
    )
    w = optimize_portfolio(alpha, cov, constraints, risk_aversion=1.0)

    assert w.shape == (4,)
    assert np.isclose(np.sum(np.abs(w)), 1.0, atol=1e-5)
    assert np.allclose(w, np.full(4, 0.25), atol=1e-3)


def test_mean_variance_zero_alpha_returns_zero_portfolio() -> None:
    alpha = np.zeros(5, dtype=float)
    cov = np.eye(5, dtype=float)
    constraints = PortfolioConstraints(
        max_weight=1.0,
        min_weight=-1.0,
        max_gross=1.0,
        max_net=1.0,
    )
    w = optimize_portfolio(alpha, cov, constraints, risk_aversion=1.0)
    assert np.linalg.norm(w, ord=1) <= 1e-6


def test_mean_variance_turnover_constraint_is_respected() -> None:
    alpha = np.array([0.3, 0.2, 0.1, 0.0], dtype=float)
    cov = np.eye(4, dtype=float) * 0.05
    prev = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)
    constraints = PortfolioConstraints(
        max_weight=1.0,
        min_weight=0.0,
        max_gross=1.0,
        max_net=1.0,
        max_turnover=0.10,
    )
    w = optimize_portfolio(alpha, cov, constraints, risk_aversion=0.2, prev_weights=prev)
    assert np.sum(np.abs(w - prev)) <= 0.1005
