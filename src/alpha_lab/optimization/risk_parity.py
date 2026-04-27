from __future__ import annotations

from typing import cast

import numpy as np
from scipy.optimize import minimize

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabExperimentError


def risk_parity_weights(
    cov: np.ndarray,
    target_risk_budget: np.ndarray | None = None,
) -> np.ndarray:
    """Solve long-only risk-parity weights."""
    cov_mat = np.asarray(cov, dtype=float)
    if cov_mat.ndim != 2 or cov_mat.shape[0] != cov_mat.shape[1]:
        raise AlphaLabConfigError("cov must be a square matrix")

    n_assets = cov_mat.shape[0]
    if n_assets == 0:
        raise AlphaLabConfigError("cov must be non-empty")

    cov_mat = 0.5 * (cov_mat + cov_mat.T)
    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

    if target_risk_budget is None:
        budget = np.full(n_assets, 1.0 / n_assets, dtype=float)
    else:
        budget = np.asarray(target_risk_budget, dtype=float).reshape(-1)
        if budget.shape != (n_assets,):
            raise AlphaLabConfigError(
                f"target_risk_budget must have shape ({n_assets},), got {budget.shape}"
            )
        if np.any(budget < 0.0):
            raise AlphaLabConfigError("target_risk_budget must be non-negative")
        total = float(budget.sum())
        if total <= 0.0:
            raise AlphaLabConfigError("target_risk_budget must sum to a positive value")
        budget = budget / total

    def objective(weights: np.ndarray) -> float:
        rc = _risk_contributions(weights, cov_psd)
        if not np.all(np.isfinite(rc)):
            return float("inf")
        return float(np.sum((rc - budget) ** 2))

    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0},)
    bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    x0 = budget.copy()
    x0 = x0 / float(x0.sum())

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not result.success or result.x is None:
        raise AlphaLabExperimentError(f"risk parity optimization failed: {result.message}")

    weights = np.clip(np.asarray(result.x, dtype=float), 0.0, None)
    total = float(weights.sum())
    if total <= 0.0:
        raise AlphaLabExperimentError("risk parity optimization returned zero weights")
    return cast(np.ndarray, weights / total)


def _risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    marginal = cov @ weights
    variance = float(weights @ marginal)
    if variance <= 0.0 or not np.isfinite(variance):
        return np.full_like(weights, np.nan, dtype=float)
    return cast(np.ndarray, weights * marginal / variance)
