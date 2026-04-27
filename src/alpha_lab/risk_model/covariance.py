from __future__ import annotations

from typing import Protocol, cast

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


class BarraExposuresLike(Protocol):
    exposures: pd.DataFrame


def sample_covariance(returns_wide: pd.DataFrame, min_history: int = 60) -> np.ndarray:
    """Sample covariance matrix with pairwise NaN-aware estimation."""
    if min_history < 2:
        raise ValueError("min_history must be >= 2")

    values = _coerce_returns_wide(returns_wide)
    n_assets = values.shape[1]
    if n_assets == 0:
        return np.empty((0, 0), dtype=float)

    cov_df = values.cov(min_periods=min_history)
    cov_df = cov_df.reindex(index=values.columns, columns=values.columns)
    return cast(np.ndarray, cov_df.to_numpy(dtype=float))


def ledoit_wolf_shrinkage(returns_wide: pd.DataFrame) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance estimator."""
    values = _coerce_returns_wide(returns_wide)
    n_assets = values.shape[1]
    if n_assets == 0:
        return np.empty((0, 0), dtype=float)

    clean = values.dropna(axis=0, how="any")
    if len(clean) < 2:
        return np.full((n_assets, n_assets), np.nan, dtype=float)

    estimator = LedoitWolf().fit(clean.to_numpy(dtype=float))
    cov = np.asarray(estimator.covariance_, dtype=float)
    return cast(np.ndarray, 0.5 * (cov + cov.T))


def factor_model_covariance(
    barra_model: BarraExposuresLike,
    factor_cov: np.ndarray,
    specific_var: np.ndarray,
) -> np.ndarray:
    """Factor-model covariance: ``Σ = B F B' + D``."""
    exposures = getattr(barra_model, "exposures", None)
    if not isinstance(exposures, pd.DataFrame):
        raise TypeError("barra_model must expose a pandas DataFrame `exposures` field")

    numeric = exposures.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("barra exposures must contain numeric factor columns")

    b_mat = numeric.to_numpy(dtype=float)
    n_assets, n_factors = b_mat.shape

    f_mat = np.asarray(factor_cov, dtype=float)
    if f_mat.ndim != 2 or f_mat.shape[0] != f_mat.shape[1]:
        raise ValueError("factor_cov must be a square matrix")
    if f_mat.shape[0] != n_factors:
        raise ValueError(
            f"factor_cov dimension {f_mat.shape[0]} does not match exposures {n_factors}"
        )

    specific = np.asarray(specific_var, dtype=float)
    if specific.ndim == 1:
        if len(specific) != n_assets:
            raise ValueError(
                f"specific_var length {len(specific)} does not match assets {n_assets}"
            )
        d_mat = np.diag(specific)
    elif specific.ndim == 2:
        if specific.shape != (n_assets, n_assets):
            raise ValueError("specific_var matrix must have shape (n_assets, n_assets)")
        d_mat = specific
    else:
        raise ValueError("specific_var must be a vector or square matrix")

    cov = b_mat @ f_mat @ b_mat.T + d_mat
    return cast(np.ndarray, 0.5 * (cov + cov.T))


def newey_west_covariance(returns_wide: pd.DataFrame, max_lag: int = 5) -> np.ndarray:
    """Newey-West HAC covariance for autocorrelated multivariate returns."""
    if max_lag < 0:
        raise ValueError("max_lag must be >= 0")

    values = _coerce_returns_wide(returns_wide)
    n_assets = values.shape[1]
    if n_assets == 0:
        return np.empty((0, 0), dtype=float)

    clean = values.dropna(axis=0, how="any")
    n_obs = len(clean)
    if n_obs < 2:
        return np.full((n_assets, n_assets), np.nan, dtype=float)

    x = clean.to_numpy(dtype=float)
    x = x - x.mean(axis=0, keepdims=True)

    lag_cap = min(int(max_lag), n_obs - 1)
    hac = (x.T @ x) / float(n_obs)

    for lag in range(1, lag_cap + 1):
        weight = 1.0 - lag / float(lag_cap + 1)
        gamma = (x[lag:].T @ x[:-lag]) / float(n_obs)
        hac = hac + weight * (gamma + gamma.T)

    return cast(np.ndarray, 0.5 * (hac + hac.T))


def _coerce_returns_wide(returns_wide: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(returns_wide, pd.DataFrame):
        raise TypeError("returns_wide must be a pandas DataFrame")
    if returns_wide.shape[1] == 0:
        return pd.DataFrame()
    out = returns_wide.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out
