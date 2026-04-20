from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.risk_model.covariance import (
    factor_model_covariance,
    ledoit_wolf_shrinkage,
    newey_west_covariance,
    sample_covariance,
)


def test_sample_covariance_matches_pandas_pairwise() -> None:
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, np.nan, 0.03, 0.01],
            "B": [0.00, 0.01, 0.01, np.nan, 0.02],
            "C": [0.03, 0.01, 0.02, 0.01, 0.00],
        }
    )
    expected = returns.cov(min_periods=2).to_numpy(dtype=float)
    actual = sample_covariance(returns, min_history=2)
    assert np.allclose(actual, expected, equal_nan=True)


def test_ledoit_wolf_covariance_is_positive_semidefinite() -> None:
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(rng.normal(0.0, 0.01, size=(180, 6)))
    cov = ledoit_wolf_shrinkage(returns)
    eigvals = np.linalg.eigvalsh(cov)
    assert float(eigvals.min()) >= -1e-10


@dataclass(frozen=True)
class _StubBarra:
    exposures: pd.DataFrame


def test_factor_model_covariance_matches_matrix_formula() -> None:
    exposures = pd.DataFrame(
        {
            "size": [1.0, 0.5, -0.5],
            "value": [0.2, -0.1, 0.3],
        }
    )
    factor_cov = np.array([[0.04, 0.01], [0.01, 0.09]], dtype=float)
    specific_var = np.array([0.01, 0.02, 0.03], dtype=float)

    actual = factor_model_covariance(
        barra_model=_StubBarra(exposures=exposures),
        factor_cov=factor_cov,
        specific_var=specific_var,
    )
    b_mat = exposures.to_numpy(dtype=float)
    expected = b_mat @ factor_cov @ b_mat.T + np.diag(specific_var)
    assert np.allclose(actual, expected)


def test_newey_west_covariance_is_symmetric() -> None:
    rng = np.random.default_rng(11)
    returns = pd.DataFrame(rng.normal(0.0, 0.01, size=(120, 4)))
    cov = newey_west_covariance(returns, max_lag=4)
    assert cov.shape == (4, 4)
    assert np.allclose(cov, cov.T, atol=1e-12)
