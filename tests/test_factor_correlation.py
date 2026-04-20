from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.reporting.factor_correlation import compute_factor_correlation


def _make_series(values: list[float], *, start: str = "2020-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start=start, periods=len(values), freq="D"))


def test_compute_factor_correlation_identifies_redundant_candidates() -> None:
    candidate = _make_series([1.0, 0.2, 0.5, 0.9, 0.1, -0.2, 0.3, 0.7, -0.4, 0.0])
    base = candidate * 1.8 + np.random.RandomState(0).normal(0.0, 1e-4, size=len(candidate))
    orthogonal = _make_series([0.2, 0.1, -0.1, -0.2, 0.05, 0.0, 0.3, -0.05, 0.1, -0.15])

    result = compute_factor_correlation(
        candidate,
        {"base_factor": pd.Series(base, index=candidate.index), "orthogonal": orthogonal},
        candidate_name="candidate_alpha",
        redundancy_threshold=0.7,
    )

    assert result.candidate_name == "candidate_alpha"
    assert len(result.correlations) == 2
    assert result.correlations[0].factor_name == "base_factor"
    assert result.max_abs_correlation >= 0.99
    assert result.likely_redundant is True
    assert result.r_squared is not None
    assert result.r_squared >= 0.98


def test_compute_factor_correlation_requires_enough_points() -> None:
    candidate = _make_series([0.1, 0.2, 0.3, 0.4], start="2020-01-01")
    existing = {"short_series": _make_series([0.2, 0.4, 0.6, 0.8], start="2020-01-01")}

    result = compute_factor_correlation(candidate, existing, candidate_name="short")

    assert result.correlations == []
    assert result.max_abs_correlation == 0.0
    assert result.likely_redundant is False
    assert result.r_squared is None


def test_compute_factor_correlation_with_non_series_values_ignores_non_series() -> None:
    candidate = _make_series([1, 2, 3, 4, 5], start="2020-01-01")
    result = compute_factor_correlation(
        candidate,
        {"not_series": [1, 2, 3, 4, 5], "good": _make_series([1, 2, 3, 4, 5])},
        redundancy_threshold=0.5,
    )

    assert len(result.correlations) == 1
    assert result.correlations[0].factor_name == "good"


def test_empty_existing_factors() -> None:
    result = compute_factor_correlation(_make_series([0.1, 0.2, 0.3, 0.4, 0.5]), {})

    assert result.max_abs_correlation == 0.0
    assert result.r_squared is None
    assert result.likely_redundant is False


def test_too_few_observations() -> None:
    result = compute_factor_correlation(
        _make_series([0.1, 0.2]),
        {"short": _make_series([0.3, 0.4])},
    )

    assert result.correlations == []
    assert result.max_abs_correlation == 0.0
    assert result.likely_redundant is False


def test_r_squared_multivariate() -> None:
    import numpy as np

    np.random.seed(42)
    n = 50
    x1, x2 = np.random.randn(n), np.random.randn(n)
    y = 0.7 * x1 + 0.3 * x2 + 0.05 * np.random.randn(n)
    dates = pd.date_range("2020-01-01", periods=n, freq="D")

    result = compute_factor_correlation(
        pd.Series(y, index=dates),
        {"f1": pd.Series(x1, index=dates), "f2": pd.Series(x2, index=dates)},
        candidate_name="composite",
    )

    assert result.r_squared is not None
    assert result.r_squared > 0.9
