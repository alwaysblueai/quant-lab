"""Smoke coverage for :mod:`alpha_lab.numba_kernels`.

Numba is an optional dependency; tests focus on the always-available
introspection surface plus a parity check (numba vs numpy) gated on
availability so the suite runs cleanly with or without numba installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from alpha_lab import numba_kernels


def test_numba_enabled_is_bool() -> None:
    assert isinstance(numba_kernels.numba_enabled(), bool)


def test_warmup_returns_bool() -> None:
    # Warmup is a no-op when numba is unavailable; either way it must return bool.
    assert isinstance(numba_kernels.warmup_numba_kernels(), bool)


@pytest.mark.skipif(not numba_kernels.numba_enabled(), reason="numba not installed")
def test_cross_sectional_corr_by_group_parity_with_numpy_pearson() -> None:
    rng = np.random.default_rng(7)
    x = rng.standard_normal(20)
    y = 2.0 * x + 0.3 * rng.standard_normal(20)
    start_idx = np.array([0, 10], dtype=np.int64)
    end_idx = np.array([10, 20], dtype=np.int64)

    out = numba_kernels.cross_sectional_corr_by_group_numba(
        x, y, start_idx, end_idx, method_code=0
    )

    expected_first = float(np.corrcoef(x[0:10], y[0:10])[0, 1])
    expected_second = float(np.corrcoef(x[10:20], y[10:20])[0, 1])
    np.testing.assert_allclose(out[0], expected_first, atol=1e-10)
    np.testing.assert_allclose(out[1], expected_second, atol=1e-10)
