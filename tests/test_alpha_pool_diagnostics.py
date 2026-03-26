from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.alpha_pool_diagnostics import alpha_pool_diagnostics


def _alpha_returns(seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 200
    base = rng.normal(0, 1, size=n)
    a1 = base + rng.normal(0, 0.2, size=n)
    a2 = base + rng.normal(0, 0.2, size=n)
    a3 = rng.normal(0, 1, size=n)
    return pd.DataFrame({"a1": a1, "a2": a2, "a3": a3})


def test_alpha_pool_diagnostics_outputs_sections() -> None:
    out = alpha_pool_diagnostics(_alpha_returns(), cluster_threshold=0.7)
    assert out.correlation_matrix.shape == (3, 3)
    assert {"alpha_a", "alpha_b", "corr"}.issubset(out.pairwise.columns)
    assert "effective_breadth" in out.breadth_summary.columns
    assert {"cluster_id", "members"}.issubset(out.clusters.columns)


def test_effective_breadth_less_than_raw_count_when_correlated() -> None:
    out = alpha_pool_diagnostics(_alpha_returns(), cluster_threshold=0.7)
    neff = out.breadth_summary["effective_breadth"].iloc[0]
    assert neff < 3.0

