from __future__ import annotations

import math

import numpy as np
import pandas as pd

from alpha_lab.multiple_testing import (
    adjust_pvalues,
    apply_multiple_testing_to_trial_log,
    bonferroni_threshold,
    effective_trial_count,
    sidak_threshold,
)


def test_bonferroni_threshold_basic() -> None:
    assert math.isclose(bonferroni_threshold(0.05, 10), 0.005)


def test_sidak_threshold_basic() -> None:
    out = sidak_threshold(0.05, 10)
    assert 0 < out < 0.05


def test_effective_trial_count_bounds() -> None:
    ident = pd.DataFrame(np.eye(4))
    all_one = pd.DataFrame(np.ones((4, 4)))
    assert math.isclose(effective_trial_count(ident), 4.0)
    assert math.isclose(effective_trial_count(all_one), 1.0)


def test_adjust_pvalues_returns_expected_shapes() -> None:
    res = adjust_pvalues([0.001, 0.01, 0.2], alpha=0.05, method="bonferroni")
    assert len(res.adjusted_pvalues) == 3
    assert len(res.reject) == 3
    assert res.adjusted_alpha < 0.05


def test_apply_multiple_testing_to_trial_log_adds_columns() -> None:
    df = pd.DataFrame(
        {
            "trial_id": ["t1", "t2", "t3"],
            "p_value": [0.001, 0.04, 0.5],
        }
    )
    out = apply_multiple_testing_to_trial_log(df, method="sidak")
    assert "mt_adjusted_pvalue" in out.columns
    assert "mt_reject" in out.columns
    assert out["mt_method"].iloc[0] == "sidak"

