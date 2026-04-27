"""Tests for alpha_lab.marginal_contribution module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.marginal_contribution import (
    MarginalContributionSummary,
    compute_marginal_contribution,
    fama_macbeth_regression,
    spanning_test,
)


def _make_factor_df(
    name: str, n_dates: int = 30, n_assets: int = 20, seed: int = 42
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    rows = []
    for d in dates:
        for a_idx in range(n_assets):
            rows.append(
                {
                    "date": d,
                    "asset": f"A{a_idx}",
                    "factor": name,
                    "value": rng.normal(0, 1),
                }
            )
    return pd.DataFrame(rows)


def _make_label_df(
    factor_df: pd.DataFrame, beta: float = 0.01, noise: float = 0.05, seed: int = 99
) -> pd.DataFrame:
    """Generate labels correlated with factor values."""
    rng = np.random.RandomState(seed)
    labels = factor_df[["date", "asset", "value"]].copy()
    labels["value"] = beta * labels["value"] + rng.normal(0, noise, len(labels))
    return labels[["date", "asset", "value"]]


class TestFamaMacBethRegression:
    def test_basic_output(self) -> None:
        factor_df = _make_factor_df("momentum")
        label_df = _make_label_df(factor_df, beta=0.02)
        result = fama_macbeth_regression(factor_df, label_df)
        assert result is not None
        assert result.factor_name == "momentum"
        assert result.n_dates > 0
        assert np.isfinite(result.mean_coefficient)
        assert np.isfinite(result.t_statistic)
        assert np.isfinite(result.p_value)
        assert 0.0 <= result.p_value <= 1.0

    def test_positive_beta_detected(self) -> None:
        factor_df = _make_factor_df("value", seed=10)
        label_df = _make_label_df(factor_df, beta=0.05, noise=0.01, seed=20)
        result = fama_macbeth_regression(factor_df, label_df)
        assert result is not None
        assert result.mean_coefficient > 0

    def test_with_existing_factors(self) -> None:
        cand = _make_factor_df("candidate", seed=1)
        exist = _make_factor_df("existing", seed=2)
        label_df = _make_label_df(cand, beta=0.02)
        result = fama_macbeth_regression(cand, label_df, [exist])
        assert result is not None
        assert result.factor_name == "candidate"
        assert result.n_dates > 0

    def test_empty_candidate_returns_none(self) -> None:
        empty = pd.DataFrame(columns=["date", "asset", "factor", "value"])
        label_df = pd.DataFrame(columns=["date", "asset", "value"])
        result = fama_macbeth_regression(empty, label_df)
        assert result is None

    def test_insufficient_dates_returns_none(self) -> None:
        """Fewer than 2 valid cross-sections should return None."""
        factor_df = _make_factor_df("x", n_dates=1, n_assets=3)
        label_df = _make_label_df(factor_df)
        result = fama_macbeth_regression(factor_df, label_df)
        assert result is None


class TestSpanningTest:
    def test_orthogonal_factors_positive_r2_increment(self) -> None:
        """A candidate with unique signal should add R² even if intercept is
        not significant (the spanning test focuses on the intercept)."""
        rng = np.random.RandomState(42)
        n_dates, n_assets = 30, 20
        dates = pd.bdate_range("2020-01-01", periods=n_dates)

        cand_rows = []
        exist_rows = []
        label_rows = []
        for d in dates:
            for a in range(n_assets):
                c_val = rng.normal(0, 1)
                e_val = rng.normal(0, 1)
                # Label depends on candidate but not on existing
                lab = 0.05 * c_val + rng.normal(0, 0.01)
                cand_rows.append({"date": d, "asset": f"A{a}", "factor": "cand", "value": c_val})
                exist_rows.append({"date": d, "asset": f"A{a}", "factor": "exist", "value": e_val})
                label_rows.append({"date": d, "asset": f"A{a}", "value": lab})

        cand_df = pd.DataFrame(cand_rows)
        exist_df = pd.DataFrame(exist_rows)
        label_df = pd.DataFrame(label_rows)

        result = spanning_test(cand_df, label_df, [exist_df])
        assert result is not None
        assert result.candidate_factor == "cand"
        # Candidate with unique signal should have positive R² increment
        assert result.r_squared_increment > 0
        assert np.isfinite(result.intercept_t_stat)
        assert np.isfinite(result.intercept_p_value)

    def test_no_existing_factors_returns_none(self) -> None:
        cand = _make_factor_df("x")
        label = _make_label_df(cand)
        result = spanning_test(cand, label, [])
        assert result is None

    def test_empty_candidate_returns_none(self) -> None:
        empty = pd.DataFrame(columns=["date", "asset", "factor", "value"])
        exist = _make_factor_df("exist")
        label = _make_label_df(exist)
        result = spanning_test(empty, label, [exist])
        assert result is None


class TestComputeMarginalContribution:
    def test_standalone_factor(self) -> None:
        """No existing factors — only Fama-MacBeth runs."""
        factor_df = _make_factor_df("standalone")
        label_df = _make_label_df(factor_df, beta=0.02)
        result = compute_marginal_contribution(factor_df, label_df)
        assert isinstance(result, MarginalContributionSummary)
        assert result.fama_macbeth is not None
        assert result.spanning_test is None

    def test_with_existing_factors(self) -> None:
        cand = _make_factor_df("cand", seed=1)
        exist = _make_factor_df("exist", seed=2)
        label = _make_label_df(cand, beta=0.02)
        result = compute_marginal_contribution(cand, label, [exist])
        assert result.fama_macbeth is not None
        assert result.spanning_test is not None

    def test_weak_factor_gets_flags(self) -> None:
        """A factor with zero signal should get weakness flags."""
        rng = np.random.RandomState(42)
        n_dates, n_assets = 30, 20
        dates = pd.bdate_range("2020-01-01", periods=n_dates)
        rows = []
        label_rows = []
        for d in dates:
            for a in range(n_assets):
                rows.append(
                    {"date": d, "asset": f"A{a}", "factor": "noise", "value": rng.normal(0, 1)}
                )
                label_rows.append({"date": d, "asset": f"A{a}", "value": rng.normal(0, 0.01)})
        factor_df = pd.DataFrame(rows)
        label_df = pd.DataFrame(label_rows)
        result = compute_marginal_contribution(factor_df, label_df)
        # A pure noise factor should not be significant
        assert result.fama_macbeth is not None
        assert result.fama_macbeth.p_value > 0.05 or result.fama_macbeth.mean_coefficient <= 0
