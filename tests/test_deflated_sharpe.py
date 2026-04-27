from __future__ import annotations

import math

import pytest

from alpha_lab.validation.deflated_sharpe import (
    deflated_sharpe_ratio,
    expected_max_sharpe,
)


def test_expected_max_sharpe_single_trial_is_zero() -> None:
    assert expected_max_sharpe(n_trials=1, n_obs=252) == pytest.approx(0.0)


def test_deflated_sharpe_ratio_sr_zero_single_trial_is_half() -> None:
    p = deflated_sharpe_ratio(
        observed_sr=0.0,
        n_trials=1,
        n_obs=252,
    )
    assert p == pytest.approx(0.5, abs=1e-12)


def test_deflated_sharpe_ratio_becomes_more_conservative_with_more_trials() -> None:
    p_single = deflated_sharpe_ratio(
        observed_sr=1.0,
        n_trials=1,
        n_obs=252,
    )
    p_many = deflated_sharpe_ratio(
        observed_sr=1.0,
        n_trials=50,
        n_obs=252,
    )
    assert p_many > p_single


def test_deflated_sharpe_ratio_respects_non_normality_inputs() -> None:
    p = deflated_sharpe_ratio(
        observed_sr=1.2,
        n_trials=10,
        n_obs=180,
        skewness=-0.2,
        excess_kurtosis=1.5,
    )
    assert 0.0 <= p <= 1.0


def test_deflated_sharpe_ratio_non_finite_sr_returns_nan() -> None:
    p = deflated_sharpe_ratio(
        observed_sr=float("nan"),
        n_trials=5,
        n_obs=120,
    )
    assert math.isnan(p)
