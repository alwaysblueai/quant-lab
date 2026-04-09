from __future__ import annotations

import math

import numpy as np

from alpha_lab.reporting.uncertainty import (
    compute_block_bootstrap_mean_ci,
    compute_bootstrap_mean_ci,
    compute_core_uncertainty,
    compute_mean_ci,
)
from alpha_lab.research_evaluation_config import UncertaintyConfig


def test_compute_mean_ci_returns_finite_interval_for_regular_sample() -> None:
    ci = compute_mean_ci([0.01, 0.02, 0.03, 0.04])
    assert math.isclose(ci.mean, 0.025)
    assert math.isfinite(ci.stderr)
    assert math.isfinite(ci.ci_lower)
    assert math.isfinite(ci.ci_upper)
    assert ci.ci_lower < ci.mean < ci.ci_upper


def test_compute_mean_ci_ignores_nan_values() -> None:
    ci = compute_mean_ci([0.01, float("nan"), 0.03, np.nan])
    assert ci.n_obs == 2
    assert math.isclose(ci.mean, 0.02)


def test_compute_mean_ci_unavailable_when_less_than_two_observations() -> None:
    ci = compute_mean_ci([0.01])
    assert ci.n_obs == 1
    assert math.isnan(ci.stderr)
    assert math.isnan(ci.ci_lower)
    assert math.isnan(ci.ci_upper)


def test_compute_core_uncertainty_sets_overlap_zero_flags() -> None:
    summary = compute_core_uncertainty(
        ic_values=[-0.01, 0.02, -0.01, 0.02],
        rank_ic_values=[0.01, 0.02, 0.01, 0.02],
        long_short_values=[-0.002, 0.003, -0.002, 0.003],
    )
    assert "ic_ci_overlaps_zero" in summary.uncertainty_flags
    assert "long_short_ci_overlaps_zero" in summary.uncertainty_flags


def test_compute_core_uncertainty_sets_unavailable_flags_for_small_samples() -> None:
    summary = compute_core_uncertainty(
        ic_values=[0.01],
        rank_ic_values=[],
        long_short_values=[0.002],
    )
    assert "ic_ci_unavailable" in summary.uncertainty_flags
    assert "rank_ic_ci_unavailable" in summary.uncertainty_flags
    assert "long_short_ci_unavailable" in summary.uncertainty_flags


def test_compute_bootstrap_mean_ci_is_deterministic_with_fixed_seed() -> None:
    values = [0.01, 0.02, 0.03, 0.04, 0.05]
    first = compute_bootstrap_mean_ci(
        values,
        confidence_level=0.95,
        n_resamples=300,
        random_seed=11,
    )
    second = compute_bootstrap_mean_ci(
        values,
        confidence_level=0.95,
        n_resamples=300,
        random_seed=11,
    )
    assert first.ci_lower == second.ci_lower
    assert first.ci_upper == second.ci_upper
    assert first.n_obs == second.n_obs


def test_compute_block_bootstrap_mean_ci_is_deterministic_with_fixed_seed() -> None:
    values = [0.01, 0.02, 0.03, 0.04, 0.05]
    first = compute_block_bootstrap_mean_ci(
        values,
        confidence_level=0.95,
        n_resamples=300,
        block_length=3,
        random_seed=11,
    )
    second = compute_block_bootstrap_mean_ci(
        values,
        confidence_level=0.95,
        n_resamples=300,
        block_length=3,
        random_seed=11,
    )
    assert first.ci_lower == second.ci_lower
    assert first.ci_upper == second.ci_upper
    assert first.n_obs == second.n_obs


def test_compute_block_bootstrap_mean_ci_handles_small_sample_and_large_block() -> None:
    one_obs = compute_block_bootstrap_mean_ci([0.01], block_length=8)
    assert one_obs.n_obs == 1
    assert math.isnan(one_obs.ci_lower)
    assert math.isnan(one_obs.ci_upper)

    small_sample = compute_block_bootstrap_mean_ci(
        [0.01, 0.02, 0.03],
        n_resamples=200,
        block_length=10,
        random_seed=9,
    )
    assert small_sample.n_obs == 3
    assert math.isfinite(small_sample.ci_lower)
    assert math.isfinite(small_sample.ci_upper)


def test_block_bootstrap_ci_is_wider_than_iid_bootstrap_for_serial_signal() -> None:
    regime_values = [0.006] * 20 + [-0.002] * 20 + [0.006] * 20 + [-0.002] * 20
    iid = compute_bootstrap_mean_ci(
        regime_values,
        confidence_level=0.95,
        n_resamples=500,
        random_seed=21,
    )
    block = compute_block_bootstrap_mean_ci(
        regime_values,
        confidence_level=0.95,
        n_resamples=500,
        block_length=10,
        random_seed=21,
    )
    assert block.half_width > iid.half_width


def test_compute_core_uncertainty_bootstrap_mode_emits_method_metadata() -> None:
    summary = compute_core_uncertainty(
        ic_values=[0.01, 0.02, 0.03, 0.02, 0.01],
        rank_ic_values=[0.01, 0.02, 0.03, 0.02, 0.01],
        long_short_values=[0.001, 0.002, 0.003, 0.002, 0.001],
        thresholds=UncertaintyConfig(
            method="bootstrap",
            bootstrap_resamples=250,
            bootstrap_confidence_level=0.90,
            bootstrap_random_seed=19,
        ),
    )
    assert summary.uncertainty_method == "bootstrap"
    assert summary.uncertainty_confidence_level == 0.90
    assert summary.uncertainty_bootstrap_resamples == 250


def test_compute_core_uncertainty_bootstrap_small_sample_preserves_unavailable_flags() -> None:
    summary = compute_core_uncertainty(
        ic_values=[0.01],
        rank_ic_values=[0.01],
        long_short_values=[0.001],
        thresholds=UncertaintyConfig(
            method="bootstrap",
            bootstrap_resamples=200,
            bootstrap_random_seed=3,
        ),
    )
    assert "ic_ci_unavailable" in summary.uncertainty_flags
    assert "rank_ic_ci_unavailable" in summary.uncertainty_flags
    assert "long_short_ci_unavailable" in summary.uncertainty_flags


def test_compute_core_uncertainty_block_bootstrap_mode_emits_method_metadata() -> None:
    summary = compute_core_uncertainty(
        ic_values=[0.01, 0.02, 0.03, 0.02, 0.01],
        rank_ic_values=[0.01, 0.02, 0.03, 0.02, 0.01],
        long_short_values=[0.001, 0.002, 0.003, 0.002, 0.001],
        thresholds=UncertaintyConfig(
            method="block_bootstrap",
            bootstrap_resamples=250,
            bootstrap_confidence_level=0.90,
            bootstrap_random_seed=19,
            block_bootstrap_block_length=4,
        ),
    )
    assert summary.uncertainty_method == "block_bootstrap"
    assert summary.uncertainty_confidence_level == 0.90
    assert summary.uncertainty_bootstrap_resamples == 250
    assert summary.uncertainty_bootstrap_block_length == 4


def test_compute_core_uncertainty_normal_mode_backward_compatible_default() -> None:
    baseline = compute_core_uncertainty(
        ic_values=[0.01, 0.02, 0.03, 0.04],
        rank_ic_values=[0.01, 0.02, 0.03, 0.04],
        long_short_values=[0.001, 0.002, 0.003, 0.004],
    )
    explicit = compute_core_uncertainty(
        ic_values=[0.01, 0.02, 0.03, 0.04],
        rank_ic_values=[0.01, 0.02, 0.03, 0.04],
        long_short_values=[0.001, 0.002, 0.003, 0.004],
        thresholds=UncertaintyConfig(method="normal"),
    )
    assert baseline.uncertainty_method == "normal"
    assert explicit.uncertainty_method == "normal"
    assert baseline.mean_ic_ci_lower == explicit.mean_ic_ci_lower
    assert baseline.mean_ic_ci_upper == explicit.mean_ic_ci_upper
    assert baseline.uncertainty_bootstrap_block_length is None
    assert explicit.uncertainty_bootstrap_block_length is None
