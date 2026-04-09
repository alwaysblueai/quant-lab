from __future__ import annotations

import pytest

from alpha_lab.reporting.neutralization_comparison import (
    EXPOSURE_DRIVEN_FLAG,
    MATERIAL_REDUCTION_FLAG,
    MODERATE_WEAKENING_FLAG,
    PRESERVES_EVIDENCE_FLAG,
    WEAKER_BUT_STABLER_FLAG,
    build_raw_vs_neutralized_comparison,
)


def test_raw_vs_neutralized_comparison_computes_core_deltas() -> None:
    comparison = build_raw_vs_neutralized_comparison(
        _metrics(
            mean_ic=0.04,
            mean_rank_ic=0.05,
            mean_long_short_return=0.0030,
            ic_ir=0.90,
            mean_ic_ci_lower=0.01,
            mean_ic_ci_upper=0.07,
            mean_rank_ic_ci_lower=0.02,
            mean_rank_ic_ci_upper=0.08,
            mean_long_short_return_ci_lower=0.0010,
            mean_long_short_return_ci_upper=0.0050,
            uncertainty_flags=[],
            rolling_instability_flags=["rolling_ic_sign_flip_instability"],
        ),
        _metrics(
            mean_ic=0.03,
            mean_rank_ic=0.04,
            mean_long_short_return=0.0024,
            ic_ir=0.70,
            mean_ic_ci_lower=-0.01,
            mean_ic_ci_upper=0.06,
            mean_rank_ic_ci_lower=0.01,
            mean_rank_ic_ci_upper=0.07,
            mean_long_short_return_ci_lower=0.0008,
            mean_long_short_return_ci_upper=0.0040,
            uncertainty_flags=["ic_ci_overlaps_zero"],
            rolling_instability_flags=[],
        ),
    )

    assert comparison.delta["mean_ic_delta"] == pytest.approx(-0.01)
    assert comparison.delta["mean_rank_ic_delta"] == pytest.approx(-0.01)
    assert comparison.delta["mean_long_short_return_delta"] == pytest.approx(-0.0006)
    assert comparison.delta["ic_ir_delta"] == pytest.approx(-0.2)
    assert comparison.delta["uncertainty_overlap_zero_count_delta"] == 1
    assert comparison.delta["rolling_instability_flag_count_delta"] == -1


def test_raw_vs_neutralized_comparison_flags_preserve_evidence() -> None:
    comparison = build_raw_vs_neutralized_comparison(
        _metrics(mean_ic=0.04, mean_rank_ic=0.05, mean_long_short_return=0.0030, ic_ir=0.8),
        _metrics(mean_ic=0.038, mean_rank_ic=0.048, mean_long_short_return=0.0028, ic_ir=0.75),
    )
    assert PRESERVES_EVIDENCE_FLAG in comparison.interpretation_flags


def test_raw_vs_neutralized_comparison_flags_material_and_exposure_driven() -> None:
    comparison = build_raw_vs_neutralized_comparison(
        _metrics(mean_ic=0.05, mean_rank_ic=0.06, mean_long_short_return=0.0040, ic_ir=1.0),
        _metrics(mean_ic=-0.001, mean_rank_ic=-0.002, mean_long_short_return=-0.0002, ic_ir=0.1),
        neutralization_mean_corr_reduction=0.30,
    )
    assert MATERIAL_REDUCTION_FLAG in comparison.interpretation_flags
    assert EXPOSURE_DRIVEN_FLAG in comparison.interpretation_flags


def test_raw_vs_neutralized_comparison_flags_weaker_but_stabler() -> None:
    comparison = build_raw_vs_neutralized_comparison(
        _metrics(
            mean_ic=0.040,
            mean_rank_ic=0.045,
            mean_long_short_return=0.0028,
            ic_ir=0.80,
            rolling_ic_positive_share=0.52,
            rolling_rank_ic_positive_share=0.54,
            rolling_long_short_positive_share=0.50,
            rolling_ic_min_mean=-0.0040,
            rolling_rank_ic_min_mean=-0.0030,
            rolling_long_short_min_mean=-0.0008,
            rolling_instability_flags=["rolling_ic_sign_flip_instability"],
        ),
        _metrics(
            mean_ic=0.032,
            mean_rank_ic=0.037,
            mean_long_short_return=0.0022,
            ic_ir=0.62,
            rolling_ic_positive_share=0.70,
            rolling_rank_ic_positive_share=0.71,
            rolling_long_short_positive_share=0.66,
            rolling_ic_min_mean=0.0010,
            rolling_rank_ic_min_mean=0.0012,
            rolling_long_short_min_mean=0.0002,
            rolling_instability_flags=[],
        ),
    )
    assert MODERATE_WEAKENING_FLAG in comparison.interpretation_flags
    assert WEAKER_BUT_STABLER_FLAG in comparison.interpretation_flags


def _metrics(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "mean_ic": 0.03,
        "mean_rank_ic": 0.04,
        "mean_long_short_return": 0.0025,
        "ic_ir": 0.7,
        "ic_valid_ratio": 0.9,
        "rank_ic_valid_ratio": 0.88,
        "eval_coverage_ratio_mean": 0.82,
        "eval_coverage_ratio_min": 0.70,
        "rolling_ic_positive_share": 0.60,
        "rolling_rank_ic_positive_share": 0.62,
        "rolling_long_short_positive_share": 0.58,
        "rolling_ic_min_mean": 0.002,
        "rolling_rank_ic_min_mean": 0.003,
        "rolling_long_short_min_mean": 0.0005,
        "mean_ic_ci_lower": 0.01,
        "mean_ic_ci_upper": 0.05,
        "mean_rank_ic_ci_lower": 0.02,
        "mean_rank_ic_ci_upper": 0.06,
        "mean_long_short_return_ci_lower": 0.0010,
        "mean_long_short_return_ci_upper": 0.0040,
        "uncertainty_flags": [],
        "rolling_instability_flags": [],
    }
    base.update(overrides)
    return base
