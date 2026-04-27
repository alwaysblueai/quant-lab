from __future__ import annotations

from alpha_lab.reporting.factor_verdict import (
    FACTOR_VERDICT_TAXONOMY,
    build_factor_verdict,
    reasons_to_text,
)


def test_factor_verdict_taxonomy_is_compact_and_stable() -> None:
    assert FACTOR_VERDICT_TAXONOMY == (
        "Strong candidate",
        "Promising but fragile",
        "Mixed evidence",
        "Weak / noisy",
        "Fails basic robustness",
    )


def test_factor_verdict_classifies_strong_candidate() -> None:
    verdict = build_factor_verdict(
        {
            "mean_ic": 0.04,
            "mean_rank_ic": 0.05,
            "ic_positive_rate": 0.63,
            "rank_ic_positive_rate": 0.67,
            "ic_valid_ratio": 0.92,
            "rank_ic_valid_ratio": 0.90,
            "mean_long_short_return": 0.0035,
            "long_short_ir": 0.75,
            "long_short_return_per_turnover": 0.011,
            "mean_long_short_turnover": 0.26,
            "subperiod_ic_positive_share": 1.0,
            "subperiod_long_short_positive_share": 1.0,
            "eval_coverage_ratio_mean": 0.83,
            "eval_coverage_ratio_min": 0.73,
            "n_dates_used": 80,
            "instability_flags": [],
        }
    )
    assert verdict.label == "Strong candidate"
    reasons = reasons_to_text(verdict.reasons)
    assert "positive IC and RankIC means" in reasons
    assert "robust across subperiods" in reasons


def test_factor_verdict_classifies_promising_but_fragile() -> None:
    verdict = build_factor_verdict(
        {
            "mean_ic": 0.03,
            "mean_rank_ic": 0.02,
            "ic_positive_rate": 0.61,
            "rank_ic_positive_rate": 0.60,
            "ic_valid_ratio": 0.88,
            "rank_ic_valid_ratio": 0.86,
            "mean_long_short_return": 0.0024,
            "long_short_ir": 0.35,
            "long_short_return_per_turnover": 0.0031,
            "mean_long_short_turnover": 0.91,
            "subperiod_ic_positive_share": 0.67,
            "subperiod_long_short_positive_share": 0.67,
            "eval_coverage_ratio_mean": 0.66,
            "eval_coverage_ratio_min": 0.45,
            "neutralization_mean_corr_reduction": 0.25,
            "n_dates_used": 64,
            "instability_flags": ["ic_subperiod_instability"],
        }
    )
    assert verdict.label == "Promising but fragile"
    reasons = reasons_to_text(verdict.reasons)
    assert "coverage is uneven across evaluation dates" in reasons
    assert "instability flags triggered" in reasons


def test_factor_verdict_classifies_mixed_evidence() -> None:
    verdict = build_factor_verdict(
        {
            "mean_ic": 0.01,
            "mean_rank_ic": 0.012,
            "ic_positive_rate": 0.53,
            "rank_ic_positive_rate": 0.54,
            "ic_valid_ratio": 0.76,
            "rank_ic_valid_ratio": 0.77,
            "mean_long_short_return": 0.0012,
            "long_short_ir": 0.07,
            "long_short_return_per_turnover": 0.0018,
            "mean_long_short_turnover": 0.72,
            "subperiod_ic_positive_share": 0.67,
            "subperiod_long_short_positive_share": 0.60,
            "eval_coverage_ratio_mean": 0.65,
            "eval_coverage_ratio_min": 0.45,
            "n_dates_used": 58,
            "instability_flags": [],
        }
    )
    assert verdict.label == "Mixed evidence"
    reasons = reasons_to_text(verdict.reasons)
    assert "positive IC and RankIC means" in reasons
    assert "subperiod robustness is mixed" in reasons


def test_factor_verdict_classifies_weak_noisy() -> None:
    verdict = build_factor_verdict(
        {
            "mean_ic": -0.002,
            "mean_rank_ic": -0.001,
            "ic_positive_rate": 0.49,
            "rank_ic_positive_rate": 0.48,
            "ic_valid_ratio": 0.86,
            "rank_ic_valid_ratio": 0.84,
            "mean_long_short_return": -0.0005,
            "long_short_ir": -0.12,
            "long_short_return_per_turnover": -0.0006,
            "mean_long_short_turnover": 0.41,
            "subperiod_ic_positive_share": 0.67,
            "subperiod_long_short_positive_share": 0.67,
            "eval_coverage_ratio_mean": 0.78,
            "eval_coverage_ratio_min": 0.60,
            "n_dates_used": 52,
            "instability_flags": [],
        }
    )
    assert verdict.label == "Weak / noisy"
    reasons = reasons_to_text(verdict.reasons)
    assert "long-short spread is not reliable" in reasons


def test_factor_verdict_classifies_basic_robustness_failure() -> None:
    verdict = build_factor_verdict(
        {
            "mean_ic": 0.008,
            "mean_rank_ic": 0.010,
            "ic_positive_rate": 0.57,
            "rank_ic_positive_rate": 0.58,
            "ic_valid_ratio": 0.55,
            "rank_ic_valid_ratio": 0.58,
            "mean_long_short_return": 0.0011,
            "long_short_ir": 0.05,
            "subperiod_ic_positive_share": 0.33,
            "subperiod_long_short_positive_share": 0.33,
            "eval_coverage_ratio_mean": 0.45,
            "eval_coverage_ratio_min": 0.25,
            "n_dates_used": 12,
            "instability_flags": ["short_eval_window"],
        }
    )
    assert verdict.label == "Fails basic robustness"
    reasons = reasons_to_text(verdict.reasons)
    assert "evaluation window is too short" in reasons
    assert "coverage is too thin" in reasons


def test_factor_verdict_adds_positive_uncertainty_reason() -> None:
    verdict = build_factor_verdict(
        {
            "mean_ic": 0.03,
            "mean_rank_ic": 0.04,
            "ic_positive_rate": 0.62,
            "rank_ic_positive_rate": 0.63,
            "ic_valid_ratio": 0.90,
            "rank_ic_valid_ratio": 0.90,
            "mean_long_short_return": 0.003,
            "long_short_ir": 0.70,
            "subperiod_ic_positive_share": 1.0,
            "subperiod_long_short_positive_share": 1.0,
            "eval_coverage_ratio_mean": 0.85,
            "eval_coverage_ratio_min": 0.75,
            "n_dates_used": 72,
            "mean_ic_ci_lower": 0.01,
            "mean_ic_ci_upper": 0.05,
            "mean_rank_ic_ci_lower": 0.02,
            "mean_rank_ic_ci_upper": 0.06,
            "mean_long_short_return_ci_lower": 0.001,
            "mean_long_short_return_ci_upper": 0.005,
        }
    )
    assert "evidence remains positive under uncertainty" in reasons_to_text(verdict.reasons)


def test_factor_verdict_adds_overlap_zero_uncertainty_reason() -> None:
    verdict = build_factor_verdict(
        {
            "mean_ic": 0.02,
            "mean_rank_ic": 0.03,
            "ic_positive_rate": 0.58,
            "rank_ic_positive_rate": 0.59,
            "ic_valid_ratio": 0.87,
            "rank_ic_valid_ratio": 0.87,
            "mean_long_short_return": 0.002,
            "long_short_ir": 0.31,
            "subperiod_ic_positive_share": 0.67,
            "subperiod_long_short_positive_share": 0.67,
            "eval_coverage_ratio_mean": 0.71,
            "eval_coverage_ratio_min": 0.55,
            "n_dates_used": 64,
            "mean_ic_ci_lower": -0.01,
            "mean_ic_ci_upper": 0.04,
            "mean_rank_ic_ci_lower": 0.01,
            "mean_rank_ic_ci_upper": 0.05,
            "mean_long_short_return_ci_lower": 0.0002,
            "mean_long_short_return_ci_upper": 0.003,
        }
    )
    assert "confidence interval overlaps zero" in reasons_to_text(verdict.reasons)


def test_factor_verdict_adds_estimation_noise_reason_from_uncertainty_flags() -> None:
    verdict = build_factor_verdict(
        {
            "mean_ic": 0.015,
            "mean_rank_ic": 0.014,
            "ic_positive_rate": 0.55,
            "rank_ic_positive_rate": 0.56,
            "ic_valid_ratio": 0.83,
            "rank_ic_valid_ratio": 0.82,
            "mean_long_short_return": 0.0015,
            "long_short_ir": 0.10,
            "subperiod_ic_positive_share": 0.67,
            "subperiod_long_short_positive_share": 0.67,
            "eval_coverage_ratio_mean": 0.70,
            "eval_coverage_ratio_min": 0.52,
            "n_dates_used": 64,
            "uncertainty_flags": ["ic_ci_wide"],
        }
    )
    assert "apparent edge is weak relative to estimation noise" in reasons_to_text(verdict.reasons)


def test_factor_verdict_adds_rolling_persistence_reason() -> None:
    verdict = build_factor_verdict(
        {
            "mean_ic": 0.03,
            "mean_rank_ic": 0.03,
            "ic_positive_rate": 0.60,
            "rank_ic_positive_rate": 0.61,
            "ic_valid_ratio": 0.90,
            "rank_ic_valid_ratio": 0.89,
            "mean_long_short_return": 0.0025,
            "long_short_ir": 0.42,
            "subperiod_ic_positive_share": 0.67,
            "subperiod_long_short_positive_share": 0.67,
            "eval_coverage_ratio_mean": 0.75,
            "eval_coverage_ratio_min": 0.62,
            "n_dates_used": 70,
            "rolling_ic_positive_share": 0.78,
            "rolling_rank_ic_positive_share": 0.74,
            "rolling_long_short_positive_share": 0.71,
            "rolling_ic_min_mean": 0.005,
            "rolling_rank_ic_min_mean": 0.004,
            "rolling_long_short_min_mean": 0.0004,
            "rolling_instability_flags": [],
        }
    )
    assert "evidence is persistent across rolling windows" in reasons_to_text(verdict.reasons)


def test_factor_verdict_flags_rebalance_decay_mismatch() -> None:
    verdict = build_factor_verdict(
        {
            "mean_ic": 0.03,
            "mean_rank_ic": 0.03,
            "ic_positive_rate": 0.62,
            "rank_ic_positive_rate": 0.63,
            "ic_valid_ratio": 0.91,
            "rank_ic_valid_ratio": 0.90,
            "mean_long_short_return": 0.0030,
            "long_short_ir": 0.62,
            "long_short_return_per_turnover": 0.009,
            "mean_long_short_turnover": 0.25,
            "subperiod_ic_positive_share": 1.0,
            "subperiod_long_short_positive_share": 1.0,
            "eval_coverage_ratio_mean": 0.82,
            "eval_coverage_ratio_min": 0.72,
            "n_dates_used": 80,
            "rebalance_step_dates": 21,
            "ic_half_life_horizon": 15.0,
            "ic_half_life_status": "estimated",
            "ic_decay_rebalance_ratio": 1.4,
            "instability_flags": [],
        }
    )
    assert verdict.label == "Promising but fragile"
    assert "rebalance cadence may be too slow for IC decay" in reasons_to_text(verdict.reasons)


def test_factor_verdict_blocks_when_rebalance_far_exceeds_half_life() -> None:
    verdict = build_factor_verdict(
        {
            "mean_ic": 0.03,
            "mean_rank_ic": 0.03,
            "ic_positive_rate": 0.62,
            "rank_ic_positive_rate": 0.63,
            "ic_valid_ratio": 0.91,
            "rank_ic_valid_ratio": 0.90,
            "mean_long_short_return": 0.0030,
            "long_short_ir": 0.62,
            "long_short_return_per_turnover": 0.009,
            "mean_long_short_turnover": 0.25,
            "subperiod_ic_positive_share": 1.0,
            "subperiod_long_short_positive_share": 1.0,
            "eval_coverage_ratio_mean": 0.82,
            "eval_coverage_ratio_min": 0.72,
            "n_dates_used": 80,
            "rebalance_step_dates": 21,
            "ic_half_life_horizon": 8.0,
            "ic_half_life_status": "estimated",
            "ic_decay_rebalance_ratio": 2.625,
            "instability_flags": [],
        }
    )
    assert verdict.label == "Fails basic robustness"
    assert "rebalance cadence materially exceeds IC half-life" in reasons_to_text(verdict.reasons)


def test_factor_verdict_adds_rolling_regime_dependence_reasons() -> None:
    verdict = build_factor_verdict(
        {
            "mean_ic": 0.02,
            "mean_rank_ic": 0.02,
            "ic_positive_rate": 0.56,
            "rank_ic_positive_rate": 0.55,
            "ic_valid_ratio": 0.86,
            "rank_ic_valid_ratio": 0.85,
            "mean_long_short_return": 0.0016,
            "long_short_ir": 0.18,
            "subperiod_ic_positive_share": 0.67,
            "subperiod_long_short_positive_share": 0.67,
            "eval_coverage_ratio_mean": 0.72,
            "eval_coverage_ratio_min": 0.55,
            "n_dates_used": 68,
            "rolling_ic_positive_share": 0.42,
            "rolling_rank_ic_positive_share": 0.45,
            "rolling_long_short_positive_share": 0.40,
            "rolling_ic_min_mean": -0.006,
            "rolling_rank_ic_min_mean": -0.005,
            "rolling_long_short_min_mean": -0.0009,
            "rolling_instability_flags": [
                "rolling_ic_sign_flip_instability",
                "rolling_regime_dependence",
            ],
        }
    )
    reasons = reasons_to_text(verdict.reasons)
    assert "rolling evidence suggests regime dependence" in reasons
    assert "rolling factor performance is unstable through time" in reasons


def test_factor_verdict_adds_neutralization_survival_reason_from_comparison_flags() -> None:
    verdict = build_factor_verdict(
        {
            "mean_ic": 0.03,
            "mean_rank_ic": 0.03,
            "ic_positive_rate": 0.61,
            "rank_ic_positive_rate": 0.61,
            "ic_valid_ratio": 0.89,
            "rank_ic_valid_ratio": 0.88,
            "mean_long_short_return": 0.0024,
            "long_short_ir": 0.40,
            "subperiod_ic_positive_share": 0.67,
            "subperiod_long_short_positive_share": 0.67,
            "eval_coverage_ratio_mean": 0.75,
            "eval_coverage_ratio_min": 0.60,
            "n_dates_used": 70,
            "neutralization_comparison_flags": [
                "neutralization preserves most evidence",
            ],
        }
    )
    assert "signal survives neutralization well" in reasons_to_text(verdict.reasons)


def test_factor_verdict_adds_neutralization_exposure_and_stability_reasons() -> None:
    verdict = build_factor_verdict(
        {
            "mean_ic": 0.02,
            "mean_rank_ic": 0.02,
            "ic_positive_rate": 0.58,
            "rank_ic_positive_rate": 0.57,
            "ic_valid_ratio": 0.85,
            "rank_ic_valid_ratio": 0.84,
            "mean_long_short_return": 0.0015,
            "long_short_ir": 0.20,
            "subperiod_ic_positive_share": 0.67,
            "subperiod_long_short_positive_share": 0.67,
            "eval_coverage_ratio_mean": 0.72,
            "eval_coverage_ratio_min": 0.55,
            "n_dates_used": 68,
            "neutralization_comparison_flags": [
                "neutralization materially reduces independent evidence",
                "raw signal appears exposure-driven",
                "neutralization improves stability despite weaker raw performance",
            ],
        }
    )
    reasons = reasons_to_text(verdict.reasons)
    assert "neutralization materially reduces independent evidence" in reasons
    assert "raw signal may be driven by common exposure" in reasons
    assert "neutralized signal is weaker but more stable" in reasons
