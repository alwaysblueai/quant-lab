from __future__ import annotations

from alpha_lab.reporting.level2_promotion import (
    LEVEL2_PROMOTION_TAXONOMY,
    build_level2_promotion,
)
from alpha_lab.research_evaluation_config import (
    Level2PromotionConfig,
    get_research_evaluation_config,
)


def _base_metrics() -> dict[str, object]:
    return {
        "factor_verdict": "Strong candidate",
        "campaign_triage": "Advance to Level 2",
        "mean_ic_ci_lower": 0.012,
        "mean_ic_ci_upper": 0.031,
        "mean_rank_ic_ci_lower": 0.015,
        "mean_rank_ic_ci_upper": 0.036,
        "mean_long_short_return_ci_lower": 0.0012,
        "mean_long_short_return_ci_upper": 0.0042,
        "rolling_ic_positive_share": 0.72,
        "rolling_rank_ic_positive_share": 0.70,
        "rolling_long_short_positive_share": 0.68,
        "rolling_ic_min_mean": 0.004,
        "rolling_rank_ic_min_mean": 0.003,
        "rolling_long_short_min_mean": 0.0004,
        "rolling_instability_flags": [],
        "subperiod_ic_positive_share": 0.75,
        "subperiod_long_short_positive_share": 0.75,
        "coverage_mean": 0.82,
        "coverage_min": 0.70,
        "ic_valid_ratio": 0.90,
        "rank_ic_valid_ratio": 0.90,
        "mean_long_short_turnover": 0.35,
        "long_short_return_per_turnover": 0.009,
        "neutralization_comparison_flags": ["neutralization preserves most evidence"],
    }


def test_level2_promotion_taxonomy_is_compact_and_stable() -> None:
    assert LEVEL2_PROMOTION_TAXONOMY == (
        "Promote to Level 2",
        "Hold for refinement",
        "Blocked from Level 2",
    )


def test_level2_promotion_classifies_promote() -> None:
    decision = build_level2_promotion(_base_metrics())
    assert decision.label == "Promote to Level 2"
    assert decision.blockers == ()
    reason_text = "; ".join(decision.reasons)
    assert "factor verdict is strong" in reason_text
    assert "stable across rolling windows" in reason_text


def test_level2_promotion_blocks_thin_coverage() -> None:
    metrics = _base_metrics()
    metrics["coverage_mean"] = 0.40
    decision = build_level2_promotion(metrics)
    assert decision.label == "Blocked from Level 2"
    assert "blocked by thin coverage" in decision.blockers


def test_level2_promotion_blocks_material_neutralization_weakness() -> None:
    metrics = _base_metrics()
    metrics["neutralization_comparison_flags"] = [
        "neutralization materially reduces independent evidence"
    ]
    decision = build_level2_promotion(metrics)
    assert decision.label == "Blocked from Level 2"
    assert "blocked by weak neutralized evidence" in decision.blockers


def test_level2_promotion_holds_when_neutralization_support_is_missing() -> None:
    metrics = _base_metrics()
    metrics["neutralization_comparison_flags"] = []
    decision = build_level2_promotion(metrics)
    assert decision.label == "Hold for refinement"
    assert "neutralization evidence is unavailable" in "; ".join(decision.reasons)


def test_level2_promotion_remains_separate_from_triage() -> None:
    metrics = _base_metrics()
    metrics["campaign_triage"] = "Advance to Level 2"
    metrics["long_short_return_per_turnover"] = -0.001
    decision = build_level2_promotion(metrics)
    assert decision.label == "Blocked from Level 2"
    assert "blocked by poor turnover efficiency" in decision.blockers


def test_level2_promotion_threshold_override_can_relax_neutralization_requirement() -> None:
    metrics = _base_metrics()
    metrics["neutralization_comparison_flags"] = []
    baseline = build_level2_promotion(metrics)
    relaxed = build_level2_promotion(
        metrics,
        thresholds=Level2PromotionConfig(require_neutralization_support_for_promote=False),
    )
    assert baseline.label == "Hold for refinement"
    assert relaxed.label == "Promote to Level 2"


def test_exploratory_profile_promotes_strong_signal_during_initial_screening() -> None:
    metrics = _base_metrics()
    metrics["factor_verdict"] = "Promising but fragile"
    metrics["neutralization_comparison_flags"] = []
    decision = build_level2_promotion(
        metrics,
        thresholds=get_research_evaluation_config("exploratory_screening").level2_promotion,
    )
    assert decision.label == "Promote to Level 2"
