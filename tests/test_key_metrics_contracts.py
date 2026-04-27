from __future__ import annotations

from alpha_lab.key_metrics_contracts import (
    CAMPAIGN_PROFILE_COMPARISON_FIELDS,
    LEVEL12_TRANSITION_TAXONOMY,
    PROFILE_AWARE_LEVEL12_OBSERVED_DIFFERENCE_FIELDS,
    normalize_transition_reason_for_rollup,
    project_campaign_profile_summary_metrics,
    project_campaign_ranking_metrics,
    project_level12_transition_distribution,
    project_level12_transition_summary,
    project_neutralization_comparison_metrics,
    project_portfolio_validation_metrics,
    project_promotion_gate_metrics,
)


def test_project_promotion_gate_metrics_covers_repeated_level12_evidence_subsets() -> None:
    projected = project_promotion_gate_metrics(
        {
            "factor_verdict": "Strong candidate",
            "campaign_triage": "Advance to Level 2",
            "ic_valid_ratio": 0.86,
            "rank_ic_valid_ratio": 0.82,
            "mean_ic_ci_lower": -0.01,
            "mean_ic_ci_upper": 0.03,
            "mean_rank_ic_ci_lower": 0.002,
            "mean_rank_ic_ci_upper": 0.028,
            "mean_long_short_return_ci_lower": 0.001,
            "mean_long_short_return_ci_upper": 0.004,
            "rolling_ic_positive_share": 0.72,
            "rolling_rank_ic_positive_share": 0.64,
            "rolling_long_short_positive_share": 0.68,
            "rolling_ic_min_mean": 0.002,
            "rolling_rank_ic_min_mean": 0.001,
            "rolling_long_short_min_mean": -0.0002,
            "rolling_instability_flags": ["rolling_ic_sign_flip_instability"],
            "neutralization_comparison": {
                "interpretation_flags": ["neutralization preserves most evidence"],
                "interpretation_reasons": ["core evidence remains close to raw"],
            },
            "neutralization_mean_ic_delta": -0.005,
        }
    )

    assert projected["factor_verdict"] == "Strong candidate"
    assert projected["campaign_triage"] == "Advance to Level 2"
    assert projected["core"]["valid_ratio_min"] == 0.82
    assert projected["uncertainty"]["uncertainty_supportive_ci_count"] == 2
    assert projected["uncertainty"]["uncertainty_overlap_zero_count"] == 1
    assert projected["rolling"]["rolling_positive_share_min"] == 0.64
    assert projected["rolling"]["rolling_worst_mean_min"] == -0.0002
    assert projected["rolling"]["rolling_instability_flags"] == (
        "rolling_ic_sign_flip_instability",
    )
    assert projected["neutralization"]["neutralization_flags"] == (
        "neutralization preserves most evidence",
    )
    assert projected["neutralization"]["neutralization_mean_ic_delta"] == -0.005


def test_project_portfolio_and_profile_summary_metrics_preserve_zero_and_reasons() -> None:
    metrics = {
        "factor_verdict": "Strong candidate",
        "factor_verdict_reasons": ["positive IC and RankIC means"],
        "campaign_triage": "Advance to Level 2",
        "campaign_triage_reasons": ["stable across rolling windows"],
        "case_name": "demo_case",
        "rebalance_frequency": "W",
        "promotion_decision": "Promote to Level 2",
        "promotion_reasons": "gate passed;robust diagnostics",
        "promotion_blockers": [],
        "portfolio_validation_status": "completed",
        "portfolio_validation_recommendation": "Credible at portfolio level",
        "portfolio_validation_robustness_label": "Credible but sensitive",
        "portfolio_validation_support_reasons": ["baseline return is positive"],
        "portfolio_validation_fragility_reasons": ["weighting sensitivity is material"],
        "portfolio_validation_major_risks": "none",
        "benchmark_excess_return": 0.0,
        "benchmark_tracking_error": "0.020",
        "mean_eval_assets_per_date": "200",
        "n_quantiles": "5",
    }

    portfolio = project_portfolio_validation_metrics(metrics)
    assert portfolio["case_name"] == "demo_case"
    assert portfolio["rebalance_frequency"] == "W"
    assert portfolio["promotion_reasons"] == ("gate passed", "robust diagnostics")
    assert portfolio["benchmark_excess_return"] == 0.0
    assert portfolio["benchmark_tracking_error"] == 0.02
    assert portfolio["portfolio_validation_robustness_label"] == "Credible but sensitive"
    assert portfolio["portfolio_validation_support_reasons"] == ("baseline return is positive",)
    assert portfolio["portfolio_validation_fragility_reasons"] == (
        "weighting sensitivity is material",
    )
    assert portfolio["mean_eval_assets_per_date"] == 200.0
    assert portfolio["n_quantiles"] == 5.0

    profile_summary = project_campaign_profile_summary_metrics(metrics)
    assert profile_summary["factor_verdict"] == "Strong candidate"
    assert profile_summary["campaign_triage"] == "Advance to Level 2"
    assert profile_summary["promotion_decision"] == "Promote to Level 2"
    assert profile_summary["portfolio_validation_recommendation"] == "Credible at portfolio level"
    assert profile_summary["level12_transition_label"] == "Confirmed at portfolio level"
    assert profile_summary["level12_transition_summary"]["level1_status"] == "Strong candidate"
    assert (
        profile_summary["level12_transition_summary"]["level2_status"]
        == "Credible at portfolio level"
    )


def test_project_neutralization_comparison_metrics_supports_legacy_top_level_fields() -> None:
    projected = project_neutralization_comparison_metrics(
        {
            "neutralization_raw_mean_ic": 0.041,
            "neutralization_raw_mean_rank_ic": 0.050,
            "neutralization_raw_mean_long_short_return": 0.0036,
            "neutralization_raw_ic_ir": 0.88,
            "neutralization_mean_ic_delta": -0.011,
            "neutralization_mean_rank_ic_delta": -0.010,
            "neutralization_mean_long_short_return_delta": -0.0006,
            "neutralization_ic_ir_delta": -0.18,
            "neutralization_comparison_flags": [
                "neutralization materially reduces independent evidence"
            ],
            "neutralization_comparison_reasons": ["top-level fallback for legacy payloads"],
        }
    )

    assert projected["neutralization_raw_mean_ic"] == 0.041
    assert projected["neutralization_mean_rank_ic_delta"] == -0.010
    assert projected["neutralization_flags"] == (
        "neutralization materially reduces independent evidence",
    )
    assert projected["neutralization_reasons"] == ("top-level fallback for legacy payloads",)


def test_shared_campaign_profile_field_sets_and_ranking_projection() -> None:
    assert CAMPAIGN_PROFILE_COMPARISON_FIELDS == (
        "factor_verdict",
        "campaign_triage",
        "promotion_decision",
        "portfolio_validation_recommendation",
    )
    assert PROFILE_AWARE_LEVEL12_OBSERVED_DIFFERENCE_FIELDS == (
        "factor_verdict",
        "campaign_triage",
        "promotion_decision",
        "level12_transition_label",
        "portfolio_validation_status",
        "portfolio_validation_recommendation",
    )

    ranking = project_campaign_ranking_metrics(
        {
            "ic_ir": "0.82",
            "mean_long_short_return": 0.0031,
            "rolling_ic_positive_share": 0.72,
            "rolling_rank_ic_positive_share": 0.68,
            "rolling_long_short_positive_share": float("nan"),
        }
    )
    assert ranking == {
        "ic_ir": 0.82,
        "mean_long_short_return": 0.0031,
        "rolling_ic_positive_share": 0.72,
        "rolling_rank_ic_positive_share": 0.68,
        "rolling_long_short_positive_share": None,
    }


def test_level12_transition_taxonomy_is_compact_and_stable() -> None:
    assert LEVEL12_TRANSITION_TAXONOMY == (
        "Confirmed at portfolio level",
        "Weakened at portfolio level",
        "Fragile after promotion",
        "Improved at portfolio level",
        "Inconclusive transition",
    )


def test_project_level12_transition_distribution_counts_proportions_and_missing() -> None:
    distribution = project_level12_transition_distribution(
        [
            {
                "case_name": "confirmed_case",
                "level12_transition_label": "Confirmed at portfolio level",
                "level12_transition_reasons": [
                    "promotion decision: Promote to Level 2",
                    "portfolio recommendation: Credible at portfolio level",
                ],
                "artifact_pointer": "dist/confirmed_case/metrics.json",
            },
            {
                "case_name": "weakened_case",
                "level12_transition_label": "Weakened at portfolio level",
                "level12_transition_reasons": [
                    "promotion decision: Promote to Level 2",
                    "portfolio recommendation: Needs portfolio refinement",
                ],
                "artifact_pointer": "dist/weakened_case/metrics.json",
            },
            {
                "case_name": "fragile_case",
                "level12_transition_label": "Fragile after promotion",
                "level12_transition_reasons": [
                    "fragility: concentration risk is elevated",
                ],
                "artifact_pointer": "dist/fragile_case/metrics.json",
            },
            {
                "case_name": "missing_case",
                "level12_transition_label": None,
            },
            {
                "case_name": "unknown_case",
                "level12_transition_label": "unknown_transition",
            },
        ]
    )

    assert distribution["n_cases"] == 5
    assert distribution["n_cases_with_transition_label"] == 3
    assert distribution["n_cases_missing_transition_label"] == 2
    assert distribution["minimum_support_met"] is True
    assert distribution["support_level"] == "tentative"
    assert distribution["support_note"] == "tentative due to low support"
    counts = distribution["counts_by_transition_label"]
    assert counts["Confirmed at portfolio level"] == 1
    assert counts["Weakened at portfolio level"] == 1
    assert counts["Fragile after promotion"] == 1
    assert counts["Improved at portfolio level"] == 0
    assert counts["Inconclusive transition"] == 0
    proportions = distribution["proportions_by_transition_label"]
    assert proportions["Confirmed at portfolio level"] == 0.2
    assert proportions["Weakened at portfolio level"] == 0.2
    assert proportions["Fragile after promotion"] == 0.2
    reps = distribution["representative_cases_by_transition_label"]
    assert reps["Confirmed at portfolio level"] == ["confirmed_case"]
    rep_names = distribution["representative_case_names_by_transition_label"]
    assert rep_names["Confirmed at portfolio level"] == ["confirmed_case"]
    artifact_hints_by_label = distribution["artifact_pointers_by_transition_label"]
    assert artifact_hints_by_label["Confirmed at portfolio level"] == [
        "dist/confirmed_case/metrics.json"
    ]
    assert "missing transition labels" in distribution["interpretation"]
    reason_rollups = distribution["reason_rollup_by_transition_label"]
    confirmed_rollup = reason_rollups["Confirmed at portfolio level"]
    assert confirmed_rollup["n_cases_with_label"] == 1
    assert confirmed_rollup["n_cases_with_any_reason"] == 1
    assert confirmed_rollup["minimum_support_met"] is False
    assert confirmed_rollup["is_sparse"] is True
    assert confirmed_rollup["dominant_reasons"] == []
    top_reason = confirmed_rollup["top_reasons"][0]
    assert top_reason["reason"] == "portfolio recommendation: Credible at portfolio level"
    assert top_reason["count"] == 1
    assert top_reason["proportion_of_label_cases"] == 1.0
    assert top_reason["minimum_support_met"] is False
    assert confirmed_rollup["representative_case_names"] == ["confirmed_case"]
    assert confirmed_rollup["supporting_case_names"] == ["confirmed_case"]
    assert confirmed_rollup["artifact_pointer_hints"] == ["dist/confirmed_case/metrics.json"]
    assert top_reason["supporting_case_names"] == ["confirmed_case"]
    assert top_reason["artifact_pointer_hints"] == ["dist/confirmed_case/metrics.json"]


def test_project_level12_transition_distribution_reason_rollup_is_grouped_and_capped() -> None:
    distribution = project_level12_transition_distribution(
        [
            {
                "case_name": "w_case_1",
                "level12_transition_label": "Weakened at portfolio level",
                "level12_transition_reasons": [
                    "promotion decision: Promote to Level 2",
                    "portfolio recommendation: Needs portfolio refinement",
                    "promotion decision: Promote to Level 2",
                ],
            },
            {
                "case_name": "w_case_2",
                "level12_transition_label": "Weakened at portfolio level",
                "level12_transition_reasons": [
                    "promotion decision: Promote to Level 2",
                    "fragility: weighting sensitivity is material",
                ],
            },
            {
                "case_name": "w_case_3",
                "level12_transition_label": "Weakened at portfolio level",
                "level12_transition_reasons": [],
            },
            {
                "case_name": "c_case_1",
                "level12_transition_label": "Confirmed at portfolio level",
                "level12_transition_reasons": [
                    "portfolio recommendation: Credible at portfolio level",
                ],
            },
        ],
        top_reason_limit=2,
        representative_reason_limit=2,
    )
    rollups = distribution["reason_rollup_by_transition_label"]
    weakened = rollups["Weakened at portfolio level"]
    assert weakened["n_cases_with_label"] == 3
    assert weakened["n_cases_with_any_reason"] == 2
    assert weakened["n_unique_reasons_observed"] == 3
    top_reasons = weakened["top_reasons"]
    assert len(top_reasons) == 2
    assert top_reasons[0]["reason"] == "promotion decision: Promote to Level 2"
    assert top_reasons[0]["count"] == 2
    assert top_reasons[0]["proportion_of_label_cases"] == 2 / 3
    assert weakened["n_unique_raw_reasons_observed"] == 3
    assert weakened["n_reasons_collapsed_by_normalization"] == 0
    assert weakened["reason_normalization"] == {
        "method": "explicit_rule_v1",
        "applies_to_rollup_only": True,
        "raw_reasons_preserved_at_case_level": True,
    }
    assert weakened["minimum_support_met"] is True
    assert weakened["support_level"] == "tentative"
    assert weakened["dominant_reasons"] == [
        {
            "reason": "promotion decision: Promote to Level 2",
            "count": 2,
            "proportion_of_label_cases": 2 / 3,
            "supporting_case_names": ["w_case_1", "w_case_2"],
            "artifact_pointer_hints": [],
            "support_level": "tentative",
            "is_sparse": False,
            "minimum_support_met": True,
            "support_note": "tentative due to low support",
            "confidence_note": "tentative due to low support",
        }
    ]
    assert weakened["representative_reasons"] == [
        "promotion decision: Promote to Level 2",
        "portfolio recommendation: Needs portfolio refinement",
    ]


def test_normalize_transition_reason_for_rollup_uses_explicit_exact_and_prefix_rules() -> None:
    assert (
        normalize_transition_reason_for_rollup("promotion decision: promote to level 2")
        == "promotion decision: Promote to Level 2"
    )
    assert (
        normalize_transition_reason_for_rollup(
            "promotion reason: blocked by unstable rolling evidence"
        )
        == "blocked by unstable rolling evidence"
    )
    assert (
        normalize_transition_reason_for_rollup("fragility: blocked by unstable rolling evidence")
        == "blocked by unstable rolling evidence"
    )


def test_project_level12_transition_distribution_normalizes_near_duplicate_reasons() -> None:
    distribution = project_level12_transition_distribution(
        [
            {
                "case_name": "w_case_1",
                "level12_transition_label": "Weakened at portfolio level",
                "level12_transition_reasons": [
                    "promotion reason: blocked by unstable rolling evidence",
                    "fragility: blocked by unstable rolling evidence",
                ],
            },
            {
                "case_name": "w_case_2",
                "level12_transition_label": "Weakened at portfolio level",
                "level12_transition_reasons": [
                    "blocked by unstable rolling evidence",
                ],
            },
        ]
    )
    weakened = distribution["reason_rollup_by_transition_label"]["Weakened at portfolio level"]
    assert weakened["n_cases_with_label"] == 2
    assert weakened["n_cases_with_any_reason"] == 2
    assert weakened["n_unique_raw_reasons_observed"] == 3
    assert weakened["n_unique_reasons_observed"] == 1
    assert weakened["n_reasons_collapsed_by_normalization"] == 2
    assert weakened["top_reasons"] == [
        {
            "reason": "blocked by unstable rolling evidence",
            "count": 2,
            "proportion_of_label_cases": 1.0,
            "supporting_case_names": ["w_case_1", "w_case_2"],
            "artifact_pointer_hints": [],
            "support_level": "tentative",
            "is_sparse": False,
            "minimum_support_met": True,
            "support_note": "tentative due to low support",
            "confidence_note": "tentative due to low support",
        }
    ]
    assert weakened["dominant_reasons"] == weakened["top_reasons"]


def test_project_level12_transition_distribution_preserves_unrelated_reason_strings() -> None:
    distribution = project_level12_transition_distribution(
        [
            {
                "case_name": "w_case_1",
                "level12_transition_label": "Weakened at portfolio level",
                "level12_transition_reasons": [
                    "portfolio risk: net return is near zero after costs",
                    "benchmark-relative: standalone alpha remains positive",
                ],
            },
        ]
    )
    weakened = distribution["reason_rollup_by_transition_label"]["Weakened at portfolio level"]
    assert weakened["n_unique_raw_reasons_observed"] == 2
    assert weakened["n_unique_reasons_observed"] == 2
    assert weakened["n_reasons_collapsed_by_normalization"] == 0
    assert weakened["top_reasons"] == [
        {
            "reason": "benchmark-relative: standalone alpha remains positive",
            "count": 1,
            "proportion_of_label_cases": 1.0,
            "supporting_case_names": ["w_case_1"],
            "artifact_pointer_hints": [],
            "support_level": "sparse",
            "is_sparse": True,
            "minimum_support_met": False,
            "support_note": "tentative due to low support",
            "confidence_note": "tentative due to low support",
        },
        {
            "reason": "portfolio risk: net return is near zero after costs",
            "count": 1,
            "proportion_of_label_cases": 1.0,
            "supporting_case_names": ["w_case_1"],
            "artifact_pointer_hints": [],
            "support_level": "sparse",
            "is_sparse": True,
            "minimum_support_met": False,
            "support_note": "tentative due to low support",
            "confidence_note": "tentative due to low support",
        },
    ]
    assert weakened["dominant_reasons"] == []


def test_transition_distribution_marks_sparse_reason_rollup_and_supported_campaign() -> None:
    sparse_distribution = project_level12_transition_distribution(
        [
            {
                "case_name": "tiny_case",
                "level12_transition_label": "Fragile after promotion",
                "level12_transition_reasons": ["cost sensitivity is elevated"],
            },
        ]
    )
    sparse_rollup = sparse_distribution["reason_rollup_by_transition_label"][
        "Fragile after promotion"
    ]
    assert sparse_distribution["minimum_support_met"] is False
    assert sparse_distribution["support_level"] == "sparse"
    assert sparse_distribution["support_note"] == "sparse transition evidence"
    assert sparse_rollup["minimum_support_met"] is False
    assert sparse_rollup["support_level"] == "sparse"
    assert sparse_rollup["dominant_reasons"] == []

    supported_distribution = project_level12_transition_distribution(
        [
            {
                "case_name": f"supported_case_{idx}",
                "level12_transition_label": "Confirmed at portfolio level",
                "level12_transition_reasons": [
                    "promotion decision: Promote to Level 2",
                    "portfolio recommendation: Credible at portfolio level",
                ],
            }
            for idx in range(1, 9)
        ]
    )
    supported_rollup = supported_distribution["reason_rollup_by_transition_label"][
        "Confirmed at portfolio level"
    ]
    assert supported_distribution["minimum_support_met"] is True
    assert supported_distribution["support_level"] == "supported"
    assert supported_distribution["support_note"] == "transition evidence is well supported"
    assert supported_rollup["minimum_support_met"] is True
    assert supported_rollup["support_level"] == "supported"
    assert supported_rollup["dominant_reasons"]


def test_project_level12_transition_summary_representative_labels() -> None:
    confirmed = project_level12_transition_summary(
        {
            "factor_verdict": "Strong candidate",
            "campaign_triage": "Advance to Level 2",
            "promotion_decision": "Promote to Level 2",
            "promotion_reasons": ["gate passed"],
            "portfolio_validation_recommendation": "Credible at portfolio level",
            "portfolio_validation_robustness_label": "Robust at portfolio level",
            "portfolio_validation_support_reasons": ["baseline return is positive"],
            "portfolio_validation_fragility_reasons": [],
            "portfolio_validation_major_risks": [],
            "portfolio_validation_cost_sensitivity_note": (
                "Portfolio return remains positive across tested transaction-cost rates."
            ),
        }
    )
    assert confirmed["transition_label"] == "Confirmed at portfolio level"
    assert (
        "portfolio-level evidence confirms" in confirmed["confirmation_vs_degradation_note"].lower()
    )

    weakened = project_level12_transition_summary(
        {
            "factor_verdict": "Strong candidate",
            "campaign_triage": "Advance to Level 2",
            "promotion_decision": "Promote to Level 2",
            "portfolio_validation_recommendation": "Needs portfolio refinement",
            "portfolio_validation_robustness_label": "Credible but sensitive",
            "portfolio_validation_fragility_reasons": ["weighting sensitivity is material"],
            "portfolio_validation_major_risks": ["net return is near zero after costs"],
            "portfolio_validation_cost_sensitivity_note": (
                "Return turns negative under higher transaction costs."
            ),
        }
    )
    assert weakened["transition_label"] == "Weakened at portfolio level"
    assert any("cost sensitivity:" in reason for reason in weakened["key_transition_reasons"])

    fragile = project_level12_transition_summary(
        {
            "factor_verdict": "Strong candidate",
            "campaign_triage": "Advance to Level 2",
            "promotion_decision": "Promote to Level 2",
            "portfolio_validation_recommendation": "Credible at portfolio level",
            "portfolio_validation_robustness_label": "Fragile at portfolio level",
            "portfolio_validation_fragility_reasons": ["concentration risk is elevated"],
            "portfolio_validation_concentration_turnover_note": (
                "Top holdings concentration is high relative to policy."
            ),
        }
    )
    assert fragile["transition_label"] == "Fragile after promotion"
    assert any("concentration/turnover:" in reason for reason in fragile["key_transition_reasons"])

    improved = project_level12_transition_summary(
        {
            "factor_verdict": "Mixed evidence",
            "campaign_triage": "Needs refinement",
            "promotion_decision": "Promote to Level 2",
            "portfolio_validation_recommendation": "Credible at portfolio level",
            "portfolio_validation_robustness_label": "Robust at portfolio level",
            "portfolio_validation_support_reasons": ["benchmark-relative evidence is supportive"],
            "portfolio_validation_benchmark_support_note": (
                "Benchmark-relative evidence supports standalone strength."
            ),
        }
    )
    assert improved["transition_label"] == "Improved at portfolio level"
    assert any("benchmark-relative:" in reason for reason in improved["key_transition_reasons"])

    inconclusive = project_level12_transition_summary(
        {
            "factor_verdict": "Strong candidate",
            "campaign_triage": "Advance to Level 2",
            "promotion_decision": "Hold for refinement",
            "promotion_blockers": ["blocked by unstable rolling evidence"],
            "portfolio_validation_recommendation": "Not evaluated (not promoted)",
        }
    )
    assert inconclusive["transition_label"] == "Inconclusive transition"
