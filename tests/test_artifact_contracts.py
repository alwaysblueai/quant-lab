from __future__ import annotations

import copy

import pytest

from alpha_lab.artifact_contracts import validate_level12_artifact_payload


def _valid_portfolio_summary() -> dict[str, object]:
    return {
        "validation_status": "completed",
        "promotion_decision": "Promote to Level 2",
        "recommendation": "Credible at portfolio level",
        "remains_credible_at_portfolio_level": True,
        "major_risks": [],
        "major_caveats": [],
        "portfolio_robustness_summary": {
            "taxonomy_label": "Robust at portfolio level",
            "support_reasons": ["baseline portfolio return is positive"],
            "fragility_reasons": [],
            "scenario_sensitivity_notes": ["holding period sensitivity is stable"],
            "benchmark_relative_support_note": "Benchmark-relative support is unavailable",
            "cost_sensitivity_note": "return remains positive across tested cost rates",
            "concentration_turnover_risk_note": "turnover and concentration within guardrails",
        },
    }


def _valid_portfolio_metrics() -> dict[str, object]:
    return {
        "protocol_settings": {"holding_period_sensitivity": [1, 5]},
        "scenario_metrics": [
            {
                "weighting_method": "rank",
                "holding_period": 1,
                "mean_portfolio_return": 0.0016,
            }
        ],
        "holding_period_sensitivity": [{"holding_period": 1}],
        "weighting_sensitivity": [{"weighting_method": "rank"}],
        "turnover_summary": {"scenario_mean_turnover_min": 0.20},
        "transaction_cost_sensitivity": {"review_cost_rate": 0.0010},
        "benchmark_relative_evaluation": {"status": "not_available"},
    }


def _valid_portfolio_package() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "package_type": "alpha_lab_level2_portfolio_validation_package",
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "input_case_identity": {"case_name": "demo_case"},
        "promotion_decision_context": {"decision": "Promote to Level 2"},
        "portfolio_validation_settings": {"holding_period_grid": [1, 5]},
        "key_portfolio_results": {"baseline_scenario": {"holding_period": 1}},
        "major_risks": [],
        "major_caveats": [],
        "portfolio_robustness_summary": {
            "taxonomy_label": "Robust at portfolio level",
            "support_reasons": ["baseline portfolio return is positive"],
            "fragility_reasons": [],
            "scenario_sensitivity_notes": ["holding period sensitivity is stable"],
            "benchmark_relative_support_note": "Benchmark-relative support is unavailable",
            "cost_sensitivity_note": "return remains positive across tested cost rates",
            "concentration_turnover_risk_note": "turnover and concentration within guardrails",
        },
        "recommendation": {"label": "Credible at portfolio level"},
    }


def _valid_metrics_payload() -> dict[str, object]:
    return {
        "metrics": {
            "research_evaluation_profile": "default_research",
            "campaign_triage": "Advance to Level 2",
            "promotion_decision": "Promote to Level 2",
            "ic_t_stat": 2.1,
            "ic_p_value": 0.038,
            "dsr_pvalue": 0.12,
            "split_description": "train<=2021-12-31 / test>=2022-01-01",
            "data_quality_status": "warn",
            "data_quality_suspended_rows": 12,
            "data_quality_stale_rows": 4,
            "data_quality_suspected_split_rows": 0,
            "data_quality_integrity_warn_count": 1,
            "data_quality_integrity_fail_count": 0,
            "data_quality_hard_fail_count": 0,
            "promotion_reasons": ["robust evidence"],
            "promotion_blockers": [],
        },
        "coverage_by_date_summary": {
            "n_dates": 42,
            "mean_coverage": 0.85,
            "min_coverage": 0.70,
        },
        "portfolio_validation_summary": _valid_portfolio_summary(),
        "portfolio_validation_metrics": _valid_portfolio_metrics(),
        "portfolio_validation_package": _valid_portfolio_package(),
    }


def _valid_factor_definition_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_factor_definition",
        "case_name": "bp_single_factor_v1",
        "package_type": "single_factor",
        "factor_name": "bp",
        "spec": {"name": "bp_single_factor_v1", "factor_name": "bp"},
        "source_artifacts": {
            "factor_definition_yaml_path": "/tmp/factor_definition.yaml",
            "run_manifest_path": "/tmp/run_manifest.json",
        },
        "fallback_derived_fields": [],
    }


def _valid_signal_validation_payload() -> dict[str, object]:
    metrics_payload = _valid_metrics_payload()
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_signal_validation",
        "case_name": "bp_single_factor_v1",
        "package_type": "single_factor",
        "metrics": metrics_payload["metrics"],
        "coverage_by_date_summary": metrics_payload["coverage_by_date_summary"],
        "neutralization_summary": [],
        "source_artifacts": {
            "metrics_path": "/tmp/metrics.json",
            "ic_timeseries_path": "/tmp/ic_timeseries.csv",
        },
        "fallback_derived_fields": [],
    }


def _valid_portfolio_recipe_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_portfolio_recipe",
        "case_name": "bp_single_factor_v1",
        "package_type": "single_factor",
        "recipe_context": {
            "factor_name": "bp",
            "rebalance_frequency": "W",
            "target_horizon": 5,
            "neutralization_enabled": False,
        },
        "portfolio_validation_summary": _valid_portfolio_summary(),
        "portfolio_validation_metrics": _valid_portfolio_metrics(),
        "portfolio_validation_package": _valid_portfolio_package(),
        "turnover_penalty_settings": "warn if mean turnover > 0.30",
        "transaction_cost_assumptions": "one-way=0.0010; grid=0.0005,0.001,0.002",
        "position_limits": "max|w|~0.1200; effective names~24.0000",
        "source_artifacts": {
            "portfolio_validation_summary_path": "/tmp/portfolio_validation_summary.json",
            "portfolio_validation_metrics_path": "/tmp/portfolio_validation_metrics.json",
            "portfolio_validation_package_path": "/tmp/portfolio_validation_package.json",
        },
        "fallback_derived_fields": [],
    }


def _valid_backtest_result_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_backtest_result",
        "case_name": "bp_single_factor_v1",
        "package_type": "single_factor",
        "rebalance_frequency": "W",
        "summary": {
            "annualized_return": None,
            "annualized_volatility": None,
            "sharpe": 0.9,
            "sortino": None,
            "max_drawdown": None,
            "calmar": None,
            "win_rate": 0.55,
            "turnover": 0.12,
            "information_ratio": 0.45,
            "excess_return_vs_benchmark": 0.01,
            "tracking_error": 0.03,
            "pre_cost_return": 0.002,
            "post_cost_return": 0.0015,
            "rolling_sharpe": None,
            "rolling_drawdown": None,
            "subperiod_analysis": None,
            "regime_analysis": None,
            "nav_points": [["2026-01-01", 1.0], ["2026-01-08", 1.02]],
            "monthly_return_table": [["2026-01", 0.015]],
            "drawdown_table": [["2026-01-08", -0.01]],
        },
        "source_artifacts": {
            "group_returns_path": "/tmp/group_returns.csv",
            "turnover_path": "/tmp/turnover.csv",
            "metrics_path": "/tmp/metrics.json",
        },
        "fallback_derived_fields": [
            "annualized_return",
            "annualized_volatility",
            "sortino",
        ],
    }


def _valid_run_manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "real_case_single_factor_bundle",
        "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
        "case_name": "bp_single_factor_v1",
        "outputs": {
            "run_manifest": "/tmp/run_manifest.json",
            "metrics": "/tmp/metrics.json",
        },
        "required_bundle_files": [
            "run_manifest.json",
            "metrics.json",
            "level2_portfolio_validation/portfolio_validation_summary.json",
        ],
        "integrity_summary": {"status": "passed"},
        "evaluation_standard": {
            "profile_name": "default_research",
            "snapshot": {"factor_verdict": {"min_ic": 0.01}},
        },
        "spec_path": None,
    }


def _valid_campaign_manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "campaign_name": "research_campaign_1",
        "campaign_description": "campaign validation",
        "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
        "execution_order": ["bp_single_factor_v1"],
        "output_root_dir": "/tmp/campaign",
        "cases": [
            {
                "case_name": "bp_single_factor_v1",
                "package_type": "single_factor",
                "spec_path": "/tmp/spec.yaml",
            }
        ],
        "evaluation_standard": {
            "profile_name": "default_research",
            "snapshot": {"campaign_triage": {"min_icir": 0.2}},
        },
    }


def _valid_campaign_results_payload() -> dict[str, object]:
    return {
        "campaign_name": "research_campaign_1",
        "run_timestamp_utc": "2026-01-01T00:00:00+00:00",
        "n_cases": 1,
        "n_success": 1,
        "n_failed": 0,
        "n_skipped": 0,
        "evaluation_profile": "default_research",
        "cases": [
            {
                "case_name": "bp_single_factor_v1",
                "package_type": "single_factor",
                "status": "success",
                "key_metrics": {
                    "research_evaluation_profile": "default_research",
                    "campaign_triage": "Advance to Level 2",
                    "promotion_decision": "Promote to Level 2",
                },
            }
        ],
    }


def _valid_research_validation_package_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "package_type": "alpha_lab_research_validation_package",
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "case_id": "bp_single_factor_v1",
        "case_name": "bp_single_factor_v1",
        "case_output_dir": "/tmp/case",
        "workflow_type": "run-single-factor",
        "experiment_name": "bp_single_factor_v1",
        "identity": {
            "workflow_summary_path": None,
            "portfolio_validation_package_path": None,
        },
        "research_intent": {
            "config_path": None,
            "workflow_type": "run-single-factor",
            "promotion_verdict": "Promote to Level 2",
            "portfolio_validation_status": "completed",
            "evaluation_profile": "default_research",
        },
        "research_results": {
            "key_metrics": {},
            "evaluation_standard": {
                "profile_name": "default_research",
                "snapshot": {"factor_verdict": {"min_ic": 0.01}},
            },
            "portfolio_validation_summary": _valid_portfolio_summary(),
            "portfolio_validation_metrics": _valid_portfolio_metrics(),
            "portfolio_validation_package": _valid_portfolio_package(),
        },
        "trial_registry_metadata": {
            "trial_log_path": None,
            "alpha_registry_path": None,
        },
        "artifact_index": [
            {
                "name": "workflow_summary_json",
                "artifact_type": "workflow_summary",
                "path": "",
                "exists": False,
                "required": True,
            }
        ],
        "interpretation": None,
        "notes": None,
    }


def _valid_campaign_profile_comparison_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "workflow_name": "campaign_profile_comparison",
        "source": "campaign",
        "output_root_dir": "/tmp/campaign_profile_comparison",
        "profiles": ["exploratory_screening", "default_research"],
        "cases": [
            {
                "case_name": "bp_single_factor_v1",
                "case_description": "demo",
                "spec_path": "/tmp/spec.yaml",
            }
        ],
        "profile_runs": [
            {
                "profile_name": "default_research",
                "ranked_case_order": ["bp_single_factor_v1"],
                "case_rows": [],
            }
        ],
        "case_comparison": [
            {
                "case_name": "bp_single_factor_v1",
                "changed_fields": [],
                "level12_transition_profile_delta": {
                    "delta_label": "transition_stable",
                    "profile_transition_labels": {
                        "exploratory_screening": "Weakened at portfolio level",
                        "default_research": "Weakened at portfolio level",
                    },
                    "profile_pair_directions": [
                        {
                            "from_profile": "exploratory_screening",
                            "to_profile": "default_research",
                            "from_label": "Weakened at portfolio level",
                            "to_label": "Weakened at portfolio level",
                            "direction": "stable",
                        }
                    ],
                },
                "profiles": {},
            }
        ],
        "field_change_index": {
            "factor_verdict": [],
            "campaign_triage": [],
            "promotion_decision": [],
            "portfolio_validation_recommendation": [],
        },
        "campaign_level_summary": {
            "stable_cases": ["bp_single_factor_v1"],
            "profile_sensitive_cases": [],
            "highly_profile_sensitive": [],
            "transition_stable_cases": ["bp_single_factor_v1"],
            "transition_sensitive_cases": [],
            "transition_delta_label_counts": {
                "transition_stable": 1,
                "transition_weakened_under_stricter_profile": 0,
                "transition_improved_under_profile_change": 0,
                "transition_mixed_or_nonmonotonic": 0,
            },
            "level12_transition_profile_delta_matrix": {
                "profile_pairs": [
                    {
                        "from_profile": "exploratory_screening",
                        "to_profile": "default_research",
                        "n_cases_compared": 1,
                        "n_cases_with_observed_transition_labels": 1,
                        "n_cases_missing_transition_labels": 0,
                        "stable_count": 1,
                        "changed_count": 0,
                        "counts_by_from_to_label": {
                            "Weakened at portfolio level": {
                                "Weakened at portfolio level": 1
                            }
                        },
                        "proportions_by_from_to_label": {
                            "Weakened at portfolio level": {
                                "Weakened at portfolio level": 1.0
                            }
                        },
                    }
                ]
            },
            "level12_transition_reason_profile_delta_matrix": {
                "profile_pairs": [
                    {
                        "from_profile": "exploratory_screening",
                        "to_profile": "default_research",
                        "n_transition_labels_with_observed_reasons": 1,
                        "n_transition_labels_with_reason_shift": 1,
                        "n_transition_labels_reason_stable": 0,
                        "reason_bucket_delta_counts": {
                            "added": 1,
                            "removed": 0,
                            "increased": 0,
                            "decreased": 0,
                            "stable": 0,
                        },
                        "reason_delta_by_transition_label": {
                            "Weakened at portfolio level": {
                                "from_profile_n_cases_with_label": 1,
                                "to_profile_n_cases_with_label": 1,
                                "from_profile_dominant_reasons": [
                                    {
                                        "reason": "factor verdict is strong",
                                        "count": 1,
                                        "n_cases_with_label": 1,
                                        "proportion_of_label_cases": 1.0,
                                    }
                                ],
                                "to_profile_dominant_reasons": [
                                    {
                                        "reason": "factor verdict is strong",
                                        "count": 1,
                                        "n_cases_with_label": 1,
                                        "proportion_of_label_cases": 1.0,
                                    },
                                    {
                                        "reason": "benchmark-relative risk is elevated",
                                        "count": 1,
                                        "n_cases_with_label": 1,
                                        "proportion_of_label_cases": 1.0,
                                    },
                                ],
                                "reason_bucket_deltas": {
                                    "added": [
                                        {
                                            "reason": "benchmark-relative risk is elevated",
                                            "from_count": 0,
                                            "from_n_cases_with_label": 1,
                                            "from_proportion_of_label_cases": 0.0,
                                            "to_count": 1,
                                            "to_n_cases_with_label": 1,
                                            "to_proportion_of_label_cases": 1.0,
                                            "delta_count": 1,
                                            "delta_proportion_of_label_cases": 1.0,
                                        }
                                    ],
                                    "removed": [],
                                    "increased": [],
                                    "decreased": [],
                                    "stable": [
                                        {
                                            "reason": "factor verdict is strong",
                                            "from_count": 1,
                                            "from_n_cases_with_label": 1,
                                            "from_proportion_of_label_cases": 1.0,
                                            "to_count": 1,
                                            "to_n_cases_with_label": 1,
                                            "to_proportion_of_label_cases": 1.0,
                                            "delta_count": 0,
                                            "delta_proportion_of_label_cases": 0.0,
                                        }
                                    ],
                                },
                                "is_reason_shifted": True,
                            }
                        },
                    }
                ]
            },
        },
    }


def _valid_factor_set_result_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_factor_set_result",
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "comparison_json_path": "/tmp/campaign_profile_comparison.json",
        "default_profile": "default_research",
        "source_artifacts": {
            "campaign_profile_comparison_json_path": "/tmp/campaign_profile_comparison.json",
            "factor_shortlist_reference": "campaign_profile_comparison.json#case_comparison",
        },
        "policy": {
            "policy_id": "factor_set_construction_v1",
            "formula_text": "selected_set=top_keep_factors_by_score",
        },
        "factor_sets": [
            {
                "factor_set_id": "set-selected-core-v1",
                "label_zh": "入选核心集合",
                "factor_ids": ["f1", "f2"],
                "factor_names": ["factor_1", "factor_2"],
                "source_shortlist_entries": ["f1#rank=1#rec=keep"],
                "construction_rule": "selected_core_top_keep_by_shortlist_score",
                "status": "selected",
                "rationale": ["high signal quality"],
                "rationale_zh": ["信号质量较高"],
                "warnings": [],
                "score_summary": {
                    "mean_shortlist_score": 0.82,
                    "mean_icir": 0.71,
                    "mean_turnover": 0.32,
                    "mean_oos_stability_share": 0.74,
                    "max_pair_correlation": 0.40,
                    "family_balance_ratio": 1.00,
                },
            }
        ],
        "selected_factor_set_ids": ["set-selected-core-v1"],
        "recommendation_summary": ["1. set-selected-core-v1"],
    }


def _valid_candidate_recipe_generation_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_candidate_recipe_generation",
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "comparison_json_path": "/tmp/campaign_profile_comparison.json",
        "default_profile": "default_research",
        "source_artifacts": {
            "campaign_profile_comparison_json_path": "/tmp/campaign_profile_comparison.json",
            "factor_set_result_json_path": "/tmp/factor_set_result.json",
        },
        "policy": {
            "policy_id": "candidate_recipe_generation_v1",
            "formula_text": "emit deterministic recipe variants",
        },
        "generated_recipes": [
            {
                "recipe_id": "candidate-set-selected-core-v1-v1",
                "recipe_name": "Candidate set-selected-core-v1 v1",
                "source_factor_set_id": "set-selected-core-v1",
                "source_factor_ids": ["f1", "f2"],
                "construction_variant": "baseline_rank_neutralized_strict",
                "weighting_scheme": "rank",
                "neutralization_mode": "neutralization_on",
                "turnover_penalty_mode": "strict",
                "benchmark_mode": "benchmark_relative",
                "rationale": ["source factor set selected"],
                "assumptions": ["cost model inherited"],
                "warnings": [],
            }
        ],
        "recommendation_summary": ["1. candidate-set-selected-core-v1-v1"],
    }


def _valid_winner_selection_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_winner_selection",
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "comparison_json_path": "/tmp/campaign_profile_comparison.json",
        "default_profile": "default_research",
        "source_artifacts": {
            "campaign_profile_comparison_json_path": "/tmp/campaign_profile_comparison.json",
            "candidate_recipe_generation_json_path": "/tmp/candidate_recipe_generation.json",
        },
        "decision_policy": {
            "decision_policy_id": "winner_selection_policy_v1",
            "policy_formula_text": "composite = weighted_average(...)",
            "component_weights": [["sharpe", 0.30], ["post_cost_return", 0.25]],
        },
        "winner_recipe_id": "recipe-a",
        "challenger_recipe_ids": ["recipe-b"],
        "watchlist_recipe_ids": ["candidate-set-selected-core-v1-v1"],
        "rejected_recipe_ids": ["recipe-c"],
        "decision_reasons": ["recipe-a selected as winner"],
        "decision_reasons_zh": ["recipe-a 被选为冠军方案"],
        "challenger_reasons": ["recipe-b remains competitive"],
        "challenger_reasons_zh": ["recipe-b 仍具竞争力"],
        "rejection_reasons": ["recipe-c failed drawdown guardrail"],
        "rejection_reasons_zh": ["recipe-c 未通过回撤护栏"],
        "next_actions": ["promote recipe-a to deeper validation"],
        "next_actions_zh": ["推进 recipe-a 进入更深入验证"],
        "missing_data_caveats": [],
        "score_table": [
            {"recipe_id": "recipe-a", "composite_score": 0.88},
            {"recipe_id": "candidate-set-selected-core-v1-v1", "composite_score": None},
        ],
    }


def _valid_next_step_recommendations_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_next_step_recommendations",
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "comparison_json_path": "/tmp/campaign_profile_comparison.json",
        "default_profile": "default_research",
        "source_artifacts": {
            "campaign_profile_comparison_json_path": "/tmp/campaign_profile_comparison.json",
            "winner_selection_json_path": "/tmp/winner_selection.json",
        },
        "policy": {
            "policy_id": "next_step_policy_v1",
            "policy_formula_text": "recommendations = deterministic_rules(...)",
        },
        "recommendations": [
            {
                "recommendation_id": "rec-01",
                "category": "promotion",
                "priority": "P1",
                "label_zh": "晋级推进",
                "action": "promote recipe-a to deeper validation",
                "action_text_zh": "推进 recipe-a 进入更深入验证",
                "rationale": "winner selected by explicit policy",
                "rationale_zh": "基于显式策略选出冠军方案",
                "triggered_by": ["recipe-a", "winner_selection_policy_v1"],
                "supporting_evidence": ["winner_score=0.88"],
            }
        ],
        "summary": ["1. [P1] promote recipe-a to deeper validation"],
        "summary_zh": ["1. [高优先级 (P1)] 推进 recipe-a 进入更深入验证"],
    }


def _valid_artifact_load_diagnostics_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_artifact_load_diagnostics",
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "comparison_json_path": "/tmp/campaign_profile_comparison.json",
        "default_profile": "default_research",
        "artifact_load_mode": "permissive",
        "artifact_load_policy_summary": [
            "mode=permissive",
            "require_canonical_artifacts=no",
        ],
        "source_artifacts": {
            "campaign_profile_comparison_json_path": "/tmp/campaign_profile_comparison.json",
            "dashboard_html_path": "/tmp/campaign_profile_dashboard_zh.html",
        },
        "diagnostics": [
            {
                "code": "MISSING_CANONICAL_ARTIFACT",
                "severity": "warning",
                "artifact_type": "canonical_artifact",
                "object_scope": "factor_definition",
                "message": "case_alpha: missing artifact path (factor_definition.json)",
                "path": None,
                "case_name": "case_alpha",
                "profile_name": "default_research",
                "mode": "permissive",
                "fallback_used": True,
                "remediation_hint": "Persist factor_definition.json before strict mode.",
            }
        ],
    }


def _valid_research_artifact_manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_research_artifact_manifest",
        "generated_at_utc": "2026-01-01T00:00:00+00:00",
        "comparison_json_path": "/tmp/campaign_profile_comparison.json",
        "default_profile": "default_research",
        "artifact_entries": [
            {
                "artifact_name": "factor_definition.json",
                "artifact_type": "alpha_lab_factor_definition",
                "artifact_layer": "canonical",
                "path": "/tmp/runs/default_research/case_a/factor_definition.json",
                "scope": "case",
                "case_name": "case_a",
                "profile_name": "default_research",
                "producer_hint": (
                    "alpha_lab.real_cases.single_factor.artifacts."
                    "export_artifact_bundle"
                ),
                "validation_status": "valid",
                "required_in_strict_mode": True,
                "lineage_role": "factor_definition",
            },
            {
                "artifact_name": "artifact_load_diagnostics.json",
                "artifact_type": "alpha_lab_artifact_load_diagnostics",
                "artifact_layer": "governance",
                "path": "/tmp/artifact_load_diagnostics.json",
                "scope": "comparison",
                "case_name": None,
                "profile_name": "default_research",
                "producer_hint": (
                    "alpha_lab.reporting.renderers.campaign_profile_dashboard."
                    "persist_workflow_closure_artifacts"
                ),
                "validation_status": "valid",
                "required_in_strict_mode": True,
                "lineage_role": "artifact_load_diagnostics",
            },
        ],
        "summary": {
            "total_entries": 2,
            "by_layer": {
                "canonical": 1,
                "governance": 1,
            },
            "by_artifact_type": {
                "alpha_lab_factor_definition": 1,
                "alpha_lab_artifact_load_diagnostics": 1,
            },
            "by_validation_status": {
                "valid": 2,
            },
        },
    }


@pytest.mark.parametrize(
    ("artifact_name", "payload"),
    [
        ("run_manifest.json", _valid_run_manifest_payload()),
        ("metrics.json", _valid_metrics_payload()),
        ("factor_definition.json", _valid_factor_definition_payload()),
        ("signal_validation.json", _valid_signal_validation_payload()),
        ("portfolio_recipe.json", _valid_portfolio_recipe_payload()),
        ("backtest_result.json", _valid_backtest_result_payload()),
        ("campaign_manifest.json", _valid_campaign_manifest_payload()),
        ("campaign_results.json", _valid_campaign_results_payload()),
        ("research_validation_package.json", _valid_research_validation_package_payload()),
        ("portfolio_validation_summary.json", _valid_portfolio_summary()),
        ("portfolio_validation_metrics.json", _valid_portfolio_metrics()),
        ("portfolio_validation_package.json", _valid_portfolio_package()),
        ("campaign_profile_comparison.json", _valid_campaign_profile_comparison_payload()),
        ("factor_set_result.json", _valid_factor_set_result_payload()),
        (
            "candidate_recipe_generation.json",
            _valid_candidate_recipe_generation_payload(),
        ),
        ("winner_selection.json", _valid_winner_selection_payload()),
        (
            "next_step_recommendations.json",
            _valid_next_step_recommendations_payload(),
        ),
        (
            "artifact_load_diagnostics.json",
            _valid_artifact_load_diagnostics_payload(),
        ),
        (
            "research_artifact_manifest.json",
            _valid_research_artifact_manifest_payload(),
        ),
    ],
)
def test_level12_artifact_validators_accept_valid_payloads(
    artifact_name: str,
    payload: dict[str, object],
) -> None:
    validate_level12_artifact_payload(payload, artifact_name=artifact_name)


def test_level12_artifact_validator_rejects_missing_required_metrics_field() -> None:
    payload = _valid_metrics_payload()
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    metrics.pop("promotion_decision")

    with pytest.raises(ValueError, match=r"metrics\.json\.metrics\.promotion_decision"):
        validate_level12_artifact_payload(payload, artifact_name="metrics.json")


def test_level12_artifact_validator_rejects_invalid_data_quality_status() -> None:
    payload = _valid_metrics_payload()
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    metrics["data_quality_status"] = "unknown"

    with pytest.raises(ValueError, match=r"metrics\.json\.metrics\.data_quality_status"):
        validate_level12_artifact_payload(payload, artifact_name="metrics.json")


def test_level12_artifact_validator_rejects_non_int_data_quality_counter() -> None:
    payload = _valid_metrics_payload()
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    metrics["data_quality_stale_rows"] = "4"

    with pytest.raises(ValueError, match=r"metrics\.json\.metrics\.data_quality_stale_rows"):
        validate_level12_artifact_payload(payload, artifact_name="metrics.json")


def test_level12_artifact_validator_rejects_non_numeric_dsr_pvalue() -> None:
    payload = _valid_metrics_payload()
    metrics = payload["metrics"]
    assert isinstance(metrics, dict)
    metrics["dsr_pvalue"] = "0.12"

    with pytest.raises(ValueError, match=r"metrics\.json\.metrics\.dsr_pvalue"):
        validate_level12_artifact_payload(payload, artifact_name="metrics.json")


def test_level12_artifact_validator_rejects_malformed_campaign_case() -> None:
    payload = _valid_campaign_results_payload()
    cases = payload["cases"]
    assert isinstance(cases, list)
    case_row = cases[0]
    assert isinstance(case_row, dict)
    case_row["key_metrics"] = "invalid"

    with pytest.raises(ValueError, match=r"campaign_results\.json\.cases\[0\]\.key_metrics"):
        validate_level12_artifact_payload(payload, artifact_name="campaign_results.json")


def test_campaign_profile_comparison_validator_accepts_example_variant() -> None:
    payload = _valid_campaign_profile_comparison_payload()
    payload.pop("workflow_name")
    payload.pop("source")
    payload["example_name"] = "profile_aware_campaign_level12"

    validate_level12_artifact_payload(payload, artifact_name="campaign_profile_comparison.json")


def test_campaign_profile_comparison_validator_rejects_invalid_changed_fields() -> None:
    payload = _valid_campaign_profile_comparison_payload()
    bad_payload = copy.deepcopy(payload)
    case_comparison = bad_payload["case_comparison"]
    assert isinstance(case_comparison, list)
    row = case_comparison[0]
    assert isinstance(row, dict)
    row["changed_fields"] = "campaign_triage"

    with pytest.raises(
        ValueError,
        match=r"campaign_profile_comparison\.json\.case_comparison\[0\]\.changed_fields",
    ):
        validate_level12_artifact_payload(
            bad_payload,
            artifact_name="campaign_profile_comparison.json",
        )


def test_backtest_result_validator_rejects_invalid_nav_point_shape() -> None:
    payload = _valid_backtest_result_payload()
    summary = payload["summary"]
    assert isinstance(summary, dict)
    summary["nav_points"] = [["2026-01-01"]]

    with pytest.raises(
        ValueError,
        match=r"backtest_result\.json\.summary\.nav_points\[0\]",
    ):
        validate_level12_artifact_payload(
            payload,
            artifact_name="backtest_result.json",
        )


def test_research_artifact_manifest_validator_rejects_invalid_layer() -> None:
    payload = _valid_research_artifact_manifest_payload()
    entries = payload["artifact_entries"]
    assert isinstance(entries, list)
    first = entries[0]
    assert isinstance(first, dict)
    first["artifact_layer"] = "invalid"

    with pytest.raises(
        ValueError,
        match=r"research_artifact_manifest\.json\.artifact_entries\[0\]\.artifact_layer",
    ):
        validate_level12_artifact_payload(
            payload,
            artifact_name="research_artifact_manifest.json",
        )


def test_portfolio_recipe_validator_rejects_invalid_recipe_control_shape() -> None:
    payload = _valid_portfolio_recipe_payload()
    payload["position_limits"] = 123

    with pytest.raises(
        ValueError,
        match=r"portfolio_recipe\.json\.position_limits",
    ):
        validate_level12_artifact_payload(
            payload,
            artifact_name="portfolio_recipe.json",
        )
