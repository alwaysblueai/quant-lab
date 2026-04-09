from __future__ import annotations

import json
from pathlib import Path

import pytest

import alpha_lab.reporting.research_validation_package as research_validation_package_module
from alpha_lab.reporting.research_validation_package import (
    RESEARCH_VALIDATION_PACKAGE_TYPE,
    build_research_validation_package,
    export_research_validation_package,
)


def _write_case_outputs(case_dir: Path) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    trial_log = case_dir / "trial_log.csv"
    trial_log.write_text("date,metric\n2024-01-02,1.0\n", encoding="utf-8")
    alpha_registry = case_dir / "alpha_registry.csv"
    alpha_registry.write_text("name,value\ndemo,1\n", encoding="utf-8")
    level2_dir = case_dir / "level2_portfolio_validation"
    level2_dir.mkdir(parents=True, exist_ok=True)
    portfolio_validation_summary = level2_dir / "portfolio_validation_summary.json"
    portfolio_validation_summary.write_text(
        json.dumps(
            {
                "validation_status": "completed",
                "promotion_decision": "Promote to Level 2",
                "recommendation": "Credible at portfolio level",
                "remains_credible_at_portfolio_level": True,
                "base_mean_portfolio_return": 0.0016,
                "base_mean_turnover": 0.23,
                "base_cost_adjusted_return_review_rate": 0.0014,
                "major_risks": [],
                "major_caveats": [
                    (
                        "Research-grade portfolio approximation only; "
                        "no execution replay or fill simulation."
                    )
                ],
                "portfolio_robustness_summary": {
                    "taxonomy_label": "Robust at portfolio level",
                    "support_reasons": [
                        "baseline portfolio return is positive under default assumptions",
                    ],
                    "fragility_reasons": [],
                    "scenario_sensitivity_notes": [
                        "holding period sensitivity is stable (range=0.000200, baseline=0.001600)."
                    ],
                    "benchmark_relative_support_note": (
                        "Benchmark-relative support is unavailable in current case evidence."
                    ),
                    "cost_sensitivity_note": (
                        "Portfolio return remains positive across tested transaction-cost rates."
                    ),
                    "concentration_turnover_risk_note": (
                        "Turnover and concentration diagnostics stay within configured guardrails."
                    ),
                },
            }
        ),
        encoding="utf-8",
    )
    portfolio_validation_metrics = level2_dir / "portfolio_validation_metrics.json"
    portfolio_validation_metrics.write_text(
        json.dumps(
            {
                "protocol_settings": {"holding_period_sensitivity": [1, 5]},
                "scenario_metrics": [
                    {
                        "weighting_method": "rank",
                        "holding_period": 1,
                        "mean_portfolio_return": 0.0016,
                        "mean_turnover": 0.23,
                    }
                ],
                "holding_period_sensitivity": [
                    {"holding_period": 1, "mean_portfolio_return": 0.0016}
                ],
                "weighting_sensitivity": [
                    {"weighting_method": "rank", "mean_portfolio_return": 0.0016}
                ],
                "turnover_summary": {"scenario_mean_turnover_min": 0.23},
                "transaction_cost_sensitivity": {"review_cost_rate": 0.0010},
                "benchmark_relative_evaluation": {
                    "status": "not_available",
                    "note": "benchmark-relative metrics are not present in case evidence",
                },
            }
        ),
        encoding="utf-8",
    )
    portfolio_validation_package = level2_dir / "portfolio_validation_package.json"
    portfolio_validation_package.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "package_type": "alpha_lab_level2_portfolio_validation_package",
                "created_at_utc": "2026-01-01T00:00:00+00:00",
                "input_case_identity": {"case_name": "demo_case"},
                "promotion_decision_context": {"decision": "Promote to Level 2"},
                "portfolio_validation_settings": {"holding_period_grid": [1, 5]},
                "key_portfolio_results": {
                    "baseline_scenario": {
                        "weighting_method": "rank",
                        "holding_period": 1,
                        "mean_portfolio_return": 0.0016,
                    }
                },
                "major_risks": [],
                "major_caveats": [],
                "portfolio_robustness_summary": {
                    "taxonomy_label": "Robust at portfolio level",
                    "support_reasons": [
                        "baseline portfolio return is positive under default assumptions"
                    ],
                    "fragility_reasons": [],
                    "scenario_sensitivity_notes": [
                        "holding period sensitivity is stable (range=0.000200, baseline=0.001600)."
                    ],
                    "benchmark_relative_support_note": (
                        "Benchmark-relative support is unavailable in current case evidence."
                    ),
                    "cost_sensitivity_note": (
                        "Portfolio return remains positive across tested transaction-cost rates."
                    ),
                    "concentration_turnover_risk_note": (
                        "Turnover and concentration diagnostics stay within configured guardrails."
                    ),
                },
                "recommendation": {"label": "Credible at portfolio level"},
            }
        ),
        encoding="utf-8",
    )
    workflow = {
        "workflow": "run-single-factor",
        "experiment_name": "demo_single_factor",
        "status": "success",
        "config_path": "configs/demo.yaml",
        "key_metrics": {
            "mean_ic": 0.03,
            "mean_ic_ci_lower": 0.015,
            "mean_ic_ci_upper": 0.045,
            "mean_rank_ic": 0.04,
            "mean_rank_ic_ci_lower": 0.020,
            "mean_rank_ic_ci_upper": 0.060,
            "ic_positive_rate": 0.62,
            "rank_ic_positive_rate": 0.64,
            "ic_valid_ratio": 0.91,
            "rank_ic_valid_ratio": 0.90,
            "mean_long_short_return": 0.003,
            "mean_long_short_return_ci_lower": 0.001,
            "mean_long_short_return_ci_upper": 0.005,
            "uncertainty_method": "bootstrap",
            "uncertainty_confidence_level": 0.90,
            "uncertainty_bootstrap_resamples": 250,
            "long_short_ir": 0.70,
            "long_short_return_per_turnover": 0.010,
            "subperiod_ic_positive_share": 1.0,
            "subperiod_long_short_positive_share": 1.0,
            "rolling_window_size": 20,
            "rolling_ic_positive_share": 0.80,
            "rolling_rank_ic_positive_share": 0.78,
            "rolling_long_short_positive_share": 0.76,
            "rolling_ic_min_mean": 0.011,
            "rolling_rank_ic_min_mean": 0.013,
            "rolling_long_short_min_mean": 0.0009,
            "rolling_instability_flags": [],
            "eval_coverage_ratio_mean": 0.82,
            "eval_coverage_ratio_min": 0.74,
            "uncertainty_flags": [],
            "n_dates_used": 72,
            "neutralization_comparison": {
                "raw": {
                    "mean_ic": 0.041,
                    "mean_rank_ic": 0.050,
                    "mean_long_short_return": 0.0036,
                    "ic_ir": 0.88,
                },
                "neutralized": {
                    "mean_ic": 0.03,
                    "mean_rank_ic": 0.04,
                    "mean_long_short_return": 0.003,
                    "ic_ir": 0.70,
                },
                "delta": {
                    "mean_ic_delta": -0.011,
                    "mean_rank_ic_delta": -0.010,
                    "mean_long_short_return_delta": -0.0006,
                    "ic_ir_delta": -0.18,
                },
                "interpretation_flags": [
                    "neutralization preserves most evidence",
                ],
                "interpretation_reasons": [
                    "core evidence remains close to raw",
                ],
            },
        },
        "promotion_decision": {"verdict": "Promote to Level 2"},
        "outputs": {
            "trial_log": str(trial_log),
            "alpha_registry": str(alpha_registry),
            "portfolio_validation_summary": str(portfolio_validation_summary),
            "portfolio_validation_metrics": str(portfolio_validation_metrics),
            "portfolio_validation_package": str(portfolio_validation_package),
        },
    }
    (case_dir / "demo_single_factor_workflow_summary.json").write_text(
        json.dumps(workflow),
        encoding="utf-8",
    )


def test_build_and_export_research_validation_package(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    _write_case_outputs(case_dir)

    package = build_research_validation_package(
        case_dir,
        case_id="demo_case",
        case_name="Demo Case",
    )
    assert package.package_type == RESEARCH_VALIDATION_PACKAGE_TYPE
    assert package.workflow_type == "run-single-factor"
    assert package.experiment_name == "demo_single_factor"
    assert package.research_results["status"] == "success"
    verdict = package.research_results.get("factor_verdict")
    assert isinstance(verdict, dict)
    assert verdict.get("label") == "Strong candidate"
    uncertainty = package.research_results.get("uncertainty")
    assert isinstance(uncertainty, dict)
    assert uncertainty.get("mean_ic_ci_lower") == 0.015
    assert uncertainty.get("mean_ic_ci_upper") == 0.045
    assert uncertainty.get("uncertainty_method") == "bootstrap"
    assert uncertainty.get("uncertainty_bootstrap_resamples") == 250
    rolling = package.research_results.get("rolling_stability")
    assert isinstance(rolling, dict)
    assert rolling.get("rolling_ic_positive_share") == 0.80
    assert rolling.get("rolling_ic_min_mean") == 0.011
    neutralization = package.research_results.get("neutralization_comparison")
    assert isinstance(neutralization, dict)
    delta = neutralization.get("delta")
    assert isinstance(delta, dict)
    assert delta.get("mean_ic_delta") == -0.011
    triage = package.research_results.get("campaign_triage")
    assert isinstance(triage, dict)
    assert triage.get("campaign_triage") == "Advance to Level 2"
    assert isinstance(triage.get("campaign_triage_reasons"), list)
    promotion = package.research_results.get("level2_promotion")
    assert isinstance(promotion, dict)
    assert promotion.get("promotion_decision") == "Promote to Level 2"
    assert isinstance(promotion.get("promotion_reasons"), list)
    assert isinstance(promotion.get("promotion_blockers"), list)
    transition = package.research_results.get("level12_transition_summary")
    assert isinstance(transition, dict)
    assert transition.get("transition_label") == "Confirmed at portfolio level"
    assert isinstance(transition.get("key_transition_reasons"), tuple)
    portfolio_summary = package.research_results.get("portfolio_validation_summary")
    assert isinstance(portfolio_summary, dict)
    assert portfolio_summary.get("validation_status") == "completed"
    assert (
        portfolio_summary.get("recommendation")
        == "Credible at portfolio level"
    )
    robustness = portfolio_summary.get("portfolio_robustness_summary")
    assert isinstance(robustness, dict)
    assert robustness.get("taxonomy_label") == "Robust at portfolio level"
    portfolio_metrics = package.research_results.get("portfolio_validation_metrics")
    assert isinstance(portfolio_metrics, dict)
    assert isinstance(portfolio_metrics.get("scenario_metrics"), list)
    portfolio_package = package.research_results.get("portfolio_validation_package")
    assert isinstance(portfolio_package, dict)
    assert (
        portfolio_package.get("package_type")
        == "alpha_lab_level2_portfolio_validation_package"
    )
    evaluation_standard = package.research_results.get("evaluation_standard")
    assert isinstance(evaluation_standard, dict)
    assert evaluation_standard.get("profile_name") == "default_research"
    assert isinstance(evaluation_standard.get("snapshot"), dict)
    assert any(ref.artifact_type == "workflow_summary" for ref in package.artifact_index)

    files = export_research_validation_package(package, case_dir / "research_validation_package")
    assert files["json"].exists()
    assert files["markdown"].exists()
    assert files["portfolio_validation_summary"].exists()
    assert files["portfolio_validation_metrics"].exists()
    assert files["portfolio_validation_package"].exists()

    payload = json.loads(files["json"].read_text(encoding="utf-8"))
    assert payload["package_type"] == RESEARCH_VALIDATION_PACKAGE_TYPE
    assert payload["case_id"] == "demo_case"
    assert payload["research_results"]["factor_verdict"]["label"] == "Strong candidate"
    assert payload["research_results"]["campaign_triage"]["campaign_triage"] == "Advance to Level 2"
    assert (
        payload["research_results"]["level2_promotion"]["promotion_decision"]
        == "Promote to Level 2"
    )
    assert (
        payload["research_results"]["level12_transition_summary"]["transition_label"]
        == "Confirmed at portfolio level"
    )
    assert payload["research_results"]["evaluation_standard"]["profile_name"] == "default_research"
    assert payload["research_results"]["uncertainty"]["mean_ic_ci_lower"] == 0.015
    assert payload["research_results"]["uncertainty"]["uncertainty_method"] == "bootstrap"
    assert payload["research_results"]["rolling_stability"]["rolling_ic_positive_share"] == 0.80
    assert (
        payload["research_results"]["neutralization_comparison"]["delta"]["mean_ic_delta"]
        == -0.011
    )
    assert (
        payload["research_results"]["portfolio_validation_summary"]["validation_status"]
        == "completed"
    )
    assert (
        payload["research_results"]["portfolio_validation_summary"]["recommendation"]
        == "Credible at portfolio level"
    )

    markdown_text = files["markdown"].read_text(encoding="utf-8")
    assert "## Campaign Triage" in markdown_text
    assert "## Level 2 Promotion Gate" in markdown_text
    assert "## Level 1 to Level 2 Transition" in markdown_text
    assert "## Level 2 Portfolio Validation" in markdown_text
    assert "Portfolio robustness taxonomy" in markdown_text
    assert "Benchmark-relative evaluation status" in markdown_text
    assert "Benchmark-relative assessment" in markdown_text
    assert "## Uncertainty" in markdown_text
    assert "## Rolling Stability" in markdown_text
    assert "## Evaluation Standard" in markdown_text
    assert "## Raw vs Neutralized Comparison" in markdown_text
    assert "Mean IC 95% CI" in markdown_text
    assert "Method / CI level / bootstrap resamples / block length" in markdown_text
    assert "Advance to Level 2" in markdown_text
    assert "neutralization preserves most evidence" in markdown_text


def test_research_validation_package_uses_shared_key_metrics_projections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_dir = tmp_path / "case_projection"
    _write_case_outputs(case_dir)

    calls = {"promotion_gate": 0, "profile_summary": 0, "portfolio": 0}
    original_promotion_gate = research_validation_package_module.project_promotion_gate_metrics
    original_profile_summary = (
        research_validation_package_module.project_campaign_profile_summary_metrics
    )
    original_portfolio = research_validation_package_module.project_portfolio_validation_metrics

    def _track_promotion_gate(metrics: dict[str, object]) -> object:
        calls["promotion_gate"] += 1
        return original_promotion_gate(metrics)

    def _track_profile_summary(metrics: dict[str, object]) -> object:
        calls["profile_summary"] += 1
        return original_profile_summary(metrics)

    def _track_portfolio(metrics: dict[str, object]) -> object:
        calls["portfolio"] += 1
        return original_portfolio(metrics)

    monkeypatch.setattr(
        research_validation_package_module,
        "project_promotion_gate_metrics",
        _track_promotion_gate,
    )
    monkeypatch.setattr(
        research_validation_package_module,
        "project_campaign_profile_summary_metrics",
        _track_profile_summary,
    )
    monkeypatch.setattr(
        research_validation_package_module,
        "project_portfolio_validation_metrics",
        _track_portfolio,
    )

    package = build_research_validation_package(
        case_dir,
        case_id="demo_case_projection",
        case_name="Demo Case Projection",
    )
    files = export_research_validation_package(
        package,
        case_dir / "research_validation_projection_package",
    )
    markdown_text = files["markdown"].read_text(encoding="utf-8")

    assert "Demo Case Projection" in markdown_text
    assert calls["promotion_gate"] >= 2
    assert calls["profile_summary"] >= 2
    assert calls["portfolio"] >= 2


def test_research_validation_package_marks_non_promoted_case_as_portfolio_validation_skipped(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case_non_promoted"
    case_dir.mkdir(parents=True, exist_ok=True)
    workflow = {
        "workflow": "run-single-factor",
        "experiment_name": "demo_non_promoted",
        "status": "success",
        "config_path": "configs/demo.yaml",
        "key_metrics": {
            "mean_ic": 0.01,
            "mean_rank_ic": 0.01,
            "ic_ir": 0.20,
            "mean_long_short_return": 0.0002,
            "mean_long_short_turnover": 0.80,
            "n_quantiles": 5,
            "rebalance_frequency": "W",
        },
        "promotion_decision": {
            "verdict": "Hold for refinement",
            "reasons": ["additional robustness evidence is required before Level 2"],
            "blockers": ["blocked by unstable rolling evidence"],
            "source": "level2_promotion_gate",
        },
        "outputs": {},
    }
    (case_dir / "demo_non_promoted_workflow_summary.json").write_text(
        json.dumps(workflow),
        encoding="utf-8",
    )

    package = build_research_validation_package(
        case_dir,
        case_id="demo_non_promoted",
        case_name="Demo Non Promoted",
    )
    summary = package.research_results.get("portfolio_validation_summary")
    assert isinstance(summary, dict)
    assert summary.get("validation_status") == "skipped_not_promoted"
    assert summary.get("recommendation") == "Not evaluated (not promoted)"


def test_research_validation_package_supports_legacy_top_level_neutralization_fields(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case_legacy_neutralization"
    case_dir.mkdir(parents=True, exist_ok=True)
    workflow = {
        "workflow": "run-single-factor",
        "experiment_name": "demo_legacy_neutralization",
        "status": "success",
        "config_path": "configs/demo.yaml",
        "key_metrics": {
            "mean_ic": 0.03,
            "mean_rank_ic": 0.04,
            "ic_ir": 0.70,
            "mean_long_short_return": 0.003,
            "mean_long_short_turnover": 0.30,
            "ic_valid_ratio": 0.90,
            "rank_ic_valid_ratio": 0.89,
            "subperiod_ic_positive_share": 1.0,
            "subperiod_long_short_positive_share": 1.0,
            "coverage_mean": 0.82,
            "coverage_min": 0.72,
            "mean_ic_ci_lower": 0.010,
            "mean_ic_ci_upper": 0.040,
            "mean_rank_ic_ci_lower": 0.012,
            "mean_rank_ic_ci_upper": 0.050,
            "mean_long_short_return_ci_lower": 0.001,
            "mean_long_short_return_ci_upper": 0.005,
            "n_quantiles": 5,
            "rebalance_frequency": "W",
            "neutralization_raw_mean_ic": 0.041,
            "neutralization_raw_mean_rank_ic": 0.050,
            "neutralization_raw_mean_long_short_return": 0.0036,
            "neutralization_raw_ic_ir": 0.88,
            "neutralization_mean_ic_delta": -0.011,
            "neutralization_mean_rank_ic_delta": -0.010,
            "neutralization_mean_long_short_return_delta": -0.0006,
            "neutralization_ic_ir_delta": -0.18,
            "neutralization_comparison_flags": [
                "neutralization preserves most evidence",
            ],
            "neutralization_comparison_reasons": [
                "legacy top-level deltas retained in package output",
            ],
        },
        "outputs": {},
    }
    (case_dir / "demo_legacy_neutralization_workflow_summary.json").write_text(
        json.dumps(workflow),
        encoding="utf-8",
    )

    package = build_research_validation_package(
        case_dir,
        case_id="demo_legacy_neutralization",
        case_name="Demo Legacy Neutralization",
    )
    neutralization = package.research_results.get("neutralization_comparison")
    assert isinstance(neutralization, dict)
    raw = neutralization.get("raw")
    neutralized = neutralization.get("neutralized")
    delta = neutralization.get("delta")
    assert isinstance(raw, dict)
    assert isinstance(neutralized, dict)
    assert isinstance(delta, dict)
    assert raw.get("mean_ic") == 0.041
    assert neutralized.get("mean_ic") == 0.03
    assert delta.get("mean_ic_delta") == -0.011
    assert neutralization.get("interpretation_flags") == [
        "neutralization preserves most evidence"
    ]


def test_research_validation_package_surfaces_block_bootstrap_metadata(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case_block_bootstrap"
    _write_case_outputs(case_dir)
    workflow_path = case_dir / "demo_single_factor_workflow_summary.json"
    workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
    key_metrics = workflow["key_metrics"]
    key_metrics["uncertainty_method"] = "block_bootstrap"
    key_metrics["uncertainty_confidence_level"] = 0.90
    key_metrics["uncertainty_bootstrap_resamples"] = 320
    key_metrics["uncertainty_bootstrap_block_length"] = 6
    workflow_path.write_text(json.dumps(workflow), encoding="utf-8")

    package = build_research_validation_package(
        case_dir,
        case_id="demo_case_block",
        case_name="Demo Case Block",
    )
    uncertainty = package.research_results["uncertainty"]
    assert uncertainty["uncertainty_method"] == "block_bootstrap"
    assert uncertainty["uncertainty_bootstrap_resamples"] == 320
    assert uncertainty["uncertainty_bootstrap_block_length"] == 6

    files = export_research_validation_package(package, case_dir / "research_validation_package")
    markdown_text = files["markdown"].read_text(encoding="utf-8")
    assert "block_bootstrap" in markdown_text
    assert "Method / CI level / bootstrap resamples / block length" in markdown_text


def test_build_research_validation_package_rejects_invalid_level2_summary_schema(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case_invalid_level2_summary"
    _write_case_outputs(case_dir)
    summary_path = case_dir / "level2_portfolio_validation" / "portfolio_validation_summary.json"
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert isinstance(summary_payload, dict)
    summary_payload.pop("recommendation")
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")

    with pytest.raises(ValueError, match=r"portfolio_validation_summary\.json\.recommendation"):
        build_research_validation_package(
            case_dir,
            case_id="demo_case",
            case_name="Demo Case",
        )
