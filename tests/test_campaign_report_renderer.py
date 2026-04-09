from __future__ import annotations

import json
from pathlib import Path

import pytest

import alpha_lab.reporting.renderers.campaign_report as campaign_report_module
from alpha_lab.reporting.renderers.campaign_report import (
    render_campaign_report,
    write_campaign_report,
)
from alpha_lab.reporting.renderers.templates import (
    CAMPAIGN_SECTION_TITLES,
    COMPARISON_TABLE_COLUMNS,
)


def test_campaign_report_renderer_builds_comparison_table(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaign_output"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    case2_metrics = campaign_dir / "case2_metrics.json"
    _write_json(
        case2_metrics,
        {
            "metrics": {
                "mean_ic": 0.018,
                "mean_ic_ci_lower": -0.002,
                "mean_ic_ci_upper": 0.038,
                "ic_ir": 0.61,
                "mean_long_short_return": 0.0029,
                "mean_long_short_turnover": 0.31,
                "coverage_mean": 0.83,
                "uncertainty_flags": ["ic_ci_overlaps_zero"],
                "neutralization_comparison": {
                    "interpretation_flags": [
                        "neutralization materially reduces independent evidence",
                        "raw signal appears exposure-driven",
                    ]
                },
                "factor_verdict": "Promising but fragile",
                "research_evaluation_profile": "default_research",
            }
        },
    )

    _write_json(
        campaign_dir / "campaign_manifest.json",
        {
            "campaign_name": "research_campaign_1",
            "campaign_description": "Compare value, quality, and composite signals.",
        },
    )

    _write_json(
        campaign_dir / "campaign_results.json",
        {
            "campaign_name": "research_campaign_1",
            "cases": [
                {
                    "case_name": "bp_single_factor_v1",
                    "package_type": "single_factor",
                    "status": "success",
                    "key_metrics": {
                        "mean_ic": 0.02,
                        "mean_ic_ci_lower": 0.008,
                        "mean_ic_ci_upper": 0.032,
                        "ic_ir": 0.73,
                        "mean_long_short_return": 0.0033,
                        "mean_long_short_turnover": 0.20,
                        "coverage_mean": 0.90,
                        "uncertainty_flags": [],
                "neutralization_comparison_flags": [
                    "neutralization preserves most evidence"
                ],
                "factor_verdict": "Strong candidate",
                "portfolio_validation_status": "completed",
                "portfolio_validation_recommendation": "Credible at portfolio level",
                "portfolio_validation_major_risks": [],
                "portfolio_validation_benchmark_relative_status": "available",
                "portfolio_validation_benchmark_relative_assessment": (
                    "supports_standalone_strength"
                ),
                "portfolio_validation_benchmark_excess_return": 0.0005,
                "portfolio_validation_benchmark_tracking_error": 0.019,
                "research_evaluation_profile": "default_research",
            },
        },
                {
                    "case_name": "value_quality_lowvol_v1",
                    "package_type": "composite",
                    "status": "success",
                    "key_metrics": {},
                    "metrics_path": str(case2_metrics),
                },
            ],
        },
    )

    report = render_campaign_report(campaign_dir)
    for section in CAMPAIGN_SECTION_TITLES:
        assert f"## {section}" in report
    for column in COMPARISON_TABLE_COLUMNS:
        assert column in report
    assert "bp_single_factor_v1" in report
    assert "value_quality_lowvol_v1" in report
    assert "single" in report
    assert "composite" in report
    assert "Strong candidate" in report
    assert "IC 95% CI" in report
    assert "rolling IC+ share" in report
    assert "ic_ci_overlaps_zero" in report
    assert "Neutralization Comparison" in report
    assert "Campaign Triage" in report
    assert "Triage Reasons" in report
    assert "Level 2 Promotion" in report
    assert "Promotion Reasons" in report
    assert "Promotion Blockers" in report
    assert "L1->L2 Transition" in report
    assert "Level 2 Portfolio Validation" in report
    assert "Portfolio Robustness" in report
    assert "Portfolio Benchmark Relative" in report
    assert "Portfolio Validation Risks" in report
    assert "Confirmed at portfolio level" in report
    assert "L1->L2 transitions (total/observed/missing)" in report
    assert "Dominant transition reasons by label" in report
    assert "Evaluation standard profile" in report
    assert "#1" in report
    assert "Top campaign triage candidate" in report
    assert (
        "Top Level 2 promotion candidate" in report
        or "Level 2 promotion gate: no case passed this run." in report
    )
    assert (
        "Top portfolio-credible Level 2 candidate" in report
        or "Level 2 portfolio validation: no case is yet portfolio-credible." in report
    )
    assert "neutralization materially reduces independent evidence" in report

    written = write_campaign_report(campaign_dir)
    assert written.exists()
    assert written.name == "campaign_report.md"


def test_campaign_report_renderer_handles_mixed_status_and_missing_metrics(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "campaign_mixed"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        campaign_dir / "campaign_results.json",
        {
            "campaign_name": "mixed_campaign",
            "cases": [
                {
                    "case_name": "roe_ttm_single_factor_v1",
                    "package_type": "single_factor",
                    "status": "success",
                    "key_metrics": {},
                },
                {
                    "case_name": "value_quality_lowvol_v1",
                    "package_type": "composite",
                    "status": "failed",
                    "key_metrics": {},
                    "error": "missing component file",
                },
            ],
        },
    )

    report = render_campaign_report(campaign_dir)
    assert "Failure Cases / Data Issues" in report
    assert "missing component file" in report
    assert "did not complete successfully" in report
    assert "Drop for now" in report
    assert "N/A" in report
    assert "L1->L2 transitions (total/observed/missing)" in report
    assert "Dominant transition reasons by label" in report
    assert "Inconclusive transition" in report


def test_campaign_report_renderer_uses_shared_key_metrics_projections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    campaign_dir = tmp_path / "campaign_projection"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        campaign_dir / "campaign_results.json",
        {
            "campaign_name": "projection_campaign",
            "cases": [
                {
                    "case_name": "bp_projection",
                    "package_type": "single_factor",
                    "status": "success",
                    "key_metrics": {
                        "mean_ic": 0.02,
                        "ic_ir": 0.75,
                        "mean_ic_ci_lower": 0.005,
                        "mean_ic_ci_upper": 0.035,
                        "factor_verdict": "Strong candidate",
                        "factor_verdict_reasons": ["stable diagnostics"],
                        "promotion_decision": "Promote to Level 2",
                        "promotion_reasons": ["gate passed"],
                        "promotion_blockers": [],
                        "portfolio_validation_status": "completed",
                        "portfolio_validation_recommendation": (
                            "Credible at portfolio level"
                        ),
                        "portfolio_validation_major_risks": [],
                        "neutralization_comparison_flags": [
                            "neutralization preserves most evidence"
                        ],
                    },
                }
            ],
        },
    )

    calls = {"promotion_gate": 0, "profile_summary": 0, "portfolio": 0}
    original_promotion_gate = campaign_report_module.project_promotion_gate_metrics
    original_profile_summary = campaign_report_module.project_campaign_profile_summary_metrics
    original_portfolio = campaign_report_module.project_portfolio_validation_metrics

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
        campaign_report_module,
        "project_promotion_gate_metrics",
        _track_promotion_gate,
    )
    monkeypatch.setattr(
        campaign_report_module,
        "project_campaign_profile_summary_metrics",
        _track_profile_summary,
    )
    monkeypatch.setattr(
        campaign_report_module,
        "project_portfolio_validation_metrics",
        _track_portfolio,
    )

    report = render_campaign_report(campaign_dir)
    assert "bp_projection" in report
    assert "Strong candidate" in report
    assert calls["promotion_gate"] >= 1
    assert calls["profile_summary"] >= 1
    assert calls["portfolio"] >= 1


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
