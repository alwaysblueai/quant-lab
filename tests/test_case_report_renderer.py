from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import alpha_lab.reporting.renderers.case_report as case_report_module
from alpha_lab.reporting.renderers.case_report import render_case_report, write_case_report
from alpha_lab.reporting.renderers.templates import CASE_SECTION_TITLES, PLACEHOLDER_OBJECTIVE


def test_case_report_renderer_loads_minimal_artifacts_and_renders_required_sections(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case_output"
    case_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        case_dir / "run_manifest.json",
        {
            "artifact_type": "real_case_single_factor_bundle",
            "case_name": "bp_single_factor_v1",
            "spec": {
                "factor_name": "bp",
                "direction": "long",
                "rebalance_frequency": "W",
                "target": {"kind": "forward_return", "horizon": 5},
                "universe": {"name": "A-share top 500"},
                "preprocess": {
                    "winsorize": True,
                    "winsorize_lower": 0.01,
                    "winsorize_upper": 0.99,
                    "standardization": "zscore",
                    "min_group_size": 3,
                },
                "neutralization": {"enabled": False},
                "transaction_cost": {"one_way_rate": 0.001},
            },
        },
    )

    _write_json(
        case_dir / "metrics.json",
        {
            "metrics": {
                "mean_ic": 0.021,
                "mean_ic_ci_lower": 0.010,
                "mean_ic_ci_upper": 0.032,
                "mean_rank_ic": 0.025,
                "mean_rank_ic_ci_lower": 0.013,
                "mean_rank_ic_ci_upper": 0.037,
                "ic_ir": 0.87,
                "mean_long_short_return": 0.0034,
                "mean_long_short_return_ci_lower": 0.0011,
                "mean_long_short_return_ci_upper": 0.0057,
                "mean_long_short_turnover": 0.22,
                "rolling_window_size": 20,
                "rolling_ic_positive_share": 0.75,
                "rolling_rank_ic_positive_share": 0.70,
                "rolling_long_short_positive_share": 0.65,
                "rolling_ic_min_mean": 0.002,
                "rolling_rank_ic_min_mean": 0.003,
                "rolling_long_short_min_mean": 0.0004,
                "rolling_instability_flags": [],
                "coverage_mean": 0.91,
                "uncertainty_flags": [],
                "uncertainty_method": "bootstrap",
                "uncertainty_confidence_level": 0.95,
                "uncertainty_bootstrap_resamples": 250,
                "neutralization_comparison": {
                    "raw": {
                        "mean_ic": 0.030,
                        "mean_rank_ic": 0.034,
                        "mean_long_short_return": 0.0040,
                        "ic_ir": 0.95,
                    },
                    "neutralized": {
                        "mean_ic": 0.021,
                        "mean_rank_ic": 0.025,
                        "mean_long_short_return": 0.0034,
                        "ic_ir": 0.87,
                    },
                    "delta": {
                        "mean_ic_delta": -0.009,
                        "mean_rank_ic_delta": -0.009,
                        "mean_long_short_return_delta": -0.0006,
                        "ic_ir_delta": -0.08,
                    },
                    "interpretation_flags": [
                        "neutralization moderately weakens evidence",
                    ],
                    "interpretation_reasons": [
                        "neutralization weakens signal but does not fully remove it",
                    ],
                },
                "factor_verdict": "Strong candidate",
                "factor_verdict_reasons": [
                    "positive IC and RankIC means",
                    "robust across subperiods",
                ],
                "promotion_decision": "Promote to Level 2",
                "promotion_reasons": [
                    "robust evidence survives neutralization",
                    "stable across rolling windows",
                ],
                "promotion_blockers": [],
                "portfolio_validation_status": "completed",
                "portfolio_validation_recommendation": "Credible at portfolio level",
                "portfolio_validation_major_risks": [],
                "portfolio_validation_base_mean_portfolio_return": 0.0018,
                "portfolio_validation_base_mean_turnover": 0.24,
                "portfolio_validation_base_cost_adjusted_return_review_rate": 0.0016,
                "portfolio_validation_benchmark_relative_status": "available",
                "portfolio_validation_benchmark_relative_assessment": (
                    "supports_standalone_strength"
                ),
                "portfolio_validation_benchmark_excess_return": 0.0006,
                "portfolio_validation_benchmark_tracking_error": 0.021,
                "portfolio_validation_benchmark_relative_risks": [],
                "research_evaluation_profile": "default_research",
                "research_evaluation_snapshot": {
                    "uncertainty": {"method": "bootstrap", "confidence_level": 0.95},
                    "rolling_stability": {"rolling_window_size": 20},
                },
            }
        },
    )

    pd.DataFrame(
        [
            {"date": "2024-01-01", "group": 1, "group_return": -0.001},
            {"date": "2024-01-01", "group": 5, "group_return": 0.002},
            {"date": "2024-01-02", "group": 1, "group_return": -0.002},
            {"date": "2024-01-02", "group": 5, "group_return": 0.004},
        ]
    ).to_csv(case_dir / "group_returns.csv", index=False)

    (case_dir / "summary.md").write_text(
        "# Summary\n\nObjective: Capture valuation mean reversion.\n",
        encoding="utf-8",
    )

    report = render_case_report(case_dir)
    for section in CASE_SECTION_TITLES:
        assert f"## {section}" in report
    assert "bp_single_factor_v1" in report
    assert "`single_factor`" in report
    assert "Capture valuation mean reversion" in report
    assert "Factor Verdict" in report
    assert "Verdict Reasons" in report
    assert "Level 2 Promotion" in report
    assert "Promotion Reasons" in report
    assert "Promotion Blockers" in report
    assert "Level 1->Level 2 Transition" in report
    assert "Transition Interpretation" in report
    assert "Transition Reasons" in report
    assert "Confirmation vs Degradation" in report
    assert "Confirmed at portfolio level" in report
    assert "Level 2 Portfolio Validation" in report
    assert "Portfolio Robustness Taxonomy" in report
    assert "Portfolio Robustness Supports" in report
    assert "Portfolio Robustness Fragilities" in report
    assert "Portfolio Scenario Sensitivity" in report
    assert "Portfolio Benchmark Support Note" in report
    assert "Portfolio Cost Sensitivity Note" in report
    assert "Portfolio Concentration/Turnover Note" in report
    assert "Portfolio Validation Risks" in report
    assert "Portfolio Validation Baseline (Return / Turnover / Cost-Adj)" in report
    assert "Portfolio Validation Benchmark Relative" in report
    assert "Portfolio Validation Benchmark Risks" in report
    assert "Evaluation Standard" in report
    assert "default_research" in report
    assert "bootstrap" in report
    assert "Uncertainty Method" in report
    assert "IC 95% CI" in report
    assert "RankIC 95% CI" in report
    assert "Long-Short Mean 95% CI" in report
    assert "Uncertainty Flags" in report
    assert "IC / ICIR" in report
    assert "Subperiod Robustness (IC / L-S)" in report
    assert "Rolling Stability" in report
    assert "Rolling IC positive share" in report
    assert "Worst rolling IC window" in report
    assert "Raw vs Neutralized Mean IC" in report
    assert "Neutralization Comparison Flags" in report
    assert "neutralization moderately weakens evidence" in report
    assert "Instability Flags" in report

    written = write_case_report(case_dir)
    assert written.exists()
    assert written.name == "case_report.md"


def test_case_report_renderer_handles_missing_fields_gracefully(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_missing"
    case_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        case_dir / "run_manifest.json",
        {
            "artifact_type": "unknown",
            "case_name": "mystery_case",
            "spec": {},
        },
    )
    _write_json(case_dir / "metrics.json", {"metrics": {}})

    report = render_case_report(case_dir)
    assert PLACEHOLDER_OBJECTIVE in report
    assert "N/A" in report
    assert "mystery_case" in report


def test_case_report_renderer_shows_block_bootstrap_method_metadata(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_block_bootstrap"
    case_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        case_dir / "run_manifest.json",
        {
            "artifact_type": "real_case_single_factor_bundle",
            "case_name": "block_bootstrap_case",
            "spec": {"target": {}, "universe": {}, "preprocess": {}, "neutralization": {}},
        },
    )
    _write_json(
        case_dir / "metrics.json",
        {
            "metrics": {
                "factor_verdict": "Strong candidate",
                "factor_verdict_reasons": ["stable diagnostics"],
                "promotion_decision": "Promote to Level 2",
                "promotion_reasons": ["gate passed"],
                "promotion_blockers": [],
                "portfolio_validation_status": "completed",
                "portfolio_validation_recommendation": "Credible at portfolio level",
                "portfolio_validation_major_risks": [],
                "mean_ic": 0.02,
                "mean_ic_ci_lower": 0.005,
                "mean_ic_ci_upper": 0.035,
                "mean_rank_ic_ci_lower": 0.006,
                "mean_rank_ic_ci_upper": 0.032,
                "mean_long_short_return_ci_lower": 0.0004,
                "mean_long_short_return_ci_upper": 0.0032,
                "ic_ir": 0.70,
                "uncertainty_method": "block_bootstrap",
                "uncertainty_confidence_level": 0.90,
                "uncertainty_bootstrap_resamples": 240,
                "uncertainty_bootstrap_block_length": 6,
                "rolling_window_size": 20,
                "rolling_ic_positive_share": 0.70,
                "rolling_rank_ic_positive_share": 0.65,
                "rolling_long_short_positive_share": 0.61,
                "uncertainty_flags": [],
                "neutralization_comparison": {},
            }
        },
    )

    report = render_case_report(case_dir)
    assert "block_bootstrap" in report
    assert "block_length=6" in report


def test_case_report_renderer_uses_shared_key_metrics_projections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_dir = tmp_path / "case_projection"
    case_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        case_dir / "run_manifest.json",
        {
            "artifact_type": "real_case_single_factor_bundle",
            "case_name": "projection_case",
            "spec": {"target": {}, "universe": {}, "preprocess": {}, "neutralization": {}},
        },
    )
    _write_json(
        case_dir / "metrics.json",
        {
            "metrics": {
                "factor_verdict": "Strong candidate",
                "factor_verdict_reasons": ["stable diagnostics"],
                "promotion_decision": "Promote to Level 2",
                "promotion_reasons": ["gate passed"],
                "promotion_blockers": [],
                "portfolio_validation_status": "completed",
                "portfolio_validation_recommendation": "Credible at portfolio level",
                "portfolio_validation_major_risks": [],
                "mean_ic": 0.02,
                "ic_ir": 0.7,
                "uncertainty_method": "bootstrap",
                "uncertainty_confidence_level": 0.95,
                "rolling_window_size": 20,
                "rolling_ic_positive_share": 0.7,
                "rolling_rank_ic_positive_share": 0.65,
                "rolling_long_short_positive_share": 0.61,
                "neutralization_comparison": {
                    "interpretation_flags": ["neutralization preserves most evidence"]
                },
            }
        },
    )

    calls = {"promotion_gate": 0, "profile_summary": 0, "portfolio": 0}
    original_promotion_gate = case_report_module.project_promotion_gate_metrics
    original_profile_summary = case_report_module.project_campaign_profile_summary_metrics
    original_portfolio = case_report_module.project_portfolio_validation_metrics

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
        case_report_module,
        "project_promotion_gate_metrics",
        _track_promotion_gate,
    )
    monkeypatch.setattr(
        case_report_module,
        "project_campaign_profile_summary_metrics",
        _track_profile_summary,
    )
    monkeypatch.setattr(
        case_report_module,
        "project_portfolio_validation_metrics",
        _track_portfolio,
    )

    report = render_case_report(case_dir)
    assert "projection_case" in report
    assert "Strong candidate" in report
    assert calls["promotion_gate"] >= 1
    assert calls["profile_summary"] >= 1
    assert calls["portfolio"] >= 1


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
