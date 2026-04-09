from __future__ import annotations

import json
from pathlib import Path

from alpha_lab.artifact_contracts import validate_level12_artifact_payload
from alpha_lab.real_cases.composite.pipeline import run_composite_case
from tests.composite_case_helpers import write_demo_composite_case


def test_composite_artifacts_have_required_files_and_fields(tmp_path: Path) -> None:
    spec_path = write_demo_composite_case(tmp_path)
    result = run_composite_case(spec_path)

    output_dir = result.output_dir
    required_files = {
        "run_manifest.json",
        "metrics.json",
        "factor_definition.json",
        "signal_validation.json",
        "portfolio_recipe.json",
        "backtest_result.json",
        "ic_timeseries.csv",
        "rolling_stability.csv",
        "group_returns.csv",
        "turnover.csv",
        "exposures.csv",
        "composite_definition.yaml",
        "summary.md",
        "experiment_card.md",
        "integrity_report.json",
        "integrity_report.md",
    }
    present_files = {p.name for p in output_dir.iterdir() if p.is_file()}
    assert required_files.issubset(present_files)
    assert (
        output_dir
        / "level2_portfolio_validation"
        / "portfolio_validation_summary.json"
    ).exists()
    assert (
        output_dir
        / "level2_portfolio_validation"
        / "portfolio_validation_metrics.json"
    ).exists()
    assert (
        output_dir
        / "level2_portfolio_validation"
        / "portfolio_validation_package.json"
    ).exists()

    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "metrics" in metrics_payload
    assert "mean_rank_ic" in metrics_payload["metrics"]
    assert "mean_long_short_return" in metrics_payload["metrics"]
    assert "mean_ic_ci_lower" in metrics_payload["metrics"]
    assert "mean_ic_ci_upper" in metrics_payload["metrics"]
    assert "mean_rank_ic_ci_lower" in metrics_payload["metrics"]
    assert "mean_rank_ic_ci_upper" in metrics_payload["metrics"]
    assert "mean_long_short_return_ci_lower" in metrics_payload["metrics"]
    assert "mean_long_short_return_ci_upper" in metrics_payload["metrics"]
    assert "uncertainty_flags" in metrics_payload["metrics"]
    assert "uncertainty_method" in metrics_payload["metrics"]
    assert "uncertainty_confidence_level" in metrics_payload["metrics"]
    assert "uncertainty_bootstrap_block_length" in metrics_payload["metrics"]
    assert "factor_verdict" in metrics_payload["metrics"]
    assert "factor_verdict_reasons" in metrics_payload["metrics"]
    assert "campaign_triage" in metrics_payload["metrics"]
    assert "campaign_triage_reasons" in metrics_payload["metrics"]
    assert "campaign_triage_priority" in metrics_payload["metrics"]
    assert "campaign_rank_primary_metric" in metrics_payload["metrics"]
    assert "campaign_rank_secondary_metric" in metrics_payload["metrics"]
    assert "campaign_rank_stability_metric" in metrics_payload["metrics"]
    assert "campaign_rank_support_count" in metrics_payload["metrics"]
    assert "campaign_rank_risk_count" in metrics_payload["metrics"]
    assert "promotion_decision" in metrics_payload["metrics"]
    assert "promotion_reasons" in metrics_payload["metrics"]
    assert "promotion_blockers" in metrics_payload["metrics"]
    assert "level12_transition_summary" in metrics_payload["metrics"]
    assert "level12_transition_label" in metrics_payload["metrics"]
    assert "level12_transition_interpretation" in metrics_payload["metrics"]
    assert "level12_transition_reasons" in metrics_payload["metrics"]
    assert "level12_transition_confirmation_note" in metrics_payload["metrics"]
    assert "portfolio_validation_status" in metrics_payload["metrics"]
    assert "portfolio_validation_recommendation" in metrics_payload["metrics"]
    assert "portfolio_validation_major_risks" in metrics_payload["metrics"]
    assert "portfolio_validation_benchmark_relative_status" in metrics_payload["metrics"]
    assert "portfolio_validation_benchmark_relative_risks" in metrics_payload["metrics"]
    assert metrics_payload["metrics"]["research_evaluation_profile"] == "default_research"
    assert "rolling_ic_positive_share" in metrics_payload["metrics"]
    assert "rolling_ic_min_mean" in metrics_payload["metrics"]
    assert "rolling_instability_flags" in metrics_payload["metrics"]
    assert "portfolio_validation_summary" in metrics_payload
    assert "portfolio_validation_metrics" in metrics_payload
    assert "portfolio_validation_package" in metrics_payload

    manifest_payload = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["artifact_type"] == "real_case_composite_bundle"
    assert manifest_payload["evaluation_standard"]["profile_name"] == "default_research"
    assert set(required_files).issubset(set(manifest_payload["required_bundle_files"]))
    assert "factor_definition.json" in manifest_payload["required_bundle_files"]
    assert "signal_validation.json" in manifest_payload["required_bundle_files"]
    assert "portfolio_recipe.json" in manifest_payload["required_bundle_files"]
    assert "backtest_result.json" in manifest_payload["required_bundle_files"]
    assert (
        "level2_portfolio_validation/portfolio_validation_summary.json"
        in manifest_payload["required_bundle_files"]
    )

    for artifact_name in (
        "factor_definition.json",
        "signal_validation.json",
        "portfolio_recipe.json",
        "backtest_result.json",
    ):
        artifact_path = output_dir / artifact_name
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict)
        validate_level12_artifact_payload(
            payload,
            artifact_name=artifact_name,
            source=artifact_path,
        )

    portfolio_recipe_payload = json.loads(
        (output_dir / "portfolio_recipe.json").read_text(encoding="utf-8")
    )
    recipe_fallback_fields = set(portfolio_recipe_payload.get("fallback_derived_fields", []))
    for key in (
        "turnover_penalty_settings",
        "transaction_cost_assumptions",
        "position_limits",
    ):
        assert isinstance(portfolio_recipe_payload.get(key), str)
        assert portfolio_recipe_payload.get(key)
        assert key not in recipe_fallback_fields

    backtest_payload = json.loads((output_dir / "backtest_result.json").read_text(encoding="utf-8"))
    summary = backtest_payload["summary"]
    assert isinstance(summary, dict)
    for key in (
        "annualized_return",
        "annualized_volatility",
        "sortino",
        "max_drawdown",
        "calmar",
        "rolling_sharpe",
        "rolling_drawdown",
        "nav_points",
        "monthly_return_table",
        "drawdown_table",
        "subperiod_analysis",
        "regime_analysis",
    ):
        assert key in summary
    backtest_fallback_fields = set(backtest_payload.get("fallback_derived_fields", []))
    for key in (
        "annualized_return",
        "annualized_volatility",
        "sortino",
        "max_drawdown",
        "calmar",
        "rolling_sharpe",
        "rolling_drawdown",
        "nav_points",
        "monthly_return_table",
        "drawdown_table",
        "subperiod_analysis",
        "regime_analysis",
    ):
        assert key not in backtest_fallback_fields

    summary_md = (output_dir / "summary.md").read_text(encoding="utf-8")
    card_md = (output_dir / "experiment_card.md").read_text(encoding="utf-8")
    assert "## Metrics" in summary_md
    assert "Mean IC 95% CI" in summary_md
    assert "Uncertainty Method" in summary_md
    assert "Uncertainty Flags" in summary_md
    assert "Rolling Stability Window" in summary_md
    assert "Factor Verdict" in summary_md
    assert "Research Evaluation Profile" in summary_md
    assert "Campaign Triage" in summary_md
    assert "Level 2 Promotion" in summary_md
    assert "Level 1->Level 2 Transition" in summary_md
    assert "Level 2 Portfolio Validation" in summary_md
    assert "## Setup" in card_md
    assert "## Results" in card_md
    assert "Mean IC 95% CI" in card_md
    assert "Uncertainty Method" in card_md
    assert "Uncertainty Flags" in card_md
    assert "Rolling Stability Window" in card_md
    assert "Factor Verdict" in card_md
    assert "Research Evaluation Profile" in card_md
    assert "Campaign Triage" in card_md
    assert "Level 2 Promotion" in card_md
    assert "Level 1->Level 2 Transition" in card_md
    assert "Level 2 Portfolio Validation" in card_md
    assert "Interpretation" in card_md
