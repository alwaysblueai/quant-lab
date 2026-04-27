from __future__ import annotations

import json
from pathlib import Path

import yaml

from alpha_lab.artifact_contracts import validate_level12_artifact_payload
from alpha_lab.real_cases.single_factor.pipeline import run_single_factor_case
from tests.single_factor_case_helpers import write_demo_single_factor_case


def test_single_factor_artifacts_have_required_files_and_fields(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="roe_ttm")
    result = run_single_factor_case(spec_path)

    output_dir = result.output_dir
    required_files = {
        "run_manifest.json",
        "metrics.json",
        "factor_definition.json",
        "signal_validation.json",
        "portfolio_recipe.json",
        "backtest_result.json",
        "purged_kfold_summary.json",
        "purged_kfold_folds.csv",
        "ic_timeseries.csv",
        "ic_decay.csv",
        "factor_autocorrelation.csv",
        "capacity_estimation.csv",
        "conditional_ic_by_magnitude.csv",
        "conditional_ic_by_cross_section_size.csv",
        "rolling_stability.csv",
        "group_returns.csv",
        "turnover.csv",
        "coverage.csv",
        "factor_definition.yaml",
        "summary.md",
        "experiment_card.md",
        "research_tearsheet.json",
        "research_tearsheet.pdf",
        "integrity_report.json",
        "integrity_report.md",
    }
    present_files = {p.name for p in output_dir.iterdir() if p.is_file()}
    assert required_files.issubset(present_files)
    assert (
        output_dir / "level2_portfolio_validation" / "portfolio_validation_summary.json"
    ).exists()
    assert (
        output_dir / "level2_portfolio_validation" / "portfolio_validation_metrics.json"
    ).exists()
    assert (
        output_dir / "level2_portfolio_validation" / "portfolio_validation_package.json"
    ).exists()

    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "metrics" in metrics_payload
    assert "mean_rank_ic" in metrics_payload["metrics"]
    assert "mean_mutual_information" in metrics_payload["metrics"]
    assert "mutual_information_ir" in metrics_payload["metrics"]
    assert "mean_long_short_return" in metrics_payload["metrics"]
    assert "ic_t_stat" in metrics_payload["metrics"]
    assert "ic_p_value" in metrics_payload["metrics"]
    assert "dsr_pvalue" in metrics_payload["metrics"]
    assert "split_description" in metrics_payload["metrics"]
    assert "coverage_min" in metrics_payload["metrics"]
    assert "data_quality_status" in metrics_payload["metrics"]
    assert "data_quality_suspended_rows" in metrics_payload["metrics"]
    assert "data_quality_stale_rows" in metrics_payload["metrics"]
    assert "data_quality_suspected_split_rows" in metrics_payload["metrics"]
    assert "data_quality_integrity_warn_count" in metrics_payload["metrics"]
    assert "data_quality_integrity_fail_count" in metrics_payload["metrics"]
    assert "data_quality_hard_fail_count" in metrics_payload["metrics"]
    assert "uncertainty_flags" in metrics_payload["metrics"]
    assert "uncertainty_method" in metrics_payload["metrics"]
    assert "uncertainty_confidence_level" in metrics_payload["metrics"]
    assert "factor_verdict" in metrics_payload["metrics"]
    assert "factor_verdict_reasons" in metrics_payload["metrics"]
    assert "campaign_triage" in metrics_payload["metrics"]
    assert "campaign_triage_reasons" in metrics_payload["metrics"]
    assert "promotion_decision" in metrics_payload["metrics"]
    assert "promotion_reasons" in metrics_payload["metrics"]
    assert "promotion_blockers" in metrics_payload["metrics"]
    assert "level12_transition_label" in metrics_payload["metrics"]
    assert "level12_transition_interpretation" in metrics_payload["metrics"]
    assert "level12_transition_reasons" in metrics_payload["metrics"]
    assert "portfolio_validation_status" in metrics_payload["metrics"]
    assert "portfolio_validation_recommendation" in metrics_payload["metrics"]
    assert "portfolio_validation_major_risks" in metrics_payload["metrics"]
    assert "ic_half_life_horizon" in metrics_payload["metrics"]
    assert "ic_decay_retention_5_over_1" in metrics_payload["metrics"]
    assert "ic_decay_rebalance_ratio" in metrics_payload["metrics"]
    assert "capacity_status" in metrics_payload["metrics"]
    assert "estimated_capacity_upper_bound" in metrics_payload["metrics"]
    assert "conditional_ic_extreme_minus_base_ic" in metrics_payload["metrics"]
    assert metrics_payload["metrics"]["research_evaluation_profile"] == "default_research"
    assert "rolling_ic_positive_share" in metrics_payload["metrics"]
    assert "rolling_ic_min_mean" in metrics_payload["metrics"]
    assert "rolling_instability_flags" in metrics_payload["metrics"]
    assert "portfolio_validation_summary" not in metrics_payload
    assert "portfolio_validation_metrics" not in metrics_payload
    assert "portfolio_validation_package" not in metrics_payload

    tearsheet_payload = json.loads(
        (output_dir / "research_tearsheet.json").read_text(encoding="utf-8")
    )
    assert tearsheet_payload["artifact_type"] == "alpha_lab_research_tearsheet"
    assert "verdict_layer" in tearsheet_payload
    assert "sections" in tearsheet_payload
    assert "appendix" in tearsheet_payload
    assert {"setup", "signal", "stability", "conversion_risk"}.issubset(
        set(tearsheet_payload["sections"])
    )
    assert (output_dir / "research_tearsheet.pdf").stat().st_size > 0

    purged_summary = json.loads(
        (output_dir / "purged_kfold_summary.json").read_text(encoding="utf-8")
    )
    assert purged_summary["artifact_type"] == "alpha_lab_purged_kfold_summary"
    assert purged_summary["status"] in {"ok", "not_available"}
    assert "n_folds" in purged_summary
    assert "mean_ic" in purged_summary

    manifest_payload = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["artifact_type"] == "real_case_single_factor_bundle"
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

    factor_definition_yaml = yaml.safe_load(
        (output_dir / "factor_definition.yaml").read_text(encoding="utf-8")
    )
    assert isinstance(factor_definition_yaml, dict)
    assert factor_definition_yaml["factor_name"] == "roe_ttm"
    assert factor_definition_yaml["n_quantiles"] == 5
    assert factor_definition_yaml["capacity"]["enabled"] is True
    assert factor_definition_yaml["capacity"]["participation_rate"] == 0.05
    assert factor_definition_yaml["capacity"]["adv_lookback"] == 20
    assert "preprocess" not in factor_definition_yaml
    assert "output" not in factor_definition_yaml
    assert factor_definition_yaml["transaction_cost"]["one_way_rate"] == 0.001

    backtest_payload = json.loads((output_dir / "backtest_result.json").read_text(encoding="utf-8"))
    summary = backtest_payload["summary"]
    assert isinstance(summary, dict)
    for key in (
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "win_rate",
        "turnover",
        "pre_cost_return",
        "post_cost_return",
        "rolling_sharpe",
        "rolling_drawdown",
        "nav_points",
        "monthly_return_table",
        "drawdown_table",
        "subperiod_analysis",
        "regime_analysis",
    ):
        assert key in summary
    assert summary["rolling_sharpe"] is None
    assert summary["rolling_drawdown"] is None
    assert isinstance(summary["nav_points"], list)
    assert len(summary["nav_points"]) >= 2
    assert summary["nav_points"][0][0]
    assert isinstance(summary["nav_points"][0][1], float)
    assert summary["monthly_return_table"] == []
    assert summary["drawdown_table"] == []
    assert summary["subperiod_analysis"] is None
    assert summary["regime_analysis"] is None
    backtest_fallback_fields = set(backtest_payload.get("fallback_derived_fields", []))
    for key in (
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "win_rate",
        "turnover",
        "pre_cost_return",
        "post_cost_return",
    ):
        assert key not in backtest_fallback_fields
    assert "nav_points" not in backtest_fallback_fields

    summary_md = (output_dir / "summary.md").read_text(encoding="utf-8")
    card_md = (output_dir / "experiment_card.md").read_text(encoding="utf-8")
    assert "## 基本信息" in summary_md
    assert "## 初筛结论" in summary_md
    assert "主要阻断项" in summary_md
    assert "## 产物路径" in summary_md
    assert "IC Half-Life" in summary_md
    assert "Mean MI" in summary_md
    assert "Capacity" in summary_md
    assert "Conditional IC" in summary_md
    assert "## 基本信息" in card_md
    assert "## 关键结果" in card_md
    assert "## 解释" in card_md
    assert "## 下一步" in card_md
    assert "## 备注" in card_md
    assert "IC Half-Life" in card_md
    assert "Mean MI" in card_md
    assert "Capacity" in card_md
    assert "Conditional IC" in card_md
