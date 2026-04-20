from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from alpha_lab.real_cases.single_factor.pipeline import run_single_factor_case
from alpha_lab.reporting.campaign_triage import build_campaign_triage
from alpha_lab.reporting.factor_verdict import build_factor_verdict
from alpha_lab.reporting.level2_promotion import build_level2_promotion
from tests.single_factor_case_helpers import write_demo_single_factor_case


def test_single_factor_pipeline_smoke_runs_end_to_end(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(
        tmp_path,
        factor_name="bp",
        enable_neutralization=True,
    )

    result = run_single_factor_case(spec_path)

    assert result.output_dir.exists()
    assert result.evaluation_result.experiment_result.summary.n_dates > 0
    assert result.evaluation_result.metrics["factor_name"] == "bp"
    assert result.evaluation_result.metrics["target_horizon"] == 5
    assert "neutralization_comparison" in result.evaluation_result.metrics
    assert "neutralization_comparison_flags" in result.evaluation_result.metrics
    assert "neutralization_mean_ic_delta" in result.evaluation_result.metrics
    assert "promotion_decision" in result.evaluation_result.metrics
    assert "promotion_reasons" in result.evaluation_result.metrics
    assert "promotion_blockers" in result.evaluation_result.metrics
    assert not result.evaluation_result.capacity_estimation.empty
    assert not result.evaluation_result.conditional_ic_by_magnitude.empty
    assert not result.evaluation_result.conditional_ic_by_cross_section_size.empty

    required_keys = {
        "run_manifest",
        "metrics",
        "factor_definition_json",
        "signal_validation_json",
        "portfolio_recipe_json",
        "backtest_result_json",
        "ic_timeseries",
        "ic_decay",
        "factor_autocorrelation",
        "capacity_estimation",
        "conditional_ic_by_magnitude",
        "conditional_ic_by_cross_section_size",
        "rolling_stability",
        "group_returns",
        "turnover",
        "coverage",
        "factor_definition",
        "summary",
        "experiment_card",
        "integrity_report_json",
        "integrity_report_markdown",
    }
    assert required_keys.issubset(set(result.artifact_paths.keys()))
    for path in result.artifact_paths.values():
        assert path.exists()


def test_single_factor_pipeline_uses_factor_input_recipe_when_factor_path_is_placeholder(
    tmp_path: Path,
) -> None:
    spec_path = write_demo_single_factor_case(
        tmp_path,
        factor_name="asym_vol_reversal_20d",
    )
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    payload["factor_path"] = str(tmp_path / "missing_placeholder_factor.csv")
    payload["preprocess"] = {
        "winsorize": False,
        "winsorize_lower": 0.01,
        "winsorize_upper": 0.99,
        "standardization": "none",
        "min_group_size": 3,
    }
    payload["factor_input"] = {
        "mode": "recipe",
        "disable_pipeline_preprocess": True,
        "recipe": {
            "base": {
                "method": "momentum",
                "window": 5,
                "skip_recent": 1,
            },
            "preprocess": {
                "standardization": {
                    "method": "zscore",
                    "min_group_size": 3,
                }
            },
        },
    }
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = run_single_factor_case(spec_path)

    assert result.output_dir.exists()
    assert result.factor_df["factor"].eq("asym_vol_reversal_20d").all()


def test_single_factor_pipeline_recipe_accepts_parquet_prices_and_universe(
    tmp_path: Path,
) -> None:
    spec_path = write_demo_single_factor_case(
        tmp_path,
        factor_name="parquet_recipe_factor",
    )
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    prices_csv = Path(payload["prices_path"])
    universe_csv = Path(payload["universe"]["path"])
    prices_parquet = prices_csv.with_suffix(".parquet")
    universe_parquet = universe_csv.with_suffix(".parquet")

    pd.read_csv(prices_csv).to_parquet(prices_parquet, index=False)
    pd.read_csv(universe_csv).to_parquet(universe_parquet, index=False)

    payload["prices_path"] = str(prices_parquet)
    payload["universe"]["path"] = str(universe_parquet)
    payload["factor_path"] = str(tmp_path / "missing_placeholder_factor.parquet")
    payload["factor_input"] = {
        "mode": "recipe",
        "disable_pipeline_preprocess": True,
        "recipe": {
            "base": {
                "method": "momentum",
                "window": 5,
                "skip_recent": 1,
            }
        },
    }
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = run_single_factor_case(spec_path)
    assert result.output_dir.exists()
    assert result.factor_df["factor"].eq("parquet_recipe_factor").all()


def test_single_factor_pipeline_accepts_parquet_factor_and_exposures(
    tmp_path: Path,
) -> None:
    spec_path = write_demo_single_factor_case(
        tmp_path,
        factor_name="bp",
        enable_neutralization=True,
    )
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    prices_csv = Path(payload["prices_path"])
    factor_csv = Path(payload["factor_path"])
    universe_csv = Path(payload["universe"]["path"])
    exposures_csv = Path(payload["neutralization"]["exposures_path"])

    prices_parquet = prices_csv.with_suffix(".parquet")
    factor_parquet = factor_csv.with_suffix(".parquet")
    universe_parquet = universe_csv.with_suffix(".parquet")
    exposures_parquet = exposures_csv.with_suffix(".parquet")

    pd.read_csv(prices_csv).to_parquet(prices_parquet, index=False)
    pd.read_csv(factor_csv).to_parquet(factor_parquet, index=False)
    pd.read_csv(universe_csv).to_parquet(universe_parquet, index=False)
    pd.read_csv(exposures_csv).to_parquet(exposures_parquet, index=False)

    payload["prices_path"] = str(prices_parquet)
    payload["factor_path"] = str(factor_parquet)
    payload["universe"]["path"] = str(universe_parquet)
    payload["neutralization"]["exposures_path"] = str(exposures_parquet)
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = run_single_factor_case(spec_path)

    assert result.output_dir.exists()
    assert result.factor_df["factor"].eq("bp").all()
    assert "neutralization_comparison" in result.evaluation_result.metrics


def test_single_factor_metrics_cover_decision_contract_keys_without_neutralization(
    tmp_path: Path,
) -> None:
    spec_path = write_demo_single_factor_case(
        tmp_path,
        factor_name="bp",
        enable_neutralization=False,
    )
    result = run_single_factor_case(spec_path)
    metrics = result.evaluation_result.metrics

    required_keys = {
        "mean_ic",
        "mean_rank_ic",
        "mean_mutual_information",
        "mutual_information_ir",
        "mutual_information_positive_rate",
        "mutual_information_valid_ratio",
        "ic_ir",
        "ic_positive_rate",
        "rank_ic_positive_rate",
        "ic_valid_ratio",
        "rank_ic_valid_ratio",
        "mean_long_short_return",
        "long_short_ir",
        "mean_long_short_turnover",
        "long_short_return_per_turnover",
        "rebalance_step_dates",
        "ic_half_life_horizon",
        "ic_half_life_status",
        "ic_half_life_not_reached",
        "ic_decay_rebalance_ratio",
        "ic_decay_mismatch_flag",
        "subperiod_ic_positive_share",
        "subperiod_long_short_positive_share",
        "eval_coverage_ratio_mean",
        "eval_coverage_ratio_min",
        "rolling_window_size",
        "rolling_ic_positive_share",
        "rolling_rank_ic_positive_share",
        "rolling_long_short_positive_share",
        "rolling_ic_min_mean",
        "rolling_rank_ic_min_mean",
        "rolling_long_short_min_mean",
        "mean_ic_ci_lower",
        "mean_ic_ci_upper",
        "mean_rank_ic_ci_lower",
        "mean_rank_ic_ci_upper",
        "mean_long_short_return_ci_lower",
        "mean_long_short_return_ci_upper",
        "uncertainty_flags",
        "uncertainty_method",
        "uncertainty_confidence_level",
        "uncertainty_bootstrap_resamples",
        "uncertainty_bootstrap_block_length",
        "rolling_instability_flags",
        "instability_flags",
        "neutralization_mean_corr_reduction",
        "neutralization_comparison",
        "neutralization_comparison_flags",
        "neutralization_comparison_reasons",
        "neutralization_raw_mean_ic",
        "neutralization_raw_mean_rank_ic",
        "neutralization_raw_mean_long_short_return",
        "neutralization_raw_ic_ir",
        "neutralization_mean_ic_delta",
        "neutralization_mean_rank_ic_delta",
        "neutralization_mean_long_short_return_delta",
        "neutralization_ic_ir_delta",
        "neutralization_valid_ratio_min_delta",
        "neutralization_eval_coverage_ratio_mean_delta",
        "neutralization_uncertainty_overlap_zero_count_delta",
        "neutralization_rolling_positive_share_min_delta",
        "neutralization_rolling_worst_mean_min_delta",
        "capacity_enabled",
        "capacity_status",
        "capacity_notes",
        "capacity_market_cap_column",
        "capacity_participation_rate",
        "capacity_adv_lookback",
        "equal_weight_mean_long_short_return",
        "market_cap_weighted_mean_long_short_return",
        "market_cap_vs_equal_weight_return_delta",
        "mean_traded_adv",
        "estimated_capacity_upper_bound",
        "conditional_ic_q1_mean_ic",
        "conditional_ic_q5_mean_ic",
        "conditional_ic_extreme_minus_base_ic",
        "conditional_ic_small_cross_section_mean_ic",
        "conditional_ic_large_cross_section_mean_ic",
        "factor_verdict",
        "factor_verdict_reasons",
        "campaign_triage",
        "campaign_triage_reasons",
        "promotion_decision",
        "promotion_reasons",
        "promotion_blockers",
    }
    missing = required_keys - set(metrics)
    assert not missing, f"missing decision-contract keys: {sorted(missing)}"

    verdict = build_factor_verdict(metrics)
    assert verdict.label == str(metrics["factor_verdict"])
    triage = build_campaign_triage(metrics)
    assert triage.label == str(metrics["campaign_triage"])
    promotion = build_level2_promotion(metrics)
    assert promotion.label == str(metrics["promotion_decision"])


def test_single_factor_pipeline_exploratory_profile_skips_heavy_diagnostics(
    tmp_path: Path,
) -> None:
    spec_path = write_demo_single_factor_case(
        tmp_path,
        factor_name="bp",
        enable_neutralization=True,
    )
    progress_events: list[tuple[str, int]] = []

    result = run_single_factor_case(
        spec_path,
        evaluation_profile="exploratory_screening",
        progress_callback=lambda message, percent: progress_events.append((message, percent)),
    )

    metrics = result.evaluation_result.metrics
    skipped = set(metrics["single_factor_skipped_diagnostics"])
    assert metrics["single_factor_diagnostics_mode"] == "streamlined"
    assert {"parameter_sensitivity", "baseline_factor_comparison", "lag_sensitivity"} <= skipped
    assert "ic_decay" not in skipped
    assert not result.evaluation_result.ic_decay.empty
    assert result.evaluation_result.factor_autocorrelation.empty
    assert result.evaluation_result.conditional_ic_by_magnitude.empty
    assert result.evaluation_result.conditional_ic_by_cross_section_size.empty
    assert result.evaluation_result.random_baseline_null.empty
    assert any(message == "运行核心回测" for message, _ in progress_events)
    assert any(message == "汇总结论与分层判定" for message, _ in progress_events)
