from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

from alpha_lab.artifact_contracts import validate_level12_artifact_payload
from alpha_lab.evaluation import compute_ic, compute_rank_ic
from alpha_lab.exceptions import AlphaLabConfigError
from alpha_lab.labels import forward_return
from alpha_lab.real_cases.single_factor.evaluate import (
    _annotate_coverage_warmup,
    _count_coverage_break_days,
    _coverage_decision_frame,
    _merge_daily_pnl_attribution_metrics,
    _summarise_effective_coverage,
    _with_split_phase,
)
from alpha_lab.real_cases.single_factor.pipeline import run_single_factor_case
from alpha_lab.splits import (
    TimeSeriesSplitContract,
    infer_default_time_series_split_contract,
    rebalance_frequency_to_step,
)
from tests.single_factor_case_helpers import write_demo_single_factor_case


def test_coverage_break_days_ignore_leading_warmup_zero_scores() -> None:
    coverage = pd.DataFrame(
        {
            "date": pd.date_range("2026-01-01", periods=5, freq="D"),
            "eligible_count": [100, 100, 100, 100, 100],
            "valid_score_count": [0, 0, 96, 95, 97],
            "valid_forward_return_count": [100, 100, 100, 100, 100],
            "valid_sample_count": [0, 0, 96, 95, 97],
            "asset_coverage": [0.0, 0.0, 0.96, 0.95, 0.97],
            "coverage": [0.0, 0.0, 0.96, 0.95, 0.97],
        }
    )

    annotated = _annotate_coverage_warmup(coverage)
    decision_frame = _coverage_decision_frame(annotated)

    assert annotated["is_warmup"].tolist() == [True, True, False, False, False]
    assert int(annotated["coverage_eval_included"].sum()) == 3
    assert _count_coverage_break_days(
        annotated,
        threshold=0.90,
        drop_threshold=0.20,
    ) == 2
    assert _count_coverage_break_days(
        decision_frame,
        threshold=0.90,
        drop_threshold=0.20,
    ) == 0
    stats = _summarise_effective_coverage(decision_frame)
    assert stats["min_asset_coverage"] == pytest.approx(0.95)
    assert stats["overall_sample_coverage"] == pytest.approx((96 + 95 + 97) / 300)


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
        "lag_sensitivity.csv",
        "random_baseline_null.csv",
        "daily_pnl_attribution.csv",
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
    metrics = metrics_payload["metrics"]
    assert "mean_rank_ic" in metrics
    assert "mean_mutual_information" in metrics
    assert "mutual_information_ir" in metrics
    assert "mean_long_short_return" in metrics

    daily_pnl = pd.read_csv(output_dir / "daily_pnl_attribution.csv", parse_dates=["date"])
    turnover = pd.read_csv(output_dir / "turnover.csv", parse_dates=["date"])
    assert {"date", "long_leg", "short_leg", "gross", "cost_drag", "net"}.issubset(
        daily_pnl.columns
    )
    assert not daily_pnl.empty
    assert pd.isna(daily_pnl["cost_drag"].iloc[0])
    assert pd.isna(daily_pnl["net"].iloc[0])
    oos_turnover = turnover.loc[turnover["split_phase"] == "OOS"].sort_values(
        "date",
        kind="mergesort",
    )
    assert pd.isna(oos_turnover["turnover"].iloc[0])
    daily_with_turnover = daily_pnl.merge(
        turnover[["date", "turnover"]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    expected_cost_drag = (
        pd.to_numeric(daily_with_turnover["turnover"], errors="coerce")
        * float(metrics["transaction_cost_one_way_rate"])
    )
    actual_cost_drag = pd.to_numeric(daily_with_turnover["cost_drag"], errors="coerce")
    pd.testing.assert_series_equal(actual_cost_drag, expected_cost_drag, check_names=False)
    expected_net = pd.to_numeric(daily_with_turnover["gross"], errors="coerce") - expected_cost_drag
    actual_net = pd.to_numeric(daily_with_turnover["net"], errors="coerce")
    pd.testing.assert_series_equal(actual_net, expected_net, check_names=False)
    assert actual_net.notna().sum() == actual_net.shape[0] - 1
    assert metrics["daily_pnl_net_mean"] == pytest.approx(float(actual_net.dropna().mean()))
    assert metrics["mean_cost_adjusted_long_short_return"] == pytest.approx(
        float(actual_net.dropna().mean())
    )
    assert "ic_t_stat" in metrics_payload["metrics"]
    assert "ic_p_value" in metrics_payload["metrics"]
    assert "dsr_pvalue" in metrics_payload["metrics"]
    assert "split_description" in metrics_payload["metrics"]
    split_contract = metrics_payload["metrics"]["split_contract"]
    metrics = metrics_payload["metrics"]
    assert split_contract["source"] == "single_factor_pipeline"
    assert metrics["oos_start"] == split_contract["oos_start"]
    assert metrics["split_embargo_days"] == split_contract["embargo_days"]
    assert metrics["metric_scope"] == "oos"
    assert metrics["report_metric_scope"] == "full_sample_with_oos_parentheses"
    assert metrics["report_timeseries_scope"] == "full_path_split_by_phase"
    assert metrics["report_split_phase_column"] == "split_phase"
    assert "mean_rank_ic_full" in metrics
    assert "mean_rank_ic_is" in metrics
    assert "mean_rank_ic_oos" in metrics
    assert "mean_rank_ic_oos_decay_ratio" in metrics
    assert metrics["mean_rank_ic_oos"] == pytest.approx(metrics["mean_rank_ic"])
    assert "random_baseline_observed_mean_rank_ic" in metrics
    assert "max_drawdown_full" in metrics
    assert "max_drawdown_is" in metrics
    assert "max_drawdown_oos" in metrics
    assert metrics["split_semantics"] == "factor_time_series_holdout"
    assert "Alpha-Lab" in metrics["split_semantics_label"]
    assert split_contract["n_oos_dates"] >= split_contract["min_oos_dates"]
    assert "coverage_min" in metrics_payload["metrics"]
    assert "coverage_break_days" in metrics_payload["metrics"]
    assert "date_coverage" in metrics_payload["metrics"]
    assert "avg_asset_coverage" in metrics_payload["metrics"]
    assert "min_asset_coverage" in metrics_payload["metrics"]
    assert "overall_sample_coverage" in metrics_payload["metrics"]
    assert "avg_assets" in metrics_payload["metrics"]
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
    random_baseline = pd.read_csv(output_dir / "random_baseline_null.csv")
    assert len(random_baseline) == int(metrics["random_baseline_n_permutations"])
    assert set(random_baseline.columns) == {"permutation", "mean_ic"}
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

    oos_start = pd.Timestamp(split_contract["oos_start"])
    ic_timeseries = pd.read_csv(output_dir / "ic_timeseries.csv")
    assert {"IS", "OOS"}.issubset(set(ic_timeseries["split_phase"]))
    assert pd.to_datetime(ic_timeseries["date"]).min() < oos_start
    assert pd.to_datetime(ic_timeseries["date"]).max() >= oos_start
    group_returns = pd.read_csv(output_dir / "group_returns.csv")
    assert {"IS", "OOS"}.issubset(set(group_returns["split_phase"]))
    turnover = pd.read_csv(output_dir / "turnover.csv")
    assert {"IS", "OOS"}.issubset(set(turnover["split_phase"]))
    coverage = pd.read_csv(output_dir / "coverage.csv")
    assert {
        "eligible_count",
        "valid_score_count",
        "valid_forward_return_count",
        "valid_sample_count",
        "asset_coverage",
        "forward_return_coverage",
        "sample_coverage",
        "coverage",
        "missing_score_count",
        "missing_forward_return_count",
        "invalid_sample_count",
        "split_phase",
    }.issubset(set(coverage.columns))
    assert {"IS", "OOS"}.issubset(set(coverage["split_phase"]))
    assert coverage["coverage"].tolist() == pytest.approx(
        coverage["asset_coverage"].tolist()
    )
    assert metrics["coverage_mean"] == pytest.approx(coverage["asset_coverage"].mean())
    assert metrics["overall_sample_coverage"] == pytest.approx(
        coverage["valid_sample_count"].sum() / coverage["eligible_count"].sum()
    )
    coverage_summary = metrics_payload["coverage_by_date_summary"]
    assert coverage_summary["mean_asset_coverage"] == pytest.approx(
        coverage["asset_coverage"].mean()
    )
    assert coverage_summary["overall_sample_coverage"] == pytest.approx(
        coverage["valid_sample_count"].sum() / coverage["eligible_count"].sum()
    )
    daily_pnl = pd.read_csv(output_dir / "daily_pnl_attribution.csv")
    daily_pnl = daily_pnl.sort_values("date", kind="mergesort").reset_index(drop=True)
    assert pd.isna(daily_pnl["cost_drag"].iloc[0])
    assert pd.isna(daily_pnl["net"].iloc[0])
    assert metrics["daily_pnl_net_mean"] == pytest.approx(daily_pnl["net"].dropna().mean())
    assert metrics["daily_pnl_cost_drag_mean"] == pytest.approx(
        daily_pnl["cost_drag"].dropna().mean()
    )

    tearsheet_payload = json.loads(
        (output_dir / "research_tearsheet.json").read_text(encoding="utf-8")
    )
    assert tearsheet_payload["artifact_type"] == "alpha_lab_research_tearsheet"
    assert tearsheet_payload["meta"]["split_contract"]["oos_start"] == split_contract["oos_start"]
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
    assert manifest_payload["split_contract"]["oos_start"] == split_contract["oos_start"]
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
    assert backtest_payload["split_contract"]["oos_start"] == split_contract["oos_start"]
    assert backtest_payload["oos_start"] == split_contract["oos_start"]
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
    nav_dates = pd.to_datetime([row[0] for row in summary["nav_points"]])
    assert nav_dates.min() < oos_start
    assert nav_dates.max() >= oos_start
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
    assert "| Level 1->2 Transition | Inconclusive transition |" in summary_md
    assert (
        "| Portfolio Validation | skipped_not_promoted (Not evaluated (not promoted)) |"
        in summary_md
    )
    summary_coverage_line = next(
        line for line in summary_md.splitlines() if line.startswith("| Coverage Mean |")
    )
    assert summary_coverage_line == "| Coverage Mean | 0.998387 (OOS: 1.000000) |"
    assert "## 基本信息" in card_md
    assert "## 关键结果" in card_md
    assert "## 解释" in card_md
    assert "## 下一步" in card_md
    assert "## 备注" in card_md
    assert "IC Half-Life" in card_md
    assert "Mean MI" in card_md
    assert "Capacity" in card_md
    assert "Conditional IC" in card_md
    assert (
        "| Level 2 Portfolio Validation | skipped_not_promoted "
        "(Not evaluated (not promoted)) |"
    ) in card_md
    card_coverage_line = next(
        line for line in card_md.splitlines() if line.startswith("| Coverage Mean |")
    )
    assert card_coverage_line == "| Coverage Mean | 0.998387 (OOS: 1.000000) |"


def test_split_phase_packaging_drops_embargo_only_for_report_curves() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    contract = TimeSeriesSplitContract(
        is_start=pd.Timestamp("2024-01-01"),
        is_end=pd.Timestamp("2024-01-01"),
        oos_start=pd.Timestamp("2024-01-04"),
        oos_end=pd.Timestamp("2024-01-04"),
        embargo_days=2,
        min_oos_dates=1,
        min_is_dates=1,
        policy="unit_test",
        source="unit_test",
        n_dates=4,
        n_is_dates=1,
        n_oos_dates=1,
        target_horizon=1,
        rebalance_step=1,
    )

    report_curve = _with_split_phase(frame, contract, drop_embargo=True)
    coverage_curve = _with_split_phase(frame, contract, drop_embargo=False)

    assert report_curve["split_phase"].tolist() == ["IS", "OOS"]
    assert report_curve["date"].tolist() == [
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-01-04"),
    ]
    assert coverage_curve["split_phase"].tolist() == [
        "IS",
        "EMBARGO",
        "EMBARGO",
        "OOS",
    ]


def test_single_factor_oos_metrics_exclude_embargo_reversal_case(tmp_path: Path) -> None:
    factor_name = "embargo_reversal"
    spec_path = write_demo_single_factor_case(tmp_path, factor_name=factor_name, n_days=160)
    spec_payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    prices = pd.read_csv(spec_payload["prices_path"], parse_dates=["date"])
    factors = pd.read_csv(spec_payload["factor_path"], parse_dates=["date"])
    labels = forward_return(prices, horizon=int(spec_payload["target"]["horizon"]))
    split_contract = infer_default_time_series_split_contract(
        prices["date"],
        target_horizon=int(spec_payload["target"]["horizon"]),
        rebalance_step=rebalance_frequency_to_step(spec_payload["rebalance_frequency"]),
        source="single_factor_pipeline",
    )

    labels_for_factor = labels[["date", "asset", "value"]].rename(
        columns={"value": "forward_return"}
    )
    factors = factors.merge(labels_for_factor, on=["date", "asset"], how="left")
    asset_codes = {asset: idx for idx, asset in enumerate(sorted(factors["asset"].unique()))}
    factors["_noise"] = factors["asset"].map(asset_codes).astype(float) * 1e-7
    finite_label = pd.to_numeric(factors["forward_return"], errors="coerce")
    dates = pd.to_datetime(factors["date"])
    embargo_mask = (dates > split_contract.is_end) & (dates < split_contract.oos_start)
    oos_mask = dates >= split_contract.oos_start
    factors.loc[embargo_mask, "value"] = -finite_label.loc[embargo_mask] + factors.loc[
        embargo_mask,
        "_noise",
    ]
    factors.loc[oos_mask, "value"] = finite_label.loc[oos_mask] + factors.loc[
        oos_mask,
        "_noise",
    ]
    factors.loc[finite_label.isna(), "value"] = factors.loc[finite_label.isna(), "_noise"]
    factors[["date", "asset", "factor", "value"]].to_csv(
        spec_payload["factor_path"],
        index=False,
    )

    result = run_single_factor_case(spec_path, vault_export_mode="skip")
    output_dir = result.output_dir
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    ic_timeseries = pd.read_csv(output_dir / "ic_timeseries.csv", parse_dates=["date"])
    coverage = pd.read_csv(output_dir / "coverage.csv", parse_dates=["date"])

    assert "EMBARGO" not in set(ic_timeseries["split_phase"])
    assert "EMBARGO" in set(coverage["split_phase"])
    oos_rank_ic = pd.to_numeric(
        ic_timeseries.loc[ic_timeseries["split_phase"] == "OOS", "rank_ic"],
        errors="coerce",
    ).dropna()
    assert metrics["mean_rank_ic"] == pytest.approx(float(oos_rank_ic.mean()))
    assert metrics["mean_rank_ic"] == pytest.approx(metrics["mean_rank_ic_oos"])
    assert float(metrics["mean_rank_ic"]) > 0.90

    embargo_factor = result.factor_df.loc[
        (pd.to_datetime(result.factor_df["date"]) > pd.Timestamp(metrics["is_end"]))
        & (pd.to_datetime(result.factor_df["date"]) < pd.Timestamp(metrics["oos_start"]))
    ]
    embargo_rank = compute_rank_ic(
        embargo_factor,
        result.evaluation_result.experiment_result.label_df,
    )
    embargo_values = pd.to_numeric(embargo_rank["rank_ic"], errors="coerce").dropna()
    assert float(embargo_values.mean()) < -0.90

    leaky_mean = float(pd.concat([oos_rank_ic, embargo_values], ignore_index=True).mean())
    assert leaky_mean < float(metrics["mean_rank_ic"]) - 0.10


def test_next_open_sensitivity_uses_oos_split_scope(tmp_path: Path) -> None:
    factor_name = "next_open_oos_scope"
    spec_path = write_demo_single_factor_case(tmp_path, factor_name=factor_name, n_days=160)
    spec_payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    horizon = int(spec_payload["target"]["horizon"])
    prices = pd.read_csv(spec_payload["prices_path"], parse_dates=["date"])
    factors = pd.read_csv(spec_payload["factor_path"], parse_dates=["date"])
    asset_codes = {asset: idx for idx, asset in enumerate(sorted(prices["asset"].unique()))}
    prices["_asset_code"] = prices["asset"].map(asset_codes).astype(float)
    prices["open"] = prices["close"] * (1.01 + prices["_asset_code"] * 0.0005)
    prices.drop(columns=["_asset_code"]).to_csv(spec_payload["prices_path"], index=False)

    next_open_labels = forward_return(
        prices.drop(columns=["_asset_code"]),
        horizon=horizon,
        execution_price_mode="next_open",
    )
    split_contract = infer_default_time_series_split_contract(
        prices["date"],
        target_horizon=horizon,
        rebalance_step=rebalance_frequency_to_step(spec_payload["rebalance_frequency"]),
        source="single_factor_pipeline",
    )
    next_open_for_factor = next_open_labels[["date", "asset", "value"]].rename(
        columns={"value": "next_open_forward_return"}
    )
    factors = factors.merge(next_open_for_factor, on=["date", "asset"], how="left")
    factors["_noise"] = factors["asset"].map(asset_codes).astype(float) * 1e-7
    finite_label = pd.to_numeric(factors["next_open_forward_return"], errors="coerce")
    dates = pd.to_datetime(factors["date"])
    pre_oos_mask = dates < split_contract.oos_start
    oos_mask = dates >= split_contract.oos_start
    factors.loc[pre_oos_mask, "value"] = -finite_label.loc[pre_oos_mask] + factors.loc[
        pre_oos_mask,
        "_noise",
    ]
    factors.loc[oos_mask, "value"] = finite_label.loc[oos_mask] + factors.loc[
        oos_mask,
        "_noise",
    ]
    factors.loc[finite_label.isna(), "value"] = factors.loc[finite_label.isna(), "_noise"]
    factors[["date", "asset", "factor", "value"]].to_csv(
        spec_payload["factor_path"],
        index=False,
    )

    result = run_single_factor_case(spec_path, vault_export_mode="skip")
    metrics = result.evaluation_result.metrics
    oos_start = pd.Timestamp(metrics["oos_start"])
    result_dates = pd.to_datetime(result.factor_df["date"])
    oos_factor = result.factor_df.loc[result_dates >= oos_start]
    oos_ic = compute_ic(oos_factor, next_open_labels)
    oos_values = pd.to_numeric(oos_ic["ic"], errors="coerce").dropna()

    assert metrics["next_open_execution_available"] is True
    assert metrics["next_open_mean_ic"] == pytest.approx(float(oos_values.mean()))
    assert float(metrics["next_open_mean_ic"]) > 0.90

    full_ic = compute_ic(result.factor_df, next_open_labels)
    full_values = pd.to_numeric(full_ic["ic"], errors="coerce").dropna()
    assert float(full_values.mean()) < float(metrics["next_open_mean_ic"]) - 0.50


def test_constant_small_cross_section_artifacts_do_not_invent_long_short(
    tmp_path: Path,
) -> None:
    factor_name = "constant_small_cross_section"
    spec_path = write_demo_single_factor_case(tmp_path, factor_name=factor_name, n_days=160)
    spec_payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec_payload["preprocess"] = {
        "winsorize": False,
        "winsorize_lower": 0.01,
        "winsorize_upper": 0.99,
        "standardization": "none",
        "min_group_size": 1,
        "min_coverage": None,
    }
    spec_path.write_text(yaml.safe_dump(spec_payload, sort_keys=False), encoding="utf-8")

    factors = pd.read_csv(spec_payload["factor_path"], parse_dates=["date"])
    dates = pd.Index(pd.to_datetime(factors["date"]).drop_duplicates()).sort_values()
    asset_order = {asset: idx for idx, asset in enumerate(sorted(factors["asset"].unique()))}
    date_order = {date: idx for idx, date in enumerate(dates)}
    keep_counts = [1, 2, 4]
    factors["_date_idx"] = pd.to_datetime(factors["date"]).map(date_order).astype(int)
    factors["_asset_idx"] = factors["asset"].map(asset_order).astype(int)
    factors["value"] = 1.0
    keep_mask = factors["_asset_idx"] < factors["_date_idx"].map(
        lambda idx: keep_counts[int(idx) % len(keep_counts)]
    )
    factors.loc[~keep_mask, "value"] = float("nan")
    factors[["date", "asset", "factor", "value"]].to_csv(
        spec_payload["factor_path"],
        index=False,
    )

    result = run_single_factor_case(spec_path, vault_export_mode="skip")
    output_dir = result.output_dir
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))["metrics"]
    group_returns = pd.read_csv(output_dir / "group_returns.csv")
    daily_pnl = pd.read_csv(output_dir / "daily_pnl_attribution.csv")
    backtest = json.loads((output_dir / "backtest_result.json").read_text(encoding="utf-8"))
    tearsheet = json.loads((output_dir / "research_tearsheet.json").read_text(encoding="utf-8"))

    assert set(group_returns["group"].dropna().astype(int)) == {1}
    assert daily_pnl.empty
    assert metrics["daily_pnl_n_dates"] == 0
    assert metrics["mean_long_short_return"] is None
    assert metrics["long_short_ir"] is None
    assert backtest["summary"].get("nav_points") in (None, [])

    chart_titles = {
        chart["title"]
        for section in tearsheet["sections"].values()
        for chart in section.get("charts", [])
    }
    assert "Cumulative Long-Short NAV" not in chart_titles


def test_daily_pnl_attribution_uses_per_date_occupied_top_bucket() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    result = SimpleNamespace(
        quantile_returns_df=pd.DataFrame(
            {
                "date": [dates[0], dates[0], dates[1], dates[1]],
                "factor": ["f", "f", "f", "f"],
                "quantile": [1, 2, 1, 5],
                "mean_return": [0.01, 0.03, 0.02, 0.07],
            }
        ),
        long_short_turnover_df=pd.DataFrame(
            {
                "date": dates,
                "factor": ["f", "f"],
                "long_short_turnover": [float("nan"), 0.5],
            }
        ),
    )
    metrics: dict[str, object] = {}

    attribution = _merge_daily_pnl_attribution_metrics(
        metrics,
        result=result,
        cost_rate=0.001,
    )

    assert attribution["date"].tolist() == list(dates)
    assert attribution["gross"].tolist() == pytest.approx([0.02, 0.05])
    assert pd.isna(attribution["cost_drag"].iloc[0])
    assert pd.isna(attribution["net"].iloc[0])
    assert float(attribution["net"].iloc[1]) == pytest.approx(0.0495)
    assert metrics["daily_pnl_n_dates"] == 2
    assert metrics["daily_pnl_net_mean"] == pytest.approx(0.0495)


def test_single_factor_case_rejects_too_short_strict_split_before_artifacts(
    tmp_path: Path,
) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="short_history", n_days=120)

    with pytest.raises(AlphaLabConfigError, match="strict IS/OOS split"):
        run_single_factor_case(spec_path)

    output_dir = tmp_path / "outputs" / "demo_short_history_single_factor"
    assert not (output_dir / "metrics.json").exists()


@pytest.mark.parametrize(
    "profile",
    ["exploratory_screening", "default_research", "stricter_research"],
)
def test_single_factor_dual_scope_report_path_respects_profile(
    tmp_path: Path,
    profile: str,
) -> None:
    spec_path = write_demo_single_factor_case(
        tmp_path,
        factor_name=f"dual_scope_{profile}",
    )

    result = run_single_factor_case(spec_path, evaluation_profile=profile)
    output_dir = result.output_dir
    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics = metrics_payload["metrics"]
    split_contract = metrics["split_contract"]

    assert metrics["research_evaluation_profile"] == profile
    # Headline metrics are OOS-gated under every profile (a split contract exists).
    assert metrics["metric_scope"] == "oos"
    assert metrics["split_semantics"] == "factor_time_series_holdout"
    assert "mean_rank_ic" in metrics
    assert "eval_coverage_ratio_mean" in metrics

    ic_timeseries = pd.read_csv(output_dir / "ic_timeseries.csv")

    if profile == "exploratory_screening":
        # Fast screening suppresses the full-sample + IS report paths: the headline
        # stays OOS, the full/IS scoped companions are dropped, and the IC timeseries
        # covers OOS only (no extra full-sample backtest).
        assert metrics["report_metric_scope"] == "suppressed_by_profile"
        assert metrics["report_timeseries_scope"] == "oos"
        for key in (
            "mean_rank_ic_full",
            "mean_rank_ic_is",
            "mean_rank_ic_oos",
            "eval_coverage_ratio_mean_full",
            "eval_coverage_ratio_mean_is",
            "eval_coverage_ratio_mean_oos",
        ):
            assert key not in metrics
        assert set(ic_timeseries["split_phase"].dropna().unique()) == {"OOS"}
        assert metrics["random_baseline_n_permutations"] > 0
    else:
        # default_research / stricter_research keep the full dual-scope contract.
        assert metrics["report_metric_scope"] == "full_sample_with_oos_parentheses"
        assert "mean_rank_ic_full" in metrics
        assert "mean_rank_ic_is" in metrics
        assert "mean_rank_ic_oos" in metrics
        assert "mean_rank_ic_oos_decay_ratio" in metrics
        assert metrics["mean_rank_ic_oos"] == pytest.approx(metrics["mean_rank_ic"])
        assert "eval_coverage_ratio_mean_full" in metrics
        assert "eval_coverage_ratio_mean_is" in metrics
        assert "eval_coverage_ratio_mean_oos" in metrics
        assert "eval_coverage_ratio_min_full" in metrics
        assert "eval_coverage_ratio_min_is" in metrics
        assert "eval_coverage_ratio_min_oos" in metrics
        assert metrics["report_timeseries_scope"] == "full_path_split_by_phase"
        assert {"IS", "OOS"}.issubset(set(ic_timeseries["split_phase"]))
        assert pd.to_datetime(ic_timeseries["date"]).min() < pd.Timestamp(
            split_contract["oos_start"]
        )
        if profile == "stricter_research":
            assert metrics["uncertainty_method"] == "block_bootstrap"
            assert metrics["strict_research_evidence"] == "enabled"
            assert "strict_bootstrap_rank_ic_ir_ci_lower" in metrics
            assert "strict_subsample_rank_ic_first_half_mean" in metrics
            assert "strict_post_split_rank_ic_gap_5_mean" in metrics


def test_membership_artifacts_tiered_by_profile(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")

    default_run = run_single_factor_case(
        spec_path,
        evaluation_profile="default_research",
        output_root_dir=tmp_path / "default_out",
    )
    exploratory_run = run_single_factor_case(
        spec_path,
        evaluation_profile="exploratory_screening",
        output_root_dir=tmp_path / "explore_out",
    )

    def _membership(run: object) -> pd.DataFrame:
        return pd.read_csv(run.output_dir / "quantile_membership.csv")  # type: ignore[attr-defined]

    def _tiers(run: object) -> dict:
        manifest = json.loads(
            (run.output_dir / "run_manifest.json").read_text(encoding="utf-8")  # type: ignore[attr-defined]
        )
        return manifest["artifact_tiers"]

    full_mem = _membership(default_run)
    sampled_mem = _membership(exploratory_run)

    # default_research keeps the full cross-section (all quantiles), tagged full.
    assert _tiers(default_run)["quantile_membership"] == "full"
    assert set(full_mem["quantile"].unique()) == {1, 2, 3, 4, 5}

    # exploratory_screening keeps only the tradeable extremes, fewer rows, tagged.
    assert _tiers(exploratory_run)["quantile_membership"] == "sampled_extreme_quantiles"
    assert set(sampled_mem["quantile"].unique()) == {1, 5}
    assert len(sampled_mem) < len(full_mem)


def test_exploratory_screening_runs_single_core_backtest(tmp_path: Path) -> None:
    """exploratory_screening must not run the full-sample / IS report backtests."""
    import alpha_lab.real_cases.single_factor.evaluate.core as sf_core

    spec_path = write_demo_single_factor_case(tmp_path, factor_name="single_backtest")

    original = sf_core.run_factor_experiment
    calls = {"n": 0}

    def _counting(*args: object, **kwargs: object):
        calls["n"] += 1
        return original(*args, **kwargs)

    with patch.object(sf_core, "run_factor_experiment", _counting):
        exploratory_result = run_single_factor_case(
            spec_path, evaluation_profile="exploratory_screening"
        )
    exploratory_calls = calls["n"]

    # The suppression is registered for downstream contract/audit consumers.
    skipped = exploratory_result.evaluation_result.metrics["single_factor_skipped_diagnostics"]
    assert "dual_scope_report_path" in skipped
    assert "is_report_path" in skipped

    calls["n"] = 0
    with patch.object(sf_core, "run_factor_experiment", _counting):
        run_single_factor_case(spec_path, evaluation_profile="default_research")
    default_calls = calls["n"]

    # Core backtest only under exploratory; default keeps core + full-sample + IS.
    assert exploratory_calls == 1
    assert default_calls == 3
