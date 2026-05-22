from __future__ import annotations

import datetime
import json
import logging
import math
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict

import pandas as pd

from alpha_lab.artifact_contracts import validate_level12_artifact_payload
from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.key_metrics_contracts import project_level12_transition_summary
from alpha_lab.real_cases.artifact_enrichment import (
    build_backtest_summary_payload,
    build_group_nav_table,
    build_portfolio_recipe_controls,
)
from alpha_lab.reporting.level2_portfolio_validation import (
    build_level2_portfolio_validation_bundle,
    export_level2_portfolio_validation_bundle,
)
from alpha_lab.reporting.purged_kfold_diagnostics import build_purged_kfold_diagnostics
from alpha_lab.reporting.research_tearsheet import (
    build_research_tearsheet_payload,
    export_research_tearsheet_pdf,
)
from alpha_lab.research_evaluation_config import (
    ResearchEvaluationConfig,
    research_evaluation_audit_snapshot,
)
from alpha_lab.research_integrity.contracts import IntegrityReport
from alpha_lab.research_integrity.reporting import (
    build_integrity_report,
    write_integrity_report_json,
    write_integrity_report_markdown,
)
from alpha_lab.vault_export import export_to_vault, resolve_vault_root

from .evaluate import SingleFactorEvaluationResult
from .spec import SingleFactorCaseSpec
from .templates import render_experiment_card_markdown, render_summary_markdown

logger = logging.getLogger(__name__)

REQUIRED_BUNDLE_FILES: tuple[str, ...] = (
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
    "group_nav.csv",
    "quantile_membership.csv",
    "quantile_equal_weights.csv",
    "portfolio_weights.csv",
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
    "level2_portfolio_validation/portfolio_validation_summary.json",
    "level2_portfolio_validation/portfolio_validation_metrics.json",
    "level2_portfolio_validation/portfolio_validation_package.json",
    "level2_portfolio_validation/portfolio_validation_package.md",
)

_FAST_SCREEN_PROFILE_NAMES = frozenset({"exploratory_screening", "quick_screening"})

_FAST_SCREEN_CORE_METRIC_KEYS: tuple[str, ...] = (
    "mean_rank_ic",
    "mean_rank_ic_full",
    "mean_rank_ic_is",
    "mean_rank_ic_oos",
    "mean_rank_ic_oos_decay_ratio",
    "mean_ic",
    "mean_ic_full",
    "mean_ic_is",
    "mean_ic_oos",
    "mean_ic_oos_decay_ratio",
    "rank_ic_ir",
    "rank_ic_ir_full",
    "rank_ic_ir_is",
    "rank_ic_ir_oos",
    "rank_ic_ir_oos_decay_ratio",
    "ic_ir",
    "ic_ir_full",
    "ic_ir_is",
    "ic_ir_oos",
    "ic_ir_oos_decay_ratio",
    "ic_positive_rate",
    "ic_positive_rate_full",
    "ic_positive_rate_is",
    "ic_positive_rate_oos",
    "group_monotonicity_summary",
    "ic_decay_half_life_summary",
    "ic_decay_retention_5_over_1",
    "mean_long_short_return",
    "mean_long_short_return_full",
    "mean_long_short_return_is",
    "mean_long_short_return_oos",
    "mean_long_short_return_oos_decay_ratio",
    "long_short_ir",
    "long_short_ir_full",
    "long_short_ir_is",
    "long_short_ir_oos",
    "long_short_ir_oos_decay_ratio",
    "mean_long_short_turnover",
    "mean_long_short_turnover_full",
    "mean_long_short_turnover_is",
    "mean_long_short_turnover_oos",
    "coverage_summary",
    "coverage_break_days",
    "eval_coverage_ratio_mean",
    "eval_coverage_ratio_mean_full",
    "eval_coverage_ratio_mean_is",
    "eval_coverage_ratio_mean_oos",
    "eval_coverage_ratio_min",
    "eval_coverage_ratio_min_full",
    "eval_coverage_ratio_min_is",
    "eval_coverage_ratio_min_oos",
    "cost_aware_long_short_ir",
    "cost_aware_long_short_ir_full",
    "cost_aware_long_short_ir_is",
    "cost_aware_long_short_ir_oos",
    "cost_aware_long_short_ir_oos_decay_ratio",
    "ic_t_stat",
    "max_drawdown",
    "max_drawdown_full",
    "max_drawdown_is",
    "max_drawdown_oos",
    "max_drawdown_oos_decay_ratio",
    "random_baseline_n_permutations",
    "random_baseline_mean_ic_mean",
    "random_baseline_mean_ic_std",
    "random_baseline_p_value",
    "random_baseline_observed_mean_rank_ic",
    "random_baseline_observed_z_score",
    "metric_scope",
    "primary_metric_scope",
    "report_metric_scope",
    "report_timeseries_scope",
    "report_split_phase_column",
    "split_semantics",
    "split_semantics_label",
)

_FAST_SCREEN_REQUIRED_CONTRACT_KEYS: tuple[str, ...] = (
    "research_evaluation_profile",
    "factor_verdict",
    "factor_verdict_reasons",
    "campaign_triage",
    "campaign_triage_reasons",
    "promotion_decision",
    "promotion_reasons",
    "promotion_blockers",
    "split_description",
    "split_contract",
    "split_policy",
    "split_source",
    "is_start",
    "is_end",
    "oos_start",
    "oos_end",
    "split_embargo_days",
    "split_min_oos_dates",
    "split_min_is_dates",
    "split_n_dates",
    "split_n_is_dates",
    "split_n_oos_dates",
    "split_target_horizon",
    "split_rebalance_step",
)


class SingleFactorArtifactPaths(TypedDict):
    run_manifest: Path
    metrics: Path
    factor_definition_json: Path
    signal_validation_json: Path
    portfolio_recipe_json: Path
    backtest_result_json: Path
    purged_kfold_summary: Path
    purged_kfold_folds: Path
    ic_timeseries: Path
    ic_decay: Path
    factor_autocorrelation: Path
    capacity_estimation: Path
    conditional_ic_by_magnitude: Path
    conditional_ic_by_cross_section_size: Path
    rolling_stability: Path
    group_returns: Path
    group_nav: Path
    quantile_membership: Path
    quantile_equal_weights: Path
    portfolio_weights: Path
    turnover: Path
    coverage: Path
    lag_sensitivity: Path
    random_baseline_null: Path
    daily_pnl_attribution: Path
    factor_definition: Path
    summary: Path
    experiment_card: Path
    research_tearsheet: Path
    research_tearsheet_pdf: Path
    integrity_report_json: Path
    integrity_report_markdown: Path
    portfolio_validation_summary: Path
    portfolio_validation_metrics: Path
    portfolio_validation_package: Path
    portfolio_validation_markdown: Path


def export_artifact_bundle(
    *,
    spec: SingleFactorCaseSpec,
    evaluation_result: SingleFactorEvaluationResult,
    evaluation_config: ResearchEvaluationConfig,
    integrity_report: IntegrityReport | None,
    output_dir: str | Path,
    spec_path: str | Path | None = None,
    vault_root: str | Path | None = None,
    vault_export_mode: str = "versioned",
    custom_factor_source: Mapping[str, object] | None = None,
    defer_vault_export: bool = False,
) -> SingleFactorArtifactPaths:
    """Write standardized artifact bundle for one single-factor case run."""

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: SingleFactorArtifactPaths = {
        "run_manifest": out_dir / "run_manifest.json",
        "metrics": out_dir / "metrics.json",
        "factor_definition_json": out_dir / "factor_definition.json",
        "signal_validation_json": out_dir / "signal_validation.json",
        "portfolio_recipe_json": out_dir / "portfolio_recipe.json",
        "backtest_result_json": out_dir / "backtest_result.json",
        "purged_kfold_summary": out_dir / "purged_kfold_summary.json",
        "purged_kfold_folds": out_dir / "purged_kfold_folds.csv",
        "ic_timeseries": out_dir / "ic_timeseries.csv",
        "ic_decay": out_dir / "ic_decay.csv",
        "factor_autocorrelation": out_dir / "factor_autocorrelation.csv",
        "capacity_estimation": out_dir / "capacity_estimation.csv",
        "conditional_ic_by_magnitude": out_dir / "conditional_ic_by_magnitude.csv",
        "conditional_ic_by_cross_section_size": (
            out_dir / "conditional_ic_by_cross_section_size.csv"
        ),
        "rolling_stability": out_dir / "rolling_stability.csv",
        "group_returns": out_dir / "group_returns.csv",
        "group_nav": out_dir / "group_nav.csv",
        "quantile_membership": out_dir / "quantile_membership.csv",
        "quantile_equal_weights": out_dir / "quantile_equal_weights.csv",
        "portfolio_weights": out_dir / "portfolio_weights.csv",
        "turnover": out_dir / "turnover.csv",
        "coverage": out_dir / "coverage.csv",
        "lag_sensitivity": out_dir / "lag_sensitivity.csv",
        "random_baseline_null": out_dir / "random_baseline_null.csv",
        "daily_pnl_attribution": out_dir / "daily_pnl_attribution.csv",
        "factor_definition": out_dir / "factor_definition.yaml",
        "summary": out_dir / "summary.md",
        "experiment_card": out_dir / "experiment_card.md",
        "research_tearsheet": out_dir / "research_tearsheet.json",
        "research_tearsheet_pdf": out_dir / "research_tearsheet.pdf",
        "integrity_report_json": out_dir / "integrity_report.json",
        "integrity_report_markdown": out_dir / "integrity_report.md",
        "portfolio_validation_summary": (
            out_dir / "level2_portfolio_validation" / "portfolio_validation_summary.json"
        ),
        "portfolio_validation_metrics": (
            out_dir / "level2_portfolio_validation" / "portfolio_validation_metrics.json"
        ),
        "portfolio_validation_package": (
            out_dir / "level2_portfolio_validation" / "portfolio_validation_package.json"
        ),
        "portfolio_validation_markdown": (
            out_dir / "level2_portfolio_validation" / "portfolio_validation_package.md"
        ),
    }

    evaluation_result.ic_timeseries.to_csv(paths["ic_timeseries"], index=False)
    evaluation_result.ic_decay.to_csv(paths["ic_decay"], index=False)
    evaluation_result.factor_autocorrelation.to_csv(
        paths["factor_autocorrelation"],
        index=False,
    )
    evaluation_result.capacity_estimation.to_csv(
        paths["capacity_estimation"],
        index=False,
    )
    evaluation_result.conditional_ic_by_magnitude.to_csv(
        paths["conditional_ic_by_magnitude"],
        index=False,
    )
    evaluation_result.conditional_ic_by_cross_section_size.to_csv(
        paths["conditional_ic_by_cross_section_size"],
        index=False,
    )
    evaluation_result.rolling_stability.to_csv(paths["rolling_stability"], index=False)
    evaluation_result.group_returns.to_csv(paths["group_returns"], index=False)
    build_group_nav_table(
        evaluation_result.group_returns,
        rebalance_frequency=spec.rebalance_frequency,
        label_horizon=int(spec.target.horizon),
    ).to_csv(paths["group_nav"], index=False)
    quantile_membership = evaluation_result.experiment_result.quantile_assignments_df.copy()
    quantile_membership.to_csv(paths["quantile_membership"], index=False)
    _build_quantile_equal_weights(quantile_membership).to_csv(
        paths["quantile_equal_weights"],
        index=False,
    )
    portfolio_weights = evaluation_result.experiment_result.portfolio_weights_df
    if portfolio_weights is None:
        portfolio_weights = pd.DataFrame(columns=["date", "asset", "weight"])
    portfolio_weights.to_csv(paths["portfolio_weights"], index=False)
    evaluation_result.turnover.to_csv(paths["turnover"], index=False)
    evaluation_result.coverage.to_csv(paths["coverage"], index=False)
    evaluation_result.lag_sensitivity.to_csv(paths["lag_sensitivity"], index=False)
    evaluation_result.random_baseline_null.to_csv(paths["random_baseline_null"], index=False)
    evaluation_result.daily_pnl_attribution.to_csv(paths["daily_pnl_attribution"], index=False)

    factor_definition_yaml = _dump_yaml_payload(_compact_spec_payload(spec))
    paths["factor_definition"].write_text(factor_definition_yaml, encoding="utf-8")

    report = integrity_report or build_integrity_report(
        (),
        context={
            "pipeline": "real_case_single_factor",
            "case_name": spec.name,
            "note": "integrity report was not provided by caller",
        },
    )
    write_integrity_report_json(report, paths["integrity_report_json"])
    write_integrity_report_markdown(report, paths["integrity_report_markdown"])

    portfolio_validation_bundle = build_level2_portfolio_validation_bundle(
        experiment_result=evaluation_result.experiment_result,
        key_metrics=evaluation_result.metrics,
        case_context={
            "case_name": spec.name,
            "case_id": spec.name,
            "case_output_dir": str(out_dir),
            "package_type": "single_factor",
            "rebalance_frequency": spec.rebalance_frequency,
            "experiment_name": spec.name,
        },
        promotion_decision={
            "verdict": evaluation_result.metrics.get("promotion_decision"),
            "reasons": evaluation_result.metrics.get("promotion_reasons"),
            "blockers": evaluation_result.metrics.get("promotion_blockers"),
            "source": "level2_promotion_gate",
        },
        config=evaluation_config.level2_portfolio_validation,
    )
    portfolio_validation_paths = export_level2_portfolio_validation_bundle(
        portfolio_validation_bundle,
        out_dir / "level2_portfolio_validation",
    )
    paths["portfolio_validation_summary"] = portfolio_validation_paths["summary"]
    paths["portfolio_validation_metrics"] = portfolio_validation_paths["metrics"]
    paths["portfolio_validation_package"] = portfolio_validation_paths["package_json"]
    paths["portfolio_validation_markdown"] = portfolio_validation_paths["package_markdown"]

    portfolio_validation_payload = portfolio_validation_bundle.to_dict()
    portfolio_validation_summary = portfolio_validation_bundle.summary
    metrics_for_payload = dict(evaluation_result.metrics)
    metrics_for_payload["portfolio_validation_status"] = portfolio_validation_summary.get(
        "validation_status"
    )
    metrics_for_payload["portfolio_validation_recommendation"] = portfolio_validation_summary.get(
        "recommendation"
    )
    metrics_for_payload["portfolio_validation_remains_credible"] = portfolio_validation_summary.get(
        "remains_credible_at_portfolio_level"
    )
    metrics_for_payload["portfolio_validation_major_risks"] = portfolio_validation_summary.get(
        "major_risks"
    )
    metrics_for_payload["portfolio_validation_base_mean_portfolio_return"] = (
        portfolio_validation_summary.get("base_mean_portfolio_return")
    )
    metrics_for_payload["portfolio_validation_base_mean_turnover"] = (
        portfolio_validation_summary.get("base_mean_turnover")
    )
    metrics_for_payload["portfolio_validation_base_cost_adjusted_return_review_rate"] = (
        portfolio_validation_summary.get("base_cost_adjusted_return_review_rate")
    )
    robustness_summary_raw = portfolio_validation_summary.get("portfolio_robustness_summary")
    robustness_summary = (
        dict(robustness_summary_raw) if isinstance(robustness_summary_raw, Mapping) else {}
    )
    metrics_for_payload["portfolio_validation_robustness_label"] = robustness_summary.get(
        "taxonomy_label"
    )
    metrics_for_payload["portfolio_validation_support_reasons"] = robustness_summary.get(
        "support_reasons"
    )
    metrics_for_payload["portfolio_validation_fragility_reasons"] = robustness_summary.get(
        "fragility_reasons"
    )
    metrics_for_payload["portfolio_validation_scenario_sensitivity_notes"] = robustness_summary.get(
        "scenario_sensitivity_notes"
    )
    metrics_for_payload["portfolio_validation_benchmark_support_note"] = robustness_summary.get(
        "benchmark_relative_support_note"
    )
    metrics_for_payload["portfolio_validation_cost_sensitivity_note"] = robustness_summary.get(
        "cost_sensitivity_note"
    )
    metrics_for_payload["portfolio_validation_concentration_turnover_note"] = (
        robustness_summary.get("concentration_turnover_risk_note")
    )
    benchmark_eval_raw = portfolio_validation_bundle.metrics.get("benchmark_relative_evaluation")
    benchmark_eval = dict(benchmark_eval_raw) if isinstance(benchmark_eval_raw, Mapping) else {}
    metrics_for_payload["portfolio_validation_benchmark_relative_status"] = benchmark_eval.get(
        "status"
    )
    metrics_for_payload["portfolio_validation_benchmark_relative_assessment"] = benchmark_eval.get(
        "assessment"
    )
    metrics_for_payload["portfolio_validation_benchmark_name"] = benchmark_eval.get(
        "benchmark_name"
    )
    metrics_for_payload["portfolio_validation_benchmark_excess_return"] = benchmark_eval.get(
        "benchmark_excess_return"
    )
    metrics_for_payload["portfolio_validation_benchmark_active_return"] = benchmark_eval.get(
        "benchmark_active_return"
    )
    metrics_for_payload["portfolio_validation_benchmark_information_ratio"] = benchmark_eval.get(
        "benchmark_information_ratio"
    )
    metrics_for_payload["portfolio_validation_benchmark_tracking_error"] = benchmark_eval.get(
        "benchmark_tracking_error"
    )
    metrics_for_payload["portfolio_validation_benchmark_relative_max_drawdown"] = (
        benchmark_eval.get("benchmark_relative_max_drawdown")
    )
    metrics_for_payload["portfolio_validation_benchmark_relative_risks"] = benchmark_eval.get(
        "risk_flags"
    )
    level12_transition = project_level12_transition_summary(metrics_for_payload)
    metrics_for_payload["level12_transition_summary"] = level12_transition
    metrics_for_payload["level12_transition_label"] = level12_transition["transition_label"]
    metrics_for_payload["level12_transition_interpretation"] = level12_transition[
        "transition_interpretation"
    ]
    metrics_for_payload["level12_transition_reasons"] = level12_transition["key_transition_reasons"]
    metrics_for_payload["level12_transition_confirmation_note"] = level12_transition[
        "confirmation_vs_degradation_note"
    ]
    split_contract_payload = _split_contract_payload(metrics_for_payload)

    summary_md = render_summary_markdown(
        spec=spec,
        metrics=metrics_for_payload,
        output_dir=out_dir,
    )
    paths["summary"].write_text(summary_md, encoding="utf-8")

    card_md = render_experiment_card_markdown(
        spec=spec,
        metrics=metrics_for_payload,
        result=evaluation_result.experiment_result,
    )
    paths["experiment_card"].write_text(card_md, encoding="utf-8")

    metrics_payload = {
        "metrics": _to_jsonable(
            _compact_metrics_payload(
                metrics_for_payload,
                profile_name=evaluation_config.profile_name,
            )
        ),
        "coverage_by_date_summary": _build_coverage_by_date_summary(
            evaluation_result.coverage
        ),
    }
    _write_json(paths["metrics"], metrics_payload)

    factor_definition_payload = _build_factor_definition_payload(
        spec=spec,
        output_paths=paths,
        custom_factor_source=custom_factor_source,
    )
    _write_json(paths["factor_definition_json"], factor_definition_payload)

    signal_validation_payload = _build_signal_validation_payload(
        spec=spec,
        metrics_payload=metrics_payload,
        output_paths=paths,
    )
    _write_json(paths["signal_validation_json"], signal_validation_payload)

    portfolio_recipe_payload = _build_portfolio_recipe_payload(
        spec=spec,
        metrics_for_payload=metrics_for_payload,
        portfolio_validation_payload=portfolio_validation_payload,
        output_paths=paths,
    )
    _write_json(paths["portfolio_recipe_json"], portfolio_recipe_payload)

    backtest_result_payload = _build_backtest_result_payload(
        spec=spec,
        metrics_for_payload=metrics_for_payload,
        group_returns_df=evaluation_result.group_returns,
        output_paths=paths,
    )
    _write_json(paths["backtest_result_json"], backtest_result_payload)

    purged_kfold = build_purged_kfold_diagnostics(
        experiment_result=evaluation_result.experiment_result,
        label_horizon=int(spec.target.horizon),
    )
    _write_json(paths["purged_kfold_summary"], purged_kfold.summary)
    purged_kfold.folds.to_csv(paths["purged_kfold_folds"], index=False)

    tearsheet_payload = build_research_tearsheet_payload(
        metrics_path=paths["metrics"],
        artifact_paths={
            "ic_decay": paths["ic_decay"],
            "group_returns": paths["group_returns"],
            "turnover": paths["turnover"],
            "rolling_stability": paths["rolling_stability"],
            "ic_timeseries": paths["ic_timeseries"],
            "coverage": paths["coverage"],
            "backtest_result_json": paths["backtest_result_json"],
        },
        meta={
            "factor_name": spec.factor_name,
            "universe_name": spec.universe.name,
            "target_kind": spec.target.kind,
            "target_horizon": spec.target.horizon,
            "target_execution_price_mode": spec.target.execution_price_mode,
            "split_contract": split_contract_payload,
        },
    )
    _write_json(paths["research_tearsheet"], tearsheet_payload)
    export_research_tearsheet_pdf(
        payload=tearsheet_payload,
        output_path=paths["research_tearsheet_pdf"],
    )

    inputs_payload: dict[str, object] = {
        "prices_path": spec.prices_path,
        "factor_path": spec.factor_path,
        "factor_name": spec.factor_name,
        "universe_path": spec.universe.path,
        "neutralization_exposures_path": spec.neutralization.exposures_path,
    }
    if custom_factor_source is not None:
        inputs_payload["custom_factor_source"] = dict(custom_factor_source)

    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "artifact_type": "real_case_single_factor_bundle",
        "run_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "case_name": spec.name,
        "spec_path": str(Path(spec_path).resolve()) if spec_path is not None else None,
        "inputs": inputs_payload,
        "outputs": {name: str(path) for name, path in paths.items()},
        "required_bundle_files": list(REQUIRED_BUNDLE_FILES),
        "integrity_summary": report.summary.to_dict(),
        "evaluation_standard": {
            "profile_name": evaluation_config.profile_name,
            "snapshot": research_evaluation_audit_snapshot(evaluation_config),
        },
        "split_contract": split_contract_payload,
        "vault_export": {
            "enabled": False,
            "mode": "skip",
            "target_paths": [],
            "status": "skipped",
            "error": None,
        },
    }
    if custom_factor_source is not None:
        manifest["custom_factor_source"] = dict(custom_factor_source)

    _write_json(paths["run_manifest"], manifest)

    if defer_vault_export:
        # Caller (CLI) will run vault export after backend contract finalize so
        # that backend_run_receipt.json / comparison_summary.json land in the
        # same export pass and the vault manifest copy reflects the final
        # backend_run_contract block.
        return paths
    resolved_vault = resolve_vault_root(vault_root)
    enabled = resolved_vault is not None and vault_export_mode.strip().lower() != "skip"
    vault_result = export_to_vault(
        {
            "experiment_card_path": paths["experiment_card"],
            "summary_path": paths["summary"],
            "manifest_path": paths["run_manifest"],
        },
        case_name=spec.name,
        vault_root=vault_root,
        mode=vault_export_mode,
    )
    manifest["vault_export"] = vault_result.to_manifest_dict(enabled=enabled)
    _write_json(paths["run_manifest"], manifest)

    if vault_result.status == "failed":
        logger.warning(
            "Vault export failed for single-factor case %s: %s",
            spec.name,
            vault_result.error,
        )
    if vault_result.success and vault_result.target_paths:
        _sync_exported_manifest_copies(paths["run_manifest"], vault_result.target_paths)

    return paths


def _build_factor_definition_payload(
    *,
    spec: SingleFactorCaseSpec,
    output_paths: SingleFactorArtifactPaths,
    custom_factor_source: Mapping[str, object] | None = None,
) -> dict[str, object]:
    archive_identity = (
        _text_or_none((custom_factor_source or {}).get("archive_identity"))
        or _text_or_none(spec.archive_identity)
        or spec.factor_name
    )
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_factor_definition",
        "case_name": spec.name,
        "package_type": "single_factor",
        "factor_name": spec.factor_name,
        "archive_identity": archive_identity,
        "spec": _compact_spec_payload(spec),
        "source_artifacts": {
            "factor_definition_yaml_path": str(output_paths["factor_definition"]),
            "run_manifest_path": str(output_paths["run_manifest"]),
        },
        "fallback_derived_fields": [],
    }
    if custom_factor_source is not None:
        payload["custom_factor_source"] = dict(custom_factor_source)
    return payload


def _build_signal_validation_payload(
    *,
    spec: SingleFactorCaseSpec,
    metrics_payload: Mapping[str, object],
    output_paths: SingleFactorArtifactPaths,
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_signal_validation",
        "case_name": spec.name,
        "package_type": "single_factor",
        "metrics": _to_jsonable(
            _compact_signal_validation_metrics(_as_object(metrics_payload.get("metrics")))
        ),
        "coverage_by_date_summary": _to_jsonable(
            _as_object(metrics_payload.get("coverage_by_date_summary"))
        ),
        "source_artifacts": {
            "metrics_path": str(output_paths["metrics"]),
            "ic_timeseries_path": str(output_paths["ic_timeseries"]),
            "ic_decay_path": str(output_paths["ic_decay"]),
            "factor_autocorrelation_path": str(output_paths["factor_autocorrelation"]),
            "capacity_estimation_path": str(output_paths["capacity_estimation"]),
            "conditional_ic_by_magnitude_path": str(output_paths["conditional_ic_by_magnitude"]),
            "conditional_ic_by_cross_section_size_path": str(
                output_paths["conditional_ic_by_cross_section_size"]
            ),
            "rolling_stability_path": str(output_paths["rolling_stability"]),
            "coverage_path": str(output_paths["coverage"]),
        },
        "fallback_derived_fields": [],
    }


def _build_portfolio_recipe_payload(
    *,
    spec: SingleFactorCaseSpec,
    metrics_for_payload: Mapping[str, object],
    portfolio_validation_payload: Mapping[str, object],
    output_paths: SingleFactorArtifactPaths,
) -> dict[str, object]:
    controls = build_portfolio_recipe_controls(
        metrics_for_payload=metrics_for_payload,
        portfolio_validation_payload=portfolio_validation_payload,
    )
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_portfolio_recipe",
        "case_name": spec.name,
        "package_type": "single_factor",
        "recipe_context": {
            "factor_name": spec.factor_name,
            "rebalance_frequency": spec.rebalance_frequency,
            "universe_name": spec.universe.name,
            "target_horizon": spec.target.horizon,
            "neutralization_enabled": bool(spec.neutralization.enabled),
        },
        "portfolio_validation_summary": _to_jsonable(
            _as_object(portfolio_validation_payload.get("portfolio_validation_summary"))
        ),
        "portfolio_validation_metrics": _to_jsonable(
            _as_object(portfolio_validation_payload.get("portfolio_validation_metrics"))
        ),
        "portfolio_validation_package": _to_jsonable(
            _as_object(portfolio_validation_payload.get("portfolio_validation_package"))
        ),
        "turnover_penalty_settings": controls["turnover_penalty_settings"],
        "transaction_cost_assumptions": controls["transaction_cost_assumptions"],
        "position_limits": controls["position_limits"],
        "source_artifacts": {
            "portfolio_validation_summary_path": str(output_paths["portfolio_validation_summary"]),
            "portfolio_validation_metrics_path": str(output_paths["portfolio_validation_metrics"]),
            "portfolio_validation_package_path": str(output_paths["portfolio_validation_package"]),
            "metrics_path": str(output_paths["metrics"]),
        },
        "fallback_derived_fields": [],
        "metrics_reference": {
            "transaction_cost_one_way_rate": _finite_if_number(
                metrics_for_payload.get("transaction_cost_one_way_rate")
            ),
            "base_weighting_method": _text_or_none(
                metrics_for_payload.get("base_weighting_method")
            ),
            "research_evaluation_profile": _text_or_none(
                metrics_for_payload.get("research_evaluation_profile")
            ),
        },
    }


def _build_backtest_result_payload(
    *,
    spec: SingleFactorCaseSpec,
    metrics_for_payload: Mapping[str, object],
    group_returns_df: pd.DataFrame,
    output_paths: SingleFactorArtifactPaths,
) -> dict[str, object]:
    summary, fallback_fields = build_backtest_summary_payload(
        group_returns_df=group_returns_df,
        rebalance_frequency=spec.rebalance_frequency,
        metrics_for_payload=metrics_for_payload,
        label_horizon=int(spec.target.horizon),
    )
    compact_summary = _compact_backtest_summary(summary)
    compact_fallback_fields = [
        field
        for field in fallback_fields
        if field in compact_summary and field not in _BACKTEST_OMITTED_DETAIL_FIELDS
    ]
    split_contract = _split_contract_payload(metrics_for_payload)
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_backtest_result",
        "case_name": spec.name,
        "package_type": "single_factor",
        "rebalance_frequency": spec.rebalance_frequency,
        "target_horizon": int(spec.target.horizon),
        "summary": compact_summary,
        "source_artifacts": {
            "group_returns_path": str(output_paths["group_returns"]),
            "group_nav_path": str(output_paths["group_nav"]),
            "turnover_path": str(output_paths["turnover"]),
            "metrics_path": str(output_paths["metrics"]),
        },
        "fallback_derived_fields": compact_fallback_fields,
    }
    if split_contract:
        payload.update(_split_contract_top_level_fields(split_contract))
    return payload


def _write_json(path: Path, payload: Mapping[str, object], *, pretty: bool = True) -> None:
    jsonable_payload = _to_jsonable(payload)
    if not isinstance(jsonable_payload, Mapping):
        raise AlphaLabDataError(f"{path} JSON payload root must be an object")
    validate_level12_artifact_payload(
        jsonable_payload,
        artifact_name=path.name,
        source=path,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(jsonable_payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        else:
            json.dump(
                jsonable_payload,
                f,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=False,
            )
        f.write("\n")


def _to_jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, float):
        return _finite_or_none(value)
    return value


def _dump_yaml_payload(payload: Mapping[str, object]) -> str:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - import guard
        raise AlphaLabDataError("PyYAML is required to serialize YAML artifacts") from exc

    text = yaml.safe_dump(
        _to_jsonable(dict(payload)),
        sort_keys=False,
        allow_unicode=False,
    )
    return str(text)


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _series_float_stat(frame: pd.DataFrame, column: str, stat: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = (
        pd.to_numeric(frame[column], errors="coerce")
        .replace([float("inf"), -float("inf")], pd.NA)
        .dropna()
    )
    if values.empty:
        return None
    if stat == "mean":
        return _finite_or_none(float(values.mean()))
    if stat == "median":
        return _finite_or_none(float(values.median()))
    if stat == "min":
        return _finite_or_none(float(values.min()))
    if stat == "max":
        return _finite_or_none(float(values.max()))
    if stat == "sum":
        return _finite_or_none(float(values.sum()))
    return None


def _coverage_int_stat(frame: pd.DataFrame, column: str, stat: str) -> int | None:
    value = _series_float_stat(frame, column, stat)
    return int(round(value)) if value is not None else None


def _build_coverage_by_date_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {
            "n_dates": 0,
            "n_valid_dates": 0,
            "date_coverage": None,
            "mean_coverage": None,
            "min_coverage": None,
            "mean_asset_coverage": None,
            "median_asset_coverage": None,
            "min_asset_coverage": None,
            "max_asset_coverage": None,
            "overall_sample_coverage": None,
            "avg_assets": None,
            "coverage_warmup_excluded_days": 0,
        }
    raw_frame = frame
    warmup_excluded_days = 0
    if "coverage_eval_included" in frame.columns:
        included = frame["coverage_eval_included"].fillna(True).astype(bool)
        warmup_excluded_days = int((~included).sum())
        frame = frame.loc[included].reset_index(drop=True)
        if frame.empty:
            frame = raw_frame
    n_dates = int(frame["date"].nunique()) if "date" in frame.columns else int(len(frame))
    valid_sample = pd.to_numeric(
        frame["valid_sample_count"]
        if "valid_sample_count" in frame.columns
        else pd.Series(dtype=float),
        errors="coerce",
    ).fillna(0)
    n_valid_dates = int((valid_sample > 0).sum())
    total_eligible = _series_float_stat(frame, "eligible_count", "sum")
    total_valid_samples = _series_float_stat(frame, "valid_sample_count", "sum")
    overall_sample_coverage = (
        _finite_or_none(float(total_valid_samples) / float(total_eligible))
        if total_eligible and total_valid_samples is not None
        else None
    )
    mean_asset_coverage = _series_float_stat(frame, "asset_coverage", "mean")
    min_asset_coverage = _series_float_stat(frame, "asset_coverage", "min")
    return {
        "n_dates": n_dates,
        "n_valid_dates": n_valid_dates,
        "date_coverage": _finite_or_none(n_valid_dates / n_dates) if n_dates else None,
        "mean_coverage": mean_asset_coverage,
        "min_coverage": min_asset_coverage,
        "mean_asset_coverage": mean_asset_coverage,
        "median_asset_coverage": _series_float_stat(frame, "asset_coverage", "median"),
        "min_asset_coverage": min_asset_coverage,
        "max_asset_coverage": _series_float_stat(frame, "asset_coverage", "max"),
        "overall_sample_coverage": overall_sample_coverage,
        "avg_assets": _series_float_stat(frame, "eligible_count", "mean"),
        "avg_valid_score_assets": _series_float_stat(frame, "valid_score_count", "mean"),
        "avg_valid_forward_return_assets": _series_float_stat(
            frame,
            "valid_forward_return_count",
            "mean",
        ),
        "total_eligible_samples": _coverage_int_stat(frame, "eligible_count", "sum"),
        "total_valid_samples": _coverage_int_stat(frame, "valid_sample_count", "sum"),
        "coverage_warmup_excluded_days": warmup_excluded_days,
        "mean_asset_coverage_raw": _series_float_stat(raw_frame, "asset_coverage", "mean"),
        "min_asset_coverage_raw": _series_float_stat(raw_frame, "asset_coverage", "min"),
    }


def _finite_if_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if not isinstance(value, (int, float)):
        return None
    return _finite_or_none(float(value))


def _text_or_none(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _as_object(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _split_contract_payload(metrics: Mapping[str, object]) -> dict[str, object]:
    raw = metrics.get("split_contract")
    return {str(key): value for key, value in raw.items()} if isinstance(raw, Mapping) else {}


def _split_contract_top_level_fields(split_contract: Mapping[str, object]) -> dict[str, object]:
    fields: dict[str, object] = {"split_contract": dict(split_contract)}
    for key in ("is_start", "is_end", "oos_start", "oos_end"):
        if key in split_contract:
            fields[key] = split_contract[key]
    return fields


def _compact_metrics_payload(
    metrics: Mapping[str, object],
    *,
    profile_name: str,
) -> dict[str, object]:
    normalized_profile = profile_name.strip().lower()
    if normalized_profile in _FAST_SCREEN_PROFILE_NAMES:
        keep = _FAST_SCREEN_REQUIRED_CONTRACT_KEYS + _FAST_SCREEN_CORE_METRIC_KEYS
        compact = {key: metrics[key] for key in keep if key in metrics}
        return compact
    return {str(key): value for key, value in metrics.items()}


def _compact_backtest_summary(summary: Mapping[str, object]) -> dict[str, object]:
    keep = (
        "annualized_return",
        "annualized_volatility",
        "sharpe",
        "sortino",
        "max_drawdown",
        "calmar",
        "win_rate",
        "turnover",
        "information_ratio",
        "excess_return_vs_benchmark",
        "tracking_error",
        "pre_cost_return",
        "post_cost_return",
    )
    compact = {key: summary[key] for key in keep if key in summary}
    compact.update(
        {
            "rolling_sharpe": None,
            "rolling_drawdown": None,
            "nav_points": summary.get("nav_points", []),
            "monthly_return_table": [],
            "drawdown_table": [],
            "subperiod_analysis": None,
            "regime_analysis": None,
            "nav_series_policy": summary.get("nav_series_policy"),
            "nav_point_interval": summary.get("nav_point_interval"),
            "nav_rebalance_step": summary.get("nav_rebalance_step"),
            "label_horizon": summary.get("label_horizon"),
            "statistics_series_policy": summary.get("statistics_series_policy"),
            "statistics_rebalance_step": summary.get("statistics_rebalance_step"),
            "statistics_periods_per_year": summary.get("statistics_periods_per_year"),
            "max_drawdown_oos": summary.get("max_drawdown_oos"),
            "pre_cost_return_oos": summary.get("pre_cost_return_oos"),
            "post_cost_return_oos": summary.get("post_cost_return_oos"),
            "turnover_oos": summary.get("turnover_oos"),
        }
    )
    return compact


def _compact_signal_validation_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    keep = (
        "research_evaluation_profile",
        "factor_verdict",
        "campaign_triage",
        "promotion_decision",
        "mean_ic",
        "mean_rank_ic",
        "mean_mutual_information",
        "mutual_information_ir",
        "ic_ir",
        "mean_long_short_return",
        "mean_long_short_turnover",
        "ic_half_life_horizon",
        "ic_decay_rebalance_ratio",
        "capacity_status",
        "estimated_capacity_upper_bound",
        "conditional_ic_extreme_minus_base_ic",
        "eval_coverage_ratio_mean",
        "eval_coverage_ratio_min",
        "rolling_instability_flags",
        "uncertainty_flags",
        "instability_flags",
    )
    return {key: metrics[key] for key in keep if key in metrics}


def _compact_spec_payload(spec: SingleFactorCaseSpec) -> dict[str, object]:
    payload = {
        "name": spec.name,
        "factor_name": spec.factor_name,
        "direction": spec.direction,
        "rebalance_frequency": spec.rebalance_frequency,
        "n_quantiles": spec.n_quantiles,
        "target": {
            "kind": spec.target.kind,
            "horizon": spec.target.horizon,
            "execution_price_mode": spec.target.execution_price_mode,
        },
        "universe": {
            "name": spec.universe.name,
            "path": spec.universe.path,
        },
        "neutralization": {
            "enabled": bool(spec.neutralization.enabled),
            "exposures_path": spec.neutralization.exposures_path,
        },
        "capacity": {
            "enabled": bool(spec.capacity.enabled),
            "participation_rate": spec.capacity.participation_rate,
            "adv_lookback": spec.capacity.adv_lookback,
        },
        "paths": {
            "prices_path": spec.prices_path,
            "factor_path": spec.factor_path,
        },
        "transaction_cost": {
            "one_way_rate": spec.transaction_cost.one_way_rate,
        },
    }
    factor_input = _compact_factor_input_payload(spec.factor_input)
    if factor_input is not None:
        payload["factor_input"] = factor_input
    if spec.archive_identity:
        payload["archive_identity"] = spec.archive_identity
    return payload


def _compact_factor_input_payload(factor_input: object) -> dict[str, object] | None:
    if factor_input is None:
        return None
    mode = _text_or_none(getattr(factor_input, "mode", None))
    disable_pipeline_preprocess = getattr(factor_input, "disable_pipeline_preprocess", None)
    recipe = getattr(factor_input, "recipe", None)
    recipe_mapping = recipe if isinstance(recipe, Mapping) else {}
    payload: dict[str, object] = {
        "mode": mode or "file",
        "disable_pipeline_preprocess": bool(disable_pipeline_preprocess),
    }
    if recipe_mapping:
        base = recipe_mapping.get("base")
        base_mapping = base if isinstance(base, Mapping) else {}
        payload["recipe_summary"] = {
            "method": _text_or_none(base_mapping.get("method")),
            "fields": sorted(str(key) for key in recipe_mapping.keys()),
        }
    return payload


_BACKTEST_OMITTED_DETAIL_FIELDS = frozenset(
    {
        "rolling_sharpe",
        "rolling_drawdown",
        "monthly_return_table",
        "drawdown_table",
        "subperiod_analysis",
        "regime_analysis",
    }
)


def _build_quantile_equal_weights(quantile_membership: pd.DataFrame) -> pd.DataFrame:
    """Derive equal-weight holdings for each quantile bucket from assignments."""

    columns = ["date", "asset", "factor", "quantile", "weight"]
    if quantile_membership.empty:
        return pd.DataFrame(columns=columns)

    required = {"date", "asset", "quantile"}
    if not required.issubset(quantile_membership.columns):
        return pd.DataFrame(columns=columns)

    frame = quantile_membership.copy()
    if "factor" not in frame.columns:
        frame["factor"] = ""
    frame = frame.dropna(subset=["date", "asset", "quantile"]).copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)

    counts = frame.groupby(["date", "factor", "quantile"], dropna=False)["asset"].transform(
        "count"
    )
    frame["weight"] = 1.0 / counts.astype(float)
    return frame[columns].reset_index(drop=True)


def _sync_exported_manifest_copies(
    local_manifest_path: Path,
    target_paths: tuple[str, ...],
) -> None:
    """Ensure vault-side manifest copies contain final vault_export payload."""

    for raw in target_paths:
        target = Path(raw)
        if not target.name.endswith("run_manifest.json"):
            continue
        try:
            shutil.copy2(local_manifest_path, target)
        except OSError as exc:
            logger.warning(
                "Failed to sync single-factor vault manifest copy %s: %s",
                target,
                exc,
            )
