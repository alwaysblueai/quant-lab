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
from alpha_lab.key_metrics_contracts import project_level12_transition_summary
from alpha_lab.model_factor import ModelFactorBuildResult, ModelSpec, resolve_model_spec_params
from alpha_lab.real_cases.artifact_enrichment import (
    build_backtest_summary_payload,
    build_portfolio_recipe_controls,
)
from alpha_lab.real_cases.common_spec import parse_mapping_payload
from alpha_lab.real_cases.single_factor.evaluate import SingleFactorEvaluationResult
from alpha_lab.reporting.level2_portfolio_validation import (
    build_level2_portfolio_validation_bundle,
    export_level2_portfolio_validation_bundle,
)
from alpha_lab.reporting.purged_kfold_diagnostics import build_purged_kfold_diagnostics
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

from .spec import ModelFactorCaseSpec, dump_spec_yaml, spec_to_dict
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
    "model_selection.json",
    "model_definition.json",
    "feature_manifest.json",
    "diagnostics.json",
    "training_log.csv",
    "feature_importance.csv",
    "ic_timeseries.csv",
    "ic_decay.csv",
    "rolling_stability.csv",
    "group_returns.csv",
    "turnover.csv",
    "coverage.csv",
    "model_factor_definition.yaml",
    "summary.md",
    "experiment_card.md",
    "integrity_report.json",
    "integrity_report.md",
    "level2_portfolio_validation/portfolio_validation_summary.json",
    "level2_portfolio_validation/portfolio_validation_metrics.json",
    "level2_portfolio_validation/portfolio_validation_package.json",
    "level2_portfolio_validation/portfolio_validation_package.md",
)


class ModelFactorArtifactPaths(TypedDict):
    run_manifest: Path
    metrics: Path
    factor_definition_json: Path
    signal_validation_json: Path
    portfolio_recipe_json: Path
    backtest_result_json: Path
    purged_kfold_summary: Path
    purged_kfold_folds: Path
    model_selection_json: Path
    model_definition_json: Path
    feature_manifest_json: Path
    diagnostics: Path
    training_log: Path
    feature_importance: Path
    ic_timeseries: Path
    ic_decay: Path
    rolling_stability: Path
    group_returns: Path
    turnover: Path
    coverage: Path
    factor_definition: Path
    summary: Path
    experiment_card: Path
    integrity_report_json: Path
    integrity_report_markdown: Path
    portfolio_validation_summary: Path
    portfolio_validation_metrics: Path
    portfolio_validation_package: Path
    portfolio_validation_markdown: Path


def export_artifact_bundle(
    *,
    spec: ModelFactorCaseSpec,
    model_factor_result: ModelFactorBuildResult,
    diagnostics_payload: Mapping[str, object] | None,
    feature_manifest_payload: Mapping[str, object],
    evaluation_result: SingleFactorEvaluationResult,
    evaluation_config: ResearchEvaluationConfig,
    integrity_report: IntegrityReport | None,
    output_dir: str | Path,
    spec_path: str | Path | None = None,
    vault_root: str | Path | None = None,
    vault_export_mode: str = "versioned",
) -> ModelFactorArtifactPaths:
    """Write standardized artifact bundle for one model-factor case run."""

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: ModelFactorArtifactPaths = {
        "run_manifest": out_dir / "run_manifest.json",
        "metrics": out_dir / "metrics.json",
        "factor_definition_json": out_dir / "factor_definition.json",
        "signal_validation_json": out_dir / "signal_validation.json",
        "portfolio_recipe_json": out_dir / "portfolio_recipe.json",
        "backtest_result_json": out_dir / "backtest_result.json",
        "purged_kfold_summary": out_dir / "purged_kfold_summary.json",
        "purged_kfold_folds": out_dir / "purged_kfold_folds.csv",
        "model_selection_json": out_dir / "model_selection.json",
        "model_definition_json": out_dir / "model_definition.json",
        "feature_manifest_json": out_dir / "feature_manifest.json",
        "diagnostics": out_dir / "diagnostics.json",
        "training_log": out_dir / "training_log.csv",
        "feature_importance": out_dir / "feature_importance.csv",
        "ic_timeseries": out_dir / "ic_timeseries.csv",
        "ic_decay": out_dir / "ic_decay.csv",
        "rolling_stability": out_dir / "rolling_stability.csv",
        "group_returns": out_dir / "group_returns.csv",
        "turnover": out_dir / "turnover.csv",
        "coverage": out_dir / "coverage.csv",
        "factor_definition": out_dir / "model_factor_definition.yaml",
        "summary": out_dir / "summary.md",
        "experiment_card": out_dir / "experiment_card.md",
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

    _write_csv(paths["ic_timeseries"], evaluation_result.ic_timeseries)
    _write_csv(paths["ic_decay"], evaluation_result.ic_decay)
    _write_csv(paths["rolling_stability"], evaluation_result.rolling_stability)
    _write_csv(paths["group_returns"], evaluation_result.group_returns)
    _write_csv(paths["turnover"], evaluation_result.turnover)
    _write_csv(paths["coverage"], evaluation_result.coverage)
    write_diagnostics_artifact(
        output_dir=out_dir,
        diagnostics_payload=diagnostics_payload
        or {
            "schema_version": "1.0.0",
            "artifact_type": "alpha_lab_model_run_diagnostics",
            "generated_at_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "run_meta": {"case_name": spec.name, "status": "succeeded"},
            "stages": [],
            "events": [],
            "warnings": [],
            "data_health": {},
        },
    )
    training_log_df, training_log_notes = _prepare_training_log_for_export(
        model_factor_result.training_log_df
    )
    feature_importance_df, feature_importance_notes = _prepare_feature_importance_for_export(
        model_factor_result.feature_importance_df,
        model_family=spec.model.family,
    )
    _write_csv(paths["training_log"], training_log_df)
    _write_csv(paths["feature_importance"], feature_importance_df)
    missing_value_notes = [*training_log_notes, *feature_importance_notes]
    feature_preprocess_payload = _feature_preprocess_payload_for_artifacts(
        spec=spec,
        spec_path=spec_path,
    )
    cross_transform_default_applied = bool(
        feature_preprocess_payload.get("cross_sectional_transform_default_applied")
    )

    paths["factor_definition"].write_text(dump_spec_yaml(spec), encoding="utf-8")
    paths["summary"].write_text(
        render_summary_markdown(
            spec=spec,
            metrics=evaluation_result.metrics,
            model_diagnostics=model_factor_result.model_diagnostics,
            output_dir=out_dir,
            artifact_missing_value_notes=missing_value_notes,
            cross_sectional_transform=str(
                feature_preprocess_payload.get("cross_sectional_transform")
                or spec.feature_preprocess.cross_sectional_transform
            ),
            cross_sectional_transform_default_applied=cross_transform_default_applied,
        ),
        encoding="utf-8",
    )
    paths["experiment_card"].write_text(
        render_experiment_card_markdown(
            spec=spec,
            metrics=evaluation_result.metrics,
            model_diagnostics=model_factor_result.model_diagnostics,
            result=evaluation_result.experiment_result,
            artifact_missing_value_notes=missing_value_notes,
            cross_sectional_transform=str(
                feature_preprocess_payload.get("cross_sectional_transform")
                or spec.feature_preprocess.cross_sectional_transform
            ),
            cross_sectional_transform_default_applied=cross_transform_default_applied,
        ),
        encoding="utf-8",
    )

    report = integrity_report or build_integrity_report(
        (),
        context={
            "pipeline": "real_case_model_factor",
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
    metrics_for_payload.update(
        {
            "model_family": spec.model.family,
            "feature_count": len(spec.feature_columns),
            "trained_model_versions": model_factor_result.model_diagnostics.get(
                "trained_model_versions"
            ),
            "model_top_features": model_factor_result.model_diagnostics.get("top_features"),
            "mean_train_rows": model_factor_result.model_diagnostics.get("mean_train_rows"),
            "mean_score_assets": model_factor_result.model_diagnostics.get("mean_score_assets"),
        }
    )
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

    metrics_payload = {
        "metrics": _to_jsonable(metrics_for_payload),
        "coverage_by_date_summary": {
            "n_dates": int(evaluation_result.coverage["date"].nunique())
            if not evaluation_result.coverage.empty
            else 0,
            "mean_coverage": _finite_or_none(
                evaluation_result.coverage["coverage"].mean()
                if not evaluation_result.coverage.empty
                else float("nan")
            ),
            "min_coverage": _finite_or_none(
                evaluation_result.coverage["coverage"].min()
                if not evaluation_result.coverage.empty
                else float("nan")
            ),
        },
        "neutralization_summary": _to_jsonable(
            evaluation_result.neutralization_summary.to_dict(orient="records")
        ),
        "portfolio_validation_summary": _to_jsonable(
            portfolio_validation_payload["portfolio_validation_summary"]
        ),
        "portfolio_validation_metrics": _to_jsonable(
            portfolio_validation_payload["portfolio_validation_metrics"]
        ),
        "portfolio_validation_package": _to_jsonable(
            portfolio_validation_payload["portfolio_validation_package"]
        ),
    }
    _write_json(paths["metrics"], metrics_payload)

    _write_json(
        paths["factor_definition_json"],
        {
            "schema_version": "1.0.0",
            "artifact_type": "alpha_lab_factor_definition",
            "case_name": spec.name,
            "package_type": "single_factor",
            "factor_name": spec.factor_name,
            "spec": _to_jsonable(spec_to_dict(spec)),
            "source_artifacts": {
                "factor_definition_yaml_path": str(paths["factor_definition"]),
                "run_manifest_path": str(paths["run_manifest"]),
            },
            "fallback_derived_fields": [],
        },
    )
    _write_json(
        paths["signal_validation_json"],
        {
            "schema_version": "1.0.0",
            "artifact_type": "alpha_lab_signal_validation",
            "case_name": spec.name,
            "package_type": "single_factor",
            "metrics": _to_jsonable(_as_object(metrics_payload.get("metrics"))),
            "coverage_by_date_summary": _to_jsonable(
                _as_object(metrics_payload.get("coverage_by_date_summary"))
            ),
            "neutralization_summary": _to_jsonable(
                metrics_payload.get("neutralization_summary") or []
            ),
            "source_artifacts": {
                "metrics_path": str(paths["metrics"]),
                "ic_timeseries_path": str(paths["ic_timeseries"]),
                "rolling_stability_path": str(paths["rolling_stability"]),
                "coverage_path": str(paths["coverage"]),
            },
            "fallback_derived_fields": [],
        },
    )
    _write_json(
        paths["portfolio_recipe_json"],
        _build_portfolio_recipe_payload(
            spec=spec,
            metrics_for_payload=metrics_for_payload,
            portfolio_validation_payload=portfolio_validation_payload,
            output_paths=paths,
        ),
    )
    _write_json(
        paths["backtest_result_json"],
        _build_backtest_result_payload(
            spec=spec,
            metrics_for_payload=metrics_for_payload,
            group_returns_df=evaluation_result.group_returns,
            output_paths=paths,
        ),
    )
    purged_kfold = build_purged_kfold_diagnostics(
        experiment_result=evaluation_result.experiment_result,
        label_horizon=int(spec.target.horizon),
    )
    _write_json(paths["purged_kfold_summary"], purged_kfold.summary)
    _write_csv(paths["purged_kfold_folds"], purged_kfold.folds)
    _write_json(
        paths["model_selection_json"],
        _build_model_selection_payload(
            spec=spec,
            model_factor_result=model_factor_result,
        ),
    )
    _write_json(
        paths["model_definition_json"],
        {
            "schema_version": "1.0.0",
            "artifact_type": "alpha_lab_model_definition",
            "case_name": spec.name,
            "factor_name": spec.factor_name,
            "model_family": spec.model.family,
            "model_params": _to_jsonable(spec.model.params),
            "resolved_model_params": _build_resolved_model_params_payload(spec=spec),
            "feature_availability": _to_jsonable(
                spec_to_dict(spec).get("feature_availability", {})
            ),
            "model_selection": _to_jsonable(spec_to_dict(spec).get("model_selection", {})),
            "feature_importance": _to_jsonable(
                spec_to_dict(spec).get("feature_importance", {})
            ),
            "model_selection_outcome": _build_model_selection_outcome_payload(
                spec=spec,
                model_factor_result=model_factor_result,
            ),
            "training": _to_jsonable(spec_to_dict(spec).get("training", {})),
            "label_temporal_contract": _build_label_temporal_contract_payload(
                spec=spec,
                model_factor_result=model_factor_result,
            ),
            "feature_preprocess": _to_jsonable(feature_preprocess_payload),
            "diagnostics": _to_jsonable(model_factor_result.model_diagnostics),
            "source_artifacts": {
                "training_log_path": str(paths["training_log"]),
                "feature_importance_path": str(paths["feature_importance"]),
                "model_selection_path": str(paths["model_selection_json"]),
                "metrics_path": str(paths["metrics"]),
            },
            "artifact_missing_value_notes": _to_jsonable(missing_value_notes),
        },
    )
    _write_json(paths["feature_manifest_json"], feature_manifest_payload)

    manifest = {
        "schema_version": "1.0.0",
        "artifact_type": "real_case_single_factor_bundle",
        "workflow": "real_case_model_factor",
        "run_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "case_name": spec.name,
        "spec_path": str(Path(spec_path).resolve()) if spec_path is not None else None,
        "inputs": {
            "prices_path": spec.prices_path,
            "features_path": spec.features_path,
            "factor_name": spec.factor_name,
            "feature_columns": list(spec.feature_columns),
            "model_family": spec.model.family,
            "feature_availability_mode": spec.feature_availability.mode,
            "model_selection_enabled": bool(spec.model_selection.enabled),
            "feature_preprocess": _to_jsonable(feature_preprocess_payload),
            "feature_importance": _to_jsonable(spec_to_dict(spec).get("feature_importance", {})),
            "universe_path": spec.universe.path,
            "neutralization_exposures_path": spec.neutralization.exposures_path,
        },
        "spec": _to_jsonable(spec_to_dict(spec)),
        "outputs": {name: str(path) for name, path in paths.items()},
        "required_bundle_files": list(REQUIRED_BUNDLE_FILES),
        "integrity_summary": report.summary.to_dict(),
        "evaluation_standard": {
            "profile_name": evaluation_config.profile_name,
            "snapshot": research_evaluation_audit_snapshot(evaluation_config),
        },
        "vault_export": {
            "enabled": False,
            "mode": "skip",
            "target_paths": [],
            "status": "skipped",
            "error": None,
        },
        "artifact_missing_value_notes": {
            "csv_na_rep": "N/A",
            "details": _to_jsonable(missing_value_notes),
        },
    }
    _write_json(paths["run_manifest"], manifest)

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
            "Vault export failed for model-factor case %s: %s",
            spec.name,
            vault_result.error,
        )
    if vault_result.success and vault_result.target_paths:
        _sync_exported_manifest_copies(paths["run_manifest"], vault_result.target_paths)

    return paths


def _build_model_selection_payload(
    *,
    spec: ModelFactorCaseSpec,
    model_factor_result: ModelFactorBuildResult,
) -> dict[str, object]:
    model_selection_df = model_factor_result.model_selection_df.copy()
    diagnostics_selection = model_factor_result.model_diagnostics.get("model_selection", {})
    if isinstance(diagnostics_selection, Mapping):
        diagnostics_selection_payload: Mapping[str, object] = diagnostics_selection
    else:
        diagnostics_selection_payload = {}
    if model_selection_df.empty:
        status = "disabled" if not spec.model_selection.enabled else "not_available"
        rows: list[dict[str, object]] = []
    else:
        status = "ok"
        rows = _to_jsonable(model_selection_df.to_dict(orient="records"))  # type: ignore[assignment]
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_model_selection",
        "case_name": spec.name,
        "factor_name": spec.factor_name,
        "status": status,
        "configured_model": _to_jsonable(spec_to_dict(spec).get("model", {})),
        "configured_model_selection": _to_jsonable(spec_to_dict(spec).get("model_selection", {})),
        "summary": _to_jsonable(dict(diagnostics_selection_payload)),
        "selection_rows": rows,
    }


def _candidate_id_for_payload(idx: int, candidate: ModelSpec) -> str:
    return f"candidate_{idx + 1}_{candidate.family}"


def _selection_candidates_for_payload(spec: ModelFactorCaseSpec) -> tuple[ModelSpec, ...]:
    if spec.model_selection.enabled and spec.model_selection.candidates:
        return spec.model_selection.candidates
    return (spec.model,)


def _build_resolved_model_params_payload(*, spec: ModelFactorCaseSpec) -> dict[str, object]:
    candidates = _selection_candidates_for_payload(spec)
    candidate_rows: list[dict[str, object]] = []
    for idx, candidate in enumerate(candidates):
        candidate_rows.append(
            {
                "candidate_id": _candidate_id_for_payload(idx, candidate),
                "family": candidate.family,
                "params": _to_jsonable(resolve_model_spec_params(candidate)),
            }
        )
    return {
        "configured_model": {
            "family": spec.model.family,
            "params": _to_jsonable(resolve_model_spec_params(spec.model)),
        },
        "selection_candidates": candidate_rows,
        "selection_candidates_count": int(len(candidate_rows)),
    }


def _build_label_temporal_contract_payload(
    *,
    spec: ModelFactorCaseSpec,
    model_factor_result: ModelFactorBuildResult,
) -> dict[str, object]:
    diagnostics = _as_object(model_factor_result.model_diagnostics)
    raw_gap = diagnostics.get("purged_train_gap_dates")
    if isinstance(raw_gap, (int, float)) and math.isfinite(float(raw_gap)):
        purge_gap = int(raw_gap)
    else:
        purge_gap = max(int(spec.target.horizon) - 1, 0)
    raw_clipped_rows = diagnostics.get("label_winsor_clipped_rows")
    if isinstance(raw_clipped_rows, (int, float)) and math.isfinite(float(raw_clipped_rows)):
        clipped_rows = int(raw_clipped_rows)
    else:
        clipped_rows = 0
    return {
        "target_kind": spec.target.kind,
        "target_horizon": int(spec.target.horizon),
        "purged_train_gap_dates": purge_gap,
        "walk_forward_purged_training": True,
        "label_winsorize_zscore": _finite_if_number(diagnostics.get("label_winsorize_zscore")),
        "label_winsor_clipped_rows": clipped_rows,
    }


def _build_model_selection_outcome_payload(
    *,
    spec: ModelFactorCaseSpec,
    model_factor_result: ModelFactorBuildResult,
) -> dict[str, object]:
    training_log = model_factor_result.training_log_df
    selection_rows = model_factor_result.model_selection_df
    diagnostics_selection = _as_object(
        _as_object(model_factor_result.model_diagnostics).get("model_selection")
    )
    raw_selection_events = diagnostics_selection.get("n_selection_events")
    if isinstance(raw_selection_events, (int, float)) and math.isfinite(
        float(raw_selection_events)
    ):
        n_selection_events = int(raw_selection_events)
    else:
        n_selection_events = 0

    status_series = (
        training_log["status"].astype(str).str.strip().str.lower()
        if "status" in training_log.columns
        else pd.Series(dtype=str)
    )
    n_fit_events = int((status_series == "fit_scored").sum()) if not status_series.empty else 0
    n_reuse_events = int((status_series == "reused_scored").sum()) if not status_series.empty else 0
    n_skip_events = int((status_series == "skipped").sum()) if not status_series.empty else 0

    selected_candidates = selection_rows
    if "selected" in selected_candidates.columns:
        selected_candidates = selected_candidates[selected_candidates["selected"] == True].copy()  # noqa: E712
    if "score_date" in selected_candidates.columns:
        selected_candidates = selected_candidates.sort_values("score_date", kind="mergesort")

    latest_candidate_id: str | None = None
    latest_candidate_score: float | None = None
    latest_selection_score_date: str | None = None
    if not selected_candidates.empty:
        latest_row = selected_candidates.iloc[-1]
        latest_candidate_id = _normalized_text_or_none(latest_row.get("candidate_id"))
        latest_candidate_score = _finite_if_number(latest_row.get("selection_score"))
        latest_selection_score_date = _normalized_text_or_none(latest_row.get("score_date"))

    if latest_candidate_id is None and "selected_candidate_id" in training_log.columns:
        candidate_series = training_log["selected_candidate_id"].map(_normalized_text_or_none)
        candidate_series = candidate_series[candidate_series.notna()]
        if not candidate_series.empty:
            latest_candidate_id = str(candidate_series.iloc[-1])

    if latest_candidate_score is None and "selected_candidate_score" in training_log.columns:
        score_series = training_log["selected_candidate_score"].map(_finite_if_number)
        score_series = score_series[score_series.notna()]
        if not score_series.empty:
            latest_candidate_score = float(score_series.iloc[-1])

    latest_candidate_turnover: float | None = None
    if "selected_candidate_turnover" in training_log.columns:
        turnover_frame = training_log
        if latest_candidate_id is not None and "selected_candidate_id" in turnover_frame.columns:
            ids = turnover_frame["selected_candidate_id"].map(_normalized_text_or_none)
            turnover_frame = turnover_frame[ids == latest_candidate_id]
        turnover_series = turnover_frame["selected_candidate_turnover"].map(_finite_if_number)
        turnover_series = turnover_series[turnover_series.notna()]
        if not turnover_series.empty:
            latest_candidate_turnover = float(turnover_series.iloc[-1])

    candidate_lookup: dict[str, ModelSpec] = {
        _candidate_id_for_payload(idx, candidate): candidate
        for idx, candidate in enumerate(_selection_candidates_for_payload(spec))
    }
    latest_candidate = (
        candidate_lookup.get(latest_candidate_id)
        if latest_candidate_id is not None
        else None
    )

    return {
        "enabled": bool(spec.model_selection.enabled),
        "selection_metric": spec.model_selection.metric if spec.model_selection.enabled else None,
        "n_selection_events": n_selection_events,
        "n_fit_events": n_fit_events,
        "n_reuse_events": n_reuse_events,
        "n_skip_events": n_skip_events,
        "latest_selected_candidate_id": latest_candidate_id,
        "latest_selected_candidate_family": (
            latest_candidate.family if latest_candidate is not None else None
        ),
        "latest_selected_candidate_params": (
            _to_jsonable(resolve_model_spec_params(latest_candidate))
            if latest_candidate is not None
            else None
        ),
        "latest_selected_candidate_score": latest_candidate_score,
        "latest_selected_candidate_turnover": latest_candidate_turnover,
        "latest_selection_score_date": latest_selection_score_date,
    }


def _build_portfolio_recipe_payload(
    *,
    spec: ModelFactorCaseSpec,
    metrics_for_payload: Mapping[str, object],
    portfolio_validation_payload: Mapping[str, object],
    output_paths: ModelFactorArtifactPaths,
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
            "model_family": spec.model.family,
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
    spec: ModelFactorCaseSpec,
    metrics_for_payload: Mapping[str, object],
    group_returns_df: pd.DataFrame,
    output_paths: ModelFactorArtifactPaths,
) -> dict[str, object]:
    summary, fallback_fields = build_backtest_summary_payload(
        group_returns_df=group_returns_df,
        rebalance_frequency=spec.rebalance_frequency,
        metrics_for_payload=metrics_for_payload,
        label_horizon=int(spec.target.horizon),
    )
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_backtest_result",
        "case_name": spec.name,
        "package_type": "single_factor",
        "rebalance_frequency": spec.rebalance_frequency,
        "target_horizon": int(spec.target.horizon),
        "summary": summary,
        "source_artifacts": {
            "group_returns_path": str(output_paths["group_returns"]),
            "turnover_path": str(output_paths["turnover"]),
            "metrics_path": str(output_paths["metrics"]),
        },
        "fallback_derived_fields": fallback_fields,
    }


def _feature_preprocess_payload_for_artifacts(
    *,
    spec: ModelFactorCaseSpec,
    spec_path: str | Path | None,
) -> dict[str, object]:
    payload = dict(_as_object(spec_to_dict(spec).get("feature_preprocess")))
    payload["cross_sectional_transform_default_applied"] = (
        _cross_sectional_transform_default_applied(spec_path)
    )
    return payload


def _cross_sectional_transform_default_applied(spec_path: str | Path | None) -> bool:
    if spec_path is None:
        return False
    path = Path(spec_path)
    if not path.exists() or not path.is_file():
        return False
    try:
        parsed = parse_mapping_payload(path.read_text(encoding="utf-8"), suffix=path.suffix.lower())
    except Exception:  # noqa: BLE001
        return False
    raw_feature_preprocess = parsed.get("feature_preprocess")
    if not isinstance(raw_feature_preprocess, Mapping):
        return True
    return "cross_sectional_transform" not in raw_feature_preprocess


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep="N/A")
    if path.exists() and path.stat().st_size > 0:
        return
    # Defensive fallback: keep file-level content non-empty even for pathological empty inputs.
    path.write_text("status,reason\nnot_available,empty_dataframe\n", encoding="utf-8")


def _prepare_training_log_for_export(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    exported = frame.copy()
    notes: list[dict[str, object]] = []
    fill_specs: tuple[tuple[str, str], ...] = (
        ("skip_reason", "not_skipped"),
        ("model_version", "N/A"),
        ("trained_date_start", "N/A"),
        ("trained_date_end", "N/A"),
        ("scale_mode", "N/A"),
    )
    missing_total = 0
    impacted_fields: list[str] = []
    for field, replacement in fill_specs:
        if field not in exported.columns:
            continue
        values = exported[field]
        missing_mask = values.isna() | values.astype(str).str.strip().eq("")
        missing_count = int(missing_mask.sum())
        if missing_count <= 0:
            continue
        missing_total += missing_count
        impacted_fields.append(field)
        exported[field] = values.astype(object)
        exported.loc[missing_mask, field] = replacement
    if missing_total > 0:
        notes.append(
            {
                "artifact": "training_log.csv",
                "fields": impacted_fields,
                "missing_value_count": missing_total,
                "reason": (
                    "训练日志存在按行条件不适用字段（例如非 skipped 行的 skip_reason、"
                    "skipped 行的模型版本/训练窗口）；导出时统一写为 N/A 或 not_skipped。"
                ),
            }
        )
    return exported, notes


def _prepare_feature_importance_for_export(
    frame: pd.DataFrame,
    *,
    model_family: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    exported = frame.copy()
    notes: list[dict[str, object]] = []
    required_defaults: dict[str, object] = {
        "feature": "N/A",
        "mean_abs_importance": "N/A",
        "latest_importance": "N/A",
        "importance_source": "unknown",
        "n_model_versions": 0,
    }
    for field, default in required_defaults.items():
        if field not in exported.columns:
            exported[field] = default

    if exported.empty:
        exported = pd.DataFrame(
            [
                {
                    "feature": "N/A",
                    "mean_abs_importance": "N/A",
                    "latest_importance": "N/A",
                    "importance_source": "not_available",
                    "n_model_versions": 0,
                    "missing_value_reason": (
                        "feature_importance 数据为空：训练阶段未产出可用重要性统计。"
                    ),
                }
            ]
        )
        notes.append(
            {
                "artifact": "feature_importance.csv",
                "fields": ["mean_abs_importance", "latest_importance"],
                "missing_value_count": 2,
                "reason": "训练阶段未产出可用重要性统计，已写入 N/A 与缺失原因。",
            }
        )
        return exported, notes

    missing_total = 0
    for field in ("mean_abs_importance", "latest_importance"):
        values = exported[field]
        missing_mask = values.isna() | values.astype(str).str.strip().eq("")
        missing_count = int(missing_mask.sum())
        if missing_count <= 0:
            continue
        missing_total += missing_count
        exported[field] = values.astype(object)
        exported.loc[missing_mask, field] = "N/A"

    source_values = exported["importance_source"].astype(str).str.strip()
    exported["importance_source"] = source_values.where(source_values.ne(""), "unknown")
    exported["missing_value_reason"] = exported.apply(
        lambda row: _feature_importance_missing_reason(row, model_family=model_family),
        axis=1,
    )

    if missing_total > 0:
        notes.append(
            {
                "artifact": "feature_importance.csv",
                "fields": ["mean_abs_importance", "latest_importance"],
                "missing_value_count": missing_total,
                "reason": (
                    "部分特征重要性字段缺失，已写入 N/A；"
                    "逐行原因见 missing_value_reason（例如模型族不支持 importance 提取）。"
                ),
            }
        )
    return exported, notes


def _feature_importance_missing_reason(row: pd.Series, *, model_family: str) -> str:
    mean_value = str(row.get("mean_abs_importance", "")).strip()
    latest_value = str(row.get("latest_importance", "")).strip()
    has_missing = mean_value in {"", "N/A"} or latest_value in {"", "N/A"}
    if not has_missing:
        return "无缺失"
    source = str(row.get("importance_source", "")).strip().lower()
    raw_versions = row.get("n_model_versions")
    versions = 0
    if isinstance(raw_versions, (int, float)) and math.isfinite(float(raw_versions)):
        versions = int(raw_versions)
    elif isinstance(raw_versions, str):
        text = raw_versions.strip()
        if text.isdigit():
            versions = int(text)
    if source == "unsupported":
        return (
            f"模型族 `{model_family}` 当前不暴露可解释特征重要性接口，"
            "因此 mean/latest importance 记为 N/A。"
        )
    if source == "disabled":
        return "feature_importance.mode='disabled'，本次运行按配置跳过重要性计算。"
    if versions <= 0:
        return "训练阶段未生成可用模型版本，无法汇总 mean/latest importance。"
    return "重要性统计不可用（存在非有限值或中间结果缺失）。"


def write_diagnostics_artifact(
    *,
    output_dir: str | Path,
    diagnostics_payload: Mapping[str, object],
    pretty: bool = False,
) -> Path:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = out_dir / "diagnostics.json"
    _write_json(diagnostics_path, diagnostics_payload, pretty=pretty)
    return diagnostics_path


def _write_json(path: Path, payload: Mapping[str, object], *, pretty: bool = True) -> None:
    jsonable_payload = _to_jsonable(payload)
    if not isinstance(jsonable_payload, Mapping):
        raise ValueError(f"{path} JSON payload root must be an object")
    validate_level12_artifact_payload(
        jsonable_payload,
        artifact_name=path.name,
        source=path,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file_obj:
        if pretty:
            json.dump(jsonable_payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        else:
            json.dump(
                jsonable_payload,
                file_obj,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=False,
            )
        file_obj.write("\n")


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


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


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


def _normalized_text_or_none(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan" or text in {"N/A", "None", "null"}:
        return None
    return text


def _as_object(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _sync_exported_manifest_copies(
    local_manifest_path: Path,
    target_paths: tuple[str, ...],
) -> None:
    for raw in target_paths:
        target = Path(raw)
        if not target.name.endswith("run_manifest.json"):
            continue
        try:
            shutil.copy2(local_manifest_path, target)
        except OSError as exc:
            logger.warning(
                "Failed to sync model-factor vault manifest copy %s: %s",
                target,
                exc,
            )
