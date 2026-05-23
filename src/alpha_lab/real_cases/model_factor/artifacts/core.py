from __future__ import annotations

import datetime
from collections.abc import Mapping
from pathlib import Path

from alpha_lab.key_metrics_contracts import project_level12_transition_summary
from alpha_lab.model_factor import ModelFactorBuildResult
from alpha_lab.real_cases.artifact_enrichment import (
    build_group_nav_table,
)
from alpha_lab.real_cases.single_factor.evaluate import SingleFactorEvaluationResult
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

from ..spec import ModelFactorCaseSpec, dump_spec_yaml, spec_to_dict
from ..templates import render_experiment_card_markdown, render_summary_markdown

# Cross-module imports (auto-added)
from ._utils import (
    ModelFactorArtifactPaths,
    _as_object,
    _finite_or_none,
    _split_contract_payload,
    _sync_exported_manifest_copies,
    _text_or_none,
    _to_jsonable,
    _write_json,
    logger,
)
from .backtest_recipe import _build_backtest_result_payload, _build_portfolio_recipe_payload
from .diagnostics import write_diagnostics_artifact
from .feature_export import (
    _feature_preprocess_payload_for_artifacts,
    _prepare_feature_importance_for_export,
    _prepare_feature_importance_ledger_for_export,
    _prepare_training_log_for_export,
    _write_csv,
)
from .model_selection import (
    _build_label_temporal_contract_payload,
    _build_model_selection_outcome_payload,
    _build_model_selection_payload,
    _build_resolved_model_params_payload,
)

REQUIRED_BUNDLE_FILES: tuple[str, ...] = (
    "run_manifest.json",
    "metrics.json",
    "factor_definition.json",
    "signal_validation.json",
    "portfolio_recipe.json",
    "backtest_result.json",
    "purged_kfold_summary.json",
    "purged_kfold_folds.csv",
    "purged_kfold_fold_daily.csv",
    "model_selection.json",
    "model_definition.json",
    "feature_manifest.json",
    "diagnostics.json",
    "research_tearsheet.json",
    "research_tearsheet.pdf",
    "training_log.csv",
    "training_metrics.csv",
    "feature_importance.csv",
    "feature_importance_ledger.csv",
    "feature_oos_ic.csv",
    "ic_timeseries.csv",
    "ic_decay.csv",
    "rolling_stability.csv",
    "group_returns.csv",
    "group_nav.csv",
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
    draft_model_source: Mapping[str, object] | None = None,
    defer_vault_export: bool = False,
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
        "purged_kfold_fold_daily": out_dir / "purged_kfold_fold_daily.csv",
        "model_selection_json": out_dir / "model_selection.json",
        "model_definition_json": out_dir / "model_definition.json",
        "feature_manifest_json": out_dir / "feature_manifest.json",
        "diagnostics": out_dir / "diagnostics.json",
        "research_tearsheet": out_dir / "research_tearsheet.json",
        "research_tearsheet_pdf": out_dir / "research_tearsheet.pdf",
        "training_log": out_dir / "training_log.csv",
        "training_metrics": out_dir / "training_metrics.csv",
        "feature_importance": out_dir / "feature_importance.csv",
        "feature_importance_ledger": out_dir / "feature_importance_ledger.csv",
        "feature_oos_ic": out_dir / "feature_oos_ic.csv",
        "ic_timeseries": out_dir / "ic_timeseries.csv",
        "ic_decay": out_dir / "ic_decay.csv",
        "rolling_stability": out_dir / "rolling_stability.csv",
        "group_returns": out_dir / "group_returns.csv",
        "group_nav": out_dir / "group_nav.csv",
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
    _write_csv(
        paths["group_nav"],
        build_group_nav_table(
            evaluation_result.group_returns,
            rebalance_frequency=spec.rebalance_frequency,
            label_horizon=int(spec.target.horizon),
        ),
    )
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
    _write_csv(paths["training_metrics"], model_factor_result.training_metrics_df)
    _write_csv(paths["feature_importance"], feature_importance_df)
    _write_csv(
        paths["feature_importance_ledger"],
        _prepare_feature_importance_ledger_for_export(
            model_factor_result.feature_importance_ledger_df,
            spec=spec,
        ),
    )
    _write_csv(paths["feature_oos_ic"], model_factor_result.feature_oos_ic_df)
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
            "split_semantics": "model_training_prediction_holdout",
            "split_semantics_label": "Model-Lab: IS=训练样本，OOS=预测样本",
            "trained_model_versions": model_factor_result.model_diagnostics.get(
                "trained_model_versions"
            ),
            "model_top_features": model_factor_result.model_diagnostics.get("top_features"),
            "mean_train_rows": model_factor_result.model_diagnostics.get("mean_train_rows"),
            "mean_score_assets": model_factor_result.model_diagnostics.get("mean_score_assets"),
            "target_execution_price_mode": spec.target.execution_price_mode,
        }
    )
    retrain_every = int(spec.training.retrain_every_n_dates)
    retrain_warning_threshold = (
        evaluation_config.model_factor_overrides.min_retrain_every_n_dates
        if evaluation_config.profile_name == "exploratory_screening"
        else None
    )
    retrain_density_warning = retrain_warning_threshold is not None and retrain_every >= int(
        retrain_warning_threshold
    )
    metrics_for_payload["retrain_density_warning"] = bool(retrain_density_warning)
    metrics_for_payload["retrain_density_warning_reason"] = (
        "快速筛选模式使用较低重训密度，OOS IC 可能偏乐观；晋级前应在 default/stricter 模式复核。"
        if retrain_density_warning
        else None
    )
    metrics_for_payload["training_retrain_every_n_dates_effective"] = retrain_every
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
    coverage_frame = evaluation_result.coverage
    coverage_eval_frame = coverage_frame
    coverage_warmup_excluded_days = 0
    if not coverage_frame.empty and "coverage_eval_included" in coverage_frame.columns:
        coverage_eval_mask = coverage_frame["coverage_eval_included"].fillna(True).astype(bool)
        coverage_warmup_excluded_days = int((~coverage_eval_mask).sum())
        coverage_eval_frame = coverage_frame.loc[coverage_eval_mask].reset_index(drop=True)
        if coverage_eval_frame.empty:
            coverage_eval_frame = coverage_frame

    metrics_payload = {
        "metrics": _to_jsonable(metrics_for_payload),
        "coverage_by_date_summary": {
            "n_dates": int(coverage_eval_frame["date"].nunique())
            if not coverage_eval_frame.empty
            else 0,
            "mean_coverage": _finite_or_none(
                coverage_eval_frame["coverage"].mean()
                if not coverage_eval_frame.empty
                else float("nan")
            ),
            "min_coverage": _finite_or_none(
                coverage_eval_frame["coverage"].min()
                if not coverage_eval_frame.empty
                else float("nan")
            ),
            "coverage_warmup_excluded_days": coverage_warmup_excluded_days,
            "mean_coverage_raw": _finite_or_none(
                coverage_frame["coverage"].mean() if not coverage_frame.empty else float("nan")
            ),
            "min_coverage_raw": _finite_or_none(
                coverage_frame["coverage"].min() if not coverage_frame.empty else float("nan")
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
            "model_family": spec.model.family,
            "feature_count": len(spec.feature_columns),
            "universe_name": spec.universe.name,
            "target_kind": spec.target.kind,
            "target_horizon": spec.target.horizon,
            "target_execution_price_mode": spec.target.execution_price_mode,
            "split_contract": split_contract_payload,
            "split_semantics": metrics_for_payload["split_semantics"],
            "split_semantics_label": metrics_for_payload["split_semantics_label"],
        },
    )
    _write_json(paths["research_tearsheet"], tearsheet_payload)
    export_research_tearsheet_pdf(
        payload=tearsheet_payload,
        output_path=paths["research_tearsheet_pdf"],
    )
    purged_kfold = build_purged_kfold_diagnostics(
        experiment_result=evaluation_result.experiment_result,
        label_horizon=int(spec.target.horizon),
    )
    _write_json(paths["purged_kfold_summary"], purged_kfold.summary)
    _write_csv(paths["purged_kfold_folds"], purged_kfold.folds)
    _write_csv(paths["purged_kfold_fold_daily"], purged_kfold.fold_daily)
    archive_identity = (
        _text_or_none((draft_model_source or {}).get("archive_identity"))
        or _text_or_none(spec.archive_identity)
        or _text_or_none((draft_model_source or {}).get("name"))
        or spec.factor_name
    )
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
            "archive_identity": archive_identity,
            "model_family": spec.model.family,
            "model_params": _to_jsonable(spec.model.params),
            "resolved_model_params": _build_resolved_model_params_payload(spec=spec),
            "feature_availability": _to_jsonable(
                spec_to_dict(spec).get("feature_availability", {})
            ),
            "target": _to_jsonable(spec_to_dict(spec).get("target", {})),
            "model_selection": _to_jsonable(spec_to_dict(spec).get("model_selection", {})),
            "feature_importance": _to_jsonable(spec_to_dict(spec).get("feature_importance", {})),
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
                "training_metrics_path": str(paths["training_metrics"]),
                "feature_importance_path": str(paths["feature_importance"]),
                "feature_oos_ic_path": str(paths["feature_oos_ic"]),
                "model_selection_path": str(paths["model_selection_json"]),
                "metrics_path": str(paths["metrics"]),
            },
            "artifact_missing_value_notes": _to_jsonable(missing_value_notes),
            **(
                {"draft_model_source": dict(draft_model_source)}
                if draft_model_source is not None
                else {}
            ),
        },
    )
    feature_manifest_with_audit: dict[str, object] = dict(feature_manifest_payload)
    if draft_model_source is not None:
        feature_manifest_with_audit["draft_model_source"] = dict(draft_model_source)
    _write_json(paths["feature_manifest_json"], feature_manifest_with_audit)

    manifest_inputs: dict[str, object] = {
        "prices_path": spec.prices_path,
        "features_path": spec.features_path,
        "factor_name": spec.factor_name,
        "archive_identity": archive_identity,
        "feature_columns": list(spec.feature_columns),
        "model_family": spec.model.family,
        "feature_availability_mode": spec.feature_availability.mode,
        "model_selection_enabled": bool(spec.model_selection.enabled),
        "feature_preprocess": _to_jsonable(feature_preprocess_payload),
        "feature_importance": _to_jsonable(spec_to_dict(spec).get("feature_importance", {})),
        "universe_path": spec.universe.path,
        "neutralization_exposures_path": spec.neutralization.exposures_path,
    }
    if draft_model_source is not None:
        manifest_inputs["draft_model_source"] = dict(draft_model_source)
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "artifact_type": "real_case_single_factor_bundle",
        "workflow": "real_case_model_factor",
        "run_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "case_name": spec.name,
        "spec_path": str(Path(spec_path).resolve()) if spec_path is not None else None,
        "inputs": manifest_inputs,
        "spec": _to_jsonable(spec_to_dict(spec)),
        "outputs": {name: str(path) for name, path in paths.items()},
        "required_bundle_files": list(REQUIRED_BUNDLE_FILES),
        "integrity_summary": report.summary.to_dict(),
        "evaluation_standard": {
            "profile_name": evaluation_config.profile_name,
            "snapshot": research_evaluation_audit_snapshot(evaluation_config),
        },
        "split_contract": split_contract_payload,
        "research_tearsheet": {
            "status": "emitted",
            "json_path": str(paths["research_tearsheet"]),
            "pdf_path": str(paths["research_tearsheet_pdf"]),
            "split_contract": split_contract_payload,
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
    if draft_model_source is not None:
        manifest["draft_model_source"] = dict(draft_model_source)
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
            "Vault export failed for model-factor case %s: %s",
            spec.name,
            vault_result.error,
        )
    if vault_result.success and vault_result.target_paths:
        _sync_exported_manifest_copies(paths["run_manifest"], vault_result.target_paths)

    return paths
