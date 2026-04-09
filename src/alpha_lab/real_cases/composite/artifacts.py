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
from alpha_lab.real_cases.artifact_enrichment import (
    build_backtest_summary_payload,
    build_portfolio_recipe_controls,
)
from alpha_lab.reporting.level2_portfolio_validation import (
    build_level2_portfolio_validation_bundle,
    export_level2_portfolio_validation_bundle,
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

from .combine import CombineResult
from .evaluate import CompositeEvaluationResult
from .spec import CompositeCaseSpec, dump_spec_yaml, spec_to_dict
from .templates import render_experiment_card_markdown, render_summary_markdown

logger = logging.getLogger(__name__)

REQUIRED_BUNDLE_FILES: tuple[str, ...] = (
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
    "level2_portfolio_validation/portfolio_validation_summary.json",
    "level2_portfolio_validation/portfolio_validation_metrics.json",
    "level2_portfolio_validation/portfolio_validation_package.json",
    "level2_portfolio_validation/portfolio_validation_package.md",
)


class CompositeArtifactPaths(TypedDict):
    run_manifest: Path
    metrics: Path
    factor_definition_json: Path
    signal_validation_json: Path
    portfolio_recipe_json: Path
    backtest_result_json: Path
    ic_timeseries: Path
    rolling_stability: Path
    group_returns: Path
    turnover: Path
    exposures: Path
    composite_definition: Path
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
    spec: CompositeCaseSpec,
    combine_result: CombineResult,
    evaluation_result: CompositeEvaluationResult,
    evaluation_config: ResearchEvaluationConfig,
    integrity_report: IntegrityReport | None,
    output_dir: str | Path,
    spec_path: str | Path | None = None,
    vault_root: str | Path | None = None,
    vault_export_mode: str = "versioned",
) -> CompositeArtifactPaths:
    """Write standardized artifact bundle for one composite case run."""

    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    paths: CompositeArtifactPaths = {
        "run_manifest": out_dir / "run_manifest.json",
        "metrics": out_dir / "metrics.json",
        "factor_definition_json": out_dir / "factor_definition.json",
        "signal_validation_json": out_dir / "signal_validation.json",
        "portfolio_recipe_json": out_dir / "portfolio_recipe.json",
        "backtest_result_json": out_dir / "backtest_result.json",
        "ic_timeseries": out_dir / "ic_timeseries.csv",
        "rolling_stability": out_dir / "rolling_stability.csv",
        "group_returns": out_dir / "group_returns.csv",
        "turnover": out_dir / "turnover.csv",
        "exposures": out_dir / "exposures.csv",
        "composite_definition": out_dir / "composite_definition.yaml",
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

    evaluation_result.ic_timeseries.to_csv(paths["ic_timeseries"], index=False)
    evaluation_result.rolling_stability.to_csv(paths["rolling_stability"], index=False)
    evaluation_result.group_returns.to_csv(paths["group_returns"], index=False)
    evaluation_result.turnover.to_csv(paths["turnover"], index=False)
    evaluation_result.exposure_summary.to_csv(paths["exposures"], index=False)

    spec_yaml = dump_spec_yaml(spec)
    paths["composite_definition"].write_text(spec_yaml, encoding="utf-8")

    summary_md = render_summary_markdown(
        spec=spec,
        metrics=evaluation_result.metrics,
        output_dir=out_dir,
    )
    paths["summary"].write_text(summary_md, encoding="utf-8")

    card_md = render_experiment_card_markdown(
        spec=spec,
        metrics=evaluation_result.metrics,
        result=evaluation_result.experiment_result,
    )
    paths["experiment_card"].write_text(card_md, encoding="utf-8")

    report = integrity_report or build_integrity_report(
        (),
        context={
            "pipeline": "real_case_composite",
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
            "package_type": "composite",
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
    metrics_for_payload["portfolio_validation_recommendation"] = (
        portfolio_validation_summary.get("recommendation")
    )
    metrics_for_payload["portfolio_validation_remains_credible"] = (
        portfolio_validation_summary.get("remains_credible_at_portfolio_level")
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
        dict(robustness_summary_raw)
        if isinstance(robustness_summary_raw, Mapping)
        else {}
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
    benchmark_eval = (
        dict(benchmark_eval_raw)
        if isinstance(benchmark_eval_raw, Mapping)
        else {}
    )
    metrics_for_payload["portfolio_validation_benchmark_relative_status"] = benchmark_eval.get(
        "status"
    )
    metrics_for_payload["portfolio_validation_benchmark_relative_assessment"] = (
        benchmark_eval.get("assessment")
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
    metrics_for_payload["level12_transition_reasons"] = level12_transition[
        "key_transition_reasons"
    ]
    metrics_for_payload["level12_transition_confirmation_note"] = level12_transition[
        "confirmation_vs_degradation_note"
    ]

    metrics_payload = {
        "metrics": _to_jsonable(metrics_for_payload),
        "component_summary": _to_jsonable(
            combine_result.component_summary.to_dict(orient="records")
        ),
        "coverage_by_date_summary": {
            "n_dates": int(combine_result.coverage_by_date["date"].nunique())
            if not combine_result.coverage_by_date.empty
            else 0,
            "mean_coverage": _finite_or_none(
                combine_result.coverage_by_date["composite_coverage"].mean()
                if not combine_result.coverage_by_date.empty
                else float("nan")
            ),
            "min_coverage": _finite_or_none(
                combine_result.coverage_by_date["composite_coverage"].min()
                if not combine_result.coverage_by_date.empty
                else float("nan")
            ),
            "mean_composite_coverage": _finite_or_none(
                combine_result.coverage_by_date["composite_coverage"].mean()
                if not combine_result.coverage_by_date.empty
                else float("nan")
            ),
            "min_composite_coverage": _finite_or_none(
                combine_result.coverage_by_date["composite_coverage"].min()
                if not combine_result.coverage_by_date.empty
                else float("nan")
            ),
        },
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

    factor_definition_payload = _build_factor_definition_payload(
        spec=spec,
        output_paths=paths,
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

    manifest = {
        "schema_version": "1.0.0",
        "artifact_type": "real_case_composite_bundle",
        "run_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "case_name": spec.name,
        "spec_path": str(Path(spec_path).resolve()) if spec_path is not None else None,
        "inputs": {
            "prices_path": spec.prices_path,
            "component_paths": [component.path for component in spec.components],
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
    }

    # Write once so manifest_path exists for optional vault sync.
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
            "Vault export failed for case %s: %s",
            spec.name,
            vault_result.error,
        )
    if vault_result.success and vault_result.target_paths:
        _sync_exported_manifest_copies(paths["run_manifest"], vault_result.target_paths)

    return paths


def _build_factor_definition_payload(
    *,
    spec: CompositeCaseSpec,
    output_paths: CompositeArtifactPaths,
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_factor_definition",
        "case_name": spec.name,
        "package_type": "composite",
        "factor_name": spec.name,
        "spec": _to_jsonable(spec_to_dict(spec)),
        "source_artifacts": {
            "composite_definition_yaml_path": str(output_paths["composite_definition"]),
            "run_manifest_path": str(output_paths["run_manifest"]),
        },
        "fallback_derived_fields": [],
    }


def _build_signal_validation_payload(
    *,
    spec: CompositeCaseSpec,
    metrics_payload: Mapping[str, object],
    output_paths: CompositeArtifactPaths,
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_signal_validation",
        "case_name": spec.name,
        "package_type": "composite",
        "metrics": _to_jsonable(_as_object(metrics_payload.get("metrics"))),
        "coverage_by_date_summary": _to_jsonable(
            _as_object(metrics_payload.get("coverage_by_date_summary"))
        ),
        "neutralization_summary": [],
        "source_artifacts": {
            "metrics_path": str(output_paths["metrics"]),
            "ic_timeseries_path": str(output_paths["ic_timeseries"]),
            "rolling_stability_path": str(output_paths["rolling_stability"]),
            "group_returns_path": str(output_paths["group_returns"]),
            "exposures_path": str(output_paths["exposures"]),
        },
        "fallback_derived_fields": [],
    }


def _build_portfolio_recipe_payload(
    *,
    spec: CompositeCaseSpec,
    metrics_for_payload: Mapping[str, object],
    portfolio_validation_payload: Mapping[str, object],
    output_paths: CompositeArtifactPaths,
) -> dict[str, object]:
    controls = build_portfolio_recipe_controls(
        metrics_for_payload=metrics_for_payload,
        portfolio_validation_payload=portfolio_validation_payload,
    )
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_portfolio_recipe",
        "case_name": spec.name,
        "package_type": "composite",
        "recipe_context": {
            "factor_name": spec.name,
            "rebalance_frequency": spec.rebalance_frequency,
            "universe_name": spec.universe.name,
            "target_horizon": spec.target.horizon,
            "neutralization_enabled": bool(spec.neutralization.enabled),
            "n_components": len(spec.components),
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
    spec: CompositeCaseSpec,
    metrics_for_payload: Mapping[str, object],
    group_returns_df: pd.DataFrame,
    output_paths: CompositeArtifactPaths,
) -> dict[str, object]:
    summary, fallback_fields = build_backtest_summary_payload(
        group_returns_df=group_returns_df,
        rebalance_frequency=spec.rebalance_frequency,
        metrics_for_payload=metrics_for_payload,
    )
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_backtest_result",
        "case_name": spec.name,
        "package_type": "composite",
        "rebalance_frequency": spec.rebalance_frequency,
        "summary": summary,
        "source_artifacts": {
            "group_returns_path": str(output_paths["group_returns"]),
            "turnover_path": str(output_paths["turnover"]),
            "metrics_path": str(output_paths["metrics"]),
        },
        "fallback_derived_fields": fallback_fields,
    }


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    jsonable_payload = _to_jsonable(payload)
    if not isinstance(jsonable_payload, Mapping):
        raise ValueError(f"{path} JSON payload root must be an object")
    validate_level12_artifact_payload(
        jsonable_payload,
        artifact_name=path.name,
        source=path,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(jsonable_payload, f, ensure_ascii=False, indent=2, sort_keys=True)
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


def _as_object(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


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
                "Failed to sync vault manifest copy %s: %s",
                target,
                exc,
            )
