from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from alpha_lab.artifact_contracts import validate_level12_artifact_payload
from alpha_lab.reporting._shared import resolve_artifact_path
from alpha_lab.reporting.display_helpers import as_object_dict
from alpha_lab.reporting.renderers.research_dashboard_schema import (
    CandidateRecipeGenerationResult,
    FactorSetConstructionResult,
    NextStepRecommendationResult,
    ResearchDashboardData,
    WinnerSelectionPolicy,
    WinnerSelectionResult,
)
from alpha_lab.reporting.research_artifact_manifest import (
    ARTIFACT_LOAD_DIAGNOSTICS_ARTIFACT_TYPE,
    CANDIDATE_RECIPE_GENERATION_ARTIFACT_TYPE,
    FACTOR_SET_RESULT_ARTIFACT_TYPE,
    NEXT_STEP_RECOMMENDATIONS_ARTIFACT_TYPE,
    RESEARCH_ARTIFACT_MANIFEST_CONTEXT_KEY,
    WINNER_SELECTION_ARTIFACT_TYPE,
    WORKFLOW_CLOSURE_ARTIFACT_FILENAMES,
    WORKFLOW_CLOSURE_SCHEMA_VERSION,
    build_research_artifact_manifest_payload,
    research_artifact_manifest_path,
)


def workflow_closure_artifact_paths(base_dir: Path) -> dict[str, Path]:
    return {
        key: (base_dir / filename).resolve()
        for key, filename in WORKFLOW_CLOSURE_ARTIFACT_FILENAMES.items()
    }


def write_validated_json_artifact(
    path: Path,
    payload: dict[str, object],
    *,
    artifact_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_level12_artifact_payload(
        payload,
        artifact_name=artifact_name,
        source=path,
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_factor_set_result_artifact_payload(
    result: FactorSetConstructionResult,
    *,
    generated_at_utc: str,
    comparison_json_path: Path,
    default_profile: str,
    source_artifacts: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": WORKFLOW_CLOSURE_SCHEMA_VERSION,
        "artifact_type": FACTOR_SET_RESULT_ARTIFACT_TYPE,
        "generated_at_utc": generated_at_utc,
        "comparison_json_path": str(comparison_json_path),
        "default_profile": default_profile,
        "source_artifacts": dict(source_artifacts),
        "policy": {
            "policy_id": result.config.policy_id,
            "formula_text": result.config.formula_text,
            "config": asdict(result.config),
        },
        "factor_sets": [
            {
                "factor_set_id": item.factor_set_id,
                "label_zh": item.label_zh,
                "factor_ids": list(item.factor_ids),
                "factor_names": list(item.factor_names),
                "source_shortlist_entries": list(item.source_shortlist_entries),
                "construction_rule": item.construction_rule,
                "status": item.status,
                "rationale": list(item.rationale),
                "rationale_zh": list(item.rationale_zh),
                "warnings": list(item.warnings),
                "score_summary": asdict(item.score_summary),
            }
            for item in result.factor_sets
        ],
        "selected_factor_set_ids": list(result.selected_factor_set_ids),
        "recommendation_summary": list(result.recommendation_summary),
    }


def build_candidate_recipe_generation_artifact_payload(
    result: CandidateRecipeGenerationResult,
    *,
    generated_at_utc: str,
    comparison_json_path: Path,
    default_profile: str,
    source_artifacts: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": WORKFLOW_CLOSURE_SCHEMA_VERSION,
        "artifact_type": CANDIDATE_RECIPE_GENERATION_ARTIFACT_TYPE,
        "generated_at_utc": generated_at_utc,
        "comparison_json_path": str(comparison_json_path),
        "default_profile": default_profile,
        "source_artifacts": dict(source_artifacts),
        "policy": {
            "policy_id": result.config.policy_id,
            "formula_text": result.config.formula_text,
            "config": asdict(result.config),
        },
        "generated_recipes": [
            {
                "recipe_id": item.recipe_id,
                "recipe_name": item.recipe_name,
                "source_factor_set_id": item.source_factor_set_id,
                "source_factor_ids": list(item.source_factor_ids),
                "construction_variant": item.construction_variant,
                "weighting_scheme": item.weighting_scheme,
                "neutralization_mode": item.neutralization_mode,
                "turnover_penalty_mode": item.turnover_penalty_mode,
                "benchmark_mode": item.benchmark_mode,
                "rationale": list(item.rationale),
                "assumptions": list(item.assumptions),
                "warnings": list(item.warnings),
            }
            for item in result.generated_recipes
        ],
        "recommendation_summary": list(result.recommendation_summary),
    }


def build_winner_selection_artifact_payload(
    result: WinnerSelectionResult,
    *,
    policy: WinnerSelectionPolicy,
    winner_policy_formula_text: str,
    generated_at_utc: str,
    comparison_json_path: Path,
    default_profile: str,
    source_artifacts: Mapping[str, str],
) -> dict[str, object]:
    missing_data_caveats = [
        f"{recipe_id}: composite score unavailable (missing metrics/components)"
        for recipe_id, score in result.score_table
        if score is None
    ]
    return {
        "schema_version": WORKFLOW_CLOSURE_SCHEMA_VERSION,
        "artifact_type": WINNER_SELECTION_ARTIFACT_TYPE,
        "generated_at_utc": generated_at_utc,
        "comparison_json_path": str(comparison_json_path),
        "default_profile": default_profile,
        "source_artifacts": dict(source_artifacts),
        "decision_policy": {
            "decision_policy_id": result.decision_policy_id or policy.decision_policy_id,
            "policy_formula_text": result.policy_formula_text or winner_policy_formula_text,
            "component_weights": list(policy.component_weights),
            "min_sharpe_for_winner": policy.min_sharpe_for_winner,
            "min_post_cost_return_for_winner": policy.min_post_cost_return_for_winner,
            "max_drawdown_floor": policy.max_drawdown_floor,
            "challenger_count": policy.challenger_count,
            "watchlist_score_min": policy.watchlist_score_min,
            "reject_score_max": policy.reject_score_max,
        },
        "winner_recipe_id": result.winner_recipe_id,
        "challenger_recipe_ids": list(result.challenger_recipe_ids),
        "watchlist_recipe_ids": list(result.watchlist_recipe_ids),
        "rejected_recipe_ids": list(result.rejected_recipe_ids),
        "decision_reasons": list(result.decision_reasons),
        "decision_reasons_zh": list(result.decision_reasons_zh),
        "challenger_reasons": list(result.challenger_reasons),
        "challenger_reasons_zh": list(result.challenger_reasons_zh),
        "rejection_reasons": list(result.rejection_reasons),
        "rejection_reasons_zh": list(result.rejection_reasons_zh),
        "next_actions": list(result.next_actions),
        "next_actions_zh": list(result.next_actions_zh),
        "missing_data_caveats": missing_data_caveats,
        "score_table": [
            {"recipe_id": recipe_id, "composite_score": score}
            for recipe_id, score in result.score_table
        ],
    }


def build_next_step_recommendations_artifact_payload(
    result: NextStepRecommendationResult,
    *,
    generated_at_utc: str,
    comparison_json_path: Path,
    default_profile: str,
    source_artifacts: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": WORKFLOW_CLOSURE_SCHEMA_VERSION,
        "artifact_type": NEXT_STEP_RECOMMENDATIONS_ARTIFACT_TYPE,
        "generated_at_utc": generated_at_utc,
        "comparison_json_path": str(comparison_json_path),
        "default_profile": default_profile,
        "source_artifacts": dict(source_artifacts),
        "policy": {
            "policy_id": result.policy_id,
            "policy_formula_text": result.policy_formula_text,
        },
        "recommendations": [
            {
                "recommendation_id": item.recommendation_id,
                "category": item.category,
                "priority": item.priority,
                "label_zh": item.label_zh,
                "action": item.action,
                "action_text_zh": item.action_text_zh,
                "rationale": item.rationale,
                "rationale_zh": item.rationale_zh,
                "triggered_by": list(item.trigger_objects),
                "supporting_evidence": list(item.supporting_evidence),
            }
            for item in result.recommendations
        ],
        "summary": list(result.summary),
        "summary_zh": list(result.summary_zh),
    }


def build_artifact_load_diagnostics_artifact_payload(
    data: ResearchDashboardData,
    *,
    generated_at_utc: str,
    comparison_json_path: Path,
    source_artifacts: Mapping[str, str],
) -> dict[str, object]:
    return {
        "schema_version": WORKFLOW_CLOSURE_SCHEMA_VERSION,
        "artifact_type": ARTIFACT_LOAD_DIAGNOSTICS_ARTIFACT_TYPE,
        "generated_at_utc": generated_at_utc,
        "comparison_json_path": str(comparison_json_path),
        "default_profile": data.default_profile,
        "artifact_load_mode": data.artifact_load_mode,
        "artifact_load_policy_summary": list(data.artifact_load_policy_summary),
        "diagnostics": [asdict(item) for item in data.artifact_load_diagnostics],
        "source_artifacts": dict(source_artifacts),
    }


def write_research_artifact_manifest_artifact(
    *,
    comparison_path: Path,
    default_profile: str,
    generated_at_utc: str,
    workflow_artifact_paths: Mapping[str, Path],
) -> Path:
    comparison_payload = _load_json(comparison_path)
    resolved_workflow_paths = resolve_workflow_closure_artifact_paths_from_comparison_payload(
        payload=comparison_payload,
        comparison_path=comparison_path,
    )
    resolved_workflow_paths.update(dict(workflow_artifact_paths))

    manifest_path = research_artifact_manifest_path(comparison_path.parent)
    write_validated_json_artifact(
        manifest_path,
        build_research_artifact_manifest_payload(
            comparison_payload=comparison_payload,
            comparison_path=comparison_path,
            generated_at_utc=generated_at_utc,
            default_profile=default_profile,
            workflow_artifact_paths=resolved_workflow_paths,
            manifest_path=manifest_path,
        ),
        artifact_name=manifest_path.name,
    )
    return manifest_path


def update_comparison_workflow_closure_context(
    *,
    comparison_path: Path,
    workflow_payload: Mapping[str, str],
) -> None:
    comparison_payload = _load_json(comparison_path)
    top_level_workflow = as_object_dict(comparison_payload.get("workflow_closure_artifacts"))
    top_level_workflow.update(dict(workflow_payload))
    comparison_payload["workflow_closure_artifacts"] = top_level_workflow

    profile_runs = comparison_payload.get("profile_runs")
    if isinstance(profile_runs, list):
        updated_profile_runs: list[dict[str, object]] = []
        for run in profile_runs:
            run_obj = as_object_dict(run)
            artifacts = as_object_dict(run_obj.get("campaign_artifacts"))
            run_workflow = as_object_dict(artifacts.get("workflow_closure_artifacts"))
            run_workflow.update(dict(workflow_payload))
            artifacts["workflow_closure_artifacts"] = run_workflow
            run_obj["campaign_artifacts"] = artifacts
            updated_profile_runs.append(run_obj)
        if updated_profile_runs:
            comparison_payload["profile_runs"] = updated_profile_runs

    summary = as_object_dict(comparison_payload.get("campaign_level_summary"))
    summary_workflow = as_object_dict(summary.get("workflow_closure_artifacts"))
    summary_workflow.update(dict(workflow_payload))
    summary["workflow_closure_artifacts"] = summary_workflow
    comparison_payload["campaign_level_summary"] = summary

    write_validated_json_artifact(
        comparison_path,
        comparison_payload,
        artifact_name=comparison_path.name,
    )


def persist_dashboard_workflow_artifacts(
    *,
    comparison_path: Path,
    data: ResearchDashboardData,
    source_artifacts: Mapping[str, str],
    generated_at_utc: str | None = None,
) -> dict[str, Path]:
    timestamp = generated_at_utc or datetime.now(UTC).isoformat()
    diagnostics_path = workflow_closure_artifact_paths(comparison_path.parent)[
        "artifact_load_diagnostics_json_path"
    ]
    write_validated_json_artifact(
        diagnostics_path,
        build_artifact_load_diagnostics_artifact_payload(
            data,
            generated_at_utc=timestamp,
            comparison_json_path=comparison_path,
            source_artifacts=source_artifacts,
        ),
        artifact_name=diagnostics_path.name,
    )
    manifest_path = write_research_artifact_manifest_artifact(
        comparison_path=comparison_path,
        default_profile=data.default_profile,
        generated_at_utc=timestamp,
        workflow_artifact_paths={
            "artifact_load_diagnostics_json_path": diagnostics_path,
        },
    )
    update_comparison_workflow_closure_context(
        comparison_path=comparison_path,
        workflow_payload={
            "artifact_load_diagnostics_json_path": str(diagnostics_path),
            RESEARCH_ARTIFACT_MANIFEST_CONTEXT_KEY: str(manifest_path),
        },
    )
    return {
        "artifact_load_diagnostics_json_path": diagnostics_path,
        RESEARCH_ARTIFACT_MANIFEST_CONTEXT_KEY: manifest_path,
    }


def persist_workflow_closure_artifacts(
    *,
    comparison_path: Path,
    data: ResearchDashboardData,
    winner_selection_policy: WinnerSelectionPolicy,
    winner_policy_formula_text: str,
    source_artifacts: Mapping[str, str],
    generated_at_utc: str | None = None,
) -> dict[str, Path]:
    timestamp = generated_at_utc or datetime.now(UTC).isoformat()
    workflow_paths = workflow_closure_artifact_paths(comparison_path.parent)

    payloads: dict[str, dict[str, object]] = {
        "factor_set_result_json_path": build_factor_set_result_artifact_payload(
            data.factor_sets,
            generated_at_utc=timestamp,
            comparison_json_path=comparison_path,
            default_profile=data.default_profile,
            source_artifacts=source_artifacts,
        ),
        "candidate_recipe_generation_json_path": (
            build_candidate_recipe_generation_artifact_payload(
                data.candidate_recipe_generation,
                generated_at_utc=timestamp,
                comparison_json_path=comparison_path,
                default_profile=data.default_profile,
                source_artifacts={
                    **dict(source_artifacts),
                    "factor_set_result_json_path": str(
                        workflow_paths["factor_set_result_json_path"]
                    ),
                },
            )
        ),
        "winner_selection_json_path": build_winner_selection_artifact_payload(
            data.winner_selection,
            policy=winner_selection_policy,
            winner_policy_formula_text=winner_policy_formula_text,
            generated_at_utc=timestamp,
            comparison_json_path=comparison_path,
            default_profile=data.default_profile,
            source_artifacts={
                **dict(source_artifacts),
                "factor_set_result_json_path": str(workflow_paths["factor_set_result_json_path"]),
                "candidate_recipe_generation_json_path": str(
                    workflow_paths["candidate_recipe_generation_json_path"]
                ),
            },
        ),
        "next_step_recommendations_json_path": (
            build_next_step_recommendations_artifact_payload(
                data.next_step_recommendations,
                generated_at_utc=timestamp,
                comparison_json_path=comparison_path,
                default_profile=data.default_profile,
                source_artifacts={
                    **dict(source_artifacts),
                    "factor_set_result_json_path": str(
                        workflow_paths["factor_set_result_json_path"]
                    ),
                    "candidate_recipe_generation_json_path": str(
                        workflow_paths["candidate_recipe_generation_json_path"]
                    ),
                    "winner_selection_json_path": str(workflow_paths["winner_selection_json_path"]),
                },
            )
        ),
        "artifact_load_diagnostics_json_path": build_artifact_load_diagnostics_artifact_payload(
            data,
            generated_at_utc=timestamp,
            comparison_json_path=comparison_path,
            source_artifacts={
                **dict(source_artifacts),
                "factor_set_result_json_path": str(workflow_paths["factor_set_result_json_path"]),
                "candidate_recipe_generation_json_path": str(
                    workflow_paths["candidate_recipe_generation_json_path"]
                ),
                "winner_selection_json_path": str(workflow_paths["winner_selection_json_path"]),
                "next_step_recommendations_json_path": str(
                    workflow_paths["next_step_recommendations_json_path"]
                ),
            },
        ),
    }
    for key, artifact_path in workflow_paths.items():
        payload = payloads[key]
        write_validated_json_artifact(
            artifact_path,
            payload,
            artifact_name=artifact_path.name,
        )

    manifest_path = write_research_artifact_manifest_artifact(
        comparison_path=comparison_path,
        default_profile=data.default_profile,
        generated_at_utc=timestamp,
        workflow_artifact_paths=workflow_paths,
    )
    workflow_paths[RESEARCH_ARTIFACT_MANIFEST_CONTEXT_KEY] = manifest_path
    update_comparison_workflow_closure_context(
        comparison_path=comparison_path,
        workflow_payload={key: str(path) for key, path in workflow_paths.items()},
    )
    return workflow_paths


def resolve_workflow_closure_artifact_paths_from_comparison_payload(
    *,
    payload: dict[str, object],
    comparison_path: Path,
) -> dict[str, Path]:
    base_dir = comparison_path.parent
    artifact_payload = as_object_dict(payload.get("workflow_closure_artifacts"))
    paths: dict[str, Path] = {}
    for key, filename in WORKFLOW_CLOSURE_ARTIFACT_FILENAMES.items():
        pointer_path = resolve_artifact_path(
            artifact_payload.get(key) if artifact_payload else None,
            base_dir=base_dir,
        )
        if pointer_path is not None and pointer_path.exists():
            paths[key] = pointer_path
            continue
        fallback_path = (base_dir / filename).resolve()
        if fallback_path.exists():
            paths[key] = fallback_path
    return paths


def _load_json(path: Path) -> dict[str, object]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return as_object_dict(loaded)
