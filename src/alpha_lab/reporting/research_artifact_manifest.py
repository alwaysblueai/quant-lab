from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from alpha_lab.artifact_contracts import (
    MODEL_FACTOR_DEFERRED_DIAGNOSTIC_CONTRACTS,
    validate_level12_artifact_payload,
)
from alpha_lab.reporting._shared import resolve_artifact_path
from alpha_lab.reporting.display_helpers import (
    as_object_dict,
    as_object_list,
    safe_text,
)

WORKFLOW_CLOSURE_SCHEMA_VERSION = "1.0.0"
FACTOR_SET_RESULT_ARTIFACT_TYPE = "alpha_lab_factor_set_result"
CANDIDATE_RECIPE_GENERATION_ARTIFACT_TYPE = "alpha_lab_candidate_recipe_generation"
WINNER_SELECTION_ARTIFACT_TYPE = "alpha_lab_winner_selection"
NEXT_STEP_RECOMMENDATIONS_ARTIFACT_TYPE = "alpha_lab_next_step_recommendations"
ARTIFACT_LOAD_DIAGNOSTICS_ARTIFACT_TYPE = "alpha_lab_artifact_load_diagnostics"
RESEARCH_ARTIFACT_MANIFEST_ARTIFACT_TYPE = "alpha_lab_research_artifact_manifest"
RESEARCH_ARTIFACT_MANIFEST_CONTEXT_KEY = "research_artifact_manifest_json_path"
RESEARCH_ARTIFACT_MANIFEST_FILENAME = "research_artifact_manifest.json"
WORKFLOW_CLOSURE_ARTIFACT_FILENAMES: dict[str, str] = {
    "factor_set_result_json_path": "factor_set_result.json",
    "candidate_recipe_generation_json_path": "candidate_recipe_generation.json",
    "winner_selection_json_path": "winner_selection.json",
    "next_step_recommendations_json_path": "next_step_recommendations.json",
    "artifact_load_diagnostics_json_path": "artifact_load_diagnostics.json",
}
CANONICAL_ARTIFACT_REQUIREMENTS: tuple[tuple[str, str, str], ...] = (
    ("factor_definition", "factor_definition.json", "factor_definition_json_path"),
    ("signal_validation", "signal_validation.json", "signal_validation_json_path"),
    ("portfolio_recipe", "portfolio_recipe.json", "portfolio_recipe_json_path"),
    ("backtest_result", "backtest_result.json", "backtest_result_json_path"),
)
WORKFLOW_CLOSURE_ARTIFACT_REQUIREMENTS: tuple[tuple[str, str, str], ...] = (
    ("factor_set_result", "factor_set_result.json", "factor_set_result_json_path"),
    (
        "candidate_recipe_generation",
        "candidate_recipe_generation.json",
        "candidate_recipe_generation_json_path",
    ),
    ("winner_selection", "winner_selection.json", "winner_selection_json_path"),
    (
        "next_step_recommendations",
        "next_step_recommendations.json",
        "next_step_recommendations_json_path",
    ),
    (
        "artifact_load_diagnostics",
        "artifact_load_diagnostics.json",
        "artifact_load_diagnostics_json_path",
    ),
)
CANONICAL_ARTIFACT_TYPE_BY_OBJECT: dict[str, str] = {
    "factor_definition": "alpha_lab_factor_definition",
    "signal_validation": "alpha_lab_signal_validation",
    "portfolio_recipe": "alpha_lab_portfolio_recipe",
    "backtest_result": "alpha_lab_backtest_result",
}
WORKFLOW_ARTIFACT_DESCRIPTOR_BY_KEY: dict[str, tuple[str, str, str, bool]] = {
    "factor_set_result_json_path": (
        "factor_set_result",
        FACTOR_SET_RESULT_ARTIFACT_TYPE,
        "workflow",
        True,
    ),
    "candidate_recipe_generation_json_path": (
        "candidate_recipe_generation",
        CANDIDATE_RECIPE_GENERATION_ARTIFACT_TYPE,
        "workflow",
        True,
    ),
    "winner_selection_json_path": (
        "winner_selection",
        WINNER_SELECTION_ARTIFACT_TYPE,
        "workflow",
        True,
    ),
    "next_step_recommendations_json_path": (
        "next_step_recommendations",
        NEXT_STEP_RECOMMENDATIONS_ARTIFACT_TYPE,
        "workflow",
        True,
    ),
    "artifact_load_diagnostics_json_path": (
        "artifact_load_diagnostics",
        ARTIFACT_LOAD_DIAGNOSTICS_ARTIFACT_TYPE,
        "governance",
        True,
    ),
    RESEARCH_ARTIFACT_MANIFEST_CONTEXT_KEY: (
        "research_artifact_manifest",
        RESEARCH_ARTIFACT_MANIFEST_ARTIFACT_TYPE,
        "governance",
        False,
    ),
}

_CANONICAL_PRODUCER_HINT = (
    "alpha_lab.real_cases.single_factor.artifacts.export_artifact_bundle "
    "or alpha_lab.real_cases.composite.artifacts.export_artifact_bundle"
)
_WORKFLOW_PRODUCER_HINT = (
    "alpha_lab.reporting.renderers.campaign_profile_dashboard."
    "persist_workflow_closure_artifacts / write_campaign_profile_dashboard_html"
)


def research_artifact_manifest_path(base_dir: Path) -> Path:
    return (base_dir / RESEARCH_ARTIFACT_MANIFEST_FILENAME).resolve()


def build_research_artifact_manifest_payload(
    *,
    comparison_payload: Mapping[str, object],
    comparison_path: Path,
    generated_at_utc: str,
    default_profile: str,
    workflow_artifact_paths: Mapping[str, Path],
    manifest_path: Path,
) -> dict[str, object]:
    payload_obj = as_object_dict(comparison_payload)
    entries = _canonical_artifact_manifest_entries(
        comparison_payload=payload_obj,
        comparison_path=comparison_path,
    )
    entries.extend(
        _workflow_artifact_manifest_entries(
            default_profile=default_profile,
            workflow_artifact_paths=workflow_artifact_paths,
            manifest_path=manifest_path,
        )
    )
    entries.extend(_deferred_diagnostic_contract_manifest_entries(default_profile=default_profile))
    return {
        "schema_version": WORKFLOW_CLOSURE_SCHEMA_VERSION,
        "artifact_type": RESEARCH_ARTIFACT_MANIFEST_ARTIFACT_TYPE,
        "generated_at_utc": generated_at_utc,
        "comparison_json_path": str(comparison_path),
        "default_profile": default_profile,
        "artifact_entries": entries,
        "summary": _research_artifact_manifest_summary(entries),
    }


def _canonical_artifact_manifest_entries(
    *,
    comparison_payload: dict[str, object],
    comparison_path: Path,
) -> list[dict[str, object]]:
    base_dir = comparison_path.parent
    entries: list[dict[str, object]] = []
    case_rows = as_object_list(comparison_payload.get("case_comparison"))
    for case_row_raw in case_rows:
        case_row = as_object_dict(case_row_raw)
        case_name = safe_text(case_row.get("case_name")) or None
        profiles_obj = as_object_dict(case_row.get("profiles"))
        for profile_name, profile_payload_raw in sorted(profiles_obj.items()):
            profile_payload = as_object_dict(profile_payload_raw)
            status = safe_text(profile_payload.get("status")) or "unknown"
            artifact_paths = as_object_dict(profile_payload.get("artifact_paths"))
            output_dir = resolve_artifact_path(
                artifact_paths.get("output_dir"),
                base_dir=base_dir,
            )
            for object_label, filename, pointer_key in CANONICAL_ARTIFACT_REQUIREMENTS:
                path = resolve_artifact_path(
                    artifact_paths.get(pointer_key),
                    base_dir=base_dir,
                )
                if path is None and output_dir is not None:
                    path = (output_dir / filename).resolve()
                entries.append(
                    {
                        "artifact_name": filename,
                        "artifact_type": CANONICAL_ARTIFACT_TYPE_BY_OBJECT.get(
                            object_label,
                            "unknown",
                        ),
                        "artifact_layer": "canonical",
                        "path": str(path) if path is not None else None,
                        "scope": "case",
                        "case_name": case_name,
                        "profile_name": profile_name,
                        "producer_hint": _CANONICAL_PRODUCER_HINT,
                        "validation_status": _artifact_validation_status(
                            path=path,
                            artifact_name=filename,
                        ),
                        "required_in_strict_mode": status == "success",
                        "lineage_role": object_label,
                    }
                )
    return entries


def _deferred_diagnostic_contract_manifest_entries(
    *,
    default_profile: str,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for contract in MODEL_FACTOR_DEFERRED_DIAGNOSTIC_CONTRACTS:
        entry = {
            "artifact_name": contract["artifact_name"],
            "artifact_type": contract["artifact_type"],
            "artifact_layer": contract["artifact_layer"],
            "path": None,
            "scope": contract["scope"],
            "case_name": None,
            "profile_name": default_profile,
            "producer_hint": contract["producer_hint"],
            "validation_status": "not_emitted_v1",
            "required_in_strict_mode": False,
            "lineage_role": contract["lineage_role"],
            "contract_status": contract["contract_status"],
            "row_grain": contract["row_grain"],
            "required_columns": _contract_string_list(contract, "required_columns"),
            "description_zh": contract.get("description_zh"),
        }
        if contract.get("optional_columns"):
            entry["optional_columns"] = _contract_string_list(contract, "optional_columns")
        if contract.get("alternative_artifact_names"):
            entry["alternative_artifact_names"] = _contract_string_list(
                contract,
                "alternative_artifact_names",
            )
        entries.append(entry)
    return entries


def _contract_string_list(contract: Mapping[str, object], key: str) -> list[str]:
    raw = contract.get(key)
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(item) for item in raw if str(item).strip()]


def _workflow_artifact_manifest_entries(
    *,
    default_profile: str,
    workflow_artifact_paths: Mapping[str, Path],
    manifest_path: Path,
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for key, (
        object_label,
        artifact_type,
        artifact_layer,
        required_in_strict_mode,
    ) in WORKFLOW_ARTIFACT_DESCRIPTOR_BY_KEY.items():
        path: Path | None
        if key == RESEARCH_ARTIFACT_MANIFEST_CONTEXT_KEY:
            path = manifest_path
            validation_status = "unchecked"
        else:
            path = workflow_artifact_paths.get(key)
            artifact_name = WORKFLOW_CLOSURE_ARTIFACT_FILENAMES.get(key, object_label + ".json")
            validation_status = _artifact_validation_status(
                path=path,
                artifact_name=artifact_name,
            )
        entries.append(
            {
                "artifact_name": (
                    RESEARCH_ARTIFACT_MANIFEST_FILENAME
                    if key == RESEARCH_ARTIFACT_MANIFEST_CONTEXT_KEY
                    else WORKFLOW_CLOSURE_ARTIFACT_FILENAMES.get(key, object_label + ".json")
                ),
                "artifact_type": artifact_type,
                "artifact_layer": artifact_layer,
                "path": str(path) if path is not None else None,
                "scope": "comparison",
                "case_name": None,
                "profile_name": default_profile,
                "producer_hint": _WORKFLOW_PRODUCER_HINT,
                "validation_status": validation_status,
                "required_in_strict_mode": required_in_strict_mode,
                "lineage_role": object_label,
            }
        )
    return entries


def _research_artifact_manifest_summary(
    entries: list[dict[str, object]],
) -> dict[str, object]:
    by_layer: dict[str, int] = {}
    by_artifact_type: dict[str, int] = {}
    by_validation_status: dict[str, int] = {}
    for entry in entries:
        layer = safe_text(entry.get("artifact_layer")) or "unknown"
        artifact_type = safe_text(entry.get("artifact_type")) or "unknown"
        validation_status = safe_text(entry.get("validation_status")) or "unknown"
        by_layer[layer] = by_layer.get(layer, 0) + 1
        by_artifact_type[artifact_type] = by_artifact_type.get(artifact_type, 0) + 1
        by_validation_status[validation_status] = by_validation_status.get(validation_status, 0) + 1
    return {
        "total_entries": len(entries),
        "by_layer": by_layer,
        "by_artifact_type": by_artifact_type,
        "by_validation_status": by_validation_status,
    }


def _artifact_validation_status(
    *,
    path: Path | None,
    artifact_name: str,
) -> str:
    if path is None:
        return "unresolved"
    if not path.exists():
        return "missing"
    payload = _load_optional_json(path)
    if not payload:
        return "invalid"
    try:
        validate_level12_artifact_payload(
            payload,
            artifact_name=artifact_name,
            source=path,
        )
    except (ValueError, TypeError, KeyError):
        return "invalid"
    return "valid"


def _load_optional_json(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return as_object_dict(loaded)
