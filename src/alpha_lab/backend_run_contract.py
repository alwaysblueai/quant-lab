from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from alpha_lab.artifact_contracts import validate_level12_artifact_payload
from alpha_lab.custom_factors import read_custom_factor_source
from alpha_lab.custom_models import read_draft_model_source

BackendRunWorkflow = Literal["single_factor", "model_factor"]

BACKEND_RUN_CONTRACT_VERSION = "backend_run_contract_v1"
BACKEND_RUN_RECEIPT_NAME = "backend_run_receipt.json"
COMPARISON_SUMMARY_NAME = "comparison_summary.json"

_COMMON_REQUIRED_FILES: tuple[str, ...] = (
    "run_manifest.json",
    "metrics.json",
    "summary.md",
    "integrity_report.json",
    "integrity_report.md",
    "signal_validation.json",
    "portfolio_recipe.json",
    "backtest_result.json",
    "research_tearsheet.json",
    "research_tearsheet.pdf",
    "experiment_card.md",
    "level2_portfolio_validation/portfolio_validation_summary.json",
    "level2_portfolio_validation/portfolio_validation_metrics.json",
    "level2_portfolio_validation/portfolio_validation_package.json",
    "level2_portfolio_validation/portfolio_validation_package.md",
)
_SINGLE_FACTOR_REQUIRED_FILES: tuple[str, ...] = (
    "factor_definition.json",
    "factor_definition.yaml",
)
_MODEL_FACTOR_REQUIRED_FILES: tuple[str, ...] = (
    "factor_definition.json",
    "model_factor_definition.yaml",
    "model_definition.json",
    "feature_manifest.json",
    "model_selection.json",
    "diagnostics.json",
    "training_log.csv",
    "training_metrics.csv",
    "feature_importance.csv",
    "feature_importance_ledger.csv",
    "feature_oos_ic.csv",
)
_LEVEL12_JSON_ARTIFACTS: tuple[str, ...] = (
    "run_manifest.json",
    "metrics.json",
    "factor_definition.json",
    "signal_validation.json",
    "portfolio_recipe.json",
    "backtest_result.json",
)
_FACTOR_AUDIT_FIELDS: tuple[str, ...] = (
    "path",
    "code_sha256",
    "factor_json_sha256",
)
_MODEL_AUDIT_FIELDS: tuple[str, ...] = (
    "path",
    "candidate_json_sha256",
    "case_spec_sha256",
    "feature_contract_sha256",
)
_COMPARISON_NUMERIC_KEYS: tuple[str, ...] = (
    "ic_mean",
    "mean_ic",
    "rank_ic_mean",
    "mean_rank_ic",
    "ic_t_stat",
    "turnover_mean",
    "mean_turnover",
    "portfolio_validation_base_mean_turnover",
    "portfolio_validation_base_cost_adjusted_return_review_rate",
)


def audit_backend_run_artifacts(
    output_dir: str | Path,
    *,
    workflow: BackendRunWorkflow,
    draft_source_path: str | Path | None = None,
    require_case_report: bool = True,
    require_comparison_summary: bool = True,
) -> dict[str, object]:
    """Audit one completed Level 1/2 backend run against the shared contract."""

    run_dir = Path(output_dir).expanduser().resolve()
    issues: list[dict[str, object]] = []
    required_files = _required_files_for_workflow(workflow)
    if require_case_report:
        required_files = (*required_files, "case_report.md")
    if require_comparison_summary:
        required_files = (*required_files, COMPARISON_SUMMARY_NAME)

    present_files: list[str] = []
    missing_files: list[str] = []
    for relative in required_files:
        path = run_dir / relative
        if path.exists():
            present_files.append(relative)
        else:
            missing_files.append(relative)
            _issue(
                issues,
                severity="error",
                code="missing_required_artifact",
                message=f"required artifact is missing: {relative}",
                artifact=relative,
            )

    artifact_contracts = _validate_level12_artifacts(run_dir, issues)
    if workflow == "model_factor":
        _validate_model_definition(run_dir / "model_definition.json", issues)
        _validate_feature_manifest(run_dir / "feature_manifest.json", issues)

    source_audit = _audit_source_blocks(
        run_dir,
        workflow=workflow,
        draft_source_path=draft_source_path,
        issues=issues,
    )

    ok = not any(str(issue.get("severity")) == "error" for issue in issues)
    return {
        "contract_version": BACKEND_RUN_CONTRACT_VERSION,
        "workflow": workflow,
        "ok": ok,
        "output_dir": str(run_dir),
        "required_files": list(required_files),
        "present_files": present_files,
        "missing_files": missing_files,
        "artifact_contracts": artifact_contracts,
        "source_audit": source_audit,
        "issues": issues,
    }


def build_comparison_summary(
    output_dir: str | Path,
    *,
    workflow: BackendRunWorkflow,
    compare_to_output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Build a stable comparison artifact, even when no baseline is supplied."""

    run_dir = Path(output_dir).expanduser().resolve()
    manifest = _read_json_object(run_dir / "run_manifest.json")
    metrics_payload = _read_json_object(run_dir / "metrics.json")
    metrics = _as_mapping(metrics_payload.get("metrics"))
    case_name = _text(manifest.get("case_name")) or run_dir.name
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_backend_comparison_summary",
        "contract_version": BACKEND_RUN_CONTRACT_VERSION,
        "workflow": workflow,
        "case_name": case_name,
        "output_dir": str(run_dir),
        "generated_at_utc": _utc_now(),
    }
    if compare_to_output_dir is None:
        payload.update(
            {
                "status": "not_available",
                "reason": "no comparator output directory supplied",
                "metric_deltas": {},
            }
        )
        return payload

    baseline_dir = Path(compare_to_output_dir).expanduser().resolve()
    baseline_manifest = _read_json_object(baseline_dir / "run_manifest.json")
    baseline_metrics_payload = _read_json_object(baseline_dir / "metrics.json")
    baseline_metrics = _as_mapping(baseline_metrics_payload.get("metrics"))
    deltas: dict[str, object] = {}
    for key in _COMPARISON_NUMERIC_KEYS:
        current_value = _finite_number(metrics.get(key))
        baseline_value = _finite_number(baseline_metrics.get(key))
        if current_value is None or baseline_value is None:
            continue
        deltas[key] = {
            "current": current_value,
            "baseline": baseline_value,
            "delta": current_value - baseline_value,
        }
    payload.update(
        {
            "status": "available" if deltas else "not_available",
            "reason": "" if deltas else "no shared numeric comparison metrics found",
            "baseline": {
                "case_name": _text(baseline_manifest.get("case_name")) or baseline_dir.name,
                "output_dir": str(baseline_dir),
            },
            "metric_deltas": deltas,
        }
    )
    return payload


def write_comparison_summary(
    output_dir: str | Path,
    *,
    workflow: BackendRunWorkflow,
    compare_to_output_dir: str | Path | None = None,
) -> Path:
    path = Path(output_dir).expanduser().resolve() / COMPARISON_SUMMARY_NAME
    _write_json(
        path,
        build_comparison_summary(
            output_dir,
            workflow=workflow,
            compare_to_output_dir=compare_to_output_dir,
        ),
    )
    return path


def build_backend_run_receipt(
    *,
    workflow: BackendRunWorkflow,
    output_dir: str | Path,
    case_spec_path: str | Path,
    draft_source_path: str | Path,
    validation_payload: Mapping[str, object],
    artifact_audit: Mapping[str, object],
    evaluation_profile: str,
    comparison_summary_path: str | Path | None = None,
    command: Sequence[str] = (),
    contract_ok: bool | None = None,
) -> dict[str, object]:
    """Build a backend-run receipt payload.

    ``contract_ok`` is the unified contract pass/fail flag combining validator
    and artifact audit. When omitted, it falls back to ``artifact_audit.ok``
    so callers that don't run a validator (and library tests) keep current
    behavior.
    """

    run_dir = Path(output_dir).expanduser().resolve()
    manifest = _read_json_object(run_dir / "run_manifest.json")
    resolved_ok = (
        bool(contract_ok)
        if contract_ok is not None
        else bool(artifact_audit.get("ok"))
    )
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_backend_run_receipt",
        "contract_version": BACKEND_RUN_CONTRACT_VERSION,
        "workflow": workflow,
        "status": "success" if resolved_ok else "failed",
        "generated_at_utc": _utc_now(),
        "case_name": _text(manifest.get("case_name")) or run_dir.name,
        "output_dir": str(run_dir),
        "case_spec_path": str(Path(case_spec_path).expanduser().resolve()),
        "draft_source_path": str(Path(draft_source_path).expanduser().resolve()),
        "evaluation_profile": evaluation_profile,
        "validation": dict(validation_payload),
        "artifact_audit": dict(artifact_audit),
        "comparison_summary_path": (
            str(Path(comparison_summary_path).expanduser().resolve())
            if comparison_summary_path is not None
            else None
        ),
        "command": list(command),
    }


def write_backend_run_receipt(
    output_dir: str | Path,
    receipt: Mapping[str, object],
) -> Path:
    path = Path(output_dir).expanduser().resolve() / BACKEND_RUN_RECEIPT_NAME
    _write_json(path, dict(receipt))
    return path


def detect_research_draft_run(
    output_dir: str | Path,
    *,
    workflow: BackendRunWorkflow,
) -> Path | None:
    """Return the draft source path if this run looks like a research draft run.

    A research draft run is one whose ``run_manifest.json`` carries a complete
    ``custom_factor_source`` (single_factor) or ``draft_model_source`` (model_factor)
    block whose ``path`` lies under ``custom_factors/research/`` or
    ``custom_models/research/``. This is the trigger used by ``real-case`` to
    decide whether to run :func:`finalize_backend_contract`.
    """

    run_dir = Path(output_dir).expanduser().resolve()
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = _read_json_object(manifest_path)
    except Exception:
        return None
    if workflow == "single_factor":
        block = _as_mapping(manifest.get("custom_factor_source"))
        required = _FACTOR_AUDIT_FIELDS
        scope_marker = "custom_factors/research/"
    elif workflow == "model_factor":
        block = _as_mapping(manifest.get("draft_model_source"))
        required = _MODEL_AUDIT_FIELDS
        scope_marker = "custom_models/research/"
    else:
        raise ValueError(f"unsupported backend workflow: {workflow!r}")
    path_text = _text(block.get("path"))
    if not path_text:
        return None
    if scope_marker not in path_text.replace("\\", "/"):
        return None
    for field in required:
        if not _text(block.get(field)):
            return None
    return Path(path_text)


def finalize_backend_contract(
    output_dir: str | Path,
    *,
    workflow: BackendRunWorkflow,
    draft_source_path: str | Path,
    case_spec_path: str | Path,
    evaluation_profile: str,
    command: Sequence[str] = (),
    validation_payload: Mapping[str, object] | None = None,
    compare_to_output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Run the four-step backend contract finalization on a completed run.

    Order:

    1. Write ``comparison_summary.json`` (sidecar checked by audit presence).
    2. Audit artifacts + source-hash audit fields.
    3. Build receipt embedding the audit result.
    4. Write receipt and attach both sidecars to ``run_manifest.json``.

    Returns the receipt payload. Callers (CLI / web) should inspect
    ``receipt['status']`` to decide process exit code; the sidecars are always
    written so failures remain auditable.
    """

    run_dir = Path(output_dir).expanduser().resolve()
    comparison_path = write_comparison_summary(
        run_dir,
        workflow=workflow,
        compare_to_output_dir=compare_to_output_dir,
    )
    artifact_audit = audit_backend_run_artifacts(
        run_dir,
        workflow=workflow,
        draft_source_path=draft_source_path,
    )
    validation_dict = dict(validation_payload or {})
    validation_ok_field = validation_dict.get("ok")
    validation_ok = (
        bool(validation_ok_field)
        if isinstance(validation_ok_field, bool)
        else True
    )
    contract_ok = validation_ok and bool(artifact_audit.get("ok"))
    receipt = build_backend_run_receipt(
        workflow=workflow,
        output_dir=run_dir,
        case_spec_path=case_spec_path,
        draft_source_path=draft_source_path,
        validation_payload=validation_dict,
        artifact_audit=artifact_audit,
        evaluation_profile=evaluation_profile,
        comparison_summary_path=comparison_path,
        command=command,
        contract_ok=contract_ok,
    )
    receipt_path = write_backend_run_receipt(run_dir, receipt)
    attach_backend_contract_to_manifest(
        run_dir,
        receipt_path=receipt_path,
        comparison_summary_path=comparison_path,
        artifact_audit=artifact_audit,
        contract_ok=contract_ok,
    )
    return receipt


def attach_backend_contract_to_manifest(
    output_dir: str | Path,
    *,
    receipt_path: str | Path,
    comparison_summary_path: str | Path,
    artifact_audit: Mapping[str, object],
    contract_ok: bool | None = None,
) -> None:
    """Record backend-run sidecar artifacts in run_manifest.json.

    ``contract_ok`` is the unified contract pass/fail flag combining validator
    and artifact audit. When omitted, it falls back to ``artifact_audit.ok``.
    """

    run_dir = Path(output_dir).expanduser().resolve()
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json_object(manifest_path)
    outputs = dict(_as_mapping(manifest.get("outputs")))
    outputs["backend_run_receipt"] = str(Path(receipt_path).expanduser().resolve())
    outputs["comparison_summary"] = str(Path(comparison_summary_path).expanduser().resolve())
    manifest["outputs"] = outputs
    raw_issues = artifact_audit.get("issues")
    issue_count = len(raw_issues) if isinstance(raw_issues, list) else 0
    resolved_ok = (
        bool(contract_ok)
        if contract_ok is not None
        else bool(artifact_audit.get("ok"))
    )
    manifest["backend_run_contract"] = {
        "contract_version": BACKEND_RUN_CONTRACT_VERSION,
        "status": "passed" if resolved_ok else "failed",
        "receipt_path": outputs["backend_run_receipt"],
        "comparison_summary_path": outputs["comparison_summary"],
        "audited_at_utc": _utc_now(),
        "issue_count": issue_count,
    }
    _write_json(manifest_path, manifest)


def _required_files_for_workflow(workflow: BackendRunWorkflow) -> tuple[str, ...]:
    if workflow == "single_factor":
        return (*_COMMON_REQUIRED_FILES, *_SINGLE_FACTOR_REQUIRED_FILES)
    if workflow == "model_factor":
        return (*_COMMON_REQUIRED_FILES, *_MODEL_FACTOR_REQUIRED_FILES)
    raise ValueError(f"unsupported backend workflow: {workflow!r}")


def _validate_level12_artifacts(
    run_dir: Path,
    issues: list[dict[str, object]],
) -> dict[str, object]:
    statuses: dict[str, object] = {}
    for filename in _LEVEL12_JSON_ARTIFACTS:
        path = run_dir / filename
        if not path.exists():
            continue
        try:
            payload = _read_json_object(path)
            validate_level12_artifact_payload(payload, artifact_name=filename, source=path)
        except Exception as exc:  # noqa: BLE001
            statuses[filename] = {"ok": False, "error": str(exc)}
            _issue(
                issues,
                severity="error",
                code="artifact_contract_invalid",
                message=f"{filename} failed Level 1/2 contract validation: {exc}",
                artifact=filename,
            )
        else:
            statuses[filename] = {"ok": True}
    return statuses


def _validate_model_definition(path: Path, issues: list[dict[str, object]]) -> None:
    if not path.exists():
        return
    payload = _read_json_object(path)
    _require_text_field(payload, "artifact_type", path, issues)
    if payload.get("artifact_type") != "alpha_lab_model_definition":
        _issue(
            issues,
            severity="error",
            code="model_definition_artifact_type",
            message="model_definition.json artifact_type must be alpha_lab_model_definition",
            artifact=str(path.name),
        )
    for key in ("case_name", "factor_name", "model_family"):
        _require_text_field(payload, key, path, issues)
    _require_mapping_field(payload, "source_artifacts", path, issues)


def _validate_feature_manifest(path: Path, issues: list[dict[str, object]]) -> None:
    if not path.exists():
        return
    payload = _read_json_object(path)
    _require_text_field(payload, "artifact_type", path, issues)
    if payload.get("artifact_type") != "alpha_lab_feature_manifest":
        _issue(
            issues,
            severity="error",
            code="feature_manifest_artifact_type",
            message="feature_manifest.json artifact_type must be alpha_lab_feature_manifest",
            artifact=str(path.name),
        )
    for key in ("case_name", "factor_name", "model_family"):
        _require_text_field(payload, key, path, issues)
    if not isinstance(payload.get("feature_columns"), list):
        _issue(
            issues,
            severity="error",
            code="feature_manifest_columns",
            message="feature_manifest.json feature_columns must be a list",
            artifact=str(path.name),
        )
    _require_mapping_field(payload, "feature_availability", path, issues)


def _audit_source_blocks(
    run_dir: Path,
    *,
    workflow: BackendRunWorkflow,
    draft_source_path: str | Path | None,
    issues: list[dict[str, object]],
) -> dict[str, object]:
    if workflow == "single_factor":
        return _audit_factor_source_blocks(run_dir, draft_source_path, issues)
    if workflow == "model_factor":
        return _audit_model_source_blocks(run_dir, draft_source_path, issues)
    raise ValueError(f"unsupported backend workflow: {workflow!r}")


def _audit_factor_source_blocks(
    run_dir: Path,
    draft_source_path: str | Path | None,
    issues: list[dict[str, object]],
) -> dict[str, object]:
    expected = (
        read_custom_factor_source(draft_source_path).to_audit_dict()
        if draft_source_path is not None
        else None
    )
    manifest = _read_json_object(run_dir / "run_manifest.json")
    factor_definition = _read_json_object(run_dir / "factor_definition.json")
    locations = {
        "run_manifest.custom_factor_source": _as_mapping(
            manifest.get("custom_factor_source")
        ),
        "run_manifest.inputs.custom_factor_source": _as_mapping(
            _as_mapping(manifest.get("inputs")).get("custom_factor_source")
        ),
        "factor_definition.custom_factor_source": _as_mapping(
            factor_definition.get("custom_factor_source")
        ),
    }
    return _audit_hash_locations(
        locations,
        required_fields=_FACTOR_AUDIT_FIELDS,
        expected=expected,
        issues=issues,
        audit_key="custom_factor_source",
    )


def _audit_model_source_blocks(
    run_dir: Path,
    draft_source_path: str | Path | None,
    issues: list[dict[str, object]],
) -> dict[str, object]:
    expected = (
        read_draft_model_source(draft_source_path).to_audit_dict()
        if draft_source_path is not None
        else None
    )
    manifest = _read_json_object(run_dir / "run_manifest.json")
    model_definition = _read_json_object(run_dir / "model_definition.json")
    feature_manifest = _read_json_object(run_dir / "feature_manifest.json")
    locations = {
        "run_manifest.draft_model_source": _as_mapping(manifest.get("draft_model_source")),
        "run_manifest.inputs.draft_model_source": _as_mapping(
            _as_mapping(manifest.get("inputs")).get("draft_model_source")
        ),
        "model_definition.draft_model_source": _as_mapping(
            model_definition.get("draft_model_source")
        ),
        "feature_manifest.draft_model_source": _as_mapping(
            feature_manifest.get("draft_model_source")
        ),
    }
    return _audit_hash_locations(
        locations,
        required_fields=_MODEL_AUDIT_FIELDS,
        expected=expected,
        issues=issues,
        audit_key="draft_model_source",
    )


def _audit_hash_locations(
    locations: Mapping[str, Mapping[str, object]],
    *,
    required_fields: Sequence[str],
    expected: Mapping[str, object] | None,
    issues: list[dict[str, object]],
    audit_key: str,
) -> dict[str, object]:
    location_payloads: dict[str, object] = {}
    for location, payload in locations.items():
        location_ok = True
        missing = [
            field
            for field in required_fields
            if not _text(payload.get(field))
        ]
        if missing:
            location_ok = False
            _issue(
                issues,
                severity="error",
                code=f"{audit_key}_missing_fields",
                message=f"{location} missing required audit fields: {missing}",
                artifact=location,
            )
        mismatches: dict[str, object] = {}
        if expected is not None:
            for field in required_fields:
                actual = _text(payload.get(field))
                expected_value = _text(expected.get(field))
                if actual and expected_value and actual != expected_value:
                    location_ok = False
                    mismatches[field] = {"actual": actual, "expected": expected_value}
            if mismatches:
                _issue(
                    issues,
                    severity="error",
                    code=f"{audit_key}_hash_mismatch",
                    message=f"{location} does not match draft source audit hashes",
                    artifact=location,
                    details=mismatches,
                )
        location_payloads[location] = {
            "ok": location_ok,
            "missing_fields": missing,
            "mismatches": mismatches,
        }
    return {
        "audit_key": audit_key,
        "expected_source_path": _text(expected.get("path")) if expected is not None else None,
        "locations": location_payloads,
    }


def _require_text_field(
    payload: Mapping[str, object],
    key: str,
    path: Path,
    issues: list[dict[str, object]],
) -> None:
    if not _text(payload.get(key)):
        _issue(
            issues,
            severity="error",
            code="missing_text_field",
            message=f"{path.name}.{key} must be a non-empty string",
            artifact=path.name,
        )


def _require_mapping_field(
    payload: Mapping[str, object],
    key: str,
    path: Path,
    issues: list[dict[str, object]],
) -> None:
    if not isinstance(payload.get(key), Mapping):
        _issue(
            issues,
            severity="error",
            code="missing_object_field",
            message=f"{path.name}.{key} must be an object",
            artifact=path.name,
        )


def _issue(
    issues: list[dict[str, object]],
    *,
    severity: str,
    code: str,
    message: str,
    artifact: str,
    details: Mapping[str, object] | None = None,
) -> None:
    payload: dict[str, object] = {
        "severity": severity,
        "code": code,
        "message": message,
        "artifact": artifact,
    }
    if details:
        payload["details"] = dict(details)
    issues.append(payload)


def _read_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} root must be an object")
    return cast(dict[str, object], payload)


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_mapping(value: object) -> Mapping[str, object]:
    return cast(Mapping[str, object], value) if isinstance(value, Mapping) else {}


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number == number and number not in {float("inf"), float("-inf")}:
            return number
    return None


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


__all__ = [
    "BACKEND_RUN_CONTRACT_VERSION",
    "BackendRunWorkflow",
    "BACKEND_RUN_RECEIPT_NAME",
    "COMPARISON_SUMMARY_NAME",
    "attach_backend_contract_to_manifest",
    "audit_backend_run_artifacts",
    "build_backend_run_receipt",
    "build_comparison_summary",
    "detect_research_draft_run",
    "finalize_backend_contract",
    "write_backend_run_receipt",
    "write_comparison_summary",
]
