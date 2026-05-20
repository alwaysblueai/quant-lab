"""Draft-model candidate metadata loader and audit helper.

Mirror of :mod:`alpha_lab.custom_factors` for Stage 3 backend draft models.
The audited artifact is ``custom_models/research/<candidate>/model_candidate.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_lab.custom_factors import sha256_text

MODEL_CANDIDATE_SCOPES: tuple[str, ...] = ("promoted", "research")


@dataclass(frozen=True)
class DraftModelSource:
    """Auditable metadata for one persisted Stage3 draft model candidate."""

    name: str
    scope: str
    path: Path
    candidate_json_sha256: str
    case_spec_sha256: str
    feature_contract_sha256: str
    contract_version: str
    implementation_status: str
    implementation_type: str
    factor_name: str
    archive_identity: str = ""
    feature_columns: tuple[str, ...] = ()
    feature_availability_mode: str | None = None
    feature_availability_column: str | None = None
    feature_availability_safety_lag_days: int | None = None
    model_family: str | None = None
    description: str = ""
    provenance: dict[str, Any] | None = None

    def to_audit_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "scope": self.scope,
            "path": str(self.path),
            "candidate_json_sha256": self.candidate_json_sha256,
            "case_spec_sha256": self.case_spec_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "contract_version": self.contract_version,
            "implementation_status": self.implementation_status,
            "implementation_type": self.implementation_type,
            "factor_name": self.factor_name,
        }
        if self.feature_columns:
            payload["feature_columns"] = list(self.feature_columns)
        if self.archive_identity:
            payload["archive_identity"] = self.archive_identity
        feature_availability: dict[str, object] = {}
        if self.feature_availability_mode is not None:
            feature_availability["mode"] = self.feature_availability_mode
        if self.feature_availability_column is not None:
            feature_availability["column"] = self.feature_availability_column
        if self.feature_availability_safety_lag_days is not None:
            feature_availability["safety_lag_days"] = (
                self.feature_availability_safety_lag_days
            )
        if feature_availability:
            payload["feature_availability"] = feature_availability
        if self.model_family is not None:
            payload["model_family"] = self.model_family
        if self.description:
            payload["description"] = self.description
        if self.provenance:
            payload["provenance"] = dict(self.provenance)
        return payload


def read_draft_model_source(path: str | Path) -> DraftModelSource:
    """Load and hash a ``model_candidate.json`` for artifact auditing."""

    candidate_path = Path(path).expanduser().resolve()
    raw_text = candidate_path.read_text(encoding="utf-8")
    payload: Any = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError(
            f"model_candidate metadata must be an object: {candidate_path}"
        )

    candidate_name = str(payload.get("candidate_name") or "").strip().lower()
    if not candidate_name:
        raise ValueError(
            f"model_candidate metadata missing candidate_name: {candidate_path}"
        )

    case_spec_payload = payload.get("case_spec_payload")
    if not isinstance(case_spec_payload, dict):
        raise ValueError(
            f"model_candidate.case_spec_payload must be an object: {candidate_path}"
        )

    case_spec_sha256 = sha256_text(
        json.dumps(case_spec_payload, ensure_ascii=False, sort_keys=True)
    )
    feature_contract_payload = {
        "feature_columns": list(case_spec_payload.get("feature_columns") or []),
        "feature_availability": dict(
            case_spec_payload.get("feature_availability") or {}
        ),
        "feature_preprocess": dict(
            case_spec_payload.get("feature_preprocess") or {}
        ),
    }
    feature_contract_sha256 = sha256_text(
        json.dumps(feature_contract_payload, ensure_ascii=False, sort_keys=True)
    )

    factor_name = str(case_spec_payload.get("factor_name") or "").strip()
    feature_columns = tuple(
        str(col).strip()
        for col in case_spec_payload.get("feature_columns") or ()
        if str(col).strip()
    )
    feature_availability = case_spec_payload.get("feature_availability") or {}
    if not isinstance(feature_availability, dict):
        feature_availability = {}
    model_section = case_spec_payload.get("model") or {}
    if not isinstance(model_section, dict):
        model_section = {}
    model_family = (
        str(model_section.get("family")).strip()
        if isinstance(model_section.get("family"), str)
        else None
    )
    safety_lag_raw = feature_availability.get("safety_lag_days")
    if isinstance(safety_lag_raw, bool) or not isinstance(safety_lag_raw, (int, float)):
        safety_lag_days: int | None = None
    else:
        safety_lag_days = int(safety_lag_raw)

    provenance_raw = payload.get("provenance")
    provenance: dict[str, Any] | None
    if isinstance(provenance_raw, dict):
        provenance = dict(provenance_raw)
    else:
        provenance = None

    return DraftModelSource(
        name=candidate_name,
        scope=_infer_scope(candidate_path),
        path=candidate_path,
        candidate_json_sha256=sha256_text(raw_text),
        case_spec_sha256=case_spec_sha256,
        feature_contract_sha256=feature_contract_sha256,
        contract_version=str(payload.get("contract_version") or "").strip(),
        implementation_status=str(payload.get("implementation_status") or "").strip(),
        implementation_type=str(payload.get("implementation_type") or "").strip(),
        factor_name=factor_name,
        archive_identity=str(
            payload.get("archive_identity")
            or case_spec_payload.get("archive_identity")
            or ""
        ).strip(),
        feature_columns=feature_columns,
        feature_availability_mode=(
            str(feature_availability.get("mode")).strip()
            if isinstance(feature_availability.get("mode"), str)
            else None
        ),
        feature_availability_column=(
            str(feature_availability.get("column")).strip()
            if isinstance(feature_availability.get("column"), str)
            and feature_availability.get("column")
            else None
        ),
        feature_availability_safety_lag_days=safety_lag_days,
        model_family=model_family,
        description=str(payload.get("human_summary") or payload.get("description") or "").strip(),
        provenance=provenance,
    )


def model_candidate_write_path(workspace_root: str | Path, name: str) -> Path:
    """Return the canonical research-scope candidate file path."""

    normalized = name.strip().lower()
    return (
        Path(workspace_root).expanduser().resolve()
        / "custom_models"
        / "research"
        / normalized
        / "model_candidate.json"
    )


def _infer_scope(path: Path) -> str:
    parts = tuple(part.lower() for part in path.parts)
    for scope in MODEL_CANDIDATE_SCOPES:
        if scope in parts:
            return scope
    return "research"
