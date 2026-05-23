"""Stage 3 backend draft-model validator.

Mirrors :mod:`alpha_lab.draft_factor_validation` but the audited artifact is a
``model_candidate.json`` whose ``case_spec_payload`` is a complete model-factor
case spec.  The validator is read-only: it parses the payload, builds a typed
``ModelFactorCaseSpec`` via the existing model-factor spec parser, and reports
audit hashes.  It never writes case YAML and never trains a model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from alpha_lab.custom_factors import sha256_text
from alpha_lab.real_cases.model_factor.spec import (
    ModelFactorCaseSpec,
    infer_fundamental_feature_columns,
    model_factor_case_spec_from_mapping,
)

_CANDIDATE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_SUPPORTED_CONTRACT_VERSIONS: frozenset[str] = frozenset(
    {"stage2_model_candidate_v1"}
)
_SUPPORTED_IMPLEMENTATION_TYPES: frozenset[str] = frozenset({"spec_variant"})
_SUPPORTED_IMPLEMENTATION_STATUS: frozenset[str] = frozenset({"draft_for_stage3"})

_REQUIRED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "contract_version",
        "candidate_name",
        "implementation_status",
        "implementation_type",
        "case_spec_payload",
    }
)

_FORBIDDEN_LEVEL3_TOKENS: tuple[str, ...] = (
    "execution_replay",
    "fill_simulation",
    "adapter_parity",
    "live_trading",
    "portfolio_construction",
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_KNOWN_AUDIENCES: frozenset[str] = frozenset(
    {"claude", "codex", "web_gpt_stage2"}
)


@dataclass(frozen=True)
class DraftModelIssue:
    """One validation finding."""

    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class DraftModelValidationResult:
    """Outcome of validating a single ``model_candidate.json``."""

    path: Path
    candidate_name: str | None
    candidate_json_sha256: str | None
    case_spec_sha256: str | None
    feature_contract_sha256: str | None
    spec: ModelFactorCaseSpec | None = None
    errors: tuple[DraftModelIssue, ...] = field(default_factory=tuple)
    warnings: tuple[DraftModelIssue, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_payload(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "path": str(self.path),
            "candidate_name": self.candidate_name,
            "candidate_json_sha256": self.candidate_json_sha256,
            "case_spec_sha256": self.case_spec_sha256,
            "feature_contract_sha256": self.feature_contract_sha256,
            "errors": [issue.__dict__ for issue in self.errors],
            "warnings": [issue.__dict__ for issue in self.warnings],
        }


def validate_draft_model_file(
    path: str | Path,
    *,
    available_fields: set[str] | None = None,
    allow_non_research_path: bool = False,
    require_features_file: bool = True,
) -> DraftModelValidationResult:
    """Validate a Stage3 backend draft model candidate file."""

    candidate_path = Path(path).expanduser().resolve()
    errors: list[DraftModelIssue] = []
    warnings: list[DraftModelIssue] = []
    candidate_name: str | None = None
    candidate_json_sha256: str | None = None
    case_spec_sha256: str | None = None
    feature_contract_sha256: str | None = None
    spec_obj: ModelFactorCaseSpec | None = None

    try:
        raw_text = candidate_path.read_text(encoding="utf-8")
    except OSError as exc:
        return _result(
            candidate_path,
            candidate_name,
            candidate_json_sha256,
            case_spec_sha256,
            feature_contract_sha256,
            spec_obj,
            errors=[_error("file_read", f"cannot read model_candidate.json: {exc}")],
            warnings=warnings,
        )

    candidate_json_sha256 = sha256_text(raw_text)
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return _result(
            candidate_path,
            candidate_name,
            candidate_json_sha256,
            case_spec_sha256,
            feature_contract_sha256,
            spec_obj,
            errors=[
                _error(
                    "json_parse",
                    f"model_candidate.json is not valid JSON: {exc}",
                )
            ],
            warnings=warnings,
        )
    if not isinstance(payload, dict):
        return _result(
            candidate_path,
            candidate_name,
            candidate_json_sha256,
            case_spec_sha256,
            feature_contract_sha256,
            spec_obj,
            errors=[
                _error("json_schema", "model_candidate.json root must be an object")
            ],
            warnings=warnings,
        )

    missing = sorted(_REQUIRED_TOP_LEVEL_KEYS.difference(payload))
    if missing:
        errors.append(
            _error("missing_keys", f"model_candidate.json missing keys: {missing}")
        )

    contract_version = _optional_text(payload.get("contract_version"))
    if contract_version is None:
        errors.append(
            _error("contract_version_missing", "contract_version must be non-empty")
        )
    elif contract_version not in _SUPPORTED_CONTRACT_VERSIONS:
        errors.append(
            _error(
                "contract_version_unsupported",
                "contract_version must be one of "
                f"{sorted(_SUPPORTED_CONTRACT_VERSIONS)}; got {contract_version!r}",
            )
        )

    implementation_status = _optional_text(payload.get("implementation_status"))
    if implementation_status is None:
        errors.append(
            _error(
                "implementation_status_missing",
                "implementation_status must be non-empty",
            )
        )
    elif implementation_status not in _SUPPORTED_IMPLEMENTATION_STATUS:
        errors.append(
            _error(
                "implementation_status_unsupported",
                "implementation_status must be one of "
                f"{sorted(_SUPPORTED_IMPLEMENTATION_STATUS)}; got "
                f"{implementation_status!r}",
            )
        )

    implementation_type = _optional_text(payload.get("implementation_type"))
    if implementation_type is None:
        errors.append(
            _error(
                "implementation_type_missing",
                "implementation_type must be non-empty",
            )
        )
    elif implementation_type not in _SUPPORTED_IMPLEMENTATION_TYPES:
        errors.append(
            _error(
                "implementation_type_unsupported",
                "v1 only supports spec-variant draft models; got "
                f"{implementation_type!r}. Custom feature builders or estimator "
                "code are out of scope.",
            )
        )

    candidate_name = _optional_text(payload.get("candidate_name"))
    if candidate_name is None:
        errors.append(
            _error("candidate_name_missing", "candidate_name must be a non-empty string")
        )
    elif not _CANDIDATE_NAME_RE.match(candidate_name):
        errors.append(
            _error(
                "candidate_name_format",
                "candidate_name must be snake_case, start with a letter, and use "
                "3-64 lowercase chars",
            )
        )

    if candidate_name and not allow_non_research_path:
        expected_tail = (
            "custom_models",
            "research",
            candidate_name,
            "model_candidate.json",
        )
        if not _path_has_tail(candidate_path, expected_tail):
            errors.append(
                _error(
                    "path_scope",
                    "draft model must live at "
                    f"custom_models/research/{candidate_name}/model_candidate.json",
                )
            )

    _check_no_level3_tokens(payload, errors)
    _validate_provenance(
        payload.get("provenance"), errors=errors, warnings=warnings
    )

    case_spec_payload = payload.get("case_spec_payload")
    if not isinstance(case_spec_payload, Mapping):
        errors.append(
            _error(
                "case_spec_payload_missing",
                "case_spec_payload must be a complete model-factor case spec object",
            )
        )
    else:
        case_spec_sha256 = sha256_text(
            json.dumps(case_spec_payload, ensure_ascii=False, sort_keys=True)
        )
        feature_contract_sha256 = _feature_contract_hash(case_spec_payload)
        try:
            spec_obj = model_factor_case_spec_from_mapping(case_spec_payload)
        except (ValueError, TypeError) as exc:
            errors.append(
                _error("case_spec_invalid", f"case_spec_payload is invalid: {exc}")
            )
        else:
            _validate_candidate_alignment(
                spec=spec_obj,
                payload=payload,
                candidate_name=candidate_name,
                errors=errors,
                warnings=warnings,
            )
            _validate_feature_columns_against_file(
                spec=spec_obj,
                candidate_path=candidate_path,
                require_features_file=require_features_file,
                errors=errors,
                warnings=warnings,
            )
            _validate_available_fields(
                spec=spec_obj,
                available_fields=available_fields,
                errors=errors,
            )
            _validate_pit_contract(spec=spec_obj, errors=errors, warnings=warnings)

    return _result(
        candidate_path,
        candidate_name,
        candidate_json_sha256,
        case_spec_sha256,
        feature_contract_sha256,
        spec_obj,
        errors=errors,
        warnings=warnings,
    )


def _validate_provenance(
    raw: object,
    *,
    errors: list[DraftModelIssue],
    warnings: list[DraftModelIssue],
) -> None:
    """Shape-check the optional ``provenance`` block on a model candidate.

    Mirrors the single-factor validator: absent block → warning (back-compat);
    present block requires non-empty ``idea_id`` and hex sha256 if
    ``stage2_payload_sha256`` is given.
    """

    if raw is None:
        warnings.append(
            _warning(
                "provenance_missing",
                "provenance block missing; new candidates should record idea_id from "
                "alpha-lab model-idea distribute output",
            )
        )
        return
    if not isinstance(raw, Mapping):
        errors.append(_error("provenance_type", "provenance must be a JSON object"))
        return
    idea_id = _optional_text(raw.get("idea_id"))
    if idea_id is None:
        errors.append(
            _error("provenance_idea_id_missing", "provenance.idea_id must be non-empty")
        )
    payload_sha = raw.get("stage2_payload_sha256")
    if payload_sha is not None:
        if not isinstance(payload_sha, str) or not _SHA256_RE.match(payload_sha):
            errors.append(
                _error(
                    "provenance_payload_sha256",
                    "provenance.stage2_payload_sha256 must be a 64-char hex sha256",
                )
            )
    audience_chain = raw.get("audience_chain")
    if audience_chain is not None:
        if not isinstance(audience_chain, list) or not all(
            isinstance(item, str) for item in audience_chain
        ):
            errors.append(
                _error(
                    "provenance_audience_chain_type",
                    "provenance.audience_chain must be a string list",
                )
            )
        else:
            unknown = [item for item in audience_chain if item not in _KNOWN_AUDIENCES]
            if unknown:
                warnings.append(
                    _warning(
                        "provenance_audience_unknown",
                        f"provenance.audience_chain contains unknown audiences: {unknown}",
                    )
                )


def _check_no_level3_tokens(
    payload: Mapping[str, object],
    errors: list[DraftModelIssue],
) -> None:
    blob = json.dumps(payload, ensure_ascii=False).lower()
    for token in _FORBIDDEN_LEVEL3_TOKENS:
        if token in blob:
            errors.append(
                _error(
                    "forbidden_level3_token",
                    f"forbidden Level 3 / portfolio-construction token in payload: "
                    f"{token!r}",
                )
            )


def _validate_candidate_alignment(
    *,
    spec: ModelFactorCaseSpec,
    payload: Mapping[str, object],
    candidate_name: str | None,
    errors: list[DraftModelIssue],
    warnings: list[DraftModelIssue],
) -> None:
    if candidate_name and spec.factor_name.strip() != candidate_name:
        warnings.append(
            _warning(
                "factor_name_mismatch",
                f"case_spec_payload.factor_name ({spec.factor_name!r}) does not match "
                f"candidate_name ({candidate_name!r}); reviewers should reconcile.",
            )
        )
    expected_horizon = _optional_text(payload.get("expected_horizon"))
    if expected_horizon is not None and expected_horizon == "t_plus_1_or_later":
        if spec.target.horizon < 1:
            errors.append(
                _error(
                    "target_horizon_invalid",
                    "expected_horizon='t_plus_1_or_later' but target.horizon < 1",
                )
            )


def _validate_feature_columns_against_file(
    *,
    spec: ModelFactorCaseSpec,
    candidate_path: Path,
    require_features_file: bool,
    errors: list[DraftModelIssue],
    warnings: list[DraftModelIssue],
) -> None:
    features_raw = spec.features_path
    base_dir = candidate_path.parent
    candidate = (
        Path(features_raw).expanduser()
        if features_raw
        else None
    )
    if candidate is None:
        return
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    if not candidate.exists():
        if require_features_file:
            errors.append(
                _error(
                    "features_file_missing",
                    f"features_path does not exist: {candidate}",
                )
            )
        else:
            warnings.append(
                _warning(
                    "features_file_skipped",
                    f"features_path not present, header check skipped: {candidate}",
                )
            )
        return
    try:
        header = _read_table_header(candidate)
    except Exception as exc:  # noqa: BLE001
        errors.append(
            _error(
                "features_header_read_failed",
                f"failed to read header of {candidate}: {exc}",
            )
        )
        return
    missing_cols = [col for col in spec.feature_columns if col not in header]
    if missing_cols:
        errors.append(
            _error(
                "feature_columns_not_in_features",
                f"feature_columns missing from features file header: {missing_cols}",
            )
        )
    if "date" not in header or "asset" not in header:
        errors.append(
            _error(
                "features_schema_invalid",
                "features file must contain 'date' and 'asset' columns",
            )
        )
    if (
        spec.feature_availability.mode == "required_timestamp"
        and spec.feature_availability.column is not None
        and spec.feature_availability.column not in header
    ):
        errors.append(
            _error(
                "feature_availability_column_missing",
                "feature_availability.column is not present in features file: "
                f"{spec.feature_availability.column!r}",
            )
        )


def _validate_available_fields(
    *,
    spec: ModelFactorCaseSpec,
    available_fields: set[str] | None,
    errors: list[DraftModelIssue],
) -> None:
    if available_fields is None:
        return
    missing = [col for col in spec.feature_columns if col not in available_fields]
    if missing:
        errors.append(
            _error(
                "feature_columns_unavailable",
                f"feature_columns missing from provided available-fields list: {missing}",
            )
        )


def _validate_pit_contract(
    *,
    spec: ModelFactorCaseSpec,
    errors: list[DraftModelIssue],
    warnings: list[DraftModelIssue],
) -> None:
    fundamental = infer_fundamental_feature_columns(spec.feature_columns)
    fa = spec.feature_availability
    if fundamental and fa.mode == "required_timestamp" and fa.column is None:
        errors.append(
            _error(
                "fundamental_feature_pit_unconfigured",
                "fundamental-like feature columns detected "
                f"({list(fundamental)}); feature_availability.mode="
                "'required_timestamp' requires a known_at column or use "
                "'safety_lag' with safety_lag_days >= 1.",
            )
        )
    if fa.mode == "required_timestamp" and fa.column is None:
        warnings.append(
            _warning(
                "feature_availability_no_known_at",
                "feature_availability.mode='required_timestamp' but no column "
                "specified; ensure features are point-in-time at row date.",
            )
        )


def _read_table_header(path: Path) -> set[str]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        import pandas as pd

        sep = "\t" if suffix == ".tsv" else ","
        head = pd.read_csv(path, sep=sep, nrows=0)
        return {str(col) for col in head.columns}
    if suffix in {".parquet", ".pq"}:
        try:
            import pyarrow.parquet as pq  # type: ignore[import-not-found]

            schema = pq.read_schema(str(path))
            return {str(name) for name in schema.names}
        except Exception:
            import pandas as pd

            head = pd.read_parquet(path)
            return {str(col) for col in head.columns}
    raise ValueError(f"unsupported features file extension: {suffix!r}")


def _feature_contract_hash(case_spec_payload: Mapping[str, object]) -> str:
    """Stable hash of feature-layer contract for cross-run audit."""

    feature_columns_raw = case_spec_payload.get("feature_columns")
    feature_columns: list[Any] = (
        list(feature_columns_raw) if isinstance(feature_columns_raw, list) else []
    )
    feature_availability_raw = case_spec_payload.get("feature_availability")
    feature_availability: dict[str, Any] = (
        dict(feature_availability_raw)
        if isinstance(feature_availability_raw, Mapping)
        else {}
    )
    feature_preprocess_raw = case_spec_payload.get("feature_preprocess")
    feature_preprocess: dict[str, Any] = (
        dict(feature_preprocess_raw)
        if isinstance(feature_preprocess_raw, Mapping)
        else {}
    )
    contract: dict[str, Any] = {
        "feature_columns": feature_columns,
        "feature_availability": feature_availability,
        "feature_preprocess": feature_preprocess,
    }
    return sha256_text(json.dumps(contract, ensure_ascii=False, sort_keys=True))


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _path_has_tail(path: Path, expected_tail: tuple[str, ...]) -> bool:
    normalized_parts = tuple(part.lower() for part in path.parts)
    expected = tuple(part.lower() for part in expected_tail)
    return (
        len(normalized_parts) >= len(expected)
        and normalized_parts[-len(expected):] == expected
    )


def _error(code: str, message: str) -> DraftModelIssue:
    return DraftModelIssue(severity="error", code=code, message=message)


def _warning(code: str, message: str) -> DraftModelIssue:
    return DraftModelIssue(severity="warning", code=code, message=message)


def _result(
    path: Path,
    candidate_name: str | None,
    candidate_json_sha256: str | None,
    case_spec_sha256: str | None,
    feature_contract_sha256: str | None,
    spec: ModelFactorCaseSpec | None,
    *,
    errors: list[DraftModelIssue],
    warnings: list[DraftModelIssue],
) -> DraftModelValidationResult:
    return DraftModelValidationResult(
        path=path,
        candidate_name=candidate_name,
        candidate_json_sha256=candidate_json_sha256,
        case_spec_sha256=case_spec_sha256,
        feature_contract_sha256=feature_contract_sha256,
        spec=spec,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
