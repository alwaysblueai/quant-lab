"""Smoke coverage for :mod:`alpha_lab.custom_models`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_lab.custom_models import (
    MODEL_CANDIDATE_SCOPES,
    DraftModelSource,
    model_candidate_write_path,
    read_draft_model_source,
)


def _write_minimal_candidate(path: Path, *, name: str = "smoke_candidate") -> None:
    payload = {
        "candidate_name": name,
        "contract_version": "stage2_model_candidate_v1",
        "implementation_status": "draft_for_stage3",
        "implementation_type": "spec_variant",
        "case_spec_payload": {
            "factor_name": name,
            "feature_columns": ["pe_ttm", "turnover_rate"],
            "feature_availability": {
                "mode": "required_timestamp",
                "column": "known_at",
            },
            "model": {"family": "ridge"},
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_scopes_are_promoted_and_research() -> None:
    assert set(MODEL_CANDIDATE_SCOPES) == {"promoted", "research"}


def test_read_draft_model_source_round_trip(tmp_path: Path) -> None:
    candidate_dir = tmp_path / "research" / "smoke_candidate"
    candidate_dir.mkdir(parents=True)
    json_path = candidate_dir / "model_candidate.json"
    _write_minimal_candidate(json_path)

    source = read_draft_model_source(json_path)

    assert isinstance(source, DraftModelSource)
    assert source.name == "smoke_candidate"
    assert source.scope == "research"
    assert source.contract_version == "stage2_model_candidate_v1"
    assert source.implementation_type == "spec_variant"
    assert source.factor_name == "smoke_candidate"
    assert source.feature_columns == ("pe_ttm", "turnover_rate")
    assert source.feature_availability_mode == "required_timestamp"
    assert source.feature_availability_column == "known_at"
    assert len(source.candidate_json_sha256) == 64
    assert len(source.case_spec_sha256) == 64


def test_to_audit_dict_includes_required_audit_fields(tmp_path: Path) -> None:
    json_path = tmp_path / "research" / "audit_candidate" / "model_candidate.json"
    json_path.parent.mkdir(parents=True)
    _write_minimal_candidate(json_path, name="audit_candidate")
    source = read_draft_model_source(json_path)

    audit = source.to_audit_dict()
    for key in (
        "name",
        "scope",
        "path",
        "candidate_json_sha256",
        "case_spec_sha256",
        "feature_contract_sha256",
        "contract_version",
        "implementation_status",
        "implementation_type",
        "factor_name",
    ):
        assert key in audit, f"audit dict missing {key}"


def test_model_candidate_write_path_in_research_dir(tmp_path: Path) -> None:
    target = model_candidate_write_path(tmp_path, "my_candidate")
    assert target.name == "model_candidate.json"
    assert target.parent.name == "my_candidate"
    assert "research" in target.parts


def test_read_draft_model_source_rejects_missing_candidate_name(tmp_path: Path) -> None:
    path = tmp_path / "model_candidate.json"
    path.write_text(json.dumps({"contract_version": "x"}), encoding="utf-8")
    with pytest.raises(ValueError, match="candidate_name"):
        read_draft_model_source(path)
