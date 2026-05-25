from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import yaml

from alpha_lab.cli import main
from alpha_lab.draft_model_validation import validate_draft_model_file
from alpha_lab.real_cases.model_factor.pipeline import run_model_factor_case
from tests.model_factor_case_helpers import write_demo_model_factor_case


def _legal_payload(spec_path: Path, *, candidate_name: str) -> dict[str, object]:
    """Build a minimal legal Stage2 model_candidate_payload from a demo case."""

    case = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    assert isinstance(case, Mapping)
    case = copy.deepcopy(dict(case))
    case["name"] = f"{candidate_name}_case"
    case["factor_name"] = candidate_name
    case.setdefault("feature_availability", {"mode": "required_timestamp", "column": "known_at"})
    return {
        "contract_version": "stage2_model_candidate_v1",
        "candidate_name": candidate_name,
        "implementation_status": "draft_for_stage3",
        "implementation_type": "spec_variant",
        "source_mechanisms": ["test_mechanism"],
        "base_case_spec_path": str(spec_path),
        "expected_horizon": "t_plus_1_or_later",
        "data_contract": {
            "prices_required_columns": ["date", "asset", "close"],
            "feature_required_columns": list(case["feature_columns"]),
            "feature_optional_columns": [],
            "feature_availability": case["feature_availability"],
        },
        "risk_controls": {
            "feature_availability_pit": "known_at column drives PIT",
            "label_leakage": "forward_return horizon=5",
            "overfit_complexity": "ridge with alpha=1",
            "turnover_cost": "one_way_rate=0.001",
            "feature_instability": "baseline check via rolling stability",
            "split_regime_fragility": "purged kfold default",
        },
        "run_controls": {
            "evaluation_profile": "exploratory_screening",
            "screening_retrain_every_n_dates": 40,
            "vault_export_mode": "skip",
        },
        "case_spec_payload": case,
        "stage3_validation_focus": ["feature_coverage", "rank_ic"],
    }


def _write_candidate(
    tmp_path: Path,
    candidate_name: str,
    payload: Mapping[str, object],
) -> Path:
    candidate_dir = tmp_path / "custom_models" / "research" / candidate_name
    candidate_dir.mkdir(parents=True)
    candidate_path = candidate_dir / "model_candidate.json"
    candidate_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return candidate_path


def test_validate_draft_model_accepts_legal_candidate(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="legal_alpha")
    payload = _legal_payload(spec_path, candidate_name="legal_alpha")
    candidate_path = _write_candidate(tmp_path, "legal_alpha", payload)

    result = validate_draft_model_file(candidate_path)

    assert result.ok, [issue.__dict__ for issue in result.errors]
    assert result.candidate_name == "legal_alpha"
    assert result.candidate_json_sha256
    assert result.case_spec_sha256
    assert result.feature_contract_sha256
    assert result.spec is not None
    assert result.spec.factor_name == "legal_alpha"


def test_validate_draft_model_rejects_missing_feature_column(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="missing_col")
    payload = _legal_payload(spec_path, candidate_name="missing_col")
    case = dict(payload["case_spec_payload"])  # type: ignore[arg-type]
    case["feature_columns"] = list(case["feature_columns"]) + ["definitely_missing"]
    payload["case_spec_payload"] = case
    candidate_path = _write_candidate(tmp_path, "missing_col", payload)

    result = validate_draft_model_file(candidate_path)

    assert not result.ok
    assert any(
        issue.code == "feature_columns_not_in_features" for issue in result.errors
    )


def test_validate_draft_model_rejects_unsupported_model_family(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="bad_family")
    payload = _legal_payload(spec_path, candidate_name="bad_family")
    case = dict(payload["case_spec_payload"])  # type: ignore[arg-type]
    case["model"] = {"family": "transformer", "params": {}}
    payload["case_spec_payload"] = case
    candidate_path = _write_candidate(tmp_path, "bad_family", payload)

    result = validate_draft_model_file(candidate_path)

    assert not result.ok
    assert any(issue.code == "case_spec_invalid" for issue in result.errors)


def test_validate_draft_model_rejects_missing_pit_for_fundamental_feature(
    tmp_path: Path,
) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="fundamental")
    payload = _legal_payload(spec_path, candidate_name="fundamental")
    case = dict(payload["case_spec_payload"])  # type: ignore[arg-type]
    # Substitute a synthetic fundamental column the heuristic will flag
    base_features = list(case["feature_columns"])
    fundamental_column = "feature_roe_proxy"
    case["feature_columns"] = base_features[:1] + [fundamental_column]
    case["feature_availability"] = {"mode": "required_timestamp"}

    # Append the fundamental-named column to the features file so the header
    # check passes and we genuinely exercise the PIT rule.
    import pandas as pd

    features_path = Path(case["features_path"])  # type: ignore[arg-type]
    features = pd.read_csv(features_path)
    features[fundamental_column] = features[base_features[0]]
    features.to_csv(features_path, index=False)

    payload["case_spec_payload"] = case
    candidate_path = _write_candidate(tmp_path, "fundamental", payload)

    result = validate_draft_model_file(candidate_path)

    assert not result.ok
    assert any(
        issue.code == "fundamental_feature_pit_unconfigured"
        for issue in result.errors
    )


def test_validate_draft_model_rejects_non_research_path(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="off_path")
    payload = _legal_payload(spec_path, candidate_name="off_path")
    misplaced_path = tmp_path / "elsewhere" / "model_candidate.json"
    misplaced_path.parent.mkdir(parents=True)
    misplaced_path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )

    result = validate_draft_model_file(misplaced_path)

    assert not result.ok
    assert any(issue.code == "path_scope" for issue in result.errors)


def test_validate_draft_model_rejects_level3_token(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="lvl3_leak")
    payload = _legal_payload(spec_path, candidate_name="lvl3_leak")
    payload["risk_controls"] = {
        **dict(payload["risk_controls"]),  # type: ignore[arg-type]
        "level3": "fill_simulation hooks added",
    }
    candidate_path = _write_candidate(tmp_path, "lvl3_leak", payload)

    result = validate_draft_model_file(candidate_path)

    assert not result.ok
    assert any(issue.code == "forbidden_level3_token" for issue in result.errors)


def test_validate_draft_model_cli_route(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="cli_route")
    payload = _legal_payload(spec_path, candidate_name="cli_route")
    candidate_path = _write_candidate(tmp_path, "cli_route", payload)

    rc = main(
        [
            "validate-draft-model",
            str(candidate_path),
            "--json",
        ]
    )

    assert rc == 0


def test_validate_draft_model_cli_fails_missing_required_field(
    tmp_path: Path,
) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="cli_missing_field")
    payload = _legal_payload(spec_path, candidate_name="cli_missing_field")
    candidate_path = _write_candidate(tmp_path, "cli_missing_field", payload)

    rc = main(
        [
            "validate-draft-model",
            str(candidate_path),
            "--available-fields",
            "date,asset",
        ]
    )

    assert rc == 1


def test_run_model_factor_case_writes_draft_model_source(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="audit_alpha")
    payload = _legal_payload(spec_path, candidate_name="audit_alpha")
    candidate_path = _write_candidate(tmp_path, "audit_alpha", payload)

    from alpha_lab.custom_models import read_draft_model_source

    draft_model_source = read_draft_model_source(candidate_path)
    result = run_model_factor_case(
        spec_path,
        draft_model_source=draft_model_source,
        evaluation_profile="exploratory_screening",
        screening_retrain_every_n_dates=40,
    )

    assert result.draft_model_source is not None
    assert result.draft_model_source.name == "audit_alpha"

    manifest = json.loads(
        (result.output_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    model_def = json.loads(
        (result.output_dir / "model_definition.json").read_text(encoding="utf-8")
    )
    feature_manifest = json.loads(
        (result.output_dir / "feature_manifest.json").read_text(encoding="utf-8")
    )

    for blob, label in (
        (manifest, "run_manifest"),
        (model_def, "model_definition"),
        (feature_manifest, "feature_manifest"),
    ):
        audit = blob.get("draft_model_source")
        assert isinstance(audit, dict), f"{label} missing draft_model_source"
        assert audit["candidate_json_sha256"]
        assert audit["case_spec_sha256"]
        assert audit["feature_contract_sha256"]
        assert audit["path"]
        assert audit["contract_version"] == "stage2_model_candidate_v1"


# ---------------------------------------------------------------------------
# provenance block (Stage 0 idea_id passthrough)
# ---------------------------------------------------------------------------


def test_validate_draft_model_warns_when_provenance_missing(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="prov_missing")
    payload = _legal_payload(spec_path, candidate_name="prov_missing")
    candidate_path = _write_candidate(tmp_path, "prov_missing", payload)
    result = validate_draft_model_file(candidate_path)
    assert result.ok
    assert any(w.code == "provenance_missing" for w in result.warnings)


def test_validate_draft_model_accepts_complete_provenance(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="prov_full")
    payload = dict(_legal_payload(spec_path, candidate_name="prov_full"))
    payload["provenance"] = {
        "idea_id": "20260511T140000Z__turnover-conditioned",
        "stage2_payload_sha256": "b" * 64,
        "audience_chain": ["claude", "codex", "web_gpt_stage2"],
    }
    candidate_path = _write_candidate(tmp_path, "prov_full", payload)
    result = validate_draft_model_file(candidate_path)
    assert result.ok
    assert not any(w.code == "provenance_missing" for w in result.warnings)


def test_validate_draft_model_rejects_provenance_without_idea_id(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="prov_noid")
    payload = dict(_legal_payload(spec_path, candidate_name="prov_noid"))
    payload["provenance"] = {"stage2_payload_sha256": "c" * 64}
    candidate_path = _write_candidate(tmp_path, "prov_noid", payload)
    result = validate_draft_model_file(candidate_path)
    assert not result.ok
    assert any(e.code == "provenance_idea_id_missing" for e in result.errors)


def test_draft_model_source_passes_provenance_to_audit(tmp_path: Path) -> None:
    from alpha_lab.custom_models import read_draft_model_source

    spec_path = write_demo_model_factor_case(tmp_path, factor_name="prov_audit")
    payload = dict(_legal_payload(spec_path, candidate_name="prov_audit"))
    payload["provenance"] = {
        "idea_id": "20260511T150000Z__audited",
        "audience_chain": ["claude", "codex", "web_gpt_stage2"],
    }
    candidate_path = _write_candidate(tmp_path, "prov_audit", payload)
    source = read_draft_model_source(candidate_path)
    audit = source.to_audit_dict()
    assert "provenance" in audit
    assert audit["provenance"]["idea_id"] == "20260511T150000Z__audited"


# ---------------------------------------------------------------------------
# Stage2 -> Stage3 materialization seam
#
# The Stage2 payload carries a *placeholder* provenance.stage2_payload_sha256
# (Web GPT cannot self-reference the canonical hash of the payload it is still
# writing). The Stage3 materialize step must replace it with a real 64-char hex
# sha. The convention is documented in docs/backend_draft_model_workflow.md and
# is the single reference implementation below: blank the field, canonical-dump
# the model_candidate_payload (sort_keys + ensure_ascii=False), sha256-hex it.
# These tests pin the convention so every consumer (Codex GUI / Web / Claude)
# produces the same, validator-accepted hash.
# ---------------------------------------------------------------------------

_STAGE2_PLACEHOLDER_SHA = "PENDING_STAGE3_MATERIALIZE"


def _materialize_stage2_payload_sha256(model_candidate_payload: Mapping[str, object]) -> str:
    """Reference impl of the documented placeholder -> real sha convention."""

    canonical = copy.deepcopy(dict(model_candidate_payload))
    provenance = dict(canonical.get("provenance") or {})
    provenance["stage2_payload_sha256"] = ""
    canonical["provenance"] = provenance
    blob = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def test_stage2_payload_sha_materialization_is_deterministic_and_hex() -> None:
    spec_path_payload = {
        "contract_version": "stage2_model_candidate_v1",
        "candidate_name": "seam_demo",
        "provenance": {
            "idea_id": "20260525T115846Z__idea-demo",
            "stage2_payload_sha256": _STAGE2_PLACEHOLDER_SHA,
            "audience_chain": ["claude", "codex", "web_gpt_stage2"],
        },
        "case_spec_payload": {"feature_columns": ["turnover_rate"]},
    }

    sha_a = _materialize_stage2_payload_sha256(spec_path_payload)
    sha_b = _materialize_stage2_payload_sha256(spec_path_payload)
    assert sha_a == sha_b, "convention must be deterministic"
    assert len(sha_a) == 64 and all(c in "0123456789abcdef" for c in sha_a)

    # A content change (not the sha field) must change the materialized sha.
    mutated = copy.deepcopy(spec_path_payload)
    mutated["candidate_name"] = "seam_demo_v2"
    assert _materialize_stage2_payload_sha256(mutated) != sha_a


def test_materialized_candidate_passes_validator_but_placeholder_fails(
    tmp_path: Path,
) -> None:
    """The Stage2->Stage3 seam: a placeholder sha must be rejected, and the
    documented materialization must produce a candidate the validator accepts
    with all three audit hashes."""

    spec_path = write_demo_model_factor_case(tmp_path, factor_name="seam_real")
    payload = dict(_legal_payload(spec_path, candidate_name="seam_real"))
    payload["provenance"] = {
        "idea_id": "20260525T115846Z__idea-seam",
        "stage2_payload_sha256": _STAGE2_PLACEHOLDER_SHA,
        "audience_chain": ["claude", "codex", "web_gpt_stage2"],
    }

    # 1) Placeholder (non-hex) must be rejected -> proves Stage3 *must* materialize.
    placeholder_path = _write_candidate(tmp_path, "seam_real", payload)
    placeholder_result = validate_draft_model_file(placeholder_path)
    assert not placeholder_result.ok
    assert any(
        e.code == "provenance_payload_sha256" for e in placeholder_result.errors
    )

    # 2) Materialize the real sha via the documented convention, rewrite, re-validate.
    real_sha = _materialize_stage2_payload_sha256(payload)
    materialized = copy.deepcopy(payload)
    materialized["provenance"]["stage2_payload_sha256"] = real_sha  # type: ignore[index]
    placeholder_path.write_text(
        json.dumps(materialized, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result = validate_draft_model_file(placeholder_path)

    assert result.ok, [issue.__dict__ for issue in result.errors]
    assert result.candidate_json_sha256
    assert result.case_spec_sha256
    assert result.feature_contract_sha256
    assert result.spec is not None
