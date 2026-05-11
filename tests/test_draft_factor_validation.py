from __future__ import annotations

import json
from pathlib import Path

from alpha_lab.cli import main
from alpha_lab.custom_factors import sha256_text
from alpha_lab.draft_factor_validation import validate_draft_factor_file

VALID_CODE = """
def build_factor(frame):
    import numpy as np
    import pandas as pd

    required = {"close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["asset", "date"], kind="mergesort")
    close_ret = data.groupby("asset", sort=False)["close"].pct_change()
    volume_ret = data.groupby("asset", sort=False)["volume"].pct_change()
    value = (close_ret - volume_ret).replace([np.inf, -np.inf], np.nan)
    return value.reindex(frame.index)
""".strip()


def test_validate_draft_factor_accepts_build_factor_contract(tmp_path: Path) -> None:
    factor_json = _write_factor_json(tmp_path, "volume_proxy", VALID_CODE)

    result = validate_draft_factor_file(
        factor_json,
        available_fields={"date", "asset", "close", "volume"},
    )

    assert result.ok
    assert result.factor_name == "volume_proxy"
    assert result.code_sha256 == sha256_text(VALID_CODE)
    assert result.factor_json_sha256 == sha256_text(factor_json.read_text(encoding="utf-8"))


def test_validate_draft_factor_rejects_file_io(tmp_path: Path) -> None:
    bad_code = """
def build_factor(frame):
    required = {"close"}
    open("/tmp/leak.txt", "w").write("x")
    return frame["close"]
""".strip()
    factor_json = _write_factor_json(tmp_path, "bad_file_io", bad_code, required=["close"])

    result = validate_draft_factor_file(factor_json)

    assert not result.ok
    assert any(issue.code == "forbidden_call" for issue in result.errors)


def test_validate_draft_factor_cli_route(tmp_path: Path) -> None:
    factor_json = _write_factor_json(tmp_path, "cli_volume_proxy", VALID_CODE)

    rc = main(
        [
            "validate-draft-factor",
            str(factor_json),
            "--available-fields",
            "date,asset,close,volume",
            "--json",
        ]
    )

    assert rc == 0


def test_validate_draft_factor_cli_fails_missing_required_field(tmp_path: Path) -> None:
    factor_json = _write_factor_json(tmp_path, "missing_field_proxy", VALID_CODE)

    rc = main(
        [
            "validate-draft-factor",
            str(factor_json),
            "--available-fields",
            "date,asset,close",
        ]
    )

    assert rc == 1


def _write_factor_json(
    tmp_path: Path,
    name: str,
    code: str,
    *,
    required: list[str] | None = None,
    provenance: dict | None = None,
) -> Path:
    factor_dir = tmp_path / "custom_factors" / "research" / name
    factor_dir.mkdir(parents=True)
    factor_json = factor_dir / "factor.json"
    payload: dict = {
        "name": name,
        "description": "daily proxy draft factor",
        "required_columns": required or ["close", "volume"],
        "optional_columns": [],
        "frequency": "daily",
        "unavailable_data_policy": "return_nan",
        "pit_assumption": "uses current and historical daily bars only",
        "code": code,
    }
    if provenance is not None:
        payload["provenance"] = provenance
    factor_json.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )
    return factor_json


# ---------------------------------------------------------------------------
# provenance block (Stage 0 idea_id passthrough)
# ---------------------------------------------------------------------------


def test_validate_draft_factor_warns_when_provenance_missing(tmp_path: Path) -> None:
    factor_json = _write_factor_json(tmp_path, "no_provenance", VALID_CODE)
    result = validate_draft_factor_file(
        factor_json, available_fields={"date", "asset", "close", "volume"}
    )
    assert result.ok
    warning_codes = {w.code for w in result.warnings}
    assert "provenance_missing" in warning_codes


def test_validate_draft_factor_accepts_complete_provenance(tmp_path: Path) -> None:
    factor_json = _write_factor_json(
        tmp_path,
        "with_provenance",
        VALID_CODE,
        provenance={
            "idea_id": "20260511T120000Z__signed-jump",
            "stage2_payload_sha256": "a" * 64,
            "audience_chain": ["claude_mechanism", "codex_review", "web_gpt_stage2"],
        },
    )
    result = validate_draft_factor_file(
        factor_json, available_fields={"date", "asset", "close", "volume"}
    )
    assert result.ok
    assert not any(w.code == "provenance_missing" for w in result.warnings)


def test_validate_draft_factor_rejects_provenance_without_idea_id(tmp_path: Path) -> None:
    factor_json = _write_factor_json(
        tmp_path,
        "bad_provenance",
        VALID_CODE,
        provenance={"stage2_payload_sha256": "a" * 64},
    )
    result = validate_draft_factor_file(
        factor_json, available_fields={"date", "asset", "close", "volume"}
    )
    assert not result.ok
    assert any(e.code == "provenance_idea_id_missing" for e in result.errors)


def test_validate_draft_factor_rejects_provenance_bad_sha(tmp_path: Path) -> None:
    factor_json = _write_factor_json(
        tmp_path,
        "bad_sha",
        VALID_CODE,
        provenance={"idea_id": "x", "stage2_payload_sha256": "not_hex"},
    )
    result = validate_draft_factor_file(
        factor_json, available_fields={"date", "asset", "close", "volume"}
    )
    assert not result.ok
    assert any(e.code == "provenance_payload_sha256" for e in result.errors)


def test_custom_factor_source_passes_provenance_to_audit(tmp_path: Path) -> None:
    from alpha_lab.custom_factors import read_custom_factor_source

    factor_json = _write_factor_json(
        tmp_path,
        "audited",
        VALID_CODE,
        provenance={
            "idea_id": "20260511T130000Z__audited",
            "audience_chain": ["claude_mechanism", "codex_review", "web_gpt_stage2"],
        },
    )
    source = read_custom_factor_source(factor_json)
    audit = source.to_audit_dict()
    assert "provenance" in audit
    assert audit["provenance"]["idea_id"] == "20260511T130000Z__audited"
