from __future__ import annotations

import json
from pathlib import Path

from alpha_lab.real_cases.composite.pipeline import run_composite_case
from tests.composite_case_helpers import write_demo_composite_case


def test_composite_pipeline_records_successful_vault_export(tmp_path: Path) -> None:
    spec_path = write_demo_composite_case(tmp_path)
    vault_root = tmp_path / "quant-knowledge"
    vault_root.mkdir(parents=True, exist_ok=True)

    result = run_composite_case(
        spec_path,
        vault_root=vault_root,
        vault_export_mode="versioned",
    )

    manifest = json.loads(result.artifact_paths["run_manifest"].read_text(encoding="utf-8"))
    vault_meta = manifest["vault_export"]

    assert vault_meta["enabled"] is True
    assert vault_meta["mode"] == "versioned"
    assert vault_meta["status"] == "success"
    assert isinstance(vault_meta["target_paths"], list)
    assert len(vault_meta["target_paths"]) >= 2

    case_dir = vault_root / "50_experiments" / result.spec.name
    assert (case_dir / "latest.md").exists()


def test_composite_pipeline_without_vault_is_skipped(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)
    spec_path = write_demo_composite_case(tmp_path)

    result = run_composite_case(spec_path)

    manifest = json.loads(result.artifact_paths["run_manifest"].read_text(encoding="utf-8"))
    vault_meta = manifest["vault_export"]
    assert vault_meta["enabled"] is False
    assert vault_meta["status"] == "skipped"
    assert vault_meta["error"] is None


def test_composite_pipeline_invalid_vault_does_not_fail_run(tmp_path: Path) -> None:
    spec_path = write_demo_composite_case(tmp_path)
    invalid_vault_root = tmp_path / "not_exists"

    result = run_composite_case(
        spec_path,
        vault_root=invalid_vault_root,
        vault_export_mode="versioned",
    )

    assert result.artifact_paths["metrics"].exists()
    assert result.artifact_paths["run_manifest"].exists()

    manifest = json.loads(result.artifact_paths["run_manifest"].read_text(encoding="utf-8"))
    vault_meta = manifest["vault_export"]
    assert vault_meta["enabled"] is True
    assert vault_meta["status"] == "failed"
    assert vault_meta["error"]
