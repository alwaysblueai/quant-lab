from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from alpha_lab.archive import (
    ArchiveRunIndex,
    apply_archive_draft,
    build_archive_preview,
    migrate_auto_exports,
    write_archive_draft,
)
from alpha_lab.custom_factors import sha256_text
from alpha_lab.exceptions import AlphaLabConfigError


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _split_frontmatter(path: Path) -> tuple[dict[str, object], str]:
    text = path.read_text(encoding="utf-8")
    _, raw, body = text.split("---\n", 2)
    loaded = yaml.safe_load(raw) or {}
    return dict(loaded), body.lstrip("\n")


def _factor_source(
    workspace: Path,
    *,
    name: str,
    archive_identity: str | None = None,
) -> dict[str, object]:
    code = "def build_factor(frame):\n    return frame['close']\n"
    payload: dict[str, object] = {
        "name": name,
        "archive_identity": archive_identity or name,
        "code": code,
    }
    source_path = workspace / "custom_factors" / "research" / name / "factor.json"
    _write_json(source_path, payload)
    return {
        "name": name,
        "archive_identity": archive_identity or name,
        "path": str(source_path),
        "code_sha256": sha256_text(code),
        "factor_json_sha256": sha256_text(
            json.dumps(payload, ensure_ascii=False, indent=2)
        ),
    }


def _single_run(
    workspace: Path,
    *,
    run_id: str,
    factor_name: str,
    archive_identity: str,
    include_manifest: bool = True,
    include_factor_definition: bool = True,
    include_source: bool = True,
) -> Path:
    run_dir = workspace / "outputs" / "real_cases" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "metrics.json", {"metrics": {"factor_name": factor_name}})
    source = (
        _factor_source(workspace, name=factor_name, archive_identity=archive_identity)
        if include_source
        else {}
    )
    if include_factor_definition:
        factor_def: dict[str, object] = {
            "artifact_type": "alpha_lab_factor_definition",
            "case_name": run_id,
            "factor_name": factor_name,
            "archive_identity": archive_identity,
            "spec": {"factor_name": factor_name, "archive_identity": archive_identity},
        }
        if source:
            factor_def["custom_factor_source"] = source
        _write_json(run_dir / "factor_definition.json", factor_def)
    if include_manifest:
        manifest: dict[str, object] = {
            "workflow": "real_case_single_factor",
            "case_name": run_id,
            "run_timestamp_utc": f"2026-05-20T00:00:0{run_id[-1:]}Z",
            "outputs": {
                "metrics": str(run_dir / "metrics.json"),
                "factor_definition_json": str(run_dir / "factor_definition.json"),
            },
            "inputs": {"custom_factor_source": source} if source else {},
        }
        _write_json(run_dir / "run_manifest.json", manifest)
    return run_dir


def test_archive_identity_groups_historical_runs(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    vault = tmp_path / "vault"
    _single_run(
        workspace,
        run_id="signed_jump_neg_5d_v1",
        factor_name="signed_jump_neg_5d_v1",
        archive_identity="signed_jump_neg_5d",
    )
    _single_run(
        workspace,
        run_id="signed_jump_neg_5d_v2_next_open",
        factor_name="signed_jump_neg_5d_v2_next_open",
        archive_identity="signed_jump_neg_5d",
    )

    preview = build_archive_preview(
        index=ArchiveRunIndex.build(workspace_root=workspace),
        vault_root=vault,
        workflow="single_factor",
        run_id="signed_jump_neg_5d_v2_next_open",
    )

    assert preview["identity"]["archive_identity"] == "signed_jump_neg_5d"  # type: ignore[index]
    assert preview["identity"]["archive_identity_inferred"] is False  # type: ignore[index]
    assert len(preview["historical_runs"]) == 2  # type: ignore[arg-type]


def test_legacy_missing_manifest_and_definition_allowed_with_source_hash(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    vault = tmp_path / "vault"
    _single_run(
        workspace,
        run_id="legacy_run",
        factor_name="legacy_factor",
        archive_identity="legacy_factor",
        include_manifest=False,
        include_factor_definition=False,
    )

    preview = build_archive_preview(
        index=ArchiveRunIndex.build(workspace_root=workspace),
        vault_root=vault,
        workflow="single_factor",
        run_id="legacy_run",
    )

    assert preview["legacy_archive_only"] is True
    assert preview["can_draft"] is True
    assert set(preview["audit"]["legacy_missing_artifacts"]) == {  # type: ignore[index]
        "run_manifest_missing",
        "factor_definition_missing",
    }


def test_missing_source_hash_blocks_draft(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    vault = tmp_path / "vault"
    _single_run(
        workspace,
        run_id="no_source",
        factor_name="no_source",
        archive_identity="no_source",
        include_source=False,
    )

    preview = build_archive_preview(
        index=ArchiveRunIndex.build(workspace_root=workspace),
        vault_root=vault,
        workflow="single_factor",
        run_id="no_source",
    )

    assert preview["can_draft"] is False
    assert "source_code_sha256_missing" in preview["audit"]["draft_blockers"]  # type: ignore[index]


def test_archive_draft_is_idempotent_for_workflow_identity(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    vault = tmp_path / "vault"
    _single_run(
        workspace,
        run_id="idem_run",
        factor_name="idem_factor",
        archive_identity="idem_factor",
    )
    preview = build_archive_preview(
        index=ArchiveRunIndex.build(workspace_root=workspace),
        vault_root=vault,
        workflow="single_factor",
        run_id="idem_run",
    )
    payload = {
        "user_notes_zh": "可以归档。",
        "existing_card_content_sha256_at_review": None,
    }

    first = write_archive_draft(
        vault_root=vault,
        project_slug="archive",
        preview=preview,
        payload=payload,
    )
    second = write_archive_draft(
        vault_root=vault,
        project_slug="archive",
        preview=preview,
        payload={**payload, "user_notes_zh": "更新后的结论。"},
    )

    assert first["draft_name"] == second["draft_name"]
    assert second["updated_existing_pending"] is True


def test_apply_rejects_when_existing_card_changed_after_review(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    vault = tmp_path / "vault"
    _single_run(
        workspace,
        run_id="lock_run",
        factor_name="lock_factor",
        archive_identity="lock_factor",
    )
    target = vault / "50_experiments" / "lock_factor" / "latest.md"
    target.parent.mkdir(parents=True)
    target.write_text("old card\n", encoding="utf-8")
    preview = build_archive_preview(
        index=ArchiveRunIndex.build(workspace_root=workspace),
        vault_root=vault,
        workflow="single_factor",
        run_id="lock_run",
    )
    draft = write_archive_draft(
        vault_root=vault,
        project_slug="archive",
        preview=preview,
        payload={
            "user_notes_zh": "可以归档。",
            "existing_card_content_sha256_at_review": preview["existing_card"][
                "content_sha256"
            ],
        },
    )
    frontmatter, body = _split_frontmatter(Path(str(draft["draft_path"])))
    frontmatter["review_status"] = "approved"
    target.write_text("changed card\n", encoding="utf-8")

    with pytest.raises(AlphaLabConfigError, match="changed after archive review"):
        apply_archive_draft(
            vault_root=vault,
            draft_path=draft["draft_path"],
            frontmatter=frontmatter,
            body=body,
        )


def test_migrate_auto_exports_skips_manually_edited_marker_card(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "vault"
    card = vault / "50_experiments" / "Exp - edited.md"
    card.parent.mkdir(parents=True)
    card.write_text(
        "---\n"
        "type: experiment\n"
        "generated_by: alpha_lab\n"
        "export_kind: pipeline_auto\n"
        "body_sha256: definitely-not-this-body\n"
        "---\n\n"
        "# Edited\n\n"
        "manual note\n",
        encoding="utf-8",
    )

    result = migrate_auto_exports(vault_root=vault, dry_run=True)

    assert str(card) in result["will_skip_manually_edited"]
    assert result["will_move"] == []


def test_archive_preview_does_not_use_recursive_walk(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    vault = tmp_path / "vault"
    _single_run(
        workspace,
        run_id="large_run",
        factor_name="large_factor",
        archive_identity="large_factor",
    )

    def fail_walk(*_: object, **__: object) -> object:
        raise AssertionError("archive preview must not recursively walk output_dir")

    monkeypatch.setattr(os, "walk", fail_walk)
    preview = build_archive_preview(
        index=ArchiveRunIndex.build(workspace_root=workspace),
        vault_root=vault,
        workflow="single_factor",
        run_id="large_run",
    )

    assert preview["ok"] is True
    assert preview["research_journey"]["llm_diagnostics"]["cache_key"]  # type: ignore[index]
    assert preview["research_journey"]["truncation"]["strategy"]  # type: ignore[index]
