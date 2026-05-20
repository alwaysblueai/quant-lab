from __future__ import annotations

import datetime as dt
import difflib
import json
import os
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from alpha_lab.custom_factors import read_custom_factor_source, sha256_text
from alpha_lab.custom_models import read_draft_model_source
from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError
from alpha_lab.research_bridge.service import PROJECTS_DIRNAME

ArchiveWorkflow = Literal["single_factor", "model_factor"]

ARCHIVE_DRAFT_TYPE = "research_archive_draft"
ARCHIVE_PROMPT_TEMPLATE_ID = "archive_summary_v1"
ARCHIVE_LLM_MODEL_ID = "deterministic_fallback"

_WORKFLOWS: frozenset[str] = frozenset({"single_factor", "model_factor"})
_DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}(?:[T ][0-9:.+-]+Z?)?\b")
_SAFE_SLUG_RE = re.compile(r"[^A-Za-z0-9_.-]+")

_SINGLE_FALLBACKS: dict[str, str] = {
    "run_manifest": "run_manifest.json",
    "metrics": "metrics.json",
    "summary": "summary.md",
    "experiment_card": "experiment_card.md",
    "factor_definition_json": "factor_definition.json",
    "integrity_report_json": "integrity_report.json",
    "integrity_report_markdown": "integrity_report.md",
}
_MODEL_FALLBACKS: dict[str, str] = {
    **_SINGLE_FALLBACKS,
    "model_definition_json": "model_definition.json",
    "feature_manifest_json": "feature_manifest.json",
    "model_selection_json": "model_selection.json",
    "feature_importance_ledger": "feature_importance_ledger.csv",
}


@dataclass(frozen=True)
class ArchiveRunRecord:
    run_id: str
    workflow: ArchiveWorkflow
    case_name: str
    output_dir: Path
    artifact_paths: Mapping[str, str] = field(default_factory=dict)
    summary: Mapping[str, object] = field(default_factory=dict)
    status: str = "succeeded"
    submitted_at_utc: str = ""
    finished_at_utc: str = ""
    evaluation_profile: str = ""
    project_slug: str = ""
    spec_path: str = ""


class ArchiveRunIndex:
    """Bounded, whitelist-based index for archive preview grouping."""

    def __init__(self, *, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).expanduser().resolve()
        self._records: dict[str, ArchiveRunRecord] = {}

    @classmethod
    def build(
        cls,
        *,
        workspace_root: str | Path,
        records: Iterable[object] = (),
    ) -> ArchiveRunIndex:
        index = cls(workspace_root=workspace_root)
        index.refresh_workspace()
        index.sync_run_records(records)
        return index

    def sync_run_records(self, records: Iterable[object]) -> None:
        for record in records:
            converted = self._from_web_record(record)
            if converted is not None:
                self._records[converted.run_id] = converted

    def refresh_workspace(self) -> None:
        outputs_root = self.workspace_root / "outputs" / "real_cases"
        web_root = outputs_root / "_web_runs"
        if web_root.exists():
            for run_dir in sorted(item for item in web_root.iterdir() if item.is_dir()):
                for output_dir in sorted(item for item in run_dir.iterdir() if item.is_dir()):
                    record = self._from_output_dir(output_dir, run_id=run_dir.name)
                    if record is not None:
                        self._records.setdefault(record.run_id, record)
        if outputs_root.exists():
            for output_dir in sorted(item for item in outputs_root.iterdir() if item.is_dir()):
                if output_dir.name == "_web_runs":
                    continue
                record = self._from_output_dir(output_dir, run_id=output_dir.name)
                if record is not None:
                    self._records.setdefault(record.run_id, record)

    def get(self, run_id: str) -> ArchiveRunRecord | None:
        return self._records.get(run_id)

    def records(self, *, workflow: ArchiveWorkflow | None = None) -> list[ArchiveRunRecord]:
        rows = list(self._records.values())
        if workflow is not None:
            rows = [item for item in rows if item.workflow == workflow]
        return sorted(rows, key=lambda item: item.submitted_at_utc or item.run_id, reverse=True)

    def _from_web_record(self, record: object) -> ArchiveRunRecord | None:
        output_dir_raw = getattr(record, "output_dir", None)
        if not output_dir_raw:
            return None
        output_dir = Path(str(output_dir_raw)).expanduser().resolve()
        if not output_dir.exists() or not output_dir.is_dir():
            return None
        workflow_raw = str(getattr(record, "workflow", "") or "")
        if workflow_raw not in _WORKFLOWS:
            return None
        run_id = str(getattr(record, "run_id", "") or "").strip()
        if not run_id:
            return None
        return ArchiveRunRecord(
            run_id=run_id,
            workflow=workflow_raw,  # type: ignore[arg-type]
            case_name=str(getattr(record, "case_name", "") or output_dir.name),
            output_dir=output_dir,
            artifact_paths={
                str(key): str(value)
                for key, value in dict(getattr(record, "artifact_paths", {}) or {}).items()
                if key and value
            },
            summary=_as_mapping(getattr(record, "summary", {})),
            status=str(getattr(record, "status", "") or ""),
            submitted_at_utc=str(getattr(record, "submitted_at_utc", "") or ""),
            finished_at_utc=str(getattr(record, "finished_at_utc", "") or ""),
            evaluation_profile=str(getattr(record, "evaluation_profile", "") or ""),
            project_slug=str(getattr(record, "project_slug", "") or ""),
            spec_path=str(getattr(record, "spec_path", "") or ""),
        )

    def _from_output_dir(self, output_dir: Path, *, run_id: str) -> ArchiveRunRecord | None:
        output_dir = output_dir.expanduser().resolve()
        manifest_path = output_dir / "run_manifest.json"
        manifest = _read_json_optional(manifest_path)
        workflow: ArchiveWorkflow | None = None
        if (output_dir / "model_definition.json").exists() or (
            output_dir / "feature_manifest.json"
        ).exists():
            workflow = "model_factor"
        elif (output_dir / "factor_definition.json").exists() or (
            output_dir / "metrics.json"
        ).exists():
            workflow = "single_factor"
        if isinstance(manifest, dict):
            raw_workflow = str(manifest.get("workflow") or "").strip()
            if raw_workflow == "real_case_model_factor":
                workflow = "model_factor"
            elif raw_workflow == "real_case_single_factor":
                workflow = "single_factor"
        if workflow is None:
            return None

        artifact_paths = _artifact_paths_from_output_dir(output_dir, workflow, manifest)
        metrics = _read_json_optional(Path(artifact_paths.get("metrics", "")))
        summary = _as_mapping(metrics.get("metrics")) if isinstance(metrics, dict) else {}
        case_name = (
            str(manifest.get("case_name") or "").strip()
            if isinstance(manifest, dict)
            else ""
        ) or output_dir.name
        timestamp = ""
        if isinstance(manifest, dict):
            timestamp = str(
                manifest.get("run_timestamp_utc")
                or manifest.get("generated_at_utc")
                or ""
            )
        return ArchiveRunRecord(
            run_id=run_id,
            workflow=workflow,
            case_name=case_name,
            output_dir=output_dir,
            artifact_paths=artifact_paths,
            summary=summary,
            status="succeeded",
            submitted_at_utc=timestamp,
            finished_at_utc=timestamp,
            evaluation_profile=_evaluation_profile_from_manifest(manifest),
            spec_path=str(manifest.get("spec_path") or "") if isinstance(manifest, dict) else "",
        )


def build_archive_preview(
    *,
    index: ArchiveRunIndex,
    vault_root: str | Path,
    workflow: str,
    run_id: str,
    include_llm: bool = True,
) -> dict[str, object]:
    workflow_l = _normalize_workflow(workflow)
    record = index.get(run_id)
    if record is None:
        index.refresh_workspace()
        record = index.get(run_id)
    if record is None or record.workflow != workflow_l:
        raise FileNotFoundError(f"archive run not found: {workflow}/{run_id}")
    if record.status not in {"succeeded", "success", ""}:
        raise AlphaLabConfigError(f"run must be succeeded before archive preview: {run_id}")

    trigger = _inspect_run(record, workspace_root=index.workspace_root)
    identity = _as_mapping(trigger.get("identity"))
    audit = _as_mapping(trigger.get("audit"))
    identity_value = str(identity.get("archive_identity") or "")
    historical: list[dict[str, object]] = []
    for candidate in index.records(workflow=workflow_l):
        try:
            inspected = _inspect_run(candidate, workspace_root=index.workspace_root)
        except Exception:
            continue
        candidate_identity = str(
            _as_mapping(inspected.get("identity")).get("archive_identity") or ""
        )
        if candidate_identity != identity_value:
            continue
        historical.append(_historical_run_row(candidate, inspected))
    historical.sort(key=lambda item: str(item.get("run_timestamp_utc") or ""), reverse=True)

    journey = _build_research_journey(trigger, include_llm=include_llm)
    target_rel = f"50_experiments/{_safe_slug(identity_value)}/latest.md"
    target_path = (Path(vault_root).expanduser().resolve() / target_rel).resolve()
    existing = _existing_card_payload(target_path, vault_root=Path(vault_root))
    draft_markdown = render_archive_card_markdown(
        preview_seed={
            "workflow": workflow_l,
            "identity": identity,
            "trigger_run": _trigger_run_payload(record),
            "historical_runs": historical,
            "research_journey": journey,
            "audit": trigger["audit"],
            "target_path": target_rel,
        },
        user_payload={},
    )
    existing_content = existing.get("content")
    existing["diff"] = _unified_diff(
        existing_content if isinstance(existing_content, str) else "",
        draft_markdown,
    )

    generated_at = _utc_now()
    return {
        "ok": True,
        "workflow": workflow_l,
        "identity": identity,
        "trigger_run": _trigger_run_payload(record),
        "historical_runs": historical,
        "research_journey": journey,
            "audit": audit,
        "draft_markdown": draft_markdown,
        "existing_card": existing,
        "target_path": target_rel,
        "timestamps": {
            "run_timestamp_utc": str(trigger.get("run_timestamp_utc") or ""),
            "archive_generated_at_utc": generated_at,
            "vault_read_timestamp_utc": str(existing.get("vault_read_timestamp_utc") or ""),
            "llm_summary_generated_at_utc": journey.get("llm_summary_generated_at_utc"),
        },
        "can_draft": not bool(audit.get("draft_blockers")),
        "can_apply": not bool(audit.get("apply_blockers")),
        "legacy_archive_only": bool(audit.get("legacy_archive_only")),
    }


def write_archive_draft(
    *,
    vault_root: str | Path,
    project_slug: str,
    preview: Mapping[str, object],
    payload: Mapping[str, object],
) -> dict[str, object]:
    audit = _as_mapping(preview.get("audit"))
    blockers = _string_list(audit.get("draft_blockers"))
    if blockers:
        raise AlphaLabConfigError("archive draft is blocked: " + "; ".join(blockers))
    legacy = bool(audit.get("legacy_archive_only"))
    if legacy and not bool(payload.get("acknowledge_legacy_archive_only")):
        raise AlphaLabConfigError("legacy archive requires acknowledge_legacy_archive_only=true")

    identity = _as_mapping(preview.get("identity"))
    workflow = str(preview.get("workflow") or "").strip()
    archive_identity = str(identity.get("archive_identity") or "").strip()
    if not workflow or not archive_identity:
        raise AlphaLabConfigError("archive preview missing workflow/archive_identity")

    expected_sha = _optional_text(payload.get("existing_card_content_sha256_at_review"))
    existing = _as_mapping(preview.get("existing_card"))
    preview_sha = _optional_text(existing.get("content_sha256"))
    if expected_sha != preview_sha:
        raise AlphaLabConfigError(
            "existing_card_content_sha256_at_review does not match preview"
        )

    target_markdown = render_archive_card_markdown(
        preview_seed=preview,
        user_payload=payload,
    )
    root = Path(vault_root).expanduser().resolve()
    drafts_dir = root / PROJECTS_DIRNAME / _safe_slug(project_slug) / "50_writeback_drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    draft_path = _find_existing_pending_archive_draft(
        drafts_dir,
        workflow=workflow,
        archive_identity=archive_identity,
    )
    updated_existing_pending = draft_path is not None
    if draft_path is None:
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        draft_path = drafts_dir / f"{stamp}__{_safe_slug(archive_identity)}__archive_draft.md"

    frontmatter = {
        "type": ARCHIVE_DRAFT_TYPE,
        "project": _safe_slug(project_slug),
        "workflow": workflow,
        "archive_identity": archive_identity,
        "archive_identity_inferred": bool(identity.get("archive_identity_inferred")),
        "target_path": str(preview.get("target_path") or ""),
        "review_status": "pending",
        "reviewed_by": "",
        "reviewed_at": "",
        "vault_export_mode": "versioned",
        "existing_card_content_sha256_at_review": expected_sha or "",
        "origin": "legacy_run" if legacy else "archive_preview",
        "audit_level": "partial_legacy" if legacy else "full",
        "legacy_missing_artifacts": _string_list(audit.get("legacy_missing_artifacts")),
        "source_hashes": _as_mapping(audit.get("source_hashes")),
        "archive_generated_at_utc": _utc_now(),
    }
    draft_path.write_text(
        _compose_markdown_with_frontmatter(frontmatter, target_markdown),
        encoding="utf-8",
    )
    return {
        "ok": True,
        "draft_name": draft_path.name,
        "draft_path": str(draft_path),
        "target_path": frontmatter["target_path"],
        "status": "pending",
        "updated_existing_pending": updated_existing_pending,
        "preview": _read_text_preview(draft_path, limit_bytes=256_000),
    }


def apply_archive_draft(
    *,
    vault_root: str | Path,
    draft_path: str | Path,
    frontmatter: Mapping[str, object],
    body: str,
    mode: str | None = None,
) -> dict[str, object]:
    review_status = str(frontmatter.get("review_status") or "").strip().lower()
    if review_status != "approved":
        raise ValueError(f"draft {draft_path} has not been approved")
    root = Path(vault_root).expanduser().resolve()
    target_rel = str(frontmatter.get("target_path") or "").strip().lstrip("/").replace("\\", "/")
    if not target_rel:
        raise ValueError("archive draft target_path is required")
    target_path = (root / target_rel).resolve()
    try:
        target_path.relative_to(root)
    except ValueError as exc:
        raise PermissionError("archive target_path must stay inside vault_root") from exc
    expected_sha = _optional_text(frontmatter.get("existing_card_content_sha256_at_review"))
    current_sha = (
        sha256_text(target_path.read_text(encoding="utf-8"))
        if target_path.exists()
        else None
    )
    if current_sha != expected_sha:
        raise AlphaLabConfigError(
            "vault card changed after archive review; regenerate archive preview before apply"
        )

    mode_l = str(mode or frontmatter.get("vault_export_mode") or "versioned").strip().lower()
    if mode_l not in {"versioned", "overwrite", "skip"}:
        raise ValueError("mode must be one of skip, overwrite, versioned")
    if mode_l == "skip":
        return {
            "ok": True,
            "success": True,
            "status": "skipped",
            "target_paths": [],
            "mode_used": "skip",
        }

    target_path.parent.mkdir(parents=True, exist_ok=True)
    targets: list[Path] = []
    if target_path.exists() and mode_l == "versioned":
        stamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H-%M-%S")
        versioned = target_path.with_name(f"{stamp}__{target_path.name}")
        versioned.write_text(body.rstrip() + "\n", encoding="utf-8")
        targets.append(versioned)
    target_path.write_text(body.rstrip() + "\n", encoding="utf-8")
    targets.append(target_path)
    _mark_draft_applied(Path(draft_path), frontmatter)
    return {
        "ok": True,
        "success": True,
        "status": "success",
        "target_paths": [str(path) for path in targets],
        "mode_used": mode_l,
    }


def render_archive_card_markdown(
    *,
    preview_seed: Mapping[str, object],
    user_payload: Mapping[str, object],
) -> str:
    identity = _as_mapping(preview_seed.get("identity"))
    audit = _as_mapping(preview_seed.get("audit"))
    trigger = _as_mapping(preview_seed.get("trigger_run"))
    journey = _as_mapping(preview_seed.get("research_journey"))
    raw_historical = preview_seed.get("historical_runs")
    historical = (
        list(raw_historical)
        if isinstance(raw_historical, Sequence) and not isinstance(raw_historical, str | bytes)
        else []
    )
    archive_identity = str(identity.get("archive_identity") or "unknown")
    generated_at = _utc_now()
    legacy = bool(audit.get("legacy_archive_only"))
    user_notes = str(user_payload.get("user_notes_zh") or "").strip()
    risks = _string_list(user_payload.get("risks"))
    next_steps = _string_list(user_payload.get("next_steps"))
    emergent_moves = _string_list(user_payload.get("emergent_moves"))
    operative_claims = _string_list(user_payload.get("operative_claims"))
    frontmatter = {
        "type": "experiment",
        "generated_by": "alpha_lab",
        "export_kind": "archive_approved",
        "archive_identity": archive_identity,
        "workflow": str(preview_seed.get("workflow") or ""),
        "status": "draft",
        "origin": "legacy_run" if legacy else "archive_preview",
        "audit_level": "partial_legacy" if legacy else "full",
        "archive_generated_from_run_id": str(trigger.get("run_id") or ""),
        "archive_generated_at_utc": generated_at,
        "emergent_moves": emergent_moves,
        "operative_claims": operative_claims,
    }
    if legacy:
        frontmatter["legacy_missing_artifacts"] = _string_list(
            audit.get("legacy_missing_artifacts")
        )
    lines = [
        f"# {archive_identity} 研究归档",
        "",
        "## 归档结论",
        "",
        user_notes or "待补充：本次归档的中文研究结论。",
        "",
        "## 当前触发 run",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Run ID | `{trigger.get('run_id', '')}` |",
        f"| Case | `{trigger.get('case_name', '')}` |",
        f"| Profile | `{trigger.get('evaluation_profile', '')}` |",
        f"| Output | `{trigger.get('output_dir', '')}` |",
        "",
        "## 研究演化过程",
        "",
        str(
            journey.get("llm_summary_zh")
            or journey.get("deterministic_summary_zh")
            or "暂无过程摘要。"
        ),
        "",
        "## 历史 runs",
        "",
        _render_historical_runs_table(historical),
        "",
        "## 风险与边界",
        "",
        *_bullet_lines(risks, fallback="待补充：失效风险、样本边界、适用条件。"),
        "",
        "## 下一步",
        "",
        *_bullet_lines(next_steps, fallback="待补充：下一步验证或复核动作。"),
        "",
        "## 回灌素材",
        "",
        "- `emergent_moves`:",
        *_bullet_lines(emergent_moves, fallback="待补充。"),
        "- `operative_claims`:",
        *_bullet_lines(operative_claims, fallback="待补充。"),
        "",
        "## 审计",
        "",
        f"- `audit_level`: `{frontmatter['audit_level']}`",
        "- `source_hashes`: "
        f"`{json.dumps(audit.get('source_hashes') or {}, ensure_ascii=False, sort_keys=True)}`",
        "- `blocking_diagnostics`: "
        f"`{json.dumps(audit.get('diagnostics') or [], ensure_ascii=False, sort_keys=True)}`",
        "",
    ]
    return _compose_markdown_with_frontmatter(frontmatter, "\n".join(lines))


def migrate_auto_exports(
    *,
    vault_root: str | Path,
    dry_run: bool = True,
) -> dict[str, object]:
    root = Path(vault_root).expanduser().resolve()
    experiments = root / "50_experiments"
    rows: dict[str, list[object]] = {
        "will_move": [],
        "will_skip_no_marker": [],
        "will_skip_manually_edited": [],
    }
    if not experiments.exists():
        return {"ok": True, "dry_run": dry_run, **rows}
    candidates = [*experiments.glob("*.md"), *experiments.glob("*/*.md")]
    archive_root = experiments / "_archived_auto_exports"
    manual_review: list[str] = []
    for path in sorted(candidates):
        if archive_root in path.parents:
            continue
        frontmatter, body = _load_markdown_with_frontmatter_optional(path)
        if (
            str(frontmatter.get("generated_by") or "") != "alpha_lab"
            or str(frontmatter.get("export_kind") or "") != "pipeline_auto"
        ):
            rows["will_skip_no_marker"].append(str(path))
            continue
        expected_body_sha = _optional_text(
            frontmatter.get("source_content_sha256")
            or frontmatter.get("body_sha256")
        )
        if expected_body_sha is None or expected_body_sha != sha256_text(body):
            rows["will_skip_manually_edited"].append(str(path))
            manual_review.append(str(path))
            continue
        target = archive_root / path.relative_to(experiments)
        rows["will_move"].append({"source": str(path), "target": str(target)})
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
    if not dry_run:
        manifest_path = archive_root / "migration_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        if manual_review:
            (archive_root / "manual_review_required.txt").write_text(
                "\n".join(manual_review) + "\n",
                encoding="utf-8",
            )
    return {"ok": True, "dry_run": dry_run, **rows}


def cleanup_deprecated_writebacks(
    *,
    vault_root: str | Path,
    older_than_days: int = 90,
) -> dict[str, object]:
    root = Path(vault_root).expanduser().resolve()
    projects_root = root / PROJECTS_DIRNAME
    if not projects_root.exists():
        return {"ok": True, "moved": []}
    cutoff = dt.datetime.now(dt.UTC).timestamp() - older_than_days * 86400
    moved: list[dict[str, str]] = []
    for drafts_dir in projects_root.glob("*/50_writeback_drafts"):
        if not drafts_dir.is_dir():
            continue
        archive_dir = drafts_dir.parent / "_archive" / "deprecated_writebacks"
        for draft in drafts_dir.glob("*__writeback_draft.md"):
            try:
                original_mtime = draft.stat().st_mtime
                frontmatter, body = _load_markdown_with_frontmatter_optional(draft)
            except OSError:
                continue
            draft_type = str(frontmatter.get("draft_type") or frontmatter.get("type") or "")
            if draft_type == "knowledge_writeback_draft":
                frontmatter = {
                    **frontmatter,
                    "type": "deprecated_manual_writeback",
                    "deprecated_from": "knowledge_writeback_draft",
                    "deprecated_at": frontmatter.get("deprecated_at") or _utc_now(),
                }
                draft.write_text(
                    _compose_markdown_with_frontmatter(frontmatter, body),
                    encoding="utf-8",
                )
                draft_type = "deprecated_manual_writeback"
            if draft_type != "deprecated_manual_writeback":
                continue
            if original_mtime > cutoff:
                continue
            target = archive_dir / draft.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.move(str(draft), str(target))
                moved.append({"source": str(draft), "target": str(target)})
    return {"ok": True, "moved": moved}


def preview_to_markdown(preview: Mapping[str, object]) -> str:
    return render_archive_card_markdown(preview_seed=preview, user_payload={})


def _inspect_run(record: ArchiveRunRecord, *, workspace_root: Path) -> dict[str, object]:
    artifacts = _resolved_artifacts(record)
    manifest = _read_json_optional(artifacts.get("run_manifest"))
    metrics = _read_json_optional(artifacts.get("metrics"))
    factor_definition = _read_json_optional(artifacts.get("factor_definition_json"))
    model_definition = _read_json_optional(artifacts.get("model_definition_json"))
    feature_manifest = _read_json_optional(artifacts.get("feature_manifest_json"))
    integrity = _read_json_optional(artifacts.get("integrity_report_json"))

    source = _source_audit(
        workflow=record.workflow,
        workspace_root=workspace_root,
        manifest=manifest,
        factor_definition=factor_definition,
        model_definition=model_definition,
        feature_manifest=feature_manifest,
        record=record,
    )
    identity = _archive_identity(
        workflow=record.workflow,
        source=source,
        manifest=manifest,
        metrics=metrics,
        factor_definition=factor_definition,
        model_definition=model_definition,
        record=record,
    )
    research_log_path = _research_log_path(
        workflow=record.workflow,
        workspace_root=workspace_root,
        source=source,
        identity=str(identity["archive_identity"]),
    )
    audit = _audit_payload(
        workflow=record.workflow,
        artifacts=artifacts,
        manifest=manifest,
        factor_definition=factor_definition,
        model_definition=model_definition,
        source=source,
        integrity=integrity,
        research_log_path=research_log_path,
    )
    return {
        "identity": identity,
        "audit": audit,
        "artifacts": {key: str(path) for key, path in artifacts.items()},
        "manifest": manifest,
        "metrics": metrics,
        "factor_definition": factor_definition,
        "model_definition": model_definition,
        "research_log_path": str(research_log_path) if research_log_path else "",
        "run_timestamp_utc": _run_timestamp(record, manifest),
    }


def _source_audit(
    *,
    workflow: ArchiveWorkflow,
    workspace_root: Path,
    manifest: Mapping[str, object] | None,
    factor_definition: Mapping[str, object] | None,
    model_definition: Mapping[str, object] | None,
    feature_manifest: Mapping[str, object] | None,
    record: ArchiveRunRecord,
) -> dict[str, object]:
    source: dict[str, object] = {}
    if workflow == "single_factor":
        for parent in (
            factor_definition,
            manifest,
            _as_mapping(manifest.get("inputs")) if manifest else None,
        ):
            raw = _as_mapping(parent.get("custom_factor_source")) if parent else {}
            if raw:
                source.update(raw)
        path = _optional_text(source.get("path"))
        if path:
            try:
                source.update(read_custom_factor_source(path).to_audit_dict())
            except Exception:
                pass
        if not source:
            for candidate in _custom_factor_candidates(workspace_root, record, factor_definition):
                if candidate.exists():
                    try:
                        source.update(read_custom_factor_source(candidate).to_audit_dict())
                        break
                    except Exception:
                        continue
    else:
        for parent in (
            model_definition,
            feature_manifest,
            manifest,
            _as_mapping(manifest.get("inputs")) if manifest else None,
        ):
            raw = _as_mapping(parent.get("draft_model_source")) if parent else {}
            if raw:
                source.update(raw)
        path = _optional_text(source.get("path")) or record.spec_path
        if path and Path(path).name == "model_candidate.json":
            try:
                source.update(read_draft_model_source(path).to_audit_dict())
            except Exception:
                pass
    return source


def _archive_identity(
    *,
    workflow: ArchiveWorkflow,
    source: Mapping[str, object],
    manifest: Mapping[str, object] | None,
    metrics: Mapping[str, object] | None,
    factor_definition: Mapping[str, object] | None,
    model_definition: Mapping[str, object] | None,
    record: ArchiveRunRecord,
) -> dict[str, object]:
    metrics_obj = _as_mapping(metrics.get("metrics")) if metrics else {}
    inferred = False
    if workflow == "single_factor":
        candidates = [
            source.get("archive_identity"),
            factor_definition.get("archive_identity") if factor_definition else None,
            _as_mapping(factor_definition.get("spec")).get("archive_identity")
            if factor_definition
            else None,
            factor_definition.get("factor_name") if factor_definition else None,
            metrics_obj.get("factor_name"),
            record.summary.get("factor_name"),
            record.case_name,
        ]
    else:
        candidates = [
            source.get("archive_identity"),
            model_definition.get("archive_identity") if model_definition else None,
            source.get("name"),
            model_definition.get("factor_name") if model_definition else None,
            metrics_obj.get("factor_name"),
            record.summary.get("factor_name"),
            record.case_name,
        ]
    value = ""
    source_field = ""
    for idx, candidate in enumerate(candidates):
        text = _optional_text(candidate)
        if text:
            value = text
            source_field = f"candidate_{idx}"
            inferred = idx >= (3 if workflow == "single_factor" else 4)
            break
    if not value:
        raise AlphaLabDataError(f"could not infer archive identity for run {record.run_id}")
    return {
        "archive_identity": value,
        "archive_identity_inferred": inferred,
        "archive_identity_source": source_field,
        "display_name": value,
    }


def _audit_payload(
    *,
    workflow: ArchiveWorkflow,
    artifacts: Mapping[str, Path],
    manifest: Mapping[str, object] | None,
    factor_definition: Mapping[str, object] | None,
    model_definition: Mapping[str, object] | None,
    source: Mapping[str, object],
    integrity: Mapping[str, object] | None,
    research_log_path: Path | None,
) -> dict[str, object]:
    diagnostics: list[dict[str, object]] = []
    draft_blockers: list[str] = []
    apply_blockers: list[str] = []
    legacy_missing: list[str] = []

    def add(
        code: str,
        severity: str,
        message: str,
        *,
        block: bool = False,
        legacy: bool = False,
    ) -> None:
        diagnostics.append({"code": code, "severity": severity, "message": message})
        if legacy:
            legacy_missing.append(code)
        elif block:
            draft_blockers.append(code)
            apply_blockers.append(code)

    if research_log_path is None:
        add("research_log_missing", "warn", "research_log.md is missing")
    if manifest is None:
        add("run_manifest_missing", "warn", "run_manifest.json is missing", legacy=True)
    if workflow == "single_factor" and factor_definition is None:
        add("factor_definition_missing", "warn", "factor_definition.json is missing", legacy=True)
    if workflow == "model_factor" and model_definition is None:
        add("model_definition_missing", "warn", "model_definition.json is missing", legacy=True)

    source_hashes: dict[str, object] = {}
    if workflow == "single_factor":
        for key in ("code_sha256", "factor_json_sha256", "path"):
            if source.get(key):
                source_hashes[key] = source[key]
        for key in ("code_sha256", "factor_json_sha256"):
            if not source.get(key):
                add(
                    f"source_{key}_missing",
                    "warn",
                    f"custom_factor_source.{key} is missing",
                    block=True,
                )
    else:
        for key in ("candidate_json_sha256", "case_spec_sha256", "feature_contract_sha256", "path"):
            if source.get(key):
                source_hashes[key] = source[key]
        for key in ("candidate_json_sha256", "case_spec_sha256", "feature_contract_sha256"):
            if not source.get(key):
                add(
                    f"source_{key}_missing",
                    "warn",
                    f"draft_model_source.{key} is missing",
                    block=True,
                )

    if _integrity_hard_fail(manifest, integrity):
        add("integrity_hard_fail", "warn", "integrity report contains hard failures", block=True)
    provenance = _as_mapping(source.get("provenance"))
    if not _optional_text(provenance.get("idea_id")):
        add("provenance_idea_id_missing", "info", "provenance.idea_id is missing")

    legacy = bool(legacy_missing)
    return {
        "diagnostics": diagnostics,
        "draft_blockers": draft_blockers,
        "apply_blockers": apply_blockers,
        "legacy_archive_only": legacy,
        "legacy_missing_artifacts": legacy_missing,
        "source_hashes": source_hashes,
        "artifact_paths": {key: str(path) for key, path in artifacts.items()},
    }


def _build_research_journey(
    inspected: Mapping[str, object],
    *,
    include_llm: bool,
) -> dict[str, object]:
    log_path_text = str(inspected.get("research_log_path") or "").strip()
    events: list[dict[str, object]] = []
    freeform_note = ""
    if log_path_text:
        path = Path(log_path_text)
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="replace")
            events, freeform_note = _parse_research_log(text)
    deterministic_input = _deterministic_summary_input(events, freeform_note)
    deterministic_summary = _render_deterministic_summary(events, freeform_note)
    llm_diag = {
        "enabled": include_llm,
        "cache_key": sha256_text(
            deterministic_input + ARCHIVE_PROMPT_TEMPLATE_ID + ARCHIVE_LLM_MODEL_ID
        ),
        "status": "not_configured"
        if not os.environ.get("ANTHROPIC_API_KEY")
        else "deterministic_fallback",
    }
    return {
        "events": events,
        "freeform_note": freeform_note,
        "deterministic_summary_zh": deterministic_summary,
        "llm_summary_zh": None,
        "llm_summary_generated_at_utc": None,
        "llm_diagnostics": llm_diag,
        "truncation": {
            "strategy": "header_first_event_last_8_events_drop_freeform_first",
            "input_sha256": sha256_text(deterministic_input),
        },
    }


def _parse_research_log(text: str) -> tuple[list[dict[str, object]], str]:
    events: list[dict[str, object]] = []
    freeform: list[str] = []
    current_heading = ""
    current_lines: list[str] = []
    current_date = ""

    def flush() -> None:
        nonlocal current_lines, current_date, current_heading
        body = "\n".join(current_lines).strip()
        if body and current_date:
            events.append({"timestamp": current_date, "title": current_heading, "body": body})
        elif body:
            freeform.append(body)
        current_lines = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        if line.startswith("##"):
            flush()
            current_heading = line.lstrip("#").strip()
            match = _DATE_RE.search(line)
            current_date = match.group(0) if match else ""
            continue
        match = _DATE_RE.search(line)
        if match and line.lstrip().startswith(("-", "*")):
            flush()
            events.append(
                {
                    "timestamp": match.group(0),
                    "title": "log_event",
                    "body": line.lstrip("-* ").strip(),
                }
            )
            current_date = ""
            current_heading = ""
            continue
        current_lines.append(line)
    flush()
    return events, "\n\n".join(item for item in freeform if item.strip()).strip()


def _deterministic_summary_input(events: Sequence[Mapping[str, object]], freeform: str) -> str:
    kept: list[str] = []
    if events:
        kept.append(json.dumps(events[0], ensure_ascii=False, sort_keys=True))
        for event in events[-8:]:
            rendered = json.dumps(event, ensure_ascii=False, sort_keys=True)
            if rendered not in kept:
                kept.append(rendered)
    if len("\n".join(kept)) < 16_000 and freeform:
        kept.append(freeform[: max(0, 16_000 - len('\n'.join(kept)))])
    return "\n".join(kept)


def _render_deterministic_summary(events: Sequence[Mapping[str, object]], freeform: str) -> str:
    if not events and not freeform:
        return "未找到可结构化的 research_log；本次只能归档最终实验结果。"
    lines = ["后端记录到的研究演化摘要："]
    selected = list(events[:1])
    selected.extend(event for event in events[-8:] if event not in selected)
    for event in selected:
        timestamp = str(event.get("timestamp") or "")
        body = str(event.get("body") or "").strip().replace("\n", " ")
        lines.append(f"- {timestamp}: {body[:260]}")
    if freeform and not events:
        lines.append(f"- freeform_note: {freeform[:500]}")
    return "\n".join(lines)


def _historical_run_row(
    record: ArchiveRunRecord, inspected: Mapping[str, object]
) -> dict[str, object]:
    metrics = _as_mapping(_as_mapping(inspected.get("metrics")).get("metrics"))
    return {
        "run_id": record.run_id,
        "case_name": record.case_name,
        "evaluation_profile": record.evaluation_profile,
        "run_timestamp_utc": inspected.get("run_timestamp_utc") or record.finished_at_utc,
        "output_dir": str(record.output_dir),
        "factor_verdict": metrics.get("factor_verdict") or record.summary.get("factor_verdict"),
        "mean_rank_ic": metrics.get("mean_rank_ic") or record.summary.get("mean_rank_ic"),
        "mean_ic": metrics.get("mean_ic") or record.summary.get("mean_ic"),
    }


def _trigger_run_payload(record: ArchiveRunRecord) -> dict[str, object]:
    return {
        "run_id": record.run_id,
        "workflow": record.workflow,
        "case_name": record.case_name,
        "status": record.status,
        "output_dir": str(record.output_dir),
        "evaluation_profile": record.evaluation_profile,
        "project_slug": record.project_slug,
    }


def _artifact_paths_from_output_dir(
    output_dir: Path,
    workflow: ArchiveWorkflow,
    manifest: Mapping[str, object] | None,
) -> dict[str, str]:
    paths: dict[str, str] = {}
    outputs = manifest.get("outputs") if isinstance(manifest, Mapping) else None
    if isinstance(outputs, Mapping):
        for key, value in outputs.items():
            key_text = str(key or "").strip()
            value_text = str(value or "").strip()
            if not key_text or not value_text:
                continue
            candidate = Path(value_text).expanduser()
            if not candidate.is_absolute():
                candidate = output_dir / candidate
            paths[key_text] = str(candidate.resolve())
    fallbacks = _MODEL_FALLBACKS if workflow == "model_factor" else _SINGLE_FALLBACKS
    for key, filename in fallbacks.items():
        candidate = output_dir / filename
        if candidate.exists():
            paths.setdefault(key, str(candidate.resolve()))
    return paths


def _resolved_artifacts(record: ArchiveRunRecord) -> dict[str, Path]:
    fallbacks = _MODEL_FALLBACKS if record.workflow == "model_factor" else _SINGLE_FALLBACKS
    artifacts: dict[str, Path] = {}
    for key, value in record.artifact_paths.items():
        text = str(value or "").strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = record.output_dir / path
        if path.exists() and path.is_file():
            artifacts[key] = path.resolve()
    for key, filename in fallbacks.items():
        candidate = record.output_dir / filename
        if candidate.exists() and candidate.is_file():
            artifacts.setdefault(key, candidate.resolve())
    return artifacts


def _research_log_path(
    *,
    workflow: ArchiveWorkflow,
    workspace_root: Path,
    source: Mapping[str, object],
    identity: str,
) -> Path | None:
    path_text = _optional_text(source.get("path"))
    if path_text:
        candidate = Path(path_text).expanduser().resolve().with_name("research_log.md")
        if candidate.exists():
            return candidate
    if workflow == "single_factor":
        candidate = workspace_root / "custom_factors" / "research" / identity / "research_log.md"
    else:
        candidate = workspace_root / "custom_models" / "research" / identity / "research_log.md"
    return candidate.resolve() if candidate.exists() else None


def _custom_factor_candidates(
    workspace_root: Path,
    record: ArchiveRunRecord,
    factor_definition: Mapping[str, object] | None,
) -> list[Path]:
    names = [
        _optional_text(factor_definition.get("factor_name")) if factor_definition else None,
        _optional_text(record.summary.get("factor_name")),
        record.case_name,
    ]
    out: list[Path] = []
    for name in names:
        if not name:
            continue
        out.append(workspace_root / "custom_factors" / "research" / name / "factor.json")
    return out


def _existing_card_payload(target_path: Path, *, vault_root: Path) -> dict[str, object]:
    read_at = _utc_now()
    try:
        rel = target_path.relative_to(vault_root.expanduser().resolve()).as_posix()
    except ValueError:
        rel = str(target_path)
    if not target_path.exists():
        return {
            "exists": False,
            "path": str(target_path),
            "relative_path": rel,
            "content_sha256": None,
            "content": "",
            "vault_read_timestamp_utc": read_at,
        }
    text = target_path.read_text(encoding="utf-8", errors="replace")
    return {
        "exists": True,
        "path": str(target_path),
        "relative_path": rel,
        "content_sha256": sha256_text(text),
        "content": text,
        "vault_read_timestamp_utc": read_at,
    }


def _find_existing_pending_archive_draft(
    drafts_dir: Path,
    *,
    workflow: str,
    archive_identity: str,
) -> Path | None:
    for path in sorted(drafts_dir.glob("*__archive_draft.md"), reverse=True):
        frontmatter, _ = _load_markdown_with_frontmatter_optional(path)
        if str(frontmatter.get("type") or "") != ARCHIVE_DRAFT_TYPE:
            continue
        if str(frontmatter.get("review_status") or "") != "pending":
            continue
        if str(frontmatter.get("workflow") or "") != workflow:
            continue
        if str(frontmatter.get("archive_identity") or "") == archive_identity:
            return path
    return None


def _mark_draft_applied(draft_path: Path, frontmatter: Mapping[str, object]) -> None:
    if not draft_path.exists():
        return
    current, body = _load_markdown_with_frontmatter_optional(draft_path)
    updated = {**current, "review_status": "applied", "applied_at": _utc_now()}
    draft_path.write_text(_compose_markdown_with_frontmatter(updated, body), encoding="utf-8")


def _render_historical_runs_table(rows: Sequence[object]) -> str:
    if not rows:
        return "暂无历史 run。"
    lines = [
        "| run_id | case | profile | timestamp | mean_rank_ic | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for raw in rows:
        row = _as_mapping(raw)
        lines.append(
            "| "
            f"`{str(row.get('run_id') or '')[:12]}` | "
            f"`{row.get('case_name') or ''}` | "
            f"`{row.get('evaluation_profile') or ''}` | "
            f"{row.get('run_timestamp_utc') or ''} | "
            f"{row.get('mean_rank_ic') or ''} | "
            f"{row.get('factor_verdict') or ''} |"
        )
    return "\n".join(lines)


def _bullet_lines(values: Sequence[str], *, fallback: str) -> list[str]:
    return [f"- {item}" for item in values] if values else [f"- {fallback}"]


def _unified_diff(old: str, new: str) -> str:
    if not old:
        return ""
    lines = list(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile="existing",
            tofile="archive_preview",
            lineterm="",
        )
    )
    if len(lines) > 240:
        lines = [*lines[:240], "... diff truncated ..."]
    return "\n".join(lines)


def _load_markdown_with_frontmatter_optional(path: Path) -> tuple[dict[str, object], str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.startswith("---\n"):
        return {}, text
    try:
        _, raw_frontmatter, body = text.split("---\n", 2)
    except ValueError:
        return {}, text
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise AlphaLabConfigError("PyYAML is required for archive markdown") from exc
    loaded = yaml.safe_load(raw_frontmatter) or {}
    return (dict(loaded) if isinstance(loaded, dict) else {}), body


def _compose_markdown_with_frontmatter(frontmatter: Mapping[str, object], body: str) -> str:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover
        raise AlphaLabConfigError("PyYAML is required for archive markdown") from exc
    rendered = yaml.safe_dump(dict(frontmatter), sort_keys=False, allow_unicode=True).strip()
    return f"---\n{rendered}\n---\n\n{body.rstrip()}\n"


def _read_json_optional(path: str | Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    try:
        p = Path(path).expanduser().resolve()
    except OSError:
        return None
    if not p.exists() or not p.is_file():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _read_text_preview(path: Path, *, limit_bytes: int) -> str:
    raw = path.read_bytes()
    return raw[:limit_bytes].decode("utf-8", errors="replace")


def _evaluation_profile_from_manifest(manifest: Mapping[str, object] | None) -> str:
    if not isinstance(manifest, Mapping):
        return ""
    evaluation_standard = manifest.get("evaluation_standard")
    if isinstance(evaluation_standard, Mapping):
        return str(evaluation_standard.get("profile_name") or "")
    return ""


def _run_timestamp(record: ArchiveRunRecord, manifest: Mapping[str, object] | None) -> str:
    if isinstance(manifest, Mapping):
        raw = manifest.get("run_timestamp_utc") or manifest.get("generated_at_utc")
        if raw:
            return str(raw)
    return record.finished_at_utc or record.submitted_at_utc


def _integrity_hard_fail(
    manifest: Mapping[str, object] | None,
    integrity: Mapping[str, object] | None,
) -> bool:
    candidates: list[Mapping[str, object]] = []
    if isinstance(manifest, Mapping):
        candidates.append(_as_mapping(manifest.get("integrity_summary")))
    if isinstance(integrity, Mapping):
        candidates.extend([integrity, _as_mapping(integrity.get("summary"))])
    for item in candidates:
        if not item:
            continue
        for key in ("hard_fail", "has_hard_fail", "failed", "has_failures"):
            if item.get(key) is True:
                return True
        for key in ("fail_count", "n_fail", "failed_checks"):
            count = _optional_int(item.get(key))
            if count is not None and count > 0:
                return True
    return False


def _normalize_workflow(workflow: str) -> ArchiveWorkflow:
    value = str(workflow or "").strip()
    if value not in _WORKFLOWS:
        raise ValueError("workflow must be one of single_factor, model_factor")
    return value  # type: ignore[return-value]


def _safe_slug(value: str) -> str:
    text = str(value or "").strip().replace("\\", "/").split("/")[-1]
    text = _SAFE_SLUG_RE.sub("-", text).strip(".-_")
    return text or "archive"


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _string_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [line.strip() for line in value.splitlines() if line.strip()]
    return []


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {str(key): item for key, item in value.items()}


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "ARCHIVE_DRAFT_TYPE",
    "ArchiveRunIndex",
    "ArchiveRunRecord",
    "apply_archive_draft",
    "build_archive_preview",
    "cleanup_deprecated_writebacks",
    "migrate_auto_exports",
    "preview_to_markdown",
    "render_archive_card_markdown",
    "write_archive_draft",
]
