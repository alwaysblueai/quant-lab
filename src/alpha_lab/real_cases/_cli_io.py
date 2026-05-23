"""Shared CLI post-run helpers for real-case pipelines.

``render_case_report``, ``update_run_manifest`` and
``finalize_contract_if_research_draft`` are the post-run hooks shared by
``single_factor/cli.py``, ``model_factor/cli.py`` and (for the rendering /
manifest helpers) ``composite/cli.py``.
"""

from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path

from alpha_lab.artifact_contracts import validate_level12_artifact_payload
from alpha_lab.backend_run_contract import (
    BackendRunWorkflow,
    detect_research_draft_run,
    finalize_backend_contract,
)
from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.reporting.renderers import write_case_report
from alpha_lab.vault_export import ExportResult, export_to_vault, resolve_vault_root

_logger = logging.getLogger(__name__)


def render_case_report(
    *,
    output_dir: Path,
    enabled: bool,
    overwrite: bool,
) -> dict[str, object]:
    if not enabled:
        return {
            "rendered_report": False,
            "rendered_report_path": None,
            "render_status": "skipped",
            "render_error": None,
        }

    try:
        report_path = write_case_report(output_dir, overwrite=overwrite)
        return {
            "rendered_report": True,
            "rendered_report_path": str(report_path),
            "render_status": "success",
            "render_error": None,
        }
    except Exception as exc:
        _logger.warning(
            "Case report rendering failed for %s: %s",
            output_dir,
            exc,
        )
        return {
            "rendered_report": False,
            "rendered_report_path": None,
            "render_status": "failed",
            "render_error": str(exc),
        }


def update_run_manifest(
    manifest_path: Path,
    render_meta: dict[str, object],
) -> None:
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise AlphaLabDataError("run_manifest.json root must be an object")
        payload.update(render_meta)
        validate_level12_artifact_payload(
            payload,
            artifact_name=manifest_path.name,
            source=manifest_path,
        )
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        _logger.warning(
            "Failed to persist render metadata into %s: %s",
            manifest_path,
            exc,
        )


def finalize_contract_if_research_draft(
    *,
    output_dir: Path,
    workflow: BackendRunWorkflow,
    case_spec_path: str | Path,
    evaluation_profile: str,
    command: Sequence[str] = (),
) -> int:
    """Run :func:`finalize_backend_contract` iff the run is a research draft.

    Returns ``0`` when contract finalization is skipped (non-draft run) or
    succeeds, ``1`` when audit fails. Sidecars are still written on failure so
    callers can read ``issues`` from ``backend_run_receipt.json``.
    """

    draft_path = detect_research_draft_run(output_dir, workflow=workflow)
    if draft_path is None:
        return 0
    case_report = output_dir / "case_report.md"
    if not case_report.exists():
        new_render_meta = render_case_report(
            output_dir=output_dir, enabled=True, overwrite=False
        )
        update_run_manifest(
            output_dir / "run_manifest.json",
            new_render_meta,
        )
    validation_payload = _validate_draft_for_contract(workflow, draft_path)
    receipt = finalize_backend_contract(
        output_dir,
        workflow=workflow,
        draft_source_path=draft_path,
        case_spec_path=case_spec_path,
        evaluation_profile=evaluation_profile,
        command=command,
        validation_payload=validation_payload,
    )
    return 0 if str(receipt.get("status") or "") == "success" else 1


def export_to_vault_after_contract(
    *,
    case_name: str,
    vault_root: str | Path | None,
    vault_export_mode: str,
    experiment_card_path: Path,
    summary_path: Path,
    manifest_path: Path,
    workflow_label: str,
) -> ExportResult:
    """Run vault export after the backend contract finalize step.

    Called by ``real-case`` CLIs that pass ``defer_vault_export=True`` to the
    pipeline. Picks up ``backend_run_receipt.json`` / ``comparison_summary.json``
    via ``vault_export.export_to_vault``'s auto-detect, then writes the
    ``vault_export`` block into the local ``run_manifest.json`` and re-syncs
    any vault-side manifest copies so they include the final
    ``backend_run_contract`` block.
    """

    resolved_vault = resolve_vault_root(vault_root)
    enabled = (
        resolved_vault is not None and vault_export_mode.strip().lower() != "skip"
    )
    vault_result = export_to_vault(
        {
            "experiment_card_path": experiment_card_path,
            "summary_path": summary_path,
            "manifest_path": manifest_path,
        },
        case_name=case_name,
        vault_root=vault_root,
        mode=vault_export_mode,
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _logger.warning(
            "vault export post-contract: cannot read %s: %s", manifest_path, exc
        )
        return vault_result
    if not isinstance(manifest, dict):
        _logger.warning(
            "vault export post-contract: %s root must be an object", manifest_path
        )
        return vault_result
    manifest["vault_export"] = vault_result.to_manifest_dict(enabled=enabled)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if vault_result.status == "failed":
        _logger.warning(
            "Vault export failed for %s case %s: %s",
            workflow_label,
            case_name,
            vault_result.error,
        )
    if vault_result.success and vault_result.target_paths:
        for raw in vault_result.target_paths:
            target = Path(raw)
            if not target.name.endswith("run_manifest.json"):
                continue
            try:
                shutil.copy2(manifest_path, target)
            except OSError as exc:
                _logger.warning(
                    "Failed to sync %s vault manifest copy %s: %s",
                    workflow_label,
                    target,
                    exc,
                )
    return vault_result


def _validate_draft_for_contract(
    workflow: BackendRunWorkflow,
    draft_path: Path,
) -> dict[str, object]:
    """Run the draft validator that matches the workflow and return its payload.

    Validation failures still produce a payload (with ``ok=False`` and
    ``errors``) so the receipt records the issues even when the contract is
    going to fail.
    """

    if workflow == "single_factor":
        from alpha_lab.draft_factor_validation import validate_draft_factor_file

        return validate_draft_factor_file(draft_path).to_payload()
    if workflow == "model_factor":
        from alpha_lab.draft_model_validation import validate_draft_model_file

        return validate_draft_model_file(draft_path).to_payload()
    raise ValueError(f"unsupported backend workflow: {workflow!r}")
