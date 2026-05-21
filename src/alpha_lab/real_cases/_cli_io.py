"""Shared CLI post-run helpers for real-case pipelines.

``render_case_report``, ``update_run_manifest`` and
``finalize_contract_if_research_draft`` are the post-run hooks shared by
``single_factor/cli.py``, ``model_factor/cli.py`` and (for the rendering /
manifest helpers) ``composite/cli.py``.
"""

from __future__ import annotations

import json
import logging
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
        try:
            write_case_report(output_dir, overwrite=False)
        except Exception as exc:  # noqa: BLE001
            _logger.warning(
                "backend contract: case_report render failed for %s: %s",
                output_dir,
                exc,
            )
    receipt = finalize_backend_contract(
        output_dir,
        workflow=workflow,
        draft_source_path=draft_path,
        case_spec_path=case_spec_path,
        evaluation_profile=evaluation_profile,
        command=command,
    )
    return 0 if str(receipt.get("status") or "") == "success" else 1
