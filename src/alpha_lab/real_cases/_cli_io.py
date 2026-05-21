"""Shared CLI post-run helpers for real-case pipelines.

`render_case_report` and `update_run_manifest` had byte-identical copies
in ``single_factor/cli.py``, ``model_factor/cli.py``, and ``composite/cli.py``.
This module is the single source of truth.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from alpha_lab.artifact_contracts import validate_level12_artifact_payload
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
