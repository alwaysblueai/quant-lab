from __future__ import annotations

import json
from pathlib import Path

from alpha_lab.real_cases.model_factor.artifacts import write_diagnostics_artifact


def test_write_diagnostics_artifact_defaults_to_compact_json(tmp_path: Path) -> None:
    path = write_diagnostics_artifact(
        output_dir=tmp_path,
        diagnostics_payload={
            "schema_version": "1.0.0",
            "artifact_type": "alpha_lab_model_run_diagnostics",
            "generated_at_utc": "2026-04-24T00:00:00+00:00",
            "run_meta": {"case_name": "demo", "status": "succeeded"},
            "stages": [{"name": "train", "duration_ms": 123.0}],
            "events": [],
            "warnings": [],
            "data_health": {},
        },
    )

    text = path.read_text(encoding="utf-8")

    assert "\n  " not in text
    assert json.loads(text)["artifact_type"] == "alpha_lab_model_run_diagnostics"
