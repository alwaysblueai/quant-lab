from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_lab.real_cases.composite.cli import main
from alpha_lab.real_cases.composite.spec import load_composite_case_spec
from tests.composite_case_helpers import write_demo_composite_case


def test_composite_cli_run_executes_and_writes_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    spec_path = write_demo_composite_case(tmp_path)
    spec = load_composite_case_spec(spec_path)

    out_root = tmp_path / "cli_out"
    rc = main(["run", str(spec_path), "--output-root-dir", str(out_root)])
    assert rc == 0

    captured = capsys.readouterr()
    assert "real-case-composite" in captured.out
    assert "Status   : success" in captured.out

    case_dir = out_root / spec.name
    assert (case_dir / "run_manifest.json").exists()
    assert (case_dir / "metrics.json").exists()
    assert (case_dir / "experiment_card.md").exists()
    assert not (case_dir / "case_report.md").exists()

    manifest = json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["render_status"] == "skipped"
    assert manifest["rendered_report"] is False
    assert manifest["rendered_report_path"] is None
    assert manifest["render_error"] is None


def test_composite_cli_run_with_render_report_writes_case_report(tmp_path: Path) -> None:
    spec_path = write_demo_composite_case(tmp_path)
    spec = load_composite_case_spec(spec_path)

    out_root = tmp_path / "cli_out"
    rc = main(
        [
            "run",
            str(spec_path),
            "--output-root-dir",
            str(out_root),
            "--render-report",
        ]
    )
    assert rc == 0

    case_dir = out_root / spec.name
    report_path = case_dir / "case_report.md"
    assert report_path.exists()

    manifest = json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["render_status"] == "success"
    assert manifest["rendered_report"] is True
    assert manifest["rendered_report_path"] == str(report_path.resolve())
    assert manifest["render_error"] is None
