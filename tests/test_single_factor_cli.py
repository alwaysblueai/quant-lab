from __future__ import annotations

import json
from pathlib import Path

import pytest

import alpha_lab.real_cases.single_factor.cli as single_factor_cli
from alpha_lab.real_cases.single_factor.spec import load_single_factor_case_spec
from tests.single_factor_case_helpers import write_demo_single_factor_case


def test_single_factor_cli_run_executes_and_writes_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")
    spec = load_single_factor_case_spec(spec_path)

    out_root = tmp_path / "cli_out"
    rc = single_factor_cli.main(["run", str(spec_path), "--output-root-dir", str(out_root)])
    assert rc == 0

    captured = capsys.readouterr()
    assert "real-case-single-factor" in captured.out
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


def test_single_factor_cli_render_report_writes_case_report(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")
    spec = load_single_factor_case_spec(spec_path)

    out_root = tmp_path / "cli_out"
    rc = single_factor_cli.main(
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


def test_single_factor_cli_render_failure_is_warning_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")
    spec = load_single_factor_case_spec(spec_path)

    def _raise_render(*args, **kwargs):
        raise RuntimeError("render failed intentionally")

    monkeypatch.setattr(single_factor_cli, "write_case_report", _raise_render)

    out_root = tmp_path / "cli_out"
    rc = single_factor_cli.main(
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
    assert (case_dir / "run_manifest.json").exists()
    assert not (case_dir / "case_report.md").exists()

    manifest = json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["render_status"] == "failed"
    assert manifest["rendered_report"] is False
    assert manifest["rendered_report_path"] is None
    assert "render failed intentionally" in str(manifest["render_error"])


def test_single_factor_cli_render_overwrite_controls_existing_report(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")
    spec = load_single_factor_case_spec(spec_path)
    out_root = tmp_path / "cli_out"

    rc_first = single_factor_cli.main(
        [
            "run",
            str(spec_path),
            "--output-root-dir",
            str(out_root),
            "--render-report",
        ]
    )
    assert rc_first == 0

    case_dir = out_root / spec.name
    report_path = case_dir / "case_report.md"
    assert report_path.exists()
    report_path.write_text("SENTINEL-CONTENT\n", encoding="utf-8")

    rc_no_overwrite = single_factor_cli.main(
        [
            "run",
            str(spec_path),
            "--output-root-dir",
            str(out_root),
            "--render-report",
        ]
    )
    assert rc_no_overwrite == 0
    assert report_path.read_text(encoding="utf-8") == "SENTINEL-CONTENT\n"

    manifest_failed = json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest_failed["render_status"] == "failed"
    assert manifest_failed["rendered_report"] is False

    rc_overwrite = single_factor_cli.main(
        [
            "run",
            str(spec_path),
            "--output-root-dir",
            str(out_root),
            "--render-report",
            "--render-overwrite",
        ]
    )
    assert rc_overwrite == 0

    report_text = report_path.read_text(encoding="utf-8")
    assert "SENTINEL-CONTENT" not in report_text
    assert report_text.startswith("# Case Report:")

    manifest_success = json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest_success["render_status"] == "success"
    assert manifest_success["rendered_report"] is True
