from __future__ import annotations

import json
from pathlib import Path

import pytest

from alpha_lab.real_cases.composite.cli import main
from alpha_lab.real_cases.composite.spec import load_composite_case_spec
from tests.composite_case_helpers import write_demo_composite_case


def test_composite_cli_help_mentions_profile_and_level2_workflow() -> None:
    from alpha_lab.real_cases.composite.cli import build_parser

    parser = build_parser()
    run_help = next(
        action.choices["run"].format_help()
        for action in parser._actions
        if (
            hasattr(action, "choices")
            and isinstance(action.choices, dict)
            and "run" in action.choices
        )
    )
    assert "--evaluation-profile" in run_help
    assert "exploratory_screening" in run_help
    assert "stricter_research" in run_help
    assert "Level 1/2" in parser.format_help()
    assert "promotion gate" in parser.format_help()


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
    assert "Evaluation Profile" in captured.out
    assert "Campaign Triage" in captured.out
    assert "Level 2 Promotion" in captured.out
    assert "Level 2 Validation" in captured.out

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
    assert manifest["evaluation_standard"]["profile_name"] == "default_research"


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


def test_composite_cli_rejects_unknown_evaluation_profile(tmp_path: Path) -> None:
    spec_path = write_demo_composite_case(tmp_path)
    with pytest.raises(SystemExit):
        main(
            [
                "run",
                str(spec_path),
                "--evaluation-profile",
                "unknown_profile",
            ]
        )
