from __future__ import annotations

import json
from pathlib import Path

import pytest

import alpha_lab.real_cases.model_factor.cli as model_factor_cli
from alpha_lab.real_cases.model_factor.pipeline import run_model_factor_case
from alpha_lab.real_cases.model_factor.spec import load_model_factor_case_spec
from tests.model_factor_case_helpers import write_demo_model_factor_case


def test_model_factor_cli_help_mentions_profile_and_level12_workflow() -> None:
    parser = model_factor_cli.build_parser()
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
    assert "Level 1/2" in parser.format_help()
    assert "canonical factor" in parser.format_help()


def test_run_model_factor_case_writes_bundle(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)

    result = run_model_factor_case(spec_path)
    case_dir = Path(spec.output.root_dir) / spec.name

    assert result.output_dir == case_dir.resolve()
    assert (case_dir / "run_manifest.json").exists()
    assert (case_dir / "metrics.json").exists()
    assert (case_dir / "model_definition.json").exists()
    assert (case_dir / "feature_manifest.json").exists()
    assert (case_dir / "training_log.csv").exists()
    assert (case_dir / "feature_importance.csv").exists()
    assert (case_dir / "purged_kfold_summary.json").exists()
    assert (case_dir / "purged_kfold_folds.csv").exists()

    manifest = json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["workflow"] == "real_case_model_factor"

    metrics_payload = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics = metrics_payload["metrics"]
    assert metrics["model_family"] == "ridge"
    assert metrics["feature_count"] == 3


def test_model_factor_cli_run_executes_and_writes_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)

    out_root = tmp_path / "cli_out"
    rc = model_factor_cli.main(["run", str(spec_path), "--output-root-dir", str(out_root)])
    assert rc == 0

    captured = capsys.readouterr()
    assert "real-case-model-factor" in captured.out
    assert "Status   : success" in captured.out
    assert "Evaluation Profile" in captured.out
    assert "Level 2 Promotion" in captured.out

    case_dir = out_root / spec.name
    assert (case_dir / "run_manifest.json").exists()
    assert (case_dir / "metrics.json").exists()
    assert (case_dir / "model_definition.json").exists()
    assert not (case_dir / "case_report.md").exists()

    manifest = json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["render_status"] == "skipped"
    assert manifest["rendered_report"] is False
    assert manifest["rendered_report_path"] is None


def test_model_factor_cli_render_report_writes_case_report(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)

    out_root = tmp_path / "cli_out"
    rc = model_factor_cli.main(
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
