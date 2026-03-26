from __future__ import annotations

from pathlib import Path

import pytest

from alpha_lab.real_cases.single_factor.cli import main
from alpha_lab.real_cases.single_factor.spec import load_single_factor_case_spec
from tests.single_factor_case_helpers import write_demo_single_factor_case


def test_single_factor_cli_run_executes_and_writes_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")
    spec = load_single_factor_case_spec(spec_path)

    out_root = tmp_path / "cli_out"
    rc = main(["run", str(spec_path), "--output-root-dir", str(out_root)])
    assert rc == 0

    captured = capsys.readouterr()
    assert "real-case-single-factor" in captured.out
    assert "Status   : success" in captured.out

    case_dir = out_root / spec.name
    assert (case_dir / "run_manifest.json").exists()
    assert (case_dir / "metrics.json").exists()
    assert (case_dir / "experiment_card.md").exists()
