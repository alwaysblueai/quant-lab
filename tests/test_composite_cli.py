from __future__ import annotations

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
