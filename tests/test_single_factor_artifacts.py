from __future__ import annotations

import json
from pathlib import Path

from alpha_lab.real_cases.single_factor.pipeline import run_single_factor_case
from tests.single_factor_case_helpers import write_demo_single_factor_case


def test_single_factor_artifacts_have_required_files_and_fields(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="roe_ttm")
    result = run_single_factor_case(spec_path)

    output_dir = result.output_dir
    required_files = {
        "run_manifest.json",
        "metrics.json",
        "ic_timeseries.csv",
        "group_returns.csv",
        "turnover.csv",
        "coverage.csv",
        "factor_definition.yaml",
        "summary.md",
        "experiment_card.md",
    }
    present_files = {p.name for p in output_dir.iterdir() if p.is_file()}
    assert required_files.issubset(present_files)

    metrics_payload = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert "metrics" in metrics_payload
    assert "mean_rank_ic" in metrics_payload["metrics"]
    assert "mean_long_short_return" in metrics_payload["metrics"]

    manifest_payload = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest_payload["artifact_type"] == "real_case_single_factor_bundle"
    assert set(required_files).issubset(set(manifest_payload["required_bundle_files"]))

    summary_md = (output_dir / "summary.md").read_text(encoding="utf-8")
    card_md = (output_dir / "experiment_card.md").read_text(encoding="utf-8")
    assert "## Metrics" in summary_md
    assert "## Setup" in card_md
    assert "## Results" in card_md
    assert "Interpretation" in card_md
