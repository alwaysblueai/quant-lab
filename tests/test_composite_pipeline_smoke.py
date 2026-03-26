from __future__ import annotations

from pathlib import Path

from alpha_lab.real_cases.composite.pipeline import run_composite_case
from tests.composite_case_helpers import write_demo_composite_case


def test_composite_pipeline_smoke_runs_end_to_end(tmp_path: Path) -> None:
    spec_path = write_demo_composite_case(tmp_path, enable_neutralization=True)

    result = run_composite_case(spec_path)

    assert result.output_dir.exists()
    assert result.evaluation_result.experiment_result.summary.n_dates > 0
    assert result.evaluation_result.metrics["n_components"] == 3
    assert result.evaluation_result.metrics["target_horizon"] == 5

    required_keys = {
        "run_manifest",
        "metrics",
        "ic_timeseries",
        "group_returns",
        "turnover",
        "exposures",
        "composite_definition",
        "summary",
        "experiment_card",
    }
    assert required_keys.issubset(set(result.artifact_paths.keys()))
    for path in result.artifact_paths.values():
        assert path.exists()
