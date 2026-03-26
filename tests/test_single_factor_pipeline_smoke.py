from __future__ import annotations

from pathlib import Path

from alpha_lab.real_cases.single_factor.pipeline import run_single_factor_case
from tests.single_factor_case_helpers import write_demo_single_factor_case


def test_single_factor_pipeline_smoke_runs_end_to_end(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(
        tmp_path,
        factor_name="bp",
        enable_neutralization=True,
    )

    result = run_single_factor_case(spec_path)

    assert result.output_dir.exists()
    assert result.evaluation_result.experiment_result.summary.n_dates > 0
    assert result.evaluation_result.metrics["factor_name"] == "bp"
    assert result.evaluation_result.metrics["target_horizon"] == 5

    required_keys = {
        "run_manifest",
        "metrics",
        "ic_timeseries",
        "group_returns",
        "turnover",
        "coverage",
        "factor_definition",
        "summary",
        "experiment_card",
    }
    assert required_keys.issubset(set(result.artifact_paths.keys()))
    for path in result.artifact_paths.values():
        assert path.exists()
