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
    assert "neutralization_comparison" in result.evaluation_result.metrics
    assert "neutralization_comparison_flags" in result.evaluation_result.metrics
    assert "neutralization_mean_ic_delta" in result.evaluation_result.metrics
    assert "promotion_decision" in result.evaluation_result.metrics
    assert "promotion_reasons" in result.evaluation_result.metrics
    assert "promotion_blockers" in result.evaluation_result.metrics

    required_keys = {
        "run_manifest",
        "metrics",
        "factor_definition_json",
        "signal_validation_json",
        "portfolio_recipe_json",
        "backtest_result_json",
        "ic_timeseries",
        "rolling_stability",
        "group_returns",
        "turnover",
        "exposures",
        "composite_definition",
        "summary",
        "experiment_card",
        "integrity_report_json",
        "integrity_report_markdown",
    }
    assert required_keys.issubset(set(result.artifact_paths.keys()))
    for path in result.artifact_paths.values():
        assert path.exists()
