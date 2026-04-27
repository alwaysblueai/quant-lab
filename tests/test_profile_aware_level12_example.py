from __future__ import annotations

import json
from pathlib import Path

import pytest

import alpha_lab.examples.profile_aware_level12 as profile_example_module
from alpha_lab.examples.profile_aware_level12 import run_profile_aware_level12_example


def test_profile_aware_level12_example_runs_and_exports_comparison(tmp_path: Path) -> None:
    result = run_profile_aware_level12_example(
        output_root_dir=tmp_path / "profile_aware_example",
        profiles=("exploratory_screening", "default_research"),
        render_report=True,
        clean_output=True,
    )

    assert result.spec_path.exists()
    assert result.comparison_json_path.exists()
    assert result.comparison_markdown_path.exists()
    assert len(result.profile_runs) == 2

    by_profile = {row.profile_name: row for row in result.profile_runs}
    assert set(by_profile) == {"exploratory_screening", "default_research"}

    for run in result.profile_runs:
        assert run.output_dir.exists()
        assert run.run_manifest_path.exists()
        assert run.metrics_path.exists()
        assert run.summary_path.exists()
        assert run.experiment_card_path.exists()
        assert run.case_report_path is not None
        assert run.case_report_path.exists()

        metrics_payload = json.loads(run.metrics_path.read_text(encoding="utf-8"))
        assert metrics_payload["metrics"]["research_evaluation_profile"] == run.profile_name

    # Regime-conditional analysis may cause both profiles to land on the same
    # triage label, so we only check that both labels are valid taxonomy members.
    from alpha_lab.reporting.campaign_triage import CAMPAIGN_TRIAGE_TAXONOMY

    assert by_profile["exploratory_screening"].campaign_triage in CAMPAIGN_TRIAGE_TAXONOMY
    assert by_profile["default_research"].campaign_triage in CAMPAIGN_TRIAGE_TAXONOMY
    assert (
        by_profile["exploratory_screening"].portfolio_validation_status
        != by_profile["default_research"].portfolio_validation_status
    )

    comparison_payload = json.loads(result.comparison_json_path.read_text(encoding="utf-8"))
    assert comparison_payload["profiles"] == ["exploratory_screening", "default_research"]
    assert comparison_payload["case_name"] == "profile_aware_bp_single_factor"
    differences = comparison_payload["observed_differences"]
    # campaign_triage may or may not differ depending on regime analysis
    assert "portfolio_validation_status" in differences


def test_profile_aware_level12_example_uses_profile_summary_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"profile_summary": 0}
    original = profile_example_module.project_campaign_profile_summary_metrics

    def _track_profile_summary(metrics: dict[str, object]) -> object:
        calls["profile_summary"] += 1
        return original(metrics)

    monkeypatch.setattr(
        profile_example_module,
        "project_campaign_profile_summary_metrics",
        _track_profile_summary,
    )

    run_profile_aware_level12_example(
        output_root_dir=tmp_path / "profile_aware_example_contracts",
        profiles=("exploratory_screening", "default_research"),
        render_report=False,
        clean_output=True,
    )

    assert calls["profile_summary"] >= 2
