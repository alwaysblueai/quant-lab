from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import alpha_lab.examples.profile_aware_campaign_level12 as campaign_example_module
from alpha_lab.examples.profile_aware_campaign_level12 import (
    run_profile_aware_campaign_level12_example,
)


def test_profile_aware_campaign_level12_example_runs_and_exports_comparison(
    tmp_path: Path,
) -> None:
    result = run_profile_aware_campaign_level12_example(
        output_root_dir=tmp_path / "profile_aware_campaign_example",
        profiles=("exploratory_screening", "default_research"),
        render_report=False,
        clean_output=True,
    )

    assert len(result.case_specs) == 3
    assert result.comparison_json_path.exists()
    assert result.comparison_markdown_path.exists()
    assert result.comparison_csv_path.exists()

    payload = json.loads(result.comparison_json_path.read_text(encoding="utf-8"))
    assert payload["profiles"] == ["exploratory_screening", "default_research"]
    case_evidence_index = payload.get("case_evidence_index")
    assert isinstance(case_evidence_index, dict)
    assert "case_short_window_sensitive" in case_evidence_index
    for run in payload["profile_runs"]:
        assert "level12_transition_distribution" in run

    case_rows = {row["case_name"]: row for row in payload["case_comparison"]}
    assert set(case_rows) == {
        "case_stable_promoted",
        "case_short_window_sensitive",
        "case_triage_sensitive",
    }

    stable_case = case_rows["case_stable_promoted"]
    assert stable_case["profile_sensitivity"] == "profile_sensitive"
    assert "promotion_decision" in stable_case["changed_fields"]
    assert (
        stable_case["level12_transition_profile_delta"]["delta_label"]
        == "transition_weakened_under_stricter_profile"
    )

    short_case = case_rows["case_short_window_sensitive"]
    assert "promotion_decision" in short_case["changed_fields"]
    assert "portfolio_validation_recommendation" in short_case["changed_fields"]
    assert (
        short_case["level12_transition_profile_delta"]["delta_label"]
        == "transition_weakened_under_stricter_profile"
    )

    triage_case = case_rows["case_triage_sensitive"]
    assert triage_case["profile_sensitivity"] == "profile_sensitive"
    assert "portfolio_validation_recommendation" in triage_case["changed_fields"]
    assert triage_case["level12_transition_profile_delta"]["delta_label"] == "transition_stable"

    summary = payload["campaign_level_summary"]
    assert "support_thresholds" in summary
    assert summary["stable_cases"] == []
    assert "case_short_window_sensitive" in summary["promoted_only_under_looser_profiles"]
    assert "case_stable_promoted" in summary["promoted_only_under_looser_profiles"]
    transition_by_profile = summary.get("level12_transition_distribution_by_profile")
    assert isinstance(transition_by_profile, dict)
    assert set(transition_by_profile) == {"exploratory_screening", "default_research"}
    rollups = transition_by_profile["default_research"]["reason_rollup_by_transition_label"]
    assert isinstance(rollups, dict)
    assert "Weakened at portfolio level" in rollups
    assert rollups["Weakened at portfolio level"]["reason_normalization"] == {
        "method": "explicit_rule_v1",
        "applies_to_rollup_only": True,
        "raw_reasons_preserved_at_case_level": True,
    }
    transition_delta_matrix = summary.get("level12_transition_profile_delta_matrix")
    assert isinstance(transition_delta_matrix, dict)
    pair_rows = transition_delta_matrix.get("profile_pairs")
    assert isinstance(pair_rows, list)
    assert len(pair_rows) == 1
    assert isinstance(pair_rows[0]["representative_case_names"], list)
    transition_reason_delta_matrix = summary.get("level12_transition_reason_profile_delta_matrix")
    assert isinstance(transition_reason_delta_matrix, dict)
    reason_pair_rows = transition_reason_delta_matrix.get("profile_pairs")
    assert isinstance(reason_pair_rows, list)
    assert len(reason_pair_rows) == 1
    assert isinstance(reason_pair_rows[0].get("reason_delta_by_transition_label"), dict)
    assert isinstance(reason_pair_rows[0]["representative_case_names"], list)
    assert summary["transition_stable_cases"] == ["case_triage_sensitive"]
    assert set(summary["transition_sensitive_cases"]) == {
        "case_stable_promoted",
        "case_short_window_sensitive",
    }
    assert summary["transition_delta_label_counts"] == {
        "transition_stable": 1,
        "transition_weakened_under_stricter_profile": 2,
        "transition_improved_under_profile_change": 0,
        "transition_mixed_or_nonmonotonic": 0,
    }
    compact_summary = summary.get("compact_comparison_summary")
    assert isinstance(compact_summary, dict)
    assert "minimum_support_thresholds" in compact_summary
    assert "transition_stability" in compact_summary
    assert "strongest_profile_pair_shifts" in compact_summary
    assert "weakened_fragile_reason_hotspots" in compact_summary
    assert "stricter_profile_impact" in compact_summary
    assert "artifact_pointer_hints" in compact_summary["transition_stability"]

    field_change_index = payload["field_change_index"]
    assert "case_short_window_sensitive" in field_change_index["promotion_decision"]
    assert isinstance(field_change_index["campaign_triage"], list)

    for campaign in result.profile_campaigns:
        assert len(campaign.case_summaries) == 3
        assert len(campaign.ranked_case_order) == 3
        for row in campaign.case_summaries:
            assert row.metrics_path.exists()
            assert row.run_manifest_path.exists()
            assert row.factor_definition_json_path.exists()
            assert row.signal_validation_json_path.exists()
            assert row.portfolio_recipe_json_path.exists()
            assert row.backtest_result_json_path.exists()
            assert row.case_report_path is None

    matrix_df = pd.read_csv(result.comparison_csv_path)
    assert len(matrix_df) == 6
    assert set(matrix_df["case_name"]) == {
        "case_stable_promoted",
        "case_short_window_sensitive",
        "case_triage_sensitive",
    }
    assert "level12_transition_label" in matrix_df.columns
    assert "level12_transition_delta_label" in matrix_df.columns
    markdown_text = result.comparison_markdown_path.read_text(encoding="utf-8")
    assert "Compact Comparison Summary" in markdown_text
    assert "Case Evidence Index" in markdown_text
    assert "Transition stability:" in markdown_text
    assert "representative cases" in markdown_text
    assert "dominant transition reasons" in markdown_text
    assert "dominant reason deltas by profile pair" in markdown_text
    assert "tentative due to low support" in markdown_text


def test_profile_aware_campaign_level12_example_uses_shared_key_metric_projections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"profile_summary": 0, "ranking": 0}
    original_profile_summary = campaign_example_module.project_campaign_profile_summary_metrics
    original_ranking = campaign_example_module.project_campaign_ranking_metrics

    def _track_profile_summary(metrics: dict[str, object]) -> object:
        calls["profile_summary"] += 1
        return original_profile_summary(metrics)

    def _track_ranking(metrics: dict[str, object]) -> object:
        calls["ranking"] += 1
        return original_ranking(metrics)

    monkeypatch.setattr(
        campaign_example_module,
        "project_campaign_profile_summary_metrics",
        _track_profile_summary,
    )
    monkeypatch.setattr(
        campaign_example_module,
        "project_campaign_ranking_metrics",
        _track_ranking,
    )

    run_profile_aware_campaign_level12_example(
        output_root_dir=tmp_path / "profile_aware_campaign_example_contracts",
        profiles=("exploratory_screening", "default_research"),
        render_report=False,
        clean_output=True,
    )

    assert calls["profile_summary"] >= 6
    assert calls["ranking"] >= 6
