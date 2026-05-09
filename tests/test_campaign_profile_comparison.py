from __future__ import annotations

import json
import runpy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from alpha_lab.campaigns.profile_comparison import (
    CampaignCaseProfileSummary,
    CampaignComparisonCase,
    _build_case_evidence_index,
    _build_case_level12_transition_profile_delta,
    _build_compact_comparison_summary,
    _build_level12_transition_profile_delta_matrix,
    _build_level12_transition_reason_profile_delta_matrix,
    run_campaign_profile_comparison,
)
from alpha_lab.cli import main
from alpha_lab.key_metrics_contracts import (
    CAMPAIGN_PROFILE_COMPARISON_FIELDS,
    project_level12_transition_distribution,
)
from tests.composite_case_helpers import write_demo_composite_case
from tests.single_factor_case_helpers import write_demo_single_factor_case


def test_campaign_profile_comparison_cli_example_source_writes_outputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root_dir = tmp_path / "profile_compare_example"
    rc = main(
        [
            "campaign",
            "compare-profiles",
            "--source",
            "example",
            "--output-root-dir",
            str(output_root_dir),
            "--profiles",
            "exploratory_screening",
            "default_research",
            "--no-render-report",
        ]
    )
    assert rc == 0

    captured = capsys.readouterr()
    assert "Workflow : campaign-profile-comparison" in captured.out
    assert "Source   : example" in captured.out
    assert "Profile-Stable Cases" in captured.out
    assert "Profile-Sensitive Cases" in captured.out
    assert "Compact Comparison Summary" in captured.out
    assert "Case Evidence Index" in captured.out
    assert "Transition stability:" in captured.out
    assert "Transition Reason Delta Pairs" in captured.out

    comparison_json_path = output_root_dir / "campaign_profile_comparison.json"
    comparison_markdown_path = output_root_dir / "campaign_profile_comparison.md"
    comparison_csv_path = output_root_dir / "campaign_profile_case_matrix.csv"
    assert comparison_json_path.exists()
    assert comparison_markdown_path.exists()
    assert comparison_csv_path.exists()

    payload = json.loads(comparison_json_path.read_text(encoding="utf-8"))
    assert payload["profiles"] == ["exploratory_screening", "default_research"]
    workflow_artifacts = payload.get("workflow_closure_artifacts")
    assert isinstance(workflow_artifacts, dict)
    assert workflow_artifacts["factor_set_result_json_path"].endswith("factor_set_result.json")
    assert workflow_artifacts["candidate_recipe_generation_json_path"].endswith(
        "candidate_recipe_generation.json"
    )
    assert workflow_artifacts["winner_selection_json_path"].endswith("winner_selection.json")
    assert workflow_artifacts["next_step_recommendations_json_path"].endswith(
        "next_step_recommendations.json"
    )
    assert workflow_artifacts["artifact_load_diagnostics_json_path"].endswith(
        "artifact_load_diagnostics.json"
    )
    assert workflow_artifacts["research_artifact_manifest_json_path"].endswith(
        "research_artifact_manifest.json"
    )
    for artifact_key in (
        "factor_set_result_json_path",
        "candidate_recipe_generation_json_path",
        "winner_selection_json_path",
        "next_step_recommendations_json_path",
        "artifact_load_diagnostics_json_path",
        "research_artifact_manifest_json_path",
    ):
        assert Path(workflow_artifacts[artifact_key]).exists()
    factor_set_payload = json.loads(
        Path(workflow_artifacts["factor_set_result_json_path"]).read_text(encoding="utf-8")
    )
    assert factor_set_payload["artifact_type"] == "alpha_lab_factor_set_result"
    winner_payload = json.loads(
        Path(workflow_artifacts["winner_selection_json_path"]).read_text(encoding="utf-8")
    )
    assert winner_payload["artifact_type"] == "alpha_lab_winner_selection"
    manifest_payload = json.loads(
        Path(workflow_artifacts["research_artifact_manifest_json_path"]).read_text(encoding="utf-8")
    )
    assert manifest_payload["artifact_type"] == "alpha_lab_research_artifact_manifest"
    entries = manifest_payload["artifact_entries"]
    assert isinstance(entries, list)
    assert any(
        isinstance(item, dict)
        and item.get("artifact_name") == "artifact_load_diagnostics.json"
        and item.get("artifact_layer") == "governance"
        and item.get("required_in_strict_mode") is True
        for item in entries
    )
    assert not any(
        isinstance(item, dict)
        and item.get("artifact_name") in {"feature_oos_ic.csv", "training_metrics.csv"}
        and item.get("validation_status") == "not_emitted_v1"
        for item in entries
    )
    case_evidence_index = payload.get("case_evidence_index")
    assert isinstance(case_evidence_index, dict)
    assert "case_stable_promoted" in case_evidence_index
    stable_index = case_evidence_index["case_stable_promoted"]
    assert stable_index["profiles_observed"] == [
        "exploratory_screening",
        "default_research",
    ]
    assert "factor_verdict_by_profile" in stable_index
    assert "portfolio_robustness_by_profile" in stable_index
    assert "artifact_pointer_hints_by_profile" in stable_index
    summary = payload.get("campaign_level_summary")
    assert isinstance(summary, dict)
    assert "support_thresholds" in summary
    by_profile = summary.get("level12_transition_distribution_by_profile")
    assert isinstance(by_profile, dict)
    assert set(by_profile) == {"exploratory_screening", "default_research"}
    assert by_profile["default_research"]["support_note"] == "tentative due to low support"
    default_rollups = by_profile["default_research"]["reason_rollup_by_transition_label"]
    assert isinstance(default_rollups, dict)
    assert "Weakened at portfolio level" in default_rollups
    weakened_rollup = default_rollups["Weakened at portfolio level"]
    assert weakened_rollup["reason_normalization"] == {
        "method": "explicit_rule_v1",
        "applies_to_rollup_only": True,
        "raw_reasons_preserved_at_case_level": True,
    }
    assert "dominant_reasons" in weakened_rollup
    transition_delta_matrix = summary.get("level12_transition_profile_delta_matrix")
    assert isinstance(transition_delta_matrix, dict)
    pair_rows = transition_delta_matrix.get("profile_pairs")
    assert isinstance(pair_rows, list)
    assert len(pair_rows) == 1
    assert pair_rows[0]["minimum_support_met"] is True
    assert pair_rows[0]["support_level"] == "tentative"
    assert pair_rows[0]["profile_pair"] == "exploratory_screening -> default_research"
    assert isinstance(pair_rows[0]["representative_case_names"], list)
    assert isinstance(pair_rows[0]["artifact_pointer_hints"], list)
    transition_reason_delta_matrix = summary.get("level12_transition_reason_profile_delta_matrix")
    assert isinstance(transition_reason_delta_matrix, dict)
    reason_pair_rows = transition_reason_delta_matrix.get("profile_pairs")
    assert isinstance(reason_pair_rows, list)
    assert len(reason_pair_rows) == 1
    assert isinstance(reason_pair_rows[0].get("reason_delta_by_transition_label"), dict)
    assert reason_pair_rows[0]["minimum_support_met"] is True
    assert reason_pair_rows[0]["support_level"] == "tentative"
    assert reason_pair_rows[0]["profile_pair"] == "exploratory_screening -> default_research"
    assert isinstance(reason_pair_rows[0]["representative_case_names"], list)
    compact_summary = summary.get("compact_comparison_summary")
    assert isinstance(compact_summary, dict)
    assert "minimum_support_thresholds" in compact_summary
    assert "transition_stability" in compact_summary
    assert "most_profile_sensitive_cases" in compact_summary
    assert "strongest_profile_pair_shifts" in compact_summary
    assert "weakened_fragile_reason_hotspots" in compact_summary
    assert "stricter_profile_impact" in compact_summary
    transition_stability = compact_summary["transition_stability"]
    assert isinstance(transition_stability["representative_case_names"], list)
    assert isinstance(transition_stability["artifact_pointer_hints"], list)
    assert pair_rows[0]["from_profile"] == "exploratory_screening"
    assert pair_rows[0]["to_profile"] == "default_research"
    assert "transition_stable_cases" in summary
    assert "transition_sensitive_cases" in summary

    case_rows = {row["case_name"]: row for row in payload["case_comparison"]}
    assert case_rows["case_stable_promoted"]["level12_transition_profile_delta"]["delta_label"] in (
        "transition_stable",
        "transition_weakened_under_stricter_profile",
    )
    assert (
        case_rows["case_short_window_sensitive"]["level12_transition_profile_delta"]["delta_label"]
        == "transition_weakened_under_stricter_profile"
    )
    artifact_paths = case_rows["case_stable_promoted"]["profiles"]["default_research"][
        "artifact_paths"
    ]
    assert artifact_paths["factor_definition_json_path"].endswith("factor_definition.json")
    assert artifact_paths["signal_validation_json_path"].endswith("signal_validation.json")
    assert artifact_paths["portfolio_recipe_json_path"].endswith("portfolio_recipe.json")
    assert artifact_paths["backtest_result_json_path"].endswith("backtest_result.json")
    profile_case_rows = payload["profile_runs"][0]["case_rows"]
    assert profile_case_rows[0]["factor_definition_json_path"].endswith("factor_definition.json")
    assert profile_case_rows[0]["signal_validation_json_path"].endswith("signal_validation.json")
    assert profile_case_rows[0]["portfolio_recipe_json_path"].endswith("portfolio_recipe.json")
    assert profile_case_rows[0]["backtest_result_json_path"].endswith("backtest_result.json")

    matrix = pd.read_csv(comparison_csv_path)
    assert set(matrix["case_name"]) == {
        "case_stable_promoted",
        "case_short_window_sensitive",
        "case_triage_sensitive",
    }
    assert "level12_transition_delta_label" in matrix.columns
    markdown_text = comparison_markdown_path.read_text(encoding="utf-8")
    assert "Compact Comparison Summary" in markdown_text
    assert "Case Evidence Index" in markdown_text
    assert "Profiles observed:" in markdown_text
    assert "Transition stability:" in markdown_text
    assert "representative cases" in markdown_text
    assert "dominant transition reasons" in markdown_text
    assert "dominant reason deltas by profile pair" in markdown_text
    assert "tentative due to low support" in markdown_text
    assert "representative_cases=" in captured.out
    assert "tentative due to low support" in captured.out


def test_campaign_profile_comparison_campaign_source_writes_outputs(
    tmp_path: Path,
) -> None:
    bp_spec = write_demo_single_factor_case(tmp_path / "bp_case", factor_name="bp")
    roe_spec = write_demo_single_factor_case(tmp_path / "roe_case", factor_name="roe_ttm")
    composite_spec = write_demo_composite_case(tmp_path / "composite_case")

    _rewrite_spec_name(bp_spec, "bp_single_factor_v1")
    _rewrite_spec_name(roe_spec, "roe_ttm_single_factor_v1")
    _rewrite_spec_name(composite_spec, "value_quality_lowvol_v1")

    campaign_manifest = _write_campaign_manifest(
        tmp_path,
        bp_spec=bp_spec,
        roe_spec=roe_spec,
        composite_spec=composite_spec,
    )

    result = run_campaign_profile_comparison(
        source="campaign",
        campaign_config=campaign_manifest,
        output_root_dir=tmp_path / "profile_compare_campaign",
        profiles=("exploratory_screening", "default_research"),
        render_report=False,
        clean_output=True,
    )

    assert result.source == "campaign"
    assert result.comparison_json_path.exists()
    assert result.comparison_markdown_path.exists()
    assert result.comparison_csv_path.exists()

    payload = json.loads(result.comparison_json_path.read_text(encoding="utf-8"))
    assert payload["source"] == "campaign"
    assert payload["profiles"] == ["exploratory_screening", "default_research"]
    workflow_artifacts = payload.get("workflow_closure_artifacts")
    assert isinstance(workflow_artifacts, dict)
    for artifact_key in (
        "factor_set_result_json_path",
        "candidate_recipe_generation_json_path",
        "winner_selection_json_path",
        "next_step_recommendations_json_path",
        "artifact_load_diagnostics_json_path",
        "research_artifact_manifest_json_path",
    ):
        assert artifact_key in workflow_artifacts
        assert Path(workflow_artifacts[artifact_key]).exists()
    manifest_payload = json.loads(
        Path(workflow_artifacts["research_artifact_manifest_json_path"]).read_text(encoding="utf-8")
    )
    assert manifest_payload["artifact_type"] == "alpha_lab_research_artifact_manifest"
    entries = manifest_payload["artifact_entries"]
    assert isinstance(entries, list)
    assert any(
        isinstance(item, dict)
        and item.get("artifact_name") == "factor_definition.json"
        and item.get("artifact_layer") == "canonical"
        and isinstance(item.get("required_in_strict_mode"), bool)
        for item in entries
    )
    case_evidence_index = payload.get("case_evidence_index")
    assert isinstance(case_evidence_index, dict)
    assert len(case_evidence_index) == 3
    assert len(payload["case_comparison"]) == 3
    assert len(payload["profile_runs"]) == 2
    assert set(payload["field_change_index"]) == set(CAMPAIGN_PROFILE_COMPARISON_FIELDS)
    for run in payload["profile_runs"]:
        assert "level12_transition_distribution" in run
        first_case_row = run["case_rows"][0]
        assert first_case_row["factor_definition_json_path"].endswith("factor_definition.json")
        assert first_case_row["signal_validation_json_path"].endswith("signal_validation.json")
        assert first_case_row["portfolio_recipe_json_path"].endswith("portfolio_recipe.json")
        assert first_case_row["backtest_result_json_path"].endswith("backtest_result.json")
    summary = payload.get("campaign_level_summary")
    assert isinstance(summary, dict)
    assert "support_thresholds" in summary
    by_profile = summary.get("level12_transition_distribution_by_profile")
    assert isinstance(by_profile, dict)
    assert set(by_profile) == {"exploratory_screening", "default_research"}
    exploratory_rollups = by_profile["exploratory_screening"]["reason_rollup_by_transition_label"]
    assert isinstance(exploratory_rollups, dict)
    assert "Inconclusive transition" in exploratory_rollups
    transition_delta_matrix = summary.get("level12_transition_profile_delta_matrix")
    assert isinstance(transition_delta_matrix, dict)
    pair_rows = transition_delta_matrix.get("profile_pairs")
    assert isinstance(pair_rows, list)
    assert len(pair_rows) == 1
    assert pair_rows[0]["support_note"] == "tentative due to low support"
    assert isinstance(pair_rows[0]["representative_case_names"], list)
    assert isinstance(pair_rows[0]["representative_case_names_by_from_to_label"], dict)
    transition_reason_delta_matrix = summary.get("level12_transition_reason_profile_delta_matrix")
    assert isinstance(transition_reason_delta_matrix, dict)
    reason_pair_rows = transition_reason_delta_matrix.get("profile_pairs")
    assert isinstance(reason_pair_rows, list)
    assert len(reason_pair_rows) == 1
    assert reason_pair_rows[0]["support_note"] in {
        "tentative due to low support",
        "sparse transition evidence",
        "reason shift observed, but only in a small number of cases",
    }
    assert isinstance(reason_pair_rows[0]["representative_case_names_by_transition_label"], dict)
    compact_summary = summary.get("compact_comparison_summary")
    assert isinstance(compact_summary, dict)
    assert "minimum_support_thresholds" in compact_summary
    assert "summary_lines" in compact_summary

    matrix = pd.read_csv(result.comparison_csv_path)
    assert len(matrix) == 6
    assert set(matrix["profile_name"]) == {"exploratory_screening", "default_research"}
    assert "level12_transition_label" in matrix.columns
    assert "level12_transition_delta_label" in matrix.columns
    markdown_text = result.comparison_markdown_path.read_text(encoding="utf-8")
    assert "Compact Comparison Summary" in markdown_text
    assert "Case Evidence Index" in markdown_text
    assert "Transition stability:" in markdown_text
    assert "representative cases" in markdown_text
    assert "dominant transition reasons" in markdown_text
    assert "dominant reason deltas by profile pair" in markdown_text
    assert "tentative due to low support" in markdown_text


def test_campaign_profile_comparison_default_pair_mode_remains_adjacent(
    tmp_path: Path,
) -> None:
    result = run_campaign_profile_comparison(
        source="example",
        output_root_dir=tmp_path / "profile_compare_default_pair_mode",
        profiles=("exploratory_screening", "default_research", "stricter_research"),
        render_report=False,
        clean_output=True,
    )
    payload = json.loads(result.comparison_json_path.read_text(encoding="utf-8"))
    assert payload["pair_mode"] == "adjacent"
    pair_rows = payload["campaign_level_summary"]["level12_transition_profile_delta_matrix"][
        "profile_pairs"
    ]
    pair_names = {(row["from_profile"], row["to_profile"]) for row in pair_rows}
    assert pair_names == {
        ("exploratory_screening", "default_research"),
        ("default_research", "stricter_research"),
    }


def test_campaign_profile_comparison_all_pairs_mode_surfaces_non_adjacent_pairs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root = tmp_path / "profile_compare_all_pairs"
    rc = main(
        [
            "campaign",
            "compare-profiles",
            "--source",
            "example",
            "--output-root-dir",
            str(output_root),
            "--profiles",
            "exploratory_screening",
            "default_research",
            "stricter_research",
            "--pair-mode",
            "all_pairs",
            "--no-render-report",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "Pair Mode: all_pairs" in captured.out
    assert "exploratory_screening -> stricter_research" in captured.out

    payload = json.loads((output_root / "campaign_profile_comparison.json").read_text("utf-8"))
    assert payload["pair_mode"] == "all_pairs"

    transition_matrix = payload["campaign_level_summary"]["level12_transition_profile_delta_matrix"]
    assert transition_matrix["pair_mode"] == "all_pairs"
    transition_pairs = transition_matrix["profile_pairs"]
    transition_pair_names = {(row["from_profile"], row["to_profile"]) for row in transition_pairs}
    assert ("exploratory_screening", "stricter_research") in transition_pair_names
    non_adjacent_transition = next(
        row
        for row in transition_pairs
        if row["from_profile"] == "exploratory_screening"
        and row["to_profile"] == "stricter_research"
    )
    assert non_adjacent_transition["support_level"] in {"sparse", "tentative", "supported"}
    assert isinstance(non_adjacent_transition["support_note"], str)

    reason_matrix = payload["campaign_level_summary"][
        "level12_transition_reason_profile_delta_matrix"
    ]
    assert reason_matrix["pair_mode"] == "all_pairs"
    reason_pairs = reason_matrix["profile_pairs"]
    non_adjacent_reason = next(
        row
        for row in reason_pairs
        if row["from_profile"] == "exploratory_screening"
        and row["to_profile"] == "stricter_research"
    )
    assert non_adjacent_reason["support_level"] in {"sparse", "tentative", "supported"}
    assert isinstance(non_adjacent_reason["support_note"], str)

    markdown_text = (output_root / "campaign_profile_comparison.md").read_text("utf-8")
    assert "profile-delta matrix (all ordered profile pairs)" in markdown_text
    assert "dominant reason deltas by profile pair (all ordered profile pairs)" in markdown_text
    assert "exploratory_screening -> stricter_research" in markdown_text


def test_campaign_profile_comparison_cli_show_case_evidence_prints_compact_drill_down(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root_dir = tmp_path / "profile_compare_case_drill_down"
    rc = main(
        [
            "campaign",
            "compare-profiles",
            "--source",
            "example",
            "--output-root-dir",
            str(output_root_dir),
            "--profiles",
            "exploratory_screening",
            "default_research",
            "--no-render-report",
            "--show-case-evidence",
            "case_short_window_sensitive",
        ]
    )
    assert rc == 0

    captured = capsys.readouterr()
    assert "Case Evidence Drill-Down:" in captured.out
    assert "case: case_short_window_sensitive" in captured.out
    assert "profiles_observed:" in captured.out
    assert "factor_verdict_by_profile:" in captured.out
    assert "promotion_decision_by_profile:" in captured.out
    assert "level12_transition_label_by_profile:" in captured.out
    assert "key_reason_hints_by_profile:" in captured.out
    assert "artifact_pointer_hints_by_profile:" in captured.out


def test_campaign_profile_comparison_default_relative_hint_rendering_is_portable(
    tmp_path: Path,
) -> None:
    first_output_root = tmp_path / "profile_compare_relative_first"
    second_output_root = tmp_path / "profile_compare_relative_second"

    first = run_campaign_profile_comparison(
        source="example",
        output_root_dir=first_output_root,
        profiles=("exploratory_screening", "default_research"),
        render_report=False,
        clean_output=True,
    )
    second = run_campaign_profile_comparison(
        source="example",
        output_root_dir=second_output_root,
        profiles=("exploratory_screening", "default_research"),
        render_report=False,
        clean_output=True,
    )

    first_payload = json.loads(first.comparison_json_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second.comparison_json_path.read_text(encoding="utf-8"))

    first_index = first_payload["case_evidence_index"]["case_short_window_sensitive"]
    second_index = second_payload["case_evidence_index"]["case_short_window_sensitive"]
    first_hint = first_index["artifact_pointer_hints_by_profile"]["default_research"][0]
    second_hint = second_index["artifact_pointer_hints_by_profile"]["default_research"][0]

    assert first_payload["artifact_hint_path_mode"] == "relative"
    assert second_payload["artifact_hint_path_mode"] == "relative"
    assert first_hint == "runs/default_research/case_short_window_sensitive/metrics.json"
    assert second_hint == first_hint
    assert not first_hint.startswith("/")

    first_summary_line = first_payload["campaign_level_summary"]["compact_comparison_summary"][
        "summary_lines"
    ][0]
    second_summary_line = second_payload["campaign_level_summary"]["compact_comparison_summary"][
        "summary_lines"
    ][0]
    assert "runs/" in first_summary_line and "/metrics.json" in first_summary_line
    assert str(first_output_root.resolve()) not in first_summary_line
    assert second_summary_line == first_summary_line

    first_markdown = first.comparison_markdown_path.read_text(encoding="utf-8")
    assert (
        "Artifact hints by profile: default_research: "
        "runs/default_research/case_short_window_sensitive/metrics.json"
    ) in first_markdown


def test_campaign_profile_comparison_absolute_hint_mode_preserves_absolute_paths(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "profile_compare_absolute_hints"
    result = run_campaign_profile_comparison(
        source="example",
        output_root_dir=output_root,
        profiles=("exploratory_screening", "default_research"),
        artifact_hint_path_mode="absolute",
        render_report=False,
        clean_output=True,
    )

    payload = json.loads(result.comparison_json_path.read_text(encoding="utf-8"))
    entry = payload["case_evidence_index"]["case_short_window_sensitive"]
    hint = entry["artifact_pointer_hints_by_profile"]["default_research"][0]
    assert payload["artifact_hint_path_mode"] == "absolute"
    assert hint.startswith(str(output_root.resolve()))


def test_campaign_profile_comparison_cli_case_evidence_uses_relative_hint_paths_by_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root_dir = tmp_path / "profile_compare_relative_cli_drill_down"
    rc = main(
        [
            "campaign",
            "compare-profiles",
            "--source",
            "example",
            "--output-root-dir",
            str(output_root_dir),
            "--profiles",
            "exploratory_screening",
            "default_research",
            "--no-render-report",
            "--show-case-evidence",
            "case_short_window_sensitive",
        ]
    )
    assert rc == 0

    captured = capsys.readouterr()
    artifact_line = next(
        row for row in captured.out.splitlines() if "artifact_pointer_hints_by_profile:" in row
    )
    assert "runs/default_research/case_short_window_sensitive/metrics.json" in artifact_line
    assert str(output_root_dir.resolve()) not in artifact_line


def test_campaign_profile_comparison_cli_show_case_evidence_missing_case_is_helpful(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_root_dir = tmp_path / "profile_compare_missing_case_drill_down"
    with pytest.raises(SystemExit):
        main(
            [
                "campaign",
                "compare-profiles",
                "--source",
                "example",
                "--output-root-dir",
                str(output_root_dir),
                "--profiles",
                "exploratory_screening",
                "default_research",
                "--no-render-report",
                "--show-case-evidence",
                "does_not_exist_case",
            ]
        )

    captured = capsys.readouterr()
    assert "not found in case_evidence_index" in captured.err
    assert "Available cases:" in captured.err
    assert "case_short_window_sensitive" in captured.err


def test_profile_aware_campaign_script_remains_backward_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _fake_main(argv: list[str] | None = None) -> int:
        assert argv is None
        return 23

    monkeypatch.setattr("alpha_lab.examples.profile_aware_campaign_level12.main", _fake_main)

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path("scripts/run_profile_aware_campaign_level12_example.py", run_name="__main__")
    assert excinfo.value.code == 23


def test_level12_transition_profile_delta_matrix_counts_stable_changed_and_pairs() -> None:
    profiles = [
        "exploratory_screening",
        "default_research",
        "stricter_research",
    ]
    cases = (
        CampaignComparisonCase("stable_case", "stable", None),
        CampaignComparisonCase("weaken_case", "weaken", None),
        CampaignComparisonCase("improve_case", "improve", None),
        CampaignComparisonCase("missing_case", "missing", None),
    )
    case_lookup = {
        "stable_case": {
            "exploratory_screening": _comparison_summary(
                "stable_case",
                "exploratory_screening",
                "Weakened at portfolio level",
            ),
            "default_research": _comparison_summary(
                "stable_case",
                "default_research",
                "Weakened at portfolio level",
            ),
            "stricter_research": _comparison_summary(
                "stable_case",
                "stricter_research",
                "Weakened at portfolio level",
            ),
        },
        "weaken_case": {
            "exploratory_screening": _comparison_summary(
                "weaken_case",
                "exploratory_screening",
                "Confirmed at portfolio level",
            ),
            "default_research": _comparison_summary(
                "weaken_case",
                "default_research",
                "Weakened at portfolio level",
            ),
            "stricter_research": _comparison_summary(
                "weaken_case",
                "stricter_research",
                "Inconclusive transition",
            ),
        },
        "improve_case": {
            "exploratory_screening": _comparison_summary(
                "improve_case",
                "exploratory_screening",
                "Inconclusive transition",
            ),
            "default_research": _comparison_summary(
                "improve_case",
                "default_research",
                "Fragile after promotion",
            ),
            "stricter_research": _comparison_summary(
                "improve_case",
                "stricter_research",
                "Weakened at portfolio level",
            ),
        },
        "missing_case": {
            "exploratory_screening": _comparison_summary(
                "missing_case",
                "exploratory_screening",
                "N/A",
            ),
            "default_research": _comparison_summary(
                "missing_case",
                "default_research",
                "Inconclusive transition",
            ),
            "stricter_research": _comparison_summary(
                "missing_case",
                "stricter_research",
                "N/A",
            ),
        },
    }
    matrix = _build_level12_transition_profile_delta_matrix(
        case_specs=cases,
        case_lookup=case_lookup,
        profiles=profiles,
    )

    pair_rows = matrix["profile_pairs"]
    assert isinstance(pair_rows, list)
    assert len(pair_rows) == 2
    first_pair = pair_rows[0]
    second_pair = pair_rows[1]
    assert first_pair["from_profile"] == "exploratory_screening"
    assert first_pair["to_profile"] == "default_research"
    assert first_pair["n_cases_compared"] == 4
    assert first_pair["n_cases_with_observed_transition_labels"] == 3
    assert first_pair["n_cases_missing_transition_labels"] == 1
    assert first_pair["stable_count"] == 1
    assert first_pair["changed_count"] == 2
    assert first_pair["minimum_support_met"] is True
    assert first_pair["support_level"] == "tentative"
    assert first_pair["representative_case_names"][:2] == ["weaken_case", "improve_case"]
    counts1 = first_pair["counts_by_from_to_label"]
    rep_cases1 = first_pair["representative_case_names_by_from_to_label"]
    assert counts1["Weakened at portfolio level"]["Weakened at portfolio level"] == 1
    assert counts1["Confirmed at portfolio level"]["Weakened at portfolio level"] == 1
    assert counts1["Inconclusive transition"]["Fragile after promotion"] == 1
    assert rep_cases1["Confirmed at portfolio level"]["Weakened at portfolio level"] == [
        "weaken_case"
    ]
    assert second_pair["from_profile"] == "default_research"
    assert second_pair["to_profile"] == "stricter_research"
    assert second_pair["n_cases_with_observed_transition_labels"] == 3
    assert second_pair["stable_count"] == 1
    assert second_pair["changed_count"] == 2
    assert second_pair["minimum_support_met"] is True
    assert second_pair["support_level"] == "tentative"
    assert second_pair["representative_case_names"][0] == "weaken_case"
    counts2 = second_pair["counts_by_from_to_label"]
    assert counts2["Weakened at portfolio level"]["Weakened at portfolio level"] == 1
    assert counts2["Weakened at portfolio level"]["Inconclusive transition"] == 1
    assert counts2["Fragile after promotion"]["Weakened at portfolio level"] == 1


def test_level12_transition_profile_delta_matrix_all_pairs_includes_non_adjacent_pair() -> None:
    matrix = _build_level12_transition_profile_delta_matrix(
        case_specs=(
            CampaignComparisonCase("case_a", "demo", None),
            CampaignComparisonCase("case_b", "demo", None),
        ),
        case_lookup={
            "case_a": {
                "exploratory_screening": _comparison_summary(
                    "case_a",
                    "exploratory_screening",
                    "Confirmed at portfolio level",
                ),
                "default_research": _comparison_summary(
                    "case_a",
                    "default_research",
                    "Weakened at portfolio level",
                ),
                "stricter_research": _comparison_summary(
                    "case_a",
                    "stricter_research",
                    "Inconclusive transition",
                ),
            },
            "case_b": {
                "exploratory_screening": _comparison_summary(
                    "case_b",
                    "exploratory_screening",
                    "Weakened at portfolio level",
                ),
                "default_research": _comparison_summary(
                    "case_b",
                    "default_research",
                    "Weakened at portfolio level",
                ),
                "stricter_research": _comparison_summary(
                    "case_b",
                    "stricter_research",
                    "Weakened at portfolio level",
                ),
            },
        },
        profiles=["exploratory_screening", "default_research", "stricter_research"],
        pair_mode="all_pairs",
    )

    assert matrix["pair_mode"] == "all_pairs"
    pair_rows = matrix["profile_pairs"]
    assert len(pair_rows) == 3
    pair_names = {(row["from_profile"], row["to_profile"]) for row in pair_rows}
    assert pair_names == {
        ("exploratory_screening", "default_research"),
        ("exploratory_screening", "stricter_research"),
        ("default_research", "stricter_research"),
    }


def test_level12_transition_reason_profile_delta_matrix_builds_adjacent_pairs() -> None:
    profiles = [
        "exploratory_screening",
        "default_research",
        "stricter_research",
    ]
    transition_distribution_by_profile = {
        "exploratory_screening": project_level12_transition_distribution(
            [
                {
                    "case_name": "case_a",
                    "level12_transition_label": "Confirmed at portfolio level",
                    "level12_transition_reasons": ["promotion decision: promote to level 2"],
                }
            ]
        ),
        "default_research": project_level12_transition_distribution(
            [
                {
                    "case_name": "case_a",
                    "level12_transition_label": "Confirmed at portfolio level",
                    "level12_transition_reasons": ["promotion decision: promote to level 2"],
                }
            ]
        ),
        "stricter_research": project_level12_transition_distribution(
            [
                {
                    "case_name": "case_a",
                    "level12_transition_label": "Weakened at portfolio level",
                    "level12_transition_reasons": ["promotion decision: hold for refinement"],
                }
            ]
        ),
    }

    matrix = _build_level12_transition_reason_profile_delta_matrix(
        transition_distribution_by_profile=transition_distribution_by_profile,
        profiles=profiles,
    )
    pair_rows = matrix["profile_pairs"]
    assert isinstance(pair_rows, list)
    assert len(pair_rows) == 2
    assert pair_rows[0]["from_profile"] == "exploratory_screening"
    assert pair_rows[0]["to_profile"] == "default_research"
    assert pair_rows[1]["from_profile"] == "default_research"
    assert pair_rows[1]["to_profile"] == "stricter_research"


def test_level12_transition_reason_profile_delta_matrix_all_pairs_includes_non_adjacent_pair() -> (
    None
):
    profiles = [
        "exploratory_screening",
        "default_research",
        "stricter_research",
    ]
    transition_distribution_by_profile = {
        "exploratory_screening": project_level12_transition_distribution(
            [
                {
                    "case_name": "case_a",
                    "level12_transition_label": "Confirmed at portfolio level",
                    "level12_transition_reasons": ["promotion decision: promote to level 2"],
                }
            ]
        ),
        "default_research": project_level12_transition_distribution(
            [
                {
                    "case_name": "case_a",
                    "level12_transition_label": "Confirmed at portfolio level",
                    "level12_transition_reasons": ["promotion decision: promote to level 2"],
                }
            ]
        ),
        "stricter_research": project_level12_transition_distribution(
            [
                {
                    "case_name": "case_a",
                    "level12_transition_label": "Weakened at portfolio level",
                    "level12_transition_reasons": ["promotion decision: hold for refinement"],
                }
            ]
        ),
    }

    matrix = _build_level12_transition_reason_profile_delta_matrix(
        transition_distribution_by_profile=transition_distribution_by_profile,
        profiles=profiles,
        pair_mode="all_pairs",
    )
    assert matrix["pair_mode"] == "all_pairs"
    pair_rows = matrix["profile_pairs"]
    assert len(pair_rows) == 3
    pair_names = {(row["from_profile"], row["to_profile"]) for row in pair_rows}
    assert pair_names == {
        ("exploratory_screening", "default_research"),
        ("exploratory_screening", "stricter_research"),
        ("default_research", "stricter_research"),
    }


def test_level12_transition_reason_profile_delta_matrix_tracks_shifted_and_stable_buckets() -> None:
    from_rows = [
        {
            "case_name": "confirmed_1",
            "level12_transition_label": "Confirmed at portfolio level",
            "level12_transition_reasons": [
                "promotion decision: promote to level 2",
                "promotion reason: benchmark-relative risk is elevated",
            ],
        },
        {
            "case_name": "confirmed_2",
            "level12_transition_label": "Confirmed at portfolio level",
            "level12_transition_reasons": ["promotion decision: promote to level 2"],
        },
        {
            "case_name": "confirmed_3",
            "level12_transition_label": "Confirmed at portfolio level",
            "level12_transition_reasons": ["fragility: concentration risk is elevated"],
        },
        {
            "case_name": "confirmed_4",
            "level12_transition_label": "Confirmed at portfolio level",
            "level12_transition_reasons": [],
        },
        {
            "case_name": "weakened_1",
            "level12_transition_label": "Weakened at portfolio level",
            "level12_transition_reasons": ["cost sensitivity is elevated"],
        },
        {
            "case_name": "weakened_2",
            "level12_transition_label": "Weakened at portfolio level",
            "level12_transition_reasons": ["benchmark-relative risk is elevated"],
        },
    ]
    to_rows = [
        {
            "case_name": "confirmed_1",
            "level12_transition_label": "Confirmed at portfolio level",
            "level12_transition_reasons": [
                "promotion decision: Promote to Level 2",
                "fragility: concentration risk is elevated",
                "portfolio recommendation: credible at portfolio level",
            ],
        },
        {
            "case_name": "confirmed_2",
            "level12_transition_label": "Confirmed at portfolio level",
            "level12_transition_reasons": [
                "promotion decision: Promote to Level 2",
                "fragility: concentration risk is elevated",
            ],
        },
        {
            "case_name": "confirmed_3",
            "level12_transition_label": "Confirmed at portfolio level",
            "level12_transition_reasons": ["fragility: concentration risk is elevated"],
        },
        {
            "case_name": "confirmed_4",
            "level12_transition_label": "Confirmed at portfolio level",
            "level12_transition_reasons": [],
        },
        {
            "case_name": "weakened_1",
            "level12_transition_label": "Weakened at portfolio level",
            "level12_transition_reasons": ["cost sensitivity is elevated"],
        },
        {
            "case_name": "weakened_2",
            "level12_transition_label": "Weakened at portfolio level",
            "level12_transition_reasons": ["benchmark-relative risk is elevated"],
        },
    ]

    matrix = _build_level12_transition_reason_profile_delta_matrix(
        transition_distribution_by_profile={
            "exploratory_screening": project_level12_transition_distribution(from_rows),
            "default_research": project_level12_transition_distribution(to_rows),
        },
        profiles=["exploratory_screening", "default_research"],
    )

    pair_rows = matrix["profile_pairs"]
    assert isinstance(pair_rows, list)
    assert len(pair_rows) == 1
    pair = pair_rows[0]
    assert pair["n_transition_labels_with_observed_reasons"] == 2
    assert pair["n_transition_labels_with_reason_shift"] == 1
    assert pair["n_transition_labels_reason_stable"] == 1
    assert pair["n_transition_labels_with_tentative_reason_shift"] == 0
    assert pair["minimum_support_met"] is True
    assert pair["profile_pair"] == "exploratory_screening -> default_research"
    assert "confirmed_1" in pair["representative_case_names"]
    assert pair["reason_bucket_delta_counts"] == {
        "added": 1,
        "removed": 0,
        "increased": 0,
        "decreased": 0,
        "stable": 1,
    }

    by_label = pair["reason_delta_by_transition_label"]
    assert isinstance(by_label, dict)
    confirmed = by_label["Confirmed at portfolio level"]
    weakened = by_label["Weakened at portfolio level"]
    assert confirmed["is_reason_shifted"] is True
    assert confirmed["is_reason_shift_tentative"] is False
    assert "confirmed_1" in confirmed["representative_case_names"]
    assert "confirmed_1" in confirmed["supporting_case_names"]
    assert weakened["is_reason_shifted"] is False
    assert weakened["is_reason_shift_tentative"] is False

    confirmed_from = confirmed["from_profile_dominant_reasons"]
    assert confirmed_from[0]["reason"] == "promotion decision: Promote to Level 2"
    assert "confirmed_1" in confirmed_from[0]["supporting_case_names"]
    confirmed_deltas = confirmed["reason_bucket_deltas"]
    assert confirmed_deltas["added"][0]["reason"] == "concentration risk is elevated"
    assert "confirmed_1" in confirmed_deltas["added"][0]["supporting_case_names"]
    assert confirmed_deltas["removed"] == []
    assert confirmed_deltas["increased"] == []
    assert confirmed_deltas["stable"][0]["reason"] == "promotion decision: Promote to Level 2"


def test_case_evidence_index_builds_compact_cross_profile_entries() -> None:
    case_comparison = [
        {
            "case_name": "case_alpha",
            "profile_sensitivity": "profile_sensitive",
            "changed_fields": ["campaign_triage", "promotion_decision"],
            "level12_transition_profile_delta": {
                "delta_label": "transition_weakened_under_stricter_profile",
                "profile_transition_labels": {
                    "exploratory_screening": "Confirmed at portfolio level",
                    "default_research": "Weakened at portfolio level",
                },
            },
            "profiles": {
                "exploratory_screening": {
                    "factor_verdict": "Strong candidate",
                    "campaign_triage": "Advance to Level 2",
                    "promotion_decision": "Promote to Level 2",
                    "portfolio_validation_recommendation": "Credible at portfolio level",
                    "level12_transition_label": "Confirmed at portfolio level",
                    "level12_transition_reasons": [
                        "reason_1",
                        "reason_2",
                        "reason_3",
                        "reason_4",
                    ],
                    "artifact_paths": {
                        "metrics_path": "/tmp/case_alpha/exploratory/metrics.json",
                        "summary_path": "/tmp/case_alpha/exploratory/summary.md",
                        "output_dir": "/tmp/case_alpha/exploratory",
                    },
                },
                "default_research": {
                    "factor_verdict": "Mixed evidence",
                    "campaign_triage": "Needs refinement",
                    "promotion_decision": "Hold for refinement",
                    "portfolio_validation_recommendation": "Needs portfolio refinement",
                    "level12_transition_label": "Weakened at portfolio level",
                    "level12_transition_reasons": ["reason_x", "reason_y"],
                    "artifact_paths": {
                        "metrics_path": "/tmp/case_alpha/default/metrics.json",
                    },
                },
            },
        }
    ]
    index = _build_case_evidence_index(
        case_comparison=case_comparison,
        profiles=["exploratory_screening", "default_research"],
    )

    entry = index["case_alpha"]
    assert entry["profiles_observed"] == ["exploratory_screening", "default_research"]
    assert entry["profile_delta_label"] == "transition_weakened_under_stricter_profile"
    assert entry["factor_verdict_by_profile"]["exploratory_screening"] == "Strong candidate"
    assert entry["promotion_decision_by_profile"]["default_research"] == "Hold for refinement"
    assert (
        entry["portfolio_robustness_by_profile"]["default_research"] == "Needs portfolio refinement"
    )
    assert (
        entry["level12_transition_label_by_profile"]["exploratory_screening"]
        == "Confirmed at portfolio level"
    )
    assert entry["key_reason_hints_by_profile"]["exploratory_screening"] == [
        "reason_1",
        "reason_2",
        "reason_3",
    ]
    assert entry["artifact_pointer_hints_by_profile"]["exploratory_screening"] == [
        "/tmp/case_alpha/exploratory/metrics.json",
        "/tmp/case_alpha/exploratory/summary.md",
    ]


def test_compact_comparison_summary_surfaces_high_signal_auditable_fields() -> None:
    case_comparison = [
        {
            "case_name": "case_stable",
            "profile_sensitivity": "profile_stable",
            "changed_fields": [],
            "level12_transition_profile_delta": {
                "delta_label": "transition_stable",
                "profile_transition_labels": {
                    "exploratory_screening": "Confirmed at portfolio level",
                    "default_research": "Confirmed at portfolio level",
                },
            },
        },
        {
            "case_name": "case_sensitive",
            "profile_sensitivity": "highly_profile_sensitive",
            "changed_fields": [
                "factor_verdict",
                "promotion_decision",
                "portfolio_validation_recommendation",
            ],
            "level12_transition_profile_delta": {
                "delta_label": "transition_weakened_under_stricter_profile",
                "profile_transition_labels": {
                    "exploratory_screening": "Confirmed at portfolio level",
                    "default_research": "Weakened at portfolio level",
                },
            },
        },
        {
            "case_name": "case_triage_sensitive",
            "profile_sensitivity": "profile_sensitive",
            "changed_fields": ["campaign_triage"],
            "level12_transition_profile_delta": {
                "delta_label": "transition_stable",
                "profile_transition_labels": {
                    "exploratory_screening": "Fragile after promotion",
                    "default_research": "Fragile after promotion",
                },
            },
        },
    ]
    transition_distribution_by_profile = {
        "default_research": project_level12_transition_distribution(
            [
                {
                    "case_name": "case_sensitive",
                    "level12_transition_label": "Weakened at portfolio level",
                    "level12_transition_reasons": ["cost sensitivity is elevated"],
                },
                {
                    "case_name": "case_triage_sensitive",
                    "level12_transition_label": "Fragile after promotion",
                    "level12_transition_reasons": ["fragility: concentration risk is elevated"],
                },
            ]
        )
    }
    transition_profile_delta_matrix = {
        "profile_pairs": [
            {
                "from_profile": "exploratory_screening",
                "to_profile": "default_research",
                "n_cases_with_observed_transition_labels": 3,
                "changed_count": 1,
                "counts_by_from_to_label": {
                    "Confirmed at portfolio level": {
                        "Confirmed at portfolio level": 1,
                        "Weakened at portfolio level": 1,
                    },
                    "Fragile after promotion": {"Fragile after promotion": 1},
                },
            }
        ]
    }
    transition_reason_profile_delta_matrix = {
        "profile_pairs": [
            {
                "from_profile": "exploratory_screening",
                "to_profile": "default_research",
                "n_transition_labels_with_observed_reasons": 2,
                "n_transition_labels_with_reason_shift": 1,
            }
        ]
    }

    compact_summary = _build_compact_comparison_summary(
        case_comparison=case_comparison,
        profiles=["exploratory_screening", "default_research"],
        transition_distribution_by_profile=transition_distribution_by_profile,
        transition_profile_delta_matrix=transition_profile_delta_matrix,
        transition_reason_profile_delta_matrix=transition_reason_profile_delta_matrix,
        transition_stable_cases=["case_stable", "case_triage_sensitive"],
        transition_sensitive_cases=["case_sensitive"],
    )

    transition_stability = compact_summary["transition_stability"]
    assert transition_stability["n_cases"] == 3
    assert transition_stability["n_transition_stable_cases"] == 2
    assert transition_stability["n_transition_sensitive_cases"] == 1
    assert transition_stability["support_level"] == "tentative"
    assert transition_stability["support_note"] == "tentative due to low support"

    sensitive_cases = compact_summary["most_profile_sensitive_cases"]
    assert sensitive_cases[0]["case_name"] == "case_sensitive"
    assert sensitive_cases[0]["n_changed_fields"] == 3

    strongest_pair = compact_summary["strongest_profile_pair_shifts"][0]
    assert strongest_pair["from_profile"] == "exploratory_screening"
    assert strongest_pair["to_profile"] == "default_research"
    assert strongest_pair["changed_count"] == 1
    assert strongest_pair["minimum_support_met"] is True
    assert strongest_pair["support_level"] == "tentative"
    assert strongest_pair["profile_pair"] == "exploratory_screening -> default_research"
    assert isinstance(strongest_pair["representative_case_names"], list)
    assert strongest_pair["top_shift_flows"][0]["from_label"] == ("Confirmed at portfolio level")
    assert strongest_pair["top_shift_flows"][0]["to_label"] == "Weakened at portfolio level"
    assert isinstance(strongest_pair["top_shift_flows"][0]["representative_case_names"], list)

    weakened_fragile = compact_summary["weakened_fragile_reason_hotspots"]
    assert weakened_fragile["profile_name"] == "default_research"
    assert weakened_fragile["top_reasons"][0]["transition_label"] in {
        "Weakened at portfolio level",
        "Fragile after promotion",
    }
    assert isinstance(weakened_fragile["top_reasons"][0]["supporting_case_names"], list)

    stricter_impact = compact_summary["stricter_profile_impact"]["aggregate"]
    assert stricter_impact["dominant_reduction_mode"] == "robustness"
    assert stricter_impact["promotion_reduction_count"] == 0
    assert stricter_impact["robustness_reduction_count"] == 1

    summary_lines = compact_summary["summary_lines"]
    assert any("Transition stability:" in row for row in summary_lines)
    assert any("Strongest profile-pair shift:" in row for row in summary_lines)
    assert any("Stricter profile impact:" in row for row in summary_lines)
    assert any("tentative due to low support" in row for row in summary_lines)


def test_transition_profile_delta_matrix_marks_sparse_support_when_pair_is_small() -> None:
    matrix = _build_level12_transition_profile_delta_matrix(
        case_specs=(
            CampaignComparisonCase("case_a", "demo", None),
            CampaignComparisonCase("case_b", "demo", None),
        ),
        case_lookup={
            "case_a": {
                "exploratory_screening": _comparison_summary(
                    "case_a",
                    "exploratory_screening",
                    "Confirmed at portfolio level",
                ),
                "default_research": _comparison_summary(
                    "case_a",
                    "default_research",
                    "Weakened at portfolio level",
                ),
            },
            "case_b": {
                "exploratory_screening": _comparison_summary(
                    "case_b",
                    "exploratory_screening",
                    "Confirmed at portfolio level",
                ),
                "default_research": _comparison_summary(
                    "case_b",
                    "default_research",
                    "Confirmed at portfolio level",
                ),
            },
        },
        profiles=["exploratory_screening", "default_research"],
    )

    pair = matrix["profile_pairs"][0]
    assert pair["n_cases_with_observed_transition_labels"] == 2
    assert pair["minimum_support_met"] is False
    assert pair["is_sparse"] is True
    assert pair["support_note"] == "sparse transition evidence"


def test_transition_reason_profile_delta_matrix_marks_tentative_sparse_shift() -> None:
    matrix = _build_level12_transition_reason_profile_delta_matrix(
        transition_distribution_by_profile={
            "exploratory_screening": project_level12_transition_distribution(
                [
                    {
                        "case_name": "case_a",
                        "level12_transition_label": "Confirmed at portfolio level",
                        "level12_transition_reasons": ["cost sensitivity is elevated"],
                    }
                ]
            ),
            "default_research": project_level12_transition_distribution(
                [
                    {
                        "case_name": "case_a",
                        "level12_transition_label": "Confirmed at portfolio level",
                        "level12_transition_reasons": ["benchmark-relative risk is elevated"],
                    },
                    {
                        "case_name": "case_b",
                        "level12_transition_label": "Confirmed at portfolio level",
                        "level12_transition_reasons": ["benchmark-relative risk is elevated"],
                    },
                    {
                        "case_name": "case_c",
                        "level12_transition_label": "Confirmed at portfolio level",
                        "level12_transition_reasons": ["benchmark-relative risk is elevated"],
                    },
                ]
            ),
        },
        profiles=["exploratory_screening", "default_research"],
    )

    pair = matrix["profile_pairs"][0]
    assert pair["n_transition_labels_with_reason_shift"] == 0
    assert pair["n_transition_labels_with_tentative_reason_shift"] == 1
    assert pair["support_note"] == "reason shift observed, but only in a small number of cases"
    confirmed = pair["reason_delta_by_transition_label"]["Confirmed at portfolio level"]
    assert confirmed["is_reason_shifted"] is False
    assert confirmed["is_reason_shift_tentative"] is True
    assert confirmed["minimum_support_met"] is False


def test_case_level_transition_delta_labels_cover_monotone_and_nonmonotonic_paths() -> None:
    profiles = [
        "exploratory_screening",
        "default_research",
        "stricter_research",
    ]
    stable = _build_case_level12_transition_profile_delta(
        {
            "exploratory_screening": _comparison_summary(
                "stable_case",
                "exploratory_screening",
                "Weakened at portfolio level",
            ),
            "default_research": _comparison_summary(
                "stable_case",
                "default_research",
                "Weakened at portfolio level",
            ),
            "stricter_research": _comparison_summary(
                "stable_case",
                "stricter_research",
                "Weakened at portfolio level",
            ),
        },
        profiles=profiles,
    )
    weakened = _build_case_level12_transition_profile_delta(
        {
            "exploratory_screening": _comparison_summary(
                "weaken_case",
                "exploratory_screening",
                "Confirmed at portfolio level",
            ),
            "default_research": _comparison_summary(
                "weaken_case",
                "default_research",
                "Weakened at portfolio level",
            ),
            "stricter_research": _comparison_summary(
                "weaken_case",
                "stricter_research",
                "Inconclusive transition",
            ),
        },
        profiles=profiles,
    )
    improved = _build_case_level12_transition_profile_delta(
        {
            "exploratory_screening": _comparison_summary(
                "improve_case",
                "exploratory_screening",
                "Inconclusive transition",
            ),
            "default_research": _comparison_summary(
                "improve_case",
                "default_research",
                "Fragile after promotion",
            ),
            "stricter_research": _comparison_summary(
                "improve_case",
                "stricter_research",
                "Weakened at portfolio level",
            ),
        },
        profiles=profiles,
    )
    mixed = _build_case_level12_transition_profile_delta(
        {
            "exploratory_screening": _comparison_summary(
                "mixed_case",
                "exploratory_screening",
                "Weakened at portfolio level",
            ),
            "default_research": _comparison_summary(
                "mixed_case",
                "default_research",
                "Improved at portfolio level",
            ),
            "stricter_research": _comparison_summary(
                "mixed_case",
                "stricter_research",
                "Inconclusive transition",
            ),
        },
        profiles=profiles,
    )

    assert stable["delta_label"] == "transition_stable"
    assert weakened["delta_label"] == "transition_weakened_under_stricter_profile"
    assert improved["delta_label"] == "transition_improved_under_profile_change"
    assert mixed["delta_label"] == "transition_mixed_or_nonmonotonic"


def _rewrite_spec_name(spec_path: Path, case_name: str) -> None:
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    payload["name"] = case_name
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_campaign_manifest(
    tmp_path: Path,
    *,
    bp_spec: Path,
    roe_spec: Path,
    composite_spec: Path,
) -> Path:
    payload = {
        "campaign_name": "research_campaign_1",
        "campaign_description": "Synthetic smoke campaign for profile comparison.",
        "output_root_dir": str(tmp_path / "campaign_outputs"),
        "case_output_root_dir": str(tmp_path / "real_case_outputs"),
        "vault_export": {"vault_root": None, "mode": "skip"},
        "cases": [
            {
                "case_name": "bp_single_factor_v1",
                "package_type": "single_factor",
                "spec_path": str(bp_spec),
            },
            {
                "case_name": "roe_ttm_single_factor_v1",
                "package_type": "single_factor",
                "spec_path": str(roe_spec),
            },
            {
                "case_name": "value_quality_lowvol_v1",
                "package_type": "composite",
                "spec_path": str(composite_spec),
            },
        ],
        "execution_order": [
            "bp_single_factor_v1",
            "roe_ttm_single_factor_v1",
            "value_quality_lowvol_v1",
        ],
    }

    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _comparison_summary(
    case_name: str,
    profile_name: str,
    transition_label: str,
) -> CampaignCaseProfileSummary:
    return CampaignCaseProfileSummary(
        case_name=case_name,
        profile_name=profile_name,
        status="success",
        output_dir=None,
        run_manifest_path=None,
        metrics_path=None,
        summary_path=None,
        experiment_card_path=None,
        factor_verdict="Strong candidate",
        factor_verdict_reasons=(),
        campaign_triage="Advance to Level 2",
        campaign_triage_reasons=(),
        promotion_decision="Promote to Level 2",
        promotion_reasons=(),
        promotion_blockers=(),
        level12_transition_label=transition_label,
        level12_transition_reasons=(),
        portfolio_validation_status="completed",
        portfolio_validation_recommendation="Needs portfolio refinement",
        portfolio_validation_major_risks=(),
    )
