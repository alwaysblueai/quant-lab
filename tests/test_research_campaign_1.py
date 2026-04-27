from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import alpha_lab.campaigns.research_campaign_1 as research_campaign_1
from tests.composite_case_helpers import write_demo_composite_case
from tests.single_factor_case_helpers import write_demo_single_factor_case


def test_research_campaign_1_help_mentions_profile_and_level2_workflow() -> None:
    help_text = research_campaign_1.build_parser().format_help()
    assert "--evaluation-profile" in help_text
    assert "exploratory_screening" in help_text
    assert "stricter_research" in help_text
    assert "Level 2 portfolio" in help_text


def test_research_campaign_1_smoke_runs_and_writes_required_outputs(tmp_path: Path) -> None:
    bp_spec = write_demo_single_factor_case(tmp_path / "bp_case", factor_name="bp")
    roe_spec = write_demo_single_factor_case(tmp_path / "roe_case", factor_name="roe_ttm")
    comp_spec = write_demo_composite_case(tmp_path / "composite_case")

    _rewrite_spec_name(bp_spec, "bp_single_factor_v1")
    _rewrite_spec_name(roe_spec, "roe_ttm_single_factor_v1")
    _rewrite_spec_name(comp_spec, "value_quality_lowvol_v1")

    campaign_path = _write_campaign_manifest(
        tmp_path,
        bp_spec=bp_spec,
        roe_spec=roe_spec,
        comp_spec=comp_spec,
    )

    result = research_campaign_1.run_research_campaign_1(campaign_path)

    assert result.output_dir.exists()
    required = {
        "campaign_manifest",
        "campaign_results",
        "campaign_summary",
        "campaign_index",
    }
    assert required.issubset(set(result.artifact_paths.keys()))
    for path in result.artifact_paths.values():
        assert path.exists()

    payload = json.loads(result.artifact_paths["campaign_results"].read_text(encoding="utf-8"))
    assert payload["campaign_name"] == "research_campaign_1"
    assert payload["evaluation_profile"] == "default_research"
    assert payload["n_cases"] == 3
    assert payload["n_success"] == 3
    assert payload["n_failed"] == 0
    transition_distribution = payload.get("level12_transition_distribution")
    assert isinstance(transition_distribution, dict)
    assert transition_distribution.get("n_cases") == 3
    assert isinstance(transition_distribution.get("counts_by_transition_label"), dict)
    assert isinstance(transition_distribution.get("proportions_by_transition_label"), dict)
    assert isinstance(transition_distribution.get("reason_rollup_by_transition_label"), dict)
    assert isinstance(transition_distribution.get("interpretation"), str)
    manifest_payload = json.loads(
        result.artifact_paths["campaign_manifest"].read_text(encoding="utf-8")
    )
    assert manifest_payload["evaluation_standard"]["profile_name"] == "default_research"
    summary_text = result.artifact_paths["campaign_summary"].read_text(encoding="utf-8")
    assert "Level 1->Level 2 Transition Distribution" in summary_text
    assert "Cases total / observed transition labels / missing labels" in summary_text
    assert "Dominant transition reasons by label" in summary_text
    assert "Transition distribution support:" in summary_text

    by_name = {row["case_name"]: row for row in payload["cases"]}
    for case_name in ("bp_single_factor_v1", "roe_ttm_single_factor_v1", "value_quality_lowvol_v1"):
        row = by_name[case_name]
        assert row["status"] == "success"
        assert Path(row["output_dir"]).exists()
        assert Path(row["summary_path"]).exists()
        assert Path(row["experiment_card_path"]).exists()
        assert Path(row["factor_definition_json_path"]).exists()
        assert Path(row["signal_validation_json_path"]).exists()
        assert Path(row["portfolio_recipe_json_path"]).exists()
        assert Path(row["backtest_result_json_path"]).exists()
        assert "ic_ir" in row["key_metrics"]
        assert "mean_ic_ci_lower" in row["key_metrics"]
        assert "mean_ic_ci_upper" in row["key_metrics"]
        assert "uncertainty_flags" in row["key_metrics"]
        assert "uncertainty_method" in row["key_metrics"]
        assert "uncertainty_confidence_level" in row["key_metrics"]
        assert "uncertainty_bootstrap_block_length" in row["key_metrics"]
        assert "factor_verdict" in row["key_metrics"]
        assert "campaign_triage" in row["key_metrics"]
        assert "campaign_triage_reasons" in row["key_metrics"]
        assert "campaign_triage_priority" in row["key_metrics"]
        assert "campaign_rank_primary_metric" in row["key_metrics"]
        assert "campaign_rank_secondary_metric" in row["key_metrics"]
        assert "campaign_rank_stability_metric" in row["key_metrics"]
        assert "campaign_rank_support_count" in row["key_metrics"]
        assert "campaign_rank_risk_count" in row["key_metrics"]
        assert "promotion_decision" in row["key_metrics"]
        assert "promotion_reasons" in row["key_metrics"]
        assert "promotion_blockers" in row["key_metrics"]
        assert "level12_transition_label" in row["key_metrics"]
        assert "portfolio_validation_status" in row["key_metrics"]
        assert "portfolio_validation_recommendation" in row["key_metrics"]
        assert "portfolio_validation_major_risks" in row["key_metrics"]
        assert "portfolio_validation_benchmark_relative_status" in row["key_metrics"]
        assert row["key_metrics"]["research_evaluation_profile"] == "default_research"
        assert "rolling_ic_positive_share" in row["key_metrics"]
        assert "rolling_ic_min_mean" in row["key_metrics"]
        assert "campaign_triage" in row
        assert "level2_promotion" in row

        pointer_path = result.output_dir / case_name / "case_output_pointer.json"
        assert pointer_path.exists()


def test_research_campaign_1_main_with_render_report_writes_campaign_report(
    tmp_path: Path,
) -> None:
    bp_spec = write_demo_single_factor_case(tmp_path / "bp_case", factor_name="bp")
    roe_spec = write_demo_single_factor_case(tmp_path / "roe_case", factor_name="roe_ttm")
    comp_spec = write_demo_composite_case(tmp_path / "composite_case")

    _rewrite_spec_name(bp_spec, "bp_single_factor_v1")
    _rewrite_spec_name(roe_spec, "roe_ttm_single_factor_v1")
    _rewrite_spec_name(comp_spec, "value_quality_lowvol_v1")

    campaign_path = _write_campaign_manifest(
        tmp_path,
        bp_spec=bp_spec,
        roe_spec=roe_spec,
        comp_spec=comp_spec,
    )

    rc = research_campaign_1.main([str(campaign_path), "--render-report"])
    assert rc == 0

    campaign_out = tmp_path / "campaign_outputs"
    report_path = campaign_out / "campaign_report.md"
    assert report_path.exists()

    results_payload = json.loads(
        (campaign_out / "campaign_results.json").read_text(encoding="utf-8")
    )
    assert results_payload["render_status"] == "success"
    assert results_payload["rendered_report"] is True
    assert results_payload["evaluation_profile"] == "default_research"
    assert results_payload["rendered_report_path"] == str(report_path.resolve())
    assert results_payload["render_error"] is None


def test_research_campaign_1_render_failure_is_warning_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    bp_spec = write_demo_single_factor_case(tmp_path / "bp_case", factor_name="bp")
    roe_spec = write_demo_single_factor_case(tmp_path / "roe_case", factor_name="roe_ttm")
    comp_spec = write_demo_composite_case(tmp_path / "composite_case")

    _rewrite_spec_name(bp_spec, "bp_single_factor_v1")
    _rewrite_spec_name(roe_spec, "roe_ttm_single_factor_v1")
    _rewrite_spec_name(comp_spec, "value_quality_lowvol_v1")

    campaign_path = _write_campaign_manifest(
        tmp_path,
        bp_spec=bp_spec,
        roe_spec=roe_spec,
        comp_spec=comp_spec,
    )

    def _raise_render(*args, **kwargs):
        raise RuntimeError("campaign render failed intentionally")

    monkeypatch.setattr(research_campaign_1, "write_campaign_report", _raise_render)

    rc = research_campaign_1.main([str(campaign_path), "--render-report"])
    assert rc == 0

    campaign_out = tmp_path / "campaign_outputs"
    assert not (campaign_out / "campaign_report.md").exists()

    results_payload = json.loads(
        (campaign_out / "campaign_results.json").read_text(encoding="utf-8")
    )
    assert results_payload["render_status"] == "failed"
    assert results_payload["rendered_report"] is False
    assert results_payload["evaluation_profile"] == "default_research"
    assert results_payload["rendered_report_path"] is None
    assert "campaign render failed intentionally" in str(results_payload["render_error"])


def test_research_campaign_1_rejects_unknown_evaluation_profile(
    tmp_path: Path,
) -> None:
    bp_spec = write_demo_single_factor_case(tmp_path / "bp_case", factor_name="bp")
    roe_spec = write_demo_single_factor_case(tmp_path / "roe_case", factor_name="roe_ttm")
    comp_spec = write_demo_composite_case(tmp_path / "composite_case")
    _rewrite_spec_name(bp_spec, "bp_single_factor_v1")
    _rewrite_spec_name(roe_spec, "roe_ttm_single_factor_v1")
    _rewrite_spec_name(comp_spec, "value_quality_lowvol_v1")
    campaign_path = _write_campaign_manifest(
        tmp_path,
        bp_spec=bp_spec,
        roe_spec=roe_spec,
        comp_spec=comp_spec,
    )

    with pytest.raises(SystemExit):
        research_campaign_1.main([str(campaign_path), "--evaluation-profile", "unknown_profile"])


def test_extract_key_metrics_uses_contract_projection_helpers(monkeypatch) -> None:
    payload = {
        "metrics": {
            "direction": "long",
            "mean_ic": "0.02",
            "mean_ic_ci_lower": 0.005,
            "mean_ic_ci_upper": 0.03,
            "ic_ir": "0.82",
            "mean_long_short_return": "0.0031",
            "mean_long_short_turnover": "0.11",
            "coverage_mean": "0.95",
            "rolling_ic_positive_share": "0.72",
            "rolling_rank_ic_positive_share": 0.68,
            "rolling_long_short_positive_share": 0.7,
            "rolling_ic_min_mean": 0.001,
            "uncertainty_flags": "ci_overlaps_zero;bootstrap_sparse",
            "factor_verdict": "Strong candidate",
            "factor_verdict_reasons": ["stable diagnostics"],
            "campaign_triage": "Advance to Level 2",
            "campaign_triage_reasons": ["robust in rolling windows"],
            "promotion_decision": "Promote to Level 2",
            "promotion_reasons": ["gate passed"],
            "promotion_blockers": [],
            "portfolio_validation_status": "completed",
            "portfolio_validation_recommendation": "Credible at portfolio level",
            "portfolio_validation_robustness_label": "Credible but sensitive",
            "portfolio_validation_major_risks": [],
            "portfolio_validation_benchmark_relative_status": "outperforming",
            "portfolio_validation_benchmark_relative_assessment": "healthy",
            "portfolio_validation_benchmark_excess_return": "0.004",
            "portfolio_validation_benchmark_tracking_error": "0.02",
            "neutralization_comparison": {
                "interpretation_flags": ["neutralization preserves most evidence"],
            },
            "neutralization_mean_ic_delta": "-0.005",
            "neutralization_mean_long_short_return_delta": "-0.0003",
            "campaign_rank_support_count": 2,
            "research_evaluation_profile": "default_research",
        }
    }
    calls = {"promotion_gate": 0, "profile_summary": 0, "portfolio": 0, "ranking": 0}

    original_promotion_gate = research_campaign_1.project_promotion_gate_metrics
    original_profile_summary = research_campaign_1.project_campaign_profile_summary_metrics
    original_portfolio = research_campaign_1.project_portfolio_validation_metrics
    original_ranking = research_campaign_1.project_campaign_ranking_metrics

    monkeypatch.setattr(research_campaign_1, "_load_json", lambda _: payload)

    def _track_promotion_gate(metrics: dict[str, object]) -> object:
        calls["promotion_gate"] += 1
        return original_promotion_gate(metrics)

    def _track_profile_summary(metrics: dict[str, object]) -> object:
        calls["profile_summary"] += 1
        return original_profile_summary(metrics)

    def _track_portfolio(metrics: dict[str, object]) -> object:
        calls["portfolio"] += 1
        return original_portfolio(metrics)

    def _track_ranking(metrics: dict[str, object]) -> object:
        calls["ranking"] += 1
        return original_ranking(metrics)

    monkeypatch.setattr(
        research_campaign_1,
        "project_promotion_gate_metrics",
        _track_promotion_gate,
    )
    monkeypatch.setattr(
        research_campaign_1,
        "project_campaign_profile_summary_metrics",
        _track_profile_summary,
    )
    monkeypatch.setattr(
        research_campaign_1,
        "project_portfolio_validation_metrics",
        _track_portfolio,
    )
    monkeypatch.setattr(
        research_campaign_1,
        "project_campaign_ranking_metrics",
        _track_ranking,
    )

    extracted = research_campaign_1._extract_key_metrics(Path("metrics.json"))

    assert calls["promotion_gate"] >= 1
    assert calls["profile_summary"] >= 1
    assert calls["portfolio"] >= 1
    assert calls["ranking"] >= 1
    assert extracted["ic_ir"] == 0.82
    assert extracted["uncertainty_flags"] == ("ci_overlaps_zero", "bootstrap_sparse")
    assert extracted["portfolio_validation_benchmark_tracking_error"] == 0.02
    assert extracted["portfolio_validation_robustness_label"] == "Credible but sensitive"
    assert extracted["level12_transition_label"] == "Confirmed at portfolio level"
    assert extracted["campaign_rank_support_count"] == 2


def test_render_campaign_summary_uses_contract_projection_helpers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    key_metrics: dict[str, object] = {
        "direction": "long",
        "mean_ic": 0.02,
        "mean_ic_ci_lower": 0.005,
        "mean_ic_ci_upper": 0.03,
        "ic_ir": 0.82,
        "mean_long_short_return": 0.0031,
        "mean_long_short_turnover": 0.11,
        "coverage_mean": 0.95,
        "rolling_ic_positive_share": 0.72,
        "rolling_rank_ic_positive_share": 0.68,
        "rolling_long_short_positive_share": 0.70,
        "rolling_ic_min_mean": 0.001,
        "uncertainty_flags": ["ci_overlaps_zero"],
        "factor_verdict": "Strong candidate",
        "campaign_triage": "Advance to Level 2",
        "campaign_triage_reasons": ["robust in rolling windows"],
        "promotion_decision": "Promote to Level 2",
        "promotion_reasons": ["gate passed"],
        "promotion_blockers": [],
        "portfolio_validation_status": "completed",
        "portfolio_validation_recommendation": "Credible at portfolio level",
        "portfolio_validation_robustness_label": "Credible but sensitive",
        "portfolio_validation_major_risks": [],
        "portfolio_validation_benchmark_relative_status": "outperforming",
        "portfolio_validation_benchmark_relative_assessment": "healthy",
        "portfolio_validation_benchmark_excess_return": 0.004,
        "portfolio_validation_benchmark_tracking_error": 0.02,
        "neutralization_comparison": {
            "interpretation_flags": ["neutralization preserves most evidence"],
        },
        "neutralization_mean_ic_delta": -0.005,
        "neutralization_mean_long_short_return_delta": -0.0003,
        "research_evaluation_profile": "default_research",
    }
    case_results = (
        research_campaign_1.CampaignCaseResult(
            case_name="bp_single_factor_v1",
            package_type="single_factor",
            status="success",
            output_dir=tmp_path / "case_output",
            summary_path=tmp_path / "summary.md",
            experiment_card_path=tmp_path / "experiment_card.md",
            run_manifest_path=tmp_path / "run_manifest.json",
            metrics_path=tmp_path / "metrics.json",
            key_metrics=key_metrics,
            vault_export={"status": "success"},
            error=None,
        ),
    )
    config = research_campaign_1.CampaignConfig(
        campaign_name="research_campaign_1",
        campaign_description="projection coverage",
        output_root_dir=str(tmp_path / "campaign_outputs"),
        cases=(),
        execution_order=(),
    )

    calls = {"promotion_gate": 0, "profile_summary": 0, "portfolio": 0, "ranking": 0}
    original_promotion_gate = research_campaign_1.project_promotion_gate_metrics
    original_profile_summary = research_campaign_1.project_campaign_profile_summary_metrics
    original_portfolio = research_campaign_1.project_portfolio_validation_metrics
    original_ranking = research_campaign_1.project_campaign_ranking_metrics

    def _track_promotion_gate(metrics: dict[str, object]) -> object:
        calls["promotion_gate"] += 1
        return original_promotion_gate(metrics)

    def _track_profile_summary(metrics: dict[str, object]) -> object:
        calls["profile_summary"] += 1
        return original_profile_summary(metrics)

    def _track_portfolio(metrics: dict[str, object]) -> object:
        calls["portfolio"] += 1
        return original_portfolio(metrics)

    def _track_ranking(metrics: dict[str, object]) -> object:
        calls["ranking"] += 1
        return original_ranking(metrics)

    monkeypatch.setattr(
        research_campaign_1,
        "project_promotion_gate_metrics",
        _track_promotion_gate,
    )
    monkeypatch.setattr(
        research_campaign_1,
        "project_campaign_profile_summary_metrics",
        _track_profile_summary,
    )
    monkeypatch.setattr(
        research_campaign_1,
        "project_portfolio_validation_metrics",
        _track_portfolio,
    )
    monkeypatch.setattr(
        research_campaign_1,
        "project_campaign_ranking_metrics",
        _track_ranking,
    )

    summary = research_campaign_1.render_campaign_summary(
        config=config,
        case_results=case_results,
        run_timestamp_utc="2026-03-27T00:00:00Z",
    )

    assert calls["promotion_gate"] >= 1
    assert calls["profile_summary"] >= 1
    assert calls["portfolio"] >= 1
    assert calls["ranking"] >= 1
    assert "bp_single_factor_v1" in summary
    assert "IC=0.020000" in summary
    assert "IC95%CI=[0.005000, 0.030000]" in summary
    assert "Neutralization=neutralization preserves most evidence" in summary
    assert "Transition=Confirmed at portfolio level" in summary
    assert "PortfolioValidation=completed (Credible at portfolio level)" in summary
    assert "PortfolioRobustness=Credible but sensitive" in summary
    assert "Level 1->Level 2 Transition Distribution" in summary


def _rewrite_spec_name(spec_path: Path, case_name: str) -> None:
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    payload["name"] = case_name
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _write_campaign_manifest(
    tmp_path: Path,
    *,
    bp_spec: Path,
    roe_spec: Path,
    comp_spec: Path,
) -> Path:
    payload = {
        "campaign_name": "research_campaign_1",
        "campaign_description": "Synthetic smoke campaign for three standardized cases.",
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
                "spec_path": str(comp_spec),
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
