from __future__ import annotations

import json
from pathlib import Path

import yaml

from alpha_lab.campaigns.research_campaign_1 import run_research_campaign_1
from alpha_lab.real_cases.composite.pipeline import run_composite_case
from alpha_lab.real_cases.single_factor.pipeline import run_single_factor_case
from alpha_lab.reporting.research_validation_package import (
    build_research_validation_package,
    export_research_validation_package,
)
from tests.composite_case_helpers import write_demo_composite_case
from tests.single_factor_case_helpers import write_demo_single_factor_case

COMMON_CASE_METRIC_FIELDS: set[str] = {
    "research_evaluation_profile",
    "campaign_triage",
    "promotion_decision",
    "level12_transition_label",
    "portfolio_validation_status",
    "portfolio_validation_recommendation",
    "portfolio_validation_benchmark_relative_status",
}


def test_level12_outputs_keep_consistent_contracts(tmp_path: Path) -> None:
    bp_spec = write_demo_single_factor_case(tmp_path / "bp_case", factor_name="bp")
    roe_spec = write_demo_single_factor_case(tmp_path / "roe_case", factor_name="roe_ttm")
    composite_spec = write_demo_composite_case(tmp_path / "composite_case")

    _rewrite_spec_name(bp_spec, "bp_single_factor_v1")
    _rewrite_spec_name(roe_spec, "roe_ttm_single_factor_v1")
    _rewrite_spec_name(composite_spec, "value_quality_lowvol_v1")

    single_run = run_single_factor_case(bp_spec)
    composite_run = run_composite_case(composite_spec)

    for output_dir in (single_run.output_dir, composite_run.output_dir):
        manifest = _load_json(output_dir / "run_manifest.json")
        metrics_payload = _load_json(output_dir / "metrics.json")
        outputs = _load_mapping(manifest.get("outputs"))
        coverage_summary = _load_mapping(metrics_payload.get("coverage_by_date_summary"))
        key_metrics = _load_mapping(metrics_payload.get("metrics"))

        assert COMMON_CASE_METRIC_FIELDS.issubset(set(key_metrics))
        assert key_metrics["research_evaluation_profile"] == "default_research"
        assert "mean_coverage" in coverage_summary
        assert "min_coverage" in coverage_summary

        required_bundle_files = manifest.get("required_bundle_files")
        assert isinstance(required_bundle_files, list)
        assert (
            "level2_portfolio_validation/portfolio_validation_summary.json"
            in required_bundle_files
        )
        assert (
            "level2_portfolio_validation/portfolio_validation_metrics.json"
            in required_bundle_files
        )
        assert (
            "level2_portfolio_validation/portfolio_validation_package.json"
            in required_bundle_files
        )
        assert "factor_definition.json" in required_bundle_files
        assert "signal_validation.json" in required_bundle_files
        assert "portfolio_recipe.json" in required_bundle_files
        assert "backtest_result.json" in required_bundle_files

        for raw_path in outputs.values():
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            assert Path(raw_path).exists()

    campaign_manifest_path = _write_campaign_manifest(
        tmp_path,
        bp_spec=bp_spec,
        roe_spec=roe_spec,
        composite_spec=composite_spec,
    )
    campaign_run = run_research_campaign_1(campaign_manifest_path)
    campaign_results = _load_json(campaign_run.artifact_paths["campaign_results"])
    assert campaign_results["evaluation_profile"] == "default_research"

    cases = campaign_results.get("cases")
    assert isinstance(cases, list)
    assert len(cases) == 3
    for row in cases:
        assert isinstance(row, dict)
        if row.get("status") != "success":
            continue
        metrics = _load_mapping(row.get("key_metrics"))
        assert COMMON_CASE_METRIC_FIELDS.issubset(set(metrics))
        assert metrics.get("research_evaluation_profile") == "default_research"
        for key in (
            "output_dir",
            "run_manifest_path",
            "metrics_path",
            "factor_definition_json_path",
            "signal_validation_json_path",
            "portfolio_recipe_json_path",
            "backtest_result_json_path",
        ):
            raw = row.get(key)
            assert isinstance(raw, str) and Path(raw).exists()

    package = build_research_validation_package(
        single_run.output_dir,
        case_id="bp_single_factor_v1",
        case_name="bp_single_factor_v1",
    )
    exported = export_research_validation_package(package, tmp_path / "research_validation_package")
    assert exported["json"].name == "research_validation_package.json"
    assert exported["markdown"].name == "research_validation_package.md"
    assert exported["portfolio_validation_summary"].name == "portfolio_validation_summary.json"
    assert exported["portfolio_validation_metrics"].name == "portfolio_validation_metrics.json"
    assert exported["portfolio_validation_package"].name == "portfolio_validation_package.json"


def _write_campaign_manifest(
    tmp_path: Path,
    *,
    bp_spec: Path,
    roe_spec: Path,
    composite_spec: Path,
) -> Path:
    payload = {
        "campaign_name": "research_campaign_1",
        "campaign_description": "Output contract consistency check.",
        "output_root_dir": str(tmp_path / "campaign_outputs"),
        "case_output_root_dir": str(tmp_path / "campaign_case_outputs"),
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


def _rewrite_spec_name(spec_path: Path, case_name: str) -> None:
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    payload["name"] = case_name
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _load_mapping(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return value
