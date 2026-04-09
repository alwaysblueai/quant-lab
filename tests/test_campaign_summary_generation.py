from __future__ import annotations

import json
from pathlib import Path

import yaml

from alpha_lab.campaigns.research_campaign_1 import run_research_campaign_1
from tests.single_factor_case_helpers import write_demo_single_factor_case


def test_campaign_summary_marks_failed_case_without_fabricating_metrics(tmp_path: Path) -> None:
    bp_spec = write_demo_single_factor_case(tmp_path / "bp_case", factor_name="bp")
    roe_spec = write_demo_single_factor_case(tmp_path / "roe_case", factor_name="roe_ttm")

    _rewrite_spec_name(bp_spec, "bp_single_factor_v1")
    _rewrite_spec_name(roe_spec, "roe_ttm_single_factor_v1")

    bad_composite_spec = tmp_path / "missing_value_quality_lowvol.yaml"

    campaign_path = _write_campaign_manifest(
        tmp_path,
        bp_spec=bp_spec,
        roe_spec=roe_spec,
        comp_spec=bad_composite_spec,
    )

    result = run_research_campaign_1(campaign_path)

    results_payload = json.loads(
        result.artifact_paths["campaign_results"].read_text(encoding="utf-8")
    )
    assert results_payload["n_cases"] == 3
    assert results_payload["n_success"] == 2
    assert results_payload["n_failed"] == 1

    by_name = {row["case_name"]: row for row in results_payload["cases"]}
    failed = by_name["value_quality_lowvol_v1"]
    assert failed["status"] == "failed"
    assert failed["output_dir"] is None
    assert failed["error"]

    summary_text = result.artifact_paths["campaign_summary"].read_text(encoding="utf-8")
    assert "## 5. Comparative Observations" in summary_text
    assert "value_quality_lowvol_v1 | composite" in summary_text
    assert "| Rank | Case |" in summary_text
    assert "IC 95% CI" in summary_text
    assert "Rolling IC+" in summary_text
    assert "Verdict" in summary_text
    assert "Campaign Triage" in summary_text
    assert "Triage Reasons" in summary_text
    assert "Level 2 Promotion" in summary_text
    assert "Promotion Reasons" in summary_text
    assert "Promotion Blockers" in summary_text
    assert "L1->L2 Transition" in summary_text
    assert "Level 2 Portfolio Validation" in summary_text
    assert "Portfolio Robustness" in summary_text
    assert "Portfolio Benchmark Relative" in summary_text
    assert "Portfolio Validation Risks" in summary_text
    assert "Neutralization" in summary_text
    assert "Level 1->Level 2 Transition Distribution" in summary_text
    assert "Cases total / observed transition labels / missing labels" in summary_text
    assert "Dominant transition reasons by label" in summary_text
    assert "Transition distribution support:" in summary_text
    assert "## 6. Campaign Triage Ranking" in summary_text
    assert "N/A" in summary_text
    assert "failed" in summary_text

    index_text = result.artifact_paths["campaign_index"].read_text(encoding="utf-8")
    assert "Campaign Index: research_campaign_1" in index_text
    assert "value_quality_lowvol_v1" in index_text


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
        "campaign_description": "Synthetic campaign with one failing case.",
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
