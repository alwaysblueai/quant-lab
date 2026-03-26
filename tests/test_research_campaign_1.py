from __future__ import annotations

import json
from pathlib import Path

import yaml

from alpha_lab.campaigns.research_campaign_1 import run_research_campaign_1
from tests.composite_case_helpers import write_demo_composite_case
from tests.single_factor_case_helpers import write_demo_single_factor_case


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

    result = run_research_campaign_1(campaign_path)

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
    assert payload["n_cases"] == 3
    assert payload["n_success"] == 3
    assert payload["n_failed"] == 0

    by_name = {row["case_name"]: row for row in payload["cases"]}
    for case_name in ("bp_single_factor_v1", "roe_ttm_single_factor_v1", "value_quality_lowvol_v1"):
        row = by_name[case_name]
        assert row["status"] == "success"
        assert Path(row["output_dir"]).exists()
        assert Path(row["summary_path"]).exists()
        assert Path(row["experiment_card_path"]).exists()
        assert "ic_ir" in row["key_metrics"]

        pointer_path = result.output_dir / case_name / "case_output_pointer.json"
        assert pointer_path.exists()


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
