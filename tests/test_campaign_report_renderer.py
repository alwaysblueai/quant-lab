from __future__ import annotations

import json
from pathlib import Path

from alpha_lab.reporting.renderers.campaign_report import (
    render_campaign_report,
    write_campaign_report,
)
from alpha_lab.reporting.renderers.templates import (
    CAMPAIGN_SECTION_TITLES,
    COMPARISON_TABLE_COLUMNS,
)


def test_campaign_report_renderer_builds_comparison_table(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaign_output"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    case2_metrics = campaign_dir / "case2_metrics.json"
    _write_json(
        case2_metrics,
        {
            "metrics": {
                "mean_ic": 0.018,
                "ic_ir": 0.61,
                "mean_long_short_return": 0.0029,
                "mean_long_short_turnover": 0.31,
                "coverage_mean": 0.83,
            }
        },
    )

    _write_json(
        campaign_dir / "campaign_manifest.json",
        {
            "campaign_name": "research_campaign_1",
            "campaign_description": "Compare value, quality, and composite signals.",
        },
    )

    _write_json(
        campaign_dir / "campaign_results.json",
        {
            "campaign_name": "research_campaign_1",
            "cases": [
                {
                    "case_name": "bp_single_factor_v1",
                    "package_type": "single_factor",
                    "status": "success",
                    "key_metrics": {
                        "mean_ic": 0.02,
                        "ic_ir": 0.73,
                        "mean_long_short_return": 0.0033,
                        "mean_long_short_turnover": 0.20,
                        "coverage_mean": 0.90,
                    },
                },
                {
                    "case_name": "value_quality_lowvol_v1",
                    "package_type": "composite",
                    "status": "success",
                    "key_metrics": {},
                    "metrics_path": str(case2_metrics),
                },
            ],
        },
    )

    report = render_campaign_report(campaign_dir)
    for section in CAMPAIGN_SECTION_TITLES:
        assert f"## {section}" in report
    for column in COMPARISON_TABLE_COLUMNS:
        assert column in report
    assert "bp_single_factor_v1" in report
    assert "value_quality_lowvol_v1" in report
    assert "single" in report
    assert "composite" in report

    written = write_campaign_report(campaign_dir)
    assert written.exists()
    assert written.name == "campaign_report.md"


def test_campaign_report_renderer_handles_mixed_status_and_missing_metrics(
    tmp_path: Path,
) -> None:
    campaign_dir = tmp_path / "campaign_mixed"
    campaign_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        campaign_dir / "campaign_results.json",
        {
            "campaign_name": "mixed_campaign",
            "cases": [
                {
                    "case_name": "roe_ttm_single_factor_v1",
                    "package_type": "single_factor",
                    "status": "success",
                    "key_metrics": {},
                },
                {
                    "case_name": "value_quality_lowvol_v1",
                    "package_type": "composite",
                    "status": "failed",
                    "key_metrics": {},
                    "error": "missing component file",
                },
            ],
        },
    )

    report = render_campaign_report(campaign_dir)
    assert "Failure Cases / Data Issues" in report
    assert "missing component file" in report
    assert "did not complete successfully" in report
    assert "N/A" in report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

