from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alpha_lab.reporting.renderers.case_report import render_case_report, write_case_report
from alpha_lab.reporting.renderers.templates import CASE_SECTION_TITLES, PLACEHOLDER_OBJECTIVE


def test_case_report_renderer_loads_minimal_artifacts_and_renders_required_sections(
    tmp_path: Path,
) -> None:
    case_dir = tmp_path / "case_output"
    case_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        case_dir / "run_manifest.json",
        {
            "artifact_type": "real_case_single_factor_bundle",
            "case_name": "bp_single_factor_v1",
            "spec": {
                "factor_name": "bp",
                "direction": "long",
                "rebalance_frequency": "W",
                "target": {"kind": "forward_return", "horizon": 5},
                "universe": {"name": "A-share top 500"},
                "preprocess": {
                    "winsorize": True,
                    "winsorize_lower": 0.01,
                    "winsorize_upper": 0.99,
                    "standardization": "zscore",
                    "min_group_size": 3,
                },
                "neutralization": {"enabled": False},
                "transaction_cost": {"one_way_rate": 0.001},
            },
        },
    )

    _write_json(
        case_dir / "metrics.json",
        {
            "metrics": {
                "mean_ic": 0.021,
                "ic_ir": 0.87,
                "mean_long_short_return": 0.0034,
                "mean_long_short_turnover": 0.22,
                "coverage_mean": 0.91,
            }
        },
    )

    pd.DataFrame(
        [
            {"date": "2024-01-01", "group": 1, "group_return": -0.001},
            {"date": "2024-01-01", "group": 5, "group_return": 0.002},
            {"date": "2024-01-02", "group": 1, "group_return": -0.002},
            {"date": "2024-01-02", "group": 5, "group_return": 0.004},
        ]
    ).to_csv(case_dir / "group_returns.csv", index=False)

    (case_dir / "summary.md").write_text(
        "# Summary\n\nObjective: Capture valuation mean reversion.\n",
        encoding="utf-8",
    )

    report = render_case_report(case_dir)
    for section in CASE_SECTION_TITLES:
        assert f"## {section}" in report
    assert "bp_single_factor_v1" in report
    assert "`single_factor`" in report
    assert "Capture valuation mean reversion" in report
    assert "IC / ICIR" in report

    written = write_case_report(case_dir)
    assert written.exists()
    assert written.name == "case_report.md"


def test_case_report_renderer_handles_missing_fields_gracefully(tmp_path: Path) -> None:
    case_dir = tmp_path / "case_missing"
    case_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        case_dir / "run_manifest.json",
        {
            "artifact_type": "unknown",
            "case_name": "mystery_case",
            "spec": {},
        },
    )
    _write_json(case_dir / "metrics.json", {"metrics": {}})

    report = render_case_report(case_dir)
    assert PLACEHOLDER_OBJECTIVE in report
    assert "N/A" in report
    assert "mystery_case" in report


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

