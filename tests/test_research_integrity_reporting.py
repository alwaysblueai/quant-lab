from __future__ import annotations

import json

from alpha_lab.research_integrity.contracts import IntegrityCheckResult
from alpha_lab.research_integrity.reporting import (
    build_integrity_report,
    render_integrity_report_markdown,
    write_integrity_report_json,
    write_integrity_report_markdown,
)


def _sample_checks() -> list[IntegrityCheckResult]:
    return [
        IntegrityCheckResult(
            check_name="check_pass",
            status="pass",
            severity="info",
            object_name="factor_df",
            module_name="tests",
            message="pass",
        ),
        IntegrityCheckResult(
            check_name="check_warn",
            status="warn",
            severity="warning",
            object_name="aux_df",
            module_name="tests",
            message="warn",
            remediation="inspect timestamps",
        ),
        IntegrityCheckResult(
            check_name="check_fail",
            status="fail",
            severity="error",
            object_name="label_df",
            module_name="tests",
            message="fail",
            remediation="drop future rows",
        ),
    ]


def test_build_integrity_report_serializes_pass_warn_fail_summary():
    report = build_integrity_report(
        _sample_checks(),
        context={"experiment": "unit_test"},
    )

    payload = report.to_dict()
    assert payload["summary"]["n_pass"] == 1
    assert payload["summary"]["n_warn"] == 1
    assert payload["summary"]["n_fail"] == 1
    assert payload["summary"]["highest_severity"] == "error"


def test_write_integrity_report_json_creates_machine_readable_artifact(tmp_path):
    report = build_integrity_report(_sample_checks())
    out_path = write_integrity_report_json(report, tmp_path / "integrity_report.json")

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "1.0.0"
    assert len(payload["checks"]) == 3
    assert payload["checks"][2]["status"] == "fail"


def test_write_integrity_report_markdown_creates_human_readable_artifact(tmp_path):
    report = build_integrity_report(_sample_checks())
    out_path = write_integrity_report_markdown(report, tmp_path / "integrity_report.md")

    text = out_path.read_text(encoding="utf-8")
    assert "# Research Integrity Report" in text
    assert "| Check | Status | Severity |" in text
    assert "`check_fail`" in text


def test_render_integrity_report_markdown_contains_all_status_labels():
    report = build_integrity_report(_sample_checks())
    text = render_integrity_report_markdown(report)

    assert "`pass`" in text
    assert "`warn`" in text
    assert "`fail`" in text
