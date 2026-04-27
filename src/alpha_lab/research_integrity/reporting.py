from __future__ import annotations

import json
from pathlib import Path

from alpha_lab.research_integrity.contracts import (
    INTEGRITY_REPORT_SCHEMA_VERSION,
    IntegrityCheckResult,
    IntegrityReport,
    summarize_checks,
    utc_now_iso,
)


def build_integrity_report(
    checks: list[IntegrityCheckResult] | tuple[IntegrityCheckResult, ...],
    *,
    context: dict[str, object] | None = None,
) -> IntegrityReport:
    """Build an integrity report object from check results."""

    check_tuple = tuple(checks)
    summary = summarize_checks(check_tuple)
    return IntegrityReport(
        schema_version=INTEGRITY_REPORT_SCHEMA_VERSION,
        generated_at_utc=utc_now_iso(),
        checks=check_tuple,
        summary=summary,
        context={} if context is None else dict(context),
    )


def write_integrity_report_json(
    report: IntegrityReport,
    output_path: str | Path,
) -> Path:
    """Write integrity report JSON to disk."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def render_integrity_report_markdown(report: IntegrityReport) -> str:
    """Render one integrity report as markdown for review workflows."""

    lines: list[str] = [
        "# Research Integrity Report",
        "",
        "## Summary",
        "",
        f"- Generated (UTC): `{report.generated_at_utc}`",
        f"- Checks: `{report.summary.n_checks}`",
        f"- Pass: `{report.summary.n_pass}`",
        f"- Warn: `{report.summary.n_warn}`",
        f"- Fail: `{report.summary.n_fail}`",
        f"- Highest Severity: `{report.summary.highest_severity}`",
    ]

    if report.context:
        lines.extend(
            [
                "",
                "## Context",
                "",
            ]
        )
        for key in sorted(report.context.keys()):
            lines.append(f"- {key}: `{report.context[key]}`")

    lines.extend(
        [
            "",
            "## Checks",
            "",
            "| Check | Status | Severity | Object | Message | Remediation |",
            "|---|---|---|---|---|---|",
        ]
    )

    for check in report.checks:
        object_name = check.object_name or "-"
        remediation = check.remediation or "-"
        message = check.message.replace("\n", " ")
        remediation_text = remediation.replace("\n", " ")
        lines.append(
            "| "
            f"`{check.check_name}` | `{check.status}` | `{check.severity}` | "
            f"`{object_name}` | {message} | {remediation_text} |"
        )

    return "\n".join(lines).rstrip() + "\n"


def write_integrity_report_markdown(
    report: IntegrityReport,
    output_path: str | Path,
) -> Path:
    """Write integrity report markdown to disk."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_integrity_report_markdown(report), encoding="utf-8")
    return path


def export_integrity_report(
    checks: list[IntegrityCheckResult] | tuple[IntegrityCheckResult, ...],
    output_dir: str | Path,
    *,
    context: dict[str, object] | None = None,
    json_name: str = "integrity_report.json",
    markdown_name: str = "integrity_report.md",
) -> dict[str, Path]:
    """Build and export integrity report artifacts (JSON + markdown)."""

    report = build_integrity_report(checks, context=context)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = write_integrity_report_json(report, out_dir / json_name)
    markdown_path = write_integrity_report_markdown(report, out_dir / markdown_name)
    return {
        "json": json_path,
        "markdown": markdown_path,
    }
