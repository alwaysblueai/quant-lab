from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Literal

from alpha_lab.exceptions import AlphaLabDataError

INTEGRITY_REPORT_SCHEMA_VERSION = "1.0.0"

IntegrityStatus = Literal["pass", "warn", "fail"]
IntegritySeverity = Literal["info", "warning", "error"]

_VALID_STATUS: frozenset[str] = frozenset({"pass", "warn", "fail"})
_VALID_SEVERITY: frozenset[str] = frozenset({"info", "warning", "error"})


def utc_now_iso() -> str:
    """Return a stable UTC timestamp string for audit artifacts."""

    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


@dataclass(frozen=True)
class IntegrityCheckResult:
    """Structured result for one integrity check."""

    check_name: str
    status: IntegrityStatus
    severity: IntegritySeverity
    message: str
    object_name: str | None = None
    module_name: str | None = None
    remediation: str | None = None
    metrics: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUS:
            raise AlphaLabDataError(f"unsupported check status {self.status!r}")
        if self.severity not in _VALID_SEVERITY:
            raise AlphaLabDataError(f"unsupported check severity {self.severity!r}")
        if not self.check_name.strip():
            raise AlphaLabDataError("check_name must be non-empty")
        if not self.message.strip():
            raise AlphaLabDataError("message must be non-empty")

    @property
    def is_failure(self) -> bool:
        return self.status == "fail"

    def to_dict(self) -> dict[str, object]:
        return {
            "check_name": self.check_name,
            "status": self.status,
            "severity": self.severity,
            "object_name": self.object_name,
            "module_name": self.module_name,
            "message": self.message,
            "remediation": self.remediation,
            "metrics": self.metrics,
        }


@dataclass(frozen=True)
class IntegrityReportSummary:
    """Aggregate counters for one integrity report."""

    n_checks: int
    n_pass: int
    n_warn: int
    n_fail: int
    highest_severity: IntegritySeverity
    hard_failure_checks: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "n_checks": self.n_checks,
            "n_pass": self.n_pass,
            "n_warn": self.n_warn,
            "n_fail": self.n_fail,
            "highest_severity": self.highest_severity,
            "hard_failure_checks": list(self.hard_failure_checks),
        }


@dataclass(frozen=True)
class IntegrityReport:
    """Serializable integrity report payload for artifact export."""

    schema_version: str
    generated_at_utc: str
    checks: tuple[IntegrityCheckResult, ...]
    summary: IntegrityReportSummary
    context: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "summary": self.summary.to_dict(),
            "context": self.context,
            "checks": [check.to_dict() for check in self.checks],
        }


def _severity_rank(severity: IntegritySeverity) -> int:
    order = {"info": 0, "warning": 1, "error": 2}
    return order[severity]


def summarize_checks(
    checks: list[IntegrityCheckResult] | tuple[IntegrityCheckResult, ...],
) -> IntegrityReportSummary:
    """Build report-level counters from a list of check results."""

    n_pass = sum(1 for check in checks if check.status == "pass")
    n_warn = sum(1 for check in checks if check.status == "warn")
    n_fail = sum(1 for check in checks if check.status == "fail")

    highest: IntegritySeverity = "info"
    if checks:
        highest = max(checks, key=lambda check: _severity_rank(check.severity)).severity

    hard_failures = tuple(
        check.check_name
        for check in checks
        if check.status == "fail" and check.severity == "error"
    )

    return IntegrityReportSummary(
        n_checks=len(checks),
        n_pass=n_pass,
        n_warn=n_warn,
        n_fail=n_fail,
        highest_severity=highest,
        hard_failure_checks=hard_failures,
    )
