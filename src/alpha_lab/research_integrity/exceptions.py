from __future__ import annotations

from dataclasses import dataclass

from alpha_lab.research_integrity.contracts import IntegrityCheckResult


class ResearchIntegrityError(RuntimeError):
    """Base class for integrity-related runtime errors."""


@dataclass(frozen=True)
class IntegrityHardFailure(ResearchIntegrityError):
    """Raised when one or more hard integrity failures are present."""

    failures: tuple[IntegrityCheckResult, ...]

    def __str__(self) -> str:
        check_names = ", ".join(check.check_name for check in self.failures)
        return f"hard integrity failures detected: {check_names}"


def raise_on_hard_failures(
    checks: list[IntegrityCheckResult] | tuple[IntegrityCheckResult, ...],
) -> None:
    """Raise an exception when any check reports fail/error."""

    hard_failures = tuple(
        check
        for check in checks
        if check.status == "fail" and check.severity == "error"
    )
    if hard_failures:
        raise IntegrityHardFailure(hard_failures)
