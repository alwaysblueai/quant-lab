from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from alpha_lab.reporting.factor_correlation import inspect_run_factor_correlation


@dataclass(frozen=True, slots=True)
class DecompositionSignal:
    redundant: bool
    top_match: str
    max_abs_correlation: float
    evidence_path: str
    reason: str


def inspect_run_decomposition(
    run_root: str | Path,
    *,
    threshold: float = 0.7,
) -> DecompositionSignal | None:
    signal = inspect_run_factor_correlation(run_root, threshold=threshold)
    if signal is None:
        return None
    return DecompositionSignal(
        redundant=signal.redundant,
        top_match=signal.top_match,
        max_abs_correlation=signal.max_abs_correlation,
        evidence_path=signal.evidence_path,
        reason=signal.reason,
    )
