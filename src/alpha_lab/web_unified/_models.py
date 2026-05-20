"""Plain data records and type aliases shared across the unified web frontend.

This is the lowest layer of ``web_unified`` — it has no dependencies on
``_RunStore`` / ``_UnifiedService`` / handler logic, so any other submodule
(subprocess helpers, future run_store/handler splits) can import from here
without creating a circular dependency.

``_RunRecord`` deliberately stays in ``__init__.py`` for now — its
``_artifact_paths_for_api`` helper calls into the big artifact resolver
defined further down in the package, which would require a circular import
to factor out cleanly. Extracting it is queued as a follow-up.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

RunStatus = Literal["queued", "running", "succeeded", "failed", "cancelled"]
RunWorkflow = Literal["single_factor", "model_factor"]


@dataclass(frozen=True)
class _RunTask:
    run_id: str
    project_slug: str
    case_name: str
    round_id: str | None
    spec_path: str
    evaluation_profile: str
    output_root_dir: str | None
    render_report: bool
    evaluation_profile_source: str = "request"
    workflow: RunWorkflow = "single_factor"
    note: str | None = None
    draft_model_candidate_path: str | None = None
    draft_model_candidate_name: str | None = None
    draft_model_candidate_hash: str | None = None
    screening_retrain_every_n_dates: int | None = None


@dataclass(frozen=True)
class _SubprocessCaseRunResult:
    output_dir: Path
    artifact_paths: Mapping[str, Path]


class _ModelLabSubprocessError(RuntimeError):
    """Raised when the model-lab subprocess exits non-zero.

    Carries the original return code and a human-readable hint that
    propagates through the run record into the UI's error pane.
    """

    def __init__(
        self,
        message: str,
        *,
        returncode: int | None,
        hint: str,
    ) -> None:
        super().__init__(message)
        self.returncode = returncode
        self.hint = hint


__all__ = [
    "RunStatus",
    "RunWorkflow",
    "_ModelLabSubprocessError",
    "_RunTask",
    "_SubprocessCaseRunResult",
]
