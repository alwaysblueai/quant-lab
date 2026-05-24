"""Resident-memory telemetry and a soft per-run memory budget.

:class:`RunMemoryMonitor` samples process RSS at stage boundaries so a run can:

1. record per-stage memory into a ``resource_usage.json`` artifact (telemetry),
   making it obvious which stage is the memory hog on a wide panel; and
2. fail fast with an auditable :class:`~alpha_lab.exceptions.AlphaLabMemoryError`
   when peak RSS exceeds ``ALPHA_LAB_MAX_RSS_MB``, instead of being silently
   killed by the OS OOM-killer.

This is a **soft** guard. Sampling happens at stage boundaries, so a single
large allocation inside one stage can still trip the OS limit before the next
sample. Set the budget below the host's hard limit (e.g. 15000 on an ~18-19GB
WSL box) and treat it as protection against gradual growth, not a hard cap.
RSS readings are non-deterministic, so the snapshot is written to a standalone
artifact and never embedded in golden-compared run manifests.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from alpha_lab.exceptions import AlphaLabMemoryError

MAX_RSS_ENV_VAR = "ALPHA_LAB_MAX_RSS_MB"
RESOURCE_USAGE_ARTIFACT_NAME = "resource_usage.json"


def _read_current_rss_mb() -> float | None:
    """Return the current process RSS in MB, or ``None`` if unavailable."""
    try:
        import psutil  # type: ignore[import-untyped]
    except Exception:  # pragma: no cover - psutil missing is an environment edge
        return None
    try:
        rss_bytes = psutil.Process().memory_info().rss
    except Exception:  # pragma: no cover - process introspection can fail
        return None
    return float(rss_bytes) / (1024.0 * 1024.0)


class RunMemoryMonitor:
    """Sample-on-demand RSS tracker with an optional soft budget.

    Enforcement (the budget) is opt-in: when ``max_rss_mb`` is ``None`` the
    monitor only records telemetry and never raises. Telemetry degrades to
    ``available=False`` when RSS cannot be read (e.g. psutil missing).
    """

    def __init__(self, max_rss_mb: float | None = None, *, label: str = "") -> None:
        self.max_rss_mb = max_rss_mb if (max_rss_mb is not None and max_rss_mb > 0) else None
        self.label = label
        self._stage_rss_mb: dict[str, float] = {}
        self._peak_rss_mb: float | None = None
        self._available = _read_current_rss_mb() is not None

    @classmethod
    def from_env(cls, *, label: str = "") -> RunMemoryMonitor:
        """Build a monitor reading the budget from ``ALPHA_LAB_MAX_RSS_MB``."""
        raw = os.environ.get(MAX_RSS_ENV_VAR, "").strip()
        budget: float | None = None
        if raw:
            try:
                parsed = float(raw)
            except ValueError:
                parsed = 0.0
            if parsed > 0:
                budget = parsed
        return cls(budget, label=label)

    @property
    def available(self) -> bool:
        return self._available

    @property
    def peak_rss_mb(self) -> float | None:
        return self._peak_rss_mb

    def sample(self, stage: str) -> float | None:
        """Record the current RSS for ``stage`` and update the running peak."""
        rss = _read_current_rss_mb()
        if rss is None:
            return None
        self._available = True
        # Keep the largest reading seen for a stage that is entered repeatedly.
        previous = self._stage_rss_mb.get(stage)
        self._stage_rss_mb[stage] = rss if previous is None else max(previous, rss)
        if self._peak_rss_mb is None or rss > self._peak_rss_mb:
            self._peak_rss_mb = rss
        return rss

    def check(self, stage: str | None = None) -> None:
        """Raise :class:`AlphaLabMemoryError` if peak RSS exceeds the budget."""
        if self.max_rss_mb is None or self._peak_rss_mb is None:
            return
        if self._peak_rss_mb <= self.max_rss_mb:
            return
        where = f" at stage {stage!r}" if stage else ""
        label = f" for {self.label!r}" if self.label else ""
        raise AlphaLabMemoryError(
            f"run memory budget exceeded{label}{where}: peak RSS "
            f"{self._peak_rss_mb:.0f} MB > {MAX_RSS_ENV_VAR}={self.max_rss_mb:.0f} MB",
            stage=stage,
            peak_rss_mb=self._peak_rss_mb,
            max_rss_mb=self.max_rss_mb,
        )

    @contextmanager
    def stage(self, name: str) -> Iterator[RunMemoryMonitor]:
        """Record RSS at the end of a stage and enforce the budget on success.

        The budget is checked only when the wrapped block completes without
        error, so a genuine failure inside the stage is never masked by a
        memory error.
        """
        try:
            yield self
        finally:
            self.sample(name)
        self.check(name)

    def snapshot(self) -> dict[str, object]:
        """Return a JSON-serializable resource-usage summary."""
        return {
            "schema_version": "1.0.0",
            "artifact_type": "alpha_lab_resource_usage",
            "monitor_available": self._available,
            "max_rss_mb_budget": self.max_rss_mb,
            "peak_rss_mb": self._peak_rss_mb,
            "stage_rss_mb": dict(self._stage_rss_mb),
            "note": (
                "Soft RSS guard sampled at stage boundaries; values are "
                "non-deterministic and excluded from golden comparisons."
            ),
        }

    def write_resource_usage(self, output_dir: str | Path) -> Path | None:
        """Write the snapshot to ``resource_usage.json`` under ``output_dir``.

        Returns the written path, or ``None`` when ``output_dir`` does not exist
        (e.g. an early failure before any artifacts were created).
        """
        directory = Path(output_dir)
        if not directory.is_dir():
            return None
        path = directory / RESOURCE_USAGE_ARTIFACT_NAME
        path.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path


__all__ = [
    "MAX_RSS_ENV_VAR",
    "RESOURCE_USAGE_ARTIFACT_NAME",
    "RunMemoryMonitor",
]
