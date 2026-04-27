"""Schemas for the single-factor fast-screen page.

Ten metric cards, four charts, one verdict. Status enum replaces silent 0.0
fallbacks: a field that was not computed renders differently from a field that
was computed and is genuinely zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class MetricStatus(StrEnum):
    """Status of a metric card or a Tier-2 module on disk.

    - ``computed``: value is present and trustworthy.
    - ``skipped``: configuration disables this computation.
    - ``not_applicable``: meaningless for this factor (e.g. sub-day half-life on daily data).
    - ``missing_input``: an upstream input (e.g. Barra exposures) is absent.
    - ``partial``: value computed but below a sample-sufficiency threshold; caller must warn.
    - ``not_computed``: Tier-2 only — module was never run for this run_id.
    - ``running``: Tier-2 only — module currently executing.
    - ``failed``: Tier-2 only — module raised.
    - ``locked``: Tier-2 only — gated behind a Tier-1 PASS/WARN.
    """

    COMPUTED = "computed"
    SKIPPED = "skipped"
    NOT_APPLICABLE = "not_applicable"
    MISSING_INPUT = "missing_input"
    PARTIAL = "partial"
    NOT_COMPUTED = "not_computed"
    RUNNING = "running"
    FAILED = "failed"
    LOCKED = "locked"


CORE_METRIC_KEYS: tuple[str, ...] = (
    "mean_rank_ic",
    "rank_ic_ir",
    "ic_positive_ratio",
    "group_monotonicity",
    "ic_half_life",
    "turnover",
    "coverage",
    "ls_sharpe_net",
    "ic_t_stat",
    "max_drawdown",
)

CORE_CHART_KEYS: tuple[str, ...] = (
    "rolling_rank_ic",
    "ic_decay",
    "group_mean_return",
    "ls_cum_nav_net",
)


@dataclass(frozen=True)
class MetricCard:
    """One metric tile on the fast-screen page.

    ``value`` is the primary display number. ``secondary`` holds optional
    companions shown in the same tile (e.g. Q5-Q1 spread alongside Kendall
    tau for group monotonicity). ``unit`` is free-form ("", "%", "d", "stk").
    ``note`` explains non-computed statuses (e.g. "missing barra exposures").
    """

    key: str
    label: str
    value: float | None
    status: MetricStatus
    unit: str = ""
    secondary: dict[str, float | int | str] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class ChartSeries:
    """One chart payload.

    Shape is chart-specific, kept as plain JSON-ish dict so the frontend can
    render without knowing the chart type a priori. ``status`` lets the UI
    render an "unavailable" placeholder rather than empty axes.
    """

    key: str
    label: str
    kind: str  # "line" | "bar" | "area"
    x: list[Any]
    y: list[float | None]
    status: MetricStatus
    extras: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "x": list(self.x),
            "y": list(self.y),
            "status": self.status.value,
            "extras": dict(self.extras),
            "note": self.note,
        }


@dataclass(frozen=True)
class Verdict:
    """Gating outcome shown in Row-3 of the page.

    ``status`` is one of ``pass`` / ``warn`` / ``fail``.
    ``triggered_rules`` is capped at three entries in the UI.
    ``next_step`` is a short human hint ("Run deep dive" / "Sample too short").
    """

    status: str
    triggered_rules: list[str]
    next_step: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "triggered_rules": list(self.triggered_rules),
            "next_step": self.next_step,
        }


@dataclass(frozen=True)
class Tier2ModuleStatus:
    """Per-module Tier-2 status, written to ``tier2/<module>/status.json``.

    ``inputs_hash`` is a short digest of the Tier-1 inputs (spec + factor_df
    shape). When Tier-1 re-runs with different inputs the Tier-2 entry is
    marked *stale* in the UI but not automatically recomputed.
    """

    module: str
    status: MetricStatus
    computed_at: str = ""
    duration_sec: float = 0.0
    inputs_hash: str = ""
    stale: bool = False
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass(frozen=True)
class FastScreenResult:
    """Tier-1 bundle. Serialised to ``tier1/result.json``."""

    factor_name: str
    run_id: str
    universe: str
    frequency: str
    window: dict[str, str]  # {"start": iso, "end": iso}
    metrics: list[MetricCard]
    charts: list[ChartSeries]
    verdict: Verdict
    inputs_hash: str
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "factor_name": self.factor_name,
            "run_id": self.run_id,
            "universe": self.universe,
            "frequency": self.frequency,
            "window": dict(self.window),
            "metrics": [m.to_dict() for m in self.metrics],
            "charts": [c.to_dict() for c in self.charts],
            "verdict": self.verdict.to_dict(),
            "inputs_hash": self.inputs_hash,
            "generated_at": self.generated_at,
        }

    def metric(self, key: str) -> MetricCard | None:
        for m in self.metrics:
            if m.key == key:
                return m
        return None


def metric_card(
    key: str,
    label: str,
    value: float | None,
    status: MetricStatus,
    *,
    unit: str = "",
    secondary: dict[str, float | int | str] | None = None,
    note: str = "",
) -> MetricCard:
    """Helper to build a MetricCard with safe defaults."""
    return MetricCard(
        key=key,
        label=label,
        value=value,
        status=status,
        unit=unit,
        secondary=dict(secondary or {}),
        note=note,
    )
