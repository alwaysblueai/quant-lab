"""Experimental handoff contracts for future Level 3 replay integrations.

These contracts remain available for forward compatibility but are not part of
the default Level 1/2 research workflow.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

from alpha_lab.exceptions import AlphaLabConfigError

HANDOFF_SCHEMA_VERSION = "2.0.0"
PORTFOLIO_CONSTRUCTION_SCHEMA_VERSION = "1.0.0"
EXECUTION_ASSUMPTIONS_SCHEMA_VERSION = "1.0.0"
HANDOFF_TIMING_SEMANTIC_KEYS: tuple[str, ...] = (
    "signal_timestamp_convention",
    "signal_bar_timestamp",
    "signal_computed_at",
    "bar_close_known_at",
    "order_submitted_at",
    "execution_bar_timestamp",
    "execution_price_rule",
    "portfolio_formation_timestamp",
    "rebalance_interval_bars",
    "holding_period_bars",
    "uses_closed_bar_only",
    "allows_incomplete_bar_features",
    "source_bar_frequency",
    "target_bar_frequency",
    "aggregation_timestamp_convention",
    "aggregated_value_known_at",
    "completed_aggregation_only",
    "intraday_daily_alignment_rule",
    "daily_feature_effective_time",
    "adjustment_mode",
    "holding_period",
)

_REQUIRED_ARTIFACT_FILES_V2: tuple[str, ...] = (
    "signal_snapshot.csv",
    "universe_mask.csv",
    "tradability_mask.csv",
    "timing.json",
    "experiment_metadata.json",
    "validation_context.json",
    "dataset_fingerprint.json",
    "portfolio_construction.json",
    "execution_assumptions.json",
    "manifest.json",
)


@dataclass(frozen=True)
class PortfolioConstructionSpec:
    """Contract describing signal-to-portfolio intent for external replay."""

    schema_version: str = PORTFOLIO_CONSTRUCTION_SCHEMA_VERSION
    construction_method: str = "top_bottom_k"
    signal_name: str | None = None
    rebalance_frequency: int = 1
    rebalance_calendar: str = "business_day"
    long_short: bool = True
    top_k: int | None = 20
    bottom_k: int | None = 20
    weight_method: str = "rank"
    weight_clip: float | None = None
    max_weight: float = 0.1
    min_weight: float | None = None
    gross_limit: float = 1.0
    net_limit: float = 0.0
    cash_buffer: float = 0.0
    neutralization_required: bool = False
    post_construction_constraints: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.schema_version != PORTFOLIO_CONSTRUCTION_SCHEMA_VERSION:
            raise AlphaLabConfigError(
                "portfolio construction schema_version must be "
                f"{PORTFOLIO_CONSTRUCTION_SCHEMA_VERSION}"
            )
        if self.rebalance_frequency <= 0:
            raise AlphaLabConfigError("rebalance_frequency must be > 0")
        if self.top_k is not None and self.top_k <= 0:
            raise AlphaLabConfigError("top_k must be > 0")
        if self.bottom_k is not None and self.bottom_k <= 0:
            raise AlphaLabConfigError("bottom_k must be > 0")
        if self.max_weight <= 0:
            raise AlphaLabConfigError("max_weight must be > 0")
        if self.min_weight is not None and self.min_weight > self.max_weight:
            raise AlphaLabConfigError("min_weight must be <= max_weight")
        if self.gross_limit <= 0:
            raise AlphaLabConfigError("gross_limit must be > 0")
        if abs(self.net_limit) > self.gross_limit:
            raise AlphaLabConfigError("abs(net_limit) must be <= gross_limit")
        if not (0.0 <= self.cash_buffer < 1.0):
            raise AlphaLabConfigError("cash_buffer must be in [0, 1)")
        if self.signal_name is not None and not self.signal_name.strip():
            raise AlphaLabConfigError("signal_name must be non-empty when provided")
        for i, constraint in enumerate(self.post_construction_constraints):
            if not isinstance(constraint, str) or not constraint.strip():
                raise AlphaLabConfigError(
                    "post_construction_constraints entries must be non-empty strings; "
                    f"invalid value at index {i}"
                )

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PortfolioConstructionSpec:
        return cls(**dict(payload))

    def with_signal_name(self, signal_name: str) -> PortfolioConstructionSpec:
        if not signal_name.strip():
            raise AlphaLabConfigError("signal_name must be non-empty")
        payload = self.to_dict()
        payload["signal_name"] = signal_name
        return PortfolioConstructionSpec(**payload)


@dataclass(frozen=True)
class ExecutionAssumptionsSpec:
    """Contract describing execution replay assumptions for external engines."""

    schema_version: str = EXECUTION_ASSUMPTIONS_SCHEMA_VERSION
    fill_price_rule: str = "next_open"
    execution_delay_bars: int = 1
    commission_model: str = "bps"
    slippage_model: str = "fixed_bps"
    lot_size_rule: str = "none"
    lot_size: int | None = None
    cash_buffer: float = 0.0
    partial_fill_policy: str = "allow_partial"
    suspension_policy: str = "skip_trade"
    price_limit_policy: str = "skip_trade"
    trade_when_not_tradable: bool = False
    allow_same_day_reentry: bool = False
    session_boundary_policy: str = "next_session"
    session_gap_policy: str = "next_session_open"
    lunch_break_policy: str = "market_default"
    allow_eod_bar_execution: bool = False
    capacity_model: str = "unbounded"
    max_participation_rate: float | None = None
    min_tradable_bar_volume: float | None = None

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_ASSUMPTIONS_SCHEMA_VERSION:
            raise AlphaLabConfigError(
                "execution assumptions schema_version must be "
                f"{EXECUTION_ASSUMPTIONS_SCHEMA_VERSION}"
            )
        if self.execution_delay_bars < 0:
            raise AlphaLabConfigError("execution_delay_bars must be >= 0")
        if self.lot_size_rule == "none" and self.lot_size is not None:
            raise AlphaLabConfigError("lot_size must be omitted when lot_size_rule='none'")
        if self.lot_size_rule != "none":
            if self.lot_size is None or self.lot_size <= 0:
                raise AlphaLabConfigError("lot_size must be > 0 when lot_size_rule is enabled")
        valid_capacity_models = {
            "unbounded",
            "participation_capped",
            "min_bar_volume_gate",
            "participation_capped_with_min_volume",
        }
        if self.capacity_model not in valid_capacity_models:
            raise AlphaLabConfigError(
                "capacity_model must be one of "
                f"{sorted(valid_capacity_models)}; got {self.capacity_model!r}"
            )
        if self.max_participation_rate is not None and not (
            0.0 < self.max_participation_rate <= 1.0
        ):
            raise AlphaLabConfigError("max_participation_rate must be in (0, 1] when provided")
        if self.min_tradable_bar_volume is not None and self.min_tradable_bar_volume < 0:
            raise AlphaLabConfigError("min_tradable_bar_volume must be >= 0 when provided")
        if self.capacity_model in {"participation_capped", "participation_capped_with_min_volume"}:
            if self.max_participation_rate is None:
                raise AlphaLabConfigError(
                    "max_participation_rate is required when capacity_model uses "
                    "participation caps"
                )
        if self.capacity_model in {"min_bar_volume_gate", "participation_capped_with_min_volume"}:
            if self.min_tradable_bar_volume is None:
                raise AlphaLabConfigError(
                    "min_tradable_bar_volume is required when capacity_model uses "
                    "bar-volume gating"
                )

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], asdict(self))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExecutionAssumptionsSpec:
        return cls(**dict(payload))


def validate_handoff_artifact(path: str | Path) -> None:
    artifact_path = Path(path).resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"handoff artifact does not exist: {artifact_path}")
    if not artifact_path.is_dir():
        raise AlphaLabConfigError(f"handoff artifact is not a directory: {artifact_path}")

    missing = [name for name in _REQUIRED_ARTIFACT_FILES_V2 if not (artifact_path / name).exists()]
    if missing:
        raise FileNotFoundError(
            "handoff artifact is missing required files: " + ", ".join(sorted(missing))
        )


def summarize_handoff_timing_semantics(
    timing_payload: Mapping[str, object],
) -> dict[str, object]:
    """Extract known timing-semantic keys from ``timing.json`` payload."""

    out: dict[str, object] = {}
    for key in HANDOFF_TIMING_SEMANTIC_KEYS:
        if key in timing_payload:
            out[key] = timing_payload[key]
    delay_spec = timing_payload.get("delay_spec")
    if isinstance(delay_spec, Mapping):
        if "execution_delay_periods" in delay_spec:
            out["execution_delay_periods"] = delay_spec["execution_delay_periods"]
    return out
