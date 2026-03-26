from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DelaySpec:
    """Explicit timing contract for research labeling/evaluation.

    This does not simulate execution. It records the assumptions used so
    downstream diagnostics and reports can be audited.
    """

    decision_timestamp: str = "close"
    execution_delay_periods: int = 1
    return_horizon_periods: int = 1
    label_start_offset_periods: int = 0
    label_end_offset_periods: int = 1
    purge_periods: int = 0
    embargo_periods: int = 0
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.decision_timestamp.strip():
            raise ValueError("decision_timestamp must be a non-empty string")
        if self.execution_delay_periods < 0:
            raise ValueError("execution_delay_periods must be >= 0")
        if self.return_horizon_periods <= 0:
            raise ValueError("return_horizon_periods must be >= 1")
        if self.label_start_offset_periods < 0:
            raise ValueError("label_start_offset_periods must be >= 0")
        if self.label_end_offset_periods <= self.label_start_offset_periods:
            raise ValueError(
                "label_end_offset_periods must be greater than label_start_offset_periods"
            )
        if self.purge_periods < 0:
            raise ValueError("purge_periods must be >= 0")
        if self.embargo_periods < 0:
            raise ValueError("embargo_periods must be >= 0")

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON-serialisable representation."""
        return {
            "decision_timestamp": self.decision_timestamp,
            "execution_delay_periods": self.execution_delay_periods,
            "return_horizon_periods": self.return_horizon_periods,
            "label_start_offset_periods": self.label_start_offset_periods,
            "label_end_offset_periods": self.label_end_offset_periods,
            "purge_periods": self.purge_periods,
            "embargo_periods": self.embargo_periods,
            "notes": self.notes,
        }

    @classmethod
    def for_horizon(
        cls,
        horizon: int,
        *,
        decision_timestamp: str = "close",
        execution_delay_periods: int = 1,
        purge_periods: int = 0,
        embargo_periods: int = 0,
        notes: str | None = None,
    ) -> DelaySpec:
        """Construct a DelaySpec aligned with the forward-return horizon."""
        return cls(
            decision_timestamp=decision_timestamp,
            execution_delay_periods=execution_delay_periods,
            return_horizon_periods=horizon,
            label_start_offset_periods=0,
            label_end_offset_periods=horizon,
            purge_periods=purge_periods,
            embargo_periods=embargo_periods,
            notes=notes,
        )


@dataclass(frozen=True)
class LabelMetadata:
    """Serializable metadata describing how labels were defined."""

    label_name: str
    horizon_periods: int
    delay: DelaySpec

    def __post_init__(self) -> None:
        if not self.label_name.strip():
            raise ValueError("label_name must be a non-empty string")
        if self.horizon_periods <= 0:
            raise ValueError("horizon_periods must be >= 1")

    def to_dict(self) -> dict[str, object]:
        return {
            "label_name": self.label_name,
            "horizon_periods": self.horizon_periods,
            "delay": self.delay.to_dict(),
        }
