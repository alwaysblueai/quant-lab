from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

HANDOFF_SCHEMA_VERSION = "2.0.0"
PORTFOLIO_CONSTRUCTION_SCHEMA_VERSION = "1.0.0"
EXECUTION_ASSUMPTIONS_SCHEMA_VERSION = "1.0.0"

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
            raise ValueError(
                "portfolio construction schema_version must be "
                f"{PORTFOLIO_CONSTRUCTION_SCHEMA_VERSION}"
            )
        if self.rebalance_frequency <= 0:
            raise ValueError("rebalance_frequency must be > 0")
        if self.top_k is not None and self.top_k <= 0:
            raise ValueError("top_k must be > 0")
        if self.bottom_k is not None and self.bottom_k <= 0:
            raise ValueError("bottom_k must be > 0")
        if self.max_weight <= 0:
            raise ValueError("max_weight must be > 0")
        if self.min_weight is not None and self.min_weight > self.max_weight:
            raise ValueError("min_weight must be <= max_weight")
        if self.gross_limit <= 0:
            raise ValueError("gross_limit must be > 0")
        if abs(self.net_limit) > self.gross_limit:
            raise ValueError("abs(net_limit) must be <= gross_limit")
        if not (0.0 <= self.cash_buffer < 1.0):
            raise ValueError("cash_buffer must be in [0, 1)")
        if self.signal_name is not None and not self.signal_name.strip():
            raise ValueError("signal_name must be non-empty when provided")
        for i, constraint in enumerate(self.post_construction_constraints):
            if not isinstance(constraint, str) or not constraint.strip():
                raise ValueError(
                    "post_construction_constraints entries must be non-empty strings; "
                    f"invalid value at index {i}"
                )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PortfolioConstructionSpec:
        return cls(**dict(payload))

    def with_signal_name(self, signal_name: str) -> PortfolioConstructionSpec:
        if not signal_name.strip():
            raise ValueError("signal_name must be non-empty")
        return PortfolioConstructionSpec(**{**self.to_dict(), "signal_name": signal_name})


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

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_ASSUMPTIONS_SCHEMA_VERSION:
            raise ValueError(
                "execution assumptions schema_version must be "
                f"{EXECUTION_ASSUMPTIONS_SCHEMA_VERSION}"
            )
        if self.execution_delay_bars < 0:
            raise ValueError("execution_delay_bars must be >= 0")
        if self.lot_size_rule == "none" and self.lot_size is not None:
            raise ValueError("lot_size must be omitted when lot_size_rule='none'")
        if self.lot_size_rule != "none":
            if self.lot_size is None or self.lot_size <= 0:
                raise ValueError("lot_size must be > 0 when lot_size_rule is enabled")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExecutionAssumptionsSpec:
        return cls(**dict(payload))


def validate_handoff_artifact(path: str | Path) -> None:
    artifact_path = Path(path).resolve()
    if not artifact_path.exists():
        raise FileNotFoundError(f"handoff artifact does not exist: {artifact_path}")
    if not artifact_path.is_dir():
        raise ValueError(f"handoff artifact is not a directory: {artifact_path}")

    missing = [name for name in _REQUIRED_ARTIFACT_FILES_V2 if not (artifact_path / name).exists()]
    if missing:
        raise FileNotFoundError(
            "handoff artifact is missing required files: " + ", ".join(sorted(missing))
        )
