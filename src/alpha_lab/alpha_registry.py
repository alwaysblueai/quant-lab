from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from alpha_lab.config import PROCESSED_DATA_DIR
from alpha_lab.experiment import ExperimentResult

ALPHA_REGISTRY_PATH = PROCESSED_DATA_DIR / "alpha_registry.csv"
ALPHA_LIFECYCLE_STAGES: tuple[str, ...] = (
    "discovery",
    "candidate",
    "approved_for_external_backtest",
    "mature",
    "decaying",
    "retired",
)

ALPHA_REGISTRY_COLUMNS: tuple[str, ...] = (
    "alpha_id",
    "taxonomy",
    "hypothesis",
    "economic_rationale",
    "neutralization_status",
    "validation_status",
    "ic_mean",
    "ic_ir",
    "decay_half_life",
    "lifecycle_stage",
    "tags",
    "notes",
    "provenance",
    "updated_at_utc",
)


@dataclass(frozen=True)
class AlphaRegistryEntry:
    """Lightweight alpha-pool registry entry."""

    alpha_id: str
    taxonomy: str | None = None
    hypothesis: str | None = None
    economic_rationale: str | None = None
    neutralization_status: str | None = None
    validation_status: str | None = None
    ic_mean: float | None = None
    ic_ir: float | None = None
    decay_half_life: float | None = None
    lifecycle_stage: str = "discovery"
    tags: tuple[str, ...] = field(default_factory=tuple)
    notes: str | None = None
    provenance: str | None = None
    updated_at_utc: str | None = None

    def __post_init__(self) -> None:
        if not self.alpha_id.strip():
            raise ValueError("alpha_id must be non-empty")
        if self.lifecycle_stage not in ALPHA_LIFECYCLE_STAGES:
            raise ValueError(
                "lifecycle_stage must be one of "
                f"{list(ALPHA_LIFECYCLE_STAGES)}, got {self.lifecycle_stage!r}"
            )

    def to_row(self) -> dict[str, object]:
        return {
            "alpha_id": self.alpha_id,
            "taxonomy": self.taxonomy,
            "hypothesis": self.hypothesis,
            "economic_rationale": self.economic_rationale,
            "neutralization_status": self.neutralization_status,
            "validation_status": self.validation_status,
            "ic_mean": self.ic_mean,
            "ic_ir": self.ic_ir,
            "decay_half_life": self.decay_half_life,
            "lifecycle_stage": self.lifecycle_stage,
            "tags": "|".join(self.tags),
            "notes": self.notes,
            "provenance": self.provenance,
            "updated_at_utc": self.updated_at_utc or _utc_now(),
        }


def load_alpha_registry(path: str | Path = ALPHA_REGISTRY_PATH) -> pd.DataFrame:
    """Load registry with stable schema."""
    target = Path(path)
    if not target.exists():
        return pd.DataFrame(columns=list(ALPHA_REGISTRY_COLUMNS))
    df = pd.read_csv(target)
    if list(df.columns) != list(ALPHA_REGISTRY_COLUMNS):
        raise ValueError("alpha registry schema mismatch")
    return df.reset_index(drop=True)


def upsert_alpha_registry_entry(
    entry: AlphaRegistryEntry,
    *,
    path: str | Path = ALPHA_REGISTRY_PATH,
) -> pd.DataFrame:
    """Insert or update one alpha registry entry and persist to disk."""
    df = load_alpha_registry(path)
    row = pd.DataFrame([entry.to_row()], columns=list(ALPHA_REGISTRY_COLUMNS))

    if entry.alpha_id in set(df["alpha_id"].astype(str)):
        df = df[df["alpha_id"].astype(str) != entry.alpha_id].copy()
    out = pd.concat([df, row], ignore_index=True)
    out = out.sort_values("alpha_id", kind="mergesort").reset_index(drop=True)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(target, index=False)
    return out


def alpha_entry_from_experiment(
    result: ExperimentResult,
    *,
    alpha_id: str,
    lifecycle_stage: str = "candidate",
    taxonomy: str | None = None,
    tags: tuple[str, ...] = (),
    notes: str | None = None,
) -> AlphaRegistryEntry:
    """Create a registry entry from one experiment result."""
    if lifecycle_stage not in ALPHA_LIFECYCLE_STAGES:
        raise ValueError(
            f"lifecycle_stage must be one of {list(ALPHA_LIFECYCLE_STAGES)}"
        )
    md = result.metadata
    return AlphaRegistryEntry(
        alpha_id=alpha_id,
        taxonomy=taxonomy or (md.factor_spec if md is not None else None),
        hypothesis=md.hypothesis if md is not None else None,
        economic_rationale=md.research_question if md is not None else None,
        neutralization_status="unknown",
        validation_status=(
            md.validation.scheme
            if (md is not None and md.validation is not None)
            else "unspecified"
        ),
        ic_mean=result.summary.mean_rank_ic,
        ic_ir=result.summary.ic_ir,
        decay_half_life=(
            result.factor_report.half_life_periods
            if result.factor_report is not None
            else None
        ),
        lifecycle_stage=lifecycle_stage,
        tags=tags,
        notes=notes,
        provenance=result.provenance.git_commit,
    )


def alpha_registry_stage_summary(registry: pd.DataFrame) -> pd.DataFrame:
    """Summarize alpha counts and mean IC by lifecycle stage."""
    if registry.empty:
        return pd.DataFrame(columns=["lifecycle_stage", "n_alphas", "mean_ic_mean", "mean_ic_ir"])
    out = (
        registry.groupby("lifecycle_stage", sort=True)
        .agg(
            n_alphas=("alpha_id", "nunique"),
            mean_ic_mean=("ic_mean", "mean"),
            mean_ic_ir=("ic_ir", "mean"),
        )
        .reset_index()
    )
    return out.sort_values("lifecycle_stage", kind="mergesort").reset_index(drop=True)


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
