from __future__ import annotations

import platform
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from alpha_lab.timing import DelaySpec


@dataclass(frozen=True)
class ValidationMetadata:
    """Validation assumptions attached to one experiment."""

    scheme: str
    train_end: pd.Timestamp | None = None
    test_start: pd.Timestamp | None = None
    val_start: pd.Timestamp | None = None
    purge_periods: int = 0
    embargo_periods: int = 0
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.scheme.strip():
            raise ValueError("scheme must be a non-empty string")
        if self.purge_periods < 0:
            raise ValueError("purge_periods must be >= 0")
        if self.embargo_periods < 0:
            raise ValueError("embargo_periods must be >= 0")

    def to_dict(self) -> dict[str, object]:
        return {
            "scheme": self.scheme,
            "train_end": self.train_end.isoformat() if self.train_end is not None else None,
            "test_start": self.test_start.isoformat() if self.test_start is not None else None,
            "val_start": self.val_start.isoformat() if self.val_start is not None else None,
            "purge_periods": self.purge_periods,
            "embargo_periods": self.embargo_periods,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class RuntimeEnvironment:
    """Minimal runtime environment needed for reproducibility audits."""

    python_version: str
    pandas_version: str
    numpy_version: str
    platform: str

    @classmethod
    def capture(cls) -> RuntimeEnvironment:
        return cls(
            python_version=platform.python_version(),
            pandas_version=pd.__version__,
            numpy_version=np.__version__,
            platform=platform.platform(),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "python_version": self.python_version,
            "pandas_version": self.pandas_version,
            "numpy_version": self.numpy_version,
            "platform": self.platform,
        }


@dataclass(frozen=True)
class ExperimentMetadata:
    """Research-governance metadata for one experiment run."""

    hypothesis: str | None = None
    research_question: str | None = None
    factor_spec: str | None = None
    dataset_id: str | None = None
    dataset_hash: str | None = None
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    caveats: tuple[str, ...] = field(default_factory=tuple)
    trial_id: str | None = None
    trial_count: int | None = None
    validation: ValidationMetadata | None = None
    delay: DelaySpec | None = None
    verdict: str | None = None
    interpretation: str | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)
    environment: RuntimeEnvironment = field(default_factory=RuntimeEnvironment.capture)

    def __post_init__(self) -> None:
        if self.trial_count is not None and self.trial_count <= 0:
            raise ValueError("trial_count must be >= 1 when provided")
        if self.verdict is not None and not self.verdict.strip():
            raise ValueError("verdict must be non-empty when provided")

    def to_dict(self) -> dict[str, object]:
        return {
            "hypothesis": self.hypothesis,
            "research_question": self.research_question,
            "factor_spec": self.factor_spec,
            "dataset_id": self.dataset_id,
            "dataset_hash": self.dataset_hash,
            "assumptions": list(self.assumptions),
            "caveats": list(self.caveats),
            "trial_id": self.trial_id,
            "trial_count": self.trial_count,
            "validation": self.validation.to_dict() if self.validation is not None else None,
            "delay": self.delay.to_dict() if self.delay is not None else None,
            "verdict": self.verdict,
            "interpretation": self.interpretation,
            "warnings": list(self.warnings),
            "environment": self.environment.to_dict(),
        }
