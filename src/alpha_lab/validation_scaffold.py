from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from alpha_lab.purged_validation import purged_fold_summary, purged_kfold_split


@dataclass(frozen=True)
class WalkForwardValidationSpec:
    """Metadata-only validation descriptor for walk-forward experiments."""

    scheme: str = "walk_forward"
    train_size: int = 0
    test_size: int = 0
    step: int = 0
    val_size: int = 0
    purge_periods: int = 0
    embargo_periods: int = 0
    notes: str | None = None

    def __post_init__(self) -> None:
        if not self.scheme.strip():
            raise ValueError("scheme must be non-empty")
        if self.train_size <= 0:
            raise ValueError("train_size must be > 0")
        if self.test_size <= 0:
            raise ValueError("test_size must be > 0")
        if self.step <= 0:
            raise ValueError("step must be > 0")
        if self.val_size < 0:
            raise ValueError("val_size must be >= 0")
        if self.purge_periods < 0:
            raise ValueError("purge_periods must be >= 0")
        if self.embargo_periods < 0:
            raise ValueError("embargo_periods must be >= 0")

    def to_dict(self) -> dict[str, object]:
        return {
            "scheme": self.scheme,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "step": self.step,
            "val_size": self.val_size,
            "purge_periods": self.purge_periods,
            "embargo_periods": self.embargo_periods,
            "notes": self.notes,
        }


def fold_windows_from_summary(fold_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Return a stable fold-window table from walk-forward summary rows."""
    if fold_summary_df.empty:
        return pd.DataFrame(
            columns=[
                "fold_id",
                "train_start",
                "train_end",
                "val_start",
                "val_end",
                "test_start",
                "test_end",
            ]
        )

    required = {"fold_id", "train_start", "train_end", "start_date", "end_date"}
    missing = required - set(fold_summary_df.columns)
    if missing:
        raise ValueError(f"fold_summary_df missing required columns: {sorted(missing)}")

    out = fold_summary_df[["fold_id", "train_start", "train_end", "start_date", "end_date"]].copy()
    out = out.rename(columns={"start_date": "test_start", "end_date": "test_end"})
    out["val_start"] = pd.NaT
    out["val_end"] = pd.NaT
    return out[
        ["fold_id", "train_start", "train_end", "val_start", "val_end", "test_start", "test_end"]
    ].reset_index(drop=True)


def purged_validation_summary(
    samples: pd.DataFrame,
    *,
    n_splits: int,
    decision_col: str = "date",
    start_col: str = "event_start",
    end_col: str = "event_end",
    embargo_periods: int = 0,
) -> pd.DataFrame:
    """Convenience bridge from validation scaffold to purged CV diagnostics."""
    folds = purged_kfold_split(
        samples,
        n_splits=n_splits,
        decision_col=decision_col,
        start_col=start_col,
        end_col=end_col,
        embargo_periods=embargo_periods,
    )
    return purged_fold_summary(folds)
