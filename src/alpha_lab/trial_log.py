from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_lab.config import PROCESSED_DATA_DIR
from alpha_lab.experiment import ExperimentResult

DEFAULT_TRIAL_LOG_PATH: Path = PROCESSED_DATA_DIR / "trial_log.csv"

TRIAL_LOG_COLUMNS: tuple[str, ...] = (
    "experiment_name",
    "trial_id",
    "trial_count",
    "factor_name",
    "label_name",
    "dataset_id",
    "dataset_hash",
    "validation_scheme",
    "verdict",
    "mean_ic",
    "mean_rank_ic",
    "ic_ir",
    "mean_long_short_return",
    "mean_long_short_turnover",
    "run_timestamp_utc",
    "git_commit",
)


def trial_row_from_result(result: ExperimentResult, *, experiment_name: str) -> pd.DataFrame:
    """Build a canonical one-row trial record from an ExperimentResult."""
    factor_name = (
        str(result.factor_df["factor"].iloc[0]) if not result.factor_df.empty else "unknown"
    )
    label_name = (
        str(result.label_df["factor"].iloc[0]) if not result.label_df.empty else "unknown"
    )
    md = result.metadata
    row = {
        "experiment_name": experiment_name,
        "trial_id": md.trial_id if md is not None else None,
        "trial_count": md.trial_count if md is not None else None,
        "factor_name": factor_name,
        "label_name": label_name,
        "dataset_id": md.dataset_id if md is not None else None,
        "dataset_hash": md.dataset_hash if md is not None else None,
        "validation_scheme": md.validation.scheme if md and md.validation else None,
        "verdict": md.verdict if md is not None else None,
        "mean_ic": result.summary.mean_ic,
        "mean_rank_ic": result.summary.mean_rank_ic,
        "ic_ir": result.summary.ic_ir,
        "mean_long_short_return": result.summary.mean_long_short_return,
        "mean_long_short_turnover": result.summary.mean_long_short_turnover,
        "run_timestamp_utc": result.provenance.run_timestamp_utc,
        "git_commit": result.provenance.git_commit,
    }
    return pd.DataFrame([row], columns=list(TRIAL_LOG_COLUMNS))


def append_trial_log(
    row: pd.DataFrame,
    path: str | Path = DEFAULT_TRIAL_LOG_PATH,
) -> None:
    """Append a one-row trial log record with strict schema checks."""
    if not isinstance(row, pd.DataFrame):
        raise TypeError(f"row must be a DataFrame, got {type(row).__name__}")
    if len(row) != 1:
        raise ValueError(f"row must contain exactly one record, got {len(row)}")
    missing = set(TRIAL_LOG_COLUMNS) - set(row.columns)
    if missing:
        raise ValueError(f"row is missing required columns: {sorted(missing)}")
    extra = set(row.columns) - set(TRIAL_LOG_COLUMNS)
    if extra:
        raise ValueError(f"row has unexpected columns: {sorted(extra)}")

    out = row[list(TRIAL_LOG_COLUMNS)]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        existing = pd.read_csv(target)
        if list(existing.columns) != list(TRIAL_LOG_COLUMNS):
            raise ValueError(
                "trial log has incompatible schema; refusing append to avoid drift"
            )
        out.to_csv(target, mode="a", header=False, index=False)
    else:
        out.to_csv(target, index=False)


def load_trial_log(path: str | Path = DEFAULT_TRIAL_LOG_PATH) -> pd.DataFrame:
    """Load trial log or return an empty schema-compatible DataFrame."""
    target = Path(path)
    if not target.exists():
        return pd.DataFrame(columns=list(TRIAL_LOG_COLUMNS))
    df = pd.read_csv(target)
    if list(df.columns) != list(TRIAL_LOG_COLUMNS):
        raise ValueError("trial log schema mismatch")
    return df.reset_index(drop=True)
