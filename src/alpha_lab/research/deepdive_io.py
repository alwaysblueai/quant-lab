from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class CaseArtifacts:
    case_dir: Path
    factor_df: pd.DataFrame
    labels_df: pd.DataFrame
    prices: pd.DataFrame
    ic_df: pd.DataFrame
    long_short_df: pd.DataFrame

    @property
    def deepdive_dir(self) -> Path:
        return self.case_dir / "deepdive"


def load_case_artifacts(case_dir: str | Path) -> CaseArtifacts:
    """Load common single-factor case artifacts for Tier-2 notebooks."""
    resolved = Path(case_dir)
    factor_df = read_first_table(
        [
            resolved / "factor_values.parquet",
            resolved / "factor_values.csv",
            resolved / "factor.parquet",
            resolved / "factor.csv",
        ]
    )
    labels_df = read_first_table([resolved / "labels.parquet", resolved / "labels.csv"])
    prices = read_first_table(
        [
            resolved / "prices.parquet",
            resolved / "prices.csv",
            resolved / "panel.parquet",
            resolved / "panel.csv",
        ],
        required=False,
    )
    ic_df = read_first_table(
        [resolved / "ic_timeseries.parquet", resolved / "ic_timeseries.csv"],
        required=False,
    )
    long_short_df = read_first_table(
        [resolved / "long_short_return.parquet", resolved / "long_short_return.csv"],
        required=False,
    )
    return CaseArtifacts(
        case_dir=resolved,
        factor_df=_clean_date_column(factor_df),
        labels_df=_clean_date_column(labels_df),
        prices=_clean_date_column(prices),
        ic_df=_clean_date_column(ic_df),
        long_short_df=_clean_date_column(long_short_df),
    )


def save_deepdive(
    frame: pd.DataFrame,
    case_dir: str | Path | CaseArtifacts,
    name: str,
    *,
    index: bool = False,
) -> Path:
    """Save a Tier-2 result under case_dir/deepdive."""
    out_dir = (
        case_dir.deepdive_dir
        if isinstance(case_dir, CaseArtifacts)
        else Path(case_dir) / "deepdive"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame.to_parquet(path, index=index)
    elif suffix in {".csv", ".txt"}:
        frame.to_csv(path, index=index)
    else:
        raise ValueError("name must end with .csv, .txt, or .parquet")
    return path


def read_first_table(paths: Iterable[str | Path], *, required: bool = True) -> pd.DataFrame:
    """Read the first existing parquet/csv path from a candidate list."""
    candidates = [Path(path) for path in paths]
    for path in candidates:
        if not path.exists():
            continue
        suffix = path.suffix.lower()
        if suffix == ".parquet":
            return pd.read_parquet(path)
        if suffix in {".csv", ".txt"}:
            return pd.read_csv(path)
    if required:
        candidate_text = [str(path) for path in candidates]
        raise FileNotFoundError(f"No table found for candidates: {candidate_text}")
    return pd.DataFrame()


def _clean_date_column(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "date" not in frame.columns:
        return frame
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out
