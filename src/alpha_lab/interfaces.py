from __future__ import annotations

from typing import Protocol

import pandas as pd

FACTOR_OUTPUT_COLUMNS = ("date", "asset", "factor", "value")


class Factor(Protocol):
    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Input:
            df must contain:
                - date
                - asset
                - required raw features

        Output:
            DataFrame with columns:
                - date
                - asset
                - factor
                - value
        """
        ...


def validate_factor_output(df: pd.DataFrame) -> None:
    """Validate the canonical factor output contract.

    Enforces the full documented contract for long-form factor DataFrames.
    Raises ``ValueError`` with an informative message on the first violation.

    Parameters
    ----------
    df:
        Candidate factor output.  Expected columns: ``date``, ``asset``,
        ``factor``, ``value``.

    Raises
    ------
    ValueError
        If any contract is violated.
    """
    required_cols = set(FACTOR_OUTPUT_COLUMNS)

    # --- Required columns ---------------------------------------------------
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # --- Not empty ----------------------------------------------------------
    if df.empty:
        raise ValueError("Factor output is empty")

    # --- All-NaN value column -----------------------------------------------
    if df["value"].isna().all():
        raise ValueError("Factor values are all NaN")

    # --- NaT in date --------------------------------------------------------
    dates = pd.to_datetime(df["date"], errors="coerce")
    n_nat = int(dates.isna().sum())
    if n_nat > 0:
        raise ValueError(
            f"Factor output 'date' column contains {n_nat} NaT or unparseable "
            "value(s).  All dates must be valid timestamps."
        )

    # --- Null or empty asset strings ----------------------------------------
    asset_null = int(df["asset"].isna().sum())
    if asset_null > 0:
        raise ValueError(
            f"Factor output 'asset' column contains {asset_null} null value(s)."
        )
    asset_empty = int((df["asset"].astype(str).str.strip() == "").sum())
    if asset_empty > 0:
        raise ValueError(
            f"Factor output 'asset' column contains {asset_empty} empty string(s)."
        )

    # --- Null or empty factor-name strings ----------------------------------
    factor_null = int(df["factor"].isna().sum())
    if factor_null > 0:
        raise ValueError(
            f"Factor output 'factor' column contains {factor_null} null value(s)."
        )
    factor_empty = int((df["factor"].astype(str).str.strip() == "").sum())
    if factor_empty > 0:
        raise ValueError(
            f"Factor output 'factor' column contains {factor_empty} empty string(s)."
        )

    # --- Duplicate (date, asset, factor) ------------------------------------
    dupes = df.duplicated(subset=["date", "asset", "factor"])
    if dupes.any():
        raise ValueError("Factor output contains duplicate (date, asset, factor) rows")
