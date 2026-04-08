"""Central validator for raw price-panel input.

This module provides a single reusable function that enforces the contract for
any raw long-form price panel before it enters the research pipeline.  Calling
``validate_price_panel`` at system entrypoints (CLI, experiment runner) ensures
that downstream code can assume a clean, consistent input rather than silently
propagating bad data.

This is a validation layer, not a data-ingestion framework.  It raises
:class:`~alpha_lab.exceptions.AlphaLabDataError` with an informative message;
it does not repair or impute data.
"""
from __future__ import annotations

import pandas as pd

from alpha_lab.exceptions import AlphaLabDataError

REQUIRED_PRICE_COLUMNS: frozenset[str] = frozenset({"date", "asset", "close"})


def validate_price_panel(df: pd.DataFrame) -> None:
    """Validate a raw long-form price panel.

    Raises ``ValueError`` on the first contract violation found.  All checks
    are ordered from cheapest/most structural (columns, emptiness) to more
    expensive (duplicate scan, value range).

    Parameters
    ----------
    df:
        Candidate price panel.  Expected columns: ``date``, ``asset``,
        ``close``.  Additional columns are permitted and ignored.

    Raises
    ------
    ValueError
        If any contract is violated.  The message identifies the specific
        violation and, where possible, the count of affected rows.
    """
    if not isinstance(df, pd.DataFrame):
        raise AlphaLabDataError(
            f"Price panel must be a pandas DataFrame, got {type(df).__name__}"
        )

    # --- Required columns ---------------------------------------------------
    missing = REQUIRED_PRICE_COLUMNS - set(df.columns)
    if missing:
        raise AlphaLabDataError(
            f"Price panel is missing required columns: {sorted(missing)}.  "
            f"Required: {sorted(REQUIRED_PRICE_COLUMNS)}."
        )

    # --- Not empty ----------------------------------------------------------
    if df.empty:
        raise AlphaLabDataError("Price panel is empty (zero rows).")

    # --- Date column: no NaT ------------------------------------------------
    dates = pd.to_datetime(df["date"], errors="coerce")
    n_nat = int(dates.isna().sum())
    if n_nat > 0:
        raise AlphaLabDataError(
            f"Price panel 'date' column contains {n_nat} unparseable or NaT "
            "value(s).  Ensure all dates are in a parseable format (e.g. YYYY-MM-DD)."
        )

    # --- Asset column: no null or empty strings -----------------------------
    asset_null = int(df["asset"].isna().sum())
    if asset_null > 0:
        raise AlphaLabDataError(
            f"Price panel 'asset' column contains {asset_null} null value(s)."
        )
    asset_empty = int((df["asset"].astype(str).str.strip() == "").sum())
    if asset_empty > 0:
        raise AlphaLabDataError(
            f"Price panel 'asset' column contains {asset_empty} empty string(s)."
        )

    # --- No duplicate (date, asset) -----------------------------------------
    n_dupes = int(df.duplicated(subset=["date", "asset"]).sum())
    if n_dupes > 0:
        raise AlphaLabDataError(
            f"Price panel contains {n_dupes} duplicate (date, asset) row(s).  "
            "Each (date, asset) pair must appear at most once."
        )

    # --- Close: no NaN ------------------------------------------------------
    n_nan_close = int(df["close"].isna().sum())
    if n_nan_close > 0:
        raise AlphaLabDataError(
            f"Price panel 'close' column contains {n_nan_close} NaN value(s)."
        )

    # --- Close: must be positive --------------------------------------------
    n_nonpos = int((df["close"] <= 0).sum())
    if n_nonpos > 0:
        raise AlphaLabDataError(
            f"Price panel 'close' column contains {n_nonpos} non-positive "
            "value(s).  Prices must be strictly positive."
        )
