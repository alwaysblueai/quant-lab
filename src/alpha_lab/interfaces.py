from __future__ import annotations

from typing import Protocol

import pandas as pd

from alpha_lab.exceptions import AlphaLabDataError

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


_VALIDATED_ATTR = "_alpha_lab_factor_output_validated"


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
    # Fast path: already validated. The fingerprint guards against in-place
    # mutation invalidating the cached result — any change to row count or
    # the top value reshuffles the fingerprint and forces revalidation.
    cached = df.attrs.get(_VALIDATED_ATTR)
    if cached is not None and cached == _validation_fingerprint(df):
        return

    required_cols = set(FACTOR_OUTPUT_COLUMNS)

    # --- Required columns ---------------------------------------------------
    missing = required_cols - set(df.columns)
    if missing:
        raise AlphaLabDataError(f"Missing required columns: {missing}")

    # --- Not empty ----------------------------------------------------------
    if df.empty:
        raise AlphaLabDataError("Factor output is empty")

    # --- All-NaN value column -----------------------------------------------
    if df["value"].isna().all():
        raise AlphaLabDataError("Factor values are all NaN")

    # --- NaT in date --------------------------------------------------------
    dates = pd.to_datetime(df["date"], errors="coerce")
    n_nat = int(dates.isna().sum())
    if n_nat > 0:
        raise AlphaLabDataError(
            f"Factor output 'date' column contains {n_nat} NaT or unparseable "
            "value(s).  All dates must be valid timestamps."
        )

    # --- Null or empty asset strings ----------------------------------------
    asset_null = int(df["asset"].isna().sum())
    if asset_null > 0:
        raise AlphaLabDataError(
            f"Factor output 'asset' column contains {asset_null} null value(s)."
        )
    asset_empty = int((df["asset"].astype(str).str.strip() == "").sum())
    if asset_empty > 0:
        raise AlphaLabDataError(
            f"Factor output 'asset' column contains {asset_empty} empty string(s)."
        )

    # --- Null or empty factor-name strings ----------------------------------
    factor_null = int(df["factor"].isna().sum())
    if factor_null > 0:
        raise AlphaLabDataError(
            f"Factor output 'factor' column contains {factor_null} null value(s)."
        )
    factor_empty = int((df["factor"].astype(str).str.strip() == "").sum())
    if factor_empty > 0:
        raise AlphaLabDataError(
            f"Factor output 'factor' column contains {factor_empty} empty string(s)."
        )

    # --- Duplicate (date, asset, factor) ------------------------------------
    dupes = df.duplicated(subset=["date", "asset", "factor"])
    if dupes.any():
        raise AlphaLabDataError("Factor output contains duplicate (date, asset, factor) rows")

    df.attrs[_VALIDATED_ATTR] = _validation_fingerprint(df)


def _validation_fingerprint(df: pd.DataFrame) -> tuple:
    """Build a cache key that is sensitive to in-place mutation.

    The fingerprint must change whenever a mutation could invalidate any of the
    checks ``validate_factor_output`` performs (row count, NaN coverage,
    boundary date/value identity, duplicate-row count). The old fingerprint
    only sampled ``(len, value.iat[0])``, which a caller could bypass by
    mutating any row past the first while keeping the length unchanged.

    Cheap O(1)/O(n_columns) summary, computed once per validated frame and
    cached on ``df.attrs``.
    """
    n = len(df)
    if n == 0:
        return (0,)

    def _scalar(value: object) -> object:
        if isinstance(value, float) and value != value:
            return "__nan__"
        return value

    value_col = df["value"]
    date_col = df["date"]
    asset_col = df["asset"]
    # ``isna`` count guards against a NaN being introduced (or removed) in any
    # position without changing the row count.
    n_value_nan = int(value_col.isna().sum())
    # Sampling boundary rows and the midpoint catches the bulk of in-place
    # edits without scanning the full column.
    mid = n // 2
    return (
        n,
        n_value_nan,
        _scalar(value_col.iat[0]),
        _scalar(value_col.iat[-1]),
        _scalar(value_col.iat[mid]),
        _scalar(date_col.iat[0]),
        _scalar(date_col.iat[-1]),
        _scalar(asset_col.iat[0]),
        _scalar(asset_col.iat[-1]),
    )
