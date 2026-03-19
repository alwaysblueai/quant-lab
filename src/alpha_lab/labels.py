from __future__ import annotations

import pandas as pd

from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS

_REQUIRED_COLS = {"date", "asset", "close"}


def forward_return(df: pd.DataFrame, *, horizon: int = 1) -> pd.DataFrame:
    """Compute forward returns using each asset's own ordered history.

    The label at date ``t`` is defined as ``close[t + horizon] / close[t] - 1``,
    where ``t + horizon`` is measured in row count within each asset's own
    sorted history. This preserves timestamp discipline: the label is stored at
    timestamp ``t`` for later evaluation against features observed at ``t``,
    while the value itself depends only on strictly future prices.

    Parameters
    ----------
    df:
        Long-form price table with columns ``[date, asset, close]``. Rows need
        not be sorted. Duplicate ``(date, asset)`` pairs are rejected.
        Non-positive prices (<= 0) are treated as missing for the return
        calculation.
    horizon:
        Number of future per-asset rows used in the forward return. Must be a
        positive integer.

    Returns
    -------
    pd.DataFrame
        Canonical long-form output with columns ``[date, asset, factor, value]``.
        ``factor`` is set to ``forward_return_{horizon}``.
    """
    if df.empty:
        return pd.DataFrame(columns=FACTOR_OUTPUT_COLUMNS)

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise KeyError(f"Input DataFrame is missing required columns: {missing}")

    if horizon <= 0:
        raise ValueError("'horizon' must be a positive integer")

    if df["date"].isna().any():
        raise ValueError("Input 'date' column contains NaN/NaT values.")

    if df["asset"].isna().any():
        raise ValueError("Input 'asset' column contains NaN values.")

    dupes = df.duplicated(subset=["date", "asset"])
    if dupes.any():
        raise ValueError(f"Duplicate (date, asset) pairs found:\n{df[dupes][['date', 'asset']]}")

    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    df_copy = df_copy.sort_values(["asset", "date"]).reset_index(drop=True)

    close = df_copy["close"].where(df_copy["close"] > 0)
    next_close = close.groupby(df_copy["asset"], sort=False).shift(-horizon)
    value = next_close.div(close).sub(1.0)

    result = df_copy[["date", "asset"]].copy()
    result["factor"] = f"forward_return_{horizon}"
    result["value"] = value
    result = result.loc[:, FACTOR_OUTPUT_COLUMNS].reset_index(drop=True)

    return result
