from __future__ import annotations

import pandas as pd

from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS

_REQUIRED_COLS = {"date", "asset", "close"}


def low_volatility(
    df: pd.DataFrame,
    *,
    window: int = 20,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute a low-volatility factor from close-only data.

    The factor at date ``t`` is the negative rolling standard deviation of
    one-row simple returns over the trailing ``window`` observations for each
    asset. Higher values therefore indicate lower realized volatility:

    ``-std(close[t] / close[t-1] - 1, ..., close[t-window+1] / close[t-window] - 1)``
    """
    if df.empty:
        return pd.DataFrame(columns=FACTOR_OUTPUT_COLUMNS)

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise KeyError(f"Input DataFrame is missing required columns: {missing}")

    if window <= 0:
        raise ValueError("'window' must be a positive integer")

    if df["date"].isna().any():
        raise ValueError("Input 'date' column contains NaN/NaT values.")

    if df["asset"].isna().any():
        raise ValueError("Input 'asset' column contains NaN values.")

    dupes = df.duplicated(subset=["date", "asset"])
    if dupes.any():
        raise ValueError(f"Duplicate (date, asset) pairs found:\n{df[dupes][['date', 'asset']]}")

    if min_periods is None:
        min_periods = window
    if min_periods <= 0:
        raise ValueError("'min_periods' must be a positive integer")
    if min_periods > window:
        raise ValueError("'min_periods' cannot exceed window")

    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    df_copy = df_copy.sort_values(["asset", "date"]).reset_index(drop=True)

    close = df_copy["close"].where(df_copy["close"] > 0)
    by_asset = close.groupby(df_copy["asset"], sort=False)
    simple_return = by_asset.pct_change(fill_method=None)
    realized_vol = (
        simple_return.groupby(df_copy["asset"], sort=False)
        .rolling(window, min_periods=min_periods)
        .std(ddof=1)
        .reset_index(level=0, drop=True)
    )

    result = df_copy[["date", "asset"]].copy()
    result["factor"] = f"low_volatility_{window}d"
    result["value"] = -realized_vol
    result = result.loc[:, FACTOR_OUTPUT_COLUMNS].reset_index(drop=True)
    return result
