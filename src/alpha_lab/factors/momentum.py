from __future__ import annotations

import pandas as pd

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError
from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS

_REQUIRED_COLS = {"date", "asset", "close"}


def momentum(
    df: pd.DataFrame,
    *,
    window: int = 20,
    skip_recent: int = 0,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute per-asset momentum factor with optional short-term skip.

    The factor at date *t* is defined as::

        close[t - skip_recent] / close[t - skip_recent - window] - 1

    where the look-back points are measured in **row count** within each
    asset's own sorted price series. Missing observations for other assets do
    not affect the lookback for the current asset. ``skip_recent=0`` reduces
    to the standard trailing ``window``-row return. Positive ``skip_recent``
    skips the most recent rows to avoid short-horizon reversal contamination.

    Parameters
    ----------
    df:
        Long-form price table with columns ``[date, asset, close]``.
        Rows need not be sorted. Duplicate ``(date, asset)`` pairs are
        rejected with ``ValueError``. Non-positive prices (<= 0) are
        treated as ``NaN`` for the return calculation.
    window:
        Look-back in prior per-asset rows. Default 20.
    skip_recent:
        Number of most recent per-asset rows to skip before measuring the
        return window. Default 0.
    min_periods:
        Minimum number of non-NaN close values required within the effective
        per-asset window of ``window + skip_recent + 1`` rows for a value to
        be emitted.
        Defaults to ``window + 1`` (strict: both endpoints and all
        observations in between for the standard no-skip case must be
        present). Lower this to tolerate intra-window NaN gaps while still
        requiring both endpoints.

    Returns
    -------
    pd.DataFrame
        Columns ``[date, asset, factor, value]``, one row per ``(date, asset)``
        present in the input. ``factor`` is set to ``momentum_{window}d``.
        Values are ``NaN`` where there is insufficient history or if prices
        are non-positive.
    """
    if df.empty:
        return pd.DataFrame(columns=FACTOR_OUTPUT_COLUMNS)

    missing = _REQUIRED_COLS - set(df.columns)
    if missing:
        raise AlphaLabDataError(f"Input DataFrame is missing required columns: {missing}")

    if window <= 0:
        raise AlphaLabConfigError("'window' must be a positive integer")
    if skip_recent < 0:
        raise AlphaLabConfigError("'skip_recent' must be a non-negative integer")

    if df["date"].isna().any():
        raise AlphaLabDataError("Input 'date' column contains NaN/NaT values.")

    if df["asset"].isna().any():
        raise AlphaLabDataError("Input 'asset' column contains NaN values.")

    dupes = df.duplicated(subset=["date", "asset"])
    if dupes.any():
        raise AlphaLabDataError(f"Duplicate (date, asset) pairs found:\n{df[dupes][['date', 'asset']]}")

    if min_periods is None:
        min_periods = window + 1

    if min_periods <= 0:
        raise AlphaLabConfigError("'min_periods' must be a positive integer")

    effective_window = window + skip_recent + 1
    if min_periods > effective_window:
        raise AlphaLabConfigError("'min_periods' cannot exceed window + skip_recent + 1")

    df_copy = df.copy()
    df_copy["date"] = pd.to_datetime(df_copy["date"])
    df_copy = df_copy.sort_values(["asset", "date"]).reset_index(drop=True)

    close = df_copy["close"].where(df_copy["close"] > 0)
    by_asset = close.groupby(df_copy["asset"], sort=False)
    end_close = by_asset.shift(skip_recent)
    prior_close = by_asset.shift(window + skip_recent)

    value = end_close.div(prior_close).sub(1.0)

    valid_count = (
        close.notna()
        .groupby(df_copy["asset"], sort=False)
        .rolling(effective_window, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )
    value = value.where(valid_count >= min_periods)

    result = df_copy[["date", "asset"]].copy()
    factor_name = f"momentum_{window}d"
    if skip_recent > 0:
        factor_name = f"momentum_{window}d_skip{skip_recent}d"
    result["factor"] = factor_name
    result["value"] = value
    result = result.loc[:, FACTOR_OUTPUT_COLUMNS].reset_index(drop=True)

    return result
