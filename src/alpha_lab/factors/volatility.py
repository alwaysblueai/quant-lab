from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError
from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS


def amplitude(
    df: pd.DataFrame,
    *,
    window: int = 20,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute a low-amplitude factor from daily high-low ranges.

    The factor at date ``t`` is the negative rolling mean of daily amplitude:

    ``-mean((high[t] - low[t]) / close[t-1])``

    Higher values therefore indicate lower recent intraday amplitude.
    """
    if df.empty:
        return pd.DataFrame(columns=FACTOR_OUTPUT_COLUMNS)

    _validate_common_inputs(df, required_cols={"date", "asset", "close", "high", "low"})
    min_periods = _resolve_min_periods(window=window, min_periods=min_periods)

    frame = _sorted_copy(df)
    prev_close = frame.groupby("asset", sort=False)["close"].shift(1)
    daily_amplitude = (frame["high"] - frame["low"]) / prev_close.where(prev_close > 0)
    rolling_mean = (
        daily_amplitude.groupby(frame["asset"], sort=False)
        .rolling(window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )

    result = frame[["date", "asset"]].copy()
    result["factor"] = f"amplitude_{window}d"
    result["value"] = -rolling_mean
    return result.loc[:, FACTOR_OUTPUT_COLUMNS].reset_index(drop=True)


def downside_volatility(
    df: pd.DataFrame,
    *,
    window: int = 20,
    min_periods: int | None = None,
) -> pd.DataFrame:
    """Compute a downside-volatility factor from close-to-close returns.

    The factor at date ``t`` is the negative rolling downside volatility:

    ``-sqrt(sum(r_t^2 for r_t < 0) / count(r_t < 0))``

    Windows with no negative returns remain missing rather than being forced
    to zero.
    """
    if df.empty:
        return pd.DataFrame(columns=FACTOR_OUTPUT_COLUMNS)

    _validate_common_inputs(df, required_cols={"date", "asset", "close"})
    min_periods = _resolve_min_periods(window=window, min_periods=min_periods)

    frame = _sorted_copy(df)
    simple_return = frame.groupby("asset", sort=False)["close"].pct_change(fill_method=None)
    neg_sq = simple_return.where(simple_return < 0).pow(2).fillna(0.0)
    neg_count = simple_return.where(simple_return < 0).notna().astype(float)

    grouped_sq = neg_sq.groupby(frame["asset"], sort=False)
    grouped_count = neg_count.groupby(frame["asset"], sort=False)
    rolling_sq_sum = (
        grouped_sq.rolling(window, min_periods=min_periods).sum().reset_index(level=0, drop=True)
    )
    rolling_neg_count = (
        grouped_count.rolling(window, min_periods=min_periods).sum().reset_index(level=0, drop=True)
    )

    downside = np.sqrt(rolling_sq_sum / rolling_neg_count.where(rolling_neg_count > 0))

    result = frame[["date", "asset"]].copy()
    result["factor"] = f"downside_volatility_{window}d"
    result["value"] = -downside
    return result.loc[:, FACTOR_OUTPUT_COLUMNS].reset_index(drop=True)


def _validate_common_inputs(df: pd.DataFrame, *, required_cols: set[str]) -> None:
    missing = required_cols - set(df.columns)
    if missing:
        raise AlphaLabDataError(f"Input DataFrame is missing required columns: {missing}")

    if df["date"].isna().any():
        raise AlphaLabDataError("Input 'date' column contains NaN/NaT values.")

    if df["asset"].isna().any():
        raise AlphaLabDataError("Input 'asset' column contains NaN values.")

    dupes = df.duplicated(subset=["date", "asset"])
    if dupes.any():
        raise AlphaLabDataError(f"Duplicate (date, asset) pairs found:\n{df[dupes][['date', 'asset']]}")


def _resolve_min_periods(*, window: int, min_periods: int | None) -> int:
    if window <= 0:
        raise AlphaLabConfigError("'window' must be a positive integer")
    if min_periods is None:
        return window
    if min_periods <= 0:
        raise AlphaLabConfigError("'min_periods' must be a positive integer")
    if min_periods > window:
        raise AlphaLabConfigError("'min_periods' cannot exceed window")
    return min_periods


def _sorted_copy(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.sort_values(["asset", "date"]).reset_index(drop=True)
