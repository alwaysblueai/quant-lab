"""Shared intraday formula utilities.

These helpers are intentionally pandas-only and side-effect free. ETL scripts
own parquet IO, partitioning, and merge policy.
"""

from __future__ import annotations

from datetime import time

import numpy as np
import pandas as pd

EPS = 1e-12


def prepare_minute_day(day: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted copy with numeric OHLCV columns and datetime parsed."""

    required = {"datetime", "open", "high", "low", "close", "volume", "amount"}
    missing = required.difference(day.columns)
    if missing:
        raise ValueError(f"missing required minute columns: {sorted(missing)}")

    out = day.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out = out.sort_values("datetime", kind="mergesort").reset_index(drop=True)
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def safe_ratio(numerator: float, denominator: float, *, eps: float = EPS) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= eps:
        return np.nan
    return float(numerator / denominator)


def safe_return(numerator: float, denominator: float, *, eps: float = EPS) -> float:
    ratio = safe_ratio(numerator, denominator, eps=eps)
    if np.isnan(ratio):
        return np.nan
    return ratio - 1.0


def time_values(frame: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(frame["datetime"]).dt.time


def first_value_at_or_after(
    frame: pd.DataFrame,
    column: str,
    boundary: time,
) -> float:
    mask = time_values(frame) >= boundary
    if not mask.any():
        return np.nan
    value = frame.loc[mask, column].iloc[0]
    return float(value) if pd.notna(value) else np.nan


def last_value_at_or_before(
    frame: pd.DataFrame,
    column: str,
    boundary: time,
) -> float:
    mask = time_values(frame) <= boundary
    if not mask.any():
        return np.nan
    value = frame.loc[mask, column].iloc[-1]
    return float(value) if pd.notna(value) else np.nan


def log_returns(closes: pd.Series) -> pd.Series:
    close = pd.to_numeric(closes, errors="coerce")
    valid = close.where(close > 0)
    return np.log(valid / valid.shift(1)).replace([np.inf, -np.inf], np.nan)


def segment_log_returns(day: pd.DataFrame, *, start: time | None, end: time | None) -> pd.Series:
    times = time_values(day)
    mask = pd.Series(True, index=day.index)
    if start is not None:
        mask &= times >= start
    if end is not None:
        mask &= times <= end
    segment = day.loc[mask, "close"]
    return log_returns(segment).dropna()


def sampled_log_returns(day: pd.DataFrame, step: int) -> pd.Series:
    """Non-overlapping close-to-close log returns sampled by traded-bar index."""

    if step <= 0:
        raise ValueError("step must be positive")
    close = pd.to_numeric(day["close"], errors="coerce")
    if close.empty:
        return pd.Series(dtype="float64")
    sampled = close.iloc[::step]
    if sampled.empty or sampled.index[-1] != close.index[-1]:
        sampled = pd.concat([sampled, close.iloc[[-1]]])
    return log_returns(sampled).dropna().reset_index(drop=True)


def realized_variance(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return np.nan
    return float(np.square(values).sum())


def bipower_variation(returns: pd.Series) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna().abs()
    if len(values) < 2:
        return np.nan
    return float((np.pi / 2.0) * (values * values.shift(1)).dropna().sum())


def signed_realized_variance(returns: pd.Series, *, positive: bool) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if values.empty:
        return np.nan
    selected = values[values > 0] if positive else values[values < 0]
    return float(np.square(selected).sum()) if not selected.empty else 0.0


def skew_with_min_nonzero(returns: pd.Series, *, min_nonzero: int) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if (values.abs() > EPS).sum() < min_nonzero:
        return np.nan
    return float(values.skew())


def kurt_with_min_nonzero(returns: pd.Series, *, min_nonzero: int) -> float:
    values = pd.to_numeric(returns, errors="coerce").dropna()
    if (values.abs() > EPS).sum() < min_nonzero:
        return np.nan
    return float(values.kurt())


def safe_corr(left: pd.Series, right: pd.Series, *, min_count: int) -> float:
    """Pearson correlation guarded against degenerate input."""

    a = pd.to_numeric(left, errors="coerce")
    b = pd.to_numeric(right, errors="coerce")
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < min_count:
        return np.nan
    if aligned.iloc[:, 0].std(ddof=0) <= EPS or aligned.iloc[:, 1].std(ddof=0) <= EPS:
        return np.nan
    value = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
    return float(value) if pd.notna(value) else np.nan


def safe_autocorr_lag1(values: pd.Series, *, min_count: int) -> float:
    """Lag-1 Pearson autocorrelation."""

    series = pd.to_numeric(values, errors="coerce").dropna().reset_index(drop=True)
    if len(series) < min_count + 1:
        return np.nan
    lagged = pd.concat({"x": series.iloc[1:].reset_index(drop=True),
                        "lag1": series.iloc[:-1].reset_index(drop=True)}, axis=1).dropna()
    if len(lagged) < min_count:
        return np.nan
    if lagged["x"].std(ddof=0) <= EPS or lagged["lag1"].std(ddof=0) <= EPS:
        return np.nan
    value = lagged["x"].corr(lagged["lag1"])
    return float(value) if pd.notna(value) else np.nan
