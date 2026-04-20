from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.interfaces import validate_factor_output


def fracdiff_weights(d: float, threshold: float = 1e-5) -> np.ndarray:
    """Compute truncated fractional differencing weights."""
    if threshold <= 0:
        raise ValueError("threshold must be > 0")

    weights = [1.0]
    k = 1
    while k < 10000:
        wk = -weights[-1] * (float(d) - k + 1.0) / float(k)
        if abs(wk) < threshold:
            break
        weights.append(float(wk))
        k += 1
    return np.asarray(weights, dtype=float)


def fracdiff_series(
    series: pd.Series,
    d: float = 0.4,
    threshold: float = 1e-5,
) -> pd.Series:
    """Apply fixed-window fractional differencing to a time series."""
    values = pd.to_numeric(series, errors="coerce")
    if values.empty:
        return pd.Series(dtype=float, index=series.index, name=series.name)
    if float(d) == 0.0:
        return values.astype(float)

    weights = fracdiff_weights(d=float(d), threshold=threshold)
    arr = values.to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    width = len(weights)
    if width == 0:
        return pd.Series(out, index=series.index, name=series.name)

    for i in range(width - 1, len(arr)):
        window = arr[i - width + 1 : i + 1]
        if np.isnan(window).any():
            continue
        out[i] = float(np.dot(weights, window[::-1]))
    return pd.Series(out, index=series.index, name=series.name)


def find_min_d(
    series: pd.Series,
    target_adf_pvalue: float = 0.05,
    d_range: tuple[float, float] = (0.0, 1.0),
    steps: int = 11,
) -> float:
    """Find the smallest fractional-differencing order passing the ADF test."""
    if not (0.0 < target_adf_pvalue < 1.0):
        raise ValueError("target_adf_pvalue must be in (0, 1)")
    if steps < 2:
        raise ValueError("steps must be >= 2")
    d_min, d_max = d_range
    if d_max < d_min:
        raise ValueError("d_range must satisfy min <= max")

    candidates = np.linspace(float(d_min), float(d_max), int(steps))
    for d in candidates:
        transformed = fracdiff_series(series, d=float(d)).dropna()
        if len(transformed) < 10:
            continue
        if transformed.nunique(dropna=True) < 2:
            continue
        try:
            _, p_value, *_ = adfuller(transformed.to_numpy(dtype=float), autolag="AIC")
        except Exception:
            continue
        if np.isfinite(p_value) and float(p_value) <= float(target_adf_pvalue):
            return float(d)
    return float(d_max)


def fracdiff_cross_section(
    df: pd.DataFrame,
    d: float = 0.4,
    threshold: float = 1e-5,
) -> pd.DataFrame:
    """Apply fractional differencing per ``(asset, factor)`` series."""
    validate_factor_output(df)

    work = df[["date", "asset", "factor", "value"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    if work["date"].isna().any():
        raise AlphaLabDataError("df contains invalid date values")

    chunks: list[pd.DataFrame] = []
    for _, group in work.groupby(["asset", "factor"], sort=False):
        block = group.sort_values("date", kind="mergesort").copy()
        block["value"] = fracdiff_series(block["value"], d=float(d), threshold=threshold)
        chunks.append(block)

    if not chunks:
        return pd.DataFrame(columns=["date", "asset", "factor", "value"])
    out = pd.concat(chunks, ignore_index=True)
    return out.sort_values(["date", "asset", "factor"], kind="mergesort").reset_index(drop=True)
