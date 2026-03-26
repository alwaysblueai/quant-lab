from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_cross_section(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    by: str = "date",
    lower: float = 0.01,
    upper: float = 0.99,
    min_group_size: int = 3,
) -> pd.DataFrame:
    """Winsorize values within each cross-sectional group."""
    _validate_transform_inputs(df, value_col=value_col, by=by, min_group_size=min_group_size)
    out = df.copy()

    def _clip(vals: pd.Series) -> pd.Series:
        n = int(vals.notna().sum())
        if n < min_group_size:
            return vals
        lo = vals.quantile(lower)
        hi = vals.quantile(upper)
        return vals.clip(lower=lo, upper=hi)

    out[value_col] = out.groupby(by, sort=True)[value_col].transform(_clip)
    return out


def zscore_cross_section(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    by: str = "date",
    min_group_size: int = 3,
) -> pd.DataFrame:
    """Compute cross-sectional z-scores with small-group safeguards."""
    _validate_transform_inputs(df, value_col=value_col, by=by, min_group_size=min_group_size)
    out = df.copy()

    def _z(vals: pd.Series) -> pd.Series:
        n = int(vals.notna().sum())
        if n < min_group_size:
            return pd.Series(np.nan, index=vals.index, dtype=float)
        mu = vals.mean()
        std = vals.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=vals.index, dtype=float)
        return (vals - mu) / std

    out[value_col] = out.groupby(by, sort=True)[value_col].transform(_z)
    return out


def rank_cross_section(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    by: str = "date",
    pct: bool = True,
    min_group_size: int = 2,
) -> pd.DataFrame:
    """Compute cross-sectional ranks per group."""
    _validate_transform_inputs(df, value_col=value_col, by=by, min_group_size=min_group_size)
    out = df.copy()

    def _rank(vals: pd.Series) -> pd.Series:
        n = int(vals.notna().sum())
        if n < min_group_size:
            return pd.Series(np.nan, index=vals.index, dtype=float)
        return vals.rank(method="average", ascending=True, na_option="keep", pct=pct)

    out[value_col] = out.groupby(by, sort=True)[value_col].transform(_rank)
    return out


def neutralize_by_group(
    df: pd.DataFrame,
    *,
    group_col: str,
    value_col: str = "value",
    by: str = "date",
    min_group_size: int = 3,
) -> pd.DataFrame:
    """Group-demean values within each (date, group) with size safeguards."""
    _validate_transform_inputs(df, value_col=value_col, by=by, min_group_size=min_group_size)
    if group_col not in df.columns:
        raise ValueError(f"df is missing group_col {group_col!r}")
    out = df.copy()

    keys = [by, group_col]

    def _demean(vals: pd.Series) -> pd.Series:
        n = int(vals.notna().sum())
        if n < min_group_size:
            return pd.Series(np.nan, index=vals.index, dtype=float)
        return vals - vals.mean()

    out[value_col] = out.groupby(keys, sort=True)[value_col].transform(_demean)
    return out


def apply_min_coverage_gate(
    df: pd.DataFrame,
    *,
    value_col: str = "value",
    by: str = "date",
    min_coverage: float = 0.3,
) -> pd.DataFrame:
    """Drop groups whose non-null coverage is below min_coverage."""
    if min_coverage <= 0 or min_coverage > 1:
        raise ValueError("min_coverage must be in (0, 1]")
    _validate_transform_inputs(df, value_col=value_col, by=by, min_group_size=1)
    out = df.copy()

    n_total = out.groupby(by, sort=True)[value_col].size()
    n_valid = out.groupby(by, sort=True)[value_col].apply(lambda s: int(s.notna().sum()))
    coverage = (n_valid / n_total).rename("_coverage")
    keep_dates = set(coverage[coverage >= min_coverage].index)
    return out[out[by].isin(keep_dates)].reset_index(drop=True)


def _validate_transform_inputs(
    df: pd.DataFrame,
    *,
    value_col: str,
    by: str,
    min_group_size: int,
) -> None:
    if value_col not in df.columns:
        raise ValueError(f"df is missing value_col {value_col!r}")
    if by not in df.columns:
        raise ValueError(f"df is missing by column {by!r}")
    if min_group_size <= 0:
        raise ValueError("min_group_size must be >= 1")
    if not pd.api.types.is_numeric_dtype(df[value_col]):
        raise ValueError(f"{value_col!r} must be numeric")
