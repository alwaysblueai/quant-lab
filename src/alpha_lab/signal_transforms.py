from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.preprocess import winsorize_series, zscore_series

_REQUIRED_COLUMNS: frozenset[str] = frozenset({"date", "asset", "value"})


def winsorize_cross_section(
    df: pd.DataFrame,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
    min_group_size: int = 3,
) -> pd.DataFrame:
    """Winsorize `value` per date cross-section."""

    _validate_signal_frame(df)
    out = df.copy()

    for idx in _date_groups(out):
        values = pd.to_numeric(out.loc[idx, "value"], errors="coerce")
        valid = values.notna()
        if int(valid.sum()) < min_group_size:
            continue
        out.loc[values.index[valid], "value"] = winsorize_series(
            values.loc[valid],
            lower=lower,
            upper=upper,
        )

    return _coerce_sort(out)


def zscore_cross_section(
    df: pd.DataFrame,
    *,
    min_group_size: int = 3,
) -> pd.DataFrame:
    """Z-score standardize `value` per date cross-section."""

    _validate_signal_frame(df)
    out = df.copy()

    for idx in _date_groups(out):
        values = pd.to_numeric(out.loc[idx, "value"], errors="coerce")
        valid = values.notna()
        if int(valid.sum()) < min_group_size:
            out.loc[idx, "value"] = np.nan
            continue
        out.loc[idx, "value"] = np.nan
        out.loc[values.index[valid], "value"] = zscore_series(values.loc[valid])

    return _coerce_sort(out)


def rank_cross_section(
    df: pd.DataFrame,
    *,
    min_group_size: int = 3,
    pct: bool = True,
) -> pd.DataFrame:
    """Cross-sectional rank transform for `value` per date."""

    _validate_signal_frame(df)
    out = df.copy()

    for idx in _date_groups(out):
        values = pd.to_numeric(out.loc[idx, "value"], errors="coerce")
        valid = values.notna()
        if int(valid.sum()) < min_group_size:
            out.loc[idx, "value"] = np.nan
            continue
        out.loc[idx, "value"] = np.nan
        out.loc[values.index[valid], "value"] = values.loc[valid].rank(
            method="average",
            ascending=True,
            pct=pct,
        )

    return _coerce_sort(out)


def apply_min_coverage_gate(
    df: pd.DataFrame,
    *,
    min_coverage: float,
) -> pd.DataFrame:
    """Set `value` to NaN on dates whose non-null coverage is below threshold."""

    _validate_signal_frame(df)
    if min_coverage <= 0 or min_coverage > 1:
        raise ValueError("min_coverage must be in (0, 1]")

    out = df.copy()
    coverage = (
        out.groupby("date", sort=False)["value"]
        .apply(lambda s: float(s.notna().mean()) if len(s) > 0 else 0.0)
        .rename("coverage")
    )
    keep_dates = set(coverage[coverage >= min_coverage].index)
    out.loc[~out["date"].isin(keep_dates), "value"] = np.nan
    return _coerce_sort(out)


def _date_groups(df: pd.DataFrame) -> list[pd.Index]:
    return [pd.Index(idx) for _, idx in df.groupby("date", sort=False).groups.items()]


def _validate_signal_frame(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"signal transform input missing columns: {sorted(missing)}")


def _coerce_sort(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
