from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError
from alpha_lab.preprocess import winsorize_series, zscore_series
from alpha_lab.research_integrity.leakage_checks import check_cross_section_transform_scope

_REQUIRED_COLUMNS: frozenset[str] = frozenset({"date", "asset", "value"})


def winsorize_cross_section(
    df: pd.DataFrame,
    *,
    lower: float = 0.01,
    upper: float = 0.99,
    min_group_size: int = 3,
    enforce_integrity: bool = True,
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

    out = _coerce_sort(out)
    return _validate_transform_scope(
        source_df=df,
        transformed_df=out,
        transform_name="winsorize_cross_section",
        enforce_integrity=enforce_integrity,
    )


def zscore_cross_section(
    df: pd.DataFrame,
    *,
    min_group_size: int = 3,
    enforce_integrity: bool = True,
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

    out = _coerce_sort(out)
    return _validate_transform_scope(
        source_df=df,
        transformed_df=out,
        transform_name="zscore_cross_section",
        enforce_integrity=enforce_integrity,
    )


def rank_cross_section(
    df: pd.DataFrame,
    *,
    min_group_size: int = 3,
    pct: bool = True,
    enforce_integrity: bool = True,
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

    out = _coerce_sort(out)
    return _validate_transform_scope(
        source_df=df,
        transformed_df=out,
        transform_name="rank_cross_section",
        enforce_integrity=enforce_integrity,
    )


def apply_min_coverage_gate(
    df: pd.DataFrame,
    *,
    min_coverage: float,
    enforce_integrity: bool = True,
) -> pd.DataFrame:
    """Set `value` to NaN on dates whose non-null coverage is below threshold."""

    _validate_signal_frame(df)
    if min_coverage <= 0 or min_coverage > 1:
        raise AlphaLabConfigError("min_coverage must be in (0, 1]")

    out = df.copy()
    grouped = out.groupby("date", sort=False)["value"]
    coverage = grouped.transform("count") / grouped.transform("size")
    out.loc[coverage < min_coverage, "value"] = np.nan
    out = _coerce_sort(out)
    return _validate_transform_scope(
        source_df=df,
        transformed_df=out,
        transform_name="apply_min_coverage_gate",
        enforce_integrity=enforce_integrity,
    )


def _date_groups(df: pd.DataFrame) -> list[pd.Index]:
    return [pd.Index(idx) for _, idx in df.groupby("date", sort=False).groups.items()]


def _validate_signal_frame(df: pd.DataFrame) -> None:
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise AlphaLabDataError(f"signal transform input missing columns: {sorted(missing)}")


def _coerce_sort(df: pd.DataFrame) -> pd.DataFrame:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def _validate_transform_scope(
    *,
    source_df: pd.DataFrame,
    transformed_df: pd.DataFrame,
    transform_name: str,
    enforce_integrity: bool,
) -> pd.DataFrame:
    if not enforce_integrity:
        return transformed_df

    check = check_cross_section_transform_scope(
        source_df,
        transformed_df,
        date_col="date",
        asset_col="asset",
        object_name=transform_name,
    )
    transformed_df.attrs["integrity_scope_check"] = check.to_dict()

    if check.status == "fail":
        raise AlphaLabDataError(f"{transform_name} failed integrity scope check: {check.message}")
    if check.status == "warn":
        warnings.warn(
            f"{transform_name} integrity warning: {check.message}",
            UserWarning,
            stacklevel=3,
        )
    return transformed_df
