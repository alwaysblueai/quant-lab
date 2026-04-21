from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError
from alpha_lab.preprocess import winsorize_series
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
    """Winsorize ``value`` per date cross-section.

    When a date has fewer than ``min_group_size`` valid rows the group is
    **skipped** and raw values flow through untouched. This is deliberately
    different from :func:`zscore_cross_section` / :func:`rank_cross_section`,
    which NaN-out such groups: clipping is defined for any sample size ≥1,
    whereas standardization/ranking require a meaningful cross-sectional
    distribution.
    """

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


def mad_winsorize_cross_section(
    df: pd.DataFrame,
    *,
    k: float = 3.0,
    min_group_size: int = 3,
    enforce_integrity: bool = True,
) -> pd.DataFrame:
    """MAD winsorize ``value`` per date cross-section."""
    if k <= 0:
        raise AlphaLabConfigError("k must be > 0")

    _validate_signal_frame(df)
    out = df.copy()

    for idx in _date_groups(out):
        values = pd.to_numeric(out.loc[idx, "value"], errors="coerce")
        valid = values.notna()
        if int(valid.sum()) < min_group_size:
            continue

        v = values.loc[valid]
        median = float(v.median())
        mad = float((v - median).abs().median())
        if not np.isfinite(mad) or mad <= 0.0:
            continue
        lower = median - float(k) * mad
        upper = median + float(k) * mad
        out.loc[v.index, "value"] = v.clip(lower=lower, upper=upper)

    out = _coerce_sort(out)
    return _validate_transform_scope(
        source_df=df,
        transformed_df=out,
        transform_name="mad_winsorize_cross_section",
        enforce_integrity=enforce_integrity,
    )


def zscore_cross_section(
    df: pd.DataFrame,
    *,
    min_group_size: int = 3,
    enforce_integrity: bool = True,
) -> pd.DataFrame:
    """Z-score standardize ``value`` per date cross-section.

    Uses **population standard deviation** (``ddof=0``) via
    :func:`~alpha_lab.preprocess.zscore_series`.  This is intentional:
    the goal is within-date normalization, not estimating a population
    parameter.  See ``zscore_series`` docstring for the full rationale
    and contrast with ``compute_ic_summary`` (which uses ``ddof=1``).

    Dates with fewer than ``min_group_size`` valid rows have the entire group
    NaN-ed: a z-score computed from a handful of observations is not
    informative and should not flow downstream. Contrast with
    :func:`winsorize_cross_section`, which skips small groups without
    nulling them.
    """

    _validate_signal_frame(df)
    out = df.copy()
    values = pd.to_numeric(out["value"], errors="coerce")
    out["value"] = values

    grouped = out.groupby("date", sort=False)["value"]
    valid_count = grouped.transform("count")
    mean = grouped.transform("mean")
    # pandas transform("std") uses sample std (ddof=1); convert to ddof=0
    # to match zscore_series semantics.
    std_sample = grouped.transform("std")
    scale = pd.Series(np.nan, index=out.index, dtype=float)
    positive_count = valid_count > 0
    scale.loc[positive_count] = np.sqrt(
        (valid_count.loc[positive_count] - 1.0) / valid_count.loc[positive_count]
    )
    std_pop = std_sample * scale

    small_group = valid_count < min_group_size
    valid_row = values.notna() & (~small_group)
    std_zero_or_nan = (~np.isfinite(std_pop)) | std_pop.eq(0.0)

    out.loc[:, "value"] = np.nan
    out.loc[valid_row & std_zero_or_nan, "value"] = 0.0
    z_mask = valid_row & (~std_zero_or_nan)
    out.loc[z_mask, "value"] = (values.loc[z_mask] - mean.loc[z_mask]) / std_pop.loc[z_mask]

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
    """Cross-sectional rank transform for ``value`` per date.

    Dates with fewer than ``min_group_size`` valid rows have the entire group
    NaN-ed (ranks from a degenerate sample are not meaningful). Ties are
    resolved with pandas' ``method="average"``.
    """

    _validate_signal_frame(df)
    out = df.copy()
    out["value"] = pd.to_numeric(out["value"], errors="coerce")

    grouped = out.groupby("date", sort=False)["value"]
    valid_count = grouped.transform("count")
    ranked = grouped.rank(
        method="average",
        ascending=True,
        pct=pct,
    )
    out["value"] = ranked
    out.loc[valid_count < min_group_size, "value"] = np.nan

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
