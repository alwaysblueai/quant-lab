from __future__ import annotations

import pandas as pd

from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS


def compute_ic(factors: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional Pearson IC by date.

    Factor values and labels are aligned only on ``(date, asset)``. Each input
    must contain exactly one factor name so the output is unambiguous. The
    returned IC for date ``t`` measures the cross-sectional association between
    features observed at ``t`` and labels stored at ``t``.
    """
    return _compute_cross_sectional_metric(
        factors,
        labels,
        method="pearson",
        value_name="ic",
    )


def compute_rank_ic(factors: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-sectional Spearman RankIC by date."""
    return _compute_cross_sectional_metric(
        factors,
        labels,
        method="spearman",
        value_name="rank_ic",
    )


def _compute_cross_sectional_metric(
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    method: str,
    value_name: str,
) -> pd.DataFrame:
    if factors.empty or labels.empty:
        return pd.DataFrame(columns=["date", "factor", "label", value_name])

    _validate_canonical_table(factors, table_name="factors")
    _validate_canonical_table(labels, table_name="labels")

    factor_name = _extract_single_factor_name(factors, table_name="factors")
    label_name = _extract_single_factor_name(labels, table_name="labels")

    merged = factors.merge(
        labels,
        on=["date", "asset"],
        how="inner",
        suffixes=("_factor", "_label"),
        validate="one_to_one",
    )

    if merged.empty:
        return pd.DataFrame(columns=["date", "factor", "label", value_name])

    values = (
        merged.groupby("date", sort=True, group_keys=False)
        .apply(lambda frame: _correlation_or_nan(frame, method=method), include_groups=False)
        .rename(value_name)
        .reset_index()
    )
    values["factor"] = factor_name
    values["label"] = label_name
    return values.loc[:, ["date", "factor", "label", value_name]]


def _validate_canonical_table(df: pd.DataFrame, *, table_name: str) -> None:
    missing = set(FACTOR_OUTPUT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")

    if df["date"].isna().any():
        raise ValueError(f"{table_name} contains NaN/NaT in 'date'")

    if df["asset"].isna().any():
        raise ValueError(f"{table_name} contains NaN in 'asset'")

    if df["factor"].isna().any():
        raise ValueError(f"{table_name} contains NaN in 'factor'")

    dupes = df.duplicated(subset=["date", "asset", "factor"])
    if dupes.any():
        raise ValueError(f"{table_name} contains duplicate (date, asset, factor) rows")


def _extract_single_factor_name(df: pd.DataFrame, *, table_name: str) -> str:
    factor_names = pd.unique(df["factor"])
    if len(factor_names) != 1:
        raise ValueError(f"{table_name} must contain exactly one factor name")
    return str(factor_names[0])


def _correlation_or_nan(frame: pd.DataFrame, *, method: str) -> float:
    subset = frame[["value_factor", "value_label"]].dropna()
    if len(subset) < 2:
        return float("nan")

    if subset["value_factor"].nunique() < 2 or subset["value_label"].nunique() < 2:
        return float("nan")

    corr = subset["value_factor"].corr(subset["value_label"], method=method)
    return float("nan") if pd.isna(corr) else float(corr)
