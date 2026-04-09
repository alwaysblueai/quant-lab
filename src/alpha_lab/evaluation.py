from __future__ import annotations

import pandas as pd

from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS, validate_factor_output


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

    validate_factor_output(factors)
    validate_factor_output(labels)

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

    values = _vectorized_cross_sectional_corr(merged, method=method, value_name=value_name)
    values["factor"] = factor_name
    values["label"] = label_name
    return values.loc[:, ["date", "factor", "label", value_name]]


def _vectorized_cross_sectional_corr(
    merged: pd.DataFrame,
    *,
    method: str,
    value_name: str,
) -> pd.DataFrame:
    """Compute per-date cross-sectional correlation without per-group Python calls.

    Uses pivot + numpy vectorised row-wise correlation for the Pearson case and
    falls back to scipy for Spearman (which needs ranking first).
    """
    import numpy as np

    # Drop rows where either value is NaN before pivoting.
    clean = merged[["date", "asset", "value_factor", "value_label"]].dropna(
        subset=["value_factor", "value_label"]
    )
    if clean.empty:
        return pd.DataFrame(columns=["date", value_name])

    # Count valid assets per date and check variance.
    date_groups = clean.groupby("date", sort=True)
    counts = date_groups.size()
    factor_nunique = date_groups["value_factor"].nunique()
    label_nunique = date_groups["value_label"].nunique()

    # Dates that can produce a valid correlation.
    valid_dates = counts.index[(counts >= 2) & (factor_nunique >= 2) & (label_nunique >= 2)]

    if len(valid_dates) == 0:
        all_dates = counts.index
        return pd.DataFrame({"date": all_dates, value_name: np.nan})

    clean_valid = clean[clean["date"].isin(valid_dates)]

    if method == "spearman":
        # Convert to ranks within each date for Spearman.
        clean_valid = clean_valid.copy()
        clean_valid["value_factor"] = clean_valid.groupby("date")["value_factor"].rank(
            method="average"
        )
        clean_valid["value_label"] = clean_valid.groupby("date")["value_label"].rank(
            method="average"
        )

    # Demean within each date (vectorised).
    clean_valid = clean_valid.copy()
    factor_mean = clean_valid.groupby("date")["value_factor"].transform("mean")
    label_mean = clean_valid.groupby("date")["value_label"].transform("mean")
    clean_valid["f_dm"] = clean_valid["value_factor"] - factor_mean
    clean_valid["l_dm"] = clean_valid["value_label"] - label_mean

    # Sum of products and sum of squares per date.
    clean_valid["fl"] = clean_valid["f_dm"] * clean_valid["l_dm"]
    clean_valid["ff"] = clean_valid["f_dm"] ** 2
    clean_valid["ll"] = clean_valid["l_dm"] ** 2

    agg = clean_valid.groupby("date", sort=True)[["fl", "ff", "ll"]].sum()
    denom = np.sqrt(agg["ff"].to_numpy() * agg["ll"].to_numpy())
    corr = np.where(denom > 0, agg["fl"].to_numpy() / denom, np.nan)

    result = pd.DataFrame({"date": agg.index, value_name: corr})

    # Add NaN rows for dates that were filtered out (< 2 assets or constant).
    invalid_dates = counts.index.difference(valid_dates)
    if len(invalid_dates) > 0:
        nan_rows = pd.DataFrame({"date": invalid_dates, value_name: np.nan})
        result = pd.concat([result, nan_rows], ignore_index=True).sort_values("date")

    return result.reset_index(drop=True)


def _extract_single_factor_name(df: pd.DataFrame, *, table_name: str) -> str:
    factor_names = pd.unique(df["factor"])
    if len(factor_names) != 1:
        raise AlphaLabDataError(f"{table_name} must contain exactly one factor name")
    return str(factor_names[0])
