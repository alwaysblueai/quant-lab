from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.interfaces import validate_factor_output
from alpha_lab.numba_kernels import (
    cross_sectional_corr_by_group_numba,
    numba_enabled,
)


def compute_mutual_information(
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    max_bins: int = 10,
    merged_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute per-date cross-sectional mutual information by quantile bins.

    Mutual information (MI) captures generic dependence, including nonlinear
    relationships that Pearson IC may miss.  It is estimated on each date by
    discretizing factor/label cross-sections into rank-quantile buckets and
    computing discrete MI in nats.
    """
    if max_bins < 2:
        raise ValueError("max_bins must be >= 2")

    if factors.empty or labels.empty:
        return pd.DataFrame(columns=["date", "factor", "label", "mutual_information"])

    validate_factor_output(factors)
    validate_factor_output(labels)

    factor_name = _extract_single_factor_name(factors, table_name="factors")
    label_name = _extract_single_factor_name(labels, table_name="labels")

    merged = _resolve_merged_pairs(
        factors=factors,
        labels=labels,
        merged_pairs=merged_pairs,
    )
    if merged.empty:
        return pd.DataFrame(columns=["date", "factor", "label", "mutual_information"])

    values = _cross_sectional_mutual_information(
        merged,
        value_name="mutual_information",
        max_bins=max_bins,
    )
    values["factor"] = factor_name
    values["label"] = label_name
    return values.loc[:, ["date", "factor", "label", "mutual_information"]]


def compute_ic(
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    merged_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
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
        merged_pairs=merged_pairs,
    )


def compute_rank_ic(
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    merged_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute cross-sectional Spearman RankIC by date."""
    return _compute_cross_sectional_metric(
        factors,
        labels,
        method="spearman",
        value_name="rank_ic",
        merged_pairs=merged_pairs,
    )


def _compute_cross_sectional_metric(
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    method: str,
    value_name: str,
    merged_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if factors.empty or labels.empty:
        return pd.DataFrame(columns=["date", "factor", "label", value_name])

    validate_factor_output(factors)
    validate_factor_output(labels)

    factor_name = _extract_single_factor_name(factors, table_name="factors")
    label_name = _extract_single_factor_name(labels, table_name="labels")

    merged = _resolve_merged_pairs(
        factors=factors,
        labels=labels,
        merged_pairs=merged_pairs,
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

    all_dates = pd.Index(pd.to_datetime(merged["date"], errors="coerce")).dropna().unique()
    all_dates = all_dates.sort_values()

    # Try numba kernel first when available; otherwise use pandas fallback.
    numba_result = _cross_sectional_corr_numba_path(
        merged,
        method=method,
        value_name=value_name,
        all_dates=all_dates,
    )
    if numba_result is not None:
        return numba_result

    # Drop rows where either value is NaN before pivoting.
    clean = merged[["date", "asset", "value_factor", "value_label"]].dropna(
        subset=["value_factor", "value_label"]
    )
    if clean.empty:
        return pd.DataFrame({"date": all_dates, value_name: np.nan})

    # Count valid assets per date and check variance.
    date_groups = clean.groupby("date", sort=True)
    counts = date_groups.size()
    factor_nunique = date_groups["value_factor"].nunique()
    label_nunique = date_groups["value_label"].nunique()

    # Dates that can produce a valid correlation.
    valid_dates = counts.index[(counts >= 2) & (factor_nunique >= 2) & (label_nunique >= 2)]

    if len(valid_dates) == 0:
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

    # Add NaN rows for dates that were filtered out (< 2 assets, constant),
    # plus dates with no clean factor/label pairs after NaN filtering.
    invalid_dates = all_dates.difference(valid_dates)
    if len(invalid_dates) > 0:
        nan_rows = pd.DataFrame({"date": invalid_dates, value_name: np.nan})
        result = pd.concat([result, nan_rows], ignore_index=True).sort_values("date")

    return result.reset_index(drop=True)


def _cross_sectional_corr_numba_path(
    merged: pd.DataFrame,
    *,
    method: str,
    value_name: str,
    all_dates: pd.Index,
) -> pd.DataFrame | None:
    if not numba_enabled():
        return None
    if method not in {"pearson", "spearman"}:
        return None

    values = merged[["date", "value_factor", "value_label"]].copy()
    values["date"] = pd.to_datetime(values["date"], errors="coerce")
    values["value_factor"] = pd.to_numeric(values["value_factor"], errors="coerce")
    values["value_label"] = pd.to_numeric(values["value_label"], errors="coerce")
    values = values.dropna(subset=["date", "value_factor", "value_label"])
    if values.empty:
        return pd.DataFrame({"date": all_dates, value_name: np.nan})

    values = values.sort_values("date", kind="mergesort").reset_index(drop=True)
    date_values = values["date"].to_numpy()
    factor_values = values["value_factor"].to_numpy(dtype=float)
    label_values = values["value_label"].to_numpy(dtype=float)

    unique_dates, start_idx = np.unique(date_values, return_index=True)
    end_idx = np.append(start_idx[1:], len(date_values))
    corr_values = cross_sectional_corr_by_group_numba(
        factor_values,
        label_values,
        start_idx.astype(np.int64, copy=False),
        end_idx.astype(np.int64, copy=False),
        method=method,
    )
    if corr_values is None:
        return None

    result = pd.DataFrame(
        {
            "date": pd.to_datetime(unique_dates),
            value_name: corr_values,
        }
    )
    invalid_dates = all_dates.difference(result["date"])
    if len(invalid_dates) > 0:
        result = pd.concat(
            [
                result,
                pd.DataFrame({"date": invalid_dates, value_name: np.nan}),
            ],
            ignore_index=True,
        )
    return result.sort_values("date").reset_index(drop=True)


def _resolve_merged_pairs(
    *,
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    merged_pairs: pd.DataFrame | None,
) -> pd.DataFrame:
    if merged_pairs is not None:
        required = {"date", "asset", "value_factor", "value_label"}
        missing = required - set(merged_pairs.columns)
        if missing:
            raise AlphaLabDataError(f"merged_pairs is missing required columns: {sorted(missing)}")
        return merged_pairs[["date", "asset", "value_factor", "value_label"]].copy()

    return factors.merge(
        labels,
        on=["date", "asset"],
        how="inner",
        suffixes=("_factor", "_label"),
        validate="one_to_one",
    )


def _cross_sectional_mutual_information(
    merged: pd.DataFrame,
    *,
    value_name: str,
    max_bins: int,
) -> pd.DataFrame:
    all_dates = pd.Index(pd.to_datetime(merged["date"], errors="coerce")).dropna().unique()
    all_dates = all_dates.sort_values()
    out = pd.DataFrame({"date": all_dates, value_name: np.nan})
    if out.empty:
        return out

    clean = merged[["date", "value_factor", "value_label"]].dropna(
        subset=["value_factor", "value_label"]
    )
    if clean.empty:
        return out

    values_by_date: dict[pd.Timestamp, float] = {}
    values = clean[["date", "value_factor", "value_label"]].copy()
    values["date"] = pd.to_datetime(values["date"], errors="coerce")
    values["value_factor"] = pd.to_numeric(values["value_factor"], errors="coerce")
    values["value_label"] = pd.to_numeric(values["value_label"], errors="coerce")
    values = values.dropna(subset=["date", "value_factor", "value_label"])
    if values.empty:
        return out

    values = values.sort_values("date", kind="mergesort").reset_index(drop=True)
    date_values = values["date"].to_numpy()
    factor_values = values["value_factor"].to_numpy(dtype=float)
    label_values = values["value_label"].to_numpy(dtype=float)

    unique_dates, start_idx = np.unique(date_values, return_index=True)
    end_idx = np.append(start_idx[1:], len(date_values))
    for i in range(len(unique_dates)):
        begin = int(start_idx[i])
        end = int(end_idx[i])
        values_by_date[pd.Timestamp(unique_dates[i])] = _estimate_mutual_information(
            factor_values[begin:end],
            label_values[begin:end],
            max_bins=max_bins,
        )

    if values_by_date:
        values = out["date"].map(values_by_date)
        out[value_name] = pd.to_numeric(values, errors="coerce")
    return out


def _estimate_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    *,
    max_bins: int,
) -> float:
    if x.size != y.size or x.size < 3:
        return float("nan")
    if np.isnan(x).all() or np.isnan(y).all():
        return float("nan")
    x_unique = pd.unique(x[~np.isnan(x)])
    y_unique = pd.unique(y[~np.isnan(y)])
    if len(x_unique) < 2 or len(y_unique) < 2:
        return float("nan")

    n_bins = int(min(max_bins, max(2, int(np.sqrt(x.size))), len(x_unique), len(y_unique)))
    if n_bins < 2:
        return float("nan")

    x_bins = _rank_quantile_bins(x, n_bins=n_bins)
    y_bins = _rank_quantile_bins(y, n_bins=n_bins)
    if x_bins is None or y_bins is None:
        return float("nan")

    encoded = x_bins * n_bins + y_bins
    contingency = (
        np.bincount(encoded, minlength=n_bins * n_bins).reshape(n_bins, n_bins).astype(float)
    )
    total = float(contingency.sum())
    if total <= 0.0:
        return float("nan")

    pxy = contingency / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    expected = px @ py
    positive = pxy > 0.0
    if not positive.any():
        return float("nan")

    with np.errstate(divide="ignore", invalid="ignore"):
        mi = float(np.sum(pxy[positive] * np.log(pxy[positive] / expected[positive])))
    if not np.isfinite(mi):
        return float("nan")
    return float(max(mi, 0.0))


def _rank_quantile_bins(values: np.ndarray, *, n_bins: int) -> np.ndarray | None:
    arr = np.asarray(values, dtype=float)
    mask = np.isfinite(arr)
    n_valid = int(mask.sum())
    if n_valid < max(2, n_bins):
        return None
    if n_bins < 2:
        return None

    valid_idx = np.flatnonzero(mask)
    # rank(method="first"): stable ordering for ties.
    order = valid_idx[np.argsort(arr[valid_idx], kind="mergesort")]
    # Equal-count binning equivalent to qcut over strict ranks.
    pos = np.arange(n_valid, dtype=float)
    bins_sorted = np.floor(pos * float(n_bins) / float(n_valid)).astype(int)

    out = np.empty(n_valid, dtype=int)
    out[np.argsort(order, kind="mergesort")] = bins_sorted
    if np.unique(out).size < 2:
        return None
    return out


def compute_ic_summary(ic_values: pd.Series) -> dict[str, float]:
    """Compute summary statistics for a series of per-date IC values.

    Parameters
    ----------
    ic_values : pd.Series
        Per-date IC (or RankIC) values, may contain NaN.

    Returns
    -------
    dict with keys: mean_ic, std_ic, ic_ir, t_stat, p_value, n_obs.
    """
    clean = ic_values.dropna()
    n = len(clean)
    if n == 0:
        return {
            "mean_ic": float("nan"),
            "std_ic": float("nan"),
            "ic_ir": float("nan"),
            "t_stat": float("nan"),
            "p_value": float("nan"),
            "n_obs": 0,
        }
    mean = float(clean.mean())
    std = float(clean.std(ddof=1)) if n > 1 else float("nan")
    std_is_effectively_zero = (not np.isfinite(std)) or (abs(std) <= np.finfo(float).eps)
    if n > 1 and not std_is_effectively_zero:
        ic_ir = mean / std
        t_stat = mean / (std / np.sqrt(n))
        p_value = float(2 * scipy_stats.t.sf(abs(t_stat), df=n - 1))
    else:
        ic_ir = float("nan")
        t_stat = float("nan")
        p_value = float("nan")
    return {
        "mean_ic": mean,
        "std_ic": std,
        "ic_ir": ic_ir,
        "t_stat": t_stat,
        "p_value": p_value,
        "n_obs": n,
    }


def _extract_single_factor_name(df: pd.DataFrame, *, table_name: str) -> str:
    factor_names = pd.unique(df["factor"])
    if len(factor_names) != 1:
        raise AlphaLabDataError(f"{table_name} must contain exactly one factor name")
    return str(factor_names[0])
