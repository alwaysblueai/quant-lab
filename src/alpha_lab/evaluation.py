from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.frame_utils import readonly_shallow_copy
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

    merged = _maybe_borrow_or_resolve_merged_pairs(
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


def compute_mean_rank_ic_permutation_null(
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
    merged_pairs: pd.DataFrame | None = None,
    min_assets_per_date: int = 3,
    batch_size: int = 256,
) -> tuple[float, np.ndarray]:
    """Sample a mean RankIC null distribution by per-date factor permutation.

    Returns
    -------
    tuple
        ``(observed_mean_rank_ic, null_samples)`` where ``null_samples`` has
        length ``n_permutations`` when valid cross-sections are available.
    """
    if n_permutations <= 0:
        raise ValueError("n_permutations must be > 0")
    if min_assets_per_date < 2:
        raise ValueError("min_assets_per_date must be >= 2")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if factors.empty or labels.empty:
        return float("nan"), np.asarray([], dtype=float)

    validate_factor_output(factors)
    validate_factor_output(labels)
    merged = _maybe_borrow_or_resolve_merged_pairs(
        factors=factors,
        labels=labels,
        merged_pairs=merged_pairs,
    )
    if merged.empty:
        return float("nan"), np.asarray([], dtype=float)

    kernel = _prepare_rank_ic_permutation_kernel(
        merged,
        min_assets_per_date=min_assets_per_date,
    )
    if kernel is None:
        return float("nan"), np.asarray([], dtype=float)
    observed_per_date, factor_centered, label_scaled = kernel
    observed_mean = float(np.mean(observed_per_date)) if observed_per_date.size else float("nan")

    n_trials = int(n_permutations)
    rng = np.random.default_rng(seed)
    n_dates = len(factor_centered)
    null_sum = np.zeros(n_trials, dtype=float)
    supports_permuted = hasattr(rng, "permuted")
    for idx in range(n_dates):
        factor_values = factor_centered[idx]
        label_values = label_scaled[idx]
        width = int(factor_values.size)
        scratch = np.empty((min(int(batch_size), n_trials), width), dtype=float)
        start = 0
        while start < n_trials:
            block = min(int(batch_size), n_trials - start)
            view = scratch[:block]
            view[:] = factor_values
            if supports_permuted:
                rng.permuted(view, axis=1, out=view)
            else:  # pragma: no cover - compatibility fallback for older NumPy
                random_keys = rng.random((block, width))
                permutation_idx = np.argsort(random_keys, axis=1, kind="quicksort")
                view[:] = np.take(factor_values, permutation_idx)
            null_sum[start : start + block] += view @ label_values
            start += block

    null = null_sum / float(n_dates)

    null = null[np.isfinite(null)]
    return observed_mean, null


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

    merged = _maybe_borrow_or_resolve_merged_pairs(
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
    """Compute per-date cross-sectional correlation.

    Priority order:
    1) Numba kernel when available.
    2) NumPy/scipy grouped kernel fallback.
    """
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

    return _cross_sectional_corr_numpy_path(
        merged,
        method=method,
        value_name=value_name,
        all_dates=all_dates,
    )


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


def _cross_sectional_corr_numpy_path(
    merged: pd.DataFrame,
    *,
    method: str,
    value_name: str,
    all_dates: pd.Index,
) -> pd.DataFrame:
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

    corr = np.empty(len(unique_dates), dtype=float)
    corr.fill(np.nan)
    for i in range(len(unique_dates)):
        begin = int(start_idx[i])
        end = int(end_idx[i])
        x = factor_values[begin:end]
        y = label_values[begin:end]
        if x.size < 2:
            continue
        if method == "spearman":
            x = scipy_stats.rankdata(x, method="average")
            y = scipy_stats.rankdata(y, method="average")
        corr[i] = _pearson_corr_numpy(x, y)

    result = pd.DataFrame(
        {
            "date": pd.to_datetime(unique_dates),
            value_name: corr,
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


def _pearson_corr_numpy(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x_dev = x - x.mean()
    y_dev = y - y.mean()
    denom = float(np.sqrt(np.dot(x_dev, x_dev) * np.dot(y_dev, y_dev)))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(np.dot(x_dev, y_dev) / denom)


def _resolve_merged_pairs(
    *,
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    merged_pairs: pd.DataFrame | None,
) -> pd.DataFrame:
    if merged_pairs is not None:
        _validate_merged_pairs_contract(merged_pairs)
        return merged_pairs[["date", "asset", "value_factor", "value_label"]].copy()

    return factors.merge(
        labels,
        on=["date", "asset"],
        how="inner",
        suffixes=("_factor", "_label"),
        validate="one_to_one",
    )


def _maybe_borrow_or_resolve_merged_pairs(
    *,
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    merged_pairs: pd.DataFrame | None,
) -> pd.DataFrame:
    if merged_pairs is not None:
        return _borrow_merged_pairs(merged_pairs)
    return _resolve_merged_pairs(factors=factors, labels=labels, merged_pairs=None)


def _borrow_merged_pairs(merged_pairs: pd.DataFrame) -> pd.DataFrame:
    """Borrow a read-only merged-pairs view for hot paths.

    Callers must treat the returned frame as read-only. New columns may be
    added to the shallow view, but existing shared column arrays are protected
    so accidental in-place mutation fails instead of corrupting the source.
    """
    _validate_merged_pairs_contract(merged_pairs)
    return readonly_shallow_copy(
        merged_pairs,
        columns=("date", "asset", "value_factor", "value_label"),
    )


def _validate_merged_pairs_contract(merged_pairs: pd.DataFrame) -> None:
    required = {"date", "asset", "value_factor", "value_label"}
    missing = required - set(merged_pairs.columns)
    if missing:
        raise AlphaLabDataError(f"merged_pairs is missing required columns: {sorted(missing)}")
    if merged_pairs.duplicated(subset=["date", "asset"]).any():
        raise AlphaLabDataError(
            "merged_pairs contains duplicate (date, asset) rows; "
            "factor/label pairs must be one-to-one"
        )


def _prepare_rank_ic_permutation_kernel(
    merged: pd.DataFrame,
    *,
    min_assets_per_date: int,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]] | None:
    required = {"date", "value_factor", "value_label"}
    missing = required - set(merged.columns)
    if missing:
        raise AlphaLabDataError(
            f"merged pairs for permutation kernel missing required columns: {sorted(missing)}"
        )

    clean = merged[["date", "value_factor", "value_label"]].dropna(
        subset=["value_factor", "value_label"]
    )
    if clean.empty:
        return None

    observed_per_date: list[float] = []
    factor_centered: list[np.ndarray] = []
    label_scaled: list[np.ndarray] = []
    for _, frame in clean.groupby("date", sort=False):
        if len(frame) < int(min_assets_per_date):
            continue
        f_rank = pd.Series(frame["value_factor"]).rank(method="average").to_numpy(dtype=float)
        y_rank = pd.Series(frame["value_label"]).rank(method="average").to_numpy(dtype=float)
        f_dev = f_rank - f_rank.mean()
        y_dev = y_rank - y_rank.mean()
        denom = float(np.sqrt(np.dot(f_dev, f_dev) * np.dot(y_dev, y_dev)))
        if not np.isfinite(denom) or denom <= 0.0:
            continue
        observed_per_date.append(float(np.dot(f_dev, y_dev) / denom))
        factor_centered.append(f_dev)
        label_scaled.append(y_dev / denom)

    if not factor_centered:
        return None
    return np.asarray(observed_per_date, dtype=float), factor_centered, label_scaled


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
    sort_pos = np.argsort(arr[valid_idx], kind="mergesort")
    # Equal-count binning equivalent to qcut over strict ranks.
    pos = np.arange(n_valid, dtype=float)
    bins_sorted = np.floor(pos * float(n_bins) / float(n_valid)).astype(int)

    out = np.empty(n_valid, dtype=int)
    out[sort_pos] = bins_sorted
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
