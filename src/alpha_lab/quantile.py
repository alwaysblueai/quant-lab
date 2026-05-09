from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError
from alpha_lab.interfaces import validate_factor_output

_QUANTILE_ASSIGNMENT_COLUMNS = ("date", "asset", "factor", "quantile")
_QUANTILE_RETURN_COLUMNS = ("date", "factor", "quantile", "mean_return")
_LONG_SHORT_COLUMNS = ("date", "factor", "long_short_return")


def quantile_assignments(
    factors: pd.DataFrame,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Compute per-asset quantile bucket assignments from factor values alone.

    Unlike :func:`quantile_returns`, no label DataFrame is required.  The
    assignment universe is all ``(date, asset)`` pairs where the factor value
    is non-NaN.  Dates with fewer than 2 non-NaN assets are excluded entirely
    (matching the :func:`~alpha_lab.quantile._assign_quantile` convention).

    **Universe note:** On the last ``horizon`` dates of a price series, factor
    values may be valid while forward-return labels are NaN.  Those dates *do*
    appear in the assignments output because a rebalancing decision would still
    be made at that date.  They are excluded from IC and quantile-return metrics
    but correctly captured by the turnover calculation.

    Parameters
    ----------
    factors:
        Canonical long-form DataFrame with columns ``[date, asset, factor,
        value]``.  Must contain exactly one factor name.
    n_quantiles:
        Number of quantile buckets.  Must be >= 2.

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, asset, factor, quantile]``.  One row per
        ``(date, asset)`` with a non-NaN factor value and a valid quantile
        assignment.  ``quantile`` is an integer in ``[1, n_quantiles]``.
    """
    if n_quantiles < 2:
        raise AlphaLabConfigError(f"n_quantiles must be >= 2, got {n_quantiles}")
    if factors.empty:
        return pd.DataFrame(columns=list(_QUANTILE_ASSIGNMENT_COLUMNS))

    validate_factor_output(factors)
    factor_name = _single_factor_name(factors, "factors")

    df = factors.loc[factors["value"].notna(), ["date", "asset", "value"]]
    if df.empty:
        return pd.DataFrame(columns=list(_QUANTILE_ASSIGNMENT_COLUMNS))

    df["quantile"] = _assign_quantiles_by_date(df, value_col="value", n_quantiles=n_quantiles)
    df = df.dropna(subset=["quantile"])
    df["quantile"] = df["quantile"].astype(int)
    df["factor"] = factor_name
    return df[list(_QUANTILE_ASSIGNMENT_COLUMNS)].reset_index(drop=True)


def quantile_returns(
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    n_quantiles: int = 5,
    *,
    merged_pairs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute average return per cross-sectional quantile bucket.

    Factor values at date ``t`` are ranked cross-sectionally to assign
    quantile labels 1 (bottom) through ``n_quantiles`` (top).  Returns are
    the corresponding ``labels`` values (e.g. forward returns stored at ``t``).
    Merging is done strictly on ``(date, asset)`` — no lookahead.

    Parameters
    ----------
    factors:
        Canonical long-form DataFrame with columns ``[date, asset, factor,
        value]``.  Must contain exactly one factor name.
    labels:
        Canonical long-form DataFrame with columns ``[date, asset, factor,
        value]``.  Must contain exactly one label name.  Typically the output
        of :func:`~alpha_lab.labels.forward_return`.
    n_quantiles:
        Number of quantile buckets.  When a date's cross-section has fewer
        than ``n_quantiles`` non-NaN assets the effective bucket count is
        reduced to the number of available assets.

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, factor, quantile, mean_return]``.

        **Bucket semantics**: this function performs *distinct-value bucket
        mapping*, not equal-count quantile splitting.  Each cross-date
        distinct factor value is mapped to a bucket label via a linear scale
        anchored at bucket 1 (lowest value) and bucket ``effective_q``
        (highest value).  When the number of distinct values is smaller than
        ``n_quantiles``, some intermediate bucket numbers will be absent; do
        not assume near-equal bucket sizes.

        Bucket 1 always contains the lowest-valued assets; the highest bucket
        (≤ ``n_quantiles``) always contains the highest-valued assets.
        Dates where fewer than 2 non-NaN assets remain after merging and
        NaN-dropping are excluded entirely.
    """
    if n_quantiles < 2:
        raise AlphaLabConfigError(f"n_quantiles must be >= 2, got {n_quantiles}")

    if factors.empty or labels.empty:
        return pd.DataFrame(columns=list(_QUANTILE_RETURN_COLUMNS))

    validate_factor_output(factors)
    validate_factor_output(labels)

    factor_name = _single_factor_name(factors, "factors")
    # Enforce a single label name so the merge below stays one-to-one.
    # If labels contains multiple horizons or label variants for the same
    # (date, asset) the merge would silently fan out rows and corrupt mean_return.
    _single_factor_name(labels, "labels")

    if merged_pairs is not None:
        required = {"date", "asset", "value_factor", "value_label"}
        missing = required - set(merged_pairs.columns)
        if missing:
            raise AlphaLabDataError(f"merged_pairs is missing required columns: {sorted(missing)}")
        dupes = merged_pairs.duplicated(subset=["date", "asset"])
        if dupes.any():
            raise AlphaLabDataError(
                "merged_pairs contains duplicate (date, asset) rows; "
                "factor/label pairs must be one-to-one"
            )
        merged = merged_pairs[["date", "asset", "value_factor", "value_label"]]
        value_col = "value_factor"
        label_col = "value_label"
    else:
        # Merge strictly on (date, asset) — the only safe join key.
        # validate="one_to_one" is a hard guard: after the single-label check above
        # the merge must be 1:1; any violation indicates a data-contract breach.
        merged = factors[["date", "asset", "value"]].merge(
            labels[["date", "asset", "value"]].rename(columns={"value": "_label"}),
            on=["date", "asset"],
            how="inner",
            validate="one_to_one",
        )
        value_col = "value"
        label_col = "_label"
    merged = merged.dropna(subset=[value_col, label_col])

    if merged.empty:
        return pd.DataFrame(columns=list(_QUANTILE_RETURN_COLUMNS))

    # Cross-sectional quantile assignment per date — uses only same-date data.
    merged["quantile"] = _assign_quantiles_by_date(
        merged,
        value_col=value_col,
        n_quantiles=n_quantiles,
    )
    merged = merged.dropna(subset=["quantile"])
    merged["quantile"] = merged["quantile"].astype(int)

    result = (
        merged.groupby(["date", "quantile"], sort=True)[label_col]
        .mean()
        .reset_index()
        .rename(columns={label_col: "mean_return"})
    )
    result["factor"] = factor_name
    return result[list(_QUANTILE_RETURN_COLUMNS)].reset_index(drop=True)


def long_short_return(quantile_ret: pd.DataFrame) -> pd.DataFrame:
    """Compute long-short return as top-quantile minus bottom-quantile.

    Parameters
    ----------
    quantile_ret:
        Output of :func:`quantile_returns` with columns
        ``[date, factor, quantile, mean_return]``.

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, factor, long_short_return]``.
        Dates where only a single quantile bucket exists produce NaN.
    """
    if quantile_ret.empty:
        return pd.DataFrame(columns=list(_LONG_SHORT_COLUMNS))

    missing = set(_QUANTILE_RETURN_COLUMNS) - set(quantile_ret.columns)
    if missing:
        raise AlphaLabDataError(f"Missing columns in quantile_ret: {missing}")

    per_bucket = (
        quantile_ret.groupby(["date", "factor", "quantile"], sort=True, as_index=False)[
            "mean_return"
        ]
        .mean()
        .sort_values(["date", "factor", "quantile"], kind="mergesort")
    )
    grouped = per_bucket.groupby(["date", "factor"], sort=True, group_keys=False)
    first_rows = (
        grouped.head(1)
        .set_index(["date", "factor"])[["quantile", "mean_return"]]
        .rename(columns={"quantile": "q_min", "mean_return": "bottom"})
    )
    last_rows = (
        grouped.tail(1)
        .set_index(["date", "factor"])[["quantile", "mean_return"]]
        .rename(columns={"quantile": "q_max", "mean_return": "top"})
    )
    n_q = per_bucket.groupby(["date", "factor"], sort=True)["quantile"].nunique()
    agg = first_rows.join(last_rows).join(n_q.rename("n_q"))
    long_short = agg["top"] - agg["bottom"]
    long_short.loc[agg["n_q"] < 2] = np.nan

    result = (
        long_short.rename("long_short_return")
        .reset_index()
        .sort_values(["date", "factor"], kind="mergesort")
    )
    return result[list(_LONG_SHORT_COLUMNS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assign_quantile(series: pd.Series, n_quantiles: int) -> pd.Series:
    """Assign cross-sectional quantile labels 1..effective_q.

    **Tie policy — row-order invariant with pinned extremes.**

    ``rank(method='dense')`` maps every distinct value to a consecutive integer
    starting at 1 for the minimum value.  Tied assets receive the same dense
    rank regardless of row order.

    Those dense ranks are then linearly mapped so that:

    - Dense rank 1 (lowest value) → bucket 1 (always).
    - Dense rank ``n_distinct`` (highest value) → bucket ``effective_q`` (always).
    - Intermediate distinct values → proportionally spaced buckets via the
      linear map above, rounded with **round-half-up** (``int(q + 0.5)``).
      Some intermediate bucket numbers may be absent when there are fewer
      distinct values than quantiles.  The exact bucket for a half-way case
      (e.g. ``q = 2.5``) is deterministically bucket 3, not 2.

    Consequences:
    - **Bottom** and **top** buckets are always occupied, so
      ``long_short_return`` always compares the true highest-valued group
      against the true lowest-valued group.
    - A constant factor (all values identical, ``n_distinct == 1``) collapses
      every asset to bucket 1; ``long_short_return`` then returns NaN —
      the correct result for an uninformative factor.

    NaN inputs produce NaN outputs.  When fewer than ``n_quantiles`` non-NaN
    assets are present, ``effective_q`` is reduced to ``n_valid``.
    """
    n_valid = int(series.notna().sum())
    if n_valid < 2:
        return pd.Series(np.nan, index=series.index, dtype=float)

    effective_q = min(n_quantiles, n_valid)
    n_distinct = int(series.dropna().nunique())

    # Constant factor: all assets land in bucket 1 → L/S returns NaN.
    if n_distinct == 1:
        result = pd.Series(np.nan, index=series.index, dtype=float)
        result[series.notna()] = 1.0
        return result

    # Dense rank: same value → same rank (1 = minimum, n_distinct = maximum).
    # Row-order invariant because the rank depends only on relative value ordering.
    dense_rank = series.rank(method="dense", na_option="keep")

    # Vectorised bucket assignment (replaces per-element .apply(lambda)).
    # Linear map: rank 1 → bucket 1, rank n_distinct → bucket effective_q.
    # Round-half-up via (q + 0.5).astype(int) matches the original int(q + 0.5).
    r = dense_rank.to_numpy(dtype=float)
    q = (r - 1) / (n_distinct - 1) * (effective_q - 1) + 1
    buckets = np.where(np.isnan(r), np.nan, np.floor(q + 0.5))
    return pd.Series(buckets, index=series.index, dtype=float)


def _assign_quantiles_by_date(
    df: pd.DataFrame,
    *,
    value_col: str,
    n_quantiles: int,
) -> pd.Series:
    """Assign dense-rank quantile buckets per date with a sorted-date fast path."""
    n_rows = len(df)
    out = np.full(n_rows, np.nan, dtype=float)
    if n_rows == 0:
        return pd.Series(out, index=df.index, dtype=float)

    dates = pd.to_datetime(df["date"], errors="coerce").to_numpy()
    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)

    valid_rows = np.isfinite(values) & ~pd.isna(dates)
    if not valid_rows.any():
        return pd.Series(out, index=df.index, dtype=float)

    valid_idx = np.flatnonzero(valid_rows)
    valid_dates = dates[valid_rows]
    valid_values = values[valid_rows]

    if len(valid_dates) > 1 and not bool(np.all(valid_dates[1:] >= valid_dates[:-1])):
        order = np.argsort(valid_dates, kind="mergesort")
        ordered_idx = valid_idx[order]
        ordered_dates = valid_dates[order]
        ordered_values = valid_values[order]
    else:
        ordered_idx = valid_idx
        ordered_dates = valid_dates
        ordered_values = valid_values

    _, start_idx = np.unique(ordered_dates, return_index=True)
    end_idx = np.append(start_idx[1:], len(ordered_dates))

    for begin, end in zip(start_idx, end_idx, strict=True):
        row_idx = ordered_idx[int(begin) : int(end)]
        group_values = ordered_values[int(begin) : int(end)]

        n_valid = int(group_values.size)
        if n_valid < 2:
            continue

        effective_q = min(n_quantiles, n_valid)
        unique_vals, inverse = np.unique(group_values, return_inverse=True)
        n_distinct = int(unique_vals.size)
        if n_distinct == 1:
            out[row_idx] = 1.0
            continue

        dense_rank = inverse.astype(float) + 1.0
        q = (dense_rank - 1.0) / float(n_distinct - 1) * float(effective_q - 1) + 1.0
        out[row_idx] = np.floor(q + 0.5)

    return pd.Series(out, index=df.index, dtype=float)


def _single_factor_name(df: pd.DataFrame, table_name: str) -> str:
    names = pd.unique(df["factor"])
    if len(names) != 1:
        raise AlphaLabDataError(f"{table_name} must contain exactly one factor name")
    return str(names[0])
