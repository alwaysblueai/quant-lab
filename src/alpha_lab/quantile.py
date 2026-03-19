from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS

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
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")
    if factors.empty:
        return pd.DataFrame(columns=list(_QUANTILE_ASSIGNMENT_COLUMNS))

    _validate_canonical(factors, "factors")
    factor_name = _single_factor_name(factors, "factors")

    df = factors[["date", "asset", "value"]].dropna(subset=["value"]).copy()
    if df.empty:
        return pd.DataFrame(columns=list(_QUANTILE_ASSIGNMENT_COLUMNS))

    df["quantile"] = df.groupby("date", sort=True)["value"].transform(
        lambda s: _assign_quantile(s, n_quantiles)
    )
    df = df.dropna(subset=["quantile"])
    df["quantile"] = df["quantile"].astype(int)
    df["factor"] = factor_name
    return df[list(_QUANTILE_ASSIGNMENT_COLUMNS)].reset_index(drop=True)


def quantile_returns(
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    n_quantiles: int = 5,
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
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")

    if factors.empty or labels.empty:
        return pd.DataFrame(columns=list(_QUANTILE_RETURN_COLUMNS))

    _validate_canonical(factors, "factors")
    _validate_canonical(labels, "labels")

    factor_name = _single_factor_name(factors, "factors")
    # Enforce a single label name so the merge below stays one-to-one.
    # If labels contains multiple horizons or label variants for the same
    # (date, asset) the merge would silently fan out rows and corrupt mean_return.
    _single_factor_name(labels, "labels")

    # Merge strictly on (date, asset) — the only safe join key.
    # validate="one_to_one" is a hard guard: after the single-label check above
    # the merge must be 1:1; any violation indicates a data-contract breach.
    merged = factors[["date", "asset", "value"]].merge(
        labels[["date", "asset", "value"]].rename(columns={"value": "_label"}),
        on=["date", "asset"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.dropna(subset=["value", "_label"])

    if merged.empty:
        return pd.DataFrame(columns=list(_QUANTILE_RETURN_COLUMNS))

    # Cross-sectional quantile assignment per date — uses only same-date data.
    merged["quantile"] = merged.groupby("date", sort=True)["value"].transform(
        lambda s: _assign_quantile(s, n_quantiles)
    )
    merged = merged.dropna(subset=["quantile"])
    merged["quantile"] = merged["quantile"].astype(int)

    result = (
        merged.groupby(["date", "quantile"], sort=True)["_label"]
        .mean()
        .reset_index()
        .rename(columns={"_label": "mean_return"})
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
        raise ValueError(f"Missing columns in quantile_ret: {missing}")

    def _ls(group: pd.DataFrame) -> float:
        q_min = int(group["quantile"].min())
        q_max = int(group["quantile"].max())
        if q_min == q_max:
            return float("nan")
        bottom = float(group.loc[group["quantile"] == q_min, "mean_return"].mean())
        top = float(group.loc[group["quantile"] == q_max, "mean_return"].mean())
        return top - bottom

    result = (
        quantile_ret.groupby(["date", "factor"], sort=True)
        .apply(_ls, include_groups=False)
        .reset_index()
        .rename(columns={0: "long_short_return"})
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

    def _bucket(r: float) -> float:
        # Linear map: rank 1 → bucket 1, rank n_distinct → bucket effective_q.
        # Endpoints are always exact integers by construction.  Half-way values
        # (exact .5) use "round half up" via int(q + 0.5), which is explicit
        # and auditable — Python's built-in round() uses banker's rounding
        # (round-half-to-even) and would be harder to reason about here.
        q = (r - 1) / (n_distinct - 1) * (effective_q - 1) + 1
        return float(int(q + 0.5))

    return dense_rank.apply(lambda r: _bucket(r) if pd.notna(r) else float("nan"))


def _validate_canonical(df: pd.DataFrame, table_name: str) -> None:
    missing = set(FACTOR_OUTPUT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{table_name} is missing required columns: {missing}")
    if df["date"].isna().any():
        raise ValueError(f"{table_name} contains NaT in 'date'")
    if df["asset"].isna().any():
        raise ValueError(f"{table_name} contains NaN in 'asset'")
    dupes = df.duplicated(subset=["date", "asset", "factor"])
    if dupes.any():
        raise ValueError(f"{table_name} contains duplicate (date, asset, factor) rows")


def _single_factor_name(df: pd.DataFrame, table_name: str) -> str:
    names = pd.unique(df["factor"])
    if len(names) != 1:
        raise ValueError(f"{table_name} must contain exactly one factor name")
    return str(names[0])
