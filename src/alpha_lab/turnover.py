from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.exceptions import AlphaLabDataError

_QUANTILE_TURNOVER_COLUMNS: tuple[str, ...] = ("date", "factor", "quantile", "turnover")
_LONG_SHORT_TURNOVER_COLUMNS: tuple[str, ...] = ("date", "factor", "long_short_turnover")


def quantile_turnover(assignments: pd.DataFrame) -> pd.DataFrame:
    """Compute period-to-period one-way turnover for each quantile bucket.

    **Turnover definition (one-way entry rate):**

    For bucket *q* at date *t* (transition from *t-1* to *t*)::

        entering(q, t) = members(q, t) − members(q, t-1)
        turnover(q, t) = |entering(q, t)| / |members(q, t)|

    This is the fraction of the portfolio at *t* that is new relative to
    *t-1*, i.e. the fraction that must be bought.  It ranges from 0 (no
    change) to 1 (complete replacement).

    **Special cases:**

    - **First observation date per factor:** NaN — no prior state is
      available to compute a transition.
    - **Empty bucket at t:** NaN — undefined.
    - **Bucket absent at t-1 but present at t:** all members are entering →
      turnover = 1.0.

    **No lookahead:** Turnover at date *t* uses only the portfolio states at
    *t* and *t-1*.  No future information is used.

    This is a minimal research friction estimate.  It does not model
    execution timing, intraday slippage, or partial fills.

    Parameters
    ----------
    assignments:
        Per-asset quantile assignments with columns
        ``[date, asset, factor, quantile]``.  Typically the output of
        :func:`~alpha_lab.quantile.quantile_assignments`.
        Must contain exactly one factor name.

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, factor, quantile, turnover]``.
        One row per ``(date, quantile)`` present in ``assignments``.
        Rows are sorted by ``(date, quantile)``.
    """
    if assignments.empty:
        return pd.DataFrame(columns=list(_QUANTILE_TURNOVER_COLUMNS))

    _check_assignment_columns(assignments)
    factor_name = _single_name(assignments["factor"], "assignments")

    dupes = assignments.duplicated(subset=["date", "asset"])
    if dupes.any():
        raise AlphaLabDataError(
            "assignments contains duplicate (date, asset) rows; "
            "each asset must appear at most once per date"
        )

    df = assignments.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["quantile"] = pd.to_numeric(df["quantile"], errors="coerce")
    df = df.dropna(subset=["quantile"])
    if df.empty:
        return pd.DataFrame(columns=list(_QUANTILE_TURNOVER_COLUMNS))
    df["quantile"] = df["quantile"].astype(int)

    dates = pd.Index(sorted(df["date"].unique()))
    date_to_idx = pd.Series(np.arange(len(dates), dtype=int), index=dates)
    df["_t"] = date_to_idx.reindex(df["date"]).to_numpy(dtype=int)

    members = df[["_t", "quantile", "asset"]].copy()
    prev_members = members.copy()
    prev_members["_t"] = prev_members["_t"] + 1
    prev_members["_in_prev"] = 1

    aligned = members.merge(
        prev_members,
        on=["_t", "quantile", "asset"],
        how="left",
        validate="one_to_one",
    )
    aligned["_in_prev"] = aligned["_in_prev"].fillna(0).astype(int)

    agg = (
        aligned.groupby(["_t", "quantile"], sort=True)
        .agg(n_curr=("asset", "size"), n_overlap=("_in_prev", "sum"))
        .reset_index()
    )
    agg["turnover"] = 1.0 - (agg["n_overlap"] / agg["n_curr"])
    agg.loc[agg["_t"] == 0, "turnover"] = np.nan
    agg["date"] = dates.to_numpy()[agg["_t"].to_numpy(dtype=int)]
    agg["factor"] = factor_name

    return (
        agg[["date", "factor", "quantile", "turnover"]]
        .sort_values(["date", "quantile"], kind="mergesort")
        .reset_index(drop=True)
    )


def long_short_turnover(quantile_turnover_df: pd.DataFrame) -> pd.DataFrame:
    """Compute long-short turnover as the average of top and bottom bucket turnover.

    The long leg is the highest occupied quantile bucket at each date; the
    short leg is the lowest.  This mirrors
    :func:`~alpha_lab.quantile.long_short_return`.

    **Definition:**

        long_short_turnover(t) = (turnover(top_q, t) + turnover(bottom_q, t)) / 2

    NaN when either leg is NaN (including the first date per factor where no
    prior portfolio state is available), or when only one bucket is occupied.

    Parameters
    ----------
    quantile_turnover_df:
        Output of :func:`quantile_turnover` with columns
        ``[date, factor, quantile, turnover]``.

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, factor, long_short_turnover]``.
    """
    if quantile_turnover_df.empty:
        return pd.DataFrame(columns=list(_LONG_SHORT_TURNOVER_COLUMNS))

    missing = set(_QUANTILE_TURNOVER_COLUMNS) - set(quantile_turnover_df.columns)
    if missing:
        raise AlphaLabDataError(f"Missing columns in quantile_turnover_df: {missing}")

    per_bucket = (
        quantile_turnover_df.groupby(["date", "factor", "quantile"], sort=True, as_index=False)[
            "turnover"
        ]
        .mean()
        .sort_values(["date", "factor", "quantile"], kind="mergesort")
    )

    grouped = per_bucket.groupby(["date", "factor"], sort=True, group_keys=False)
    first_rows = (
        grouped.head(1)
        .set_index(["date", "factor"])[["quantile", "turnover"]]
        .rename(columns={"quantile": "q_min", "turnover": "bot"})
    )
    last_rows = (
        grouped.tail(1)
        .set_index(["date", "factor"])[["quantile", "turnover"]]
        .rename(columns={"quantile": "q_max", "turnover": "top"})
    )
    n_q = per_bucket.groupby(["date", "factor"], sort=True)["quantile"].nunique()
    agg = first_rows.join(last_rows).join(n_q.rename("n_q"))
    ls_turn = (agg["bot"] + agg["top"]) / 2.0
    ls_turn.loc[(agg["n_q"] < 2) | agg["bot"].isna() | agg["top"].isna()] = np.nan

    result = (
        ls_turn.rename("long_short_turnover")
        .reset_index()
        .sort_values(["date", "factor"], kind="mergesort")
    )
    return result[list(_LONG_SHORT_TURNOVER_COLUMNS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_assignment_columns(df: pd.DataFrame) -> None:
    required = {"date", "asset", "factor", "quantile"}
    missing = required - set(df.columns)
    if missing:
        raise AlphaLabDataError(f"assignments is missing required columns: {missing}")


def _single_name(series: pd.Series, table_name: str) -> str:  # type: ignore[type-arg]
    names = pd.unique(series)
    if len(names) != 1:
        raise AlphaLabDataError(f"{table_name} must contain exactly one factor name, got {names!r}")
    return str(names[0])
