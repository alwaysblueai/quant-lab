from __future__ import annotations

import math

import pandas as pd

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
        raise ValueError(
            "assignments contains duplicate (date, asset) rows; "
            "each asset must appear at most once per date"
        )

    df = assignments.copy()
    df["date"] = pd.to_datetime(df["date"])

    dates = sorted(df["date"].unique())
    rows: list[dict[str, object]] = []
    prev_buckets: dict[int, frozenset[str]] = {}

    for date in dates:
        day_df = df[df["date"] == date]
        curr_buckets: dict[int, frozenset[str]] = {
            int(q): frozenset(g["asset"])
            for q, g in day_df.groupby("quantile")
        }

        for q in sorted(curr_buckets):
            curr_members = curr_buckets[q]
            n_curr = len(curr_members)
            if n_curr == 0 or not prev_buckets:
                turn: float = float("nan")
            else:
                prev_members = prev_buckets.get(q, frozenset())
                entering = curr_members - prev_members
                turn = len(entering) / n_curr

            rows.append(
                {"date": date, "factor": factor_name, "quantile": q, "turnover": turn}
            )

        prev_buckets = curr_buckets

    if not rows:
        return pd.DataFrame(columns=list(_QUANTILE_TURNOVER_COLUMNS))
    return pd.DataFrame(rows, columns=list(_QUANTILE_TURNOVER_COLUMNS)).reset_index(
        drop=True
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
        raise ValueError(f"Missing columns in quantile_turnover_df: {missing}")

    def _ls_turn(group: pd.DataFrame) -> float:
        q_min = int(group["quantile"].min())
        q_max = int(group["quantile"].max())
        if q_min == q_max:
            return float("nan")
        bot = float(group.loc[group["quantile"] == q_min, "turnover"].iloc[0])
        top = float(group.loc[group["quantile"] == q_max, "turnover"].iloc[0])
        if math.isnan(bot) or math.isnan(top):
            return float("nan")
        return (bot + top) / 2.0

    result = (
        quantile_turnover_df.groupby(["date", "factor"], sort=True)
        .apply(_ls_turn, include_groups=False)
        .reset_index()
        .rename(columns={0: "long_short_turnover"})
    )
    return result[list(_LONG_SHORT_TURNOVER_COLUMNS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_assignment_columns(df: pd.DataFrame) -> None:
    required = {"date", "asset", "factor", "quantile"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"assignments is missing required columns: {missing}")


def _single_name(series: pd.Series, table_name: str) -> str:  # type: ignore[type-arg]
    names = pd.unique(series)
    if len(names) != 1:
        raise ValueError(
            f"{table_name} must contain exactly one factor name, got {names!r}"
        )
    return str(names[0])
