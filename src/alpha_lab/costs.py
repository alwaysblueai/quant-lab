from __future__ import annotations

import pandas as pd

_COST_ADJUSTED_COLUMNS: tuple[str, ...] = (
    "date",
    "factor",
    "long_short_return",
    "turnover",
    "adjusted_return",
)


def apply_linear_cost(
    returns: pd.Series,  # type: ignore[type-arg]
    turnover: pd.Series,  # type: ignore[type-arg]
    cost_rate: float,
) -> pd.Series:  # type: ignore[type-arg]
    """Apply a one-way linear transaction cost to a return series.

    **Cost model:**

        adjusted_return(t) = return(t) - cost_rate × turnover(t)

    ``cost_rate`` is the **one-way** cost per unit of turnover (e.g., 0.001
    for 10 basis points per one-way transaction).  Turnover of 1.0 means the
    portfolio is fully replaced; the cost incurred is ``cost_rate × 1.0``.

    **This is a minimal research friction estimate only.**  It does not model
    bid-ask spread variation, market impact, short-borrow fees, timing risk,
    or partial fills.  Cost is charged once per period per unit of one-way
    turnover.

    **NaN propagation:** If either ``return(t)`` or ``turnover(t)`` is NaN
    the adjusted return at *t* is NaN.  The first evaluation date typically
    has NaN turnover (no prior portfolio state), so its adjusted return is
    also NaN.

    Parameters
    ----------
    returns:
        Per-period return series (e.g. long-short return indexed by date).
    turnover:
        Per-period one-way turnover series.  Must share the exact same index
        as ``returns``.  Values are expected in ``[0, 1]``; NaN where no
        prior portfolio state is available.
    cost_rate:
        Non-negative one-way cost rate.  Must be ``>= 0``.

    Returns
    -------
    pd.Series
        Cost-adjusted returns, same index as inputs.

    Raises
    ------
    ValueError
        If ``cost_rate < 0``.
    ValueError
        If ``returns`` and ``turnover`` do not share the same index.
    """
    if cost_rate < 0:
        raise ValueError(f"cost_rate must be >= 0, got {cost_rate}")
    if not returns.index.equals(turnover.index):
        raise ValueError(
            "returns and turnover must share the same index; "
            f"got lengths {len(returns)} and {len(turnover)} with non-matching indices"
        )
    return returns - cost_rate * turnover


def cost_adjusted_long_short(
    long_short_df: pd.DataFrame,
    long_short_turnover_df: pd.DataFrame,
    cost_rate: float,
) -> pd.DataFrame:
    """Apply linear transaction costs to long-short returns.

    Merges ``long_short_df`` and ``long_short_turnover_df`` on
    ``(date, factor)`` then subtracts ``cost_rate × turnover`` from each
    period's long-short return.

    **Timing assumption:** Turnover at date *t* is the one-way portfolio
    replacement rate entering the period that earns the return labeled at *t*.
    The cost is charged once per period at the start of the period.

    **This is a minimal research friction estimate only** (see
    :func:`apply_linear_cost` for the full disclaimer).

    Parameters
    ----------
    long_short_df:
        Output of :func:`~alpha_lab.quantile.long_short_return` with columns
        ``[date, factor, long_short_return]``.
    long_short_turnover_df:
        Output of :func:`~alpha_lab.turnover.long_short_turnover` with columns
        ``[date, factor, long_short_turnover]``.
    cost_rate:
        Non-negative one-way cost rate.  Passed to :func:`apply_linear_cost`.

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, factor, long_short_return, turnover,
        adjusted_return]``.  Rows sorted by ``(date, factor)``.
        ``adjusted_return`` is NaN on dates where ``turnover`` is NaN
        (e.g. the first evaluation date).

    Raises
    ------
    ValueError
        If ``cost_rate < 0``.
    """
    if cost_rate < 0:
        raise ValueError(f"cost_rate must be >= 0, got {cost_rate}")

    if long_short_df.empty or long_short_turnover_df.empty:
        return pd.DataFrame(columns=list(_COST_ADJUSTED_COLUMNS))

    merged = long_short_df.merge(
        long_short_turnover_df.rename(columns={"long_short_turnover": "turnover"}),
        on=["date", "factor"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=list(_COST_ADJUSTED_COLUMNS))

    merged = merged.sort_values(["date", "factor"]).reset_index(drop=True)
    merged["adjusted_return"] = apply_linear_cost(
        merged["long_short_return"], merged["turnover"], cost_rate=cost_rate
    )
    return merged[list(_COST_ADJUSTED_COLUMNS)].reset_index(drop=True)
