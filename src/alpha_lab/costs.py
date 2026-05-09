from __future__ import annotations

import pandas as pd

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError

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
        raise AlphaLabConfigError(f"cost_rate must be >= 0, got {cost_rate}")
    if not returns.index.equals(turnover.index):
        raise AlphaLabDataError(
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
        raise AlphaLabConfigError(f"cost_rate must be >= 0, got {cost_rate}")

    if long_short_df.empty or long_short_turnover_df.empty:
        return pd.DataFrame(columns=list(_COST_ADJUSTED_COLUMNS))

    required_ls = {"date", "factor", "long_short_return"}
    missing_ls = required_ls - set(long_short_df.columns)
    if missing_ls:
        raise AlphaLabDataError(f"long_short_df is missing required columns: {sorted(missing_ls)}")
    required_turnover = {"date", "factor", "long_short_turnover"}
    missing_turnover = required_turnover - set(long_short_turnover_df.columns)
    if missing_turnover:
        raise AlphaLabDataError(
            "long_short_turnover_df is missing required columns: "
            f"{sorted(missing_turnover)}"
        )
    if long_short_df.duplicated(subset=["date", "factor"]).any():
        raise AlphaLabDataError(
            "long_short_df contains duplicate (date, factor) rows; "
            "long-short returns must be one-to-one by date and factor"
        )
    if long_short_turnover_df.duplicated(subset=["date", "factor"]).any():
        raise AlphaLabDataError(
            "long_short_turnover_df contains duplicate (date, factor) rows; "
            "turnover rows must be one-to-one by date and factor"
        )

    merged = long_short_df.merge(
        long_short_turnover_df.rename(columns={"long_short_turnover": "turnover"}),
        on=["date", "factor"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        return pd.DataFrame(columns=list(_COST_ADJUSTED_COLUMNS))

    merged = merged.sort_values(["date", "factor"]).reset_index(drop=True)
    merged["adjusted_return"] = apply_linear_cost(
        merged["long_short_return"], merged["turnover"], cost_rate=cost_rate
    )
    return merged[list(_COST_ADJUSTED_COLUMNS)].reset_index(drop=True)


def apply_short_borrow_cost(
    returns: pd.Series,  # type: ignore[type-arg]
    short_weights: pd.Series,  # type: ignore[type-arg]
    annual_rate: float = 0.08,
) -> pd.Series:  # type: ignore[type-arg]
    """Apply daily short-borrow financing costs to a return series.

    **Rate convention — note the asymmetry with** :func:`apply_linear_cost`:
    this function takes an **annualized** rate and divides by ``252``
    internally (daily convention). :func:`apply_linear_cost` takes a
    **per-period** ``cost_rate`` and applies it directly. Callers mixing the
    two must ensure the return-series cadence matches each function's
    assumption.
    """
    if annual_rate < 0:
        raise AlphaLabConfigError(f"annual_rate must be >= 0, got {annual_rate}")
    if not returns.index.equals(short_weights.index):
        raise AlphaLabDataError(
            "returns and short_weights must share the same index; "
            "pass aligned time series before borrow-cost adjustment"
        )

    short_leg = (-pd.to_numeric(short_weights, errors="coerce").clip(upper=0.0)).fillna(0.0)
    daily_cost = short_leg * float(annual_rate) / 252.0
    return returns - daily_cost
