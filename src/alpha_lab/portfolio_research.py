from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column name constants
# ---------------------------------------------------------------------------

_WEIGHT_COLUMNS: tuple[str, ...] = ("date", "asset", "weight")
_PORTFOLIO_RETURN_COLUMNS: tuple[str, ...] = ("date", "portfolio_return")
_PORTFOLIO_TURNOVER_COLUMNS: tuple[str, ...] = ("date", "portfolio_turnover")
_PORTFOLIO_COST_ADJ_COLUMNS: tuple[str, ...] = ("date", "portfolio_return", "adjusted_return")

_WEIGHT_METHODS: frozenset[str] = frozenset({"equal", "rank", "score"})


# ---------------------------------------------------------------------------
# portfolio_weights
# ---------------------------------------------------------------------------


def portfolio_weights(
    factor_df: pd.DataFrame,
    method: str = "equal",
    *,
    top_k: int | None = None,
    bottom_k: int | None = None,
) -> pd.DataFrame:
    """Compute cross-sectional portfolio weights from factor values.

    On each date the factor values are ranked cross-sectionally and weights
    are assigned to the top and/or bottom assets according to ``method``.

    **Weight conventions**

    - Long-only (only ``top_k`` provided, or neither ``top_k`` nor
      ``bottom_k``): weights over the selected assets sum to +1.
    - Long-short (both ``top_k`` and ``bottom_k``): long-leg weights sum to
      +1 and short-leg weights sum to -1, so the net portfolio weight is 0.

    Parameters
    ----------
    factor_df:
        Canonical long-form DataFrame with columns ``[date, asset, factor,
        value]``.  Must contain exactly one factor name.
    method:
        Weight scheme.  One of:

        - ``"equal"``: each selected asset receives equal weight.
        - ``"rank"``: weights proportional to cross-sectional rank; the
          highest-valued asset receives the highest weight.
        - ``"score"``: weights proportional to ``value - min(value)``; the
          minimum-valued asset in the selection always receives zero weight.

    top_k:
        Number of highest-valued assets to include in the long leg.
        Defaults to all assets when ``None``.
    bottom_k:
        Number of lowest-valued assets to include in the short leg.
        ``None`` means no short leg (long-only portfolio).

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, asset, weight]``.  One row per selected asset per
        date.  Dates with fewer than 2 non-NaN assets are excluded.

    Raises
    ------
    ValueError
        If ``method`` is not one of ``{"equal", "rank", "score"}``, if
        required columns are missing, or if ``factor_df`` contains more than
        one factor name.
    """
    if method not in _WEIGHT_METHODS:
        raise ValueError(
            f"method must be one of {sorted(_WEIGHT_METHODS)}, got {method!r}"
        )

    required = {"date", "asset", "factor", "value"}
    missing = required - set(factor_df.columns)
    if missing:
        raise ValueError(f"factor_df is missing required columns: {missing}")

    if factor_df.empty:
        return pd.DataFrame(columns=list(_WEIGHT_COLUMNS))

    factor_names = pd.unique(factor_df["factor"])
    if len(factor_names) != 1:
        raise ValueError("factor_df must contain exactly one factor name")

    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if bottom_k is not None and bottom_k <= 0:
        raise ValueError("bottom_k must be a positive integer")

    all_rows: list[dict[str, object]] = []

    for date, group in factor_df.groupby("date", sort=True):
        group = group.dropna(subset=["value"]).copy()
        n = len(group)
        if n < 2:
            continue

        # Sort descending: index 0 = highest factor value.
        group = group.sort_values("value", ascending=False).reset_index(drop=True)

        eff_top = min(top_k, n) if top_k is not None else n
        # Bottom_k cannot consume assets already in the long leg.
        eff_bottom = (
            min(bottom_k, n - eff_top) if bottom_k is not None else 0
        )

        long_vals = group["value"].iloc[:eff_top].to_numpy(dtype=float)
        long_assets = group["asset"].iloc[:eff_top].to_numpy()
        long_w = _compute_weights(long_vals, method)

        if eff_bottom > 0:
            # For the short leg, negate values so the *lowest* factor value
            # maps to the *highest* negated value → highest short weight.
            short_vals = -group["value"].iloc[n - eff_bottom :].to_numpy(dtype=float)
            short_assets = group["asset"].iloc[n - eff_bottom :].to_numpy()
            short_w = _compute_weights(short_vals, method)

            for asset, w in zip(long_assets, long_w, strict=True):
                all_rows.append({"date": date, "asset": asset, "weight": float(w)})
            for asset, w in zip(short_assets, short_w, strict=True):
                all_rows.append({"date": date, "asset": asset, "weight": -float(w)})
        else:
            for asset, w in zip(long_assets, long_w, strict=True):
                all_rows.append({"date": date, "asset": asset, "weight": float(w)})

    if not all_rows:
        return pd.DataFrame(columns=list(_WEIGHT_COLUMNS))

    return pd.DataFrame(all_rows, columns=list(_WEIGHT_COLUMNS)).reset_index(drop=True)


# ---------------------------------------------------------------------------
# simulate_portfolio_returns
# ---------------------------------------------------------------------------


def simulate_portfolio_returns(
    weights_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    *,
    holding_period: int = 1,
    rebalance_frequency: int = 1,
) -> pd.DataFrame:
    """Simulate portfolio returns with overlapping holdings.

    Models a staggered-portfolio strategy where positions are initiated at
    every *active rebalance date* (spaced ``rebalance_frequency`` apart within
    the dates present in ``weights_df``) and held for ``holding_period``
    rebalance periods.

    **Overlapping-position model**

    When ``holding_period > 1``, multiple positions are simultaneously active.
    At each evaluation date *t* the portfolio return is the *mean* of the
    weighted returns of all positions that are currently being held.

    **Active-position rule**

    A position entered at active rebalance date *r_i* is active at evaluation
    date *t* if and only if::

        r_i <= t < r_{i + holding_period}

    where ``r_{i + holding_period}`` is the ``(i + holding_period)``-th active
    rebalance date.  Positions entered at the last few rebalance dates (where
    ``i + holding_period >= number of active rebalances``) remain active until
    the end of the data.

    **Return column convention**

    ``returns_df`` must contain a ``value`` column (the canonical column from
    :func:`~alpha_lab.labels.forward_return` and similar).  Pass
    ``result.label_df`` or any label/return DataFrame that follows the
    canonical long-form schema.

    Parameters
    ----------
    weights_df:
        Columns ``[date, asset, weight]``.  Typically the output of
        :func:`portfolio_weights`.
    returns_df:
        Columns ``[date, asset, value]``.  ``value`` holds the per-period
        return for each ``(date, asset)`` pair.  Typically
        ``result.label_df`` from a factor experiment.
    holding_period:
        Number of *rebalance periods* to hold each position.  Must be >= 1.
    rebalance_frequency:
        Rebalance every *R* dates within the ``weights_df`` date grid.
        Must be >= 1.  ``rebalance_frequency=1`` means rebalance at every
        date where weights are available.

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, portfolio_return]``.  One row per evaluation date
        that produced at least one active position.

    Raises
    ------
    ValueError
        If required columns are missing or parameter constraints are violated.
    """
    if holding_period < 1:
        raise ValueError(f"holding_period must be >= 1, got {holding_period}")
    if rebalance_frequency < 1:
        raise ValueError(f"rebalance_frequency must be >= 1, got {rebalance_frequency}")

    for df, name, cols in [
        (weights_df, "weights_df", {"date", "asset", "weight"}),
        (returns_df, "returns_df", {"date", "asset", "value"}),
    ]:
        missing = cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    if weights_df.empty or returns_df.empty:
        return pd.DataFrame(columns=list(_PORTFOLIO_RETURN_COLUMNS))

    w_df = weights_df.copy()
    w_df["date"] = pd.to_datetime(w_df["date"])
    r_df = returns_df[["date", "asset", "value"]].copy()
    r_df["date"] = pd.to_datetime(r_df["date"])

    # Identify active rebalance dates (every rebalance_frequency steps).
    all_rebalance_dates = (
        w_df["date"].drop_duplicates().sort_values().reset_index(drop=True)
    )
    active_rebalance_dates = all_rebalance_dates.iloc[::rebalance_frequency].reset_index(
        drop=True
    )
    n_active = len(active_rebalance_dates)

    # For each active rebalance date, compute the weighted return stream over
    # the date range where that position is active.
    # position_returns[eval_date] accumulates returns from all active positions.
    position_return_sums: dict[pd.Timestamp, list[float]] = {}

    for i, r_date in enumerate(active_rebalance_dates):
        # Determine when this position expires.
        if i + holding_period < n_active:
            expiry: pd.Timestamp | None = active_rebalance_dates.iloc[i + holding_period]
        else:
            expiry = None  # Active until the end of the evaluation data.

        weights_at_r = w_df[w_df["date"] == r_date][["asset", "weight"]]
        if weights_at_r.empty:
            continue

        # Slice evaluation returns for the active window.
        if expiry is not None:
            window_returns = r_df[(r_df["date"] >= r_date) & (r_df["date"] < expiry)]
        else:
            window_returns = r_df[r_df["date"] >= r_date]

        if window_returns.empty:
            continue

        merged = window_returns.merge(weights_at_r, on="asset", how="inner")
        if merged.empty:
            continue

        # Compute weighted portfolio return at each evaluation date.
        daily_ret = (
            merged.groupby("date", sort=True)
            .apply(
                lambda g: float((g["weight"] * g["value"]).sum()),
                include_groups=False,
            )
        )

        for eval_date, ret_val in daily_ret.items():
            ts = pd.Timestamp(eval_date)
            if ts not in position_return_sums:
                position_return_sums[ts] = []
            position_return_sums[ts].append(float(ret_val))

    if not position_return_sums:
        return pd.DataFrame(columns=list(_PORTFOLIO_RETURN_COLUMNS))

    rows = [
        {"date": d, "portfolio_return": float(np.mean(vals))}
        for d, vals in sorted(position_return_sums.items())
    ]
    return pd.DataFrame(rows, columns=list(_PORTFOLIO_RETURN_COLUMNS)).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# portfolio_turnover
# ---------------------------------------------------------------------------


def portfolio_turnover(weights_df: pd.DataFrame) -> pd.DataFrame:
    """Compute portfolio turnover between consecutive rebalance dates.

    Uses the standard two-way turnover definition::

        turnover(t) = 0.5 × Σ |w_new_i − w_old_i|

    This equals the fraction of the total portfolio (by absolute weight) that
    is traded when moving from the old to the new portfolio.  A value of 0
    means no change; a value of 1 means the portfolio is fully replaced.

    **Note on sign convention**: for long-short portfolios (where some weights
    are negative), the formula correctly accounts for flips in direction.

    The first rebalance date always produces NaN — no prior portfolio state is
    available.

    Parameters
    ----------
    weights_df:
        Columns ``[date, asset, weight]``.  Typically the output of
        :func:`portfolio_weights`.

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, portfolio_turnover]``.  One row per unique date in
        ``weights_df``, sorted chronologically.

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    missing = {"date", "asset", "weight"} - set(weights_df.columns)
    if missing:
        raise ValueError(f"weights_df is missing required columns: {missing}")

    if weights_df.empty:
        return pd.DataFrame(columns=list(_PORTFOLIO_TURNOVER_COLUMNS))

    df = weights_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    dates = sorted(df["date"].unique())
    rows: list[dict[str, object]] = []
    prev_weights: dict[str, float] = {}

    for date in dates:
        curr_weights: dict[str, float] = (
            df[df["date"] == date].set_index("asset")["weight"].to_dict()
        )

        if not prev_weights:
            rows.append({"date": date, "portfolio_turnover": math.nan})
        else:
            all_assets = set(curr_weights) | set(prev_weights)
            total_abs_change = sum(
                abs(curr_weights.get(a, 0.0) - prev_weights.get(a, 0.0))
                for a in all_assets
            )
            rows.append({"date": date, "portfolio_turnover": total_abs_change / 2.0})

        prev_weights = curr_weights

    return pd.DataFrame(rows, columns=list(_PORTFOLIO_TURNOVER_COLUMNS)).reset_index(
        drop=True
    )


# ---------------------------------------------------------------------------
# portfolio_cost_adjusted_returns
# ---------------------------------------------------------------------------


def portfolio_cost_adjusted_returns(
    portfolio_return_df: pd.DataFrame,
    portfolio_turnover_df: pd.DataFrame,
    *,
    cost_rate: float,
) -> pd.DataFrame:
    """Compute cost-adjusted portfolio returns from a flat one-way transaction cost.

    Transaction costs are deducted only on dates present in
    ``portfolio_turnover_df`` (i.e. the actual rebalance dates).  Non-rebalance
    evaluation dates carry zero cost.  The first rebalance date has NaN turnover
    (no prior portfolio state), so its adjusted return is also NaN.

    The adjustment is:

        adjusted_return(t) = portfolio_return(t) − cost_rate × turnover(t)

    where ``turnover(t)`` is 0 on non-rebalance evaluation dates and NaN on the
    first rebalance date.

    **Research disclaimer**: this is a minimal friction estimate.  It does not
    model market impact, intraday slippage, bid-ask spread variation,
    short-borrow costs, execution timing, or partial fills.

    Parameters
    ----------
    portfolio_return_df:
        Columns ``[date, portfolio_return]``.  Typically the output of
        :func:`simulate_portfolio_returns`.
    portfolio_turnover_df:
        Columns ``[date, portfolio_turnover]``.  Must be restricted to the
        **active rebalance schedule** (not every evaluation date) so that
        costs are only charged when trades actually occur.  Typically the
        output of :func:`portfolio_turnover` called on weights filtered to
        active rebalance dates.
    cost_rate:
        One-way transaction cost rate (e.g. ``0.001`` for 10 bps).  Must
        be >= 0.

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, portfolio_return, adjusted_return]``.  One row per
        date in ``portfolio_return_df``.  ``adjusted_return`` is NaN on the
        first rebalance date (unknown entry cost) and equal to
        ``portfolio_return`` on non-rebalance dates (no trade cost).

    Raises
    ------
    ValueError
        If ``cost_rate < 0`` or required columns are missing.
    """
    if cost_rate < 0:
        raise ValueError(f"cost_rate must be >= 0, got {cost_rate}")

    for df, name, cols in [
        (portfolio_return_df, "portfolio_return_df", {"date", "portfolio_return"}),
        (portfolio_turnover_df, "portfolio_turnover_df", {"date", "portfolio_turnover"}),
    ]:
        missing = cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} is missing required columns: {missing}")

    if portfolio_return_df.empty:
        return pd.DataFrame(columns=list(_PORTFOLIO_COST_ADJ_COLUMNS))

    ret_df = portfolio_return_df[["date", "portfolio_return"]].copy()
    ret_df["date"] = pd.to_datetime(ret_df["date"])

    to_df = portfolio_turnover_df[["date", "portfolio_turnover"]].copy()
    to_df["date"] = pd.to_datetime(to_df["date"])

    # Left-join so every evaluation date is present.
    merged = ret_df.merge(to_df, on="date", how="left")

    # Dates not in turnover_df are non-rebalance evaluation dates: no trade, no cost.
    rebal_dates = set(to_df["date"])
    not_rebal = ~merged["date"].isin(rebal_dates)
    merged.loc[not_rebal, "portfolio_turnover"] = 0.0

    # First rebalance date has NaN turnover (no prior state) → NaN adjusted_return.
    merged["adjusted_return"] = (
        merged["portfolio_return"] - cost_rate * merged["portfolio_turnover"]
    )

    return merged[["date", "portfolio_return", "adjusted_return"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_weights(values: np.ndarray, method: str) -> np.ndarray:
    """Compute normalised positive weights from an array of factor values.

    Higher values → higher weights.  All output weights are non-negative and
    sum to 1.  Callers negate the input array to invert ordering for the
    short leg.

    Parameters
    ----------
    values:
        1-D array of finite (or NaN-free) factor values.  NaN values are not
        handled here; callers must drop NaNs before calling.
    method:
        ``"equal"``, ``"rank"``, or ``"score"``.

    Returns
    -------
    np.ndarray
        Non-negative weights summing to 1, same length as *values*.
    """
    n = len(values)
    if n == 0:
        return np.array([], dtype=float)

    if method == "equal":
        return np.full(n, 1.0 / n)

    if method == "rank":
        series = pd.Series(values)
        # ascending=True: highest value → highest rank → highest weight.
        ranks = series.rank(method="average", ascending=True).to_numpy(dtype=float)
        total = ranks.sum()
        return ranks / total if total > 0 else np.full(n, 1.0 / n)

    if method == "score":
        shifted = values - values.min()
        total = shifted.sum()
        return shifted / total if total > 0 else np.full(n, 1.0 / n)

    raise ValueError(f"Unknown weight method: {method!r}")
