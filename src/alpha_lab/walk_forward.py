from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from alpha_lab.costs import cost_adjusted_long_short
from alpha_lab.experiment import ExperimentResult, run_factor_experiment
from alpha_lab.splits import walk_forward_split
from alpha_lab.strategy import StrategySpec

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

_FOLD_SUMMARY_COLUMNS: tuple[str, ...] = (
    "fold_id",
    "train_start",
    "train_end",
    "start_date",
    "end_date",
    "mean_ic",
    "mean_rank_ic",
    "ic_ir",
    "mean_long_short",
    "mean_turnover",
    "mean_cost_adjusted_return",
    # Portfolio path — NaN when holding_period/rebalance_frequency not provided.
    "mean_portfolio_return",
    "portfolio_hit_rate",
    "mean_portfolio_turnover",
    "mean_cost_adjusted_portfolio_return",
)


@dataclass(frozen=True)
class WalkForwardAggregate:
    """Aggregate statistics across all walk-forward folds.

    All ``mean_*`` fields are means-of-means: the mean of the per-fold metric
    averaged across all folds with a finite value.  ``std_*`` fields are the
    sample standard deviation across folds (ddof=1; NaN when fewer than two
    folds have finite values).
    """

    n_folds: int
    """Number of folds generated."""

    mean_ic: float
    """Mean of per-fold mean IC."""

    std_ic: float
    """Std across folds of per-fold mean IC."""

    mean_ic_ir: float
    """Mean of per-fold IC-IR."""

    std_ic_ir: float
    """Std across folds of per-fold IC-IR."""

    mean_long_short: float
    """Mean of per-fold mean long-short return."""

    std_long_short: float
    """Std across folds of per-fold mean long-short return."""

    mean_turnover: float
    """Mean of per-fold mean long-short turnover."""

    std_turnover: float
    """Std across folds of per-fold mean long-short turnover."""

    mean_cost_adjusted_return: float
    """Mean of per-fold mean cost-adjusted L/S return.  NaN when cost_rate was
    not provided or all folds produced NaN."""

    std_cost_adjusted_return: float
    """Std across folds of per-fold mean cost-adjusted L/S return."""

    best_fold: int
    """``fold_id`` of the fold with the highest ``mean_ic``."""

    worst_fold: int
    """``fold_id`` of the fold with the lowest ``mean_ic``."""

    pooled_ic_mean: float
    """Mean IC across all pooled fold observations.

    More statistically sound than :attr:`mean_ic` (which is a mean-of-fold-means)
    when fold sizes differ.  Computed directly from the concatenated IC series
    across all folds.
    """

    pooled_ic_std: float
    """Std of IC across all pooled fold observations (ddof=1)."""

    pooled_ic_ir: float
    """Pooled IC-IR: :attr:`pooled_ic_mean` / :attr:`pooled_ic_std`.
    NaN when fewer than two observations or std is zero.
    """

    n_ic_obs: int
    """Total number of finite IC observations across all folds."""

    mean_portfolio_return: float
    """Mean of per-fold mean portfolio return.  NaN when ``holding_period`` and
    ``rebalance_frequency`` were not provided to :func:`run_walk_forward_experiment`."""

    std_portfolio_return: float
    """Std across folds of per-fold mean portfolio return."""

    portfolio_hit_rate: float
    """Mean of per-fold portfolio hit rate (fraction of dates with positive return).
    NaN when portfolio simulation was not enabled."""

    mean_portfolio_turnover: float
    """Mean of per-fold mean portfolio turnover on active rebalance dates."""

    mean_cost_adjusted_portfolio_return: float
    """Mean of per-fold mean cost-adjusted portfolio return.  NaN when
    ``portfolio_cost_rate`` was not provided."""

    std_cost_adjusted_portfolio_return: float
    """Std across folds of per-fold mean cost-adjusted portfolio return."""

    pooled_portfolio_return_mean: float
    """Mean portfolio return across all pooled fold observations.

    Analogous to :attr:`pooled_ic_mean`: computed directly from the
    concatenated portfolio-return series rather than as a mean-of-fold-means.
    NaN when portfolio simulation was not enabled or produced no finite values.
    """

    pooled_portfolio_return_std: float
    """Std of portfolio return across all pooled fold observations (ddof=1)."""

    pooled_portfolio_hit_rate: float
    """Fraction of all pooled portfolio-return observations that are > 0.
    NaN when portfolio simulation was not enabled.
    """

    n_portfolio_obs: int
    """Total number of finite portfolio-return observations across all folds."""

    pooled_cost_adjusted_return_mean: float
    """Mean cost-adjusted portfolio return across all pooled fold observations.

    Computed from the ``adjusted_return`` column of the concatenated
    :attr:`WalkForwardResult.pooled_cost_adjusted_portfolio_return_df`.
    NaN when ``portfolio_cost_rate`` was not provided or produced no finite
    values.
    """

    pooled_cost_adjusted_return_std: float
    """Std of cost-adjusted portfolio return across all pooled observations
    (ddof=1).  NaN when fewer than two finite values are available."""

    n_cost_adjusted_obs: int
    """Total number of finite cost-adjusted portfolio-return observations
    across all folds (NaN first-rebalance dates are excluded)."""

    pooled_portfolio_turnover_mean: float
    """Mean two-way portfolio turnover across all pooled fold observations.

    Computed from the concatenated
    :attr:`WalkForwardResult.pooled_portfolio_turnover_df` (active rebalance
    dates only; first rebalance date per fold is NaN and excluded).
    NaN when portfolio simulation was not enabled or produced no finite values.
    """


@dataclass
class WalkForwardResult:
    """Full output of :func:`run_walk_forward_experiment`.

    ``per_fold_results`` contains one :class:`~alpha_lab.experiment.ExperimentResult`
    per fold; their ordering matches ``fold_summary_df``.  Each result is
    restricted to its own test window — there is no shared state between folds.
    """

    per_fold_results: list[ExperimentResult]
    """Per-fold experiment results, ordered by fold_id."""

    fold_summary_df: pd.DataFrame
    """One-row-per-fold summary DataFrame.

    Core columns: ``fold_id, train_start, train_end, start_date, end_date,
    mean_ic, mean_rank_ic, ic_ir, mean_long_short, mean_turnover,
    mean_cost_adjusted_return``.

    Portfolio columns (NaN when portfolio simulation is not enabled):
    ``mean_portfolio_return, portfolio_hit_rate, mean_portfolio_turnover,
    mean_cost_adjusted_portfolio_return``.

    **Note on val_size**: ``train_start`` and ``train_end`` reflect the full
    training window *including* the trailing validation buffer when
    ``val_size > 0``.  No validation-period metrics are surfaced — the
    validation dates are excluded from both training and test evaluation and
    serve only as a construction-side gap.
    """

    aggregate_summary: WalkForwardAggregate
    """Aggregate statistics across all folds."""

    pooled_ic_df: pd.DataFrame
    """Concatenated IC observations from every fold.

    Columns: ``[fold_id, date, ic]``.  Use this for distribution analysis
    across the full OOS evaluation span — e.g. to visualise the empirical
    IC density or run statistical tests that require the raw series.
    """

    pooled_portfolio_return_df: pd.DataFrame
    """Concatenated portfolio-return observations from every fold.

    Columns: ``[fold_id, date, portfolio_return]``.  Empty DataFrame when
    ``holding_period`` / ``rebalance_frequency`` were not provided to
    :func:`run_walk_forward_experiment`.  Analogous to :attr:`pooled_ic_df`
    but for the portfolio simulation path.
    """

    pooled_cost_adjusted_portfolio_return_df: pd.DataFrame
    """Concatenated cost-adjusted portfolio-return observations from every fold.

    Columns: ``[fold_id, date, portfolio_return, adjusted_return]``.
    ``portfolio_return`` is the gross return (before cost deduction);
    ``adjusted_return`` is the net return after deducting
    ``cost_rate × turnover`` on active rebalance dates.  ``adjusted_return``
    is NaN on the first active rebalance date of each fold (no prior portfolio
    state is available).

    Empty DataFrame when ``portfolio_cost_rate`` was not provided or portfolio
    simulation was not enabled.  Every row in this DataFrame is from a fold
    test window — there is no IS or non-OOS contamination.
    """

    pooled_portfolio_turnover_df: pd.DataFrame
    """Concatenated portfolio-turnover observations from every fold.

    Columns: ``[fold_id, date, portfolio_turnover]``.  Restricted to active
    rebalance dates (every ``rebalance_frequency``-th weight date per fold);
    the first rebalance date of each fold is NaN (no prior portfolio state).

    Empty DataFrame when portfolio simulation was not enabled.  Symmetric
    with :attr:`pooled_portfolio_return_df` and
    :attr:`pooled_cost_adjusted_portfolio_return_df` for the turnover path.
    """


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_walk_forward_experiment(
    prices: pd.DataFrame,
    factor_fn: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    train_size: int,
    test_size: int,
    step: int,
    horizon: int = 1,
    n_quantiles: int = 5,
    cost_rate: float | None = None,
    val_size: int = 0,
    holding_period: int | None = None,
    rebalance_frequency: int | None = None,
    weighting_method: str = "equal",
    portfolio_cost_rate: float | None = None,
    strategy: StrategySpec | None = None,
) -> WalkForwardResult:
    """Run a factor experiment over rolling walk-forward folds.

    Uses :func:`~alpha_lab.splits.walk_forward_split` to partition the
    date axis into non-overlapping test windows.  For each fold the factor
    is evaluated on the test window only, guaranteeing strict out-of-sample
    evaluation with no data leakage between folds.

    **Temporal discipline**

    - ``factor_fn`` receives prices filtered to each fold's visible window
      (up to and including ``test_end``).  Factor values are computed at
      every date in that window, but evaluation metrics are restricted to the
      fold's test window.  This is correct for look-back factors (e.g.
      momentum) that require historical context: the factor value at date *t*
      uses only prices up to *t* by construction.
    - ``factor_fn`` must be a **deterministic, stateless** callable.  It
      must not fit model parameters on the training window or carry state
      between fold calls.  For factors that require per-fold fitting, the
      caller must wrap the fitting logic inside ``factor_fn`` using only the
      prices passed to it.
    - Test windows are strictly non-overlapping and temporally ordered.
    - Labels at test date *t* use ``close[t + horizon] / close[t] - 1``,
      where ``t + horizon`` references strictly future prices.  The same
      timestamp discipline as :func:`~alpha_lab.experiment.run_factor_experiment`
      applies.

    Parameters
    ----------
    prices:
        Long-form price panel with columns ``[date, asset, close]``.
    factor_fn:
        Callable ``(prices) -> factor_df``.  Must return canonical
        ``[date, asset, factor, value]`` output.
    train_size:
        Number of unique dates in the training window.
    test_size:
        Number of unique dates in the test window.
    step:
        Number of unique dates to advance the window between folds.
    horizon:
        Forward-return look-ahead in per-asset rows.
    n_quantiles:
        Number of quantile buckets.
    cost_rate:
        Optional one-way transaction cost rate.  When provided,
        ``mean_cost_adjusted_return`` is computed per fold.
    val_size:
        Number of trailing dates within the training window reserved for
        validation (passed through to :func:`~alpha_lab.splits.walk_forward_split`).
        These dates are excluded from both train and test evaluation.

        **Current limitation**: ``val_size`` is a fold-construction parameter
        only.  No validation-period outputs (IC, quantile metrics, etc.) are
        produced for the validation window.  Set this to 0 (the default)
        unless you are using it to create a gap between training and test
        windows to avoid label overlap.
    holding_period:
        Optional number of rebalance periods to hold each portfolio position.
        Must be provided together with ``rebalance_frequency``.  When both are
        given, portfolio weights, returns, and turnover are computed for each
        fold and surfaced in ``fold_summary_df`` and
        :attr:`WalkForwardAggregate`.
    rebalance_frequency:
        Optional rebalance interval in dates.  Must be provided together with
        ``holding_period``.
    weighting_method:
        Weight method passed to
        :func:`~alpha_lab.portfolio_research.portfolio_weights`.
        One of ``"equal"``, ``"rank"``, ``"score"``.  Ignored unless both
        ``holding_period`` and ``rebalance_frequency`` are provided.
    portfolio_cost_rate:
        Optional one-way transaction cost rate for the portfolio path.
        When provided together with ``holding_period`` / ``rebalance_frequency``,
        ``mean_cost_adjusted_portfolio_return`` is computed per fold and
        pooled cost-adjusted observations are surfaced in
        :attr:`WalkForwardResult.pooled_cost_adjusted_portfolio_return_df`.
    strategy:
        Optional :class:`~alpha_lab.strategy.StrategySpec` that explicitly
        specifies portfolio construction intent.  When provided, it overrides
        ``holding_period``, ``rebalance_frequency``, and ``weighting_method``
        (a ``UserWarning`` is raised if those are also passed explicitly); it
        also controls ``long_top_k`` / ``short_bottom_k`` for the weight-based
        portfolio path.  ``n_quantiles`` and ``portfolio_cost_rate`` are not
        part of the strategy spec and must be provided separately.

    Returns
    -------
    WalkForwardResult

    Raises
    ------
    ValueError
        If ``prices`` contains NaT dates, if the parameters produce no folds,
        or if any argument constraint is violated.
    """
    # --- Validate inputs ----------------------------------------------------
    if train_size <= 0 or test_size <= 0 or step <= 0:
        raise ValueError("train_size, test_size, and step must be positive integers")
    if cost_rate is not None and cost_rate < 0:
        raise ValueError("cost_rate must be >= 0")
    if strategy is not None:
        if holding_period is not None or rebalance_frequency is not None:
            warnings.warn(
                "holding_period and rebalance_frequency are ignored when strategy is "
                "provided.  Set them via StrategySpec instead.",
                UserWarning,
                stacklevel=2,
            )
    elif (holding_period is None) != (rebalance_frequency is None):
        raise ValueError(
            "holding_period and rebalance_frequency must both be provided or both be omitted."
        )
    if portfolio_cost_rate is not None and portfolio_cost_rate < 0:
        raise ValueError("portfolio_cost_rate must be >= 0")

    # Warn when portfolio_cost_rate is supplied but portfolio mode is not active.
    _portfolio_active = (strategy is not None) or (holding_period is not None)
    if portfolio_cost_rate is not None and not _portfolio_active:
        warnings.warn(
            "portfolio_cost_rate is ignored because portfolio simulation is not "
            "enabled.  Provide holding_period and rebalance_frequency (or a "
            "StrategySpec) to enable the portfolio path.",
            UserWarning,
            stacklevel=2,
        )

    # --- Extract unique sorted dates ----------------------------------------
    dates_raw = pd.to_datetime(prices["date"])
    if dates_raw.isna().any():
        raise ValueError("prices contains NaT in 'date'")

    unique_dates = dates_raw.drop_duplicates().sort_values().reset_index(drop=True)

    # walk_forward_split requires unique sorted dates
    splits = walk_forward_split(
        unique_dates,
        train_size=train_size,
        test_size=test_size,
        step=step,
        val_size=val_size,
    )

    if not splits:
        raise ValueError(
            f"No walk-forward folds generated with train_size={train_size}, "
            f"test_size={test_size}, step={step} on {len(unique_dates)} unique dates.  "
            "Increase the dataset size or reduce train_size/test_size."
        )

    # --- Run per-fold experiments -------------------------------------------
    per_fold_results: list[ExperimentResult] = []
    fold_rows: list[dict[str, object]] = []
    pooled_ic_parts: list[pd.DataFrame] = []
    pooled_port_ret_parts: list[pd.DataFrame] = []
    pooled_port_to_parts: list[pd.DataFrame] = []
    pooled_cost_adj_parts: list[pd.DataFrame] = []

    for fold_id, masks in enumerate(splits):
        train_mask: pd.Series = pd.Series(masks["train"])
        test_mask: pd.Series = pd.Series(masks["test"])

        train_dates = unique_dates[train_mask.to_numpy()]
        test_dates = unique_dates[test_mask.to_numpy()]

        train_start_ts: pd.Timestamp = train_dates.iloc[0]
        train_end_ts: pd.Timestamp = train_dates.iloc[-1]
        test_start_ts: pd.Timestamp = test_dates.iloc[0]
        test_end_ts: pd.Timestamp = test_dates.iloc[-1]

        # Restrict prices to this fold's visible window (up to and including
        # test_end_ts) so that factor_fn cannot access future price data beyond
        # this fold's test period.  This makes each fold strictly independent.
        fold_prices = prices[
            pd.to_datetime(prices["date"]) <= test_end_ts
        ].reset_index(drop=True)

        result = run_factor_experiment(
            fold_prices,
            factor_fn,
            horizon=horizon,
            n_quantiles=n_quantiles,
            train_end=train_end_ts,
            test_start=test_start_ts,
            holding_period=holding_period,
            rebalance_frequency=rebalance_frequency,
            weighting_method=weighting_method,
            portfolio_cost_rate=portfolio_cost_rate,
            strategy=strategy,
        )
        per_fold_results.append(result)

        # Accumulate IC observations for the pooled OOS series.
        if not result.ic_df.empty:
            fold_ic = result.ic_df[["date", "ic"]].copy()
            fold_ic.insert(0, "fold_id", fold_id)
            pooled_ic_parts.append(fold_ic)

        # Accumulate portfolio-return observations for the pooled OOS series.
        if result.portfolio_return_df is not None and not result.portfolio_return_df.empty:
            fold_port = result.portfolio_return_df[["date", "portfolio_return"]].copy()
            fold_port.insert(0, "fold_id", fold_id)
            pooled_port_ret_parts.append(fold_port)

        # Accumulate portfolio-turnover observations (active rebalance dates only).
        if result.portfolio_turnover_df is not None and not result.portfolio_turnover_df.empty:
            fold_to = result.portfolio_turnover_df[["date", "portfolio_turnover"]].copy()
            fold_to.insert(0, "fold_id", fold_id)
            pooled_port_to_parts.append(fold_to)

        # Accumulate cost-adjusted portfolio-return observations (OOS only).
        # portfolio_cost_adjusted_return_df columns: [date, portfolio_return, adjusted_return]
        # portfolio_return = gross return (before cost); adjusted_return = net return.
        if (
            result.portfolio_cost_adjusted_return_df is not None
            and not result.portfolio_cost_adjusted_return_df.empty
        ):
            fold_cost_adj = result.portfolio_cost_adjusted_return_df[
                ["date", "portfolio_return", "adjusted_return"]
            ].copy()
            fold_cost_adj.insert(0, "fold_id", fold_id)
            pooled_cost_adj_parts.append(fold_cost_adj)

        # Compute long-short cost-adjusted return for this fold if requested.
        if cost_rate is not None and not result.long_short_df.empty:
            adj_df = cost_adjusted_long_short(
                result.long_short_df,
                result.long_short_turnover_df,
                cost_rate=cost_rate,
            )
            adj_vals = adj_df["adjusted_return"].dropna()
            mean_cost_adj: float = float(adj_vals.mean()) if len(adj_vals) > 0 else math.nan
        else:
            mean_cost_adj = math.nan

        # Extract per-fold portfolio summary (None when portfolio params omitted).
        ps = result.portfolio_summary
        s = result.summary
        fold_rows.append(
            {
                "fold_id": fold_id,
                "train_start": train_start_ts,
                "train_end": train_end_ts,
                "start_date": test_start_ts,
                "end_date": test_end_ts,
                "mean_ic": s.mean_ic,
                "mean_rank_ic": s.mean_rank_ic,
                "ic_ir": s.ic_ir,
                "mean_long_short": s.mean_long_short_return,
                "mean_turnover": s.mean_long_short_turnover,
                "mean_cost_adjusted_return": mean_cost_adj,
                "mean_portfolio_return": ps.mean_portfolio_return if ps else math.nan,
                "portfolio_hit_rate": ps.portfolio_hit_rate if ps else math.nan,
                "mean_portfolio_turnover": ps.mean_portfolio_turnover if ps else math.nan,
                "mean_cost_adjusted_portfolio_return": (
                    ps.mean_cost_adjusted_return if ps else math.nan
                ),
            }
        )

    fold_summary_df = pd.DataFrame(fold_rows, columns=list(_FOLD_SUMMARY_COLUMNS))
    pooled_ic_df = (
        pd.concat(pooled_ic_parts, ignore_index=True)
        if pooled_ic_parts
        else pd.DataFrame(columns=["fold_id", "date", "ic"])
    )
    pooled_portfolio_return_df = (
        pd.concat(pooled_port_ret_parts, ignore_index=True)
        if pooled_port_ret_parts
        else pd.DataFrame(columns=["fold_id", "date", "portfolio_return"])
    )
    pooled_cost_adjusted_portfolio_return_df = (
        pd.concat(pooled_cost_adj_parts, ignore_index=True)
        if pooled_cost_adj_parts
        else pd.DataFrame(
            columns=["fold_id", "date", "portfolio_return", "adjusted_return"]
        )
    )
    pooled_portfolio_turnover_df = (
        pd.concat(pooled_port_to_parts, ignore_index=True)
        if pooled_port_to_parts
        else pd.DataFrame(columns=["fold_id", "date", "portfolio_turnover"])
    )
    aggregate = _compute_aggregate(
        fold_summary_df,
        pooled_ic_df,
        pooled_portfolio_return_df,
        pooled_cost_adjusted_portfolio_return_df,
        pooled_portfolio_turnover_df,
    )

    return WalkForwardResult(
        per_fold_results=per_fold_results,
        fold_summary_df=fold_summary_df,
        aggregate_summary=aggregate,
        pooled_ic_df=pooled_ic_df,
        pooled_portfolio_return_df=pooled_portfolio_return_df,
        pooled_cost_adjusted_portfolio_return_df=pooled_cost_adjusted_portfolio_return_df,
        pooled_portfolio_turnover_df=pooled_portfolio_turnover_df,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_aggregate(
    fold_df: pd.DataFrame,
    pooled_ic_df: pd.DataFrame,
    pooled_portfolio_return_df: pd.DataFrame,
    pooled_cost_adjusted_portfolio_return_df: pd.DataFrame,
    pooled_portfolio_turnover_df: pd.DataFrame,
) -> WalkForwardAggregate:
    """Compute aggregate statistics from the fold summary, pooled IC, pooled
    portfolio-return, pooled cost-adjusted portfolio-return, and pooled
    portfolio-turnover DataFrames."""

    def _mean(col: str) -> float:
        vals = fold_df[col].dropna()
        return float(vals.mean()) if len(vals) > 0 else math.nan

    def _std(col: str) -> float:
        vals = fold_df[col].dropna()
        return float(vals.std(ddof=1)) if len(vals) > 1 else math.nan

    ic_vals = fold_df["mean_ic"].dropna()
    if len(ic_vals) > 0:
        best_fold = int(
            fold_df.loc[fold_df["mean_ic"] == ic_vals.max(), "fold_id"].iloc[0]
        )
        worst_fold = int(
            fold_df.loc[fold_df["mean_ic"] == ic_vals.min(), "fold_id"].iloc[0]
        )
    else:
        # All NaN — fall back to first and last fold ids.
        best_fold = int(fold_df["fold_id"].iloc[0])
        worst_fold = int(fold_df["fold_id"].iloc[-1])

    # Pooled IC statistics across all fold observations.
    pooled_vals = (
        pooled_ic_df["ic"].dropna()
        if not pooled_ic_df.empty and "ic" in pooled_ic_df.columns
        else pd.Series(dtype=float)
    )
    n_ic_obs = len(pooled_vals)
    pooled_ic_mean = float(pooled_vals.mean()) if n_ic_obs > 0 else math.nan
    pooled_ic_std = float(pooled_vals.std(ddof=1)) if n_ic_obs > 1 else math.nan
    if math.isnan(pooled_ic_std) or pooled_ic_std == 0.0:
        pooled_ic_ir = math.nan
    else:
        pooled_ic_ir = pooled_ic_mean / pooled_ic_std

    # Pooled portfolio-return statistics across all fold observations.
    port_vals = (
        pooled_portfolio_return_df["portfolio_return"].dropna()
        if not pooled_portfolio_return_df.empty
        and "portfolio_return" in pooled_portfolio_return_df.columns
        else pd.Series(dtype=float)
    )
    n_portfolio_obs = len(port_vals)
    pooled_port_mean = float(port_vals.mean()) if n_portfolio_obs > 0 else math.nan
    pooled_port_std = float(port_vals.std(ddof=1)) if n_portfolio_obs > 1 else math.nan
    pooled_port_hit = float((port_vals > 0).mean()) if n_portfolio_obs > 0 else math.nan

    # Pooled cost-adjusted portfolio-return statistics.
    # adjusted_return is NaN on the first rebalance date of each fold;
    # dropna() correctly excludes those entries.
    cost_adj_vals = (
        pooled_cost_adjusted_portfolio_return_df["adjusted_return"].dropna()
        if not pooled_cost_adjusted_portfolio_return_df.empty
        and "adjusted_return" in pooled_cost_adjusted_portfolio_return_df.columns
        else pd.Series(dtype=float)
    )
    n_cost_adjusted_obs = len(cost_adj_vals)
    pooled_cost_adj_mean = (
        float(cost_adj_vals.mean()) if n_cost_adjusted_obs > 0 else math.nan
    )
    pooled_cost_adj_std = (
        float(cost_adj_vals.std(ddof=1)) if n_cost_adjusted_obs > 1 else math.nan
    )

    # Pooled portfolio-turnover statistics (active rebalance dates only).
    to_vals = (
        pooled_portfolio_turnover_df["portfolio_turnover"].dropna()
        if not pooled_portfolio_turnover_df.empty
        and "portfolio_turnover" in pooled_portfolio_turnover_df.columns
        else pd.Series(dtype=float)
    )
    pooled_to_mean = float(to_vals.mean()) if len(to_vals) > 0 else math.nan

    return WalkForwardAggregate(
        n_folds=len(fold_df),
        mean_ic=_mean("mean_ic"),
        std_ic=_std("mean_ic"),
        mean_ic_ir=_mean("ic_ir"),
        std_ic_ir=_std("ic_ir"),
        mean_long_short=_mean("mean_long_short"),
        std_long_short=_std("mean_long_short"),
        mean_turnover=_mean("mean_turnover"),
        std_turnover=_std("mean_turnover"),
        mean_cost_adjusted_return=_mean("mean_cost_adjusted_return"),
        std_cost_adjusted_return=_std("mean_cost_adjusted_return"),
        best_fold=best_fold,
        worst_fold=worst_fold,
        pooled_ic_mean=pooled_ic_mean,
        pooled_ic_std=pooled_ic_std,
        pooled_ic_ir=pooled_ic_ir,
        n_ic_obs=n_ic_obs,
        mean_portfolio_return=_mean("mean_portfolio_return"),
        std_portfolio_return=_std("mean_portfolio_return"),
        portfolio_hit_rate=_mean("portfolio_hit_rate"),
        mean_portfolio_turnover=_mean("mean_portfolio_turnover"),
        mean_cost_adjusted_portfolio_return=_mean("mean_cost_adjusted_portfolio_return"),
        std_cost_adjusted_portfolio_return=_std("mean_cost_adjusted_portfolio_return"),
        pooled_portfolio_return_mean=pooled_port_mean,
        pooled_portfolio_return_std=pooled_port_std,
        pooled_portfolio_hit_rate=pooled_port_hit,
        n_portfolio_obs=n_portfolio_obs,
        pooled_cost_adjusted_return_mean=pooled_cost_adj_mean,
        pooled_cost_adjusted_return_std=pooled_cost_adj_std,
        n_cost_adjusted_obs=n_cost_adjusted_obs,
        pooled_portfolio_turnover_mean=pooled_to_mean,
    )
