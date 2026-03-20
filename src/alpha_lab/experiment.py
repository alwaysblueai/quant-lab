from __future__ import annotations

import datetime
import subprocess
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.data_validation import validate_price_panel
from alpha_lab.evaluation import compute_ic, compute_rank_ic
from alpha_lab.interfaces import validate_factor_output
from alpha_lab.labels import forward_return
from alpha_lab.quantile import long_short_return, quantile_assignments, quantile_returns
from alpha_lab.splits import time_split
from alpha_lab.strategy import StrategySpec
from alpha_lab.turnover import long_short_turnover, quantile_turnover


@dataclass(frozen=True)
class ExperimentProvenance:
    """Minimal provenance record for one :func:`run_factor_experiment` call.

    Captured automatically at run time.  Stored on
    :attr:`ExperimentResult.provenance` so every result object carries its own
    audit trail without requiring a separate registry entry.
    """

    factor_name: str
    """Name of the factor as reported in the factor DataFrame."""

    horizon: int
    """Forward-return horizon (per-asset rows) used to compute labels."""

    n_quantiles: int
    """Number of quantile buckets used in the factor evaluation path."""

    run_timestamp_utc: str
    """ISO-8601 UTC timestamp of when the experiment was executed."""

    git_commit: str | None
    """Short git commit hash of the current HEAD, or ``None`` if the project
    is not in a git repository or git is unavailable."""

    portfolio_cost_rate: float | None
    """One-way transaction cost rate passed to the experiment, or ``None``."""

    strategy_repr: str | None
    """``repr()`` of the :class:`~alpha_lab.strategy.StrategySpec` passed to
    the experiment, or ``None`` if no strategy was provided."""


@dataclass(frozen=True)
class ExperimentSummary:
    """Scalar summary of one experiment (or one split thereof).

    All metrics are computed over the evaluation period.  Fields are NaN
    when the evaluation period contains insufficient data.
    """

    mean_ic: float
    """Cross-sectional Pearson IC averaged across evaluation dates."""

    mean_rank_ic: float
    """Cross-sectional Spearman RankIC averaged across evaluation dates."""

    ic_ir: float
    """Information ratio: mean_ic / std_ic (ddof=1).  NaN when fewer than
    two non-NaN IC observations are available or std_ic == 0."""

    mean_long_short_return: float
    """Average top-minus-bottom quantile return across evaluation dates."""

    long_short_hit_rate: float
    """Fraction of evaluation dates on which the long-short return was > 0."""

    n_dates: int
    """Number of distinct evaluation dates that produced a finite IC value."""

    mean_long_short_turnover: float
    """Average one-way long-short turnover across evaluation dates.

    Defined as the mean of :func:`~alpha_lab.turnover.long_short_turnover`
    over all dates with a finite value.  NaN when no finite turnover values
    are available (e.g. fewer than two evaluation dates).
    """


@dataclass(frozen=True)
class PortfolioSummary:
    """Scalar summary of the portfolio simulation path inside one experiment.

    Populated only when ``holding_period`` and ``rebalance_frequency`` are
    provided to :func:`run_factor_experiment`.  All fields are NaN when the
    corresponding series is empty or all-NaN.

    This is a companion to :class:`ExperimentSummary`: the latter covers the
    IC / quantile / long-short evaluation path; this covers the portfolio
    weight / staggered-return / turnover path.
    """

    mean_portfolio_return: float
    """Mean of :attr:`ExperimentResult.portfolio_return_df` over all evaluation
    dates with a finite value."""

    portfolio_hit_rate: float
    """Fraction of evaluation dates on which ``portfolio_return > 0``."""

    mean_portfolio_turnover: float
    """Mean two-way portfolio turnover across active rebalance dates (NaN
    entries — the first rebalance date — are excluded from the average)."""

    mean_cost_adjusted_return: float
    """Mean of ``adjusted_return`` from
    :attr:`ExperimentResult.portfolio_cost_adjusted_return_df`.  NaN when
    ``portfolio_cost_rate`` was not supplied to :func:`run_factor_experiment`
    or all adjusted returns are NaN.
    """

    n_portfolio_dates: int
    """Number of evaluation dates with a finite ``portfolio_return``."""


@dataclass
class ExperimentResult:
    """Full output of one :func:`run_factor_experiment` call.

    ``factor_df`` and ``label_df`` always cover the **full sample** so the
    caller can inspect the complete time-series.  All other DataFrames and
    ``summary`` are restricted to the **evaluation period** (test split when
    a split is requested, full sample otherwise).

    ``n_quantiles``, ``train_end``, and ``test_start`` carry the exact
    parameters used to produce this result so downstream consumers (e.g.
    reporting utilities) can label results accurately without the caller
    repeating them.
    """

    factor_df: pd.DataFrame
    """Canonical long-form factor output for the full sample."""

    label_df: pd.DataFrame
    """Canonical long-form forward-return labels for the full sample."""

    ic_df: pd.DataFrame
    """Pearson IC by date over the evaluation period."""

    rank_ic_df: pd.DataFrame
    """Spearman RankIC by date over the evaluation period."""

    quantile_returns_df: pd.DataFrame
    """Mean return per quantile bucket and date over the evaluation period."""

    long_short_df: pd.DataFrame
    """Long-short (top minus bottom quantile) return by date."""

    summary: ExperimentSummary
    """Scalar summary metrics for the evaluation period."""

    n_quantiles: int
    """Number of quantile buckets requested in the experiment."""

    train_end: pd.Timestamp | None
    """Last date (inclusive) of the training period, or None when no split was requested."""

    test_start: pd.Timestamp | None
    """First date (inclusive) of the evaluation period, or None when no split was requested."""

    quantile_assignments_df: pd.DataFrame
    """Per-asset quantile bucket assignments over the evaluation period.

    Columns: ``[date, asset, factor, quantile]``.  Universe is all
    ``(date, asset)`` pairs with a non-NaN factor value, which may extend
    slightly beyond the :attr:`quantile_returns_df` universe (the latter
    excludes dates where forward-return labels are NaN).
    """

    quantile_turnover_df: pd.DataFrame
    """Period-to-period one-way turnover per quantile bucket.

    Columns: ``[date, factor, quantile, turnover]``.  First evaluation date
    is always NaN (no prior portfolio state).
    """

    long_short_turnover_df: pd.DataFrame
    """Long-short one-way turnover by date.

    Columns: ``[date, factor, long_short_turnover]``.  Average of top and
    bottom bucket turnover; NaN on the first evaluation date.
    """

    provenance: ExperimentProvenance
    """Minimal audit record: factor name, horizon, quantiles, run timestamp,
    git commit, cost rate, and strategy repr.  Captured automatically."""

    n_eval_dates: int
    """Number of distinct dates in the evaluation period."""

    n_eval_assets: int
    """Number of distinct assets in the evaluation period."""

    n_label_nan_dates: int
    """Number of evaluation dates for which **no** valid (non-NaN) forward
    return label existed for any asset.  These dates are excluded from IC and
    quantile-return computation.  Typically equals the label ``horizon``
    (the last ``horizon`` dates have no future price to compute a return)."""

    portfolio_weights_df: pd.DataFrame | None = None
    """Portfolio weights by date and asset when portfolio parameters are
    provided to :func:`run_factor_experiment`.

    Columns: ``[date, asset, weight]``.  ``None`` when ``holding_period`` and
    ``rebalance_frequency`` are not supplied.
    """

    portfolio_return_df: pd.DataFrame | None = None
    """Simulated portfolio return by date when portfolio parameters are
    provided to :func:`run_factor_experiment`.

    Columns: ``[date, portfolio_return]``.  ``None`` when ``holding_period``
    and ``rebalance_frequency`` are not supplied.
    """

    portfolio_turnover_df: pd.DataFrame | None = None
    """Portfolio two-way turnover on **active rebalance dates** only.

    Columns: ``[date, portfolio_turnover]``.  Restricted to the same
    active rebalance schedule used by
    :func:`~alpha_lab.portfolio_research.simulate_portfolio_returns`
    (i.e. every ``rebalance_frequency``-th weight date).  First active
    rebalance date is always NaN (no prior portfolio state).  ``None``
    when ``holding_period`` and ``rebalance_frequency`` are not supplied.
    """

    portfolio_cost_adjusted_return_df: pd.DataFrame | None = None
    """Cost-adjusted portfolio returns when ``portfolio_cost_rate`` is provided.

    Columns: ``[date, portfolio_return, adjusted_return]``.  ``adjusted_return``
    deducts ``portfolio_cost_rate × turnover`` on each active rebalance date
    and is NaN on the first rebalance date.  ``None`` when
    ``portfolio_cost_rate`` is not supplied or portfolio simulation is not
    enabled.
    """

    portfolio_summary: PortfolioSummary | None = None
    """Scalar summary of the portfolio simulation path.

    Populated whenever ``holding_period`` and ``rebalance_frequency`` are
    supplied to :func:`run_factor_experiment`.  ``None`` otherwise.
    """


def run_factor_experiment(
    prices: pd.DataFrame,
    factor_fn: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    horizon: int = 1,
    n_quantiles: int = 5,
    train_end: str | pd.Timestamp | None = None,
    test_start: str | pd.Timestamp | None = None,
    val_start: str | pd.Timestamp | None = None,
    holding_period: int | None = None,
    rebalance_frequency: int | None = None,
    weighting_method: str = "equal",
    portfolio_cost_rate: float | None = None,
    strategy: StrategySpec | None = None,
) -> ExperimentResult:
    """Run a factor experiment end-to-end.

    **Timestamp discipline**

    - Factor values are computed by ``factor_fn`` and must only use
      information available at or before their observation date.  The runner
      validates the canonical schema but cannot verify the internals of
      ``factor_fn`` — that responsibility rests with the factor author.
    - Labels are ``forward_return(prices, horizon=horizon)``.  The label
      stored at date *t* equals ``close[t + horizon] / close[t] - 1``,
      where ``t + horizon`` is measured in per-asset row count.  The label
      value uses strictly future prices; the label is *stored* at *t* so it
      can be merged with the factor on ``(date, asset)`` without lookahead.
    - Evaluation metrics are computed on the **test period** when a split is
      provided, or on the full sample otherwise.  Training-period dates are
      never included in IC or quantile metrics.

    Parameters
    ----------
    prices:
        Long-form price panel with columns ``[date, asset, close]``.
    factor_fn:
        Callable that accepts a price panel and returns a canonical
        ``[date, asset, factor, value]`` DataFrame.  Must produce exactly
        one factor name.
    horizon:
        Forward-return look-ahead in per-asset rows.
    n_quantiles:
        Number of quantile buckets for :func:`~alpha_lab.quantile.quantile_returns`.
    train_end:
        Last date (inclusive) of the training period.  Provide together
        with ``test_start`` to restrict evaluation to the test period.
    test_start:
        First date (inclusive) of the evaluation period.
    val_start:
        Optional start of a validation window between ``train_end`` and
        ``test_start``; passed through to :func:`~alpha_lab.splits.time_split`.
    holding_period:
        Optional number of rebalance periods to hold each portfolio position.
        Must be provided together with ``rebalance_frequency``.  When both are
        provided, :func:`~alpha_lab.portfolio_research.portfolio_weights` and
        :func:`~alpha_lab.portfolio_research.simulate_portfolio_returns` are
        called and results are attached to
        :attr:`ExperimentResult.portfolio_weights_df` and
        :attr:`ExperimentResult.portfolio_return_df`.
    rebalance_frequency:
        Optional rebalance interval in dates.  Must be provided together with
        ``holding_period``.
    weighting_method:
        Weight method passed to :func:`~alpha_lab.portfolio_research.portfolio_weights`.
        One of ``"equal"``, ``"rank"``, ``"score"``.  Ignored unless both
        ``holding_period`` and ``rebalance_frequency`` are provided.
    portfolio_cost_rate:
        Optional one-way transaction cost rate for portfolio simulation
        (e.g. ``0.001`` for 10 bps).  When provided together with
        ``holding_period`` / ``rebalance_frequency``, populates
        :attr:`ExperimentResult.portfolio_cost_adjusted_return_df`.
        Must be >= 0.  Ignored when portfolio simulation is not enabled.
    strategy:
        Optional :class:`~alpha_lab.strategy.StrategySpec` that explicitly
        specifies portfolio construction intent.  When provided, it overrides
        ``holding_period``, ``rebalance_frequency``, and ``weighting_method``
        (a ``UserWarning`` is raised if those are also passed explicitly); it
        also supplies ``long_top_k`` and ``short_bottom_k`` to
        :func:`~alpha_lab.portfolio_research.portfolio_weights` so that asset
        selection is explicit rather than implicit.  ``n_quantiles`` and
        ``portfolio_cost_rate`` are not part of the strategy spec — they remain
        separate parameters.

    Returns
    -------
    ExperimentResult
    """
    # --- Step -1: validate raw price panel ----------------------------------
    # Catch bad input early before any downstream computation occurs.  This
    # also validates programmatic callers, not just the CLI path.
    validate_price_panel(prices)

    # --- Step 0: resolve strategy overrides ---------------------------------
    # StrategySpec is the explicit domain boundary between the factor research
    # layer and the portfolio research layer.  When provided it overrides all
    # individual portfolio construction parameters so that construction intent
    # is expressed in one place rather than scattered across call sites.
    if strategy is not None:
        if holding_period is not None or rebalance_frequency is not None:
            warnings.warn(
                "holding_period and rebalance_frequency are ignored when strategy is "
                "provided.  Set them via StrategySpec instead.",
                UserWarning,
                stacklevel=2,
            )
        holding_period = strategy.holding_period
        rebalance_frequency = strategy.rebalance_frequency
        weighting_method = strategy.weighting_method

    # --- Step 0b: validate portfolio arguments --------------------------------
    if (holding_period is None) != (rebalance_frequency is None):
        raise ValueError(
            "holding_period and rebalance_frequency must both be provided or "
            "both be omitted."
        )
    if holding_period is not None and holding_period < 1:
        raise ValueError("holding_period must be >= 1")
    if rebalance_frequency is not None and rebalance_frequency < 1:
        raise ValueError("rebalance_frequency must be >= 1")

    # Warn when portfolio_cost_rate is supplied but portfolio mode is not active.
    # After strategy override, holding_period is None iff portfolio mode is off.
    if portfolio_cost_rate is not None and holding_period is None:
        warnings.warn(
            "portfolio_cost_rate is ignored because portfolio simulation is not "
            "enabled.  Provide holding_period and rebalance_frequency (or a "
            "StrategySpec) to enable the portfolio path.",
            UserWarning,
            stacklevel=2,
        )

    # --- Step 0b: validate split arguments ----------------------------------
    # Require both or neither.  A lone train_end / test_start, or a val_start
    # without a complete split, silently evaluates on the full sample — that is
    # a dangerous misuse path and must be an explicit error.
    if (train_end is None) != (test_start is None):
        raise ValueError(
            "train_end and test_start must both be provided or both be omitted; "
            f"got train_end={train_end!r}, test_start={test_start!r}."
        )
    if val_start is not None and train_end is None:
        raise ValueError(
            "val_start requires both train_end and test_start to be specified."
        )

    # --- Step 1: factor values (full sample) --------------------------------
    factor_df = factor_fn(prices)
    validate_factor_output(factor_df)

    # --- Step 2: forward-return labels (full sample) ------------------------
    # Labels at date t: close[t+horizon]/close[t] - 1 (strictly future prices).
    # Stored at t so they merge with factor on (date, asset) without lookahead.
    label_df = forward_return(prices, horizon=horizon)

    # --- Step 3: resolve evaluation period ----------------------------------
    if train_end is not None and test_start is not None:
        # time_split is date-comparison-based and is panel-safe: all rows
        # sharing a date receive the same mask value.
        masks = time_split(
            factor_df["date"],
            train_end=train_end,
            test_start=test_start,
            val_start=val_start,
        )
        eval_factor = factor_df[masks["test"]].reset_index(drop=True)
        # Filter labels by the exact test-period dates (not by positional mask,
        # since label_df may have a different row structure than factor_df).
        eval_date_index = pd.DatetimeIndex(eval_factor["date"].unique())
        eval_label = label_df[label_df["date"].isin(eval_date_index)].reset_index(
            drop=True
        )
    else:
        eval_factor = factor_df.copy()
        eval_label = label_df.copy()
        eval_date_index = pd.DatetimeIndex(eval_factor["date"].unique())

    # --- Step 3b: diagnostics -----------------------------------------------
    n_eval_dates = int(eval_factor["date"].nunique())
    n_eval_assets = int(eval_factor["asset"].nunique())
    # Count eval dates that have no valid (non-NaN) forward return label for
    # any asset.  These are excluded from IC and quantile-return computation.
    dates_with_labels: set[object] = set(
        eval_label.loc[eval_label["value"].notna(), "date"].unique()
    )
    eval_factor_dates: set[object] = set(eval_factor["date"].unique())
    n_label_nan_dates = len(eval_factor_dates - dates_with_labels)

    # --- Step 4: IC / RankIC -----------------------------------------------
    ic_df = compute_ic(eval_factor, eval_label)
    rank_ic_df = compute_rank_ic(eval_factor, eval_label)

    # --- Step 5: quantile returns and long-short ----------------------------
    qr_df = quantile_returns(eval_factor, eval_label, n_quantiles=n_quantiles)
    ls_df = long_short_return(qr_df)

    # --- Step 5b: portfolio assignments and turnover ------------------------
    # Assignments are computed from eval_factor only (no label required).
    # The universe may include the last `horizon` dates where labels are NaN —
    # those dates still represent real rebalancing events.
    asgn_df = quantile_assignments(eval_factor, n_quantiles=n_quantiles)
    qto_df = quantile_turnover(asgn_df)
    lsto_df = long_short_turnover(qto_df)

    # --- Step 6: summary ----------------------------------------------------
    # Restrict turnover to dates present in long_short_df so that
    # mean_long_short_turnover is averaged over the same universe as
    # mean_long_short_return and cost-adjusted returns.
    ls_dates = set(ls_df["date"].unique()) if not ls_df.empty else set()
    lsto_for_summary = (
        lsto_df[lsto_df["date"].isin(ls_dates)] if not lsto_df.empty else lsto_df
    )
    summary = _summarise(ic_df, rank_ic_df, ls_df, lsto_for_summary)

    # --- Step 7: optional portfolio simulation ------------------------------
    port_weights_df: pd.DataFrame | None = None
    port_return_df: pd.DataFrame | None = None
    port_turnover_df: pd.DataFrame | None = None
    port_cost_adj_df: pd.DataFrame | None = None
    port_summary: PortfolioSummary | None = None

    if holding_period is not None and rebalance_frequency is not None:
        port_weights_df, port_return_df, port_turnover_df, port_cost_adj_df = (
            _run_portfolio_block(
                eval_factor=eval_factor,
                prices=prices,
                eval_date_index=eval_date_index,
                holding_period=holding_period,
                rebalance_frequency=rebalance_frequency,
                weighting_method=weighting_method,
                portfolio_cost_rate=portfolio_cost_rate,
                strategy=strategy,
            )
        )
        port_summary = _summarise_portfolio(
            port_return_df, port_turnover_df, port_cost_adj_df
        )

    # --- Step 8: build provenance -------------------------------------------
    _factor_names = factor_df["factor"].unique() if not factor_df.empty else []
    _factor_name_str = str(_factor_names[0]) if len(_factor_names) > 0 else "unknown"
    prov = ExperimentProvenance(
        factor_name=_factor_name_str,
        horizon=horizon,
        n_quantiles=n_quantiles,
        run_timestamp_utc=_utc_now(),
        git_commit=_get_git_commit(),
        portfolio_cost_rate=portfolio_cost_rate,
        strategy_repr=repr(strategy) if strategy is not None else None,
    )

    return ExperimentResult(
        factor_df=factor_df,
        label_df=label_df,
        ic_df=ic_df,
        rank_ic_df=rank_ic_df,
        quantile_returns_df=qr_df,
        long_short_df=ls_df,
        summary=summary,
        n_quantiles=n_quantiles,
        train_end=pd.Timestamp(train_end) if train_end is not None else None,
        test_start=pd.Timestamp(test_start) if test_start is not None else None,
        quantile_assignments_df=asgn_df,
        quantile_turnover_df=qto_df,
        long_short_turnover_df=lsto_df,
        provenance=prov,
        n_eval_dates=n_eval_dates,
        n_eval_assets=n_eval_assets,
        n_label_nan_dates=n_label_nan_dates,
        portfolio_weights_df=port_weights_df,
        portfolio_return_df=port_return_df,
        portfolio_turnover_df=port_turnover_df,
        portfolio_cost_adjusted_return_df=port_cost_adj_df,
        portfolio_summary=port_summary,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utc_now() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _get_git_commit() -> str | None:
    """Return the short git commit hash of the current HEAD, or ``None``.

    Uses ``git rev-parse --short HEAD`` in the package directory.  Returns
    ``None`` if git is not available, the project is not in a repository, or
    the subprocess call fails for any reason.  Failures are intentionally
    silent — a missing commit hash is not a hard error.
    """
    try:
        from pathlib import Path

        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=str(Path(__file__).resolve().parent),
        )
        if result.returncode == 0:
            return result.stdout.strip() or None
    except Exception:
        pass
    return None


def _summarise(
    ic_df: pd.DataFrame,
    rank_ic_df: pd.DataFrame,
    ls_df: pd.DataFrame,
    ls_turnover_df: pd.DataFrame,
) -> ExperimentSummary:
    """Compute scalar summary metrics from per-date evaluation DataFrames."""
    ic_vals = ic_df["ic"].dropna() if not ic_df.empty else pd.Series(dtype=float)
    rank_ic_vals = (
        rank_ic_df["rank_ic"].dropna()
        if not rank_ic_df.empty
        else pd.Series(dtype=float)
    )
    ls_vals = (
        ls_df["long_short_return"].dropna()
        if not ls_df.empty
        else pd.Series(dtype=float)
    )
    ls_turn_vals = (
        ls_turnover_df["long_short_turnover"].dropna()
        if not ls_turnover_df.empty
        else pd.Series(dtype=float)
    )

    mean_ic = float(ic_vals.mean()) if len(ic_vals) > 0 else float("nan")
    mean_rank_ic = float(rank_ic_vals.mean()) if len(rank_ic_vals) > 0 else float("nan")

    # ic_ir = mean_ic / std_ic  (ddof=1; NaN when std unavailable or zero)
    ic_std = float(ic_vals.std(ddof=1)) if len(ic_vals) > 1 else float("nan")
    if np.isnan(ic_std) or ic_std == 0.0:
        ic_ir: float = float("nan")
    else:
        ic_ir = mean_ic / ic_std

    mean_ls = float(ls_vals.mean()) if len(ls_vals) > 0 else float("nan")
    hit_rate = float((ls_vals > 0).mean()) if len(ls_vals) > 0 else float("nan")
    # n_dates: dates with a finite IC value (matches the denominator of mean_ic and ic_ir).
    n_dates = (
        int(ic_df.loc[ic_df["ic"].notna(), "date"].nunique())
        if not ic_df.empty
        else 0
    )
    mean_ls_turnover = (
        float(ls_turn_vals.mean()) if len(ls_turn_vals) > 0 else float("nan")
    )

    return ExperimentSummary(
        mean_ic=mean_ic,
        mean_rank_ic=mean_rank_ic,
        ic_ir=ic_ir,
        mean_long_short_return=mean_ls,
        long_short_hit_rate=hit_rate,
        n_dates=n_dates,
        mean_long_short_turnover=mean_ls_turnover,
    )


def _summarise_portfolio(
    portfolio_return_df: pd.DataFrame,
    portfolio_turnover_df: pd.DataFrame,
    portfolio_cost_adjusted_df: pd.DataFrame | None,
) -> PortfolioSummary:
    """Compute scalar portfolio summary metrics from per-date simulation output."""
    ret_vals = portfolio_return_df["portfolio_return"].dropna()
    mean_return = float(ret_vals.mean()) if len(ret_vals) > 0 else float("nan")
    hit_rate = float((ret_vals > 0).mean()) if len(ret_vals) > 0 else float("nan")
    n_dates = len(ret_vals)

    turn_vals = portfolio_turnover_df["portfolio_turnover"].dropna()
    mean_turnover = float(turn_vals.mean()) if len(turn_vals) > 0 else float("nan")

    if portfolio_cost_adjusted_df is not None:
        adj_vals = portfolio_cost_adjusted_df["adjusted_return"].dropna()
        mean_cost_adj: float = float(adj_vals.mean()) if len(adj_vals) > 0 else float("nan")
    else:
        mean_cost_adj = float("nan")

    return PortfolioSummary(
        mean_portfolio_return=mean_return,
        portfolio_hit_rate=hit_rate,
        mean_portfolio_turnover=mean_turnover,
        mean_cost_adjusted_return=mean_cost_adj,
        n_portfolio_dates=n_dates,
    )


def _run_portfolio_block(
    eval_factor: pd.DataFrame,
    prices: pd.DataFrame,
    eval_date_index: pd.DatetimeIndex,
    holding_period: int,
    rebalance_frequency: int,
    weighting_method: str,
    portfolio_cost_rate: float | None,
    strategy: StrategySpec | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Run the optional portfolio simulation block inside :func:`run_factor_experiment`.

    Keeps portfolio weight, return, turnover, and cost-adjusted return
    computation isolated from the main experiment pipeline so that
    :func:`run_factor_experiment` does not grow further as new portfolio
    metrics are added.

    When ``strategy`` is provided, asset selection (``long_top_k`` /
    ``short_bottom_k``) and weight method are taken from the spec via
    :func:`~alpha_lab.strategy.portfolio_weights_from_strategy`, making
    construction intent explicit.  When ``strategy`` is ``None``, the
    existing ``weighting_method`` parameter is used with no asset-count
    constraints (all assets in the long leg, no short leg) — the same
    default behaviour as before the strategy layer was introduced.

    Returns
    -------
    tuple of (weights_df, return_df, turnover_df, cost_adjusted_df | None)
        - ``weights_df``: ``[date, asset, weight]``
        - ``return_df``: ``[date, portfolio_return]``
        - ``turnover_df``: ``[date, portfolio_turnover]`` on active rebalance dates only
        - ``cost_adjusted_df``: ``[date, portfolio_return, adjusted_return]`` or ``None``
    """
    from alpha_lab.portfolio_research import (
        portfolio_cost_adjusted_returns,
        portfolio_turnover,
        portfolio_weights,
        simulate_portfolio_returns,
    )

    if strategy is not None:
        from alpha_lab.strategy import portfolio_weights_from_strategy

        port_weights_df = portfolio_weights_from_strategy(eval_factor, strategy)
    else:
        port_weights_df = portfolio_weights(eval_factor, method=weighting_method)

    # Use 1-period step returns for the simulation so that each position held
    # for ``holding_period`` periods contributes one period's P&L per step.
    # H-period forward returns (used for IC/quantile evaluation) would
    # incorrectly compound in the staggered-portfolio model when
    # holding_period > 1.
    one_period_labels = forward_return(prices, horizon=1)
    eval_1p = one_period_labels[
        one_period_labels["date"].isin(eval_date_index)
    ].reset_index(drop=True)

    port_return_df = simulate_portfolio_returns(
        port_weights_df,
        eval_1p,
        holding_period=holding_period,
        rebalance_frequency=rebalance_frequency,
    )

    # Restrict turnover to the active rebalance schedule — the same subset of
    # dates used by simulate_portfolio_returns().  When rebalance_frequency > 1,
    # computing turnover on all weight dates would include non-trade dates and
    # overstate transaction costs.
    all_weight_dates = (
        pd.to_datetime(port_weights_df["date"])
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    active_rebal_dates = set(all_weight_dates.iloc[::rebalance_frequency])
    port_weights_active = port_weights_df[
        pd.to_datetime(port_weights_df["date"]).isin(active_rebal_dates)
    ].reset_index(drop=True)
    port_turnover_df = portfolio_turnover(port_weights_active)

    if portfolio_cost_rate is not None:
        port_cost_adj_df: pd.DataFrame | None = portfolio_cost_adjusted_returns(
            port_return_df,
            port_turnover_df,
            cost_rate=portfolio_cost_rate,
        )
    else:
        port_cost_adj_df = None

    return port_weights_df, port_return_df, port_turnover_df, port_cost_adj_df
