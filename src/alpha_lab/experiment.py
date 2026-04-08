from __future__ import annotations

import datetime
import subprocess
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.data_validation import validate_price_panel
from alpha_lab.exceptions import AlphaLabConfigError
from alpha_lab.evaluation import compute_ic, compute_rank_ic
from alpha_lab.interfaces import validate_factor_output
from alpha_lab.labels import forward_return
from alpha_lab.quantile import long_short_return, quantile_assignments, quantile_returns
from alpha_lab.research_evaluation_config import (
    DEFAULT_RESEARCH_EVALUATION_CONFIG,
    RollingStabilityConfig,
)
from alpha_lab.research_integrity.contracts import IntegrityCheckResult, IntegrityReport
from alpha_lab.research_integrity.exceptions import raise_on_hard_failures
from alpha_lab.research_integrity.asof import pit_check
from alpha_lab.research_integrity.leakage_checks import (
    check_cross_section_transform_scope,
    check_factor_label_temporal_order,
)
from alpha_lab.research_integrity.reporting import build_integrity_report
from alpha_lab.splits import time_split
from alpha_lab.strategy import StrategySpec
from alpha_lab.turnover import long_short_turnover, quantile_turnover

DEFAULT_ROLLING_STABILITY_THRESHOLDS = DEFAULT_RESEARCH_EVALUATION_CONFIG.rolling_stability


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

    git_dirty: bool | None
    """``True`` if the working tree has uncommitted changes, ``False`` if
    clean, ``None`` if git is unavailable.  Annotates provenance so
    experiments run on a dirty tree are flagged in the audit trail."""

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

    ic_positive_rate: float
    """Fraction of finite IC observations that are strictly positive."""

    rank_ic_positive_rate: float
    """Fraction of finite RankIC observations that are strictly positive."""

    ic_valid_ratio: float
    """Finite IC observations divided by all IC rows in the evaluation period."""

    rank_ic_valid_ratio: float
    """Finite RankIC observations divided by all RankIC rows in the evaluation period."""

    long_short_ir: float
    """Information ratio of long-short return (mean / std, ddof=1)."""

    long_short_return_per_turnover: float
    """Mean long-short return divided by mean one-way long-short turnover."""

    subperiod_ic_positive_share: float
    """Share of chronological subperiods with a positive mean IC."""

    subperiod_long_short_positive_share: float
    """Share of chronological subperiods with a positive mean long-short return."""

    subperiod_ic_min_mean: float
    """Minimum subperiod mean IC across chronological subperiods."""

    subperiod_long_short_min_mean: float
    """Minimum subperiod mean long-short return across chronological subperiods."""

    rolling_window_size: int
    """Rolling stability window size (in evaluation observations)."""

    rolling_ic_positive_share: float
    """Share of rolling IC mean windows with a strictly positive mean."""

    rolling_rank_ic_positive_share: float
    """Share of rolling RankIC mean windows with a strictly positive mean."""

    rolling_long_short_positive_share: float
    """Share of rolling long-short mean windows with a strictly positive mean."""

    rolling_ic_min_mean: float
    """Worst (minimum) rolling IC mean across windows."""

    rolling_rank_ic_min_mean: float
    """Worst (minimum) rolling RankIC mean across windows."""

    rolling_long_short_min_mean: float
    """Worst (minimum) rolling long-short mean across windows."""

    rolling_instability_flags: tuple[str, ...]
    """Rolling-window-specific instability warning flags."""

    mean_eval_assets_per_date: float
    """Mean number of assets with finite factor and label values per eval date."""

    min_eval_assets_per_date: float
    """Minimum number of assets with finite factor and label values per eval date."""

    eval_coverage_ratio_mean: float
    """Mean per-date coverage ratio: valid assets / total eval assets."""

    eval_coverage_ratio_min: float
    """Minimum per-date coverage ratio: valid assets / total eval assets."""

    instability_flags: tuple[str, ...] = ()
    """Heuristic warning flags for weak/unstable diagnostic patterns."""


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

    rolling_stability_df: pd.DataFrame
    """Rolling stability diagnostics by date over the evaluation period.

    Columns:
    ``[date, rolling_mean_ic, rolling_ic_positive_rate, rolling_mean_rank_ic,
    rolling_rank_ic_positive_rate, rolling_mean_long_short_return,
    rolling_long_short_positive_rate]``.
    """

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

    integrity_checks: tuple[IntegrityCheckResult, ...] = ()
    """Structured integrity checks captured during experiment execution."""

    integrity_report: IntegrityReport | None = None
    """Aggregate integrity report object for this run."""


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
    rolling_stability_thresholds: RollingStabilityConfig = DEFAULT_ROLLING_STABILITY_THRESHOLDS,
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
    integrity_checks: list[IntegrityCheckResult] = []

    def _record_integrity(check: IntegrityCheckResult) -> None:
        integrity_checks.append(check)
        raise_on_hard_failures((check,))

    max_price_date = pd.Timestamp(pd.to_datetime(prices["date"]).max())
    _record_integrity(
        pit_check(prices, max_allowed_date=max_price_date, object_name="prices")
    )

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
        raise AlphaLabConfigError(
            "holding_period and rebalance_frequency must both be provided or "
            "both be omitted."
        )
    if holding_period is not None and holding_period < 1:
        raise AlphaLabConfigError("holding_period must be >= 1")
    if rebalance_frequency is not None and rebalance_frequency < 1:
        raise AlphaLabConfigError("rebalance_frequency must be >= 1")

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
        raise AlphaLabConfigError(
            "train_end and test_start must both be provided or both be omitted; "
            f"got train_end={train_end!r}, test_start={test_start!r}."
        )
    if val_start is not None and train_end is None:
        raise AlphaLabConfigError(
            "val_start requires both train_end and test_start to be specified."
        )

    # --- Step 1: factor values (full sample) --------------------------------
    factor_df = factor_fn(prices)
    validate_factor_output(factor_df)
    _record_integrity(
        pit_check(factor_df, max_allowed_date=max_price_date, object_name="factor_df")
    )
    _record_integrity(
        check_cross_section_transform_scope(
            prices[["date", "asset"]],
            factor_df[["date", "asset", "value"]],
            date_col="date",
            asset_col="asset",
            object_name="factor_vs_prices_scope",
        )
    )

    # --- Step 2: forward-return labels (full sample) ------------------------
    # Labels at date t: close[t+horizon]/close[t] - 1 (strictly future prices).
    # Stored at t so they merge with factor on (date, asset) without lookahead.
    label_df = forward_return(prices, horizon=horizon)
    _record_integrity(
        pit_check(label_df, max_allowed_date=max_price_date, object_name="label_df")
    )
    _record_integrity(
        check_factor_label_temporal_order(
            factor_df,
            label_df,
            join_keys=("date", "asset"),
            factor_date_col="date",
            label_date_col="date",
            object_name="factor_label_alignment",
        )
    )

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

    merged_eval = eval_factor[["date", "asset", "value"]].merge(
        eval_label[["date", "asset", "value"]].rename(
            columns={"value": "_label_value"}
        ),
        on=["date", "asset"],
        how="inner",
        validate="one_to_one",
    )
    valid_eval = merged_eval.dropna(subset=["value", "_label_value"])
    if valid_eval.empty:
        valid_assets_by_date = pd.Series(dtype=float)
    else:
        valid_assets_by_date = valid_eval.groupby("date")["asset"].nunique()
    mean_eval_assets_per_date = (
        float(valid_assets_by_date.mean())
        if len(valid_assets_by_date) > 0
        else float("nan")
    )
    min_eval_assets_per_date = (
        float(valid_assets_by_date.min())
        if len(valid_assets_by_date) > 0
        else float("nan")
    )
    if n_eval_assets > 0 and np.isfinite(mean_eval_assets_per_date):
        eval_coverage_ratio_mean = mean_eval_assets_per_date / float(n_eval_assets)
    else:
        eval_coverage_ratio_mean = float("nan")
    if n_eval_assets > 0 and np.isfinite(min_eval_assets_per_date):
        eval_coverage_ratio_min = min_eval_assets_per_date / float(n_eval_assets)
    else:
        eval_coverage_ratio_min = float("nan")

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
    rolling_stability_df = _build_rolling_stability_frame(
        ic_df,
        rank_ic_df,
        ls_df,
        window=rolling_stability_thresholds.rolling_window_size,
    )
    summary = _summarise(
        ic_df,
        rank_ic_df,
        ls_df,
        lsto_for_summary,
        rolling_stability_df=rolling_stability_df,
        mean_eval_assets_per_date=mean_eval_assets_per_date,
        min_eval_assets_per_date=min_eval_assets_per_date,
        eval_coverage_ratio_mean=eval_coverage_ratio_mean,
        eval_coverage_ratio_min=eval_coverage_ratio_min,
        rolling_stability_thresholds=rolling_stability_thresholds,
    )

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
        eval_max_date = (
            pd.Timestamp(eval_date_index.max()) if len(eval_date_index) > 0 else max_price_date
        )
        if port_weights_df is not None and not port_weights_df.empty:
            _record_integrity(
                pit_check(port_weights_df, max_allowed_date=eval_max_date, object_name="portfolio_weights_df")
            )
        if port_return_df is not None and not port_return_df.empty:
            _record_integrity(
                pit_check(port_return_df, max_allowed_date=eval_max_date, object_name="portfolio_return_df")
            )
        if port_turnover_df is not None and not port_turnover_df.empty:
            _record_integrity(
                pit_check(port_turnover_df, max_allowed_date=eval_max_date, object_name="portfolio_turnover_df")
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
        git_dirty=_is_git_dirty(),
        portfolio_cost_rate=portfolio_cost_rate,
        strategy_repr=repr(strategy) if strategy is not None else None,
    )
    integrity_report = build_integrity_report(
        tuple(integrity_checks),
        context={
            "pipeline": "run_factor_experiment",
            "horizon": horizon,
            "n_quantiles": n_quantiles,
            "train_end": str(train_end) if train_end is not None else None,
            "test_start": str(test_start) if test_start is not None else None,
            "portfolio_path_enabled": bool(
                holding_period is not None and rebalance_frequency is not None
            ),
        },
    )

    return ExperimentResult(
        factor_df=factor_df,
        label_df=label_df,
        ic_df=ic_df,
        rank_ic_df=rank_ic_df,
        quantile_returns_df=qr_df,
        long_short_df=ls_df,
        rolling_stability_df=rolling_stability_df,
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
        integrity_checks=tuple(integrity_checks),
        integrity_report=integrity_report,
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
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _is_git_dirty() -> bool | None:
    """Return True if the working tree has uncommitted changes, else False.

    Uses ``git diff --quiet HEAD`` in the package directory.  Returns
    ``None`` if git is unavailable or the call fails.
    """
    try:
        from pathlib import Path as _Path

        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"],
            capture_output=True,
            timeout=2,
            cwd=str(_Path(__file__).resolve().parent),
        )
        # Exit code 0 = clean, 1 = dirty, other = error
        if result.returncode in (0, 1):
            return result.returncode == 1
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _summarise(
    ic_df: pd.DataFrame,
    rank_ic_df: pd.DataFrame,
    ls_df: pd.DataFrame,
    ls_turnover_df: pd.DataFrame,
    *,
    rolling_stability_df: pd.DataFrame,
    mean_eval_assets_per_date: float,
    min_eval_assets_per_date: float,
    eval_coverage_ratio_mean: float,
    eval_coverage_ratio_min: float,
    rolling_stability_thresholds: RollingStabilityConfig,
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
    ic_positive_rate = (
        float((ic_vals > 0).mean()) if len(ic_vals) > 0 else float("nan")
    )
    rank_ic_positive_rate = (
        float((rank_ic_vals > 0).mean()) if len(rank_ic_vals) > 0 else float("nan")
    )
    ic_valid_ratio = (
        float(ic_df["ic"].notna().mean()) if not ic_df.empty else float("nan")
    )
    rank_ic_valid_ratio = (
        float(rank_ic_df["rank_ic"].notna().mean())
        if not rank_ic_df.empty
        else float("nan")
    )
    ls_std = float(ls_vals.std(ddof=1)) if len(ls_vals) > 1 else float("nan")
    if np.isnan(ls_std) or ls_std == 0.0:
        long_short_ir = float("nan")
    else:
        long_short_ir = mean_ls / ls_std
    if np.isnan(mean_ls_turnover) or mean_ls_turnover <= 0.0:
        long_short_return_per_turnover = float("nan")
    else:
        long_short_return_per_turnover = mean_ls / mean_ls_turnover

    (
        subperiod_ic_positive_share,
        subperiod_ic_min_mean,
    ) = _subperiod_stability_metrics(ic_df, value_col="ic")
    (
        subperiod_long_short_positive_share,
        subperiod_long_short_min_mean,
    ) = _subperiod_stability_metrics(ls_df, value_col="long_short_return")
    (
        rolling_ic_positive_share,
        rolling_ic_min_mean,
    ) = _rolling_positive_share_and_min_mean(
        rolling_stability_df,
        value_col="rolling_mean_ic",
    )
    (
        rolling_rank_ic_positive_share,
        rolling_rank_ic_min_mean,
    ) = _rolling_positive_share_and_min_mean(
        rolling_stability_df,
        value_col="rolling_mean_rank_ic",
    )
    (
        rolling_long_short_positive_share,
        rolling_long_short_min_mean,
    ) = _rolling_positive_share_and_min_mean(
        rolling_stability_df,
        value_col="rolling_mean_long_short_return",
    )
    rolling_instability_flags = _collect_rolling_instability_flags(
        rolling_stability_df,
        thresholds=rolling_stability_thresholds,
    )
    base_instability_flags = _collect_instability_flags(
        n_dates=n_dates,
        ic_positive_rate=ic_positive_rate,
        ic_valid_ratio=ic_valid_ratio,
        rank_ic_valid_ratio=rank_ic_valid_ratio,
        subperiod_ic_positive_share=subperiod_ic_positive_share,
        subperiod_long_short_positive_share=subperiod_long_short_positive_share,
        eval_coverage_ratio_mean=eval_coverage_ratio_mean,
        mean_long_short_return=mean_ls,
        mean_long_short_turnover=mean_ls_turnover,
        long_short_ir=long_short_ir,
        thresholds=rolling_stability_thresholds,
    )
    instability_flags = _merge_flags(base_instability_flags, rolling_instability_flags)

    return ExperimentSummary(
        mean_ic=mean_ic,
        mean_rank_ic=mean_rank_ic,
        ic_ir=ic_ir,
        mean_long_short_return=mean_ls,
        long_short_hit_rate=hit_rate,
        n_dates=n_dates,
        mean_long_short_turnover=mean_ls_turnover,
        ic_positive_rate=ic_positive_rate,
        rank_ic_positive_rate=rank_ic_positive_rate,
        ic_valid_ratio=ic_valid_ratio,
        rank_ic_valid_ratio=rank_ic_valid_ratio,
        long_short_ir=long_short_ir,
        long_short_return_per_turnover=long_short_return_per_turnover,
        subperiod_ic_positive_share=subperiod_ic_positive_share,
        subperiod_long_short_positive_share=subperiod_long_short_positive_share,
        subperiod_ic_min_mean=subperiod_ic_min_mean,
        subperiod_long_short_min_mean=subperiod_long_short_min_mean,
        rolling_window_size=rolling_stability_thresholds.rolling_window_size,
        rolling_ic_positive_share=rolling_ic_positive_share,
        rolling_rank_ic_positive_share=rolling_rank_ic_positive_share,
        rolling_long_short_positive_share=rolling_long_short_positive_share,
        rolling_ic_min_mean=rolling_ic_min_mean,
        rolling_rank_ic_min_mean=rolling_rank_ic_min_mean,
        rolling_long_short_min_mean=rolling_long_short_min_mean,
        rolling_instability_flags=rolling_instability_flags,
        mean_eval_assets_per_date=mean_eval_assets_per_date,
        min_eval_assets_per_date=min_eval_assets_per_date,
        eval_coverage_ratio_mean=eval_coverage_ratio_mean,
        eval_coverage_ratio_min=eval_coverage_ratio_min,
        instability_flags=instability_flags,
    )


def _build_rolling_stability_frame(
    ic_df: pd.DataFrame,
    rank_ic_df: pd.DataFrame,
    ls_df: pd.DataFrame,
    *,
    window: int,
) -> pd.DataFrame:
    """Build a compact rolling stability DataFrame across core diagnostics."""
    rolling_ic = _rolling_metric_frame(
        ic_df,
        source_col="ic",
        mean_col="rolling_mean_ic",
        positive_rate_col="rolling_ic_positive_rate",
        window=window,
    )
    rolling_rank_ic = _rolling_metric_frame(
        rank_ic_df,
        source_col="rank_ic",
        mean_col="rolling_mean_rank_ic",
        positive_rate_col="rolling_rank_ic_positive_rate",
        window=window,
    )
    rolling_ls = _rolling_metric_frame(
        ls_df,
        source_col="long_short_return",
        mean_col="rolling_mean_long_short_return",
        positive_rate_col="rolling_long_short_positive_rate",
        window=window,
    )

    out = rolling_ic.merge(rolling_rank_ic, on="date", how="outer").merge(
        rolling_ls,
        on="date",
        how="outer",
    )
    if out.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "rolling_mean_ic",
                "rolling_ic_positive_rate",
                "rolling_mean_rank_ic",
                "rolling_rank_ic_positive_rate",
                "rolling_mean_long_short_return",
                "rolling_long_short_positive_rate",
            ]
        )
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.sort_values("date", kind="mergesort").reset_index(drop=True)


def _rolling_metric_frame(
    frame: pd.DataFrame,
    *,
    source_col: str,
    mean_col: str,
    positive_rate_col: str,
    window: int,
) -> pd.DataFrame:
    """Compute rolling mean and rolling positive-rate for one metric series."""
    if frame.empty or source_col not in frame.columns:
        return pd.DataFrame(columns=["date", mean_col, positive_rate_col])

    finite = frame.loc[frame[source_col].notna(), ["date", source_col]].copy()
    if finite.empty:
        return pd.DataFrame(columns=["date", mean_col, positive_rate_col])

    finite["date"] = pd.to_datetime(finite["date"], errors="coerce")
    finite[source_col] = pd.to_numeric(finite[source_col], errors="coerce")
    finite = finite.dropna(subset=["date", source_col])
    if finite.empty:
        return pd.DataFrame(columns=["date", mean_col, positive_rate_col])

    finite = finite.sort_values("date", kind="mergesort").reset_index(drop=True)
    finite[mean_col] = finite[source_col].rolling(window=window, min_periods=window).mean()
    finite[positive_rate_col] = (
        (finite[source_col] > 0).astype(float).rolling(window=window, min_periods=window).mean()
    )
    return finite[["date", mean_col, positive_rate_col]]


def _rolling_positive_share_and_min_mean(
    frame: pd.DataFrame,
    *,
    value_col: str,
) -> tuple[float, float]:
    """Return (positive-share, minimum) for a rolling-mean metric column."""
    if frame.empty or value_col not in frame.columns:
        return float("nan"), float("nan")
    vals = pd.to_numeric(frame[value_col], errors="coerce").dropna()
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float((vals > 0).mean()), float(vals.min())


def _collect_rolling_instability_flags(
    rolling_stability_df: pd.DataFrame,
    *,
    thresholds: RollingStabilityConfig = DEFAULT_ROLLING_STABILITY_THRESHOLDS,
) -> tuple[str, ...]:
    """Derive compact rolling-window instability flags."""
    metric_specs = (
        ("rolling_mean_ic", "rolling_ic"),
        ("rolling_mean_rank_ic", "rolling_rank_ic"),
        ("rolling_mean_long_short_return", "rolling_long_short"),
    )

    flags: list[str] = []
    regime_dependent = False
    for value_col, prefix in metric_specs:
        if value_col not in rolling_stability_df.columns:
            continue
        values = pd.to_numeric(
            rolling_stability_df[value_col],
            errors="coerce",
        ).dropna()
        if len(values) == 0:
            continue

        positive_share = float((values > 0).mean())
        if positive_share < thresholds.rolling_regime_min_positive_share:
            flags.append(f"{prefix}_below_zero_share_high")
            regime_dependent = True

        sign_flip_rate = _sign_flip_rate(values.to_numpy(dtype=float))
        if (
            np.isfinite(sign_flip_rate)
            and len(values) >= thresholds.rolling_regime_min_windows_for_sign_flip
            and sign_flip_rate > thresholds.rolling_regime_sign_flip_threshold
        ):
            flags.append(f"{prefix}_sign_flip_instability")
            regime_dependent = True

    if regime_dependent:
        flags.append("rolling_regime_dependence")
    return tuple(flags)


def _sign_flip_rate(values: np.ndarray) -> float:
    """Return adjacent sign-flip rate for a numeric series (ignoring zeros)."""
    if len(values) < 2:
        return float("nan")
    signs = np.sign(values)
    signs = signs[signs != 0]
    if len(signs) < 2:
        return float("nan")
    changes = int(np.sum(signs[1:] != signs[:-1]))
    return float(changes / (len(signs) - 1))


def _merge_flags(*groups: tuple[str, ...]) -> tuple[str, ...]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for flag in group:
            if flag not in seen:
                seen.add(flag)
                merged.append(flag)
    return tuple(merged)


def _subperiod_stability_metrics(
    frame: pd.DataFrame,
    *,
    value_col: str,
    n_subperiods: int = 3,
) -> tuple[float, float]:
    """Return (positive-share, min-subperiod-mean) over chronological subperiods."""
    if frame.empty or value_col not in frame.columns:
        return float("nan"), float("nan")

    finite = frame.loc[frame[value_col].notna(), ["date", value_col]].copy()
    if len(finite) < n_subperiods:
        return float("nan"), float("nan")

    finite["date"] = pd.to_datetime(finite["date"])
    finite = finite.sort_values("date", kind="mergesort").reset_index(drop=True)
    idx_chunks = np.array_split(np.arange(len(finite)), n_subperiods)
    means: list[float] = []
    for chunk in idx_chunks:
        if len(chunk) == 0:
            continue
        mean_val = float(finite.iloc[chunk][value_col].mean())
        if np.isfinite(mean_val):
            means.append(mean_val)
    if not means:
        return float("nan"), float("nan")

    positive_share = float(np.mean(np.array(means) > 0))
    min_mean = float(np.min(means))
    return positive_share, min_mean


def _collect_instability_flags(
    *,
    n_dates: int,
    ic_positive_rate: float,
    ic_valid_ratio: float,
    rank_ic_valid_ratio: float,
    subperiod_ic_positive_share: float,
    subperiod_long_short_positive_share: float,
    eval_coverage_ratio_mean: float,
    mean_long_short_return: float,
    mean_long_short_turnover: float,
    long_short_ir: float,
    thresholds: RollingStabilityConfig = DEFAULT_ROLLING_STABILITY_THRESHOLDS,
) -> tuple[str, ...]:
    flags: list[str] = []
    if n_dates < thresholds.instability_short_eval_window_dates:
        flags.append("short_eval_window")
    if np.isfinite(ic_valid_ratio) and ic_valid_ratio < thresholds.instability_ic_valid_ratio_min:
        flags.append("low_ic_valid_ratio")
    if (
        np.isfinite(rank_ic_valid_ratio)
        and rank_ic_valid_ratio < thresholds.instability_rank_ic_valid_ratio_min
    ):
        flags.append("low_rank_ic_valid_ratio")
    if (
        np.isfinite(ic_positive_rate)
        and ic_positive_rate < thresholds.instability_ic_positive_rate_min
    ):
        flags.append("ic_sign_instability")
    if (
        np.isfinite(subperiod_ic_positive_share)
        and subperiod_ic_positive_share < thresholds.instability_subperiod_positive_share_min
    ):
        flags.append("ic_subperiod_instability")
    if (
        np.isfinite(subperiod_long_short_positive_share)
        and subperiod_long_short_positive_share
        < thresholds.instability_subperiod_positive_share_min
    ):
        flags.append("long_short_subperiod_instability")
    if (
        np.isfinite(eval_coverage_ratio_mean)
        and eval_coverage_ratio_mean < thresholds.instability_eval_coverage_ratio_mean_min
    ):
        flags.append("thin_universe_coverage")
    if (
        np.isfinite(mean_long_short_turnover)
        and mean_long_short_turnover > thresholds.instability_high_turnover
        and np.isfinite(mean_long_short_return)
        and mean_long_short_return
        <= thresholds.instability_high_turnover_negative_spread_max_return
    ):
        flags.append("high_turnover_negative_spread")
    if np.isfinite(long_short_ir) and long_short_ir < thresholds.instability_long_short_ir_min:
        flags.append("negative_long_short_ir")
    return tuple(flags)


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
