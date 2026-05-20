from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from alpha_lab.baseline_factor_suite import (
    baseline_required_columns_available,
    iter_baseline_factor_specs,
)
from alpha_lab.data_quality.tradability import (
    apply_untradable_mask_to_labels,
    detect_limit_moves,
    summarise_tradability,
)
from alpha_lab.evaluation import (
    compute_mean_rank_ic_permutation_null,
)
from alpha_lab.factor_recipe import build_factor_from_recipe_mapping
from alpha_lab.labels import forward_return
from alpha_lab.marginal_contribution import compute_marginal_contribution
from alpha_lab.reporting.neutralization_comparison import (
    NeutralizationComparison,
    build_raw_vs_neutralized_comparison,
)
from alpha_lab.research_evaluation_config import (
    NeutralizationComparisonConfig,
    ResearchEvaluationConfig,
)
from alpha_lab.splits import TimeSeriesSplitContract
from alpha_lab.validation.haircut_sharpe import haircut_sharpe_ratio

# Cross-module imports (auto-added by split)
from ._utils import (
    _evaluate_variant_lightweight,
    _finite_or_nan,
    _parse_flags,
    _parse_optional_float,
    _parse_optional_int,
    _resolve_variant_label_df,
)


def _build_neutralization_comparison(
    *,
    raw_row: pd.Series,
    neutralized_row: pd.Series,
    neutralization_mean_corr_reduction: float,
    thresholds: NeutralizationComparisonConfig,
) -> NeutralizationComparison:
    raw_metrics = _comparison_metrics_from_summary_row(raw_row)
    neutralized_metrics = _comparison_metrics_from_summary_row(neutralized_row)
    return build_raw_vs_neutralized_comparison(
        raw_metrics,
        neutralized_metrics,
        neutralization_mean_corr_reduction=(
            neutralization_mean_corr_reduction
            if np.isfinite(neutralization_mean_corr_reduction)
            else None
        ),
        thresholds=thresholds,
    )


def _comparison_metrics_from_summary_row(row: pd.Series) -> dict[str, object]:
    return {
        "mean_ic": float(row["mean_ic"]),
        "mean_rank_ic": float(row["mean_rank_ic"]),
        "mean_long_short_return": float(row["mean_long_short_return"]),
        "ic_ir": float(row["ic_ir"]),
        "ic_valid_ratio": float(row["ic_valid_ratio"]),
        "rank_ic_valid_ratio": float(row["rank_ic_valid_ratio"]),
        "eval_coverage_ratio_mean": float(row["eval_coverage_ratio_mean"]),
        "eval_coverage_ratio_min": float(row["eval_coverage_ratio_min"]),
        "rolling_ic_positive_share": float(row["rolling_ic_positive_share"]),
        "rolling_rank_ic_positive_share": float(row["rolling_rank_ic_positive_share"]),
        "rolling_long_short_positive_share": float(row["rolling_long_short_positive_share"]),
        "rolling_ic_min_mean": float(row["rolling_ic_min_mean"]),
        "rolling_rank_ic_min_mean": float(row["rolling_rank_ic_min_mean"]),
        "rolling_long_short_min_mean": float(row["rolling_long_short_min_mean"]),
        "mean_ic_ci_lower": float(row["mean_ic_ci_lower"]),
        "mean_ic_ci_upper": float(row["mean_ic_ci_upper"]),
        "mean_rank_ic_ci_lower": float(row["mean_rank_ic_ci_lower"]),
        "mean_rank_ic_ci_upper": float(row["mean_rank_ic_ci_upper"]),
        "mean_long_short_return_ci_lower": float(row["mean_long_short_return_ci_lower"]),
        "mean_long_short_return_ci_upper": float(row["mean_long_short_return_ci_upper"]),
        "uncertainty_flags": _parse_flags(row["uncertainty_flags"]),
        "rolling_instability_flags": _parse_flags(row["rolling_instability_flags"]),
    }


def _merge_tradability_metrics(
    metrics: dict[str, object],
    *,
    prices: pd.DataFrame,
    factor_df: pd.DataFrame,
    label_df: pd.DataFrame,
    horizon: int,
    n_quantiles: int,
    evaluation_config: ResearchEvaluationConfig,
    cost_rate: float,
    enabled: bool = True,
) -> None:
    """Flag rows whose next-day execution is blocked (limit-up/down, suspended)
    and re-run the evaluation with those labels blanked to quantify the
    tradability leakage.

    The delta fields surface how much of the headline IC / long-short return is
    attributable to days that could never have been traded in practice — the
    single largest silent leakage source for short-horizon price-volume
    factors.
    """
    counts = _empty_tradability_summary()
    metrics.update(counts)
    metrics["tradability_filtered_mean_ic"] = float("nan")
    metrics["tradability_filtered_mean_long_short_return"] = float("nan")
    metrics["tradability_ic_delta"] = float("nan")
    metrics["tradability_long_short_return_delta"] = float("nan")
    metrics["tradability_filter_applied"] = False
    if not enabled:
        return
    try:
        flagged = detect_limit_moves(prices)
        counts.update(summarise_tradability(flagged))
        metrics.update(counts)
    except Exception:
        return

    untradable_rows = _parse_optional_int(counts.get("tradability_untradable_rows"))
    if untradable_rows is None or untradable_rows <= 0:
        return

    try:
        masked_labels = apply_untradable_mask_to_labels(
            label_df,
            prices,
            direction="both",
        )
    except Exception:
        return

    try:
        filtered_summary = _evaluate_variant_lightweight(
            factor_df=factor_df,
            label_df=masked_labels[["date", "asset", "factor", "value"]],
            n_quantiles=n_quantiles,
        )
    except Exception:
        return

    filtered_mean_ic = float(filtered_summary.mean_ic)
    filtered_mean_ls = float(filtered_summary.mean_long_short_return)
    base_mean_ic = _finite_or_nan(_parse_optional_float(metrics.get("mean_ic")))
    base_mean_ls = _finite_or_nan(_parse_optional_float(metrics.get("mean_long_short_return")))

    metrics["tradability_filter_applied"] = True
    metrics["tradability_filtered_mean_ic"] = filtered_mean_ic
    metrics["tradability_filtered_mean_long_short_return"] = filtered_mean_ls
    if np.isfinite(filtered_mean_ic) and np.isfinite(base_mean_ic):
        metrics["tradability_ic_delta"] = filtered_mean_ic - base_mean_ic
    if np.isfinite(filtered_mean_ls) and np.isfinite(base_mean_ls):
        metrics["tradability_long_short_return_delta"] = filtered_mean_ls - base_mean_ls


def _empty_tradability_summary() -> dict[str, object]:
    return {
        "tradability_total_rows": 0,
        "tradability_limit_up_rows": 0,
        "tradability_limit_down_rows": 0,
        "tradability_untradable_rows": 0,
        "tradability_untradable_rate": float("nan"),
    }


def _merge_execution_price_sensitivity_metrics(
    metrics: dict[str, object],
    *,
    prices: pd.DataFrame,
    factor_df: pd.DataFrame,
    horizon: int,
    n_quantiles: int,
    evaluation_config: ResearchEvaluationConfig,
    cost_rate: float,
    split_contract: TimeSeriesSplitContract | None = None,
    enabled: bool = True,
) -> None:
    """Re-run the evaluation with next-open execution prices and record the
    delta vs the default close-to-close convention.

    Price-volume factors that look strong at close-to-close but disappear
    under next-open execution are capturing the signal-day close itself —
    a bar the real-world trader cannot transact on.  The delta fields here
    make that failure mode visible without requiring the caller to re-run
    the pipeline manually.
    """
    metrics["next_open_execution_available"] = False
    metrics["next_open_mean_ic"] = float("nan")
    metrics["next_open_mean_long_short_return"] = float("nan")
    metrics["next_open_long_short_ir"] = float("nan")
    metrics["next_open_mean_ic_delta"] = float("nan")
    metrics["next_open_mean_long_short_return_delta"] = float("nan")
    if not enabled:
        return

    if "open" not in prices.columns:
        return

    try:
        next_open_label_df = forward_return(
            prices,
            horizon=horizon,
            execution_price_mode="next_open",
        )
        if split_contract is not None:
            oos_start = pd.Timestamp(split_contract.oos_start)
            factor_dates = pd.to_datetime(factor_df["date"], errors="coerce")
            label_dates = pd.to_datetime(next_open_label_df["date"], errors="coerce")
            factor_for_variant = factor_df.loc[factor_dates >= oos_start].reset_index(drop=True)
            label_for_variant = next_open_label_df.loc[label_dates >= oos_start].reset_index(
                drop=True
            )
        else:
            factor_for_variant = factor_df
            label_for_variant = next_open_label_df
        next_open_summary = _evaluate_variant_lightweight(
            factor_df=factor_for_variant,
            label_df=label_for_variant,
            n_quantiles=n_quantiles,
        )
    except Exception:
        return

    next_open_mean_ic = float(next_open_summary.mean_ic)
    next_open_mean_ls = float(next_open_summary.mean_long_short_return)
    base_mean_ic = _finite_or_nan(_parse_optional_float(metrics.get("mean_ic")))
    base_mean_ls = _finite_or_nan(_parse_optional_float(metrics.get("mean_long_short_return")))

    metrics["next_open_execution_available"] = True
    metrics["next_open_mean_ic"] = next_open_mean_ic
    metrics["next_open_mean_long_short_return"] = next_open_mean_ls
    metrics["next_open_long_short_ir"] = float(next_open_summary.long_short_ir)
    if np.isfinite(next_open_mean_ic) and np.isfinite(base_mean_ic):
        metrics["next_open_mean_ic_delta"] = next_open_mean_ic - base_mean_ic
    if np.isfinite(next_open_mean_ls) and np.isfinite(base_mean_ls):
        metrics["next_open_mean_long_short_return_delta"] = next_open_mean_ls - base_mean_ls


def _merge_haircut_sharpe_metrics(metrics: dict[str, object]) -> None:
    """Add a multiple-testing-adjusted Sharpe ratio to the metrics dict.

    ``n_trials`` is conservatively set to 1 — the default single-factor case
    assumes no grid search.  Callers running a parameter sweep should
    overwrite these fields with a higher ``n_trials`` after collecting
    variant results.
    """
    observed_sharpe = _parse_optional_float(metrics.get("long_short_ir"))
    n_obs = _parse_optional_int(metrics.get("n_dates_used"))

    metrics["haircut_sharpe_n_trials_assumed"] = 1
    if observed_sharpe is None or n_obs is None or n_obs < 2:
        metrics["haircut_sharpe_observed"] = (
            float(observed_sharpe) if observed_sharpe is not None else float("nan")
        )
        metrics["haircut_sharpe_expected_max"] = float("nan")
        metrics["haircut_sharpe_adjusted"] = float("nan")
        metrics["haircut_sharpe_ratio"] = float("nan")
        return

    result = haircut_sharpe_ratio(
        observed_sharpe=float(observed_sharpe),
        n_trials=1,
        n_obs=int(n_obs),
    )
    metrics["haircut_sharpe_observed"] = result.observed_sharpe
    metrics["haircut_sharpe_expected_max"] = result.expected_max_sharpe
    metrics["haircut_sharpe_adjusted"] = result.haircut_sharpe
    metrics["haircut_sharpe_ratio"] = result.haircut_ratio


def _merge_marginal_contribution_metrics(
    metrics: dict[str, object],
    *,
    factor_df: pd.DataFrame,
    label_df: pd.DataFrame,
    enabled: bool = True,
) -> None:
    """Run independent Fama-MacBeth regression (no existing factors) and
    write scalar fields into ``metrics``.

    In single-factor scenarios the spanning test is not applicable (no
    existing factors to control for), so those fields remain ``None``.
    The Fama-MacBeth regression is still informative: it cross-validates
    the IC evidence via an explicit cross-sectional predictive-power
    coefficient.
    """
    metrics["fama_macbeth_mean_coefficient"] = None
    metrics["fama_macbeth_t_statistic"] = None
    metrics["fama_macbeth_p_value"] = None
    metrics["fama_macbeth_n_dates"] = None
    metrics["spanning_is_spanned"] = None
    metrics["spanning_r_squared_increment"] = None
    metrics["marginal_flags"] = []
    if not enabled:
        return

    summary = compute_marginal_contribution(
        candidate_factor_df=factor_df,
        label_df=label_df,
        existing_factor_dfs=None,
    )
    fm = summary.fama_macbeth
    metrics["fama_macbeth_mean_coefficient"] = (
        float(fm.mean_coefficient) if fm is not None else None
    )
    metrics["fama_macbeth_t_statistic"] = float(fm.t_statistic) if fm is not None else None
    metrics["fama_macbeth_p_value"] = float(fm.p_value) if fm is not None else None
    metrics["fama_macbeth_n_dates"] = int(fm.n_dates) if fm is not None else None
    metrics["spanning_is_spanned"] = None
    metrics["spanning_r_squared_increment"] = None
    metrics["marginal_flags"] = list(summary.marginal_flags)


def _merge_neutralization_comparison_metrics(
    metrics: dict[str, object],
    *,
    comparison: NeutralizationComparison,
) -> None:
    payload = comparison.to_dict()
    delta = comparison.delta
    raw = comparison.raw
    metrics["neutralization_comparison"] = payload
    metrics["neutralization_comparison_flags"] = list(comparison.interpretation_flags)
    metrics["neutralization_comparison_reasons"] = list(comparison.interpretation_reasons)
    metrics["neutralization_raw_mean_ic"] = raw.get("mean_ic")
    metrics["neutralization_raw_mean_rank_ic"] = raw.get("mean_rank_ic")
    metrics["neutralization_raw_mean_long_short_return"] = raw.get("mean_long_short_return")
    metrics["neutralization_raw_ic_ir"] = raw.get("ic_ir")
    metrics["neutralization_mean_ic_delta"] = delta.get("mean_ic_delta")
    metrics["neutralization_mean_rank_ic_delta"] = delta.get("mean_rank_ic_delta")
    metrics["neutralization_mean_long_short_return_delta"] = delta.get(
        "mean_long_short_return_delta"
    )
    metrics["neutralization_ic_ir_delta"] = delta.get("ic_ir_delta")
    metrics["neutralization_valid_ratio_min_delta"] = delta.get("valid_ratio_min_delta")
    metrics["neutralization_eval_coverage_ratio_mean_delta"] = delta.get(
        "eval_coverage_ratio_mean_delta"
    )
    metrics["neutralization_uncertainty_overlap_zero_count_delta"] = delta.get(
        "uncertainty_overlap_zero_count_delta"
    )
    metrics["neutralization_rolling_positive_share_min_delta"] = delta.get(
        "rolling_positive_share_min_delta"
    )
    metrics["neutralization_rolling_worst_mean_min_delta"] = delta.get(
        "rolling_worst_mean_min_delta"
    )


def _merge_param_sensitivity_metrics(
    metrics: dict[str, object],
    *,
    prices: pd.DataFrame,
    factor_df: pd.DataFrame,
    horizon: int,
    base_n_quantiles: int,
    evaluation_config: ResearchEvaluationConfig,
    label_df: pd.DataFrame | None = None,
    label_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    enabled: bool = True,
) -> None:
    """Re-run the factor at alternative ``n_quantiles`` settings and record
    the IC/IR spread across variants.

    A factor whose headline IC collapses when you move from 5 to 3 or 10
    buckets is capturing a single quirky slice of the distribution rather
    than a monotone cross-sectional signal.  The spread fields surface that
    instability without requiring the caller to re-run the pipeline by hand.
    """
    defaults = {
        "param_sensitivity_n_variants": 0,
        "param_sensitivity_n_quantiles_grid": (),
        "param_sensitivity_mean_ic_min": float("nan"),
        "param_sensitivity_mean_ic_max": float("nan"),
        "param_sensitivity_mean_ic_std": float("nan"),
        "param_sensitivity_mean_ic_range": float("nan"),
        "param_sensitivity_long_short_ir_min": float("nan"),
        "param_sensitivity_long_short_ir_max": float("nan"),
        "param_sensitivity_long_short_ir_std": float("nan"),
        "param_sensitivity_long_short_ir_range": float("nan"),
    }
    metrics.update(defaults)
    if not enabled:
        return

    candidate_grid = tuple(q for q in (3, 5, 10) if q != base_n_quantiles and q >= 2)
    if not candidate_grid:
        return

    resolved_label_df = _resolve_variant_label_df(
        prices=prices,
        horizon=horizon,
        label_fn=label_fn,
        label_df=label_df,
    )

    ic_values: list[float] = []
    ir_values: list[float] = []
    used_grid: list[int] = []
    for q in candidate_grid:
        try:
            variant = _evaluate_variant_lightweight(
                factor_df=factor_df,
                label_df=resolved_label_df,
                n_quantiles=q,
            )
        except Exception:
            continue
        mic = float(variant.mean_ic)
        ir = float(variant.long_short_ir)
        if np.isfinite(mic):
            ic_values.append(mic)
        if np.isfinite(ir):
            ir_values.append(ir)
        used_grid.append(int(q))

    if not used_grid:
        return

    metrics["param_sensitivity_n_variants"] = len(used_grid)
    metrics["param_sensitivity_n_quantiles_grid"] = tuple(used_grid)
    if ic_values:
        arr = np.asarray(ic_values, dtype=float)
        metrics["param_sensitivity_mean_ic_min"] = float(arr.min())
        metrics["param_sensitivity_mean_ic_max"] = float(arr.max())
        metrics["param_sensitivity_mean_ic_std"] = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
        metrics["param_sensitivity_mean_ic_range"] = float(arr.max() - arr.min())
    if ir_values:
        arr = np.asarray(ir_values, dtype=float)
        metrics["param_sensitivity_long_short_ir_min"] = float(arr.min())
        metrics["param_sensitivity_long_short_ir_max"] = float(arr.max())
        metrics["param_sensitivity_long_short_ir_std"] = (
            float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
        )
        metrics["param_sensitivity_long_short_ir_range"] = float(arr.max() - arr.min())


def _merge_baseline_factor_comparison_metrics(
    metrics: dict[str, object],
    *,
    prices: pd.DataFrame,
    factor_df: pd.DataFrame,
    horizon: int,
    n_quantiles: int,
    evaluation_config: ResearchEvaluationConfig,
    label_df: pd.DataFrame | None = None,
    label_fn: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    enabled: bool = True,
) -> None:
    """Compare the factor against the default research baseline suite."""
    metrics["baseline_momentum_mean_ic"] = float("nan")
    metrics["baseline_momentum_long_short_ir"] = float("nan")
    metrics["baseline_momentum_factor_rank_corr"] = float("nan")
    metrics["baseline_reversal_mean_ic"] = float("nan")
    metrics["baseline_reversal_long_short_ir"] = float("nan")
    metrics["baseline_reversal_factor_rank_corr"] = float("nan")
    metrics["baseline_best_mean_ic"] = float("nan")
    metrics["baseline_suite_best_name"] = ""
    metrics["baseline_suite_best_family"] = ""
    metrics["baseline_suite_best_mean_ic"] = float("nan")
    metrics["baseline_suite_best_long_short_ir"] = float("nan")
    metrics["baseline_suite_best_factor_rank_corr"] = float("nan")
    metrics["baseline_suite_count"] = 0
    metrics["baseline_suite_evaluated_count"] = 0
    metrics["baseline_suite_skipped_count"] = 0
    metrics["baseline_suite_evaluated_names"] = ()
    metrics["baseline_suite_skipped_names"] = ()
    metrics["baseline_factor_mean_ic_advantage"] = float("nan")
    if not enabled:
        return

    if "close" not in prices.columns:
        return

    resolved_label_df = _resolve_variant_label_df(
        prices=prices,
        horizon=horizon,
        label_fn=label_fn,
        label_df=label_df,
    )
    base_mean_ic = _finite_or_nan(_parse_optional_float(metrics.get("mean_ic")))
    best = float("nan")
    best_name = ""
    best_family = ""
    best_ir = float("nan")
    best_rank_corr = float("nan")
    evaluated_names: list[str] = []
    skipped_names: list[str] = []
    specs = iter_baseline_factor_specs()
    metrics["baseline_suite_count"] = len(specs)

    for spec in specs:
        if not baseline_required_columns_available(spec, prices.columns):
            skipped_names.append(spec.name)
            continue
        try:
            base_df = build_factor_from_recipe_mapping(
                prices=prices,
                recipe=spec.recipe,
                factor_name=spec.name,
            )
            base_df["date"] = pd.to_datetime(base_df["date"], errors="coerce")
        except Exception:
            skipped_names.append(spec.name)
            continue
        if base_df.empty:
            skipped_names.append(spec.name)
            continue
        try:
            result = _evaluate_variant_lightweight(
                factor_df=base_df,
                label_df=resolved_label_df,
                n_quantiles=n_quantiles,
            )
        except Exception:
            skipped_names.append(spec.name)
            continue
        evaluated_names.append(spec.name)
        mic = float(result.mean_ic)
        ir = float(result.long_short_ir)
        metrics[f"baseline_{spec.name}_mean_ic"] = mic
        metrics[f"baseline_{spec.name}_long_short_ir"] = ir
        try:
            rank_corr = _cross_sectional_rank_corr(factor_df, base_df)
        except Exception:
            rank_corr = float("nan")
        metrics[f"baseline_{spec.name}_factor_rank_corr"] = float(rank_corr)
        if spec.name == "mom_20d":
            metrics["baseline_momentum_mean_ic"] = mic
            metrics["baseline_momentum_long_short_ir"] = ir
            metrics["baseline_momentum_factor_rank_corr"] = float(rank_corr)
        elif spec.name == "rev_5d":
            metrics["baseline_reversal_mean_ic"] = mic
            metrics["baseline_reversal_long_short_ir"] = ir
            metrics["baseline_reversal_factor_rank_corr"] = float(rank_corr)
        if np.isfinite(mic):
            if not np.isfinite(best) or mic > best:
                best = mic
                best_name = spec.name
                best_family = spec.family
                best_ir = ir
                best_rank_corr = float(rank_corr)

    metrics["baseline_suite_evaluated_count"] = len(evaluated_names)
    metrics["baseline_suite_skipped_count"] = len(skipped_names)
    metrics["baseline_suite_evaluated_names"] = tuple(evaluated_names)
    metrics["baseline_suite_skipped_names"] = tuple(skipped_names)
    if np.isfinite(best):
        metrics["baseline_best_mean_ic"] = best
        metrics["baseline_suite_best_name"] = best_name
        metrics["baseline_suite_best_family"] = best_family
        metrics["baseline_suite_best_mean_ic"] = best
        metrics["baseline_suite_best_long_short_ir"] = best_ir
        metrics["baseline_suite_best_factor_rank_corr"] = best_rank_corr
        if np.isfinite(base_mean_ic):
            metrics["baseline_factor_mean_ic_advantage"] = base_mean_ic - best


def _cross_sectional_rank_corr(
    left_factor: pd.DataFrame,
    right_factor: pd.DataFrame,
) -> float:
    """Mean per-date Spearman correlation between two factor series."""
    left = left_factor[["date", "asset", "value"]].rename(columns={"value": "_l"})
    right = right_factor[["date", "asset", "value"]].rename(columns={"value": "_r"})
    merged = left.merge(right, on=["date", "asset"], how="inner").dropna()
    if merged.empty:
        return float("nan")
    per_date = merged.groupby("date", sort=False)[["_l", "_r"]].apply(
        lambda g: g["_l"].corr(g["_r"], method="spearman")
    )
    per_date = per_date.replace([np.inf, -np.inf], np.nan).dropna()
    if per_date.empty:
        return float("nan")
    return float(per_date.mean())


def _merge_random_factor_baseline_metrics(
    metrics: dict[str, object],
    *,
    factor_df: pd.DataFrame,
    label_df: pd.DataFrame,
    n_permutations: int = 50,
    seed: int = 20260418,
    enabled: bool = True,
) -> pd.DataFrame:
    """Estimate a null distribution for mean rank-IC by shuffling the factor
    values within each date, and report a p-value for the observed value.

    This is the cheapest possible "is the observed IC distinguishable from
    noise on this universe?" check.  It does not replace Deflated Sharpe or
    Haircut Sharpe — those account for multiple testing — but it does catch
    the case where a factor's IC is numerically positive but inside the
    cross-sectional noise floor of this particular universe.
    """
    metrics["random_baseline_n_permutations"] = 0
    metrics["random_baseline_mean_ic_mean"] = float("nan")
    metrics["random_baseline_mean_ic_std"] = float("nan")
    metrics["random_baseline_mean_ic_p95"] = float("nan")
    metrics["random_baseline_mean_ic_p99"] = float("nan")
    metrics["random_baseline_p_value"] = float("nan")
    metrics["random_baseline_observed_mean_rank_ic"] = float("nan")
    metrics["random_baseline_observed_z_score"] = float("nan")
    empty_null = pd.DataFrame(columns=["permutation", "mean_ic"])
    if not enabled:
        return empty_null

    if factor_df.empty or label_df.empty:
        return empty_null

    observed_mean, arr = compute_mean_rank_ic_permutation_null(
        factor_df,
        label_df,
        n_permutations=int(n_permutations),
        seed=int(seed),
        min_assets_per_date=3,
    )
    if not np.isfinite(observed_mean) or arr.size == 0:
        return empty_null

    metrics["random_baseline_n_permutations"] = int(arr.size)
    metrics["random_baseline_mean_ic_mean"] = float(arr.mean())
    metrics["random_baseline_mean_ic_std"] = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
    metrics["random_baseline_mean_ic_p95"] = float(np.quantile(arr, 0.95))
    metrics["random_baseline_mean_ic_p99"] = float(np.quantile(arr, 0.99))
    metrics["random_baseline_observed_mean_rank_ic"] = float(observed_mean)
    tail = (
        float((arr >= observed_mean).sum())
        if observed_mean >= 0
        else float((arr <= observed_mean).sum())
    )
    metrics["random_baseline_p_value"] = (tail + 1.0) / (arr.size + 1.0)
    std = metrics["random_baseline_mean_ic_std"]
    if isinstance(std, float) and std > 0.0:
        metrics["random_baseline_observed_z_score"] = float((observed_mean - arr.mean()) / std)
    return pd.DataFrame(
        {
            "permutation": np.arange(1, arr.size + 1, dtype=int),
            "mean_ic": arr,
        }
    )
