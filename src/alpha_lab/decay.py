"""IC decay and factor autocorrelation analysis.

Measures how factor predictive power decays across holding horizons and
how persistent factor rankings are over time.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from alpha_lab.evaluation import compute_ic, compute_ic_summary, compute_rank_ic
from alpha_lab.labels import forward_return
from alpha_lab.sorted_panel import ensure_sorted

AUTOCORR_FAST_PATH_COVERAGE = 0.995
AUTOCORR_MIN_ASSETS = 3


def estimate_ic_half_life(decay_df: pd.DataFrame) -> dict[str, object]:
    """Estimate the horizon where |mean IC| decays to 50% of its baseline.

    Baseline is the smallest finite-horizon ``abs(mean_ic)`` observation.
    The crossing point is linearly interpolated between adjacent horizons.

    Returns a compact payload with:
    ``ic_half_life_horizon``, ``ic_half_life_status``, ``ic_half_life_not_reached``,
    ``ic_half_life_baseline_horizon``, and ``ic_half_life_baseline_abs_mean_ic``.
    """
    payload: dict[str, object] = {
        "ic_half_life_horizon": float("nan"),
        "ic_half_life_status": "unavailable",
        "ic_half_life_not_reached": False,
        "ic_half_life_baseline_horizon": float("nan"),
        "ic_half_life_baseline_abs_mean_ic": float("nan"),
    }
    if decay_df.empty or "horizon" not in decay_df.columns or "mean_ic" not in decay_df.columns:
        return payload

    working = decay_df.loc[:, ["horizon", "mean_ic"]].copy()
    working["horizon"] = pd.to_numeric(working["horizon"], errors="coerce")
    working["mean_ic"] = pd.to_numeric(working["mean_ic"], errors="coerce")
    working = working.dropna(subset=["horizon", "mean_ic"]).sort_values(
        "horizon",
        kind="mergesort",
    )
    if len(working) < 2:
        return payload

    working["abs_mean_ic"] = working["mean_ic"].abs()
    baseline = working.iloc[0]
    baseline_abs = float(baseline["abs_mean_ic"])
    if not np.isfinite(baseline_abs) or baseline_abs <= 0.0:
        return payload

    payload["ic_half_life_baseline_horizon"] = float(baseline["horizon"])
    payload["ic_half_life_baseline_abs_mean_ic"] = baseline_abs
    target = 0.5 * baseline_abs

    later = working.iloc[1:].copy()
    if later.empty:
        return payload

    prev_h = float(baseline["horizon"])
    prev_abs = baseline_abs
    for row in later.itertuples(index=False):
        curr_h = float(row.horizon)
        curr_abs = float(row.abs_mean_ic)
        if not np.isfinite(curr_abs):
            continue
        if curr_abs <= target:
            if prev_abs <= target or curr_h <= prev_h or prev_abs == curr_abs:
                half_life = curr_h
            else:
                weight = (target - prev_abs) / (curr_abs - prev_abs)
                half_life = prev_h + weight * (curr_h - prev_h)
            payload["ic_half_life_horizon"] = float(half_life)
            payload["ic_half_life_status"] = "estimated"
            return payload
        prev_h = curr_h
        prev_abs = curr_abs

    payload["ic_half_life_status"] = "not_reached"
    payload["ic_half_life_not_reached"] = True
    return payload


def compute_ic_decay(
    factor_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    horizons: tuple[int, ...] = (1, 2, 3, 5, 10, 20),
    precomputed_labels_by_horizon: Mapping[int, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Compute IC summary at multiple forward-return horizons.

    For each horizon *h*, computes ``forward_return(prices, h)`` and then
    cross-sectional Pearson IC and Spearman RankIC between the factor and
    those labels.

    Parameters
    ----------
    factor_df:
        Canonical ``[date, asset, factor, value]`` factor output.
    prices_df:
        Price panel with at least ``[date, asset, close]``.
    horizons:
        Tuple of forward-return horizons to evaluate.
    precomputed_labels_by_horizon:
        Optional mapping ``horizon -> forward-return labels``. When provided,
        the function reuses cached labels for matching horizons and falls back
        to ``forward_return(prices_df, horizon=h)`` only for missing horizons.

    Returns
    -------
    pd.DataFrame
        One row per horizon with columns:
        ``[horizon, mean_ic, mean_rank_ic, ic_ir, t_stat, p_value, n_dates]``.
    """
    rows: list[dict[str, object]] = []
    for h in horizons:
        labels: pd.DataFrame | None = None
        if precomputed_labels_by_horizon is not None:
            cached = precomputed_labels_by_horizon.get(int(h))
            if cached is not None:
                labels = cached.copy()
        if labels is None:
            labels = forward_return(prices_df, horizon=h)
        if labels.empty:
            rows.append(
                {
                    "horizon": h,
                    "mean_ic": float("nan"),
                    "mean_rank_ic": float("nan"),
                    "ic_ir": float("nan"),
                    "t_stat": float("nan"),
                    "p_value": float("nan"),
                    "n_dates": 0,
                }
            )
            continue

        merged_eval = (
            factor_df[["date", "asset", "value"]]
            .rename(columns={"value": "value_factor"})
            .merge(
                labels[["date", "asset", "value"]].rename(columns={"value": "value_label"}),
                on=["date", "asset"],
                how="inner",
                validate="one_to_one",
            )
        )
        ic_df = compute_ic(factor_df, labels, merged_pairs=merged_eval)
        rank_ic_df = compute_rank_ic(factor_df, labels, merged_pairs=merged_eval)

        ic_vals = ic_df["ic"] if not ic_df.empty else pd.Series(dtype=float)
        rank_ic_vals = rank_ic_df["rank_ic"] if not rank_ic_df.empty else pd.Series(dtype=float)

        summary = compute_ic_summary(ic_vals)
        mean_rank_ic = (
            float(rank_ic_vals.dropna().mean()) if len(rank_ic_vals.dropna()) > 0 else float("nan")
        )

        rows.append(
            {
                "horizon": h,
                "mean_ic": summary["mean_ic"],
                "mean_rank_ic": mean_rank_ic,
                "ic_ir": summary["ic_ir"],
                "t_stat": summary["t_stat"],
                "p_value": summary["p_value"],
                "n_dates": summary["n_obs"],
            }
        )

    return pd.DataFrame(rows)


def compute_factor_autocorrelation(
    factor_df: pd.DataFrame,
    lags: tuple[int, ...] = (1, 2, 3, 5, 10),
) -> pd.DataFrame:
    """Cross-sectional rank autocorrelation of factor values at different lags.

    For each lag *k*, computes the Spearman rank correlation between the
    factor cross-section at date *t* and date *t-k*.  High autocorrelation
    implies low turnover and better tradability.

    Parameters
    ----------
    factor_df:
        Canonical ``[date, asset, factor, value]``.
    lags:
        Tuple of lag values (in number of dates) to evaluate.

    Returns
    -------
    pd.DataFrame
        One row per lag with columns: ``[lag, mean_autocorr, std_autocorr, n_dates]``.
    """
    factor_df = ensure_sorted(factor_df, by=("date", "asset"))
    wide = factor_df.pivot(index="date", columns="asset", values="value")
    wide = wide.sort_index()
    if wide.empty:
        return _empty_autocorr_rows(lags)

    coverage_min = float(wide.notna().mean(axis=1).min())
    if coverage_min >= AUTOCORR_FAST_PATH_COVERAGE:
        return _autocorr_preranked_fast_path(wide, lags)

    return _autocorr_per_pair_rank_fallback(wide, lags)


def _autocorr_preranked_fast_path(
    wide: pd.DataFrame,
    lags: tuple[int, ...],
) -> pd.DataFrame:
    ranked = wide.rank(axis=1, method="average")
    ranked_values = ranked.to_numpy(dtype=float, copy=False)
    n_dates = int(ranked_values.shape[0])

    rows: list[dict[str, object]] = []
    for lag in lags:
        lag_int = int(lag)
        if lag_int >= n_dates:
            rows.append(
                {
                    "lag": lag,
                    "mean_autocorr": float("nan"),
                    "std_autocorr": float("nan"),
                    "n_dates": 0,
                }
            )
            continue

        corrs = _rowwise_pearson_corr(
            ranked_values[lag_int:],
            ranked_values[:-lag_int],
            min_assets=AUTOCORR_MIN_ASSETS,
        )
        clean = corrs[np.isfinite(corrs)]
        rows.append(
            {
                "lag": lag,
                "mean_autocorr": float(np.mean(clean)) if clean.size else float("nan"),
                "std_autocorr": (
                    float(np.std(clean, ddof=1)) if int(clean.size) > 1 else float("nan")
                ),
                "n_dates": int(clean.size),
            }
        )

    return pd.DataFrame(rows)


def _autocorr_per_pair_rank_fallback(
    wide: pd.DataFrame,
    lags: tuple[int, ...],
) -> pd.DataFrame:
    values = wide.to_numpy(dtype=float, copy=False)
    n_dates = int(values.shape[0])

    rows: list[dict[str, object]] = []
    for lag in lags:
        lag_int = int(lag)
        if lag_int >= n_dates:
            rows.append(
                {
                    "lag": lag,
                    "mean_autocorr": float("nan"),
                    "std_autocorr": float("nan"),
                    "n_dates": 0,
                }
            )
            continue

        corrs: list[float] = []
        for i in range(lag_int, n_dates):
            corr = _spearman_corr_on_overlap(
                values[i],
                values[i - lag_int],
                min_assets=AUTOCORR_MIN_ASSETS,
            )
            if not np.isfinite(corr):
                continue
            corrs.append(corr)

        clean = [c for c in corrs if np.isfinite(c)]
        rows.append(
            {
                "lag": lag,
                "mean_autocorr": float(np.mean(clean)) if clean else float("nan"),
                "std_autocorr": float(np.std(clean, ddof=1)) if len(clean) > 1 else float("nan"),
                "n_dates": len(clean),
            }
        )

    return pd.DataFrame(rows)


def _empty_autocorr_rows(lags: tuple[int, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "lag": lag,
                "mean_autocorr": float("nan"),
                "std_autocorr": float("nan"),
                "n_dates": 0,
            }
            for lag in lags
        ]
    )


def _rowwise_pearson_corr(
    current: np.ndarray,
    previous: np.ndarray,
    *,
    min_assets: int,
) -> np.ndarray:
    valid = np.isfinite(current) & np.isfinite(previous)
    n = valid.sum(axis=1).astype(float)
    current_zeroed = np.where(valid, current, 0.0)
    previous_zeroed = np.where(valid, previous, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        current_mean = current_zeroed.sum(axis=1) / n
        previous_mean = previous_zeroed.sum(axis=1) / n
        current_centered = np.where(valid, current - current_mean[:, None], 0.0)
        previous_centered = np.where(valid, previous - previous_mean[:, None], 0.0)
        numerator = (current_centered * previous_centered).sum(axis=1)
        denom = np.sqrt(
            (current_centered * current_centered).sum(axis=1)
            * (previous_centered * previous_centered).sum(axis=1)
        )
        corr = numerator / denom
    corr[(n < float(min_assets)) | ~np.isfinite(corr)] = np.nan
    return cast(np.ndarray, corr)


def _spearman_corr_on_overlap(
    current: np.ndarray,
    previous: np.ndarray,
    *,
    min_assets: int,
) -> float:
    valid = np.isfinite(current) & np.isfinite(previous)
    if int(valid.sum()) < min_assets:
        return float("nan")

    current_rank = scipy_stats.rankdata(current[valid], method="average")
    previous_rank = scipy_stats.rankdata(previous[valid], method="average")

    current_dev = current_rank - current_rank.mean()
    previous_dev = previous_rank - previous_rank.mean()
    denom = float(np.sqrt(np.dot(current_dev, current_dev) * np.dot(previous_dev, previous_dev)))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(np.dot(current_dev, previous_dev) / denom)
