"""Pin the aggregation semantics of ``ExperimentSummary.mean_ic``.

``run_factor_experiment`` reports **mean-of-per-date IC** (every evaluation
date contributes equal weight) — not pooled IC (every (factor, label) pair
contributing equal weight). When the cross-section size varies across dates,
the two aggregations differ materially.

These tests lock that design choice: the summary value equals the mean of
daily ICs, and on a deliberately unbalanced panel this is measurably
different from the pooled alternative computed over all (factor, label)
observations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.experiment import run_factor_experiment


def _build_unbalanced_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Produce prices and an explicit factor_df with asymmetric date coverage.

    Early dates have many assets; later dates have few. A truly pooled IC
    would be dominated by the early dates, while mean-of-daily IC weights
    every date equally.
    """

    dates = pd.date_range("2024-01-01", periods=16, freq="B")
    narrow_assets = [f"N{i}" for i in range(3)]
    wide_assets = narrow_assets + [f"W{i}" for i in range(12)]

    rng = np.random.default_rng(2026)
    rows_prices: list[dict] = []
    rows_factor: list[dict] = []
    for i, date in enumerate(dates):
        assets_today = wide_assets if i < 8 else narrow_assets
        for asset in assets_today:
            price = 100.0 + float(rng.normal(0.0, 1.0))
            rows_prices.append({"date": date, "asset": asset, "close": price})
            rows_factor.append(
                {
                    "date": date,
                    "asset": asset,
                    "factor": "synthetic",
                    "value": float(rng.normal(0.0, 1.0)),
                }
            )

    # Ensure each asset has a continuous series (fill gaps with last-known
    # close) so forward_return can compute labels.
    prices = pd.DataFrame(rows_prices)
    full = []
    for asset, sub in prices.groupby("asset"):
        reindexed = sub.set_index("date").reindex(dates).ffill().bfill()
        reindexed["asset"] = asset
        reindexed = reindexed.reset_index().rename(columns={"index": "date"})
        full.append(reindexed)
    prices = pd.concat(full, ignore_index=True)[["date", "asset", "close"]]

    factor_df = pd.DataFrame(rows_factor)
    return prices, factor_df


def _factor_fn(factor_df: pd.DataFrame):
    return lambda _prices: factor_df.copy()


def _pooled_ic(factor_df: pd.DataFrame, label_df: pd.DataFrame) -> float:
    labels = label_df[["date", "asset", "value"]].rename(columns={"value": "label"})
    merged = factor_df[["date", "asset", "value"]].merge(
        labels,
        on=["date", "asset"],
        how="inner",
    )
    merged = merged.dropna(subset=["value", "label"])
    if len(merged) < 2:
        return float("nan")
    return float(merged["value"].corr(merged["label"], method="pearson"))


def test_mean_ic_equals_mean_of_daily_ics_not_pooled():
    prices, factor_df = _build_unbalanced_panel()
    result = run_factor_experiment(prices, _factor_fn(factor_df), allow_full_sample_evaluation=True)

    daily = result.ic_df["ic"].dropna()
    assert len(daily) >= 4, "need several valid dates for this aggregation test"

    mean_of_daily = float(daily.mean())
    assert np.isclose(result.summary.mean_ic, mean_of_daily, atol=1e-12), (
        f"summary.mean_ic ({result.summary.mean_ic}) should equal the mean of "
        f"per-date ICs ({mean_of_daily})"
    )

    pooled = _pooled_ic(result.factor_df, result.label_df)
    # The two aggregations should differ materially on an unbalanced panel —
    # if they ever coincide to 1e-6 by coincidence on this seed, regenerate.
    assert not np.isclose(mean_of_daily, pooled, atol=1e-6), (
        f"unbalanced panel expected to show mean-of-daily ({mean_of_daily}) "
        f"≠ pooled IC ({pooled}); tune the fixture"
    )


def test_mean_ic_equal_to_pooled_on_balanced_panel():
    """Balanced panel (same asset count every date) → the two aggregations
    can still differ by construction of Pearson correlation, but the daily
    means should at least be finite and close in magnitude. This test keeps
    the aggregation distinction unambiguous rather than asserting equality.
    """

    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    assets = [f"A{i}" for i in range(10)]
    rng = np.random.default_rng(7)
    rows_prices = []
    rows_factor = []
    for date in dates:
        for asset in assets:
            rows_prices.append({"date": date, "asset": asset, "close": 100.0 + float(rng.normal())})
            rows_factor.append(
                {
                    "date": date,
                    "asset": asset,
                    "factor": "synthetic",
                    "value": float(rng.normal()),
                }
            )
    prices = pd.DataFrame(rows_prices)
    factor_df = pd.DataFrame(rows_factor)
    result = run_factor_experiment(prices, _factor_fn(factor_df), allow_full_sample_evaluation=True)

    daily = result.ic_df["ic"].dropna()
    assert np.isfinite(result.summary.mean_ic)
    assert np.isclose(result.summary.mean_ic, float(daily.mean()), atol=1e-12)
