from __future__ import annotations

import math

import numpy as np
import pandas as pd

from alpha_lab.factor_report import (
    build_factor_report,
    compute_ic_summary,
    estimate_half_life,
    quantile_monotonicity,
)
from alpha_lab.factors.momentum import momentum


def _make_prices(n_assets: int = 6, n_days: int = 80, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows: list[dict[str, object]] = []
    for asset in [f"A{i}" for i in range(n_assets)]:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


def test_compute_ic_summary_returns_expected_shape() -> None:
    ic_df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=3), "ic": [1.0, 2.0, 3.0]})
    out = compute_ic_summary(ic_df, value_col="ic")
    assert len(out) == 1
    assert out["metric"].iloc[0] == "ic"
    assert out["mean"].iloc[0] == 2.0
    assert out["n_obs"].iloc[0] == 3


def test_quantile_monotonicity_flags_monotonic_and_non_monotonic_dates() -> None:
    qret = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")] * 3 + [pd.Timestamp("2024-01-02")] * 3,
            "quantile": [1, 2, 3, 1, 2, 3],
            "mean_return": [0.01, 0.02, 0.03, 0.01, 0.03, 0.02],
        }
    )
    out = quantile_monotonicity(qret)
    out = out.sort_values("date").reset_index(drop=True)
    assert bool(out.loc[0, "is_monotonic"]) is True
    assert bool(out.loc[1, "is_monotonic"]) is False


def test_estimate_half_life_returns_positive_for_decay_profile() -> None:
    profile = pd.DataFrame(
        {
            "horizon": [1, 2, 5, 10],
            "mean_rank_ic": [0.10, 0.08, 0.04, 0.02],
        }
    )
    half_life = estimate_half_life(profile)
    assert half_life > 0
    assert math.isfinite(half_life)


def test_build_factor_report_smoke() -> None:
    prices = _make_prices()
    factors = momentum(prices, window=5)
    report = build_factor_report(prices=prices, factors=factors, horizon=5, n_quantiles=5)
    assert report.horizon == 5
    assert report.n_quantiles == 5
    assert isinstance(report.ic_df, pd.DataFrame)
    assert isinstance(report.coverage_df, pd.DataFrame)
    assert "n_overlap_assets" in report.coverage_df.columns
