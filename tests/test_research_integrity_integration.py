"""Integration tests: research_integrity core wired into experiment / walk_forward.

These tests verify that the integrity checks committed in Round 1 (research_integrity
package) are correctly invoked by the two primary consumers: run_factor_experiment
and run_walk_forward_experiment.

Round 2 / Bucket C tests (single_factor pipeline, neutralization) are not included
here — they belong to a separate commit.
"""

from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.experiment import run_factor_experiment
from alpha_lab.factors.momentum import momentum
from alpha_lab.research_integrity.exceptions import IntegrityHardFailure
from alpha_lab.walk_forward import run_walk_forward_experiment


def _make_prices(n_assets: int = 4, n_days: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows: list[dict[str, object]] = []
    for i in range(n_assets):
        asset = f"A{i:03d}"
        price = 100.0 + i
        for date in dates:
            price *= 1.001
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


def test_experiment_fails_fast_on_obvious_future_factor_dates() -> None:
    prices = _make_prices()

    def leaky_factor_fn(input_prices: pd.DataFrame) -> pd.DataFrame:
        factor = momentum(input_prices, window=5)
        factor = factor.copy()
        factor["date"] = pd.to_datetime(factor["date"]) + pd.Timedelta(days=45)
        return factor

    with pytest.raises(IntegrityHardFailure):
        run_factor_experiment(prices, leaky_factor_fn, horizon=5, n_quantiles=5)


def test_walk_forward_builds_fold_and_aggregate_integrity_reports() -> None:
    prices = _make_prices(n_days=80)

    result = run_walk_forward_experiment(
        prices,
        lambda p: momentum(p, window=5),
        train_size=30,
        test_size=10,
        step=10,
        horizon=5,
    )

    assert result.aggregate_integrity_report is not None
    assert len(result.fold_integrity_reports) == result.aggregate_summary.n_folds
    assert result.aggregate_integrity_report.summary.n_checks > 0


def test_walk_forward_raises_when_fold_factor_has_future_dates() -> None:
    prices = _make_prices(n_days=80)

    def leaky_factor_fn(input_prices: pd.DataFrame) -> pd.DataFrame:
        factor = momentum(input_prices, window=5)
        factor = factor.copy()
        factor["date"] = pd.to_datetime(factor["date"]) + pd.Timedelta(days=45)
        return factor

    with pytest.raises(IntegrityHardFailure):
        run_walk_forward_experiment(
            prices,
            leaky_factor_fn,
            train_size=30,
            test_size=10,
            step=10,
            horizon=5,
        )
