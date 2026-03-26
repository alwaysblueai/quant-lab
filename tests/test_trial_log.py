from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.experiment import run_factor_experiment
from alpha_lab.experiment_metadata import ExperimentMetadata
from alpha_lab.factors.momentum import momentum
from alpha_lab.trial_log import (
    TRIAL_LOG_COLUMNS,
    append_trial_log,
    load_trial_log,
    trial_row_from_result,
)


def _make_prices(n_assets: int = 4, n_days: int = 30, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows: list[dict[str, object]] = []
    for asset in [f"A{i}" for i in range(n_assets)]:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


def _momentum_fn(prices: pd.DataFrame) -> pd.DataFrame:
    return momentum(prices, window=5)


def test_trial_row_from_result_contains_canonical_columns() -> None:
    result = run_factor_experiment(
        _make_prices(),
        _momentum_fn,
        metadata=ExperimentMetadata(
            trial_id="t1",
            trial_count=3,
            dataset_id="prices_v1",
            dataset_hash="abc123",
            verdict="promising",
        ),
    )
    row = trial_row_from_result(result, experiment_name="momentum_h1")
    assert list(row.columns) == list(TRIAL_LOG_COLUMNS)
    assert row["trial_id"].iloc[0] == "t1"
    assert row["trial_count"].iloc[0] == 3
    assert row["dataset_id"].iloc[0] == "prices_v1"


def test_append_and_load_trial_log_round_trip(tmp_path) -> None:
    result = run_factor_experiment(_make_prices(), _momentum_fn)
    row = trial_row_from_result(result, experiment_name="momentum_h1")
    path = tmp_path / "trial_log.csv"
    append_trial_log(row, path)
    loaded = load_trial_log(path)
    assert len(loaded) == 1
    assert list(loaded.columns) == list(TRIAL_LOG_COLUMNS)


def test_append_trial_log_rejects_schema_drift(tmp_path) -> None:
    bad = pd.DataFrame([{"foo": 1}])
    with pytest.raises(ValueError, match="missing required columns"):
        append_trial_log(bad, tmp_path / "trial_log.csv")
