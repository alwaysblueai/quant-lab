from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.experiment import run_factor_experiment
from alpha_lab.reporting.purged_kfold_diagnostics import (
    PURGED_KFOLD_FOLDS_COLUMNS,
    build_purged_kfold_diagnostics,
)


def _demo_prices(n_assets: int = 8, n_days: int = 120, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows: list[dict[str, object]] = []
    for asset_idx in range(n_assets):
        asset = f"A{asset_idx:03d}"
        close = 100.0 + float(asset_idx)
        for date in dates:
            close *= 1.0 + float(rng.normal(0.0, 0.01))
            rows.append({"date": date, "asset": asset, "close": close})
    return pd.DataFrame(rows)


def _factor_fn(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices[["date", "asset", "close"]].copy()
    frame = frame.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
    frame["value"] = frame.groupby("asset", sort=False)["close"].pct_change(fill_method=None)
    frame["factor"] = "demo_factor"
    return frame[["date", "asset", "factor", "value"]]


def test_build_purged_kfold_diagnostics_outputs_summary_and_fold_table() -> None:
    result = run_factor_experiment(
        _demo_prices(),
        _factor_fn,
        horizon=5,
        n_quantiles=5,
    )
    diagnostics = build_purged_kfold_diagnostics(
        experiment_result=result,
        label_horizon=5,
        n_splits=5,
        embargo_pct=0.02,
    )

    assert diagnostics.summary["artifact_type"] == "alpha_lab_purged_kfold_summary"
    assert diagnostics.summary["status"] == "ok"
    assert diagnostics.summary["n_folds"] == 5
    assert diagnostics.summary["n_splits_used"] == 5
    assert "verdict" in diagnostics.summary
    assert tuple(diagnostics.folds.columns) == PURGED_KFOLD_FOLDS_COLUMNS
    assert len(diagnostics.folds) == 5


def test_build_purged_kfold_diagnostics_returns_not_available_when_dates_missing() -> None:
    result = run_factor_experiment(
        _demo_prices(),
        _factor_fn,
        horizon=5,
        n_quantiles=5,
    )
    result.ic_df = result.ic_df.iloc[0:0].copy()
    result.rank_ic_df = result.rank_ic_df.iloc[0:0].copy()
    result.long_short_df = result.long_short_df.iloc[0:0].copy()

    diagnostics = build_purged_kfold_diagnostics(
        experiment_result=result,
        label_horizon=5,
    )

    assert diagnostics.summary["status"] == "not_available"
    assert diagnostics.summary["verdict"] == "not_available"
    assert diagnostics.summary["n_folds"] == 0
    assert diagnostics.folds.empty
