from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.decay import compute_ic_decay
from alpha_lab.evaluation import compute_ic, compute_ic_summary, compute_rank_ic
from alpha_lab.labels import forward_return


def _inventory_prices_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=30)
    rows: list[dict[str, object]] = []
    for asset_idx in range(5):
        asset = f"A{asset_idx}"
        price = 100.0 + asset_idx
        for date_idx, date in enumerate(dates):
            daily_return = 0.0005 * (asset_idx - 2) + 0.0001 * (date_idx % 7)
            price *= 1.0 + daily_return
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


def test_inventory_target_horizon_ic_decay_matches_public_ic_paths() -> None:
    horizon = 5
    prices = _inventory_prices_panel()
    labels = forward_return(prices, horizon=horizon)
    factor_df = labels.copy()
    factor_df["factor"] = "perfect_forward_return_proxy"
    asset_offsets = {
        "A0": 0.00015,
        "A1": -0.00005,
        "A2": 0.00020,
        "A3": -0.00010,
        "A4": 0.0,
    }
    label_values = pd.to_numeric(factor_df["value"], errors="coerce")
    factor_df["value"] = label_values.fillna(0.0) + factor_df["asset"].map(asset_offsets)

    direct_ic = compute_ic(factor_df, labels)
    direct_rank_ic = compute_rank_ic(factor_df, labels)
    direct_summary = compute_ic_summary(pd.to_numeric(direct_ic["ic"], errors="coerce"))
    direct_rank_values = pd.to_numeric(direct_rank_ic["rank_ic"], errors="coerce").dropna()
    expected_n_dates = prices["date"].nunique() - horizon

    cached_decay = compute_ic_decay(
        factor_df,
        prices,
        horizons=(horizon,),
        precomputed_labels_by_horizon={horizon: labels},
    )
    fast_decay = compute_ic_decay(factor_df, prices, horizons=(horizon,))

    for decay_df in (cached_decay, fast_decay):
        row = decay_df.iloc[0]
        assert int(row["horizon"]) == horizon
        assert int(row["n_dates"]) == expected_n_dates
        assert row["mean_ic"] == pytest.approx(float(direct_summary["mean_ic"]))
        assert row["ic_ir"] == pytest.approx(float(direct_summary["ic_ir"]))
        assert row["t_stat"] == pytest.approx(float(direct_summary["t_stat"]))
        assert row["p_value"] == pytest.approx(float(direct_summary["p_value"]))
        assert row["mean_rank_ic"] == pytest.approx(float(direct_rank_values.mean()))
