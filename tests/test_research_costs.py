from __future__ import annotations

import warnings

import pandas as pd

from alpha_lab.research_costs import layered_research_costs


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
            "asset": ["A", "B"],
            "trade_dollar": [1_000_000.0, -500_000.0],
            "adv_dollar": [20_000_000.0, 5_000_000.0],
            "daily_volatility": [0.02, 0.03],
        }
    )


def test_layered_research_costs_outputs_components() -> None:
    out = layered_research_costs(_trades())
    cols = set(out.per_trade.columns)
    assert "cost_flat_fee" in cols
    assert "cost_spread_proxy" in cols
    assert "cost_impact_proxy" in cols
    assert "cost_total" in cols
    assert "cost_total_bps" in cols
    assert out.summary["total_cost_dollar"].iloc[0] > 0


def test_layered_research_costs_all_zero_trade_group_no_runtime_warning() -> None:
    trades = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
            "asset": ["A", "B"],
            "trade_dollar": [0.0, 0.0],
            "adv_dollar": [20_000_000.0, 5_000_000.0],
            "daily_volatility": [0.02, 0.03],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        out = layered_research_costs(trades)
    assert pd.isna(float(out.by_date["p95_cost_bps"].iloc[0]))
