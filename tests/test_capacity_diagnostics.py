from __future__ import annotations

import pandas as pd

from alpha_lab.capacity_diagnostics import run_capacity_diagnostics


def _trades() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
            ],
            "asset": ["A", "B", "A"],
            "trade_dollar": [1_000_000.0, 100_000.0, 500_000.0],
            "adv_dollar": [10_000_000.0, 1_000_000.0, 5_000_000.0],
            "target_weight": [0.08, 0.01, 0.06],
        }
    )


def test_run_capacity_diagnostics_outputs_tables() -> None:
    out = run_capacity_diagnostics(
        _trades(),
        portfolio_value=20_000_000.0,
        max_adv_participation=0.05,
        concentration_weight_threshold=0.05,
    )
    assert {"adv_participation", "adv_limit_flag"}.issubset(out.adv_penetration.columns)
    assert {"turnover", "turnover_x_liquidity"}.issubset(out.turnover_liquidity.columns)
    assert "concentration_liquidity_flag" in out.concentration_flags.columns
    assert "n_adv_limit_flags" in out.warnings.columns

