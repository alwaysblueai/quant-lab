from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.factor_selection import screen_factors


def _synthetic_factor_data(seed: int = 123) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=30, freq="B")
    assets = [f"A{i:03d}" for i in range(40)]
    rows_f: list[dict[str, object]] = []
    rows_l: list[dict[str, object]] = []
    for date in dates:
        latent = rng.normal(0, 1, size=len(assets))
        label = latent + rng.normal(0, 0.5, size=len(assets))
        f1 = latent + rng.normal(0, 0.2, size=len(assets))
        f2 = f1 + rng.normal(0, 0.02, size=len(assets))  # highly redundant
        f3 = rng.normal(0, 1, size=len(assets))
        miss = rng.random(len(assets)) < 0.5
        f3 = np.where(miss, np.nan, f3)

        for i, asset in enumerate(assets):
            rows_l.append(
                {
                    "date": date,
                    "asset": asset,
                    "factor": "forward_label",
                    "value": float(label[i]),
                }
            )
            rows_f.append({"date": date, "asset": asset, "factor": "f1", "value": float(f1[i])})
            rows_f.append({"date": date, "asset": asset, "factor": "f2", "value": float(f2[i])})
            rows_f.append({"date": date, "asset": asset, "factor": "f3", "value": float(f3[i])})
    return pd.DataFrame(rows_f), pd.DataFrame(rows_l)


def test_screen_factors_outputs_expected_sections() -> None:
    factors, labels = _synthetic_factor_data()
    report = screen_factors(
        factors,
        labels,
        min_coverage=0.7,
        min_abs_monotonicity=0.05,
        max_pairwise_corr=0.95,
        max_vif=8.0,
    )
    assert {"factor", "coverage", "rank_ic_mean"}.issubset(report.summary.columns)
    assert {"factor_a", "factor_b", "corr"}.issubset(report.pairwise_correlation.columns)
    assert {"factor", "vif"}.issubset(report.vif.columns)
    assert {"factor", "marginal_delta"}.issubset(report.marginal_contribution.columns)


def test_screen_factors_flags_weak_and_redundant_candidates() -> None:
    factors, labels = _synthetic_factor_data()
    report = screen_factors(
        factors,
        labels,
        min_coverage=0.7,
        min_abs_monotonicity=0.05,
        max_pairwise_corr=0.95,
        max_vif=8.0,
    )
    decisions = report.decisions.set_index("factor")
    assert decisions.loc["f3", "decision"] == "weak_factor"
    assert decisions.loc["f2", "decision"] in {"redundant_factor", "weak_factor"}

