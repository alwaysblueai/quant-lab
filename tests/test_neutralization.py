from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.neutralization import (
    exposure_diagnostics,
    neutralize_industry,
    neutralize_signal,
    neutralize_size,
)


def _make_cross_section(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    rows: list[dict[str, object]] = []
    for date in dates:
        for i in range(30):
            size = float(rng.normal(0.0, 1.0))
            beta = float(rng.normal(1.0, 0.2))
            industry = "A" if i < 10 else ("B" if i < 20 else "C")
            ind_effect = {"A": 0.3, "B": -0.1, "C": 0.0}[industry]
            noise = float(rng.normal(0.0, 0.1))
            value = 1.5 * size + 0.8 * beta + ind_effect + noise
            rows.append(
                {
                    "date": date,
                    "asset": f"stk_{i:03d}",
                    "value": value,
                    "size": size,
                    "beta": beta,
                    "industry": industry,
                }
            )
    return pd.DataFrame(rows)


def test_neutralize_size_reduces_size_exposure() -> None:
    df = _make_cross_section()
    result = neutralize_size(df, size_col="size", min_obs=10)
    diag = result.diagnostics.set_index("exposure")
    assert "size_exposure" in diag.index
    assert (
        diag.loc["size_exposure", "mean_abs_corr_after"]
        < diag.loc["size_exposure", "mean_abs_corr_before"]
    )


def test_neutralize_industry_has_near_zero_group_mean() -> None:
    df = _make_cross_section()
    result = neutralize_industry(df, industry_col="industry", min_obs=10)
    means = (
        result.data.dropna(subset=["value_neutralized"])
        .groupby(["date", "industry"], sort=True)["value_neutralized"]
        .mean()
    )
    assert (means.abs() < 1e-6).all()


def test_neutralize_signal_combined_controls() -> None:
    df = _make_cross_section()
    result = neutralize_signal(
        df,
        size_col="size",
        beta_col="beta",
        industry_col="industry",
        min_obs=10,
    )
    assert "value_neutralized" in result.data.columns
    assert len(result.coefficients) > 0
    exps = set(result.diagnostics["exposure"])
    assert "size_exposure" in exps
    assert "beta_exposure" in exps


def test_exposure_diagnostics_empty_when_no_valid_dates() -> None:
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "v_before": [1.0, np.nan],
            "v_after": [np.nan, 1.0],
            "exp": [0.1, 0.2],
        }
    )
    out = exposure_diagnostics(
        df,
        value_before_col="v_before",
        value_after_col="v_after",
        exposure_cols=["exp"],
    )
    assert len(out) == 1
    assert pd.isna(out["mean_abs_corr_before"].iloc[0])

