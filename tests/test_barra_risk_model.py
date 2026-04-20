from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.risk_model.barra import (
    BarraExposures,
    build_barra_exposures,
    estimate_factor_returns,
)


def _build_synthetic_inputs(
    n_assets: int = 6,
    n_days: int = 320,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.date_range("2023-01-02", periods=n_days, freq="B")
    assets = [f"A{i}" for i in range(n_assets)]
    price_rows: list[dict[str, object]] = []
    basic_rows: list[dict[str, object]] = []
    for idx, asset in enumerate(assets):
        px = 20.0 + idx
        for t, date in enumerate(dates):
            px = px * (1.0 + 0.0004 * (idx + 1) + 0.002 * np.sin(t / 20.0))
            price_rows.append({"date": date, "asset": asset, "close": px})
            basic_rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "circ_mv": 1e9 * (1.0 + idx * 0.1),
                    "pb": 1.2 + 0.15 * idx,
                    "industry": "Bank" if idx % 2 == 0 else "Tech",
                }
            )
    return pd.DataFrame(price_rows), pd.DataFrame(basic_rows)


def test_build_barra_exposures_returns_expected_columns() -> None:
    prices, basic = _build_synthetic_inputs()
    out = build_barra_exposures(prices, basic, industry_col="industry")

    assert isinstance(out, BarraExposures)
    assert set(["date", "asset", "circ_mv"]).issubset(out.exposures.columns)
    for col in ("size", "value", "momentum", "volatility", "beta"):
        assert col in out.exposures.columns
    assert len(out.industry_factors) >= 1
    assert out.exposures.duplicated(subset=["date", "asset"]).sum() == 0


def test_estimate_factor_returns_recovers_known_linear_coefficients() -> None:
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    assets = ["A", "B", "C", "D", "E"]
    rows: list[dict[str, object]] = []
    ret_rows: list[dict[str, object]] = []
    for date in dates:
        for idx, asset in enumerate(assets):
            size = float(idx + 1)
            value = float((idx % 2) * 2 - 1)
            ret = 0.02 * size - 0.01 * value
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "circ_mv": 1e9 + idx * 1e8,
                    "size": size,
                    "value": value,
                }
            )
            ret_rows.append({"date": date, "asset": asset, "value": ret})

    exposures = BarraExposures(
        exposures=pd.DataFrame(rows),
        style_factors=("size", "value"),
        industry_factors=(),
    )
    factor_returns = estimate_factor_returns(exposures, pd.DataFrame(ret_rows))
    pivot = factor_returns.pivot(index="date", columns="factor", values="factor_return")

    assert np.allclose(pivot["size"].to_numpy(dtype=float), 0.02, atol=1e-8)
    assert np.allclose(pivot["value"].to_numpy(dtype=float), -0.01, atol=1e-8)
