from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def write_demo_composite_case(tmp_path: Path, *, enable_neutralization: bool = False) -> Path:
    """Create a fully runnable synthetic composite-case spec and input files."""

    data_dir = tmp_path / "inputs"
    data_dir.mkdir(parents=True, exist_ok=True)

    prices_path = data_dir / "prices.csv"
    bp_path = data_dir / "bp.csv"
    roe_path = data_dir / "roe_ttm.csv"
    rv_path = data_dir / "resid_vol_60d.csv"
    universe_path = data_dir / "universe.csv"
    exposures_path = data_dir / "exposures.csv"

    prices, bp, roe, rv, universe, exposures = _synthetic_case_tables()

    prices.to_csv(prices_path, index=False)
    bp.to_csv(bp_path, index=False)
    roe.to_csv(roe_path, index=False)
    rv.to_csv(rv_path, index=False)
    universe.to_csv(universe_path, index=False)
    exposures.to_csv(exposures_path, index=False)

    spec = {
        "name": "demo_value_quality_lowvol",
        "prices_path": str(prices_path),
        "rebalance_frequency": "W",
        "n_quantiles": 5,
        "universe": {
            "name": "demo_universe",
            "path": str(universe_path),
            "in_universe_column": "in_universe",
        },
        "target": {"kind": "forward_return", "horizon": 5},
        "components": [
            {
                "name": "bp",
                "path": str(bp_path),
                "factor": "bp",
                "weight": 0.4,
                "direction": "positive",
                "transform": "zscore",
            },
            {
                "name": "roe_ttm",
                "path": str(roe_path),
                "factor": "roe_ttm",
                "weight": 0.4,
                "direction": "positive",
                "transform": "zscore",
            },
            {
                "name": "resid_vol_60d",
                "path": str(rv_path),
                "factor": "resid_vol_60d",
                "weight": 0.2,
                "direction": "negative",
                "transform": "zscore",
            },
        ],
        "preprocess": {
            "winsorize": True,
            "winsorize_lower": 0.01,
            "winsorize_upper": 0.99,
            "min_group_size": 3,
            "min_coverage": 0.5,
        },
        "neutralization": {
            "enabled": enable_neutralization,
            "exposures_path": str(exposures_path),
            "size_col": "size_exposure",
            "industry_col": "industry",
            "beta_col": "beta_exposure",
            "min_obs": 5,
            "ridge": 1e-8,
        },
        "transaction_cost": {"one_way_rate": 0.001},
        "output": {"root_dir": str(tmp_path / "outputs")},
    }

    spec_path = tmp_path / "demo_composite_case.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return spec_path


def _synthetic_case_tables(
    *,
    n_assets: int = 12,
    n_days: int = 90,
    seed: int = 20260326,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    assets = [f"A{i:03d}" for i in range(n_assets)]

    rows_price: list[dict[str, object]] = []
    rows_bp: list[dict[str, object]] = []
    rows_roe: list[dict[str, object]] = []
    rows_rv: list[dict[str, object]] = []
    rows_universe: list[dict[str, object]] = []
    rows_exposure: list[dict[str, object]] = []

    for i, asset in enumerate(assets):
        price = 50.0 + i
        latent = rng.normal(0.0, 1.0, size=n_days)
        industry = ["IND_A", "IND_B", "IND_C"][i % 3]

        for t, date in enumerate(dates):
            pred = latent[t - 1] if t > 0 else 0.0
            ret = 0.0015 * pred + rng.normal(0.0, 0.01)
            price = max(price * (1.0 + ret), 1.0)

            bp = latent[t] + rng.normal(0.0, 0.25)
            roe = 0.7 * latent[t] + rng.normal(0.0, 0.25)
            resid_vol = -0.8 * latent[t] + rng.normal(0.0, 0.30)

            rows_price.append({"date": date, "asset": asset, "close": float(price)})
            rows_bp.append({"date": date, "asset": asset, "factor": "bp", "value": float(bp)})
            rows_roe.append(
                {
                    "date": date,
                    "asset": asset,
                    "factor": "roe_ttm",
                    "value": float(roe),
                }
            )
            rows_rv.append(
                {
                    "date": date,
                    "asset": asset,
                    "factor": "resid_vol_60d",
                    "value": float(resid_vol),
                }
            )

            rows_universe.append(
                {
                    "date": date,
                    "asset": asset,
                    "in_universe": not (i == 0 and t < 3),
                }
            )

            rows_exposure.append(
                {
                    "date": date,
                    "asset": asset,
                    "size_exposure": float(np.log(price)),
                    "beta_exposure": float(0.8 + (i % 5) * 0.1),
                    "industry": industry,
                }
            )

    return (
        pd.DataFrame(rows_price),
        pd.DataFrame(rows_bp),
        pd.DataFrame(rows_roe),
        pd.DataFrame(rows_rv),
        pd.DataFrame(rows_universe),
        pd.DataFrame(rows_exposure),
    )
