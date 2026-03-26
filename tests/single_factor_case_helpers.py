from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def write_demo_single_factor_case(
    tmp_path: Path,
    *,
    factor_name: str = "bp",
    direction: str = "long",
    enable_neutralization: bool = False,
) -> Path:
    """Create a fully runnable synthetic single-factor spec and input files."""

    data_dir = tmp_path / "inputs"
    data_dir.mkdir(parents=True, exist_ok=True)

    prices, factors, universe, exposures = _synthetic_case_tables(factor_name=factor_name)

    prices_path = data_dir / "prices.csv"
    factor_path = data_dir / f"{factor_name}.csv"
    universe_path = data_dir / "universe.csv"
    exposures_path = data_dir / "exposures.csv"

    prices.to_csv(prices_path, index=False)
    factors.to_csv(factor_path, index=False)
    universe.to_csv(universe_path, index=False)
    exposures.to_csv(exposures_path, index=False)

    spec = {
        "name": f"demo_{factor_name}_single_factor",
        "factor_name": factor_name,
        "factor_path": str(factor_path),
        "prices_path": str(prices_path),
        "rebalance_frequency": "W",
        "n_quantiles": 5,
        "direction": direction,
        "universe": {
            "name": "demo_universe",
            "path": str(universe_path),
            "in_universe_column": "in_universe",
        },
        "target": {"kind": "forward_return", "horizon": 5},
        "preprocess": {
            "winsorize": True,
            "winsorize_lower": 0.01,
            "winsorize_upper": 0.99,
            "standardization": "zscore",
            "min_group_size": 3,
            "min_coverage": 0.5,
        },
        "neutralization": {
            "enabled": enable_neutralization,
            "exposures_path": str(exposures_path),
            "size_col": "size_exposure",
            "industry_col": "industry",
            "min_obs": 5,
            "ridge": 1e-8,
        },
        "transaction_cost": {"one_way_rate": 0.001},
        "output": {"root_dir": str(tmp_path / "outputs")},
    }

    spec_path = tmp_path / f"{factor_name}_single_factor_case.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return spec_path


def _synthetic_case_tables(
    *,
    factor_name: str,
    n_assets: int = 12,
    n_days: int = 90,
    seed: int = 20260326,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    assets = [f"A{i:03d}" for i in range(n_assets)]

    rows_price: list[dict[str, object]] = []
    rows_factor: list[dict[str, object]] = []
    rows_universe: list[dict[str, object]] = []
    rows_exposure: list[dict[str, object]] = []

    for i, asset in enumerate(assets):
        price = 50.0 + i
        latent = rng.normal(0.0, 1.0, size=n_days)
        industry = ["IND_A", "IND_B", "IND_C"][i % 3]

        for t, date in enumerate(dates):
            pred = latent[t - 1] if t > 0 else 0.0
            ret = 0.0018 * pred + rng.normal(0.0, 0.01)
            price = max(price * (1.0 + ret), 1.0)

            factor_val = latent[t] + rng.normal(0.0, 0.25)

            rows_price.append({"date": date, "asset": asset, "close": float(price)})
            rows_factor.append(
                {
                    "date": date,
                    "asset": asset,
                    "factor": factor_name,
                    "value": float(factor_val),
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
                    "industry": industry,
                }
            )

    return (
        pd.DataFrame(rows_price),
        pd.DataFrame(rows_factor),
        pd.DataFrame(rows_universe),
        pd.DataFrame(rows_exposure),
    )
