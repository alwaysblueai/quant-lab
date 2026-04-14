from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def write_demo_model_factor_case(
    tmp_path: Path,
    *,
    factor_name: str = "model_alpha",
    direction: str = "long",
    enable_neutralization: bool = False,
    include_known_at: bool = True,
) -> Path:
    """Create a fully runnable synthetic model-factor spec and input files."""

    data_dir = tmp_path / "inputs"
    data_dir.mkdir(parents=True, exist_ok=True)

    prices, features, universe, exposures = _synthetic_case_tables()

    prices_path = data_dir / "prices.csv"
    features_path = data_dir / "features.csv"
    universe_path = data_dir / "universe.csv"
    exposures_path = data_dir / "exposures.csv"

    if not include_known_at:
        features = features.drop(columns=["known_at"])

    prices.to_csv(prices_path, index=False)
    features.to_csv(features_path, index=False)
    universe.to_csv(universe_path, index=False)
    exposures.to_csv(exposures_path, index=False)

    spec = {
        "name": f"demo_{factor_name}_model_factor",
        "factor_name": factor_name,
        "features_path": str(features_path),
        "feature_columns": ["feature_momentum", "feature_quality", "feature_noise"],
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
        "feature_preprocess": {
            "missing_policy": "median_impute",
            "scale_features": "auto",
        },
        "model": {
            "family": "ridge",
            "params": {"alpha": 1.0},
        },
        "training": {
            "window_type": "rolling",
            "train_window_n_dates": 45,
            "min_train_dates": 20,
            "min_train_rows": 120,
            "retrain_every_n_dates": 5,
            "min_score_assets": 5,
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

    spec_path = tmp_path / f"{factor_name}_model_factor_case.yaml"
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return spec_path


def _synthetic_case_tables(
    *,
    n_assets: int = 12,
    n_days: int = 100,
    seed: int = 20260401,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    assets = [f"A{i:03d}" for i in range(n_assets)]

    rows_price: list[dict[str, object]] = []
    rows_feature: list[dict[str, object]] = []
    rows_universe: list[dict[str, object]] = []
    rows_exposure: list[dict[str, object]] = []

    for i, asset in enumerate(assets):
        price = 50.0 + i
        momentum_latent = rng.normal(0.0, 1.0, size=n_days)
        quality_latent = rng.normal(0.0, 1.0, size=n_days)
        industry = ["IND_A", "IND_B", "IND_C"][i % 3]

        for t, date in enumerate(dates):
            pred = 0.8 * momentum_latent[t - 1] + 0.4 * quality_latent[t - 1] if t > 0 else 0.0
            ret = 0.0025 * pred + rng.normal(0.0, 0.01)
            price = max(price * (1.0 + ret), 1.0)

            rows_price.append({"date": date, "asset": asset, "close": float(price)})
            rows_feature.append(
                {
                    "date": date,
                    "asset": asset,
                    "known_at": date,
                    "feature_momentum": float(momentum_latent[t]),
                    "feature_quality": float(quality_latent[t]),
                    "feature_noise": float(rng.normal(0.0, 1.0)),
                }
            )
            rows_universe.append(
                {
                    "date": date,
                    "asset": asset,
                    "in_universe": not (i == 0 and t < 5),
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
        pd.DataFrame(rows_feature),
        pd.DataFrame(rows_universe),
        pd.DataFrame(rows_exposure),
    )
