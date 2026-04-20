from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.factor_recipe import FactorRecipeError, build_factor_from_recipe_mapping


def _sample_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=18, freq="D")
    assets = ["000001.SZ", "000002.SZ", "000003.SZ", "600000.SH", "600004.SH", "600009.SH"]
    rows: list[dict[str, object]] = []
    for asset_idx, asset in enumerate(assets):
        for date_idx, date in enumerate(dates):
            base = 10.0 + float(asset_idx) * 2.0
            drift = 0.03 + 0.005 * float(asset_idx)
            seasonal = 0.2 * np.sin(float(date_idx) / 3.0 + float(asset_idx))
            close = base + drift * float(date_idx) + seasonal
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "asset": asset,
                    "close": close,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "volume": 1_000_000 + asset_idx * 100_000 + date_idx * 10_000,
                    "amount": (1_000_000 + asset_idx * 100_000 + date_idx * 10_000) * close,
                }
            )
    return pd.DataFrame(rows)


def test_build_factor_from_recipe_mapping_supports_preprocess() -> None:
    prices = _sample_prices()
    recipe = {
        "base": {
            "method": "momentum",
            "window": 3,
        },
        "preprocess": {
            "winsorize": {
                "enabled": True,
                "lower": 0.05,
                "upper": 0.95,
                "min_group_size": 3,
            },
            "standardization": {"method": "zscore", "min_group_size": 3},
            "min_coverage": 0.4,
        },
    }

    factor = build_factor_from_recipe_mapping(
        prices=prices,
        recipe=recipe,
        factor_name="mom3_zscore",
    )

    assert list(factor.columns) == ["date", "asset", "factor", "value"]
    assert set(factor["factor"]) == {"mom3_zscore"}
    assert factor["value"].notna().sum() > 0


def test_build_factor_from_recipe_mapping_supports_skip_recent_momentum() -> None:
    prices = _sample_prices()
    recipe = {
        "base": {
            "method": "momentum",
            "window": 5,
            "skip_recent": 2,
        },
    }

    factor = build_factor_from_recipe_mapping(
        prices=prices,
        recipe=recipe,
        factor_name="mom5_ex2",
    )

    assert list(factor.columns) == ["date", "asset", "factor", "value"]
    assert set(factor["factor"]) == {"mom5_ex2"}
    assert factor["value"].notna().sum() > 0


def test_build_factor_from_recipe_mapping_supports_amplitude() -> None:
    prices = _sample_prices()
    recipe = {"base": {"method": "amplitude", "window": 5}}

    factor = build_factor_from_recipe_mapping(
        prices=prices,
        recipe=recipe,
        factor_name="amp5",
    )

    assert list(factor.columns) == ["date", "asset", "factor", "value"]
    assert set(factor["factor"]) == {"amp5"}
    assert factor["value"].notna().sum() > 0


def test_build_factor_from_recipe_mapping_supports_downside_volatility() -> None:
    prices = _sample_prices()
    recipe = {"base": {"method": "downside_volatility", "window": 5}}

    factor = build_factor_from_recipe_mapping(
        prices=prices,
        recipe=recipe,
        factor_name="dvol5",
    )

    assert list(factor.columns) == ["date", "asset", "factor", "value"]
    assert set(factor["factor"]) == {"dvol5"}
    assert factor["value"].notna().sum() > 0


def test_build_factor_from_recipe_mapping_supports_orthogonalize() -> None:
    prices = _sample_prices()
    recipe = {
        "base": {"method": "momentum", "window": 3},
        "orthogonalize": {
            "enabled": True,
            "exposures": [{"method": "low_volatility", "window": 5}],
            "min_obs": 4,
            "ridge": 1e-8,
        },
    }

    factor = build_factor_from_recipe_mapping(
        prices=prices,
        recipe=recipe,
        factor_name="mom3_resid_lv5",
    )
    assert not factor.empty
    assert factor["value"].notna().sum() > 0


def test_build_factor_from_recipe_mapping_rejects_unknown_method() -> None:
    prices = _sample_prices()
    recipe = {"base": {"method": "not_a_factor"}}
    with pytest.raises(FactorRecipeError, match="unsupported base.method"):
        build_factor_from_recipe_mapping(
            prices=prices,
            recipe=recipe,
            factor_name="bad_factor",
        )


def test_build_factor_from_recipe_mapping_rejects_recipe_signal_direction() -> None:
    prices = _sample_prices()
    recipe = {
        "base": {"method": "momentum", "window": 3},
        "signal": {"direction": "short"},
    }
    with pytest.raises(FactorRecipeError, match="recipe.signal.direction"):
        build_factor_from_recipe_mapping(
            prices=prices,
            recipe=recipe,
            factor_name="bad_signal_direction",
        )


def test_build_factor_from_recipe_mapping_supports_vcimom() -> None:
    prices = _sample_prices()
    recipe = {
        "base": {
            "method": "vcimom",
            "residual_window": 8,
            "momentum_window": 6,
            "skip_recent": 2,
            "confirm_window": 4,
            "penalty_window": 3,
            "amount_window": 5,
            "confirm_weight": 0.5,
            "penalty_weight": 0.3,
        }
    }

    factor = build_factor_from_recipe_mapping(
        prices=prices,
        recipe=recipe,
        factor_name="vcimom20_5",
    )

    assert list(factor.columns) == ["date", "asset", "factor", "value"]
    assert set(factor["factor"]) == {"vcimom20_5"}
    assert factor["value"].notna().sum() > 0
