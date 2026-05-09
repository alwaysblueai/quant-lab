from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.factor_recipe import (
    FactorRecipeError,
    build_factor_from_recipe_mapping,
    factor_registry,
)


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


def test_build_factor_from_recipe_mapping_forwards_extra_base_kwargs() -> None:
    prices = _sample_prices()
    seen: dict[str, object] = {}

    def passthrough_builder(
        df: pd.DataFrame,
        *,
        window: int = 20,
        skip_recent: int = 0,
        shock_gate_mode: str | None = None,
        neutralize_basic: bool | None = None,
    ) -> pd.DataFrame:
        seen.update(
            {
                "window": window,
                "skip_recent": skip_recent,
                "shock_gate_mode": shock_gate_mode,
                "neutralize_basic": neutralize_basic,
            }
        )
        out = df[["date", "asset"]].copy()
        out["factor"] = "passthrough"
        out["value"] = 1.0
        return out

    factor_registry.register("test_passthrough_kwargs", passthrough_builder)
    recipe = {
        "base": {
            "method": "test_passthrough_kwargs",
            "window": 5,
            "skip_recent": 1,
            "shock_gate_mode": "cs_quantile",
            "neutralize_basic": True,
        },
    }

    factor = build_factor_from_recipe_mapping(
        prices=prices,
        recipe=recipe,
        factor_name="passthrough_kwargs",
    )

    assert seen == {
        "window": 5,
        "skip_recent": 1,
        "shock_gate_mode": "cs_quantile",
        "neutralize_basic": True,
    }
    assert set(factor["factor"]) == {"passthrough_kwargs"}


def test_build_factor_from_recipe_mapping_accepts_legacy_lookback() -> None:
    prices = _sample_prices()
    factor_from_window = build_factor_from_recipe_mapping(
        prices=prices,
        recipe={"base": {"method": "momentum", "window": 4, "skip_recent": 1}},
        factor_name="mom4_window",
    )
    factor_from_lookback = build_factor_from_recipe_mapping(
        prices=prices,
        recipe={"base": {"method": "momentum", "lookback": 4, "skip_recent": 1}},
        factor_name="mom4_lookback",
    )

    pd.testing.assert_series_equal(
        factor_from_window["value"],
        factor_from_lookback["value"],
        check_names=False,
    )


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


def test_build_factor_from_recipe_mapping_preserves_daily_research_columns() -> None:
    prices = _sample_prices()
    prices["open"] = prices["close"] * 0.99
    prices["pre_close"] = prices.groupby("asset", sort=False)["close"].shift(1)
    prices["vwap"] = prices["close"] * 1.001
    prices["is_suspended"] = 0

    def open_close_gap(prices: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        assert {
            "open",
            "high",
            "low",
            "pre_close",
            "volume",
            "amount",
            "vwap",
            "is_suspended",
        } <= set(prices.columns)
        out = prices[["date", "asset"]].copy()
        out["value"] = prices["open"] / prices["close"] - 1.0
        return out

    factor_registry.register("open_close_gap_test", open_close_gap)

    factor = build_factor_from_recipe_mapping(
        prices=prices,
        recipe={"base": {"method": "open_close_gap_test"}},
        factor_name="open_close_gap",
    )

    assert list(factor.columns) == ["date", "asset", "factor", "value"]
    assert set(factor["factor"]) == {"open_close_gap"}
    assert factor["value"].notna().sum() == len(prices)


def test_build_factor_from_recipe_mapping_passes_custom_base_kwargs() -> None:
    prices = _sample_prices()
    captured: dict[str, object] = {}

    def custom_research_factor(
        prices: pd.DataFrame,
        *,
        window: int = 20,
        skip_recent: int = 0,
        min_periods: int | None = None,
        **kwargs: object,
    ) -> pd.DataFrame:
        captured["window"] = window
        captured["skip_recent"] = skip_recent
        captured["min_periods"] = min_periods
        captured.update(kwargs)
        out = prices[["date", "asset"]].copy()
        out["value"] = 1.0
        return out

    factor_registry.register("custom_kwargs_test", custom_research_factor)
    try:
        factor = build_factor_from_recipe_mapping(
            prices=prices,
            recipe={
                "base": {
                    "method": "custom_kwargs_test",
                    "window": 7,
                    "skip_recent": 1,
                    "shock_gate_mode": "cs_quantile",
                    "neutralize_basic": True,
                    "outside_event_policy": "nan",
                    "invert": False,
                    "exclude_limit": True,
                }
            },
            factor_name="custom_kwargs",
        )
    finally:
        factor_registry._builders.pop("custom_kwargs_test", None)

    assert set(factor["factor"]) == {"custom_kwargs"}
    assert captured == {
        "window": 7,
        "skip_recent": 1,
        "min_periods": None,
        "shock_gate_mode": "cs_quantile",
        "neutralize_basic": True,
        "outside_event_policy": "nan",
        "invert": False,
        "exclude_limit": True,
    }


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
