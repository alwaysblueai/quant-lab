from __future__ import annotations

import pandas as pd

from alpha_lab.baseline_factor_suite import (
    baseline_factor_suite_payload,
    baseline_required_columns_available,
    iter_baseline_factor_specs,
)
from alpha_lab.factor_recipe import build_factor_from_recipe_mapping


def _prices(n_days: int = 150) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    assets = ["a", "b", "c", "d", "e", "f"]
    rows: list[dict[str, object]] = []
    for asset_idx, asset in enumerate(assets):
        for date_idx, date in enumerate(dates):
            close = 10.0 + asset_idx + 0.03 * date_idx + 0.1 * ((date_idx + asset_idx) % 5)
            volume = 1_000_000 + 10_000 * asset_idx + 1_000 * date_idx
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "close": close,
                    "high": close * 1.02,
                    "low": close * 0.98,
                    "volume": volume,
                    "amount": volume * close,
                }
            )
    return pd.DataFrame(rows)


def test_baseline_factor_suite_separates_default_and_extended_specs() -> None:
    default_specs = iter_baseline_factor_specs()
    all_specs = iter_baseline_factor_specs(include_non_default=True)
    payload = baseline_factor_suite_payload(include_non_default=True)

    default_names = {spec.name for spec in default_specs}
    all_names = {spec.name for spec in all_specs}

    assert "mom_20d" in default_names
    assert "rev_5d" in default_names
    assert "vcimom_20_5" not in default_names
    assert "vcimom_20_5" in all_names
    assert len(payload) == len(all_specs)
    assert {item["name"] for item in payload} == all_names


def test_default_baseline_factor_suite_builds_canonical_factors() -> None:
    prices = _prices()

    for spec in iter_baseline_factor_specs():
        assert baseline_required_columns_available(spec, prices.columns)
        factor = build_factor_from_recipe_mapping(
            prices=prices,
            recipe=spec.recipe,
            factor_name=spec.name,
        )
        assert list(factor.columns) == ["date", "asset", "factor", "value"]
        assert set(factor["factor"]) == {spec.name}
        assert factor["value"].notna().sum() > 0


def test_baseline_required_columns_allow_vcimom_amount_or_volume() -> None:
    spec = next(
        item
        for item in iter_baseline_factor_specs(include_non_default=True)
        if item.name == "vcimom_20_5"
    )

    assert baseline_required_columns_available(spec, ["date", "asset", "close", "amount"])
    assert baseline_required_columns_available(spec, ["date", "asset", "close", "volume"])
    assert not baseline_required_columns_available(spec, ["date", "asset", "close"])
