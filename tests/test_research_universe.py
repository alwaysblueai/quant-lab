from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.research_universe import (
    ResearchUniverseRules,
    construct_research_universe,
)


def _prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    rows: list[dict[str, object]] = []
    for asset, base in [("A", 100.0), ("B", 50.0)]:
        for i, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "close": base + i,
                    "volume": 100_000 if asset == "A" else 500,
                }
            )
    return pd.DataFrame(rows)


def test_construct_research_universe_outputs_expected_tables() -> None:
    prices = _prices()
    metadata = pd.DataFrame(
        {
            "asset": ["A", "B"],
            "listing_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2024-01-03")],
            "is_st": [False, False],
        }
    )
    state = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-04"), pd.Timestamp("2024-01-05")],
            "asset": ["A", "A"],
            "is_halted": [True, False],
            "is_limit_locked": [False, True],
        }
    )
    result = construct_research_universe(
        prices,
        asset_metadata=metadata,
        market_state=state,
        rules=ResearchUniverseRules(
            min_listing_age_days=2,
            min_adv=1_000_000.0,
            adv_window=2,
        ),
    )

    assert set(result.universe.columns) == {"date", "asset", "in_universe"}
    assert set(result.tradability.columns) == {"date", "asset", "is_tradable"}
    assert set(result.exclusion_reasons.columns) == {"date", "asset", "reason", "detail"}
    assert {"n_assets", "n_in_universe", "n_tradable"}.issubset(result.diagnostics.columns)


def test_construct_research_universe_records_exclusion_reasons() -> None:
    prices = _prices()
    metadata = pd.DataFrame(
        {
            "asset": ["A", "B"],
            "listing_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2024-01-04")],
            "is_st": [False, True],
        }
    )
    result = construct_research_universe(
        prices,
        asset_metadata=metadata,
        rules=ResearchUniverseRules(min_listing_age_days=60),
    )
    reasons = set(result.exclusion_reasons["reason"].unique())
    assert "st_filter" in reasons
    assert "listing_age_or_missing_listing_date" in reasons


def test_construct_research_universe_is_deterministic() -> None:
    prices = _prices()
    metadata = pd.DataFrame(
        {
            "asset": ["A", "B"],
            "listing_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")],
        }
    )
    rules = ResearchUniverseRules(min_adv=0.0)
    first = construct_research_universe(
        prices.sample(frac=1.0, random_state=0).reset_index(drop=True),
        asset_metadata=metadata,
        rules=rules,
    )
    second = construct_research_universe(
        prices.sample(frac=1.0, random_state=1).reset_index(drop=True),
        asset_metadata=metadata,
        rules=rules,
    )
    pd.testing.assert_frame_equal(first.universe, second.universe)
    pd.testing.assert_frame_equal(first.tradability, second.tradability)
    pd.testing.assert_frame_equal(first.exclusion_reasons, second.exclusion_reasons)


def test_construct_research_universe_rejects_duplicate_market_state_keys() -> None:
    prices = _prices()
    state = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "asset": ["A", "A"],
            "is_halted": [True, False],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        construct_research_universe(prices, market_state=state)


def test_construct_research_universe_with_dollar_volume_column() -> None:
    prices = _prices().copy()
    prices["dollar_volume"] = np.linspace(1_000, 5_000, len(prices))
    metadata = pd.DataFrame(
        {
            "asset": ["A", "B"],
            "listing_date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")],
        }
    )
    result = construct_research_universe(
        prices,
        asset_metadata=metadata,
        rules=ResearchUniverseRules(min_adv=2_000.0, adv_window=2),
    )
    assert "min_adv_filter" in set(result.exclusion_reasons["reason"])
