from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.research_contracts import (
    ResearchBundle,
    validate_canonical_signal_table,
    validate_prices_table,
)


def _prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-01",
                "2024-01-02",
            ],
            "asset": ["A", "A", "B", "B"],
            "close": [100.0, 101.0, 50.0, 49.5],
        }
    )


def _signals(name: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
            "asset": ["A", "A", "B", "B"],
            "factor": [name, name, name, name],
            "value": [1.0, 1.5, -0.5, -0.2],
        }
    )


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
            "asset": ["A", "A", "B", "B"],
            "in_universe": [True, True, True, False],
        }
    )


def _tradability() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-01", "2024-01-02"],
            "asset": ["A", "A", "B", "B"],
            "is_tradable": [True, True, True, True],
        }
    )


def test_validate_prices_table_rejects_non_monotonic_asset_history() -> None:
    bad = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-01"],
            "asset": ["A", "A"],
            "close": [101.0, 100.0],
        }
    )
    with pytest.raises(ValueError, match="monotonic-increasing"):
        validate_prices_table(bad)


def test_validate_canonical_signal_table_rejects_duplicate_key() -> None:
    dup = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "asset": ["A", "A"],
            "factor": ["mom", "mom"],
            "value": [1.0, 2.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_canonical_signal_table(dup, table_name="factors")


def test_research_bundle_validate_success() -> None:
    bundle = ResearchBundle(
        prices=_prices(),
        factors=_signals("mom"),
        labels=_signals("forward_return_1"),
        universe=_universe(),
        tradability=_tradability(),
    )
    bundle.validate()


def test_research_bundle_rejects_universe_tradability_mismatch() -> None:
    tradability = _tradability().iloc[:-1].reset_index(drop=True)
    bundle = ResearchBundle(
        prices=_prices(),
        factors=_signals("mom"),
        labels=_signals("forward_return_1"),
        universe=_universe(),
        tradability=tradability,
    )
    with pytest.raises(ValueError, match="share the same"):
        bundle.validate()
