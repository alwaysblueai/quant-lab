"""Smoke coverage for :mod:`alpha_lab.research_contracts`."""

from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.research_contracts import (
    validate_canonical_signal_table,
    validate_prices_table,
)


def _valid_prices() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "asset": ["A", "B", "A", "B"],
            "close": [10.0, 20.0, 11.0, 21.0],
        }
    )


def _valid_signal() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "asset": ["A", "B", "A", "B"],
            "factor": ["smoke"] * 4,
            "value": [0.1, -0.2, 0.3, -0.4],
        }
    )


def test_validate_prices_table_accepts_canonical() -> None:
    validate_prices_table(_valid_prices())


def test_validate_prices_table_rejects_missing_close() -> None:
    bad = _valid_prices().drop(columns=["close"])
    with pytest.raises((ValueError, AlphaLabDataError)):
        validate_prices_table(bad)


def test_validate_canonical_signal_table_accepts_canonical() -> None:
    validate_canonical_signal_table(_valid_signal())


def test_validate_canonical_signal_table_wraps_to_alpha_lab_data_error() -> None:
    bad = _valid_signal().drop(columns=["factor"])
    with pytest.raises(AlphaLabDataError, match="signal violates"):
        validate_canonical_signal_table(bad)


def test_validate_canonical_signal_table_custom_label_in_message() -> None:
    bad = _valid_signal().drop(columns=["factor"])
    with pytest.raises(AlphaLabDataError, match="alpha_signal"):
        validate_canonical_signal_table(bad, table_name="alpha_signal")
