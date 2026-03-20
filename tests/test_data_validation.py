"""Tests for alpha_lab.data_validation.validate_price_panel."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.data_validation import REQUIRED_PRICE_COLUMNS, validate_price_panel

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_prices(n_assets: int = 3, n_days: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows = []
    for asset in [f"A{i}" for i in range(n_assets)]:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


def test_valid_panel_passes() -> None:
    validate_price_panel(_make_prices())


def test_extra_columns_are_allowed() -> None:
    df = _make_prices()
    df["volume"] = 1_000_000
    validate_price_panel(df)  # should not raise


# ---------------------------------------------------------------------------
# 2. Required columns
# ---------------------------------------------------------------------------


def test_missing_all_required_columns_raises() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        validate_price_panel(pd.DataFrame({"x": [1]}))


def test_missing_close_column_raises() -> None:
    df = _make_prices().drop(columns=["close"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_price_panel(df)


def test_missing_asset_column_raises() -> None:
    df = _make_prices().drop(columns=["asset"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_price_panel(df)


def test_missing_date_column_raises() -> None:
    df = _make_prices().drop(columns=["date"])
    with pytest.raises(ValueError, match="missing required columns"):
        validate_price_panel(df)


# ---------------------------------------------------------------------------
# 3. Empty DataFrame
# ---------------------------------------------------------------------------


def test_empty_dataframe_raises() -> None:
    df = pd.DataFrame(columns=["date", "asset", "close"])
    with pytest.raises(ValueError, match="empty"):
        validate_price_panel(df)


# ---------------------------------------------------------------------------
# 4. NaT / unparseable dates
# ---------------------------------------------------------------------------


def test_nat_date_raises() -> None:
    df = _make_prices()
    df.iloc[0, df.columns.get_loc("date")] = pd.NaT
    with pytest.raises(ValueError, match="NaT"):
        validate_price_panel(df)


def test_unparseable_date_string_raises() -> None:
    df = pd.DataFrame(
        {"date": ["2024-01-01", "not-a-date"], "asset": ["A", "A"], "close": [100.0, 101.0]}
    )
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    with pytest.raises(ValueError, match="NaT"):
        validate_price_panel(df)


# ---------------------------------------------------------------------------
# 5. Null / empty asset strings
# ---------------------------------------------------------------------------


def test_null_asset_raises() -> None:
    df = _make_prices()
    df.iloc[0, df.columns.get_loc("asset")] = None
    with pytest.raises(ValueError, match="null"):
        validate_price_panel(df)


def test_empty_string_asset_raises() -> None:
    df = _make_prices()
    df.iloc[0, df.columns.get_loc("asset")] = "   "
    with pytest.raises(ValueError, match="empty string"):
        validate_price_panel(df)


# ---------------------------------------------------------------------------
# 6. Duplicate (date, asset)
# ---------------------------------------------------------------------------


def test_duplicate_date_asset_raises() -> None:
    df = _make_prices()
    dupe = df.iloc[[0]].copy()
    df = pd.concat([df, dupe], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_price_panel(df)


# ---------------------------------------------------------------------------
# 7. NaN close
# ---------------------------------------------------------------------------


def test_nan_close_raises() -> None:
    df = _make_prices()
    df.iloc[0, df.columns.get_loc("close")] = float("nan")
    with pytest.raises(ValueError, match="NaN"):
        validate_price_panel(df)


# ---------------------------------------------------------------------------
# 8. Non-positive close
# ---------------------------------------------------------------------------


def test_zero_close_raises() -> None:
    df = _make_prices()
    df.iloc[0, df.columns.get_loc("close")] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        validate_price_panel(df)


def test_negative_close_raises() -> None:
    df = _make_prices()
    df.iloc[0, df.columns.get_loc("close")] = -5.0
    with pytest.raises(ValueError, match="non-positive"):
        validate_price_panel(df)


# ---------------------------------------------------------------------------
# 9. Non-DataFrame input
# ---------------------------------------------------------------------------


def test_non_dataframe_raises() -> None:
    with pytest.raises(ValueError, match="DataFrame"):
        validate_price_panel([1, 2, 3])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 10. REQUIRED_PRICE_COLUMNS constant
# ---------------------------------------------------------------------------


def test_required_columns_constant() -> None:
    assert REQUIRED_PRICE_COLUMNS == frozenset({"date", "asset", "close"})
