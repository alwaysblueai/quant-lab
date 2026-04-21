from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.real_cases.common_io import load_prices, load_tabular_frame, load_universe_mask
from alpha_lab.real_cases.common_spec import UniverseSpec


def test_load_universe_mask_parses_string_flags_strictly(tmp_path: Path) -> None:
    universe_path = tmp_path / "universe.csv"
    universe_path.write_text(
        "\n".join(
            [
                "date,asset,in_universe",
                "2024-01-02,000001.SZ,1",
                "2024-01-02,000002.SZ,0",
                "2024-01-02,000003.SZ,true",
                "2024-01-02,000004.SZ,false",
            ]
        ),
        encoding="utf-8",
    )
    spec = UniverseSpec(path=str(universe_path), in_universe_column="in_universe")

    mask = load_universe_mask(spec)

    assert mask is not None
    values = mask.sort_values("asset", kind="mergesort")["in_universe"].tolist()
    assert values == [True, False, True, False]


def test_load_universe_mask_rejects_unsupported_flag_values(tmp_path: Path) -> None:
    universe_path = tmp_path / "universe.csv"
    universe_path.write_text(
        "\n".join(
            [
                "date,asset,in_universe",
                "2024-01-02,000001.SZ,2",
                "2024-01-02,000002.SZ,maybe",
            ]
        ),
        encoding="utf-8",
    )
    spec = UniverseSpec(path=str(universe_path), in_universe_column="in_universe")

    with pytest.raises(AlphaLabDataError):
        load_universe_mask(spec)


def test_load_tabular_frame_prefers_fresh_parquet_sibling_for_csv(tmp_path: Path) -> None:
    prices_csv = tmp_path / "prices.csv"
    prices_parquet = prices_csv.with_suffix(".parquet")
    pd.DataFrame({"source": ["csv"], "value": [1.0]}).to_csv(prices_csv, index=False)
    pd.DataFrame({"source": ["parquet"], "value": [2.0]}).to_parquet(prices_parquet, index=False)

    now = time.time()
    os.utime(prices_csv, (now, now))
    os.utime(prices_parquet, (now + 5.0, now + 5.0))

    loaded = load_tabular_frame(str(prices_csv), object_name="prices")

    assert loaded["source"].tolist() == ["parquet"]
    assert loaded["value"].tolist() == [2.0]


def test_load_tabular_frame_falls_back_to_csv_when_parquet_is_stale(tmp_path: Path) -> None:
    prices_csv = tmp_path / "prices.csv"
    prices_parquet = prices_csv.with_suffix(".parquet")
    pd.DataFrame({"source": ["csv"], "value": [1.0]}).to_csv(prices_csv, index=False)
    pd.DataFrame({"source": ["parquet"], "value": [2.0]}).to_parquet(prices_parquet, index=False)

    now = time.time()
    os.utime(prices_parquet, (now, now))
    os.utime(prices_csv, (now + 5.0, now + 5.0))

    loaded = load_tabular_frame(str(prices_csv), object_name="prices")

    assert loaded["source"].tolist() == ["csv"]
    assert loaded["value"].tolist() == [1.0]


def test_load_prices_applies_default_dividend_adjustment_when_column_present(
    tmp_path: Path,
) -> None:
    prices_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "asset": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "close": [10.0, 10.5, 10.8],
            "dividend_per_share": [0.0, 0.0, 0.2],
        }
    ).to_csv(prices_path, index=False)

    loaded = load_prices(str(prices_path))

    ratio = 1.0 - 0.2 / 10.5
    assert loaded.loc[0, "close"] == pytest.approx(10.0 * ratio)
    assert loaded.loc[1, "close"] == pytest.approx(10.5 * ratio)
    assert loaded.loc[2, "close"] == pytest.approx(10.8)


def test_load_prices_dividend_adjustment_skips_missing_rows(tmp_path: Path) -> None:
    prices_path = tmp_path / "prices.csv"
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "asset": ["000001.SZ", "000001.SZ", "000001.SZ"],
            "close": [10.0, 11.0, 12.0],
            "dividend_per_share": [float("nan"), float("nan"), 0.55],
        }
    ).to_csv(prices_path, index=False)

    loaded = load_prices(str(prices_path))

    assert loaded["close"].isna().sum() == 0
    ratio = 1.0 - 0.55 / 11.0
    assert loaded.loc[0, "close"] == pytest.approx(10.0 * ratio)
    assert loaded.loc[1, "close"] == pytest.approx(11.0 * ratio)
    assert loaded.loc[2, "close"] == pytest.approx(12.0)
