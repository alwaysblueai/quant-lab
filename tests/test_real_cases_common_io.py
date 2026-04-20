from __future__ import annotations

from pathlib import Path

import pytest

from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.real_cases.common_io import load_universe_mask
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
