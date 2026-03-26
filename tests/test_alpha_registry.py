from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.alpha_registry import (
    AlphaRegistryEntry,
    alpha_entry_from_experiment,
    alpha_registry_stage_summary,
    load_alpha_registry,
    upsert_alpha_registry_entry,
)
from alpha_lab.experiment import run_factor_experiment
from alpha_lab.factors.momentum import momentum


def _prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=40, freq="B")
    rows: list[dict[str, object]] = []
    for asset in ["A", "B", "C", "D"]:
        px = 100.0
        for d in dates:
            px *= 1.001
            rows.append({"date": d, "asset": asset, "close": px})
    return pd.DataFrame(rows)


def _factor_fn(prices: pd.DataFrame) -> pd.DataFrame:
    return momentum(prices, window=5)


def test_upsert_and_load_alpha_registry(tmp_path) -> None:
    path = tmp_path / "alpha_registry.csv"
    entry = AlphaRegistryEntry(alpha_id="alpha_1", lifecycle_stage="candidate")
    out = upsert_alpha_registry_entry(entry, path=path)
    loaded = load_alpha_registry(path)
    assert len(out) == 1
    assert len(loaded) == 1
    assert loaded["alpha_id"].iloc[0] == "alpha_1"


def test_upsert_replaces_same_alpha_id(tmp_path) -> None:
    path = tmp_path / "alpha_registry.csv"
    upsert_alpha_registry_entry(
        AlphaRegistryEntry(alpha_id="alpha_1", lifecycle_stage="candidate", notes="v1"),
        path=path,
    )
    out = upsert_alpha_registry_entry(
        AlphaRegistryEntry(alpha_id="alpha_1", lifecycle_stage="mature", notes="v2"),
        path=path,
    )
    assert len(out) == 1
    assert out["lifecycle_stage"].iloc[0] == "mature"
    assert out["notes"].iloc[0] == "v2"


def test_alpha_entry_from_experiment_builds_registry_row() -> None:
    result = run_factor_experiment(_prices(), _factor_fn, generate_factor_report=True)
    entry = alpha_entry_from_experiment(result, alpha_id="mom_5d", lifecycle_stage="candidate")
    assert entry.alpha_id == "mom_5d"
    assert entry.lifecycle_stage == "candidate"


def test_alpha_registry_stage_summary_counts() -> None:
    df = pd.DataFrame(
        {
            "alpha_id": ["a", "b", "c"],
            "lifecycle_stage": ["candidate", "candidate", "retired"],
            "ic_mean": [0.01, 0.02, -0.01],
            "ic_ir": [0.3, 0.4, -0.2],
        }
    )
    out = alpha_registry_stage_summary(df)
    cand = out.set_index("lifecycle_stage").loc["candidate", "n_alphas"]
    assert cand == 2


def test_alpha_registry_entry_rejects_invalid_stage() -> None:
    with pytest.raises(ValueError, match="lifecycle_stage"):
        AlphaRegistryEntry(alpha_id="x", lifecycle_stage="invalid")

