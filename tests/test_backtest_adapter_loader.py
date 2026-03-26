from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.backtest_adapter.loader import load_backtest_input_bundle
from alpha_lab.experiment import run_factor_experiment
from alpha_lab.factors.momentum import momentum
from alpha_lab.handoff import export_handoff_artifact
from alpha_lab.timing import DelaySpec


def _make_prices(n_assets: int = 5, n_days: int = 45, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows: list[dict[str, object]] = []
    for asset in [f"A{i}" for i in range(n_assets)]:
        px = 100.0
        for date in dates:
            px *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": px})
    return pd.DataFrame(rows)


def _factor_fn(prices: pd.DataFrame) -> pd.DataFrame:
    return momentum(prices, window=5)


def _bundle_path(tmp_path: Path) -> Path:
    prices = _make_prices()
    result = run_factor_experiment(
        prices,
        _factor_fn,
        horizon=5,
        delay_spec=DelaySpec.for_horizon(5, execution_delay_periods=1),
    )
    base = prices[["date", "asset"]].drop_duplicates().reset_index(drop=True)
    universe = base.copy()
    universe["in_universe"] = True
    tradability = base.copy()
    tradability["is_tradable"] = True
    export = export_handoff_artifact(
        result,
        output_dir=tmp_path,
        artifact_name="adapter_loader_bundle",
        universe_df=universe,
        tradability_df=tradability,
    )
    return export.artifact_path


def test_loader_loads_valid_schema_v2_bundle(tmp_path: Path) -> None:
    artifact_path = _bundle_path(tmp_path)
    bundle = load_backtest_input_bundle(artifact_path)
    assert bundle.schema_version == "2.0.0"
    assert bundle.signal_name == bundle.portfolio_construction.signal_name
    assert len(bundle.signal_snapshot_df) > 0
    assert set(bundle.signal_snapshot_df.columns) == {
        "date",
        "asset",
        "signal_name",
        "signal_value",
    }


def test_loader_fails_on_missing_required_file(tmp_path: Path) -> None:
    artifact_path = _bundle_path(tmp_path)
    (artifact_path / "tradability_mask.csv").unlink()
    with pytest.raises(ValueError, match="missing required file"):
        load_backtest_input_bundle(artifact_path)


def test_loader_schema_mismatch_failure_path(tmp_path: Path) -> None:
    artifact_path = _bundle_path(tmp_path)
    manifest_path = artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "1.0.0"
    manifest.pop("bundle_type", None)
    manifest.pop("bundle_components", None)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="supports only handoff schema"):
        load_backtest_input_bundle(artifact_path)


def test_loader_fails_on_malformed_bundle_json(tmp_path: Path) -> None:
    artifact_path = _bundle_path(tmp_path)
    target = artifact_path / "portfolio_construction.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["weight_method"] = "not_supported"
    target.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_backtest_input_bundle(artifact_path)


def test_loader_accepts_universe_tradability_superset_of_signal_keys(tmp_path: Path) -> None:
    artifact_path = _bundle_path(tmp_path)
    signal = pd.read_csv(artifact_path / "signal_snapshot.csv")
    date = signal["date"].iloc[0]

    universe = pd.read_csv(artifact_path / "universe_mask.csv")
    tradability = pd.read_csv(artifact_path / "tradability_mask.csv")
    universe = pd.concat(
        [
            universe,
            pd.DataFrame([{"date": date, "asset": "EXTRA_ASSET", "in_universe": True}]),
        ],
        ignore_index=True,
    )
    tradability = pd.concat(
        [
            tradability,
            pd.DataFrame([{"date": date, "asset": "EXTRA_ASSET", "is_tradable": True}]),
        ],
        ignore_index=True,
    )
    universe.to_csv(artifact_path / "universe_mask.csv", index=False)
    tradability.to_csv(artifact_path / "tradability_mask.csv", index=False)
    manifest_path = artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for file_name in ("universe_mask.csv", "tradability_mask.csv"):
        file_path = artifact_path / file_name
        manifest["files"][file_name]["sha256"] = hashlib.sha256(file_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    bundle = load_backtest_input_bundle(artifact_path)
    assert bundle.signal_snapshot_df["asset"].nunique() < bundle.universe_mask_df["asset"].nunique()
