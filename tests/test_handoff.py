from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.experiment import run_factor_experiment
from alpha_lab.experiment_metadata import ExperimentMetadata, ValidationMetadata
from alpha_lab.factors.momentum import momentum
from alpha_lab.handoff import (
    EXECUTION_ASSUMPTIONS_SCHEMA_VERSION,
    HANDOFF_SCHEMA_VERSION,
    PORTFOLIO_CONSTRUCTION_SCHEMA_VERSION,
    ExecutionAssumptionsSpec,
    PortfolioConstructionSpec,
    compute_dataset_fingerprint,
    dataframe_fingerprint,
    export_handoff_artifact,
    export_walk_forward_handoff_artifacts,
    validate_handoff_artifact,
)
from alpha_lab.timing import DelaySpec
from alpha_lab.walk_forward import run_walk_forward_experiment


def _make_prices(n_assets: int = 6, n_days: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows: list[dict[str, object]] = []
    for asset in [f"A{i}" for i in range(n_assets)]:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


def _factor_fn(prices: pd.DataFrame) -> pd.DataFrame:
    return momentum(prices, window=5)


def _universe_and_tradability(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = prices[["date", "asset"]].drop_duplicates().reset_index(drop=True)
    universe = base.copy()
    universe["in_universe"] = True
    tradability = base.copy()
    tradability["is_tradable"] = True
    return universe, tradability


def _rewrite_json_and_patch_manifest(
    artifact_path: Path,
    *,
    filename: str,
    payload: dict[str, object],
) -> None:
    target = artifact_path / filename
    text = json.dumps(payload, sort_keys=True, indent=2) + "\n"
    target.write_text(text, encoding="utf-8")
    manifest_path = artifact_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][filename]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def test_dataframe_fingerprint_is_order_insensitive() -> None:
    df1 = pd.DataFrame({"b": [2, 1], "a": ["x", "y"]})
    df2 = pd.DataFrame({"a": ["y", "x"], "b": [1, 2]})
    assert dataframe_fingerprint(df1) == dataframe_fingerprint(df2)


def test_compute_dataset_fingerprint_changes_on_content_change() -> None:
    t1 = {"signal": pd.DataFrame({"date": ["2024-01-01"], "asset": ["A"], "value": [1.0]})}
    t2 = {"signal": pd.DataFrame({"date": ["2024-01-01"], "asset": ["A"], "value": [2.0]})}
    fp1 = compute_dataset_fingerprint(t1)["fingerprint"]
    fp2 = compute_dataset_fingerprint(t2)["fingerprint"]
    assert fp1 != fp2


def test_portfolio_construction_spec_validation() -> None:
    with pytest.raises(ValueError, match="bottom_k"):
        PortfolioConstructionSpec(long_short=False, bottom_k=10)
    with pytest.raises(ValueError, match="rebalance_frequency"):
        PortfolioConstructionSpec(rebalance_frequency=0)
    with pytest.raises(ValueError, match="construction_method"):
        PortfolioConstructionSpec(construction_method="invalid")
    spec = PortfolioConstructionSpec()
    assert spec.schema_version == PORTFOLIO_CONSTRUCTION_SCHEMA_VERSION


def test_execution_assumptions_spec_validation() -> None:
    with pytest.raises(ValueError, match="execution_delay_bars"):
        ExecutionAssumptionsSpec(execution_delay_bars=-1)
    with pytest.raises(ValueError, match="cash_buffer"):
        ExecutionAssumptionsSpec(cash_buffer=1.0)
    with pytest.raises(ValueError, match="lot_size"):
        ExecutionAssumptionsSpec(lot_size_rule="round_to_lot", lot_size=0)
    spec = ExecutionAssumptionsSpec()
    assert spec.schema_version == EXECUTION_ASSUMPTIONS_SCHEMA_VERSION


def test_export_handoff_artifact_writes_expected_package(tmp_path: Path) -> None:
    prices = _make_prices()
    universe, tradability = _universe_and_tradability(prices)
    result = run_factor_experiment(
        prices,
        _factor_fn,
        horizon=5,
        metadata=ExperimentMetadata(
            trial_id="trial-1",
            trial_count=2,
            dataset_id="dataset-v1",
            dataset_hash="abc123",
            validation=ValidationMetadata(scheme="time_split"),
        ),
        delay_spec=DelaySpec.for_horizon(5, purge_periods=1, embargo_periods=1),
    )
    export = export_handoff_artifact(
        result,
        output_dir=tmp_path,
        artifact_name="exp_handoff",
        experiment_id="exp-001",
        universe_df=universe,
        tradability_df=tradability,
        include_label_snapshot=True,
        exclusion_reasons_df=pd.DataFrame(
            {
                "date": [prices["date"].iloc[0]],
                "asset": [prices["asset"].iloc[0]],
                "reason": ["suspended"],
            }
        ),
    )
    assert export.artifact_path.exists()
    assert (export.artifact_path / "manifest.json").exists()
    assert (export.artifact_path / "signal_snapshot.csv").exists()
    assert (export.artifact_path / "label_snapshot.csv").exists()
    assert (export.artifact_path / "portfolio_construction.json").exists()
    assert (export.artifact_path / "execution_assumptions.json").exists()
    manifest = json.loads((export.artifact_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == HANDOFF_SCHEMA_VERSION
    assert manifest["bundle_type"] == "alpha_lab_handoff_bundle"
    assert "bundle_components" in manifest
    assert manifest["experiment_id"] == "exp-001"
    assert manifest["dataset_fingerprint"] == export.dataset_fingerprint
    assert manifest["bundle_components"]["portfolio_construction"]["file"] == (
        "portfolio_construction.json"
    )
    assert manifest["bundle_components"]["execution_assumptions"]["file"] == (
        "execution_assumptions.json"
    )
    validate_handoff_artifact(export.artifact_path)


def test_handoff_export_is_reproducible(tmp_path: Path) -> None:
    prices = _make_prices()
    universe, tradability = _universe_and_tradability(prices)
    result = run_factor_experiment(prices, _factor_fn, horizon=5)
    first = export_handoff_artifact(
        result,
        output_dir=tmp_path,
        artifact_name="stable",
        universe_df=universe,
        tradability_df=tradability,
    )
    second = export_handoff_artifact(
        result,
        output_dir=tmp_path,
        artifact_name="stable",
        universe_df=universe,
        tradability_df=tradability,
        overwrite=True,
    )
    fp1 = json.loads(
        (first.artifact_path / "dataset_fingerprint.json").read_text(encoding="utf-8")
    )
    fp2 = json.loads(
        (second.artifact_path / "dataset_fingerprint.json").read_text(encoding="utf-8")
    )
    assert fp1["fingerprint"] == fp2["fingerprint"]


def test_export_handoff_with_custom_portfolio_and_execution_specs(tmp_path: Path) -> None:
    prices = _make_prices()
    universe, tradability = _universe_and_tradability(prices)
    result = run_factor_experiment(prices, _factor_fn, horizon=5)
    export = export_handoff_artifact(
        result,
        output_dir=tmp_path,
        artifact_name="custom_specs",
        universe_df=universe,
        tradability_df=tradability,
        portfolio_construction=PortfolioConstructionSpec(
            signal_name="momentum_5d",
            long_short=True,
            top_k=15,
            bottom_k=15,
            cash_buffer=0.02,
            post_construction_constraints=("sector_neutral",),
        ),
        execution_assumptions=ExecutionAssumptionsSpec(
            execution_delay_bars=1,
            fill_price_rule="next_open",
            cash_buffer=0.02,
            lot_size_rule="round_to_lot",
            lot_size=100,
        ),
    )
    portfolio = json.loads((export.artifact_path / "portfolio_construction.json").read_text())
    execution = json.loads((export.artifact_path / "execution_assumptions.json").read_text())
    assert portfolio["signal_name"] == "momentum_5d"
    assert execution["lot_size"] == 100
    validate_handoff_artifact(export.artifact_path)


def test_export_rejects_missing_universe_keys(tmp_path: Path) -> None:
    prices = _make_prices()
    universe, tradability = _universe_and_tradability(prices)
    broken_universe = universe.iloc[:-1].reset_index(drop=True)
    result = run_factor_experiment(prices, _factor_fn, horizon=5)
    with pytest.raises(ValueError, match="missing"):
        export_handoff_artifact(
            result,
            output_dir=tmp_path,
            artifact_name="broken",
            universe_df=broken_universe,
            tradability_df=tradability,
        )


def test_export_rejects_portfolio_signal_name_mismatch(tmp_path: Path) -> None:
    prices = _make_prices()
    universe, tradability = _universe_and_tradability(prices)
    result = run_factor_experiment(prices, _factor_fn, horizon=5)
    with pytest.raises(ValueError, match="signal_name"):
        export_handoff_artifact(
            result,
            output_dir=tmp_path,
            artifact_name="signal_name_mismatch",
            universe_df=universe,
            tradability_df=tradability,
            portfolio_construction=PortfolioConstructionSpec(signal_name="other_signal"),
        )


def test_export_requires_exclusion_reasons_for_non_tradable_keys(tmp_path: Path) -> None:
    prices = _make_prices()
    universe, tradability = _universe_and_tradability(prices)
    result = run_factor_experiment(prices, _factor_fn, horizon=5)
    tradability = tradability.copy()
    signal_key = result.quantile_assignments_df[["date", "asset"]].drop_duplicates().iloc[0]
    mask = (
        pd.to_datetime(tradability["date"]) == pd.to_datetime(signal_key["date"])
    ) & (tradability["asset"] == str(signal_key["asset"]))
    tradability.loc[mask, "is_tradable"] = False
    with pytest.raises(ValueError, match="exclusion_reasons"):
        export_handoff_artifact(
            result,
            output_dir=tmp_path,
            artifact_name="missing_exclusion_for_non_tradable",
            universe_df=universe,
            tradability_df=tradability,
        )


def test_export_rejects_partial_exclusion_reason_coverage(tmp_path: Path) -> None:
    prices = _make_prices()
    universe, tradability = _universe_and_tradability(prices)
    result = run_factor_experiment(prices, _factor_fn, horizon=5)
    tradability = tradability.copy()
    signal_keys = (
        result.quantile_assignments_df[["date", "asset"]]
        .drop_duplicates()
        .head(2)
        .reset_index(drop=True)
    )
    for row in signal_keys.itertuples(index=False):
        mask = (
            pd.to_datetime(tradability["date"]) == pd.to_datetime(row.date)
        ) & (tradability["asset"] == str(row.asset))
        tradability.loc[mask, "is_tradable"] = False
    exclusion = signal_keys.loc[[0], ["date", "asset"]].copy()
    exclusion["reason"] = "min_adv_filter"
    with pytest.raises(ValueError, match="missing_count"):
        export_handoff_artifact(
            result,
            output_dir=tmp_path,
            artifact_name="partial_exclusion_coverage",
            universe_df=universe,
            tradability_df=tradability,
            exclusion_reasons_df=exclusion,
        )


def test_validate_handoff_artifact_detects_corruption(tmp_path: Path) -> None:
    prices = _make_prices()
    universe, tradability = _universe_and_tradability(prices)
    result = run_factor_experiment(prices, _factor_fn, horizon=5)
    export = export_handoff_artifact(
        result,
        output_dir=tmp_path,
        artifact_name="corrupt_me",
        universe_df=universe,
        tradability_df=tradability,
    )
    signal_path = export.artifact_path / "signal_snapshot.csv"
    signal_path.write_text("date,asset,signal_name,signal_value\n2024-01-01,A,mom,999\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_handoff_artifact(export.artifact_path)


def test_validate_handoff_artifact_rejects_malformed_portfolio_spec(tmp_path: Path) -> None:
    prices = _make_prices()
    universe, tradability = _universe_and_tradability(prices)
    result = run_factor_experiment(prices, _factor_fn, horizon=5)
    export = export_handoff_artifact(
        result,
        output_dir=tmp_path,
        artifact_name="bad_portfolio",
        universe_df=universe,
        tradability_df=tradability,
    )
    _rewrite_json_and_patch_manifest(
        export.artifact_path,
        filename="portfolio_construction.json",
        payload={
            "schema_version": PORTFOLIO_CONSTRUCTION_SCHEMA_VERSION,
            "construction_method": "top_bottom_k",
            "signal_name": "momentum_5d",
            "rebalance_frequency": 1,
            "rebalance_calendar": "business_day",
            "long_short": False,
            "top_k": 10,
            "bottom_k": 10,
            "weight_method": "rank",
            "max_weight": 0.1,
            "gross_limit": 1.0,
            "net_limit": 0.5,
            "cash_buffer": 0.0,
            "neutralization_required": False,
            "post_construction_constraints": [],
        },
    )
    with pytest.raises(ValueError, match="bottom_k"):
        validate_handoff_artifact(export.artifact_path)


def test_validate_handoff_artifact_rejects_malformed_execution_spec(tmp_path: Path) -> None:
    prices = _make_prices()
    universe, tradability = _universe_and_tradability(prices)
    result = run_factor_experiment(prices, _factor_fn, horizon=5)
    export = export_handoff_artifact(
        result,
        output_dir=tmp_path,
        artifact_name="bad_execution",
        universe_df=universe,
        tradability_df=tradability,
    )
    _rewrite_json_and_patch_manifest(
        export.artifact_path,
        filename="execution_assumptions.json",
        payload={
            "schema_version": EXECUTION_ASSUMPTIONS_SCHEMA_VERSION,
            "fill_price_rule": "next_open",
            "execution_delay_bars": -1,
            "commission_model": "bps",
            "slippage_model": "fixed_bps",
            "lot_size_rule": "none",
            "lot_size": None,
            "cash_buffer": 0.0,
            "partial_fill_policy": "allow_partial",
            "suspension_policy": "skip_trade",
            "price_limit_policy": "skip_trade",
            "trade_when_not_tradable": False,
            "allow_same_day_reentry": False,
        },
    )
    with pytest.raises(ValueError, match="execution_delay_bars"):
        validate_handoff_artifact(export.artifact_path)


def test_walk_forward_handoff_exports_selected_folds(tmp_path: Path) -> None:
    prices = _make_prices(n_days=80)
    universe, tradability = _universe_and_tradability(prices)
    wf = run_walk_forward_experiment(
        prices,
        _factor_fn,
        train_size=30,
        test_size=10,
        step=10,
        horizon=5,
    )
    exports = export_walk_forward_handoff_artifacts(
        wf,
        output_dir=tmp_path,
        universe_df=universe,
        tradability_df=tradability,
        artifact_prefix="wf",
        fold_ids=[0, 2],
    )
    assert len(exports) == 2
    names = sorted(p.artifact_path.name for p in exports)
    assert names == ["wf_fold_000", "wf_fold_002"]
    for exp in exports:
        assert (exp.artifact_path / "portfolio_construction.json").exists()
        assert (exp.artifact_path / "execution_assumptions.json").exists()
        validate_handoff_artifact(exp.artifact_path)


def test_handoff_metadata_and_validation_propagation(tmp_path: Path) -> None:
    prices = _make_prices()
    universe, tradability = _universe_and_tradability(prices)
    result = run_factor_experiment(
        prices,
        _factor_fn,
        horizon=5,
        metadata=ExperimentMetadata(
            trial_id="trial-9",
            trial_count=9,
            hypothesis="momentum persistence",
            validation=ValidationMetadata(scheme="time_split"),
        ),
    )
    export = export_handoff_artifact(
        result,
        output_dir=tmp_path,
        artifact_name="meta",
        experiment_id="exp-meta",
        fold_id=3,
        validation_context={"custom_tag": "wf-check"},
        universe_df=universe,
        tradability_df=tradability,
    )
    metadata = json.loads((export.artifact_path / "experiment_metadata.json").read_text())
    validation = json.loads((export.artifact_path / "validation_context.json").read_text())
    timing = json.loads((export.artifact_path / "timing.json").read_text())
    assert metadata["experiment_id"] == "exp-meta"
    assert metadata["experiment_metadata"]["trial_id"] == "trial-9"
    assert validation["fold_id"] == 3
    assert validation["custom_tag"] == "wf-check"
    assert timing["delay_spec"]["return_horizon_periods"] == 5


def test_export_rejects_timing_execution_delay_contradiction(tmp_path: Path) -> None:
    prices = _make_prices()
    universe, tradability = _universe_and_tradability(prices)
    result = run_factor_experiment(
        prices,
        _factor_fn,
        horizon=5,
        delay_spec=DelaySpec.for_horizon(5, execution_delay_periods=2),
    )
    with pytest.raises(ValueError, match="execution_delay_bars"):
        export_handoff_artifact(
            result,
            output_dir=tmp_path,
            artifact_name="delay_contradiction",
            universe_df=universe,
            tradability_df=tradability,
            execution_assumptions=ExecutionAssumptionsSpec(execution_delay_bars=1),
        )


def test_validate_handoff_artifact_accepts_legacy_v1_layout(tmp_path: Path) -> None:
    prices = _make_prices()
    universe, tradability = _universe_and_tradability(prices)
    result = run_factor_experiment(prices, _factor_fn, horizon=5)
    export = export_handoff_artifact(
        result,
        output_dir=tmp_path,
        artifact_name="legacy_compat",
        universe_df=universe,
        tradability_df=tradability,
    )
    artifact = export.artifact_path
    (artifact / "portfolio_construction.json").unlink()
    (artifact / "execution_assumptions.json").unlink()

    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "1.0.0"
    manifest.pop("bundle_type", None)
    manifest.pop("bundle_components", None)
    manifest["files"].pop("portfolio_construction.json", None)
    manifest["files"].pop("execution_assumptions.json", None)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    validate_handoff_artifact(artifact)
