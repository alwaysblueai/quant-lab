from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alpha_lab.backtest_adapter.schema import (
    BacktestInputBundle,
    validate_adapter_schema_versions,
)
from alpha_lab.backtest_adapter.validators import validate_backtest_input_bundle
from alpha_lab.handoff import (
    HANDOFF_SCHEMA_VERSION,
    ExecutionAssumptionsSpec,
    PortfolioConstructionSpec,
    validate_handoff_artifact,
)


def load_backtest_input_bundle(path: str | Path) -> BacktestInputBundle:
    """Load a schema-2.0.0 handoff artifact into a typed in-memory bundle."""
    artifact_path = Path(path).resolve()
    validate_handoff_artifact(artifact_path)

    manifest = _read_json(artifact_path / "manifest.json")
    bundle_schema_version = str(manifest.get("schema_version") or "")
    if bundle_schema_version != HANDOFF_SCHEMA_VERSION:
        raise ValueError(
            f"backtest adapter supports only handoff schema {HANDOFF_SCHEMA_VERSION}; "
            f"got {bundle_schema_version}"
        )

    signal_snapshot_df = _read_csv(artifact_path / "signal_snapshot.csv")
    _coerce_dates(signal_snapshot_df, "signal_snapshot.csv")
    _assert_has(
        signal_snapshot_df, ("date", "asset", "signal_name", "signal_value"), "signal_snapshot.csv"
    )
    signal_snapshot_df = signal_snapshot_df.sort_values(
        ["date", "asset"], kind="mergesort"
    ).reset_index(drop=True)

    universe_mask_df = _read_csv(artifact_path / "universe_mask.csv")
    _coerce_dates(universe_mask_df, "universe_mask.csv")
    _assert_has(universe_mask_df, ("date", "asset", "in_universe"), "universe_mask.csv")
    universe_mask_df["in_universe"] = _coerce_bool_series(
        universe_mask_df["in_universe"], "in_universe"
    )
    universe_mask_df = universe_mask_df.sort_values(
        ["date", "asset"], kind="mergesort"
    ).reset_index(drop=True)

    tradability_mask_df = _read_csv(artifact_path / "tradability_mask.csv")
    _coerce_dates(tradability_mask_df, "tradability_mask.csv")
    _assert_has(tradability_mask_df, ("date", "asset", "is_tradable"), "tradability_mask.csv")
    tradability_mask_df["is_tradable"] = _coerce_bool_series(
        tradability_mask_df["is_tradable"], "is_tradable"
    )
    tradability_mask_df = tradability_mask_df.sort_values(
        ["date", "asset"], kind="mergesort"
    ).reset_index(drop=True)

    exclusion_path = artifact_path / "exclusion_reasons.csv"
    exclusion_reasons_df: pd.DataFrame | None = None
    if exclusion_path.exists():
        exclusion_reasons_df = _read_csv(exclusion_path)
        _coerce_dates(exclusion_reasons_df, "exclusion_reasons.csv")
        required = {"date", "asset", "reason"}
        missing = required - set(exclusion_reasons_df.columns)
        if missing:
            raise ValueError(
                f"exclusion_reasons.csv is missing required columns: {sorted(missing)}"
            )
        reason_text = exclusion_reasons_df["reason"].astype(str).str.strip()
        if (reason_text == "").any():
            raise ValueError("exclusion_reasons.csv contains empty reason values")
        exclusion_reasons_df = exclusion_reasons_df.sort_values(
            ["date", "asset", "reason"], kind="mergesort"
        ).reset_index(drop=True)

    label_path = artifact_path / "label_snapshot.csv"
    label_snapshot_df: pd.DataFrame | None = None
    if label_path.exists():
        label_snapshot_df = _read_csv(label_path)
        _coerce_dates(label_snapshot_df, "label_snapshot.csv")

    timing_payload = _read_json(artifact_path / "timing.json")
    experiment_metadata_payload = _read_json(artifact_path / "experiment_metadata.json")
    validation_context_payload = _read_json(artifact_path / "validation_context.json")
    dataset_fingerprint_payload = _read_json(artifact_path / "dataset_fingerprint.json")
    portfolio_payload = _read_json(artifact_path / "portfolio_construction.json")
    execution_payload = _read_json(artifact_path / "execution_assumptions.json")

    bundle = BacktestInputBundle(
        artifact_path=artifact_path,
        schema_version=bundle_schema_version,
        manifest=manifest,
        signal_snapshot_df=signal_snapshot_df,
        universe_mask_df=universe_mask_df,
        tradability_mask_df=tradability_mask_df,
        timing_payload=timing_payload,
        experiment_metadata_payload=experiment_metadata_payload,
        validation_context_payload=validation_context_payload,
        dataset_fingerprint_payload=dataset_fingerprint_payload,
        portfolio_construction=PortfolioConstructionSpec.from_dict(portfolio_payload),
        execution_assumptions=ExecutionAssumptionsSpec.from_dict(execution_payload),
        exclusion_reasons_df=exclusion_reasons_df,
        label_snapshot_df=label_snapshot_df,
    )
    validate_adapter_schema_versions(bundle)
    validate_backtest_input_bundle(bundle)
    return bundle


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required file missing: {path}")
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _coerce_dates(table: pd.DataFrame, name: str) -> None:
    if "date" not in table.columns:
        raise ValueError(f"{name} is missing required date column 'date'")
    table["date"] = pd.to_datetime(table["date"], errors="coerce")
    if table["date"].isna().any():
        raise ValueError(f"{name} contains invalid timestamp values in 'date' column")


def _coerce_bool_series(series: pd.Series, column_name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    lowered = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False}
    bad_values = sorted(set(lowered[~lowered.isin(mapping.keys())].tolist()))
    if bad_values:
        raise ValueError(f"column {column_name!r} contains non-boolean values: {bad_values[:10]}")
    return lowered.map(mapping).astype(bool)


def _assert_has(table: pd.DataFrame, cols: tuple[str, ...], name: str) -> None:
    missing = set(cols) - set(table.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")
