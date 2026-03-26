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
    schema_version = str(manifest.get("schema_version"))
    if schema_version != HANDOFF_SCHEMA_VERSION:
        raise ValueError(
            "backtest adapter supports only handoff schema "
            f"{HANDOFF_SCHEMA_VERSION!r}; got {schema_version!r}"
        )

    signal_snapshot_df = _read_csv(
        artifact_path / "signal_snapshot.csv",
        date_cols=("date",),
    )
    universe_mask_df = _read_csv(
        artifact_path / "universe_mask.csv",
        date_cols=("date",),
    )
    universe_mask_df["in_universe"] = _coerce_bool_series(
        universe_mask_df["in_universe"],
        column_name="in_universe",
    )
    tradability_mask_df = _read_csv(
        artifact_path / "tradability_mask.csv",
        date_cols=("date",),
    )
    tradability_mask_df["is_tradable"] = _coerce_bool_series(
        tradability_mask_df["is_tradable"],
        column_name="is_tradable",
    )

    exclusion_path = artifact_path / "exclusion_reasons.csv"
    exclusion_reasons_df = (
        _read_csv(exclusion_path, date_cols=("date",)) if exclusion_path.exists() else None
    )
    if exclusion_reasons_df is not None:
        required = {"date", "asset", "reason"}
        missing = required - set(exclusion_reasons_df.columns)
        if missing:
            raise ValueError(
                "exclusion_reasons.csv is missing required columns: "
                f"{sorted(missing)}"
            )
        reason_text = exclusion_reasons_df["reason"].astype(str).str.strip()
        if (reason_text == "").any():
            raise ValueError("exclusion_reasons.csv contains empty reason values")
        exclusion_reasons_df = exclusion_reasons_df.sort_values(
            ["date", "asset", "reason"],
            kind="mergesort",
        ).reset_index(drop=True)
    label_path = artifact_path / "label_snapshot.csv"
    label_snapshot_df = _read_csv(label_path, date_cols=("date",)) if label_path.exists() else None

    timing_payload = _read_json(artifact_path / "timing.json")
    experiment_metadata_payload = _read_json(artifact_path / "experiment_metadata.json")
    validation_context_payload = _read_json(artifact_path / "validation_context.json")
    dataset_fingerprint_payload = _read_json(artifact_path / "dataset_fingerprint.json")
    portfolio_payload = _read_json(artifact_path / "portfolio_construction.json")
    execution_payload = _read_json(artifact_path / "execution_assumptions.json")

    portfolio_construction = PortfolioConstructionSpec.from_dict(portfolio_payload)
    execution_assumptions = ExecutionAssumptionsSpec.from_dict(execution_payload)
    validate_adapter_schema_versions(
        portfolio_construction,
        execution_assumptions,
        bundle_schema_version=schema_version,
    )

    bundle = BacktestInputBundle(
        artifact_path=artifact_path,
        schema_version=schema_version,
        manifest=manifest,
        signal_snapshot_df=signal_snapshot_df.sort_values(["date", "asset"]).reset_index(drop=True),
        universe_mask_df=universe_mask_df.sort_values(["date", "asset"]).reset_index(drop=True),
        tradability_mask_df=tradability_mask_df.sort_values(["date", "asset"]).reset_index(
            drop=True
        ),
        exclusion_reasons_df=exclusion_reasons_df,
        label_snapshot_df=label_snapshot_df,
        timing_payload=timing_payload,
        experiment_metadata_payload=experiment_metadata_payload,
        validation_context_payload=validation_context_payload,
        dataset_fingerprint_payload=dataset_fingerprint_payload,
        portfolio_construction=portfolio_construction,
        execution_assumptions=execution_assumptions,
    )
    validate_backtest_input_bundle(bundle)
    return bundle


def _read_csv(path: Path, *, date_cols: tuple[str, ...]) -> pd.DataFrame:
    if not path.exists():
        raise ValueError(f"required file missing: {path.name}")
    df = pd.read_csv(path)
    for col in date_cols:
        if col not in df.columns:
            raise ValueError(f"{path.name} is missing required date column {col!r}")
        df[col] = pd.to_datetime(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"{path.name} contains invalid timestamp values in {col!r}")
    return df


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise ValueError(f"required file missing: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _coerce_bool_series(series: pd.Series, *, column_name: str) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)

    lowered = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    if not lowered.isin(mapping).all():
        bad_values = sorted(set(series[~lowered.isin(mapping)].astype(str).tolist()))
        raise ValueError(
            f"column {column_name!r} contains non-boolean values: {bad_values[:5]}"
        )
    return lowered.map(mapping).astype(bool)
