from __future__ import annotations

import pandas as pd

from alpha_lab.model_factor.dataset_cache import (
    ResolvedFeatureAvailability,
)

from ..spec import (
    FeatureAvailabilitySpec,
    ModelFactorCaseSpec,
)


def _resolve_feature_availability_contract(
    features: pd.DataFrame,
    *,
    prices: pd.DataFrame,
    contract: FeatureAvailabilitySpec,
) -> tuple[pd.DataFrame, ResolvedFeatureAvailability]:
    if contract.mode == "required_timestamp":
        source_column = _resolve_known_at_column(features, preferred=contract.column)
        if source_column is None:
            raise ValueError(
                "feature_availability.mode='required_timestamp' requires a "
                "known_at/available_at column, or set feature_availability.column explicitly."
            )
        resolved = features.copy()
        resolved[source_column] = pd.to_datetime(resolved[source_column], errors="coerce")
        if resolved[source_column].isna().any():
            raise ValueError(
                f"feature availability column {source_column!r} contains invalid timestamps"
            )
        return (
            resolved.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True),
            ResolvedFeatureAvailability(
                mode=contract.mode,
                requested_column=contract.column,
                source_column=source_column,
                known_at_col=source_column,
                safety_lag_days=None,
                shifted_rows=0,
                dropped_rows=0,
            ),
        )

    lag = int(contract.safety_lag_days or 0)
    if lag <= 0:
        raise ValueError(
            "feature_availability.safety_lag_days must be > 0 when "
            "feature_availability.mode='safety_lag'"
        )
    resolved = features.copy()
    feature_dates = pd.to_datetime(resolved["date"], errors="coerce")
    price_axis = pd.Index(
        pd.to_datetime(prices["date"], errors="coerce").dropna().unique()
    ).sort_values()
    feature_axis = pd.Index(feature_dates.dropna().unique()).sort_values()
    missing_dates = feature_axis.difference(price_axis)
    if len(missing_dates) > 0:
        raise ValueError(
            "feature_availability.mode='safety_lag' requires feature dates to exist on the "
            "price date axis; missing dates: "
            f"{[pd.Timestamp(value).date().isoformat() for value in missing_dates[:5]]}"
        )
    lag_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for idx, raw_date in enumerate(price_axis):
        shifted_idx = idx + lag
        if shifted_idx >= len(price_axis):
            continue
        lag_map[pd.Timestamp(raw_date)] = pd.Timestamp(price_axis[shifted_idx])
    shifted_dates = feature_dates.map(lag_map)
    keep_mask = shifted_dates.notna()
    dropped_rows = int((~keep_mask).sum())
    resolved = resolved.loc[keep_mask].copy()
    shifted_dates = pd.to_datetime(shifted_dates.loc[keep_mask], errors="coerce")
    synthetic_known_at_col = "__feature_availability_known_at"
    resolved["date"] = shifted_dates.to_numpy()
    resolved[synthetic_known_at_col] = shifted_dates.to_numpy()
    return (
        resolved.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True),
        ResolvedFeatureAvailability(
            mode=contract.mode,
            requested_column=None,
            source_column=None,
            known_at_col=synthetic_known_at_col,
            safety_lag_days=lag,
            shifted_rows=int(len(resolved)),
            dropped_rows=dropped_rows,
        ),
    )


def _resolve_known_at_column(features: pd.DataFrame, *, preferred: str | None) -> str | None:
    if preferred is not None:
        return preferred if preferred in features.columns else None
    if "known_at" in features.columns:
        return "known_at"
    if "available_at" in features.columns:
        return "available_at"
    return None


def _build_feature_manifest_payload(
    *,
    spec: ModelFactorCaseSpec,
    features: pd.DataFrame | None,
    resolved_feature_availability: ResolvedFeatureAvailability,
    cache_metadata: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata = cache_metadata or {}
    raw_stats = metadata.get("feature_manifest_stats")
    stats_by_feature: dict[str, dict[str, object]] = {}
    if isinstance(raw_stats, list):
        for item in raw_stats:
            if isinstance(item, dict) and isinstance(item.get("feature"), str):
                stats_by_feature[str(item["feature"])] = dict(item)
    rows: list[dict[str, object]] = []
    for column in spec.feature_columns:
        if features is not None and column in features.columns:
            series = pd.to_numeric(features[column], errors="coerce")
            rows.append(
                {
                    "feature": column,
                    "non_null_ratio": (float(series.notna().mean()) if not series.empty else None),
                    "mean": float(series.mean()) if series.notna().any() else None,
                    "std": float(series.std(ddof=1)) if series.notna().sum() > 1 else None,
                }
            )
        else:
            stats = stats_by_feature.get(column, {})
            rows.append(
                {
                    "feature": column,
                    "non_null_ratio": stats.get("non_null_ratio"),
                    "mean": stats.get("mean"),
                    "std": stats.get("std"),
                }
            )
    features_loaded = features is not None and not features.empty
    feature_frame = features if features_loaded else None
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_feature_manifest",
        "case_name": spec.name,
        "factor_name": spec.factor_name,
        "model_family": spec.model.family,
        "manifest_source": "features" if features_loaded else "cache_metadata",
        "n_rows": (
            int(len(feature_frame))
            if feature_frame is not None
            else _metadata_int_or_none(metadata.get("n_features_rows"))
        ),
        "n_dates": (
            int(feature_frame["date"].nunique())
            if feature_frame is not None and "date" in feature_frame
            else _metadata_int_or_none(metadata.get("n_features_dates"))
        ),
        "n_assets": (
            int(feature_frame["asset"].nunique())
            if feature_frame is not None and "asset" in feature_frame
            else _metadata_int_or_none(metadata.get("n_features_assets"))
        ),
        "known_at_column": resolved_feature_availability.known_at_col,
        "feature_availability": {
            "mode": resolved_feature_availability.mode,
            "requested_column": resolved_feature_availability.requested_column,
            "source_column": resolved_feature_availability.source_column,
            "known_at_column": resolved_feature_availability.known_at_col,
            "safety_lag_days": resolved_feature_availability.safety_lag_days,
            "shifted_rows": resolved_feature_availability.shifted_rows,
            "dropped_rows": resolved_feature_availability.dropped_rows,
        },
        "feature_columns": list(spec.feature_columns),
        "features": rows,
    }


def _metadata_int_or_none(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    return None


def _unique_columns(columns: list[str], *, exclude: set[str] | None = None) -> tuple[str, ...]:
    excluded = exclude or set()
    out: list[str] = []
    seen: set[str] = set()
    for raw in columns:
        column = str(raw).strip()
        if not column or column in excluded or column in seen:
            continue
        out.append(column)
        seen.add(column)
    return tuple(out)
