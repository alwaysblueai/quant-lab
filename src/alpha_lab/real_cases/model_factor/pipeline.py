from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from alpha_lab.interfaces import validate_factor_output
from alpha_lab.model_factor import (
    ModelFactorBuildConfig,
    ModelFactorBuildResult,
    build_model_factor,
)
from alpha_lab.neutralization import neutralize_signal
from alpha_lab.real_cases.common_io import (
    apply_universe_to_factor,
    apply_universe_to_prices,
    load_prices,
    load_universe_mask,
)
from alpha_lab.real_cases.single_factor.evaluate import (
    SingleFactorEvaluationResult,
    evaluate_single_factor_case,
)
from alpha_lab.research_contracts import validate_prices_table
from alpha_lab.research_evaluation_config import get_research_evaluation_config
from alpha_lab.research_integrity.contracts import IntegrityCheckResult, IntegrityReport
from alpha_lab.research_integrity.exceptions import raise_on_hard_failures
from alpha_lab.research_integrity.leakage_checks import (
    check_asof_inputs_not_after_signal_date,
    check_cross_section_transform_scope,
    check_factor_label_temporal_order,
    check_no_future_dates_in_input,
)
from alpha_lab.research_integrity.reporting import build_integrity_report

from .artifacts import ModelFactorArtifactPaths, export_artifact_bundle
from .spec import ModelFactorCaseSpec, load_model_factor_case_spec


@dataclass(frozen=True)
class ModelFactorCaseRunResult:
    """End-to-end run result for one real-case model-factor research package."""

    spec: ModelFactorCaseSpec
    output_dir: Path
    factor_df: pd.DataFrame
    evaluation_result: SingleFactorEvaluationResult
    artifact_paths: ModelFactorArtifactPaths
    integrity_report: IntegrityReport
    model_factor_result: ModelFactorBuildResult


def run_model_factor_case(
    spec_or_path: ModelFactorCaseSpec | str | Path,
    *,
    output_root_dir: str | Path | None = None,
    evaluation_profile: str = "default_research",
    vault_root: str | Path | None = None,
    vault_export_mode: str = "versioned",
) -> ModelFactorCaseRunResult:
    """Run one real-case model-factor study end-to-end and export artifacts."""

    integrity_checks: list[IntegrityCheckResult] = []

    def _record_integrity(check: IntegrityCheckResult) -> None:
        integrity_checks.append(check)
        raise_on_hard_failures((check,))

    spec_path: Path | None = None
    if isinstance(spec_or_path, ModelFactorCaseSpec):
        spec = spec_or_path
    else:
        spec_path = Path(spec_or_path).resolve()
        spec = load_model_factor_case_spec(spec_path)

    evaluation_config = get_research_evaluation_config(evaluation_profile)

    universe_mask = load_universe_mask(spec.universe)
    prices = load_prices(spec.prices_path)
    features = _load_features(spec.features_path)
    max_price_date = pd.Timestamp(prices["date"].max())

    _record_integrity(
        check_no_future_dates_in_input(
            prices,
            max_allowed_date=max_price_date,
            date_col="date",
            object_name="model_factor_prices",
        )
    )
    _record_integrity(
        check_no_future_dates_in_input(
            features,
            max_allowed_date=max_price_date,
            date_col="date",
            object_name="model_factor_features_raw",
        )
    )

    if universe_mask is not None:
        _record_integrity(
            check_no_future_dates_in_input(
                universe_mask,
                max_allowed_date=max_price_date,
                date_col="date",
                object_name="model_factor_universe",
            )
        )
        _record_integrity(
            check_asof_inputs_not_after_signal_date(
                prices[["date", "asset"]],
                universe_mask,
                by=("asset",),
                signal_date_col="date",
                aux_effective_date_col="date",
                aux_known_at_col=None,
                object_name="model_factor_universe_asof",
            )
        )
        prices = apply_universe_to_prices(prices, universe_mask)
        features = apply_universe_to_factor(features, universe_mask)

    known_at_col = _detect_known_at_col(features)
    build_result = build_model_factor(
        features,
        prices,
        ModelFactorBuildConfig(
            factor_name=spec.factor_name,
            feature_columns=spec.feature_columns,
            target_horizon=spec.target.horizon,
            feature_preprocess=spec.feature_preprocess,
            model=spec.model,
            training=spec.training,
            known_at_col=known_at_col,
        ),
    )
    integrity_checks.extend(build_result.integrity_checks)

    factor_df = build_result.factor_df.copy()
    if spec.direction == "short":
        factor_df["value"] = -factor_df["value"]
    raw_factor_df = factor_df.copy()

    factor_df, neutral_diag = _maybe_neutralize_factor(
        factor_df,
        spec=spec,
        universe_mask=universe_mask,
        integrity_checks=integrity_checks,
        max_price_date=max_price_date,
    )
    coverage_by_date = _coverage_by_date(factor_df)

    validate_factor_output(factor_df)
    _record_integrity(
        check_cross_section_transform_scope(
            prices[["date", "asset"]],
            factor_df[["date", "asset", "value"]],
            date_col="date",
            asset_col="asset",
            object_name="model_factor_final_factor_scope",
        )
    )

    evaluation_result = evaluate_single_factor_case(
        prices=prices,
        factor_df=factor_df,
        raw_factor_df=raw_factor_df,
        spec=spec,
        coverage_by_date=coverage_by_date,
        neutralization_summary=neutral_diag,
        evaluation_config=evaluation_config,
    )
    for check in evaluation_result.experiment_result.integrity_checks:
        _record_integrity(check)
    _record_integrity(
        check_factor_label_temporal_order(
            evaluation_result.experiment_result.factor_df,
            evaluation_result.experiment_result.label_df,
            join_keys=("date", "asset"),
            factor_date_col="date",
            label_date_col="date",
            object_name="model_factor_label_alignment",
        )
    )

    integrity_report = build_integrity_report(
        tuple(integrity_checks),
        context={
            "pipeline": "run_model_factor_case",
            "case_name": spec.name,
            "prices_path": spec.prices_path,
            "features_path": spec.features_path,
            "factor_name": spec.factor_name,
            "feature_columns": list(spec.feature_columns),
            "model_family": spec.model.family,
            "neutralization_enabled": bool(spec.neutralization.enabled),
        },
    )

    root_dir = (
        Path(output_root_dir).resolve()
        if output_root_dir is not None
        else Path(spec.output.root_dir)
    )
    output_dir = (root_dir.resolve() / spec.name).resolve()

    artifact_paths = export_artifact_bundle(
        spec=spec,
        model_factor_result=build_result,
        feature_manifest_payload=_build_feature_manifest_payload(
            spec=spec,
            features=features,
            known_at_col=known_at_col,
        ),
        evaluation_result=evaluation_result,
        integrity_report=integrity_report,
        output_dir=output_dir,
        spec_path=spec_path,
        evaluation_config=evaluation_config,
        vault_root=vault_root,
        vault_export_mode=vault_export_mode,
    )

    return ModelFactorCaseRunResult(
        spec=spec,
        output_dir=output_dir,
        factor_df=factor_df,
        evaluation_result=evaluation_result,
        artifact_paths=artifact_paths,
        integrity_report=integrity_report,
        model_factor_result=build_result,
    )


def _load_features(path_value: str) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"features file does not exist: {path}")

    features = pd.read_csv(path)
    required = {"date", "asset"}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"features is missing required columns: {sorted(missing)}")

    features = features.copy()
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    if "known_at" in features.columns:
        features["known_at"] = pd.to_datetime(features["known_at"], errors="coerce")
    if "available_at" in features.columns:
        features["available_at"] = pd.to_datetime(features["available_at"], errors="coerce")
    return features.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def _maybe_neutralize_factor(
    factor_df: pd.DataFrame,
    *,
    spec: ModelFactorCaseSpec,
    universe_mask: pd.DataFrame | None,
    integrity_checks: list[IntegrityCheckResult] | None = None,
    max_price_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    if not spec.neutralization.enabled:
        return factor_df, None

    exposures_path = spec.neutralization.exposures_path
    if exposures_path is None:
        raise ValueError("neutralization.exposures_path is required when neutralization is enabled")

    exposures = pd.read_csv(exposures_path)
    exposures["date"] = pd.to_datetime(exposures["date"], errors="coerce")

    required = {"date", "asset"}
    if spec.neutralization.size_col is not None:
        required.add(spec.neutralization.size_col)
    if spec.neutralization.industry_col is not None:
        required.add(spec.neutralization.industry_col)

    missing = required - set(exposures.columns)
    if missing:
        raise ValueError(
            "neutralization exposure file is missing required columns: "
            f"{sorted(missing)}"
        )
    known_at_col = None
    if "known_at" in exposures.columns:
        known_at_col = "known_at"
    elif "available_at" in exposures.columns:
        known_at_col = "available_at"

    if integrity_checks is not None and max_price_date is not None:
        no_future_check = check_no_future_dates_in_input(
            exposures,
            max_allowed_date=max_price_date,
            date_col="date",
            object_name="model_factor_neutralization_exposures",
        )
        integrity_checks.append(no_future_check)
        raise_on_hard_failures((no_future_check,))

        asof_check = check_asof_inputs_not_after_signal_date(
            factor_df[["date", "asset"]],
            exposures,
            by=("asset",),
            signal_date_col="date",
            aux_effective_date_col="date",
            aux_known_at_col=known_at_col,
            object_name="model_factor_neutralization_exposures_asof",
        )
        integrity_checks.append(asof_check)
        raise_on_hard_failures((asof_check,))

    if universe_mask is not None:
        active = universe_mask[universe_mask["in_universe"]][["date", "asset"]]
        exposures = exposures.merge(
            active,
            on=["date", "asset"],
            how="inner",
            validate="many_to_one",
        )

    merged = factor_df[["date", "asset", "value"]].merge(
        exposures,
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )

    size_col = spec.neutralization.size_col
    industry_col = spec.neutralization.industry_col

    if size_col is not None:
        merged["__size_input"] = merged[size_col]
        size_col = "__size_input"
    if industry_col is not None:
        merged["__industry_input"] = merged[industry_col]
        industry_col = "__industry_input"
    known_at_input = None
    if known_at_col is not None:
        merged["__known_at_input"] = pd.to_datetime(
            merged[known_at_col],
            errors="coerce",
        )
        known_at_input = "__known_at_input"

    cols = ["date", "asset", "value"]
    for col in (size_col, industry_col):
        if col is not None:
            cols.append(col)
    if known_at_input is not None:
        cols.append(known_at_input)

    neutralized = neutralize_signal(
        merged[cols].copy(),
        value_col="value",
        by="date",
        size_col=size_col,
        industry_col=industry_col,
        beta_col=None,
        min_obs=spec.neutralization.min_obs,
        ridge=spec.neutralization.ridge,
        output_col="value_neutralized",
        known_at_col=known_at_input,
        enforce_integrity=True,
    )
    if integrity_checks is not None:
        integrity_checks.extend(list(neutralized.integrity_checks))
        raise_on_hard_failures(neutralized.integrity_checks)

    out = factor_df[["date", "asset", "factor"]].copy()
    out = out.merge(
        neutralized.data[["date", "asset", "value_neutralized"]],
        on=["date", "asset"],
        how="left",
        validate="one_to_one",
    )
    out = out.rename(columns={"value_neutralized": "value"})
    return out, neutralized.diagnostics


def _coverage_by_date(factor_df: pd.DataFrame) -> pd.DataFrame:
    if factor_df.empty:
        return pd.DataFrame(columns=["date", "n_assets", "coverage", "missingness"])

    summary = factor_df.groupby("date", sort=True).agg(
        n_assets=("asset", "nunique"),
        n_non_null=("value", lambda s: int(s.notna().sum())),
    )
    summary["coverage"] = summary["n_non_null"] / summary["n_assets"].replace(0, pd.NA)
    summary["missingness"] = 1.0 - summary["coverage"]
    return summary.reset_index()[["date", "n_assets", "coverage", "missingness"]]


def _detect_known_at_col(features: pd.DataFrame) -> str | None:
    if "known_at" in features.columns:
        return "known_at"
    if "available_at" in features.columns:
        return "available_at"
    return None


def _build_feature_manifest_payload(
    *,
    spec: ModelFactorCaseSpec,
    features: pd.DataFrame,
    known_at_col: str | None,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for column in spec.feature_columns:
        series = pd.to_numeric(features[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "non_null_ratio": float(series.notna().mean()),
                "mean": float(series.mean()) if series.notna().any() else None,
                "std": float(series.std(ddof=1)) if series.notna().sum() > 1 else None,
            }
        )
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_feature_manifest",
        "case_name": spec.name,
        "factor_name": spec.factor_name,
        "model_family": spec.model.family,
        "n_rows": int(len(features)),
        "n_dates": int(features["date"].nunique()),
        "n_assets": int(features["asset"].nunique()),
        "known_at_column": known_at_col,
        "feature_columns": list(spec.feature_columns),
        "features": rows,
    }
