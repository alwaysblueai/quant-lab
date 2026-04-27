from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from alpha_lab.interfaces import validate_factor_output
from alpha_lab.labels import LabelCache
from alpha_lab.model_factor import (
    FeaturePreprocessConfig,
    ModelFactorBuildConfig,
    ModelFactorBuildResult,
    build_model_factor,
)
from alpha_lab.model_factor.diagnostics import ModelFactorDiagnosticsRecorder
from alpha_lab.neutralization import neutralize_signal
from alpha_lab.real_cases.common_io import (
    apply_universe_to_factor,
    apply_universe_to_prices,
    ensure_parquet_tabular_frame,
    load_prices,
    load_tabular_frame,
    load_universe_mask,
)
from alpha_lab.real_cases.single_factor.evaluate import (
    SingleFactorEvaluationResult,
    evaluate_single_factor_case,
)
from alpha_lab.research_evaluation_config import (
    ResearchEvaluationConfig,
    get_research_evaluation_config,
)
from alpha_lab.research_integrity.contracts import IntegrityCheckResult, IntegrityReport
from alpha_lab.research_integrity.exceptions import raise_on_hard_failures
from alpha_lab.research_integrity.leakage_checks import (
    check_asof_inputs_not_after_signal_date,
    check_cross_section_transform_scope,
    check_factor_label_temporal_order,
    check_no_future_dates_in_input,
)
from alpha_lab.research_integrity.reporting import build_integrity_report

from .artifacts import (
    ModelFactorArtifactPaths,
    export_artifact_bundle,
    write_diagnostics_artifact,
)
from .spec import (
    FeatureAvailabilitySpec,
    ModelFactorCaseSpec,
    infer_fundamental_feature_columns,
    load_model_factor_case_spec,
)


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


@dataclass(frozen=True)
class _ResolvedFeatureAvailability:
    mode: str
    requested_column: str | None
    source_column: str | None
    known_at_col: str
    safety_lag_days: int | None
    shifted_rows: int
    dropped_rows: int


def run_model_factor_case(
    spec_or_path: ModelFactorCaseSpec | str | Path,
    *,
    output_root_dir: str | Path | None = None,
    evaluation_profile: str = "default_research",
    vault_root: str | Path | None = None,
    vault_export_mode: str = "versioned",
    progress_callback: Callable[[str, int], None] | None = None,
) -> ModelFactorCaseRunResult:
    """Run one real-case model-factor study end-to-end and export artifacts."""

    integrity_checks: list[IntegrityCheckResult] = []
    diagnostics = ModelFactorDiagnosticsRecorder()
    spec: ModelFactorCaseSpec | None = None
    spec_path: Path | None = None
    output_dir: Path | None = None
    evaluation_config = None
    universe_mask: pd.DataFrame | None = None
    prices = pd.DataFrame()
    features = pd.DataFrame()
    resolved_feature_availability: _ResolvedFeatureAvailability | None = None
    feature_manifest_payload: dict[str, object] | None = None

    def _emit_progress(message: str, percent: int) -> None:
        if progress_callback is not None:
            progress_callback(message, percent)

    def _emit_training_progress(message: str, percent: int) -> None:
        mapped_percent = 30 + round(max(0, min(int(percent), 100)) * 38 / 100)
        _emit_progress(message, mapped_percent)

    def _record_integrity(check: IntegrityCheckResult) -> None:
        integrity_checks.append(check)
        raise_on_hard_failures((check,))

    try:
        _emit_progress("读取模型因子实验合同文件", 3)
        with diagnostics.stage("spec_load") as spec_stage:
            if isinstance(spec_or_path, ModelFactorCaseSpec):
                spec = spec_or_path
            else:
                spec_path = Path(spec_or_path).resolve()
                spec = load_model_factor_case_spec(spec_path)
            evaluation_config = get_research_evaluation_config(evaluation_profile)
            output_dir = _resolve_case_output_dir(spec, output_root_dir=output_root_dir)
            fundamental_features = infer_fundamental_feature_columns(spec.feature_columns)
            if fundamental_features and spec.feature_availability.mode != "safety_lag":
                diagnostics.warning(
                    title="基本面特征可用性风险",
                    severity="warning",
                    stage="spec_load",
                    description=(
                        "检测到疑似基本面特征 "
                        f"{list(fundamental_features)}，但 feature_availability.mode="
                        f"'{spec.feature_availability.mode}'。若公告在收盘后发布，"
                        "同日使用可能产生前视偏差。"
                    ),
                    suggested_action=(
                        "建议优先使用 feature_availability.mode='safety_lag' "
                        "并设置 safety_lag_days>=1；或提供可审计的已知时间戳列。"
                    ),
                )
                _record_integrity(
                    IntegrityCheckResult(
                        check_name="fundamental_feature_availability_contract",
                        status="warn",
                        severity="warning",
                        object_name="model_factor_features",
                        module_name="run_model_factor_case",
                        message=(
                            "detected fundamental-like feature columns without safety_lag mode"
                        ),
                        remediation=(
                            "prefer safety_lag>=1 or audited known_at timestamps for "
                            "fundamental features"
                        ),
                        metrics={
                            "feature_availability_mode": spec.feature_availability.mode,
                            "fundamental_feature_columns": list(fundamental_features),
                        },
                    )
                )
            spec_stage.attach(
                case_name=spec.name,
                factor_name=spec.factor_name,
                model_family=spec.model.family,
                feature_count=len(spec.feature_columns),
                fundamental_feature_columns=list(fundamental_features),
                target_horizon=int(spec.target.horizon),
                evaluation_profile=evaluation_profile,
                output_dir=str(output_dir),
            )
        _emit_progress("实验合同与评估配置已加载", 10)
        assert spec is not None
        assert output_dir is not None
        assert evaluation_config is not None

        _emit_progress("加载行情、特征与可选股票池", 15)
        with diagnostics.stage("data_load") as data_stage:
            universe_mask = load_universe_mask(spec.universe)
            price_columns, optional_price_columns = _model_factor_price_read_columns(
                evaluation_config
            )
            prices = load_prices(
                spec.prices_path,
                columns=price_columns,
                optional_columns=optional_price_columns,
            )
            feature_storage = ensure_parquet_tabular_frame(
                spec.features_path,
                object_name="features",
            )
            features = _load_features(
                str(feature_storage.path),
                feature_columns=spec.feature_columns,
                feature_availability=spec.feature_availability,
                feature_preprocess=spec.feature_preprocess,
            )
            max_price_date = pd.Timestamp(prices["date"].max())
            data_stage.attach(
                n_prices_rows=int(len(prices)),
                n_prices_assets=int(prices["asset"].nunique()),
                n_prices_dates=int(prices["date"].nunique()),
                prices_requested_columns=list(price_columns),
                prices_optional_columns=list(optional_price_columns),
                prices_loaded_columns=list(prices.columns),
                n_features_rows=int(len(features)),
                n_features_assets=int(features["asset"].nunique()),
                n_features_dates=int(features["date"].nunique()),
                features_requested_path=str(feature_storage.source_path),
                features_storage_path=str(feature_storage.path),
                features_storage_format=feature_storage.path.suffix.lower().lstrip("."),
                features_parquet_materialized=bool(feature_storage.materialized),
                universe_enabled=bool(universe_mask is not None),
                max_price_date=max_price_date.date().isoformat(),
            )

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
            features, resolved_feature_availability = _resolve_feature_availability_contract(
                features,
                prices=prices,
                contract=spec.feature_availability,
            )
            _record_integrity(
                check_no_future_dates_in_input(
                    features,
                    max_allowed_date=max_price_date,
                    date_col="date",
                    object_name="model_factor_features_resolved",
                )
            )
            data_stage.attach(
                feature_availability_mode=resolved_feature_availability.mode,
                feature_known_at_column=resolved_feature_availability.known_at_col,
                feature_availability_source_column=resolved_feature_availability.source_column,
                feature_safety_lag_days=resolved_feature_availability.safety_lag_days,
                feature_shifted_rows=resolved_feature_availability.shifted_rows,
                feature_dropped_rows=resolved_feature_availability.dropped_rows,
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

        _emit_progress("训练模型生成因子", 30)
        assert resolved_feature_availability is not None
        known_at_col = resolved_feature_availability.known_at_col
        build_result = build_model_factor(
            features,
            prices,
            ModelFactorBuildConfig(
                factor_name=spec.factor_name,
                feature_columns=spec.feature_columns,
                target_horizon=spec.target.horizon,
                feature_preprocess=spec.feature_preprocess,
                feature_importance=spec.feature_importance,
                model=spec.model,
                model_selection=spec.model_selection,
                training=spec.training,
                known_at_col=known_at_col,
                label_winsorize_zscore=spec.target.winsorize_zscore,
            ),
            observer=diagnostics,
            progress_callback=_emit_training_progress,
        )
        integrity_checks.extend(build_result.integrity_checks)
        feature_manifest_payload = _build_feature_manifest_payload(
            spec=spec,
            features=features,
            resolved_feature_availability=resolved_feature_availability,
        )
        # The wide feature matrix is no longer needed after training and manifest
        # summarization. Release it before evaluation/artifact export to reduce
        # peak RSS for large model-lab runs.
        features = pd.DataFrame()
        gc.collect()

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
        universe_mask = None
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
        forward_label_cache = _build_forward_label_cache(
            prices=prices,
            target_horizon=int(spec.target.horizon),
            target_label_df=build_result.forward_label_df,
            evaluation_config=evaluation_config,
        )

        _emit_progress("运行模型因子评估", 68)
        with diagnostics.stage("evaluate") as evaluate_stage:
            evaluation_result = evaluate_single_factor_case(
                prices=prices,
                factor_df=factor_df,
                raw_factor_df=raw_factor_df,
                spec=cast(Any, spec),
                coverage_by_date=coverage_by_date,
                neutralization_summary=neutral_diag,
                precomputed_forward_labels=forward_label_cache,
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
            evaluate_stage.attach(
                n_factor_rows=int(len(factor_df)),
                n_factor_dates=int(factor_df["date"].nunique()),
                n_factor_assets=int(factor_df["asset"].nunique()),
                evaluation_profile=evaluation_profile,
                neutralization_enabled=bool(spec.neutralization.enabled),
                forward_label_cache_horizons=sorted(forward_label_cache),
                forward_label_cache_rows={
                    int(horizon): int(len(labels))
                    for horizon, labels in forward_label_cache.items()
                },
            )
        forward_label_cache.clear()
        prices = pd.DataFrame()
        gc.collect()

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
                "feature_availability_mode": spec.feature_availability.mode,
                "model_selection_enabled": bool(spec.model_selection.enabled),
                "neutralization_enabled": bool(spec.neutralization.enabled),
            },
        )

        _emit_progress("导出模型因子研究产物", 90)
        assert feature_manifest_payload is not None
        diagnostics_run_meta = _build_diagnostics_run_meta(
            spec=spec,
            evaluation_profile=evaluation_profile,
            output_dir=output_dir,
            status="succeeded",
        )
        with diagnostics.stage("artifact_export") as export_stage:
            artifact_paths = export_artifact_bundle(
                spec=spec,
                model_factor_result=build_result,
                diagnostics_payload=None,
                feature_manifest_payload=feature_manifest_payload,
                evaluation_result=evaluation_result,
                integrity_report=integrity_report,
                output_dir=output_dir,
                spec_path=spec_path,
                evaluation_config=evaluation_config,
                vault_root=vault_root,
                vault_export_mode=vault_export_mode,
            )
            export_stage.attach(
                n_artifacts=int(len(artifact_paths)),
                output_dir=str(output_dir),
                vault_export_mode=str(vault_export_mode),
            )
        diagnostics_payload = diagnostics.build_payload(
            run_meta=diagnostics_run_meta,
            raw_log_ref="training_log.csv",
        )
        artifact_paths["diagnostics"] = write_diagnostics_artifact(
            output_dir=output_dir,
            diagnostics_payload=diagnostics_payload,
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
    except Exception as exc:
        diagnostics.event(
            level="error",
            stage="pipeline",
            message="model-factor pipeline failed",
            payload={"error_type": type(exc).__name__, "error_message": str(exc)},
        )
        diagnostics.warning(
            title="Run 执行失败",
            severity="error",
            stage="pipeline",
            description=f"{type(exc).__name__}: {exc}",
            suggested_action="检查失败阶段日志与 traceback，优先排查路径、schema、时序约束。",
        )
        diagnostics_path: Path | None = None
        if output_dir is not None:
            diagnostics_run_meta = _build_diagnostics_run_meta(
                spec=spec,
                evaluation_profile=evaluation_profile,
                output_dir=output_dir,
                status="failed",
            )
            try:
                with diagnostics.stage(
                    "artifact_export",
                    payload={"mode": "failure_flush"},
                ) as failure_stage:
                    failure_stage.attach(
                        mode="failure_flush",
                        output_dir=str(output_dir),
                    )
                diagnostics_payload = diagnostics.build_payload(
                    run_meta=diagnostics_run_meta,
                    raw_log_ref="training_log.csv",
                )
                diagnostics_path = write_diagnostics_artifact(
                    output_dir=output_dir,
                    diagnostics_payload=diagnostics_payload,
                )
            except Exception as flush_exc:  # noqa: BLE001
                diagnostics.event(
                    level="error",
                    stage="artifact_export",
                    message="failed to flush diagnostics on error path",
                    payload={
                        "error_type": type(flush_exc).__name__,
                        "error_message": str(flush_exc),
                    },
                )
        _annotate_exception_with_diagnostics(
            exc,
            output_dir=output_dir,
            diagnostics_path=diagnostics_path,
        )
        raise


def _resolve_case_output_dir(
    spec: ModelFactorCaseSpec,
    *,
    output_root_dir: str | Path | None,
) -> Path:
    root_dir = (
        Path(output_root_dir).resolve()
        if output_root_dir is not None
        else Path(spec.output.root_dir)
    )
    return (root_dir.resolve() / spec.name).resolve()


def _build_diagnostics_run_meta(
    *,
    spec: ModelFactorCaseSpec | None,
    evaluation_profile: str,
    output_dir: Path | None,
    status: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "workflow": "real_case_model_factor",
        "status": status,
        "evaluation_profile": evaluation_profile,
        "output_dir": str(output_dir) if output_dir is not None else None,
    }
    if spec is not None:
        payload.update(
            {
                "case_name": spec.name,
                "factor_name": spec.factor_name,
                "model_family": spec.model.family,
                "target_horizon": int(spec.target.horizon),
                "feature_count": len(spec.feature_columns),
                "prices_path": spec.prices_path,
                "features_path": spec.features_path,
            }
        )
    return payload


def _annotate_exception_with_diagnostics(
    exc: Exception,
    *,
    output_dir: Path | None,
    diagnostics_path: Path | None,
) -> None:
    try:
        if output_dir is not None:
            exc.model_lab_output_dir = str(output_dir)  # type: ignore[attr-defined]
        if diagnostics_path is not None:
            exc.model_lab_artifact_paths = {  # type: ignore[attr-defined]
                "diagnostics": str(diagnostics_path),
            }
    except Exception:  # noqa: BLE001
        return


def _load_features(
    path_value: str,
    *,
    feature_columns: tuple[str, ...] = (),
    feature_availability: FeatureAvailabilitySpec | None = None,
    feature_preprocess: FeaturePreprocessConfig | None = None,
) -> pd.DataFrame:
    if not feature_columns and feature_availability is None and feature_preprocess is None:
        features = load_tabular_frame(path_value, object_name="features")
        return _normalize_loaded_features(features)

    required_columns: list[str] = ["date", "asset", *feature_columns]
    optional_columns: list[str] = []

    if feature_availability is not None:
        if feature_availability.mode == "required_timestamp":
            if feature_availability.column is not None:
                required_columns.append(feature_availability.column)
            else:
                optional_columns.extend(["known_at", "available_at"])
    else:
        optional_columns.extend(["known_at", "available_at"])

    if (
        feature_preprocess is not None
        and feature_preprocess.cross_sectional_group_scope == "date_and_industry"
        and feature_preprocess.industry_group_column is not None
    ):
        required_columns.append(feature_preprocess.industry_group_column)

    features = load_tabular_frame(
        path_value,
        object_name="features",
        columns=required_columns,
        optional_columns=optional_columns,
    )
    return _normalize_loaded_features(features)


def _normalize_loaded_features(features: pd.DataFrame) -> pd.DataFrame:
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

    required = {"date", "asset"}
    if spec.neutralization.size_col is not None:
        required.add(spec.neutralization.size_col)
    if spec.neutralization.industry_col is not None:
        required.add(spec.neutralization.industry_col)

    exposures = load_tabular_frame(
        exposures_path,
        object_name="neutralization exposure",
        columns=sorted(required),
        optional_columns=("known_at", "available_at"),
    )
    exposures["date"] = pd.to_datetime(exposures["date"], errors="coerce")

    missing = required - set(exposures.columns)
    if missing:
        raise ValueError(
            f"neutralization exposure file is missing required columns: {sorted(missing)}"
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


def _model_factor_price_read_columns(
    evaluation_config: ResearchEvaluationConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    required = ("date", "asset", "close")
    diagnostics_cfg = evaluation_config.single_factor_diagnostics
    optional: list[str] = [
        # Preserve default dividend back-adjustment and data-quality summaries
        # while still avoiding unrelated wide price columns.
        "dividend_per_share",
        "volume",
    ]
    if diagnostics_cfg.run_tradability_checks:
        optional.extend(["open", "high", "low", "volume"])
    if diagnostics_cfg.run_execution_price_sensitivity:
        optional.append("open")
    if diagnostics_cfg.compute_capacity_estimation:
        optional.extend(["amount", "market_cap", "circ_mv", "total_mv", "value"])
    if diagnostics_cfg.run_baseline_comparison:
        optional.extend(["ret_5d", "ret_20d"])
    return required, _unique_columns(optional, exclude=set(required))


def _build_forward_label_cache(
    *,
    prices: pd.DataFrame,
    target_horizon: int,
    target_label_df: pd.DataFrame,
    evaluation_config: ResearchEvaluationConfig,
) -> dict[int, pd.DataFrame]:
    horizons = {int(target_horizon)}
    if evaluation_config.single_factor_diagnostics.compute_ic_decay:
        horizons.update(_model_factor_decay_horizons(target_horizon))

    label_cache = LabelCache(prices)
    cache: dict[int, pd.DataFrame] = {}
    for horizon in sorted(horizons):
        if horizon == int(target_horizon):
            cache[horizon] = target_label_df.copy()
        else:
            cache[horizon] = label_cache.forward_return(horizon)
    return cache


def _model_factor_decay_horizons(target_horizon: int) -> tuple[int, ...]:
    horizons = {1, 2, 3, 5, 10, 20}
    if target_horizon > 0:
        horizons.add(int(target_horizon))
    return tuple(sorted(horizons))


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


def _resolve_feature_availability_contract(
    features: pd.DataFrame,
    *,
    prices: pd.DataFrame,
    contract: FeatureAvailabilitySpec,
) -> tuple[pd.DataFrame, _ResolvedFeatureAvailability]:
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
            _ResolvedFeatureAvailability(
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
        _ResolvedFeatureAvailability(
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
    features: pd.DataFrame,
    resolved_feature_availability: _ResolvedFeatureAvailability,
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
