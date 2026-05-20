from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pandas as pd

from alpha_lab.custom_models import DraftModelSource
from alpha_lab.interfaces import validate_factor_output
from alpha_lab.model_factor import (
    ModelFactorBuildConfig,
    ModelFactorBuildResult,
    build_model_factor,
)
from alpha_lab.model_factor._memory import release_unused_memory
from alpha_lab.model_factor.dataset_cache import (
    ModelFactorDatasetCache,
    ResolvedFeatureAvailability,
)
from alpha_lab.model_factor.diagnostics import (
    ModelFactorDiagnosticsRecorder,
    StageLifecycleCallback,
)
from alpha_lab.real_cases.common_io import (
    apply_universe_to_factor,
    apply_universe_to_prices,
    ensure_parquet_tabular_frame,
    load_prices,
    load_universe_mask,
)
from alpha_lab.real_cases.single_factor.evaluate import (
    SingleFactorEvaluationResult,
    evaluate_single_factor_case,
)
from alpha_lab.research_evaluation_config import (
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
from alpha_lab.splits import (
    TimeSeriesSplitContract,
    infer_default_time_series_split_contract,
    rebalance_frequency_to_step,
)

from ..artifacts import (
    ModelFactorArtifactPaths,
    export_artifact_bundle,
    write_diagnostics_artifact,
)
from ..spec import (
    ModelFactorCaseSpec,
    infer_fundamental_feature_columns,
    load_model_factor_case_spec,
)

# Cross-module imports (auto-added)
from .cache import (
    _build_preparation_cache_key,
    _resolve_case_output_dir,
    _resolve_preparation_cache_dir,
    _resolve_screening_training_override,
)
from .diagnostics import _annotate_exception_with_diagnostics, _build_diagnostics_run_meta
from .feature_manifest import (
    _build_feature_manifest_payload,
    _metadata_int_or_none,
    _resolve_feature_availability_contract,
)
from .features import _coverage_by_date, _load_features, _maybe_neutralize_factor
from .labels import (
    _build_forward_label_cache,
    _enrich_label_extreme_samples,
    _group_return_extreme_rows,
    _model_factor_price_read_columns,
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
    stage_timings: dict[str, float]
    draft_model_source: DraftModelSource | None = None


def _strict_split_contract_check(
    contract: TimeSeriesSplitContract,
    *,
    object_name: str,
    module_name: str,
) -> IntegrityCheckResult:
    metadata = contract.to_metadata()
    return IntegrityCheckResult(
        check_name="strict_time_series_split_contract",
        status="pass",
        severity="info",
        object_name=object_name,
        module_name=module_name,
        message=(
            "Strict chronological IS/OOS split resolved before model training: "
            f"IS {metadata['is_start']}..{metadata['is_end']}, "
            f"OOS {metadata['oos_start']}..{metadata['oos_end']}, "
            f"embargo={metadata['embargo_days']}."
        ),
        metrics=metadata,
    )


def run_model_factor_case(
    spec_or_path: ModelFactorCaseSpec | str | Path,
    *,
    output_root_dir: str | Path | None = None,
    cache_root_dir: str | Path | None = None,
    evaluation_profile: str = "default_research",
    use_preparation_cache: bool = True,
    screening_retrain_every_n_dates: int | None = None,
    vault_root: str | Path | None = None,
    vault_export_mode: str = "versioned",
    progress_callback: Callable[[str, int], None] | None = None,
    stage_lifecycle_callback: StageLifecycleCallback | None = None,
    draft_model_source: DraftModelSource | None = None,
) -> ModelFactorCaseRunResult:
    """Run one real-case model-factor study end-to-end and export artifacts."""

    integrity_checks: list[IntegrityCheckResult] = []
    diagnostics = ModelFactorDiagnosticsRecorder(stage_lifecycle_callback=stage_lifecycle_callback)
    spec: ModelFactorCaseSpec | None = None
    spec_path: Path | None = None
    output_dir: Path | None = None
    evaluation_config = None
    universe_mask: pd.DataFrame | None = None
    prices = pd.DataFrame()
    features = pd.DataFrame()
    resolved_feature_availability: ResolvedFeatureAvailability | None = None
    feature_manifest_payload: dict[str, object] | None = None
    preparation_cache_key: str | None = None
    preparation_cache_dir: Path | None = None
    dataset_cache: ModelFactorDatasetCache | None = None
    model_build_prepared_cache_hit = False
    prepared_cache_metadata: dict[str, object] | None = None
    split_contract: TimeSeriesSplitContract | None = None

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
            preparation_cache_dir = _resolve_preparation_cache_dir(
                output_dir,
                cache_root_dir=cache_root_dir,
            )
            dataset_cache = ModelFactorDatasetCache(preparation_cache_dir)
            original_retrain_every_n_dates = int(spec.training.retrain_every_n_dates)
            profile_retrain_every_n_dates = (
                evaluation_config.model_factor_overrides.min_retrain_every_n_dates
                if evaluation_profile == "exploratory_screening"
                else None
            )
            effective_screening_retrain_every_n_dates = (
                screening_retrain_every_n_dates
                if screening_retrain_every_n_dates is not None
                else profile_retrain_every_n_dates
            )
            effective_training = _resolve_screening_training_override(
                spec=spec,
                evaluation_profile=evaluation_profile,
                screening_retrain_every_n_dates=effective_screening_retrain_every_n_dates,
                diagnostics=diagnostics,
            )
            if effective_training != spec.training:
                spec = replace(spec, training=effective_training)
            effective_retrain_every_n_dates = int(spec.training.retrain_every_n_dates)
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
                training_retrain_every_n_dates_original=original_retrain_every_n_dates,
                training_retrain_every_n_dates_effective=effective_retrain_every_n_dates,
                screening_retrain_every_n_dates=effective_screening_retrain_every_n_dates,
                preparation_cache_enabled=bool(use_preparation_cache),
                preparation_cache_dir=str(preparation_cache_dir),
                output_dir=str(output_dir),
            )
            if evaluation_profile == "exploratory_screening":
                diagnostics.event(
                    level="info",
                    stage="spec_load",
                    message="model-factor run uses exploratory screening profile",
                    payload={
                        "profile": evaluation_profile,
                        "diagnostic_max_dates": (
                            evaluation_config.single_factor_diagnostics.diagnostic_max_dates
                        ),
                        "training_retrain_every_n_dates_effective": (
                            effective_retrain_every_n_dates
                        ),
                    },
                )
            if evaluation_profile == "exploratory_screening" and spec.model_selection.enabled:
                diagnostics.warning(
                    title="快速筛选模式启用了模型选参",
                    severity="warning",
                    stage="spec_load",
                    description=(
                        "exploratory_screening 主要用于快速迭代，但当前 "
                        "model_selection.enabled=True，会显著增加 walk-forward 训练成本。"
                    ),
                    suggested_action=(
                        "开发筛选阶段建议关闭 model_selection；完整研究复核时再启用 "
                        "Purged CV 选参。"
                    ),
                )
        _emit_progress("实验合同与评估配置已加载", 10)
        assert spec is not None
        assert output_dir is not None
        assert evaluation_config is not None

        _emit_progress("加载行情、特征与可选股票池", 15)
        with diagnostics.stage("data_load") as data_stage:
            universe_mask = load_universe_mask(spec.universe)
            price_columns, optional_price_columns = _model_factor_price_read_columns(
                evaluation_config,
                target_price_column=spec.target.price_column,
            )
            feature_storage = ensure_parquet_tabular_frame(
                spec.features_path,
                object_name="features",
            )
            assert dataset_cache is not None
            preparation_cache_key = _build_preparation_cache_key(
                dataset_cache=dataset_cache,
                spec=spec,
                feature_storage_path=feature_storage.path,
                feature_source_path=feature_storage.source_path,
                price_columns=price_columns,
                optional_price_columns=optional_price_columns,
                evaluation_profile=evaluation_profile,
            )
            prepared_cache_metadata = (
                dataset_cache.prepared_inputs_metadata(preparation_cache_key)
                if use_preparation_cache
                else None
            )
            model_build_prepared_cache_hit = (
                use_preparation_cache
                and prepared_cache_metadata is not None
                and dataset_cache.prepared_inputs_exists(preparation_cache_key)
            )
            skip_data_load_features = (
                model_build_prepared_cache_hit
                and evaluation_profile == "exploratory_screening"
                and not spec.model_selection.enabled
            )
            cache_hit = (
                dataset_cache.load_data_load(
                    preparation_cache_key,
                    include_features=not skip_data_load_features,
                )
                if use_preparation_cache
                else None
            )
            if cache_hit is not None:
                prices = cache_hit.prices
                features = cache_hit.features
                resolved_feature_availability = cache_hit.resolved_feature_availability
                features_loaded_for_data_load = bool(cache_hit.features_loaded)
            else:
                prices = load_prices(
                    spec.prices_path,
                    columns=price_columns,
                    optional_columns=optional_price_columns,
                )
                features = _load_features(
                    str(feature_storage.path),
                    feature_columns=spec.feature_columns,
                    feature_availability=spec.feature_availability,
                    feature_preprocess=spec.feature_preprocess,
                )
                features_loaded_for_data_load = True
            max_price_date = pd.Timestamp(prices["date"].max())
            cache_metadata = cache_hit.metadata if cache_hit is not None else {}
            data_stage.attach(
                preparation_cache_enabled=bool(use_preparation_cache),
                preparation_cache_hit=bool(cache_hit is not None),
                prepared_inputs_cache_hit=bool(model_build_prepared_cache_hit),
                preparation_cache_key=preparation_cache_key,
                preparation_cache_dir=str(preparation_cache_dir),
                n_prices_rows=int(len(prices)),
                n_prices_assets=int(prices["asset"].nunique()),
                n_prices_dates=int(prices["date"].nunique()),
                prices_requested_columns=list(price_columns),
                prices_optional_columns=list(optional_price_columns),
                prices_loaded_columns=list(prices.columns),
                n_features_rows=(
                    int(len(features))
                    if features_loaded_for_data_load
                    else _metadata_int_or_none(cache_metadata.get("n_features_rows"))
                ),
                n_features_assets=(
                    int(features["asset"].nunique())
                    if features_loaded_for_data_load
                    else cache_metadata.get("n_features_assets")
                ),
                n_features_dates=(
                    int(features["date"].nunique())
                    if features_loaded_for_data_load
                    else cache_metadata.get("n_features_dates")
                ),
                features_loaded_for_data_load=bool(features_loaded_for_data_load),
                features_skipped_due_to_prepared_cache=bool(skip_data_load_features),
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
            if features_loaded_for_data_load:
                _record_integrity(
                    check_no_future_dates_in_input(
                        features,
                        max_allowed_date=max_price_date,
                        date_col="date",
                        object_name="model_factor_features_raw",
                    )
                )
            elif skip_data_load_features:
                diagnostics.event(
                    level="info",
                    stage="data_load",
                    message="skipped feature table reload because prepared inputs cache is warm",
                    payload={
                        "evaluation_profile": evaluation_profile,
                        "preparation_cache_key": preparation_cache_key,
                    },
                )
            if cache_hit is None:
                features, resolved_feature_availability = _resolve_feature_availability_contract(
                    features,
                    prices=prices,
                    contract=spec.feature_availability,
                )
            assert resolved_feature_availability is not None
            if features_loaded_for_data_load:
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
                if cache_hit is None:
                    prices = apply_universe_to_prices(prices, universe_mask)
                    features = apply_universe_to_factor(features, universe_mask)
            split_contract = infer_default_time_series_split_contract(
                prices["date"],
                target_horizon=int(spec.target.horizon),
                rebalance_step=rebalance_frequency_to_step(spec.rebalance_frequency),
                source="model_factor_pipeline",
            )
            _record_integrity(
                _strict_split_contract_check(
                    split_contract,
                    object_name="model_factor_strict_split",
                    module_name="real_cases.model_factor.pipeline",
                )
            )
            data_stage.attach(split_contract=split_contract.to_metadata())
        if use_preparation_cache and cache_hit is None:
            assert dataset_cache is not None
            dataset_cache.write_data_load(
                cache_key=preparation_cache_key,
                prices=prices,
                features=features,
                resolved_feature_availability=resolved_feature_availability,
            )

        _emit_progress("训练模型生成因子", 30)
        assert resolved_feature_availability is not None
        assert dataset_cache is not None
        assert preparation_cache_key is not None
        feature_manifest_payload = _build_feature_manifest_payload(
            spec=spec,
            features=features if not features.empty else None,
            resolved_feature_availability=resolved_feature_availability,
            cache_metadata=(
                prepared_cache_metadata or (cache_hit.metadata if cache_hit is not None else None)
            ),
        )
        if model_build_prepared_cache_hit:
            diagnostics.event(
                level="info",
                stage="data_load",
                message="model-build prepared input cache hit before training",
                payload={"preparation_cache_key": preparation_cache_key},
            )
            features_for_build = pd.DataFrame()
            features = pd.DataFrame()
            release_unused_memory()
        else:
            features_for_build = features
        known_at_col = resolved_feature_availability.known_at_col
        build_result = build_model_factor(
            features_for_build,
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
                target_price_column=spec.target.price_column,
                max_abs_forward_return=spec.target.max_abs_forward_return,
                label_winsorize_zscore=spec.target.winsorize_zscore,
                preparation_cache_dir=(
                    str(dataset_cache.prepared_inputs_root_dir)
                    if use_preparation_cache and dataset_cache is not None
                    else None
                ),
                preparation_cache_key=preparation_cache_key if use_preparation_cache else None,
                compute_feature_oos_ic=evaluation_profile != "exploratory_screening",
            ),
            observer=diagnostics,
            progress_callback=_emit_training_progress,
        )
        integrity_checks.extend(build_result.integrity_checks)
        # The wide feature matrix is no longer needed after training and manifest
        # summarization. Release it before evaluation/artifact export to reduce
        # peak RSS for large model-lab runs.
        features_for_build = pd.DataFrame()
        features = pd.DataFrame()
        release_unused_memory()

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
        coverage_by_date = _coverage_by_date(
            factor_df,
            coverage_base_df=build_result.coverage_base_df,
            target_label_df=build_result.forward_label_df,
        )

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
            target_price_column=spec.target.price_column,
            max_abs_forward_return=spec.target.max_abs_forward_return,
            evaluation_config=evaluation_config,
        )
        assert split_contract is not None

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
                split_contract=split_contract,
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
            group_return_extremes = _group_return_extreme_rows(evaluation_result.group_returns)
            if group_return_extremes:
                diagnostics.warning(
                    title="分组收益存在极端值",
                    severity="warning",
                    stage="evaluate",
                    description=(
                        "group_returns.csv 中检测到单组单期收益绝对值超过 30%，"
                        "请优先排查复权价格、停复牌和占位价格。"
                    ),
                    suggested_action=(
                        "查看 diagnostics 中的 group_return_extreme_rows，并重建可信价格 label。"
                    ),
                )
                diagnostics.event(
                    level="warning",
                    stage="evaluate",
                    message="extreme group-return rows detected",
                    payload={
                        "threshold": 0.30,
                        "rows": group_return_extremes,
                    },
                )
            enriched_label_extremes = _enrich_label_extreme_samples(
                build_result.target_diagnostics.get("label_extreme_top_samples"),
                assignments=evaluation_result.experiment_result.quantile_assignments_df,
                group_returns=evaluation_result.group_returns,
            )
            if enriched_label_extremes:
                diagnostics.event(
                    level="warning",
                    stage="evaluate",
                    message="extreme label samples with quantile groups",
                    payload={"rows": enriched_label_extremes},
                )
            evaluate_stage.attach(
                n_factor_rows=int(len(factor_df)),
                n_factor_dates=int(factor_df["date"].nunique()),
                n_factor_assets=int(factor_df["asset"].nunique()),
                n_group_return_extreme_rows=len(group_return_extremes),
                n_label_extreme_sample_rows=len(enriched_label_extremes),
                evaluation_profile=evaluation_profile,
                evaluation_stage_timings=dict(evaluation_result.stage_timings),
                neutralization_enabled=bool(spec.neutralization.enabled),
                forward_label_cache_horizons=sorted(forward_label_cache),
                forward_label_cache_rows={
                    int(horizon): int(len(labels))
                    for horizon, labels in forward_label_cache.items()
                },
                split_contract=split_contract.to_metadata(),
            )
        forward_label_cache.clear()
        prices = pd.DataFrame()
        release_unused_memory()

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
                "split_contract": split_contract.to_metadata(),
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
                draft_model_source=(
                    draft_model_source.to_audit_dict() if draft_model_source is not None else None
                ),
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
        stage_timings = diagnostics.stage_timings_summary()
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
            stage_timings=stage_timings,
            draft_model_source=draft_model_source,
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
