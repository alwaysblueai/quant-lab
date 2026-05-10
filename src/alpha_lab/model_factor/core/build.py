from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import numpy as np
import pandas as pd

from alpha_lab.interfaces import validate_factor_output
from alpha_lab.labels import forward_return
from alpha_lab.model_factor._memory import release_unused_memory
from alpha_lab.model_factor.dataset_cache import (
    load_prepared_inputs_cache,
    write_prepared_inputs_cache,
)
from alpha_lab.model_factor.diagnostics import (
    ModelFactorDiagnosticsObserver,
    compute_data_health_snapshot,
    derive_data_health_warnings,
    diagnostics_observer_or_null,
)
from alpha_lab.research_integrity.contracts import IntegrityCheckResult
from alpha_lab.research_integrity.leakage_checks import check_no_future_dates_in_input

# Cross-module imports (auto-added)
from ._utils import _finite_or_none, _object_to_int, _weakref_or_none
from .config import (
    ModelFactorBuildConfig,
    ModelFactorBuildResult,
    _feature_importance_enabled,
    _feature_importance_over_time_enabled,
    _feature_importance_over_time_source,
    _feature_importance_over_time_top_k,
    _feature_importance_permutation_enabled,
    _feature_importance_permutation_latest_only,
    _feature_importance_permutation_n_repeats,
    _feature_importance_permutation_sample_rows,
    _feature_importance_permutation_top_k_features,
)
from .diagnostics_build import (
    _build_model_diagnostics,
    _check_feature_known_at_not_after_signal_date,
    _raise_on_integrity_failures,
)
from .estimator import _fit_model_bundle, _fit_model_bundle_from_arrays
from .importance import (
    _combine_feature_importance_frames,
    _combine_feature_importance_ledger_frames,
    _estimated_permutation_predict_calls,
    _feature_importance_frame,
    _feature_importance_training_slice,
    _permutation_importance_guardrail_reason,
)
from .internals import _FeatureImportanceRequest, _FittedModelBundle, _PreparedModelArrays
from .preprocess import (
    _apply_cross_sectional_transform,
    _apply_forward_return_extreme_filter,
    _build_score_coverage_base_frame,
    _industry_group_temporal_profile,
    _normalize_features,
    _normalize_prices,
    _prices_for_target_labels,
    _target_diagnostics_from_data_health,
    _winsorize_labels_per_date,
)
from .selection import (
    _build_feature_oos_ic_frame,
    _build_training_metrics_frame,
    _feature_oos_ic_rows,
    _oos_training_metrics_row,
    _select_model_candidate,
    _selection_candidates,
    _selection_has_mlp_early_stopping,
    _training_metrics_row_from_arrays,
    _training_metrics_row_from_frame,
)
from .training_arrays import (
    _build_date_indexed_rows,
    _build_training_window_cache,
    _feature_importance_training_slice_from_arrays,
    _prepare_model_arrays,
    _prepare_model_arrays_from_numpy_cache,
    _row_selection_mode,
)
from .types import _RowSelection


def build_model_factor(
    features_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    config: ModelFactorBuildConfig,
    *,
    observer: ModelFactorDiagnosticsObserver | None = None,
    progress_callback: Callable[[str, int], None] | None = None,
) -> ModelFactorBuildResult:
    """Train walk-forward models and emit canonical `[date, asset, factor, value]` output."""

    diagnostics = diagnostics_observer_or_null(observer)

    def _emit_progress(message: str, percent: int) -> None:
        if progress_callback is not None:
            progress_callback(message, max(0, min(int(percent), 100)))

    with diagnostics.stage("preprocess", payload={"object": "prices"}) as prices_stage:
        prices = _normalize_prices(prices_df)
        prices_stage.attach(
            n_rows=int(len(prices)),
            n_assets=int(prices["asset"].nunique()),
            n_dates=int(prices["date"].nunique()),
        )
    prepared_cache = load_prepared_inputs_cache(
        cache_dir=config.preparation_cache_dir,
        cache_key=config.preparation_cache_key,
    )
    if (
        prepared_cache is not None
        and prepared_cache.numpy_entry is not None
        and config.model_selection.enabled
    ):
        prepared_cache = None
    with diagnostics.stage(
        "feature_validate",
        payload={"feature_count": len(config.feature_columns)},
    ) as feature_stage:
        if prepared_cache is not None and prepared_cache.numpy_entry is not None:
            features = prepared_cache.numpy_entry.index_df.copy()
            cross_mode = config.feature_preprocess.cross_sectional_transform
            feature_stage.attach(
                cache_hit=True,
                cache_layout="numpy_v2",
                row_order=prepared_cache.metadata.get("row_order"),
                feature_dtype=prepared_cache.metadata.get("feature_dtype"),
            )
        elif prepared_cache is not None:
            features = prepared_cache.features
            cross_mode = config.feature_preprocess.cross_sectional_transform
            feature_stage.attach(cache_hit=True, cache_layout="dataframe_v1")
        else:
            features = _normalize_features(features_df, config=config)
            if (
                config.feature_preprocess.cross_sectional_group_scope == "date_and_industry"
                and config.feature_preprocess.industry_group_column is not None
            ):
                industry_profile = _industry_group_temporal_profile(
                    features,
                    industry_group_column=config.feature_preprocess.industry_group_column,
                )
                feature_stage.attach(
                    industry_group_assets_total=industry_profile["n_assets_total"],
                    industry_group_assets_eligible=industry_profile["n_assets_eligible"],
                    industry_group_static_assets=industry_profile["n_assets_static"],
                    industry_group_static_asset_ratio=industry_profile["static_ratio"],
                )
                if bool(industry_profile["all_assets_static"]):
                    diagnostics.warning(
                        title="行业分组列疑似静态映射",
                        severity="warning",
                        stage="feature_validate",
                        description=(
                            "cross_sectional_group_scope='date_and_industry' 下检测到 "
                            "asset->industry 在样本期内全部不变。该行业列可能来自静态快照，"
                            "存在将未来重分类回填到历史截面的风险。"
                        ),
                        suggested_action=(
                            "优先使用带生效时间的行业历史映射表；若暂时无法提供，请在报告中标注 "
                            "industry 分组的 PIT 假设边界。"
                        ),
                    )
            cross_mode = config.feature_preprocess.cross_sectional_transform
            features = _apply_cross_sectional_transform(
                features,
                feature_columns=config.feature_columns,
                mode=cross_mode,
                group_scope=config.feature_preprocess.cross_sectional_group_scope,
                industry_group_column=config.feature_preprocess.industry_group_column,
            )
            feature_stage.attach(cache_hit=False)
        feature_stage.attach(
            n_rows=int(len(features)),
            n_assets=int(features["asset"].nunique()),
            n_dates=int(features["date"].nunique()),
            feature_count=int(len(config.feature_columns)),
            cross_sectional_transform=cross_mode,
        )

    integrity_checks: list[IntegrityCheckResult] = []
    max_price_date = pd.Timestamp(prices["date"].max())
    integrity_checks.append(
        check_no_future_dates_in_input(
            prices,
            max_allowed_date=max_price_date,
            date_col="date",
            object_name="model_factor_prices",
        )
    )
    integrity_checks.append(
        check_no_future_dates_in_input(
            features,
            max_allowed_date=max_price_date,
            date_col="date",
            object_name="model_factor_features",
        )
    )
    if config.known_at_col is not None:
        integrity_checks.append(
            _check_feature_known_at_not_after_signal_date(
                features,
                known_at_col=config.known_at_col,
            )
        )
    _raise_on_integrity_failures(integrity_checks)

    with diagnostics.stage(
        "target_build",
        payload={"target_horizon": int(config.target_horizon)},
    ) as target_stage:
        price_universe_counts = prices.groupby("date", sort=True)["asset"].nunique()
        if prepared_cache is not None:
            forward_label_df = prepared_cache.forward_label_df
            labels = prepared_cache.labels
            data_health = dict(prepared_cache.data_health)
            target_diagnostics = _target_diagnostics_from_data_health(data_health)
            winsor_clip_count = int(prepared_cache.winsor_clip_count)
            winsor_z = config.label_winsorize_zscore
            target_stage.attach(cache_hit=True)
        else:
            label_prices = _prices_for_target_labels(
                prices,
                price_column=config.target_price_column,
            )
            label_df = forward_return(label_prices, horizon=config.target_horizon)
            label_name = f"forward_return_{config.target_horizon}"
            forward_label_df = (
                label_df[label_df["factor"] == label_name][["date", "asset", "factor", "value"]]
                .copy()
                .reset_index(drop=True)
            )
            labels = forward_label_df[["date", "asset", "value"]].rename(columns={"value": "label"})
            forward_label_df, labels, target_diagnostics = _apply_forward_return_extreme_filter(
                forward_label_df,
                labels,
                label_prices=label_prices,
                target_price_column=config.target_price_column,
                horizon=int(config.target_horizon),
                max_abs_forward_return=config.max_abs_forward_return,
            )
            del label_prices
            del label_df
            winsor_z = config.label_winsorize_zscore
            winsor_clip_count = 0
            if winsor_z is not None and not labels.empty:
                labels, winsor_clip_count = _winsorize_labels_per_date(labels, z=float(winsor_z))
            data_health = compute_data_health_snapshot(
                features=features,
                labels=labels,
                feature_columns=config.feature_columns,
            )
            data_health.update(target_diagnostics)
            cache_write_succeeded = write_prepared_inputs_cache(
                cache_dir=config.preparation_cache_dir,
                cache_key=config.preparation_cache_key,
                features=features,
                labels=labels,
                forward_label_df=forward_label_df,
                data_health=data_health,
                winsor_clip_count=winsor_clip_count,
                feature_columns=config.feature_columns,
            )
            prepared_cache_adopted_for_run = False
            if cache_write_succeeded and not config.model_selection.enabled:
                written_prepared_cache = load_prepared_inputs_cache(
                    cache_dir=config.preparation_cache_dir,
                    cache_key=config.preparation_cache_key,
                )
                if (
                    written_prepared_cache is not None
                    and written_prepared_cache.numpy_entry is not None
                ):
                    prepared_cache = written_prepared_cache
                    features = written_prepared_cache.numpy_entry.index_df.copy()
                    labels = written_prepared_cache.labels
                    forward_label_df = written_prepared_cache.forward_label_df
                    prepared_cache_adopted_for_run = True
                    diagnostics.event(
                        level="info",
                        stage="target_build",
                        message="prepared input cache adopted for cold run",
                        payload={
                            "preparation_cache_key": config.preparation_cache_key,
                            "cache_layout": written_prepared_cache.metadata.get("layout"),
                        },
                    )
                    release_unused_memory()
            target_stage.attach(
                cache_hit=False,
                prepared_cache_write_succeeded=bool(cache_write_succeeded),
                prepared_cache_adopted_for_run=bool(prepared_cache_adopted_for_run),
            )
        diagnostics.set_data_health(data_health)
        warnings_emitted = 0
        for warning in derive_data_health_warnings(data_health):
            diagnostics.warning(**warning)
            warnings_emitted += 1
        target_stage.attach(
            n_label_rows=int(len(labels)),
            n_forward_label_cache_rows=int(len(forward_label_df)),
            target_price_column=config.target_price_column,
            max_abs_forward_return=config.max_abs_forward_return,
            label_extreme_filtered_rows=target_diagnostics.get("label_extreme_filtered_rows"),
            label_extreme_max_abs_raw_return=target_diagnostics.get(
                "label_extreme_max_abs_raw_return"
            ),
            target_mean=data_health.get("target_mean"),
            target_std=data_health.get("target_std"),
            outlier_ratio=data_health.get("outlier_ratio"),
            coverage_ratio=data_health.get("coverage_ratio"),
            warnings_emitted=warnings_emitted,
            label_winsorize_zscore=winsor_z if winsor_z is not None else "none",
            label_winsor_clipped_rows=winsor_clip_count,
        )
        if _object_to_int(target_diagnostics.get("label_extreme_filtered_rows")) > 0:
            diagnostics.warning(
                title="目标收益存在极端值",
                severity="warning",
                stage="target_build",
                description=(
                    "forward return 中检测到超过 target.max_abs_forward_return 的样本，"
                    "这些 label 已置为 NaN 并从训练/评估中排除。"
                ),
                suggested_action=(
                    "优先检查复权价格列和异常停复牌/占位价格；必要时重建输入价格面板。"
                ),
            )
            diagnostics.event(
                level="warning",
                stage="target_build",
                message="extreme forward-return labels filtered",
                payload={
                    "target_price_column": config.target_price_column,
                    "max_abs_forward_return": config.max_abs_forward_return,
                    "filtered_rows": target_diagnostics.get("label_extreme_filtered_rows"),
                    "max_abs_raw_return": target_diagnostics.get(
                        "label_extreme_max_abs_raw_return"
                    ),
                    "top_samples": target_diagnostics.get("label_extreme_top_samples"),
                },
            )
    prepared_numpy_entry = prepared_cache.numpy_entry if prepared_cache is not None else None
    prepared_cache = None
    prices = pd.DataFrame()
    release_unused_memory()

    with diagnostics.stage("preprocess", payload={"step": "merge_labels"}) as merge_stage:
        if prepared_numpy_entry is not None:
            if "label" not in features.columns:
                raise ValueError("prepared numpy cache index is missing label column")
            label_source = "prepared_numpy_index"
        else:
            features = features.merge(
                labels,
                on=["date", "asset"],
                how="left",
                validate="one_to_one",
            )
            label_source = "label_merge"
        merge_stage.attach(
            n_rows=int(len(features)),
            n_labeled_rows=int(features["label"].notna().sum()),
            label_source=label_source,
        )
    coverage_base_df = _build_score_coverage_base_frame(
        features,
        feature_columns=config.feature_columns,
        price_universe_counts=price_universe_counts,
    )
    labels = pd.DataFrame()
    price_universe_counts = pd.Series(dtype="int64")
    release_unused_memory()
    score_dates = list(
        pd.Index(features["date"].drop_duplicates()).sort_values().to_pydatetime().tolist()
    )
    score_date_index = pd.DatetimeIndex(score_dates)
    total_score_dates = len(score_dates)
    with diagnostics.stage(
        "training_window_index",
        payload={"n_score_dates": total_score_dates},
    ) as window_stage:
        feature_date_index = _build_date_indexed_rows(features, score_date_index)
        if prepared_numpy_entry is not None and not config.model_selection.enabled:
            labeled_date_index = _build_date_indexed_rows(
                prepared_numpy_entry.labeled_index_df,
                score_date_index,
                allow_missing=True,
            )
            training_window_cache = _build_training_window_cache(
                date_index=labeled_date_index,
                merged=prepared_numpy_entry.labeled_index_df,
                training=config.training,
                target_horizon=config.target_horizon,
            )
            row_index_cache_mode = "compact_labeled_numpy_windows_no_retention"
        else:
            training_window_cache = _build_training_window_cache(
                date_index=feature_date_index,
                merged=features,
                training=config.training,
                target_horizon=config.target_horizon,
            )
            row_index_cache_mode = "on_demand_fit_windows_no_retention"
        n_train_dates_values = training_window_cache.n_train_dates_by_pos
        n_labeled_rows_values = training_window_cache.n_labeled_rows_by_pos
        window_stage.attach(
            n_score_dates=total_score_dates,
            window_type=config.training.window_type,
            target_horizon=int(config.target_horizon),
            purged_train_gap_dates=max(int(config.target_horizon) - 1, 0),
            min_train_dates=(
                int(n_train_dates_values.min()) if len(n_train_dates_values) > 0 else 0
            ),
            max_train_dates=(
                int(n_train_dates_values.max()) if len(n_train_dates_values) > 0 else 0
            ),
            mean_train_dates=_finite_or_none(
                float(n_train_dates_values.mean())
                if len(n_train_dates_values) > 0
                else float("nan")
            ),
            min_labeled_train_rows=(
                int(n_labeled_rows_values.min()) if len(n_labeled_rows_values) > 0 else 0
            ),
            max_labeled_train_rows=(
                int(n_labeled_rows_values.max()) if len(n_labeled_rows_values) > 0 else 0
            ),
            mean_labeled_train_rows=_finite_or_none(
                float(n_labeled_rows_values.mean())
                if len(n_labeled_rows_values) > 0
                else float("nan")
            ),
            row_index_cache_mode=row_index_cache_mode,
        )

    prepared_arrays: _PreparedModelArrays | None = None
    if not config.model_selection.enabled:
        with diagnostics.stage(
            "preprocess",
            payload={
                "step": "prepare_model_arrays",
                "feature_count": len(config.feature_columns),
            },
        ) as array_stage:
            if prepared_numpy_entry is not None:
                source_features_ref = None
                prepared_arrays = _prepare_model_arrays_from_numpy_cache(prepared_numpy_entry)
                cache_layout = "numpy_v2_mmap"
            else:
                source_features_ref = _weakref_or_none(features)
                prepared_arrays = _prepare_model_arrays(
                    features,
                    feature_columns=config.feature_columns,
                )
                cache_layout = "runtime_dataframe_to_numpy"
                features = pd.DataFrame()
                release_unused_memory()
            array_stage.attach(
                model_matrix_mode="numpy_arrays_after_window_index",
                cache_layout=cache_layout,
                n_rows=int(len(prepared_arrays.labels)),
                n_features=int(prepared_arrays.feature_values.shape[1]),
                feature_dtype=str(prepared_arrays.feature_values.dtype),
                label_dtype=str(prepared_arrays.labels.dtype),
                asset_categories=int(len(prepared_arrays.assets.categories)),
                has_compact_training_matrix=prepared_arrays.training_feature_values is not None,
                source_dataframe_released=(
                    None if source_features_ref is None else source_features_ref() is None
                ),
            )

    factor_frames: list[pd.DataFrame] = []
    training_log_rows: list[dict[str, object]] = []
    training_metrics_rows: list[dict[str, object]] = []
    oos_metrics_rows: list[dict[str, object]] = []
    feature_oos_ic_rows: list[dict[str, object]] = []
    per_fit_importance_frames: list[pd.DataFrame] = []
    latest_importance_request: _FeatureImportanceRequest | None = None
    model_selection_rows: list[dict[str, object]] = []
    current_bundle: _FittedModelBundle | None = None
    pending_prediction_bundle: _FittedModelBundle | None = None
    pending_prediction_indices: list[np.ndarray] = []
    pending_prediction_dates: list[pd.Timestamp] = []
    pending_prediction_statuses: list[str] = []
    last_fit_score_idx: int | None = None
    skipped_score_dates_by_reason: Counter[str] = Counter()
    skipped_score_date_samples: dict[str, list[str]] = {}
    model_version = 0

    def flush_pending_predictions() -> None:
        nonlocal pending_prediction_bundle
        if pending_prediction_bundle is None or not pending_prediction_indices:
            pending_prediction_bundle = None
            pending_prediction_indices.clear()
            pending_prediction_dates.clear()
            pending_prediction_statuses.clear()
            return

        row_idx = np.concatenate(pending_prediction_indices)
        score_dates_text = [date.date().isoformat() for date in pending_prediction_dates]
        payload: dict[str, object] = {
            "score_date_start": score_dates_text[0],
            "score_date_end": score_dates_text[-1],
            "model_version": pending_prediction_bundle.model_version,
            "n_score_dates": len(pending_prediction_dates),
            "n_score_rows": int(len(row_idx)),
        }
        with diagnostics.stage("predict", payload=payload) as predict_stage:
            if prepared_arrays is not None:
                score_features = prepared_arrays.feature_values[row_idx]
                predictions = pending_prediction_bundle.pipeline.predict(score_features)
                factor_dates = prepared_arrays.dates[row_idx]
                labels = prepared_arrays.labels[row_idx]
                oos_metrics_rows.append(
                    _oos_training_metrics_row(
                        model_version=pending_prediction_bundle.model_version,
                        dates=factor_dates,
                        labels=labels,
                        predictions=np.asarray(predictions, dtype=float),
                    )
                )
                if config.compute_feature_oos_ic:
                    feature_oos_ic_rows.extend(
                        _feature_oos_ic_rows(
                            model_version=pending_prediction_bundle.model_version,
                            feature_columns=config.feature_columns,
                            dates=factor_dates,
                            labels=labels,
                            feature_values=score_features,
                        )
                    )
                factor_assets = np.asarray(
                    prepared_arrays.assets.take(row_idx).astype(str),
                    dtype=object,
                )
                factor_frames.append(
                    pd.DataFrame(
                        {
                            "date": pd.to_datetime(factor_dates),
                            "asset": factor_assets,
                            "factor": config.factor_name,
                            "value": np.asarray(predictions, dtype=float),
                        }
                    )
                )
                del score_features, predictions, factor_dates, factor_assets, labels
            else:
                score_slice = features.take(row_idx)
                score_features = score_slice.loc[:, list(config.feature_columns)]
                predictions = pending_prediction_bundle.pipeline.predict(score_features)
                oos_metrics_rows.append(
                    _oos_training_metrics_row(
                        model_version=pending_prediction_bundle.model_version,
                        dates=score_slice["date"].to_numpy(),
                        labels=score_slice["label"].to_numpy(),
                        predictions=np.asarray(predictions, dtype=float),
                    )
                )
                if config.compute_feature_oos_ic:
                    feature_oos_ic_rows.extend(
                        _feature_oos_ic_rows(
                            model_version=pending_prediction_bundle.model_version,
                            feature_columns=config.feature_columns,
                            dates=score_slice["date"].to_numpy(),
                            labels=score_slice["label"].to_numpy(),
                            feature_values=score_features,
                        )
                    )
                factor_frames.append(
                    pd.DataFrame(
                        {
                            "date": pd.to_datetime(score_slice["date"]).to_numpy(),
                            "asset": score_slice["asset"].astype(str).to_numpy(),
                            "factor": config.factor_name,
                            "value": np.asarray(predictions, dtype=float),
                        }
                    )
                )
                del score_slice, score_features, predictions
            predict_stage.attach(
                score_date_start=score_dates_text[0],
                score_date_end=score_dates_text[-1],
                score_date_first_values=score_dates_text[:3],
                score_date_last_values=score_dates_text[-3:],
                model_version=pending_prediction_bundle.model_version,
                n_score_dates=len(pending_prediction_dates),
                n_score_rows=int(len(row_idx)),
                model_matrix_mode=(
                    "numpy_arrays_after_window_index"
                    if prepared_arrays is not None
                    else "dataframe"
                ),
                statuses=sorted(set(pending_prediction_statuses)),
                status="batch_scored",
            )

        pending_prediction_bundle = None
        pending_prediction_indices.clear()
        pending_prediction_dates.clear()
        pending_prediction_statuses.clear()
        del row_idx

    def append_feature_importance(request: _FeatureImportanceRequest) -> None:
        sample_rows = _feature_importance_permutation_sample_rows(config.feature_importance)
        permutation_guardrail = _permutation_importance_guardrail_reason(
            config.feature_importance,
            model_family=request.model_family,
            n_versions_for_estimate=(
                1
                if _feature_importance_permutation_latest_only(config.feature_importance)
                else max(1, request.model_version)
            ),
            n_features=len(request.feature_columns),
        )
        payload = {
            "model_version": request.model_version,
            "model_family": request.model_family,
            "mode": config.feature_importance.mode,
            "method": config.feature_importance.method,
            "save_ledger": config.feature_importance.save_ledger,
            "over_time_source": _feature_importance_over_time_source(config.feature_importance),
            "permutation_enabled": _feature_importance_permutation_enabled(
                config.feature_importance
            ),
            "permutation_latest_only": _feature_importance_permutation_latest_only(
                config.feature_importance
            ),
            "permutation_sample_rows": sample_rows,
            "permutation_n_repeats": _feature_importance_permutation_n_repeats(
                config.feature_importance
            ),
            "permutation_guardrail_reason": permutation_guardrail,
            "n_train_rows": int(len(request.train_slice)),
        }
        with diagnostics.stage("feature_importance", payload=payload) as importance_stage:
            frame = _feature_importance_frame(
                request.pipeline,
                train_slice=request.train_slice,
                feature_columns=request.feature_columns,
                model_family=request.model_family,
                model_version=request.model_version,
                fit_date=request.fit_date,
                trained_until=request.trained_until,
                config=config.feature_importance,
                permutation_guardrail_reason=permutation_guardrail,
            )
            per_fit_importance_frames.append(frame)
            importance_stage.attach(
                model_version=request.model_version,
                model_family=request.model_family,
                mode=config.feature_importance.mode,
                method=config.feature_importance.method,
                save_ledger=config.feature_importance.save_ledger,
                over_time_enabled=_feature_importance_over_time_enabled(config.feature_importance),
                over_time_top_k=_feature_importance_over_time_top_k(config.feature_importance),
                over_time_source=_feature_importance_over_time_source(config.feature_importance),
                permutation_enabled=_feature_importance_permutation_enabled(
                    config.feature_importance
                ),
                permutation_sample_rows=sample_rows,
                permutation_n_repeats=_feature_importance_permutation_n_repeats(
                    config.feature_importance
                ),
                permutation_top_k_features=_feature_importance_permutation_top_k_features(
                    config.feature_importance
                ),
                estimated_predict_calls=_estimated_permutation_predict_calls(
                    config.feature_importance,
                    n_versions_for_estimate=(
                        1
                        if _feature_importance_permutation_latest_only(config.feature_importance)
                        else max(1, request.model_version)
                    ),
                    n_features=len(request.feature_columns),
                ),
                permutation_guardrail_reason=permutation_guardrail,
                n_features=int(len(request.feature_columns)),
                n_train_rows=int(len(request.train_slice)),
                importance_sources=sorted(
                    {str(value) for value in frame["importance_source"].dropna().unique().tolist()}
                ),
            )

    if _selection_has_mlp_early_stopping(config):
        diagnostics.warning(
            title="MLP 早停验证存在时序局限",
            severity="warning",
            stage="model_fit",
            description=(
                "检测到 MLP 启用 early_stopping。sklearn 的早停验证集会跨日期抽样，"
                "不是严格时间序列验证。"
            ),
            suggested_action=(
                "如需时间感知验证，优先使用 model_selection 内层 Purged CV 进行选模，"
                "不要将该早停验证分数作为时序外推依据。"
            ),
        )

    for score_idx, raw_score_date in enumerate(score_dates):
        score_date = pd.Timestamp(raw_score_date)
        score_row_indices = feature_date_index.row_indices_by_pos[score_idx]
        score_step = score_idx + 1
        if total_score_dates:
            _emit_progress(
                (
                    "训练模型生成因子："
                    f"第 {score_step}/{total_score_dates} 个评分日 "
                    f"{score_date.date().isoformat()}，正在切分训练窗口"
                ),
                round(score_idx * 100 / total_score_dates),
            )
        n_score_assets = int(len(score_row_indices))
        n_train_dates = int(training_window_cache.n_train_dates_by_pos[score_idx])
        n_train_rows = int(training_window_cache.n_labeled_rows_by_pos[score_idx])

        status = "reused_scored"
        skip_reason: str | None = None
        selection_status = "disabled" if not config.model_selection.enabled else "not_run"
        selection_metric = config.model_selection.metric if config.model_selection.enabled else None
        selected_candidate_id: str | None = None
        selected_candidate_score: float | None = None
        selected_candidate_turnover: float | None = None
        should_fit = current_bundle is None or (
            last_fit_score_idx is not None
            and score_idx - last_fit_score_idx >= config.training.retrain_every_n_dates
        )

        if n_score_assets < config.training.min_score_assets:
            current_bundle = current_bundle
            status = "skipped"
            skip_reason = "insufficient_score_assets"
        elif current_bundle is None and n_train_dates < config.training.min_train_dates:
            status = "skipped"
            skip_reason = "insufficient_train_dates"
        elif current_bundle is None and n_train_rows < config.training.min_train_rows:
            status = "skipped"
            skip_reason = "insufficient_train_rows"
        elif (
            should_fit
            and n_train_dates >= config.training.min_train_dates
            and n_train_rows >= config.training.min_train_rows
        ):
            flush_pending_predictions()
            train_labeled: pd.DataFrame | None = None
            train_row_selection: _RowSelection | None = None
            if prepared_arrays is not None:
                train_row_selection = training_window_cache.labeled_row_selection(score_idx)
            else:
                train_labeled = training_window_cache.labeled_slice(features, score_idx)
            model_version += 1
            selected_model = config.model
            n_selection_splits = 0
            if config.model_selection.enabled:
                assert train_labeled is not None
                selection_metric = config.model_selection.metric
                with diagnostics.stage(
                    "model_selection",
                    payload={
                        "score_date": score_date.date().isoformat(),
                        "model_version": model_version,
                        "n_train_rows": n_train_rows,
                        "n_train_dates": n_train_dates,
                        "candidate_count": len(_selection_candidates(config)),
                        "selection_metric": config.model_selection.metric,
                    },
                ) as selection_stage:
                    selection = _select_model_candidate(
                        train_slice=train_labeled,
                        config=config,
                        score_date=score_date,
                        model_version=model_version,
                    )
                    selection_stage.attach(
                        score_date=score_date.date().isoformat(),
                        model_version=model_version,
                        status=selection.status,
                        selected_candidate_id=selection.selected_candidate_id,
                        selected_model_family=selection.selected_model.family,
                        selected_score=selection.selected_score,
                        n_splits_used=selection.n_splits_used,
                    )
                selected_model = selection.selected_model
                selection_status = selection.status
                selected_candidate_id = selection.selected_candidate_id
                selected_candidate_score = selection.selected_score
                selected_candidate_turnover = selection.selected_turnover
                n_selection_splits = selection.n_splits_used
                model_selection_rows.extend(selection.rows)
            with diagnostics.stage(
                "model_fit",
                payload={
                    "score_date": score_date.date().isoformat(),
                    "model_version": model_version,
                    "n_train_rows": n_train_rows,
                    "n_train_dates": n_train_dates,
                },
            ) as fit_stage:
                if prepared_arrays is not None and train_row_selection is not None:
                    fitted = _fit_model_bundle_from_arrays(
                        prepared_arrays=prepared_arrays,
                        row_selection=train_row_selection,
                        config=config,
                        model_version=model_version,
                        model_spec=selected_model,
                        selected_candidate_id=selected_candidate_id,
                        selection_score=selected_candidate_score,
                        selected_candidate_turnover=selected_candidate_turnover,
                    )
                else:
                    assert train_labeled is not None
                    fitted = _fit_model_bundle(
                        train_slice=train_labeled,
                        config=config,
                        model_version=model_version,
                        model_spec=selected_model,
                        selected_candidate_id=selected_candidate_id,
                        selection_score=selected_candidate_score,
                        selected_candidate_turnover=selected_candidate_turnover,
                    )
                if prepared_arrays is not None and train_row_selection is not None:
                    train_metrics_row = _training_metrics_row_from_arrays(
                        fitted=fitted,
                        prepared_arrays=prepared_arrays,
                        row_selection=train_row_selection,
                        selected_candidate_id=selected_candidate_id,
                        selected_candidate_score=selected_candidate_score,
                    )
                else:
                    assert train_labeled is not None
                    train_metrics_row = _training_metrics_row_from_frame(
                        fitted=fitted,
                        train_slice=train_labeled,
                        feature_columns=config.feature_columns,
                        selected_candidate_id=selected_candidate_id,
                        selected_candidate_score=selected_candidate_score,
                    )
                training_metrics_rows.append(train_metrics_row)
                fit_stage.attach(
                    score_date=score_date.date().isoformat(),
                    model_version=fitted.model_version,
                    n_train_rows=fitted.n_train_rows,
                    n_train_dates=fitted.n_train_dates,
                    scale_mode=fitted.scale_mode,
                    model_family=fitted.model_family,
                    model_matrix_mode=(
                        "numpy_arrays_after_window_index"
                        if prepared_arrays is not None
                        else "dataframe"
                    ),
                    model_matrix_selection=(
                        _row_selection_mode(train_row_selection)
                        if train_row_selection is not None
                        else "dataframe"
                    ),
                    train_start=fitted.train_start.date().isoformat(),
                    train_end=fitted.train_end.date().isoformat(),
                    selection_status=selection_status,
                    selected_candidate_id=selected_candidate_id,
                    selected_candidate_score=selected_candidate_score,
                    selected_candidate_turnover=selected_candidate_turnover,
                    n_selection_splits=n_selection_splits,
                    train_ic=train_metrics_row.get("train_ic"),
                    train_rank_ic=train_metrics_row.get("train_rank_ic"),
                    train_loss=train_metrics_row.get("train_loss"),
                )
            current_bundle = fitted
            last_fit_score_idx = score_idx
            status = "fit_scored"
            if _feature_importance_enabled(config.feature_importance):
                if _feature_importance_permutation_enabled(config.feature_importance):
                    if prepared_arrays is not None and train_row_selection is not None:
                        importance_train_slice = _feature_importance_training_slice_from_arrays(
                            prepared_arrays,
                            row_selection=train_row_selection,
                            feature_columns=config.feature_columns,
                            model_version=fitted.model_version,
                            max_rows=_feature_importance_permutation_sample_rows(
                                config.feature_importance
                            ),
                        )
                    else:
                        assert train_labeled is not None
                        importance_train_slice = _feature_importance_training_slice(
                            train_labeled,
                            feature_columns=config.feature_columns,
                            model_version=fitted.model_version,
                            max_rows=_feature_importance_permutation_sample_rows(
                                config.feature_importance
                            ),
                        )
                else:
                    importance_train_slice = pd.DataFrame(
                        columns=[*config.feature_columns, "label"]
                    )
                importance_request = _FeatureImportanceRequest(
                    pipeline=fitted.pipeline,
                    train_slice=importance_train_slice,
                    feature_columns=config.feature_columns,
                    model_family=fitted.model_family,
                    model_version=fitted.model_version,
                    fit_date=score_date,
                    trained_until=fitted.train_end,
                )
                if config.feature_importance.mode == "every_fit":
                    append_feature_importance(importance_request)
                elif config.feature_importance.mode == "latest_only":
                    latest_importance_request = importance_request
            del train_labeled, train_row_selection
        elif current_bundle is None:
            status = "skipped"
            skip_reason = "model_not_ready"

        if current_bundle is not None and status != "skipped":
            if (
                pending_prediction_bundle is not None
                and pending_prediction_bundle.model_version != current_bundle.model_version
            ):
                flush_pending_predictions()
            pending_prediction_bundle = current_bundle
            pending_prediction_indices.append(score_row_indices)
            pending_prediction_dates.append(score_date)
            pending_prediction_statuses.append(status)
        elif skip_reason is None and current_bundle is None:
            skip_reason = "model_not_ready"
        if status == "skipped" and skip_reason:
            skipped_score_dates_by_reason.update([skip_reason])
            samples = skipped_score_date_samples.setdefault(skip_reason, [])
            if len(samples) < 5:
                samples.append(score_date.date().isoformat())
        if total_score_dates:
            status_label = {
                "fit_scored": "重新训练并打分",
                "reused_scored": "复用模型打分",
                "skipped": f"跳过：{skip_reason or 'unknown'}",
            }.get(status, status)
            _emit_progress(
                (
                    "训练模型生成因子："
                    f"第 {score_step}/{total_score_dates} 个评分日 "
                    f"{score_date.date().isoformat()} 已完成（{status_label}）"
                ),
                min(round(score_step * 99 / total_score_dates), 99),
            )

        training_log_rows.append(
            {
                "score_date": score_date,
                "status": status,
                "skip_reason": skip_reason,
                "model_version": (
                    current_bundle.model_version if current_bundle is not None else None
                ),
                "trained_date_start": (
                    current_bundle.train_start if current_bundle is not None else None
                ),
                "trained_date_end": (
                    current_bundle.train_end if current_bundle is not None else None
                ),
                "n_train_dates": n_train_dates,
                "n_train_rows": n_train_rows,
                "n_score_assets": n_score_assets,
                "model_family": (
                    current_bundle.model_family
                    if current_bundle is not None
                    else config.model.family
                ),
                "scale_mode": current_bundle.scale_mode if current_bundle is not None else "N/A",
                "selection_status": selection_status if should_fit else "reused",
                "selection_metric": selection_metric,
                "selected_candidate_id": (
                    selected_candidate_id
                    if should_fit
                    else (
                        current_bundle.selected_candidate_id if current_bundle is not None else None
                    )
                ),
                "selected_candidate_score": (
                    selected_candidate_score
                    if should_fit
                    else (current_bundle.selection_score if current_bundle is not None else None)
                ),
                "selected_candidate_turnover": (
                    selected_candidate_turnover
                    if should_fit
                    else (
                        current_bundle.selected_candidate_turnover
                        if current_bundle is not None
                        else None
                    )
                ),
            }
        )

    flush_pending_predictions()
    if skipped_score_dates_by_reason:
        diagnostics.event(
            level="warning",
            stage="predict",
            message="score dates skipped",
            payload={
                "skip_reason_counts": dict(skipped_score_dates_by_reason),
                "score_date_samples_by_reason": skipped_score_date_samples,
            },
        )
    if (
        _feature_importance_enabled(config.feature_importance)
        and config.feature_importance.mode == "latest_only"
        and latest_importance_request is not None
    ):
        _emit_progress("训练模型生成因子：正在计算最后一个模型版本的特征重要性", 99)
        append_feature_importance(latest_importance_request)
    if total_score_dates:
        _emit_progress(f"训练模型生成因子：全部 {total_score_dates} 个评分日已完成", 100)
    prepared_arrays = None
    release_unused_memory()

    factor_df = (
        pd.concat(factor_frames, ignore_index=True)
        if factor_frames
        else pd.DataFrame(columns=["date", "asset", "factor", "value"])
    )
    if factor_df.empty:
        raise ValueError("model factor build produced no scored rows")
    factor_df = factor_df.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
    validate_factor_output(factor_df)
    factor_frames.clear()
    features = pd.DataFrame()
    release_unused_memory()

    training_log_df = (
        pd.DataFrame(
            training_log_rows,
            columns=[
                "score_date",
                "status",
                "skip_reason",
                "model_version",
                "trained_date_start",
                "trained_date_end",
                "n_train_dates",
                "n_train_rows",
                "n_score_assets",
                "model_family",
                "scale_mode",
                "selection_status",
                "selection_metric",
                "selected_candidate_id",
                "selected_candidate_score",
                "selected_candidate_turnover",
            ],
        )
        .sort_values("score_date", kind="mergesort")
        .reset_index(drop=True)
    )

    model_selection_df = (
        pd.DataFrame(model_selection_rows)
        if model_selection_rows
        else pd.DataFrame(
            columns=[
                "score_date",
                "model_version",
                "candidate_id",
                "candidate_family",
                "candidate_params",
                "selection_metric",
                "selection_score",
                "mean_ic",
                "mean_rank_ic",
                "n_splits_used",
                "fold_metrics_available",
                "selected",
                "selection_status",
            ]
        )
    )
    if not model_selection_df.empty:
        model_selection_df = model_selection_df.sort_values(
            ["score_date", "candidate_id"],
            kind="mergesort",
        ).reset_index(drop=True)
    training_metrics_df = _build_training_metrics_frame(
        training_metrics_rows=training_metrics_rows,
        oos_metrics_rows=oos_metrics_rows,
    )
    feature_oos_ic_df = _build_feature_oos_ic_frame(feature_oos_ic_rows)
    feature_importance_df = _combine_feature_importance_frames(
        per_fit_importance_frames,
        feature_columns=config.feature_columns,
        disabled=not _feature_importance_enabled(config.feature_importance),
    )
    feature_importance_ledger_df = _combine_feature_importance_ledger_frames(
        per_fit_importance_frames,
        save_ledger=(
            _feature_importance_enabled(config.feature_importance)
            and config.feature_importance.save_ledger
        ),
    )
    model_diagnostics = _build_model_diagnostics(
        config=config,
        training_log_df=training_log_df,
        training_metrics_df=training_metrics_df,
        feature_importance_df=feature_importance_df,
        feature_oos_ic_df=feature_oos_ic_df,
        model_selection_df=model_selection_df,
        label_winsorize_zscore=config.label_winsorize_zscore,
        label_winsor_clipped_rows=winsor_clip_count,
    )

    return ModelFactorBuildResult(
        factor_df=factor_df,
        training_log_df=training_log_df,
        training_metrics_df=training_metrics_df,
        feature_importance_df=feature_importance_df,
        feature_importance_ledger_df=feature_importance_ledger_df,
        coverage_base_df=coverage_base_df,
        feature_oos_ic_df=feature_oos_ic_df,
        forward_label_df=forward_label_df,
        model_selection_df=model_selection_df,
        model_diagnostics=model_diagnostics,
        integrity_checks=tuple(integrity_checks),
        target_diagnostics=dict(target_diagnostics),
    )
