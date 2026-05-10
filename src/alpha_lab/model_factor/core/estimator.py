from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Cross-module imports (auto-added)
from .config import ModelFactorBuildConfig, ModelSpec
from .internals import _FittedModelBundle, _PreparedModelArrays
from .training_arrays import _row_selection_length, _training_matrix_from_selection
from .types import _TREE_MODEL_FAMILIES, ModelFamily, ScaleFeatures, _RowSelection


def _fit_model_bundle(
    *,
    train_slice: pd.DataFrame,
    config: ModelFactorBuildConfig | None,
    model_version: int,
    model_spec: ModelSpec,
    selected_candidate_id: str | None = None,
    selection_score: float | None = None,
    selected_candidate_turnover: float | None = None,
    feature_columns_override: tuple[str, ...] | None = None,
) -> _FittedModelBundle:
    if config is None and feature_columns_override is None:
        raise ValueError("feature_columns_override is required when config is None")
    feature_columns = list(feature_columns_override or (config.feature_columns if config else ()))
    scale_features = config.feature_preprocess.scale_features if config is not None else "auto"
    pipeline, scale_mode = _build_model_pipeline(
        model_spec=model_spec,
        scale_features=scale_features,
    )
    train_features = _prepare_training_matrix(
        train_slice,
        feature_columns=tuple(feature_columns),
        model_family=model_spec.family,
    )
    pipeline.fit(train_features, train_slice["label"])
    train_dates = pd.Index(train_slice["date"].drop_duplicates()).sort_values()
    return _FittedModelBundle(
        pipeline=pipeline,
        model_version=model_version,
        train_start=pd.Timestamp(train_dates.min()),
        train_end=pd.Timestamp(train_dates.max()),
        n_train_dates=int(len(train_dates)),
        n_train_rows=int(len(train_slice)),
        scale_mode=scale_mode,
        model_family=model_spec.family,
        model_params=resolve_model_spec_params(model_spec),
        selected_candidate_id=selected_candidate_id,
        selection_score=selection_score,
        selected_candidate_turnover=selected_candidate_turnover,
    )


def _fit_model_bundle_from_arrays(
    *,
    prepared_arrays: _PreparedModelArrays,
    row_selection: _RowSelection,
    config: ModelFactorBuildConfig,
    model_version: int,
    model_spec: ModelSpec,
    selected_candidate_id: str | None = None,
    selection_score: float | None = None,
    selected_candidate_turnover: float | None = None,
) -> _FittedModelBundle:
    pipeline, scale_mode = _build_model_pipeline(
        model_spec=model_spec,
        scale_features=config.feature_preprocess.scale_features,
    )
    feature_values = (
        prepared_arrays.training_feature_values
        if prepared_arrays.training_feature_values is not None
        else prepared_arrays.feature_values
    )
    labels_array = (
        prepared_arrays.training_labels
        if prepared_arrays.training_labels is not None
        else prepared_arrays.labels
    )
    dates_array = (
        prepared_arrays.training_dates
        if prepared_arrays.training_dates is not None
        else prepared_arrays.dates
    )
    train_features = _training_matrix_from_selection(
        feature_values,
        row_selection,
    )
    train_labels = labels_array[row_selection]
    pipeline.fit(train_features, train_labels)
    train_dates = pd.DatetimeIndex(pd.to_datetime(dates_array[row_selection]))
    unique_train_dates = train_dates.unique().sort_values()
    return _FittedModelBundle(
        pipeline=pipeline,
        model_version=model_version,
        train_start=pd.Timestamp(unique_train_dates.min()),
        train_end=pd.Timestamp(unique_train_dates.max()),
        n_train_dates=int(len(unique_train_dates)),
        n_train_rows=_row_selection_length(
            row_selection,
            n_rows=len(labels_array),
        ),
        scale_mode=scale_mode,
        model_family=model_spec.family,
        model_params=resolve_model_spec_params(model_spec),
        selected_candidate_id=selected_candidate_id,
        selection_score=selection_score,
        selected_candidate_turnover=selected_candidate_turnover,
    )


def _build_model_pipeline(
    *,
    model_spec: ModelSpec,
    scale_features: ScaleFeatures,
) -> tuple[Pipeline, str]:
    estimator = _build_estimator(model_spec)
    scale_mode = _resolve_scale_mode(scale_features, model_spec.family)
    steps: list[tuple[str, object]] = [
        (
            "imputer",
            SimpleImputer(strategy="median", keep_empty_features=True, copy=False),
        )
    ]
    if scale_mode == "standard":
        steps.append(("scaler", StandardScaler(copy=False)))
    steps.append(("model", estimator))
    return Pipeline(steps=steps), scale_mode


def _build_estimator(spec: ModelSpec) -> object:
    params = resolve_model_spec_params(spec)
    family = spec.family
    if family == "linear":
        return LinearRegression(**params)
    if family == "ridge":
        return Ridge(**params)
    if family == "lasso":
        return Lasso(**params)
    if family == "elastic_net":
        return ElasticNet(**params)
    if family == "gbdt":
        return HistGradientBoostingRegressor(**params)
    if family == "xgboost":
        try:
            from xgboost import XGBRegressor  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "model.family='xgboost' requires optional dependency 'xgboost'. "
                "Install it with `uv add xgboost` or `pip install xgboost`."
            ) from exc
        return XGBRegressor(**params)
    if family == "lightgbm":
        try:
            from lightgbm import LGBMRegressor  # type: ignore[import-not-found]
        except Exception as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "model.family='lightgbm' requires optional dependency 'lightgbm'. "
                "Install it with `uv add lightgbm` or `pip install lightgbm`."
            ) from exc
        return LGBMRegressor(**cast(dict[str, Any], params))
    return MLPRegressor(**params)


def _training_dtype_for_family(model_family: str) -> np.dtype:
    if model_family in _TREE_MODEL_FAMILIES:
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def _prepare_training_matrix(
    frame: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    model_family: str,
) -> pd.DataFrame:
    matrix = frame.loc[:, list(feature_columns)]
    target_dtype = _training_dtype_for_family(model_family)
    if all(dtype == target_dtype for dtype in matrix.dtypes):
        return matrix
    return matrix.astype(target_dtype)


def resolve_model_spec_params(spec: ModelSpec) -> dict[str, object]:
    """Return effective estimator params after alpha-lab defaults are applied."""

    # Finance-aware defaults: applied via setdefault so user-provided params win.
    # Rationale: A-share daily cross-sections have ~5k assets; the sklearn defaults
    # (min_samples_leaf=20, num_leaves=31, MLP without early stopping, etc.) overfit
    # extreme single-day returns. We pin tighter regularization here, the same way
    # standardize/winsorize defaults harden the feature side.
    params = dict(spec.params)
    family = spec.family
    if family in {"linear", "ridge", "lasso", "elastic_net"}:
        params.setdefault("copy_X", False)
    if family == "lasso":
        params.setdefault("random_state", 0)
        params.setdefault("max_iter", 5000)
        return params
    if family == "elastic_net":
        params.setdefault("random_state", 0)
        params.setdefault("max_iter", 5000)
        return params
    if family == "gbdt":
        params.setdefault("random_state", 0)
        params.setdefault("loss", "absolute_error")
        params.setdefault("min_samples_leaf", 200)
        params.setdefault("l2_regularization", 1.0)
        return params
    if family == "xgboost":
        params.setdefault("random_state", 0)
        params.setdefault("min_child_weight", 5)
        params.setdefault("max_depth", 6)
        params.setdefault("subsample", 0.8)
        params.setdefault("colsample_bytree", 0.8)
        params.setdefault("reg_lambda", 1.0)
        params.setdefault("tree_method", "hist")
        return params
    if family == "lightgbm":
        params.setdefault("random_state", 0)
        params.setdefault("min_data_in_leaf", 200)
        params.setdefault("num_leaves", 63)
        params.setdefault("subsample", 0.8)
        params.setdefault("colsample_bytree", 0.8)
        params.setdefault("reg_lambda", 1.0)
        return params
    if family == "mlp":
        params.setdefault("random_state", 0)
        params.setdefault("max_iter", 500)
        params.setdefault("early_stopping", True)
        params.setdefault("validation_fraction", 0.15)
        params.setdefault("n_iter_no_change", 10)
        params.setdefault("alpha", 1e-3)
        return params
    return params


def _resolve_scale_mode(scale_features: ScaleFeatures, model_family: ModelFamily) -> str:
    if scale_features == "standard":
        return "standard"
    if scale_features == "none":
        return "none"
    if model_family in {"linear", "ridge", "lasso", "elastic_net", "mlp"}:
        return "standard"
    return "none"
