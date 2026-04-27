from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, cast, get_args

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from alpha_lab.interfaces import validate_factor_output
from alpha_lab.labels import forward_return
from alpha_lab.model_factor.diagnostics import (
    ModelFactorDiagnosticsObserver,
    compute_data_health_snapshot,
    derive_data_health_warnings,
    diagnostics_observer_or_null,
)
from alpha_lab.research_contracts import validate_prices_table
from alpha_lab.research_integrity.contracts import IntegrityCheckResult
from alpha_lab.research_integrity.leakage_checks import check_no_future_dates_in_input
from alpha_lab.validation.purged_kfold import purged_kfold_split

ModelFamily = Literal[
    "linear",
    "ridge",
    "lasso",
    "elastic_net",
    "gbdt",
    "xgboost",
    "lightgbm",
    "mlp",
]

_PERMUTATION_IMPORTANCE_MAX_ROWS = 1024
_PERMUTATION_IMPORTANCE_RANDOM_SEED = 20260423
_TREE_MODEL_FAMILIES: frozenset[str] = frozenset({"gbdt", "xgboost", "lightgbm"})
_MODEL_FAMILY_IMPORTANCE_EXTRACTORS: dict[str, tuple[str, ...]] = {
    "linear": ("coef", "permutation"),
    "ridge": ("coef", "permutation"),
    "lasso": ("coef", "permutation"),
    "elastic_net": ("coef", "permutation"),
    "gbdt": ("permutation",),
    "xgboost": ("feature_importances", "permutation"),
    "lightgbm": ("feature_importances", "permutation"),
    "mlp": ("permutation",),
}

MissingPolicy = Literal["median_impute"]
ScaleFeatures = Literal["auto", "standard", "none"]
WindowType = Literal["rolling", "expanding"]
ModelSelectionMetric = Literal[
    "rank_ic",
    "ic",
    "rank_ic_minus_turnover_penalty",
    "ic_minus_turnover_penalty",
]
FeatureImportanceMode = Literal["disabled", "latest_only", "every_fit"]
CrossSectionalTransform = Literal[
    "none",
    "zscore",
    "rank",
    "winsorize_zscore",
]
CrossSectionalGroupScope = Literal["date", "date_and_industry"]

_RESERVED_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "date",
        "asset",
        "factor",
        "value",
        "label",
        "target",
        "forward_return",
    }
)


def list_model_contracts() -> dict[str, object]:
    """Return model-factor contract literals for explorer/CLI consumers."""

    return {
        "supported_model_families": list(get_args(ModelFamily)),
        "supported_feature_preprocess": {
            "missing_policy": list(get_args(MissingPolicy)),
            "scale_features": list(get_args(ScaleFeatures)),
            "cross_sectional_transform": list(get_args(CrossSectionalTransform)),
            "cross_sectional_group_scope": list(get_args(CrossSectionalGroupScope)),
        },
        "supported_training": {
            "window_type": list(get_args(WindowType)),
        },
        "supported_selection_metrics": list(get_args(ModelSelectionMetric)),
        "supported_feature_importance": {
            "mode": list(get_args(FeatureImportanceMode)),
            "default_mode": FeatureImportanceConfig().mode,
            "default_permutation_max_rows": _PERMUTATION_IMPORTANCE_MAX_ROWS,
        },
    }


@dataclass(frozen=True)
class FeaturePreprocessConfig:
    """Training-time feature preprocessing controls."""

    missing_policy: MissingPolicy = "median_impute"
    scale_features: ScaleFeatures = "auto"
    cross_sectional_transform: CrossSectionalTransform = "winsorize_zscore"
    cross_sectional_group_scope: CrossSectionalGroupScope = "date"
    industry_group_column: str | None = None

    def __post_init__(self) -> None:
        if self.missing_policy != "median_impute":
            raise ValueError(
                "feature_preprocess.missing_policy currently supports only 'median_impute'"
            )
        if self.scale_features not in {"auto", "standard", "none"}:
            raise ValueError(
                "feature_preprocess.scale_features must be one of ['auto', 'standard', 'none']"
            )
        if self.cross_sectional_transform not in {
            "none",
            "zscore",
            "rank",
            "winsorize_zscore",
        }:
            raise ValueError(
                "feature_preprocess.cross_sectional_transform must be one of "
                "['none', 'zscore', 'rank', 'winsorize_zscore']"
            )
        if self.cross_sectional_group_scope not in {"date", "date_and_industry"}:
            raise ValueError(
                "feature_preprocess.cross_sectional_group_scope must be one of "
                "['date', 'date_and_industry']"
            )
        if self.cross_sectional_group_scope == "date_and_industry":
            if self.industry_group_column is None or not str(self.industry_group_column).strip():
                raise ValueError(
                    "feature_preprocess.industry_group_column is required when "
                    "cross_sectional_group_scope='date_and_industry'"
                )
            object.__setattr__(
                self,
                "industry_group_column",
                str(self.industry_group_column).strip(),
            )
        if self.industry_group_column is not None and not str(self.industry_group_column).strip():
            raise ValueError("feature_preprocess.industry_group_column must be a non-empty string")


@dataclass(frozen=True)
class ModelSpec:
    """Estimator family and fixed hyperparameters."""

    family: ModelFamily = "ridge"
    params: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in {
            "linear",
            "ridge",
            "lasso",
            "elastic_net",
            "gbdt",
            "xgboost",
            "lightgbm",
            "mlp",
        }:
            raise ValueError(
                "model.family must be one of ['linear', 'ridge', 'lasso', "
                "'elastic_net', 'gbdt', 'xgboost', 'lightgbm', 'mlp']"
            )
        if not isinstance(self.params, dict):
            raise ValueError("model.params must be an object")
        for key in self.params:
            if not isinstance(key, str) or not key.strip():
                raise ValueError("model.params keys must be non-empty strings")


@dataclass(frozen=True)
class ModelSelectionSpec:
    """Inner-loop model selection controls within each outer walk-forward fit."""

    enabled: bool = False
    n_splits: int = 4
    embargo_pct: float = 0.05
    metric: ModelSelectionMetric = "rank_ic"
    turnover_penalty_lambda: float = 0.0
    turnover_bucket_quantile: float = 0.2
    candidates: tuple[ModelSpec, ...] = ()

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("model_selection.n_splits must be >= 2")
        if self.embargo_pct < 0.0 or self.embargo_pct >= 1.0:
            raise ValueError("model_selection.embargo_pct must be in [0, 1)")
        if self.metric not in {
            "rank_ic",
            "ic",
            "rank_ic_minus_turnover_penalty",
            "ic_minus_turnover_penalty",
        }:
            raise ValueError(
                "model_selection.metric must be one of ["
                "'rank_ic', 'ic', 'rank_ic_minus_turnover_penalty', "
                "'ic_minus_turnover_penalty']"
            )
        if not isinstance(self.turnover_penalty_lambda, (int, float)):
            raise ValueError("model_selection.turnover_penalty_lambda must be numeric")
        if not pd.notna(float(self.turnover_penalty_lambda)) or self.turnover_penalty_lambda < 0:
            raise ValueError(
                "model_selection.turnover_penalty_lambda must be >= 0 and finite"
            )
        if not isinstance(self.turnover_bucket_quantile, (int, float)):
            raise ValueError("model_selection.turnover_bucket_quantile must be numeric")
        if not 0.0 < float(self.turnover_bucket_quantile) <= 0.5:
            raise ValueError(
                "model_selection.turnover_bucket_quantile must be in (0, 0.5]"
            )
        if not isinstance(self.candidates, tuple):
            raise ValueError("model_selection.candidates must be a tuple of model specs")
        for idx, candidate in enumerate(self.candidates):
            if not isinstance(candidate, ModelSpec):
                raise ValueError(
                    f"model_selection.candidates[{idx}] must be a ModelSpec instance"
                )


@dataclass(frozen=True)
class TrainingSpec:
    """Walk-forward training schedule for model-generated factor scores."""

    window_type: WindowType = "rolling"
    train_window_n_dates: int | None = 60
    min_train_dates: int = 40
    min_train_rows: int = 200
    retrain_every_n_dates: int = 1
    min_score_assets: int = 5

    def __post_init__(self) -> None:
        if self.window_type not in {"rolling", "expanding"}:
            raise ValueError("training.window_type must be one of ['rolling', 'expanding']")
        if self.window_type == "rolling" and (
            self.train_window_n_dates is None or self.train_window_n_dates <= 0
        ):
            raise ValueError(
                "training.train_window_n_dates must be > 0 when training.window_type='rolling'"
            )
        if self.train_window_n_dates is not None and self.train_window_n_dates <= 0:
            raise ValueError("training.train_window_n_dates must be > 0 when provided")
        if self.min_train_dates <= 0:
            raise ValueError("training.min_train_dates must be > 0")
        if self.min_train_rows <= 0:
            raise ValueError("training.min_train_rows must be > 0")
        if self.retrain_every_n_dates <= 0:
            raise ValueError("training.retrain_every_n_dates must be > 0")
        if self.min_score_assets <= 0:
            raise ValueError("training.min_score_assets must be > 0")


@dataclass(frozen=True)
class FeatureImportanceConfig:
    """Controls for model feature importance diagnostics."""

    mode: FeatureImportanceMode = "latest_only"
    permutation_max_rows: int = _PERMUTATION_IMPORTANCE_MAX_ROWS

    def __post_init__(self) -> None:
        if self.mode not in {"disabled", "latest_only", "every_fit"}:
            raise ValueError(
                "feature_importance.mode must be one of "
                "['disabled', 'latest_only', 'every_fit']"
            )
        if isinstance(self.permutation_max_rows, bool) or not isinstance(
            self.permutation_max_rows,
            int,
        ):
            raise ValueError("feature_importance.permutation_max_rows must be an integer")
        if self.permutation_max_rows <= 0:
            raise ValueError("feature_importance.permutation_max_rows must be > 0")


@dataclass(frozen=True)
class ModelFactorBuildConfig:
    """Configuration for converting research features into a factor table."""

    factor_name: str
    feature_columns: tuple[str, ...]
    target_horizon: int
    feature_preprocess: FeaturePreprocessConfig = FeaturePreprocessConfig()
    model: ModelSpec = ModelSpec()
    model_selection: ModelSelectionSpec = ModelSelectionSpec()
    training: TrainingSpec = TrainingSpec()
    feature_importance: FeatureImportanceConfig = FeatureImportanceConfig()
    known_at_col: str | None = None
    label_winsorize_zscore: float | None = None

    def __post_init__(self) -> None:
        if not self.factor_name.strip():
            raise ValueError("factor_name must be non-empty")
        if not self.feature_columns:
            raise ValueError("feature_columns must be a non-empty tuple")
        seen: set[str] = set()
        for column in self.feature_columns:
            if not isinstance(column, str) or not column.strip():
                raise ValueError("feature_columns must contain non-empty strings")
            if column in _RESERVED_FEATURE_COLUMNS:
                raise ValueError(f"feature_columns may not include reserved column {column!r}")
            if column in seen:
                raise ValueError(f"feature_columns must be unique; duplicate: {column!r}")
            seen.add(column)
        if self.target_horizon <= 0:
            raise ValueError("target_horizon must be > 0")
        if self.known_at_col is not None and not self.known_at_col.strip():
            raise ValueError("known_at_col must be non-empty when provided")
        if self.label_winsorize_zscore is not None and self.label_winsorize_zscore <= 0:
            raise ValueError("label_winsorize_zscore must be > 0 when provided")


@dataclass(frozen=True)
class ModelFactorBuildResult:
    """Canonical factor output plus training diagnostics."""

    factor_df: pd.DataFrame
    training_log_df: pd.DataFrame
    feature_importance_df: pd.DataFrame
    model_diagnostics: dict[str, object]
    forward_label_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_selection_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    integrity_checks: tuple[IntegrityCheckResult, ...] = ()


@dataclass(frozen=True)
class _FittedModelBundle:
    pipeline: Pipeline
    model_version: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    n_train_dates: int
    n_train_rows: int
    scale_mode: str
    model_family: str
    model_params: dict[str, object]
    selected_candidate_id: str | None = None
    selection_score: float | None = None
    selected_candidate_turnover: float | None = None


@dataclass(frozen=True)
class _FeatureImportanceRequest:
    pipeline: Pipeline
    train_slice: pd.DataFrame
    feature_columns: tuple[str, ...]
    model_family: str
    model_version: int
    trained_until: pd.Timestamp


@dataclass(frozen=True)
class _ModelSelectionOutcome:
    selected_model: ModelSpec
    selected_candidate_id: str
    selected_score: float | None
    selected_turnover: float | None
    status: str
    n_splits_used: int
    rows: list[dict[str, object]]


@dataclass(frozen=True)
class _DateIndexedRows:
    ordered_dates: pd.DatetimeIndex
    row_indices_by_pos: tuple[np.ndarray, ...]


@dataclass
class _TrainingWindowCache:
    row_indices_by_date_pos: tuple[np.ndarray, ...]
    labeled_row_indices_by_date_pos: tuple[np.ndarray, ...]
    window_start_by_pos: np.ndarray
    eligible_end_by_pos: np.ndarray
    n_train_dates_by_pos: np.ndarray
    n_labeled_rows_by_pos: np.ndarray
    _labeled_window_rows: dict[int, np.ndarray] = field(default_factory=dict)

    def labeled_row_indices(self, score_idx: int) -> np.ndarray:
        cached = self._labeled_window_rows.get(int(score_idx))
        if cached is not None:
            return cached

        window_start = int(self.window_start_by_pos[score_idx])
        eligible_end = int(self.eligible_end_by_pos[score_idx])
        if window_start >= eligible_end:
            row_idx = np.empty(0, dtype=np.intp)
        else:
            selected = self.labeled_row_indices_by_date_pos[window_start:eligible_end]
            non_empty = [indices for indices in selected if len(indices) > 0]
            row_idx = (
                np.concatenate(non_empty).astype(np.intp, copy=False)
                if non_empty
                else np.empty(0, dtype=np.intp)
            )
        self._labeled_window_rows[int(score_idx)] = row_idx
        return row_idx

    def labeled_slice(self, merged: pd.DataFrame, score_idx: int) -> pd.DataFrame:
        row_idx = self.labeled_row_indices(score_idx)
        if len(row_idx) == 0:
            return merged.iloc[0:0].copy()
        return merged.take(row_idx).copy()


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
    with diagnostics.stage(
        "feature_validate",
        payload={"feature_count": len(config.feature_columns)},
    ) as feature_stage:
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
        label_df = forward_return(prices, horizon=config.target_horizon)
        label_name = f"forward_return_{config.target_horizon}"
        forward_label_df = (
            label_df[label_df["factor"] == label_name][["date", "asset", "factor", "value"]]
            .copy()
            .reset_index(drop=True)
        )
        labels = forward_label_df[["date", "asset", "value"]].rename(
            columns={"value": "label"}
        )
        winsor_z = config.label_winsorize_zscore
        winsor_clip_count = 0
        if winsor_z is not None and not labels.empty:
            labels, winsor_clip_count = _winsorize_labels_per_date(labels, z=float(winsor_z))
        data_health = compute_data_health_snapshot(
            features=features,
            labels=labels,
            feature_columns=config.feature_columns,
        )
        diagnostics.set_data_health(data_health)
        warnings_emitted = 0
        for warning in derive_data_health_warnings(data_health):
            diagnostics.warning(**warning)
            warnings_emitted += 1
        target_stage.attach(
            n_label_rows=int(len(labels)),
            n_forward_label_cache_rows=int(len(forward_label_df)),
            target_mean=data_health.get("target_mean"),
            target_std=data_health.get("target_std"),
            outlier_ratio=data_health.get("outlier_ratio"),
            coverage_ratio=data_health.get("coverage_ratio"),
            warnings_emitted=warnings_emitted,
            label_winsorize_zscore=winsor_z if winsor_z is not None else "none",
            label_winsor_clipped_rows=winsor_clip_count,
        )

    with diagnostics.stage("preprocess", payload={"step": "merge_labels"}) as merge_stage:
        merged = features.merge(labels, on=["date", "asset"], how="left", validate="one_to_one")
        merge_stage.attach(
            n_rows=int(len(merged)),
            n_labeled_rows=int(merged["label"].notna().sum()),
        )
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
        merged_date_index = _build_date_indexed_rows(merged, score_date_index)
        training_window_cache = _build_training_window_cache(
            date_index=merged_date_index,
            merged=merged,
            training=config.training,
            target_horizon=config.target_horizon,
        )
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
            row_index_cache_mode="lazy_materialize_fit_windows",
        )

    factor_rows: list[dict[str, object]] = []
    training_log_rows: list[dict[str, object]] = []
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
        score_slice = features.take(row_idx).copy()
        score_dates_text = [date.date().isoformat() for date in pending_prediction_dates]
        payload: dict[str, object] = {
            "score_date_start": score_dates_text[0],
            "score_date_end": score_dates_text[-1],
            "model_version": pending_prediction_bundle.model_version,
            "n_score_dates": len(pending_prediction_dates),
            "n_score_rows": int(len(score_slice)),
        }
        with diagnostics.stage("predict", payload=payload) as predict_stage:
            score_features = score_slice.loc[:, list(config.feature_columns)]
            predictions = pending_prediction_bundle.pipeline.predict(score_features)
            for date_value, asset, value in zip(
                score_slice["date"],
                score_slice["asset"],
                predictions,
                strict=True,
            ):
                factor_rows.append(
                    {
                        "date": pd.Timestamp(date_value),
                        "asset": str(asset),
                        "factor": config.factor_name,
                        "value": float(value),
                    }
                )
            predict_stage.attach(
                score_date_start=score_dates_text[0],
                score_date_end=score_dates_text[-1],
                score_date_first_values=score_dates_text[:3],
                score_date_last_values=score_dates_text[-3:],
                model_version=pending_prediction_bundle.model_version,
                n_score_dates=len(pending_prediction_dates),
                n_score_rows=int(len(score_slice)),
                statuses=sorted(set(pending_prediction_statuses)),
                status="batch_scored",
            )

        pending_prediction_bundle = None
        pending_prediction_indices.clear()
        pending_prediction_dates.clear()
        pending_prediction_statuses.clear()

    def append_feature_importance(request: _FeatureImportanceRequest) -> None:
        payload = {
            "model_version": request.model_version,
            "model_family": request.model_family,
            "mode": config.feature_importance.mode,
            "permutation_max_rows": config.feature_importance.permutation_max_rows,
            "n_train_rows": int(len(request.train_slice)),
        }
        with diagnostics.stage("feature_importance", payload=payload) as importance_stage:
            frame = _feature_importance_frame(
                request.pipeline,
                train_slice=request.train_slice,
                feature_columns=request.feature_columns,
                model_family=request.model_family,
                model_version=request.model_version,
                trained_until=request.trained_until,
                permutation_max_rows=config.feature_importance.permutation_max_rows,
            )
            per_fit_importance_frames.append(frame)
            importance_stage.attach(
                model_version=request.model_version,
                model_family=request.model_family,
                mode=config.feature_importance.mode,
                permutation_max_rows=config.feature_importance.permutation_max_rows,
                n_features=int(len(request.feature_columns)),
                n_train_rows=int(len(request.train_slice)),
                importance_sources=sorted(
                    {
                        str(value)
                        for value in frame["importance_source"].dropna().unique().tolist()
                    }
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
            train_labeled = training_window_cache.labeled_slice(merged, score_idx)
            model_version += 1
            selected_model = config.model
            n_selection_splits = 0
            if config.model_selection.enabled:
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
                fitted = _fit_model_bundle(
                    train_slice=train_labeled,
                    config=config,
                    model_version=model_version,
                    model_spec=selected_model,
                    selected_candidate_id=selected_candidate_id,
                    selection_score=selected_candidate_score,
                    selected_candidate_turnover=selected_candidate_turnover,
                )
                fit_stage.attach(
                    score_date=score_date.date().isoformat(),
                    model_version=fitted.model_version,
                    n_train_rows=fitted.n_train_rows,
                    n_train_dates=fitted.n_train_dates,
                    scale_mode=fitted.scale_mode,
                    model_family=fitted.model_family,
                    train_start=fitted.train_start.date().isoformat(),
                    train_end=fitted.train_end.date().isoformat(),
                    selection_status=selection_status,
                    selected_candidate_id=selected_candidate_id,
                    selected_candidate_score=selected_candidate_score,
                    selected_candidate_turnover=selected_candidate_turnover,
                    n_selection_splits=n_selection_splits,
                )
            current_bundle = fitted
            last_fit_score_idx = score_idx
            status = "fit_scored"
            importance_request = _FeatureImportanceRequest(
                pipeline=fitted.pipeline,
                train_slice=train_labeled,
                feature_columns=config.feature_columns,
                model_family=fitted.model_family,
                model_version=fitted.model_version,
                trained_until=fitted.train_end,
            )
            if config.feature_importance.mode == "every_fit":
                append_feature_importance(importance_request)
            elif config.feature_importance.mode == "latest_only":
                latest_importance_request = importance_request
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
                        current_bundle.selected_candidate_id
                        if current_bundle is not None
                        else None
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
        config.feature_importance.mode == "latest_only"
        and latest_importance_request is not None
    ):
        _emit_progress("训练模型生成因子：正在计算最后一个模型版本的特征重要性", 99)
        append_feature_importance(latest_importance_request)
    if total_score_dates:
        _emit_progress(f"训练模型生成因子：全部 {total_score_dates} 个评分日已完成", 100)

    factor_df = pd.DataFrame(factor_rows, columns=["date", "asset", "factor", "value"])
    if factor_df.empty:
        raise ValueError("model factor build produced no scored rows")
    factor_df = factor_df.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
    validate_factor_output(factor_df)

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
    feature_importance_df = _combine_feature_importance_frames(
        per_fit_importance_frames,
        feature_columns=config.feature_columns,
        disabled=config.feature_importance.mode == "disabled",
    )
    model_diagnostics = _build_model_diagnostics(
        config=config,
        training_log_df=training_log_df,
        feature_importance_df=feature_importance_df,
        model_selection_df=model_selection_df,
        label_winsorize_zscore=config.label_winsorize_zscore,
        label_winsor_clipped_rows=winsor_clip_count,
    )

    return ModelFactorBuildResult(
        factor_df=factor_df,
        training_log_df=training_log_df,
        feature_importance_df=feature_importance_df,
        forward_label_df=forward_label_df,
        model_selection_df=model_selection_df,
        model_diagnostics=model_diagnostics,
        integrity_checks=tuple(integrity_checks),
    )


def _normalize_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    prices = prices_df.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
    validate_prices_table(prices)
    return prices


def _apply_cross_sectional_transform(
    features: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    mode: CrossSectionalTransform,
    group_scope: CrossSectionalGroupScope,
    industry_group_column: str | None,
) -> pd.DataFrame:
    """Per-date feature transform. All modes operate strictly within one date, so
    they are PIT-safe even when applied to the full feature frame at once."""

    if mode == "none":
        return features
    cols = list(feature_columns)
    if not cols:
        return features
    frame = features.copy()
    if group_scope == "date_and_industry" and industry_group_column is not None:
        grouped = frame.groupby(["date", industry_group_column], sort=False)[cols]
    else:
        grouped = frame.groupby("date", sort=False)[cols]
    if mode == "zscore":
        frame[cols] = grouped.transform(_zscore_finite)
    elif mode == "rank":
        frame[cols] = grouped.transform(_rank_centered)
    elif mode == "winsorize_zscore":
        frame[cols] = grouped.transform(_winsorize_then_zscore)
    return frame


def _industry_group_temporal_profile(
    features: pd.DataFrame,
    *,
    industry_group_column: str,
) -> dict[str, object]:
    """Summarize whether an industry grouping column behaves like a static snapshot."""

    n_dates = int(features["date"].nunique()) if "date" in features.columns else 0
    n_assets_total = int(features["asset"].nunique()) if "asset" in features.columns else 0
    if n_dates < 2 or n_assets_total == 0:
        return {
            "n_dates": n_dates,
            "n_assets_total": n_assets_total,
            "n_assets_eligible": 0,
            "n_assets_static": 0,
            "static_ratio": None,
            "all_assets_static": False,
        }

    frame = features[["asset", "date", industry_group_column]].copy()
    frame[industry_group_column] = (
        frame[industry_group_column].astype(str).str.strip().replace({"": "__unknown__"})
    )
    date_counts = frame.groupby("asset", sort=False)["date"].nunique(dropna=True)
    eligible_assets = date_counts[date_counts >= 2].index
    n_assets_eligible = int(len(eligible_assets))
    if n_assets_eligible == 0:
        return {
            "n_dates": n_dates,
            "n_assets_total": n_assets_total,
            "n_assets_eligible": 0,
            "n_assets_static": 0,
            "static_ratio": None,
            "all_assets_static": False,
        }

    unique_counts = (
        frame[frame["asset"].isin(eligible_assets)]
        .groupby("asset", sort=False)[industry_group_column]
        .nunique(dropna=False)
    )
    n_assets_static = int((unique_counts <= 1).sum())
    static_ratio = float(n_assets_static / n_assets_eligible)
    return {
        "n_dates": n_dates,
        "n_assets_total": n_assets_total,
        "n_assets_eligible": n_assets_eligible,
        "n_assets_static": n_assets_static,
        "static_ratio": static_ratio,
        "all_assets_static": n_assets_static == n_assets_eligible,
    }


def _zscore_finite(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mu = float(values.mean()) if values.notna().any() else 0.0
    sd = float(values.std(ddof=0)) if values.notna().any() else 0.0
    if not np.isfinite(sd) or sd == 0.0:
        return values - mu
    return (values - mu) / sd


def _rank_centered(series: pd.Series) -> pd.Series:
    # Percentile rank on a per-date cross-section, then map to [-1, 1]. NaN stays NaN.
    values = pd.to_numeric(series, errors="coerce")
    ranks = values.rank(method="average", pct=True)
    return (ranks - 0.5) * 2.0


def _winsorize_labels_per_date(labels: pd.DataFrame, *, z: float) -> tuple[pd.DataFrame, int]:
    """Per-date symmetric winsorization of labels to ±z·std. Returns (frame, n_clipped)."""

    if labels.empty:
        return labels, 0
    frame = labels.copy()
    raw = pd.to_numeric(frame["label"], errors="coerce")
    grouped = raw.groupby(frame["date"], sort=False)
    mu = grouped.transform("mean")
    sd = grouped.transform(lambda s: float(s.std(ddof=0)) if s.notna().any() else 0.0)
    finite_sd = sd.where(np.isfinite(sd) & (sd > 0.0), other=0.0)
    upper = mu + float(z) * finite_sd
    lower = mu - float(z) * finite_sd
    clipped = raw.clip(lower=lower, upper=upper)
    # Where sd==0, clip is no-op (lower==upper==mu); count strictly modified rows.
    diff_mask = (raw.notna()) & ((raw != clipped) & raw.notna())
    n_clipped = int(diff_mask.sum())
    frame["label"] = clipped
    return frame, n_clipped


def _winsorize_then_zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if not values.notna().any():
        return values
    mu = float(values.mean())
    sd = float(values.std(ddof=0))
    if not np.isfinite(sd) or sd == 0.0:
        return values - mu
    clipped = values.clip(lower=mu - 3.0 * sd, upper=mu + 3.0 * sd)
    cmu = float(clipped.mean())
    csd = float(clipped.std(ddof=0))
    if not np.isfinite(csd) or csd == 0.0:
        return clipped - cmu
    return (clipped - cmu) / csd


def _normalize_features(
    features_df: pd.DataFrame,
    *,
    config: ModelFactorBuildConfig,
) -> pd.DataFrame:
    required = {"date", "asset", *config.feature_columns}
    if config.known_at_col is not None:
        required.add(config.known_at_col)
    if (
        config.feature_preprocess.cross_sectional_group_scope == "date_and_industry"
        and config.feature_preprocess.industry_group_column is not None
    ):
        required.add(config.feature_preprocess.industry_group_column)
    missing = required - set(features_df.columns)
    if missing:
        raise ValueError(f"features_df is missing required columns: {sorted(missing)}")
    if "factor" in features_df.columns or "value" in features_df.columns:
        raise ValueError(
            "features_df may not contain canonical signal columns 'factor'/'value'; "
            "provide a wide feature table instead"
        )

    selected_columns = ["date", "asset", *config.feature_columns]
    if config.known_at_col is not None:
        selected_columns.append(config.known_at_col)
    if (
        config.feature_preprocess.cross_sectional_group_scope == "date_and_industry"
        and config.feature_preprocess.industry_group_column is not None
    ):
        selected_columns.append(config.feature_preprocess.industry_group_column)
    frame = features_df.loc[:, selected_columns].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("features_df.date contains invalid timestamps")
    asset_values = frame["asset"].astype(str).str.strip()
    if frame["asset"].isna().any() or (asset_values == "").any():
        raise ValueError("features_df.asset contains null or empty values")
    frame["asset"] = asset_values.astype("category")
    if frame.duplicated(subset=["date", "asset"]).any():
        raise ValueError("features_df contains duplicate (date, asset) rows")
    for column in config.feature_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].notna().sum() == 0:
            raise ValueError(f"feature column {column!r} contains no numeric observations")
    if config.known_at_col is not None:
        frame[config.known_at_col] = pd.to_datetime(frame[config.known_at_col], errors="coerce")
        if frame[config.known_at_col].isna().any():
            raise ValueError(f"{config.known_at_col} contains invalid timestamps")
    if config.feature_preprocess.cross_sectional_group_scope == "date_and_industry":
        group_col = config.feature_preprocess.industry_group_column
        if group_col is not None:
            group_values = frame[group_col].astype(str).str.strip()
            group_values = group_values.replace({"nan": "__unknown__", "": "__unknown__"})
            frame[group_col] = group_values.astype("category")
    return frame.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def _build_date_indexed_rows(
    frame: pd.DataFrame,
    ordered_dates: pd.DatetimeIndex,
) -> _DateIndexedRows:
    grouped = {
        pd.Timestamp(date): np.asarray(indices, dtype=np.intp)
        for date, indices in frame.groupby("date", sort=False).indices.items()
    }
    row_indices_by_pos: list[np.ndarray] = []
    missing_dates: list[str] = []
    for raw_date in ordered_dates:
        date = pd.Timestamp(raw_date)
        indices = grouped.get(date)
        if indices is None:
            missing_dates.append(date.date().isoformat())
            continue
        row_indices_by_pos.append(indices)
    if missing_dates:
        preview = ", ".join(missing_dates[:5])
        raise ValueError(f"date-indexed frame is missing score dates: {preview}")
    return _DateIndexedRows(
        ordered_dates=pd.DatetimeIndex(ordered_dates),
        row_indices_by_pos=tuple(row_indices_by_pos),
    )


def _build_training_window_cache(
    *,
    date_index: _DateIndexedRows,
    merged: pd.DataFrame,
    training: TrainingSpec,
    target_horizon: int,
) -> _TrainingWindowCache:
    n_dates = int(len(date_index.row_indices_by_pos))
    labeled_by_pos = _build_labeled_date_indexed_rows(merged, date_index.ordered_dates)
    labeled_counts = np.asarray([len(indices) for indices in labeled_by_pos], dtype=np.intp)
    labeled_prefix = np.concatenate(
        [np.zeros(1, dtype=np.intp), np.cumsum(labeled_counts, dtype=np.intp)]
    )
    window_start_by_pos = np.zeros(n_dates, dtype=np.intp)
    eligible_end_by_pos = np.zeros(n_dates, dtype=np.intp)
    n_train_dates_by_pos = np.zeros(n_dates, dtype=np.intp)
    n_labeled_rows_by_pos = np.zeros(n_dates, dtype=np.intp)

    for score_idx in range(n_dates):
        window_start, eligible_end = _training_window_bounds(
            score_idx=score_idx,
            training=training,
            target_horizon=target_horizon,
        )
        window_start_by_pos[score_idx] = window_start
        eligible_end_by_pos[score_idx] = eligible_end
        n_train_dates_by_pos[score_idx] = max(eligible_end - window_start, 0)
        n_labeled_rows_by_pos[score_idx] = int(
            labeled_prefix[eligible_end] - labeled_prefix[window_start]
        )

    return _TrainingWindowCache(
        row_indices_by_date_pos=date_index.row_indices_by_pos,
        labeled_row_indices_by_date_pos=labeled_by_pos,
        window_start_by_pos=window_start_by_pos,
        eligible_end_by_pos=eligible_end_by_pos,
        n_train_dates_by_pos=n_train_dates_by_pos,
        n_labeled_rows_by_pos=n_labeled_rows_by_pos,
    )


def _build_labeled_date_indexed_rows(
    merged: pd.DataFrame,
    ordered_dates: pd.DatetimeIndex,
) -> tuple[np.ndarray, ...]:
    if "label" not in merged.columns:
        return tuple(np.empty(0, dtype=np.intp) for _ in ordered_dates)
    labeled = merged[merged["label"].notna()]
    grouped = {
        pd.Timestamp(date): np.asarray(indices, dtype=np.intp)
        for date, indices in labeled.groupby("date", sort=False).groups.items()
    }
    return tuple(
        grouped.get(pd.Timestamp(raw_date), np.empty(0, dtype=np.intp))
        for raw_date in ordered_dates
    )


def _training_window_bounds(
    *,
    score_idx: int,
    training: TrainingSpec,
    target_horizon: int,
) -> tuple[int, int]:
    horizon = max(int(target_horizon), 1)
    purge = horizon - 1
    eligible_end = int(score_idx) - purge
    if eligible_end <= 0:
        return 0, 0

    if training.window_type == "expanding":
        window_start = 0
    else:
        window_start = max(0, eligible_end - int(training.train_window_n_dates or 0))
    if window_start >= eligible_end:
        return 0, 0
    return window_start, eligible_end


def _training_slice_from_preindexed_dates(
    merged: pd.DataFrame,
    *,
    date_index: _DateIndexedRows,
    score_idx: int,
    training: TrainingSpec,
    target_horizon: int,
) -> pd.DataFrame:
    # Same purged walk-forward contract as _training_slice, but using date
    # positions instead of rescanning the full table for every score date.
    window_start, eligible_end = _training_window_bounds(
        score_idx=score_idx,
        training=training,
        target_horizon=target_horizon,
    )
    if window_start >= eligible_end:
        return merged.iloc[0:0].copy()

    selected_indices = date_index.row_indices_by_pos[window_start:eligible_end]
    if not selected_indices:
        return merged.iloc[0:0].copy()
    row_idx = np.concatenate(selected_indices)
    return merged.take(row_idx).copy()


def _training_slice(
    merged: pd.DataFrame,
    *,
    score_date: pd.Timestamp,
    training: TrainingSpec,
    target_horizon: int,
) -> pd.DataFrame:
    # Purged walk-forward: a sample at date t has label that resolves at t+h, so we
    # drop the most-recent (h-1) dates from the training pool. This guarantees every
    # label in the training slice was realized strictly before score_date.
    horizon = max(int(target_horizon), 1)
    all_dates = pd.Index(
        merged.loc[merged["date"] < score_date, "date"].drop_duplicates()
    ).sort_values()
    if len(all_dates) == 0:
        return merged.iloc[0:0].copy()
    purge = horizon - 1
    if purge > 0:
        eligible_dates = all_dates[:-purge] if len(all_dates) > purge else all_dates[:0]
    else:
        eligible_dates = all_dates
    if len(eligible_dates) == 0:
        return merged.iloc[0:0].copy()

    if training.window_type == "expanding":
        return merged[merged["date"].isin(eligible_dates)].copy()

    tail_dates = eligible_dates[-int(training.train_window_n_dates or 0) :]
    return merged[merged["date"].isin(tail_dates)].copy()


def _selection_candidates(config: ModelFactorBuildConfig) -> tuple[ModelSpec, ...]:
    if config.model_selection.candidates:
        return config.model_selection.candidates
    return (config.model,)


def _selection_has_mlp_early_stopping(config: ModelFactorBuildConfig) -> bool:
    return any(
        _mlp_early_stopping_enabled(candidate)
        for candidate in _selection_candidates(config)
    )


def _mlp_early_stopping_enabled(spec: ModelSpec) -> bool:
    if spec.family != "mlp":
        return False
    params = dict(spec.params)
    params.setdefault("early_stopping", True)
    return bool(params.get("early_stopping"))


def _select_model_candidate(
    *,
    train_slice: pd.DataFrame,
    config: ModelFactorBuildConfig,
    score_date: pd.Timestamp,
    model_version: int,
) -> _ModelSelectionOutcome:
    candidates = _selection_candidates(config)
    train_dates = pd.Index(train_slice["date"].drop_duplicates()).sort_values()
    n_splits_used = min(config.model_selection.n_splits, int(len(train_dates)))
    if n_splits_used < 2:
        fallback_id = _candidate_id(0, candidates[0])
        return _ModelSelectionOutcome(
            selected_model=candidates[0],
            selected_candidate_id=fallback_id,
            selected_score=None,
            selected_turnover=None,
            status="insufficient_dates",
            n_splits_used=0,
            rows=[
                {
                    "score_date": score_date,
                    "model_version": model_version,
                    "candidate_id": fallback_id,
                    "candidate_family": candidates[0].family,
                    "candidate_params": dict(candidates[0].params),
                    "selection_metric": config.model_selection.metric,
                    "selection_score": None,
                    "mean_ic": None,
                    "mean_rank_ic": None,
                    "mean_turnover": None,
                    "mean_cost_adjusted_score": None,
                    "n_splits_used": 0,
                    "fold_metrics_available": 0,
                    "selected": True,
                    "selection_status": "insufficient_dates",
                }
            ],
        )

    splits = purged_kfold_split(
        train_dates.to_numpy(),
        n_splits=n_splits_used,
        label_horizon=int(config.target_horizon),
        embargo_pct=float(config.model_selection.embargo_pct),
    )
    rows: list[dict[str, object]] = []
    best_model = candidates[0]
    best_candidate_id = _candidate_id(0, candidates[0])
    best_score = float("-inf")
    best_turnover: float | None = None
    found_finite_score = False

    for idx, candidate in enumerate(candidates):
        mean_ic, mean_rank_ic, mean_turnover, fold_metrics_available = _evaluate_model_candidate_cv(
            train_slice=train_slice,
            feature_columns=config.feature_columns,
            candidate=candidate,
            splits=splits,
            train_dates=train_dates,
            turnover_bucket_quantile=float(config.model_selection.turnover_bucket_quantile),
        )
        selection_score = _selection_score(
            metric=config.model_selection.metric,
            mean_ic=mean_ic,
            mean_rank_ic=mean_rank_ic,
            mean_turnover=mean_turnover,
            turnover_penalty=float(config.model_selection.turnover_penalty_lambda),
        )
        candidate_id = _candidate_id(idx, candidate)
        score_value = float(selection_score) if selection_score is not None else float("-inf")
        if np.isfinite(score_value):
            if (not found_finite_score) or score_value > best_score:
                best_model = candidate
                best_candidate_id = candidate_id
                best_score = score_value
                best_turnover = mean_turnover
            found_finite_score = True
        rows.append(
            {
                "score_date": score_date,
                "model_version": model_version,
                "candidate_id": candidate_id,
                "candidate_family": candidate.family,
                "candidate_params": dict(candidate.params),
                "selection_metric": config.model_selection.metric,
                "selection_score": selection_score,
                "mean_ic": mean_ic,
                "mean_rank_ic": mean_rank_ic,
                "mean_turnover": mean_turnover,
                "mean_cost_adjusted_score": selection_score,
                "n_splits_used": n_splits_used,
                "fold_metrics_available": fold_metrics_available,
                "selected": False,
                "selection_status": "ok" if found_finite_score else "no_finite_scores",
            }
        )

    if not found_finite_score:
        best_model = candidates[0]
        best_candidate_id = _candidate_id(0, candidates[0])
        best_score = float("-inf")

    for row in rows:
        row["selected"] = str(row["candidate_id"]) == best_candidate_id
        row["selection_status"] = "ok" if found_finite_score else "no_finite_scores"

    return _ModelSelectionOutcome(
        selected_model=best_model,
        selected_candidate_id=best_candidate_id,
        selected_score=_finite_or_none(best_score),
        selected_turnover=_finite_or_none(best_turnover),
        status="ok" if found_finite_score else "no_finite_scores",
        n_splits_used=n_splits_used,
        rows=rows,
    )


def _evaluate_model_candidate_cv(
    *,
    train_slice: pd.DataFrame,
    feature_columns: tuple[str, ...],
    candidate: ModelSpec,
    splits: list[dict[str, np.ndarray]],
    train_dates: pd.Index,
    turnover_bucket_quantile: float,
) -> tuple[float | None, float | None, float | None, int]:
    ic_values: list[float] = []
    rank_ic_values: list[float] = []
    turnover_values: list[float] = []
    for masks in splits:
        fold_train_dates = train_dates[masks["train"]]
        fold_test_dates = train_dates[masks["test"]]
        if len(fold_train_dates) == 0 or len(fold_test_dates) == 0:
            continue
        fold_train = train_slice[train_slice["date"].isin(fold_train_dates)].copy()
        fold_test = train_slice[train_slice["date"].isin(fold_test_dates)].copy()
        if fold_train.empty or fold_test.empty:
            continue
        fitted = _fit_model_bundle(
            train_slice=fold_train,
            config=None,
            model_version=0,
            model_spec=candidate,
            feature_columns_override=feature_columns,
        )
        predictions = fitted.pipeline.predict(fold_test.loc[:, list(feature_columns)])
        fold_mean_ic, fold_mean_rank_ic, fold_turnover = _score_prediction_cross_sections(
            fold_test=fold_test,
            predictions=np.asarray(predictions, dtype=float),
            turnover_bucket_quantile=turnover_bucket_quantile,
        )
        if fold_mean_ic is not None:
            ic_values.append(float(fold_mean_ic))
        if fold_mean_rank_ic is not None:
            rank_ic_values.append(float(fold_mean_rank_ic))
        if fold_turnover is not None:
            turnover_values.append(float(fold_turnover))
    return (
        _finite_or_none(float(np.mean(ic_values))) if ic_values else None,
        _finite_or_none(float(np.mean(rank_ic_values))) if rank_ic_values else None,
        _finite_or_none(float(np.mean(turnover_values))) if turnover_values else None,
        max(len(ic_values), len(rank_ic_values), len(turnover_values)),
    )


def _score_prediction_cross_sections(
    *,
    fold_test: pd.DataFrame,
    predictions: np.ndarray,
    turnover_bucket_quantile: float,
) -> tuple[float | None, float | None, float | None]:
    scored = fold_test[["date", "label"]].copy()
    scored["prediction"] = np.asarray(predictions, dtype=float)
    per_date_ic: list[float] = []
    per_date_rank_ic: list[float] = []
    per_date_turnover: list[float] = []
    previous_positions: pd.Series | None = None
    for _, group in scored.groupby("date", sort=True):
        subset = group[["prediction", "label"]].dropna()
        if len(subset) < 3:
            previous_positions = pd.Series(dtype=float)
            continue
        ic = float(subset["prediction"].corr(subset["label"], method="pearson"))
        rank_ic = float(subset["prediction"].corr(subset["label"], method="spearman"))
        if np.isfinite(ic):
            per_date_ic.append(ic)
        if np.isfinite(rank_ic):
            per_date_rank_ic.append(rank_ic)

        n_assets = int(subset["prediction"].size)
        k = max(1, int(n_assets * turnover_bucket_quantile))
        rank = subset["prediction"].rank(method="first")
        long_positions = (rank > (n_assets - k)).astype(float)
        short_positions = (rank <= k).astype(float) * -1.0
        positions = long_positions + short_positions
        if previous_positions is None:
            turnover = float(positions.abs().mean())
        else:
            aligned = positions.align(previous_positions, fill_value=0.0)
            turnover = float((aligned[0] - aligned[1]).abs().sum()) / (2.0 * max(1, n_assets))
        per_date_turnover.append(turnover)
        previous_positions = positions

    return (
        _finite_or_none(float(np.mean(per_date_ic))) if per_date_ic else None,
        _finite_or_none(float(np.mean(per_date_rank_ic))) if per_date_rank_ic else None,
        _finite_or_none(float(np.mean(per_date_turnover))) if per_date_turnover else None,
    )


def _selection_score(
    *,
    metric: ModelSelectionMetric,
    mean_ic: float | None,
    mean_rank_ic: float | None,
    mean_turnover: float | None,
    turnover_penalty: float,
) -> float | None:
    if metric == "ic":
        return mean_ic
    if metric == "rank_ic":
        if mean_turnover is not None and turnover_penalty > 0:
            return None if mean_rank_ic is None else mean_rank_ic
        return mean_rank_ic
    if metric == "rank_ic_minus_turnover_penalty":
        if mean_rank_ic is None:
            return None
        return mean_rank_ic - (float(turnover_penalty) * float(mean_turnover or 0.0))
    if metric == "ic_minus_turnover_penalty":
        if mean_ic is None:
            return None
        return mean_ic - (float(turnover_penalty) * float(mean_turnover or 0.0))
    return mean_rank_ic


def _candidate_id(idx: int, candidate: ModelSpec) -> str:
    return f"candidate_{idx + 1}_{candidate.family}"


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
    estimator = _build_estimator(model_spec)
    scale_mode = _resolve_scale_mode(scale_features, model_spec.family)
    steps: list[tuple[str, object]] = [
        (
            "imputer",
            SimpleImputer(strategy="median", keep_empty_features=True),
        )
    ]
    if scale_mode == "standard":
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    pipeline = Pipeline(steps=steps)
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


def _feature_importance_extractors_for_family(model_family: str) -> tuple[str, ...]:
    extractors = _MODEL_FAMILY_IMPORTANCE_EXTRACTORS.get(model_family)
    if extractors is None:
        raise ValueError(
            "model family is missing feature importance extractor registration: "
            f"{model_family!r}. "
            "Please update _MODEL_FAMILY_IMPORTANCE_EXTRACTORS."
        )
    return extractors


def _extract_coef_importance(estimator: object, *, n_features: int) -> np.ndarray | None:
    if not hasattr(estimator, "coef_"):
        return None
    try:
        coef = np.asarray(estimator.coef_, dtype=float).reshape(-1)  # type: ignore[attr-defined]
    except Exception:
        return None
    if coef.size != n_features or not np.isfinite(coef).all():
        return None
    return coef


def _extract_tree_importance(estimator: object, *, n_features: int) -> np.ndarray | None:
    if not hasattr(estimator, "feature_importances_"):
        return None
    try:
        values = np.asarray(
            estimator.feature_importances_,  # type: ignore[attr-defined]
            dtype=float,
        ).reshape(-1)
    except Exception:
        return None
    if values.size != n_features or not np.isfinite(values).all():
        return None
    return values


def _estimate_permutation_importance(
    pipeline: Pipeline,
    *,
    train_slice: pd.DataFrame,
    feature_columns: tuple[str, ...],
    seed: int,
    max_rows: int,
) -> np.ndarray | None:
    if not feature_columns or "label" not in train_slice.columns:
        return None

    labels = pd.to_numeric(train_slice["label"], errors="coerce")
    valid_mask = labels.notna()
    if int(valid_mask.sum()) < 2:
        return None

    features = train_slice.loc[valid_mask, list(feature_columns)].copy().reset_index(drop=True)
    target = labels.loc[valid_mask].to_numpy(dtype=float)

    if len(features) > max_rows:
        rng = np.random.default_rng(seed)
        sample_idx = np.sort(
            rng.choice(
                len(features),
                size=max_rows,
                replace=False,
            )
        )
        features = features.iloc[sample_idx].reset_index(drop=True)
        target = target[sample_idx]

    if target.size <= 1 or not np.isfinite(target).all():
        return None

    try:
        baseline_pred = np.asarray(pipeline.predict(features), dtype=float).reshape(-1)
    except Exception:
        return None
    if baseline_pred.size != target.size or not np.isfinite(baseline_pred).all():
        return None

    baseline_loss = float(np.mean((baseline_pred - target) ** 2))
    if not np.isfinite(baseline_loss):
        return None

    rng = np.random.default_rng(seed + 1_000_003)
    permuted = features.copy()
    importances = np.zeros(len(feature_columns), dtype=float)
    for idx, feature in enumerate(feature_columns):
        original_values = permuted[feature].to_numpy(copy=True)
        shuffled_values = original_values.copy()
        rng.shuffle(shuffled_values)
        permuted.loc[:, feature] = shuffled_values
        try:
            permuted_pred = np.asarray(pipeline.predict(permuted), dtype=float).reshape(-1)
        except Exception:
            permuted.loc[:, feature] = original_values
            return None
        permuted.loc[:, feature] = original_values
        if permuted_pred.size != target.size or not np.isfinite(permuted_pred).all():
            return None
        permuted_loss = float(np.mean((permuted_pred - target) ** 2))
        if not np.isfinite(permuted_loss):
            return None
        importances[idx] = max(0.0, permuted_loss - baseline_loss)
    return importances


def _feature_importance_frame(
    pipeline: Pipeline,
    *,
    train_slice: pd.DataFrame,
    feature_columns: tuple[str, ...],
    model_family: str,
    model_version: int,
    trained_until: pd.Timestamp,
    permutation_max_rows: int,
) -> pd.DataFrame:
    estimator = pipeline.named_steps["model"]
    importance_source = "unsupported"
    signed: np.ndarray | None = None

    for extractor in _feature_importance_extractors_for_family(model_family):
        if extractor == "coef":
            signed = _extract_coef_importance(estimator, n_features=len(feature_columns))
        elif extractor == "feature_importances":
            signed = _extract_tree_importance(estimator, n_features=len(feature_columns))
        elif extractor == "permutation":
            signed = _estimate_permutation_importance(
                pipeline,
                train_slice=train_slice,
                feature_columns=feature_columns,
                seed=_PERMUTATION_IMPORTANCE_RANDOM_SEED + model_version,
                max_rows=permutation_max_rows,
            )
        else:
            raise ValueError(
                f"unknown feature importance extractor {extractor!r} for {model_family!r}"
            )
        if signed is not None:
            importance_source = extractor
            break

    if signed is None or signed.size != len(feature_columns):
        signed = np.full(len(feature_columns), np.nan, dtype=float)
        importance_source = "unsupported"

    frame = pd.DataFrame(
        {
            "model_version": model_version,
            "trained_until": trained_until,
            "feature": list(feature_columns),
            "importance": signed,
            "abs_importance": np.abs(signed),
            "importance_source": importance_source,
        }
    )
    return frame


def _combine_feature_importance_frames(
    frames: list[pd.DataFrame],
    *,
    feature_columns: tuple[str, ...],
    disabled: bool = False,
) -> pd.DataFrame:
    if not frames:
        importance_source = "disabled" if disabled else "unsupported"
        return pd.DataFrame(
            {
                "feature": list(feature_columns),
                "mean_abs_importance": [float("nan")] * len(feature_columns),
                "latest_importance": [float("nan")] * len(feature_columns),
                "importance_source": [importance_source] * len(feature_columns),
                "n_model_versions": [0] * len(feature_columns),
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    latest_version = int(combined["model_version"].max())
    latest = combined[combined["model_version"] == latest_version][
        ["feature", "importance", "importance_source"]
    ].rename(columns={"importance": "latest_importance"})
    summary = (
        combined.groupby("feature", sort=False)
        .agg(
            mean_abs_importance=("abs_importance", "mean"),
            n_model_versions=("model_version", "nunique"),
        )
        .reset_index()
    )
    summary = summary.merge(latest, on="feature", how="left", validate="one_to_one")
    return summary.sort_values("feature", kind="mergesort").reset_index(drop=True)


def _build_model_diagnostics(
    *,
    config: ModelFactorBuildConfig,
    training_log_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    model_selection_df: pd.DataFrame,
    label_winsorize_zscore: float | None,
    label_winsor_clipped_rows: int,
) -> dict[str, object]:
    trained_rows = training_log_df[training_log_df["status"] != "skipped"].copy()
    skip_counts = Counter(
        str(value).strip()
        for value in training_log_df["skip_reason"]
        if isinstance(value, str) and value.strip()
    )
    selection_status_counts = Counter(
        str(value).strip()
        for value in training_log_df["selection_status"]
        if isinstance(value, str) and value.strip()
    )
    selected_families = (
        sorted(
            {
                str(value).strip()
                for value in training_log_df["model_family"]
                if isinstance(value, str) and value.strip()
            }
        )
        if not training_log_df.empty
        else [config.model.family]
    )
    if "selected" in model_selection_df.columns:
        selected_candidate_rows = model_selection_df[model_selection_df["selected"]].copy()
    else:
        selected_candidate_rows = model_selection_df.iloc[0:0].copy()
    importance_values = pd.to_numeric(
        feature_importance_df.get("mean_abs_importance", pd.Series(dtype=float)),
        errors="coerce",
    )
    ranked_importance = feature_importance_df.assign(_mean_abs_importance=importance_values)
    ranked_importance = ranked_importance[ranked_importance["_mean_abs_importance"].notna()]
    top_features = (
        ranked_importance.sort_values(
            ["_mean_abs_importance", "feature"],
            ascending=[False, True],
            kind="mergesort",
        )
        .head(5)["feature"]
        .tolist()
    )
    version_values = pd.to_numeric(
        feature_importance_df.get("n_model_versions", pd.Series(dtype=float)),
        errors="coerce",
    )
    importance_model_versions = (
        int(version_values.max())
        if not version_values.empty and version_values.notna().any()
        else 0
    )
    importance_source_counts = Counter(
        str(value).strip()
        for value in feature_importance_df.get("importance_source", pd.Series(dtype=str))
        if isinstance(value, str) and value.strip()
    )
    return {
        "factor_name": config.factor_name,
        "configured_model_family": config.model.family,
        "selected_model_families": selected_families,
        "feature_columns": list(config.feature_columns),
        "feature_count": len(config.feature_columns),
        "target_horizon": config.target_horizon,
        "purged_train_gap_dates": max(int(config.target_horizon) - 1, 0),
        "label_winsorize_zscore": (
            float(label_winsorize_zscore) if label_winsorize_zscore is not None else None
        ),
        "label_winsor_clipped_rows": int(label_winsor_clipped_rows),
        "trained_model_versions": int(training_log_df["model_version"].dropna().nunique()),
        "n_score_dates_total": int(len(training_log_df)),
        "n_score_dates_scored": int(len(trained_rows)),
        "mean_train_rows": _finite_or_none(
            trained_rows["n_train_rows"].mean() if not trained_rows.empty else float("nan")
        ),
        "mean_score_assets": _finite_or_none(
            trained_rows["n_score_assets"].mean() if not trained_rows.empty else float("nan")
        ),
        "skip_reason_counts": dict(skip_counts),
        "selection_status_counts": dict(selection_status_counts),
        "model_selection": {
            "enabled": bool(config.model_selection.enabled),
            "metric": config.model_selection.metric if config.model_selection.enabled else None,
            "candidate_count": int(len(_selection_candidates(config))),
            "n_selection_events": (
                int(selected_candidate_rows["score_date"].nunique())
                if not selected_candidate_rows.empty
                else 0
            ),
            "latest_selected_candidate_id": (
                str(selected_candidate_rows.iloc[-1]["candidate_id"])
                if not selected_candidate_rows.empty
                else None
            ),
        },
        "feature_importance": {
            "mode": config.feature_importance.mode,
            "permutation_max_rows": int(config.feature_importance.permutation_max_rows),
            "n_importance_model_versions": importance_model_versions,
            "importance_source_counts": dict(importance_source_counts),
        },
        "top_features": top_features,
    }


def _raise_on_integrity_failures(checks: list[IntegrityCheckResult]) -> None:
    failures = [check for check in checks if check.status == "fail"]
    if failures:
        first = failures[0]
        raise ValueError(first.message)


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value) if np.isfinite(value) else None


def _check_feature_known_at_not_after_signal_date(
    features: pd.DataFrame,
    *,
    known_at_col: str,
) -> IntegrityCheckResult:
    known_at = pd.to_datetime(features[known_at_col], errors="coerce")
    signal_date = pd.to_datetime(features["date"], errors="coerce")
    violations = int((known_at > signal_date).sum())
    if violations > 0:
        return IntegrityCheckResult(
            check_name="check_model_factor_feature_known_at",
            status="fail",
            severity="error",
            object_name="model_factor_feature_known_at",
            module_name="model_factor.core",
            message=f"{violations} feature rows have known_at later than signal date",
            remediation="Shift features to the first date when they are actually known.",
            metrics={"violations": violations},
        )
    return IntegrityCheckResult(
        check_name="check_model_factor_feature_known_at",
        status="pass",
        severity="info",
        object_name="model_factor_feature_known_at",
        module_name="model_factor.core",
        message="feature known_at timestamps are not later than signal dates",
        metrics={"rows_checked": int(len(features))},
    )
