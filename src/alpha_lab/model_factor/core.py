from __future__ import annotations

import weakref
from collections import Counter
from collections.abc import Callable, Mapping
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
from alpha_lab.model_factor._memory import release_unused_memory
from alpha_lab.model_factor.dataset_cache import (
    PreparedInputsNumpyCacheEntry,
    load_prepared_inputs_cache,
    write_prepared_inputs_cache,
)
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

_PERMUTATION_IMPORTANCE_MAX_ROWS = 50_000
_PERMUTATION_IMPORTANCE_RANDOM_SEED = 20260423
_PERMUTATION_IMPORTANCE_MAX_PREDICT_CALLS = 100
_MODEL_FEATURE_DTYPE = "float32"
_TREE_MODEL_FAMILIES: frozenset[str] = frozenset({"gbdt", "xgboost", "lightgbm"})
_RowSelection = slice | np.ndarray
_MODEL_FAMILY_IMPORTANCE_EXTRACTORS: dict[str, tuple[str, ...]] = {
    "linear": ("coef",),
    "ridge": ("coef",),
    "lasso": ("coef",),
    "elastic_net": ("coef",),
    "gbdt": ("feature_importances",),
    "xgboost": ("feature_importances",),
    "lightgbm": ("feature_importances",),
    "mlp": (),
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
FeatureImportanceMethod = Literal["auto", "cheap", "permutation"]
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

TRAINING_METRICS_COLUMNS: tuple[str, ...] = (
    "model_version",
    "model_family",
    "train_start",
    "train_end",
    "oos_start",
    "oos_end",
    "train_ic",
    "train_rank_ic",
    "train_loss",
    "oos_ic",
    "oos_rank_ic",
    "oos_loss",
    "n_train_obs",
    "n_train_dates",
    "n_oos_obs",
    "n_oos_dates",
    "selected_candidate_id",
    "selected_candidate_score",
)

FEATURE_OOS_IC_COLUMNS: tuple[str, ...] = (
    "feature",
    "window_start",
    "window_end",
    "model_version",
    "ic",
    "rank_ic",
    "n_obs",
    "n_dates",
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
            "method": list(get_args(FeatureImportanceMethod)),
            "default_mode": FeatureImportanceConfig().mode,
            "default_method": FeatureImportanceConfig().method,
            "default_save_ledger": FeatureImportanceConfig().save_ledger,
            "over_time_source": "cheap_ledger_only",
            "permutation_default_enabled": False,
            "default_permutation_max_rows": _PERMUTATION_IMPORTANCE_MAX_ROWS,
        },
        "supported_diagnostics": {
            "training_metrics": "training_metrics.csv",
            "feature_oos_ic": "feature_oos_ic.csv",
            "feature_oos_ic_default_enabled": True,
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


def _mapping_bool(mapping: Mapping[str, object] | None, key: str, default: bool) -> bool:
    if not isinstance(mapping, Mapping) or key not in mapping:
        return default
    value = mapping.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


def _mapping_int(mapping: Mapping[str, object] | None, key: str, default: int) -> int:
    if not isinstance(mapping, Mapping) or key not in mapping:
        return int(default)
    value = mapping.get(key)
    if isinstance(value, bool):
        return int(default)
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return int(default)


def _mapping_text(mapping: Mapping[str, object] | None, key: str, default: str) -> str:
    if not isinstance(mapping, Mapping) or key not in mapping:
        return default
    return str(mapping.get(key) or default).strip() or default


def _feature_importance_enabled(config: object) -> bool:
    return bool(getattr(config, "enabled", True)) and str(getattr(config, "mode", "")) != "disabled"


def _feature_importance_over_time_enabled(config: object) -> bool:
    over_time = cast(Mapping[str, object] | None, getattr(config, "over_time", None))
    return _mapping_bool(over_time, "enabled", True)


def _feature_importance_over_time_top_k(config: object) -> int:
    over_time = cast(Mapping[str, object] | None, getattr(config, "over_time", None))
    return max(1, _mapping_int(over_time, "top_k", 5))


def _feature_importance_over_time_source(config: object) -> str:
    over_time = cast(Mapping[str, object] | None, getattr(config, "over_time", None))
    return _mapping_text(over_time, "source", "cheap_ledger_only")


def _feature_importance_permutation_enabled(config: object) -> bool:
    permutation = cast(Mapping[str, object] | None, getattr(config, "permutation", None))
    return _mapping_bool(permutation, "enabled", False)


def _feature_importance_permutation_latest_only(config: object) -> bool:
    permutation = cast(Mapping[str, object] | None, getattr(config, "permutation", None))
    return _mapping_bool(permutation, "latest_only", True)


def _feature_importance_permutation_sample_rows(config: object) -> int:
    permutation = cast(Mapping[str, object] | None, getattr(config, "permutation", None))
    legacy = int(getattr(config, "permutation_max_rows", _PERMUTATION_IMPORTANCE_MAX_ROWS))
    return max(1, _mapping_int(permutation, "sample_rows", legacy))


def _feature_importance_permutation_sample_rows_configured(config: object) -> bool:
    permutation = cast(Mapping[str, object] | None, getattr(config, "permutation", None))
    if not isinstance(permutation, Mapping) or "sample_rows" not in permutation:
        return False
    value = permutation.get("sample_rows")
    return value is not None and str(value).strip() != ""


def _feature_importance_permutation_n_repeats(config: object) -> int:
    permutation = cast(Mapping[str, object] | None, getattr(config, "permutation", None))
    return max(1, _mapping_int(permutation, "n_repeats", 3))


def _feature_importance_permutation_top_k_features(config: object) -> int | None:
    permutation = cast(Mapping[str, object] | None, getattr(config, "permutation", None))
    if not isinstance(permutation, Mapping) or "top_k_features" not in permutation:
        return 20
    value = permutation.get("top_k_features")
    if value is None:
        return None
    if isinstance(value, bool):
        return 20
    return int(value)


def _feature_importance_permutation_random_state(config: object) -> int:
    permutation = cast(Mapping[str, object] | None, getattr(config, "permutation", None))
    return _mapping_int(permutation, "random_state", 42)


def _feature_importance_permutation_force(config: object) -> bool:
    permutation = cast(Mapping[str, object] | None, getattr(config, "permutation", None))
    return _mapping_bool(permutation, "force", False)


@dataclass(frozen=True)
class FeatureImportanceConfig:
    """Controls for model feature importance diagnostics."""

    enabled: bool = True
    method: FeatureImportanceMethod = "auto"
    save_ledger: bool = True
    mode: FeatureImportanceMode = "every_fit"
    permutation_max_rows: int = _PERMUTATION_IMPORTANCE_MAX_ROWS
    over_time: Mapping[str, object] | None = None
    permutation: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("feature_importance.enabled must be a boolean")
        if self.method not in {"auto", "cheap", "permutation"}:
            raise ValueError(
                "feature_importance.method must be one of ['auto', 'cheap', 'permutation']"
            )
        if not isinstance(self.save_ledger, bool):
            raise ValueError("feature_importance.save_ledger must be a boolean")
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
        if self.over_time is not None and not isinstance(self.over_time, Mapping):
            raise ValueError("feature_importance.over_time must be an object when provided")
        if self.permutation is not None and not isinstance(self.permutation, Mapping):
            raise ValueError("feature_importance.permutation must be an object when provided")
        repeats = _feature_importance_permutation_n_repeats(self)
        if repeats <= 0:
            raise ValueError("feature_importance.permutation.n_repeats must be > 0")
        top_k = _feature_importance_permutation_top_k_features(self)
        if top_k is not None and top_k <= 0:
            raise ValueError("feature_importance.permutation.top_k_features must be > 0")
        sample_rows = _feature_importance_permutation_sample_rows(self)
        if sample_rows <= 0:
            raise ValueError("feature_importance.permutation.sample_rows must be > 0")


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
    target_price_column: str = "close"
    max_abs_forward_return: float | None = None
    label_winsorize_zscore: float | None = None
    preparation_cache_dir: str | None = None
    preparation_cache_key: str | None = None
    compute_feature_oos_ic: bool = True

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
        if not self.target_price_column.strip():
            raise ValueError("target_price_column must be non-empty")
        if self.max_abs_forward_return is not None and self.max_abs_forward_return <= 0:
            raise ValueError("max_abs_forward_return must be > 0 when provided")
        if self.label_winsorize_zscore is not None and self.label_winsorize_zscore <= 0:
            raise ValueError("label_winsorize_zscore must be > 0 when provided")
        if self.preparation_cache_dir is not None and not self.preparation_cache_dir.strip():
            raise ValueError("preparation_cache_dir must be non-empty when provided")
        if self.preparation_cache_key is not None and not self.preparation_cache_key.strip():
            raise ValueError("preparation_cache_key must be non-empty when provided")
        if not isinstance(self.compute_feature_oos_ic, bool):
            raise ValueError("compute_feature_oos_ic must be a boolean")


@dataclass(frozen=True)
class ModelFactorBuildResult:
    """Canonical factor output plus training diagnostics."""

    factor_df: pd.DataFrame
    training_log_df: pd.DataFrame
    training_metrics_df: pd.DataFrame
    feature_importance_df: pd.DataFrame
    model_diagnostics: dict[str, object]
    feature_importance_ledger_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    coverage_base_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    feature_oos_ic_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    forward_label_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    model_selection_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    integrity_checks: tuple[IntegrityCheckResult, ...] = ()
    target_diagnostics: dict[str, object] = field(default_factory=dict)


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
    fit_date: pd.Timestamp
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


@dataclass(frozen=True)
class _PreparedModelArrays:
    feature_values: np.ndarray
    labels: np.ndarray
    dates: np.ndarray
    assets: pd.Categorical
    training_feature_values: np.ndarray | None = None
    training_labels: np.ndarray | None = None
    training_dates: np.ndarray | None = None


@dataclass
class _TrainingWindowCache:
    row_indices_by_date_pos: tuple[np.ndarray, ...]
    labeled_row_indices_by_date_pos: tuple[np.ndarray, ...]
    window_start_by_pos: np.ndarray
    eligible_end_by_pos: np.ndarray
    n_train_dates_by_pos: np.ndarray
    n_labeled_rows_by_pos: np.ndarray
    _labeled_window_rows: dict[int, np.ndarray] = field(default_factory=dict)

    def labeled_row_selection(self, score_idx: int) -> _RowSelection:
        window_start = int(self.window_start_by_pos[score_idx])
        eligible_end = int(self.eligible_end_by_pos[score_idx])
        if window_start >= eligible_end:
            return np.empty(0, dtype=np.intp)

        start: int | None = None
        end: int | None = None
        for indices in self.labeled_row_indices_by_date_pos[window_start:eligible_end]:
            if len(indices) == 0:
                continue
            date_slice = _indices_as_contiguous_slice(indices)
            if date_slice is None:
                return self.labeled_row_indices(score_idx)
            if start is None:
                start = int(date_slice.start)
                end = int(date_slice.stop)
                continue
            if int(date_slice.start) != end:
                return self.labeled_row_indices(score_idx)
            end = int(date_slice.stop)

        if start is None or end is None:
            return np.empty(0, dtype=np.intp)
        return slice(start, end)

    def labeled_row_indices(self, score_idx: int) -> np.ndarray:
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
            labels = forward_label_df[["date", "asset", "value"]].rename(
                columns={"value": "label"}
            )
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
            "over_time_source": _feature_importance_over_time_source(
                config.feature_importance
            ),
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
                over_time_enabled=_feature_importance_over_time_enabled(
                    config.feature_importance
                ),
                over_time_top_k=_feature_importance_over_time_top_k(
                    config.feature_importance
                ),
                over_time_source=_feature_importance_over_time_source(
                    config.feature_importance
                ),
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
                        if _feature_importance_permutation_latest_only(
                            config.feature_importance
                        )
                        else max(1, request.model_version)
                    ),
                    n_features=len(request.feature_columns),
                ),
                permutation_guardrail_reason=permutation_guardrail,
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


def _normalize_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    prices = prices_df.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
    validate_prices_table(prices)
    return prices


def _build_score_coverage_base_frame(
    features: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    price_universe_counts: pd.Series,
) -> pd.DataFrame:
    columns = [
        "date",
        "universe_count",
        "feature_row_count",
        "complete_feature_count",
        "feature_nan_row_count",
        "label_available_count",
        "eligible_count",
        "missing_feature_count",
        "missing_label_count",
        "filtered_count",
    ]
    if features.empty or "date" not in features.columns or "asset" not in features.columns:
        return pd.DataFrame(columns=columns)

    frame = features[["date", "asset"]].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    label_available = (
        pd.to_numeric(features["label"], errors="coerce").notna()
        if "label" in features.columns
        else pd.Series(False, index=features.index)
    )
    frame["_label_available"] = label_available.to_numpy(dtype=bool)
    grouped = frame.groupby("date", sort=True)
    summary = grouped.agg(
        feature_row_count=("asset", "nunique"),
        label_available_count=("_label_available", "sum"),
    )

    complete_feature_count: pd.Series | None = None
    if feature_columns and set(feature_columns).issubset(features.columns):
        complete_mask = features.loc[:, list(feature_columns)].notna().all(axis=1)
        complete_frame = pd.DataFrame(
            {
                "date": frame["date"].to_numpy(),
                "_complete_feature": complete_mask.to_numpy(dtype=bool),
            }
        )
        complete_feature_count = complete_frame.groupby("date", sort=True)[
            "_complete_feature"
        ].sum()

    if complete_feature_count is not None:
        summary["complete_feature_count"] = complete_feature_count.reindex(summary.index).fillna(0)
        summary["feature_nan_row_count"] = (
            summary["feature_row_count"] - summary["complete_feature_count"]
        ).clip(lower=0)
    else:
        summary["complete_feature_count"] = pd.NA
        summary["feature_nan_row_count"] = pd.NA

    price_counts = price_universe_counts.copy()
    price_counts.index = pd.to_datetime(price_counts.index, errors="coerce")
    summary["universe_count"] = price_counts.reindex(summary.index)
    summary["universe_count"] = summary["universe_count"].fillna(summary["feature_row_count"])
    summary["eligible_count"] = summary["label_available_count"]
    summary["missing_feature_count"] = (
        summary["universe_count"] - summary["feature_row_count"]
    ).clip(lower=0)
    summary["missing_label_count"] = (
        summary["feature_row_count"] - summary["label_available_count"]
    ).clip(lower=0)
    summary["filtered_count"] = (
        summary["universe_count"] - summary["eligible_count"]
    ).clip(lower=0)
    summary = summary.reset_index()
    for column in columns:
        if column not in summary.columns:
            summary[column] = pd.NA
    return summary[columns]


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
    frame = features
    if group_scope == "date_and_industry" and industry_group_column is not None:
        group_keys: str | list[str] = ["date", industry_group_column]
    else:
        group_keys = "date"
    transform_fn: Callable[[pd.Series], pd.Series]
    if mode == "zscore":
        transform_fn = _zscore_finite
    elif mode == "rank":
        transform_fn = _rank_centered
    elif mode == "winsorize_zscore":
        transform_fn = _winsorize_then_zscore
    else:
        return frame
    for column in cols:
        transformed = frame.groupby(group_keys, sort=False)[column].transform(transform_fn)
        frame[column] = pd.to_numeric(transformed, errors="coerce").astype(_MODEL_FEATURE_DTYPE)
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


def _prices_for_target_labels(prices: pd.DataFrame, *, price_column: str) -> pd.DataFrame:
    column = str(price_column or "").strip()
    if not column:
        raise ValueError("target_price_column must be non-empty")
    if column not in prices.columns:
        raise ValueError(f"target price column {column!r} is missing from prices")
    required = ["date", "asset", column]
    frame = prices.loc[:, required].copy()
    if column != "close":
        frame = frame.rename(columns={column: "close"})
    return frame.loc[:, ["date", "asset", "close"]]


def _apply_forward_return_extreme_filter(
    forward_label_df: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    label_prices: pd.DataFrame,
    target_price_column: str,
    horizon: int,
    max_abs_forward_return: float | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    values = pd.to_numeric(forward_label_df["value"], errors="coerce")
    finite_values = values[np.isfinite(values)]
    max_abs_raw = (
        float(finite_values.abs().max())
        if not finite_values.empty and finite_values.notna().any()
        else None
    )
    diagnostics: dict[str, object] = {
        "target_price_column": str(target_price_column),
        "label_extreme_max_abs_raw_return": max_abs_raw,
        "label_extreme_filter_threshold": (
            float(max_abs_forward_return) if max_abs_forward_return is not None else None
        ),
        "label_extreme_filtered_rows": 0,
        "label_extreme_top_samples": [],
    }
    if max_abs_forward_return is None:
        return forward_label_df, labels, diagnostics

    threshold = float(max_abs_forward_return)
    mask = values.abs() > threshold
    filtered_rows = int(mask.sum())
    diagnostics["label_extreme_filtered_rows"] = filtered_rows
    if filtered_rows <= 0:
        return forward_label_df, labels, diagnostics

    samples = _forward_return_extreme_samples(
        forward_label_df.loc[mask, ["date", "asset", "value"]],
        label_prices=label_prices,
        horizon=int(horizon),
        limit=20,
    )
    diagnostics["label_extreme_top_samples"] = samples
    filtered_label_df = forward_label_df.copy()
    filtered_labels = labels.copy()
    filtered_label_df.loc[mask, "value"] = np.nan
    filtered_labels.loc[mask, "label"] = np.nan
    return filtered_label_df, filtered_labels, diagnostics


def _forward_return_extreme_samples(
    outlier_rows: pd.DataFrame,
    *,
    label_prices: pd.DataFrame,
    horizon: int,
    limit: int,
) -> list[dict[str, object]]:
    if outlier_rows.empty:
        return []
    price_frame = label_prices.loc[:, ["date", "asset", "close"]].copy()
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    price_frame = price_frame.sort_values(["asset", "date"], kind="mergesort")
    price_frame["entry_price"] = pd.to_numeric(price_frame["close"], errors="coerce")
    price_frame["exit_price"] = price_frame.groupby("asset", sort=False)["entry_price"].shift(
        -int(horizon)
    )
    sidecar = price_frame.loc[:, ["date", "asset", "entry_price", "exit_price"]]
    frame = outlier_rows.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["raw_return"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.merge(sidecar, on=["date", "asset"], how="left", validate="one_to_one")
    frame["_abs"] = frame["raw_return"].abs()
    frame = frame.sort_values("_abs", ascending=False, kind="mergesort").head(int(limit))
    samples: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        samples.append(
            {
                "date": pd.Timestamp(row.date).date().isoformat(),
                "asset": str(row.asset),
                "raw_return": _sample_float_or_none(row.raw_return),
                "entry_price": _sample_float_or_none(row.entry_price),
                "exit_price": _sample_float_or_none(row.exit_price),
            }
        )
    return samples


def _sample_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(cast(Any, value))
    except (TypeError, ValueError):
        return None
    return _finite_or_none(number)


def _target_diagnostics_from_data_health(data_health: dict[str, object]) -> dict[str, object]:
    keys = (
        "target_price_column",
        "label_extreme_max_abs_raw_return",
        "label_extreme_filter_threshold",
        "label_extreme_filtered_rows",
        "label_extreme_top_samples",
    )
    out = {key: data_health.get(key) for key in keys if key in data_health}
    out.setdefault("label_extreme_filtered_rows", 0)
    out.setdefault("label_extreme_top_samples", [])
    return out


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
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype(_MODEL_FEATURE_DTYPE)
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
    *,
    allow_missing: bool = False,
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
            if allow_missing:
                row_indices_by_pos.append(np.empty(0, dtype=np.intp))
            else:
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


def _prepare_model_arrays(
    merged: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
) -> _PreparedModelArrays:
    feature_frame = merged.loc[:, list(feature_columns)]
    feature_values = feature_frame.to_numpy(dtype=np.float32, copy=False)
    if not feature_values.flags.c_contiguous:
        feature_values = np.ascontiguousarray(feature_values, dtype=np.float32)
    labels = pd.to_numeric(merged["label"], errors="coerce").to_numpy(
        dtype=np.float32,
        copy=True,
    )
    dates = pd.to_datetime(merged["date"]).to_numpy(copy=True)
    assets = pd.Categorical(merged["asset"])
    return _PreparedModelArrays(
        feature_values=feature_values,
        labels=labels,
        dates=dates,
        assets=assets,
    )


def _prepare_model_arrays_from_numpy_cache(
    entry: PreparedInputsNumpyCacheEntry,
) -> _PreparedModelArrays:
    index_df = entry.index_df
    labeled_index_df = entry.labeled_index_df
    return _PreparedModelArrays(
        feature_values=entry.feature_values,
        labels=pd.to_numeric(index_df["label"], errors="coerce").to_numpy(
            dtype=np.float32,
            copy=True,
        ),
        dates=pd.to_datetime(index_df["date"]).to_numpy(copy=True),
        assets=pd.Categorical(index_df["asset"]),
        training_feature_values=entry.labeled_feature_values,
        training_labels=pd.to_numeric(labeled_index_df["label"], errors="coerce").to_numpy(
            dtype=np.float32,
            copy=True,
        ),
        training_dates=pd.to_datetime(labeled_index_df["date"]).to_numpy(copy=True),
    )


def _weakref_or_none(value: object) -> weakref.ReferenceType[object] | None:
    try:
        return weakref.ref(value)
    except TypeError:
        return None


def _indices_as_contiguous_slice(indices: np.ndarray) -> slice | None:
    if len(indices) == 0:
        return slice(0, 0)
    first = int(indices[0])
    last = int(indices[-1])
    if last < first:
        return None
    if last - first + 1 != len(indices):
        return None
    expected = np.arange(first, last + 1, dtype=np.intp)
    if not np.array_equal(indices, expected):
        return None
    return slice(first, last + 1)


def _row_selection_mode(selection: _RowSelection) -> str:
    return "contiguous_slice" if isinstance(selection, slice) else "advanced_index"


def _row_selection_length(selection: _RowSelection, *, n_rows: int) -> int:
    if isinstance(selection, slice):
        start, stop, step = selection.indices(n_rows)
        if step <= 0:
            return 0
        return max((stop - start + step - 1) // step, 0)
    return int(len(selection))


def _row_selection_to_indices(selection: _RowSelection, *, n_rows: int) -> np.ndarray:
    if isinstance(selection, slice):
        start, stop, step = selection.indices(n_rows)
        return np.arange(start, stop, step, dtype=np.intp)
    return selection.astype(np.intp, copy=False)


def _training_matrix_from_selection(
    feature_values: np.ndarray,
    selection: _RowSelection,
) -> np.ndarray:
    selected = feature_values[selection]
    if isinstance(selection, slice):
        return np.array(selected, dtype=np.float32, order="C", copy=True)
    return np.asarray(selected, dtype=np.float32, order="C")


def _feature_importance_training_slice_from_arrays(
    prepared_arrays: _PreparedModelArrays,
    *,
    row_selection: _RowSelection,
    feature_columns: tuple[str, ...],
    model_version: int,
    max_rows: int,
) -> pd.DataFrame:
    columns = [*feature_columns, "label"]
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
    row_indices = _row_selection_to_indices(
        row_selection,
        n_rows=len(labels_array),
    )
    if len(row_indices) == 0:
        return pd.DataFrame(columns=columns)

    labels = labels_array[row_indices]
    valid_pos = np.flatnonzero(np.isfinite(labels))
    if valid_pos.size < 2:
        return pd.DataFrame(columns=columns)
    selected_rows = row_indices[valid_pos]
    if len(selected_rows) > max_rows:
        rng = np.random.default_rng(_PERMUTATION_IMPORTANCE_RANDOM_SEED + model_version)
        sample_pos = np.sort(rng.choice(len(selected_rows), size=max_rows, replace=False))
        selected_rows = selected_rows[sample_pos]

    data: dict[str, object] = {
        column: feature_values[selected_rows, idx]
        for idx, column in enumerate(feature_columns)
    }
    data["label"] = labels_array[selected_rows].astype(float, copy=False)
    return pd.DataFrame(data, columns=columns)


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


def _training_metrics_row_from_frame(
    *,
    fitted: _FittedModelBundle,
    train_slice: pd.DataFrame,
    feature_columns: tuple[str, ...],
    selected_candidate_id: str | None,
    selected_candidate_score: float | None,
) -> dict[str, object]:
    train_features = _prepare_training_matrix(
        train_slice,
        feature_columns=feature_columns,
        model_family=fitted.model_family,
    )
    predictions = fitted.pipeline.predict(train_features)
    metrics = _prediction_diagnostics_by_date(
        dates=train_slice["date"].to_numpy(),
        labels=train_slice["label"].to_numpy(),
        predictions=np.asarray(predictions, dtype=float),
    )
    return _training_metrics_base_row(
        fitted=fitted,
        metrics=metrics,
        selected_candidate_id=selected_candidate_id,
        selected_candidate_score=selected_candidate_score,
    )


def _training_metrics_row_from_arrays(
    *,
    fitted: _FittedModelBundle,
    prepared_arrays: _PreparedModelArrays,
    row_selection: _RowSelection,
    selected_candidate_id: str | None,
    selected_candidate_score: float | None,
) -> dict[str, object]:
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
    train_features = _training_matrix_from_selection(feature_values, row_selection)
    predictions = fitted.pipeline.predict(train_features)
    row_indices = _row_selection_to_indices(row_selection, n_rows=len(labels_array))
    metrics = _prediction_diagnostics_by_date(
        dates=dates_array[row_indices],
        labels=labels_array[row_indices],
        predictions=np.asarray(predictions, dtype=float),
    )
    return _training_metrics_base_row(
        fitted=fitted,
        metrics=metrics,
        selected_candidate_id=selected_candidate_id,
        selected_candidate_score=selected_candidate_score,
    )


def _training_metrics_base_row(
    *,
    fitted: _FittedModelBundle,
    metrics: dict[str, object],
    selected_candidate_id: str | None,
    selected_candidate_score: float | None,
) -> dict[str, object]:
    return {
        "model_version": int(fitted.model_version),
        "model_family": fitted.model_family,
        "train_start": fitted.train_start.date().isoformat(),
        "train_end": fitted.train_end.date().isoformat(),
        "train_ic": metrics.get("ic"),
        "train_rank_ic": metrics.get("rank_ic"),
        "train_loss": metrics.get("loss"),
        "n_train_obs": metrics.get("n_obs"),
        "n_train_dates": metrics.get("n_dates"),
        "selected_candidate_id": selected_candidate_id,
        "selected_candidate_score": _finite_or_none(selected_candidate_score),
    }


def _oos_training_metrics_row(
    *,
    model_version: int,
    dates: object,
    labels: object,
    predictions: np.ndarray,
) -> dict[str, object]:
    metrics = _prediction_diagnostics_by_date(
        dates=dates,
        labels=labels,
        predictions=predictions,
    )
    return {
        "model_version": int(model_version),
        "oos_start": metrics.get("start"),
        "oos_end": metrics.get("end"),
        "oos_ic": metrics.get("ic"),
        "oos_rank_ic": metrics.get("rank_ic"),
        "oos_loss": metrics.get("loss"),
        "n_oos_obs": metrics.get("n_obs"),
        "n_oos_dates": metrics.get("n_dates"),
    }


def _prediction_diagnostics_by_date(
    *,
    dates: object,
    labels: object,
    predictions: object,
) -> dict[str, object]:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(pd.Series(dates), errors="coerce"),
            "label": pd.to_numeric(pd.Series(labels), errors="coerce"),
            "prediction": pd.to_numeric(pd.Series(predictions), errors="coerce"),
        }
    )
    frame = frame.dropna(subset=["date"])
    finite_mask = np.isfinite(frame["label"].to_numpy(dtype=float)) & np.isfinite(
        frame["prediction"].to_numpy(dtype=float)
    )
    valid = frame.loc[finite_mask, ["date", "label", "prediction"]].copy()
    if valid.empty:
        return {
            "start": None,
            "end": None,
            "ic": None,
            "rank_ic": None,
            "loss": None,
            "n_obs": 0,
            "n_dates": 0,
        }

    per_date_ic: list[float] = []
    per_date_rank_ic: list[float] = []
    for _, group in valid.groupby("date", sort=True):
        if len(group) < 3:
            continue
        if (
            group["prediction"].nunique(dropna=True) < 2
            or group["label"].nunique(dropna=True) < 2
        ):
            continue
        ic = float(group["prediction"].corr(group["label"], method="pearson"))
        rank_ic = float(group["prediction"].corr(group["label"], method="spearman"))
        if np.isfinite(ic):
            per_date_ic.append(ic)
        if np.isfinite(rank_ic):
            per_date_rank_ic.append(rank_ic)
    residual = valid["prediction"].to_numpy(dtype=float) - valid["label"].to_numpy(dtype=float)
    dates_index = pd.DatetimeIndex(valid["date"])
    return {
        "start": dates_index.min().date().isoformat(),
        "end": dates_index.max().date().isoformat(),
        "ic": _finite_or_none(float(np.mean(per_date_ic))) if per_date_ic else None,
        "rank_ic": (
            _finite_or_none(float(np.mean(per_date_rank_ic))) if per_date_rank_ic else None
        ),
        "loss": _finite_or_none(float(np.mean(np.square(residual)))),
        "n_obs": int(len(valid)),
        "n_dates": int(valid["date"].nunique()),
    }


def _feature_oos_ic_rows(
    *,
    model_version: int,
    feature_columns: tuple[str, ...],
    dates: object,
    labels: object,
    feature_values: object,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if isinstance(feature_values, pd.DataFrame):
        feature_frame = feature_values.loc[:, list(feature_columns)]
    else:
        feature_frame = pd.DataFrame(np.asarray(feature_values), columns=list(feature_columns))
    for feature in feature_columns:
        metrics = _prediction_diagnostics_by_date(
            dates=dates,
            labels=labels,
            predictions=feature_frame[feature].to_numpy(),
        )
        rows.append(
            {
                "feature": feature,
                "window_start": metrics.get("start"),
                "window_end": metrics.get("end"),
                "model_version": int(model_version),
                "ic": metrics.get("ic"),
                "rank_ic": metrics.get("rank_ic"),
                "n_obs": metrics.get("n_obs"),
                "n_dates": metrics.get("n_dates"),
            }
        )
    return rows


def _build_training_metrics_frame(
    *,
    training_metrics_rows: list[dict[str, object]],
    oos_metrics_rows: list[dict[str, object]],
) -> pd.DataFrame:
    if not training_metrics_rows:
        return pd.DataFrame(columns=TRAINING_METRICS_COLUMNS)
    oos_by_version: dict[int, dict[str, object]] = {}
    for row in oos_metrics_rows:
        raw_version = row.get("model_version")
        if raw_version is None or pd.isna(raw_version):
            continue
        version = _object_to_int(raw_version)
        existing = oos_by_version.get(version)
        if existing is None:
            oos_by_version[version] = dict(row)
            continue
        oos_by_version[version] = _merge_oos_metric_rows(existing, row)
    merged_rows: list[dict[str, object]] = []
    for row in training_metrics_rows:
        version = _object_to_int(row["model_version"])
        merged = dict(row)
        merged.update(
            {
                "oos_start": None,
                "oos_end": None,
                "oos_ic": None,
                "oos_rank_ic": None,
                "oos_loss": None,
                "n_oos_obs": 0,
                "n_oos_dates": 0,
            }
        )
        merged.update(oos_by_version.get(version, {}))
        merged_rows.append(merged)
    return (
        pd.DataFrame(merged_rows, columns=list(TRAINING_METRICS_COLUMNS))
        .sort_values("model_version", kind="mergesort")
        .reset_index(drop=True)
    )


def _merge_oos_metric_rows(
    left: dict[str, object],
    right: dict[str, object],
) -> dict[str, object]:
    # Defensive path for unusual future batching; current pipeline flushes once per model version.
    return {
        "model_version": left.get("model_version"),
        "oos_start": min(
            str(left.get("oos_start") or right.get("oos_start")),
            str(right.get("oos_start") or left.get("oos_start")),
        ),
        "oos_end": max(
            str(left.get("oos_end") or right.get("oos_end")),
            str(right.get("oos_end") or left.get("oos_end")),
        ),
        "oos_ic": _finite_weighted_average(left, right, "oos_ic", "n_oos_dates"),
        "oos_rank_ic": _finite_weighted_average(left, right, "oos_rank_ic", "n_oos_dates"),
        "oos_loss": _finite_weighted_average(left, right, "oos_loss", "n_oos_obs"),
        "n_oos_obs": _object_to_int(left.get("n_oos_obs"))
        + _object_to_int(right.get("n_oos_obs")),
        "n_oos_dates": _object_to_int(left.get("n_oos_dates"))
        + _object_to_int(right.get("n_oos_dates")),
    }


def _finite_weighted_average(
    left: dict[str, object],
    right: dict[str, object],
    value_key: str,
    weight_key: str,
) -> float | None:
    values: list[tuple[float, float]] = []
    for row in (left, right):
        value = _finite_or_none(_object_to_float(row.get(value_key)))
        weight = _finite_or_none(_object_to_float(row.get(weight_key)))
        if value is not None and weight is not None and weight > 0:
            values.append((value, weight))
    if not values:
        return None
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return None
    return _finite_or_none(sum(value * weight for value, weight in values) / total_weight)


def _build_feature_oos_ic_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=FEATURE_OOS_IC_COLUMNS)
    return (
        pd.DataFrame(rows, columns=list(FEATURE_OOS_IC_COLUMNS))
        .sort_values(["model_version", "feature"], kind="mergesort")
        .reset_index(drop=True)
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


def _feature_importance_extractors_for_family(model_family: str) -> tuple[str, ...]:
    extractors = _MODEL_FAMILY_IMPORTANCE_EXTRACTORS.get(model_family)
    if extractors is None:
        raise ValueError(
            "model family is missing feature importance extractor registration: "
            f"{model_family!r}. "
            "Please update _MODEL_FAMILY_IMPORTANCE_EXTRACTORS."
        )
    return extractors or ("unsupported",)


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
    n_repeats: int = 3,
    top_k_features: int | None = None,
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
        baseline_pred = np.asarray(
            _predict_feature_frame(pipeline, features),
            dtype=float,
        ).reshape(-1)
    except Exception:
        return None
    if baseline_pred.size != target.size or not np.isfinite(baseline_pred).all():
        return None

    baseline_loss = float(np.mean((baseline_pred - target) ** 2))
    if not np.isfinite(baseline_loss):
        return None

    rng = np.random.default_rng(seed + 1_000_003)
    permuted = features.copy()
    importances = np.full(len(feature_columns), np.nan, dtype=float)
    feature_indices = np.arange(len(feature_columns), dtype=int)
    if top_k_features is not None and top_k_features < len(feature_indices):
        feature_indices = feature_indices[: max(1, int(top_k_features))]
    for idx in feature_indices:
        feature = feature_columns[int(idx)]
        original_values = permuted[feature].to_numpy(copy=True)
        losses: list[float] = []
        for _ in range(max(1, int(n_repeats))):
            shuffled_values = original_values.copy()
            rng.shuffle(shuffled_values)
            permuted.loc[:, feature] = shuffled_values
            try:
                permuted_pred = np.asarray(
                    _predict_feature_frame(pipeline, permuted),
                    dtype=float,
                ).reshape(-1)
            except Exception:
                permuted.loc[:, feature] = original_values
                return None
            permuted.loc[:, feature] = original_values
            if permuted_pred.size != target.size or not np.isfinite(permuted_pred).all():
                return None
            permuted_loss = float(np.mean((permuted_pred - target) ** 2))
            if not np.isfinite(permuted_loss):
                return None
            losses.append(max(0.0, permuted_loss - baseline_loss))
        importances[int(idx)] = float(np.mean(losses)) if losses else np.nan
    return importances


def _predict_feature_frame(pipeline: Pipeline, features: pd.DataFrame) -> np.ndarray:
    imputer = pipeline.named_steps.get("imputer")
    if hasattr(imputer, "feature_names_in_"):
        return np.asarray(pipeline.predict(features.copy()))
    return np.asarray(
        pipeline.predict(features.to_numpy(dtype=np.float32, copy=True)),
    )


def _feature_importance_training_slice(
    train_slice: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    model_version: int,
    max_rows: int,
) -> pd.DataFrame:
    """Keep only the rows/columns feature importance can inspect.

    The permutation extractor already caps itself with the same deterministic sample.
    Sampling here avoids retaining a full walk-forward training window until the end
    of latest-only runs while preserving the rows the extractor would have used.
    """

    columns = [*feature_columns, "label"]
    slim = train_slice.loc[:, columns]
    labels = pd.to_numeric(slim["label"], errors="coerce")
    valid_mask = labels.notna()
    if int(valid_mask.sum()) < 2:
        return slim.iloc[0:0].copy()
    slim = slim.loc[valid_mask]
    if len(slim) > max_rows:
        rng = np.random.default_rng(_PERMUTATION_IMPORTANCE_RANDOM_SEED + model_version)
        sample_idx = np.sort(rng.choice(len(slim), size=max_rows, replace=False))
        slim = slim.iloc[sample_idx]
    return slim.reset_index(drop=True).copy()


def _estimated_permutation_predict_calls(
    config: FeatureImportanceConfig,
    *,
    n_versions_for_estimate: int,
    n_features: int,
) -> int:
    top_k = _feature_importance_permutation_top_k_features(config)
    n_features_used = int(n_features if top_k is None else min(n_features, top_k))
    return (
        max(1, int(n_versions_for_estimate))
        * max(1, n_features_used)
        * max(1, _feature_importance_permutation_n_repeats(config))
    )


def _permutation_importance_guardrail_reason(
    config: FeatureImportanceConfig,
    *,
    model_family: str,
    n_versions_for_estimate: int,
    n_features: int,
) -> str:
    if not _feature_importance_permutation_enabled(config):
        return ""
    if _feature_importance_permutation_force(config):
        return ""
    latest_only = _feature_importance_permutation_latest_only(config)
    sample_rows_configured = _feature_importance_permutation_sample_rows_configured(config)
    n_repeats = _feature_importance_permutation_n_repeats(config)
    estimated_calls = _estimated_permutation_predict_calls(
        config,
        n_versions_for_estimate=n_versions_for_estimate,
        n_features=n_features,
    )
    if n_versions_for_estimate > 5 and not latest_only:
        return (
            "Permutation importance over all versions is expensive and disabled by "
            "default. Use latest_only=true, reduce sample_rows, reduce n_repeats, "
            "or persist cheap importance ledger."
        )
    if n_features > 50 and not sample_rows_configured:
        return "Permutation importance skipped: n_features > 50 and sample_rows is not set."
    if n_repeats > 5 and n_versions_for_estimate > 1:
        return "Permutation importance skipped: n_repeats > 5 across multiple versions."
    if str(model_family).lower() == "mlp" and not latest_only:
        return "Permutation importance for MLP over multiple versions is disabled by default."
    if estimated_calls > _PERMUTATION_IMPORTANCE_MAX_PREDICT_CALLS:
        return (
            "Permutation importance estimated_predict_calls exceeds guardrail "
            f"({_PERMUTATION_IMPORTANCE_MAX_PREDICT_CALLS})."
        )
    return ""


def _feature_importance_frame(
    pipeline: Pipeline,
    *,
    train_slice: pd.DataFrame,
    feature_columns: tuple[str, ...],
    model_family: str,
    model_version: int,
    fit_date: pd.Timestamp,
    trained_until: pd.Timestamp,
    config: FeatureImportanceConfig,
    permutation_guardrail_reason: str = "",
) -> pd.DataFrame:
    estimator = pipeline.named_steps["model"]
    importance_source = "unsupported"
    signed: np.ndarray | None = None
    abs_importance: np.ndarray | None = None

    extractors = list(_feature_importance_extractors_for_family(model_family))
    if config.method == "permutation":
        extractors = ["permutation"] if _feature_importance_permutation_enabled(config) else []
    elif (
        config.method == "auto"
        and _feature_importance_permutation_enabled(config)
        and "permutation" not in extractors
    ):
        extractors.append("permutation")
    for extractor in extractors:
        if extractor == "unsupported":
            continue
        if extractor == "coef":
            signed = _extract_coef_importance(estimator, n_features=len(feature_columns))
            if signed is not None:
                abs_importance = np.abs(signed)
                importance_source = "coefficient"
        elif extractor == "feature_importances":
            tree_importance = _extract_tree_importance(estimator, n_features=len(feature_columns))
            if tree_importance is not None:
                signed = np.full(len(feature_columns), np.nan, dtype=float)
                abs_importance = np.abs(tree_importance)
                importance_source = "built_in"
        elif extractor == "permutation":
            if permutation_guardrail_reason:
                importance_source = "permutation_skipped_guardrail"
                continue
            signed = _estimate_permutation_importance(
                pipeline,
                train_slice=train_slice,
                feature_columns=feature_columns,
                seed=_feature_importance_permutation_random_state(config) + model_version,
                max_rows=_feature_importance_permutation_sample_rows(config),
                n_repeats=_feature_importance_permutation_n_repeats(config),
                top_k_features=_feature_importance_permutation_top_k_features(config),
            )
            if signed is not None:
                abs_importance = np.abs(signed)
                signed = np.full(len(feature_columns), np.nan, dtype=float)
                importance_source = "permutation_sampled"
        else:
            raise ValueError(
                f"unknown feature importance extractor {extractor!r} for {model_family!r}"
            )
        if abs_importance is not None:
            break

    if abs_importance is None or abs_importance.size != len(feature_columns):
        abs_importance = np.full(len(feature_columns), np.nan, dtype=float)
    if signed is None or signed.size != len(feature_columns):
        signed = np.full(len(feature_columns), np.nan, dtype=float)
    if not np.isfinite(abs_importance).any() and importance_source == "unsupported":
        if str(model_family).lower() == "mlp":
            importance_source = "unsupported_mlp_default"
        elif str(model_family).lower() == "gbdt":
            importance_source = "built_in_unavailable"

    total_abs = float(np.nansum(abs_importance))
    if not np.isfinite(total_abs) or total_abs <= 0:
        normalized = np.full(len(feature_columns), np.nan, dtype=float)
        ranks = np.full(len(feature_columns), np.nan, dtype=float)
    else:
        normalized = abs_importance / total_abs
        ranks = (
            pd.Series(abs_importance)
            .rank(method="first", ascending=False, na_option="bottom")
            .to_numpy(dtype=float)
        )
    legacy_importance = np.where(np.isfinite(signed), signed, abs_importance)

    frame = pd.DataFrame(
        {
            "model_version": model_version,
            "fit_date": fit_date,
            "trained_until": trained_until,
            "model_family": model_family,
            "feature": list(feature_columns),
            "signed_importance": signed,
            "importance": legacy_importance,
            "abs_importance": abs_importance,
            "normalized_share": normalized,
            "rank": ranks,
            "importance_source": importance_source,
            "permutation_sampled": importance_source == "permutation_sampled",
            "permutation_sample_rows": (
                _feature_importance_permutation_sample_rows(config)
                if importance_source == "permutation_sampled"
                else np.nan
            ),
            "permutation_n_repeats": (
                _feature_importance_permutation_n_repeats(config)
                if importance_source == "permutation_sampled"
                else np.nan
            ),
            "permutation_guardrail_reason": permutation_guardrail_reason,
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
                "mean_signed_importance": [float("nan")] * len(feature_columns),
                "latest_abs_importance": [float("nan")] * len(feature_columns),
                "positive_version_count": [0] * len(feature_columns),
                "negative_version_count": [0] * len(feature_columns),
                "zero_version_count": [0] * len(feature_columns),
                "sign_stability": [float("nan")] * len(feature_columns),
                "importance_source": [importance_source] * len(feature_columns),
                "n_model_versions": [0] * len(feature_columns),
            }
        )

    combined = pd.concat(frames, ignore_index=True)
    latest_version = int(combined["model_version"].max())
    latest = combined[combined["model_version"] == latest_version][
        [
            "feature",
            "signed_importance",
            "abs_importance",
            "importance_source",
        ]
    ].rename(
        columns={
            "signed_importance": "latest_importance",
            "abs_importance": "latest_abs_importance",
        }
    )
    signed_values = pd.to_numeric(combined.get("signed_importance"), errors="coerce")
    combined = combined.assign(_signed_importance=signed_values)
    sign_summary = (
        combined.assign(
            _positive=combined["_signed_importance"] > 0,
            _negative=combined["_signed_importance"] < 0,
            _zero=combined["_signed_importance"] == 0,
            _signed_available=combined["_signed_importance"].notna(),
        )
        .groupby("feature", sort=False)
        .agg(
            positive_version_count=("_positive", "sum"),
            negative_version_count=("_negative", "sum"),
            zero_version_count=("_zero", "sum"),
            signed_available_count=("_signed_available", "sum"),
        )
        .reset_index()
    )
    summary = (
        combined.groupby("feature", sort=False)
        .agg(
            mean_abs_importance=("abs_importance", "mean"),
            mean_signed_importance=("_signed_importance", "mean"),
            n_model_versions=("model_version", "nunique"),
        )
        .reset_index()
    )
    summary = summary.merge(latest, on="feature", how="left", validate="one_to_one")
    summary = summary.merge(sign_summary, on="feature", how="left", validate="one_to_one")
    sources = (
        combined.groupby("feature", sort=False)["importance_source"]
        .apply(lambda values: "|".join(sorted({str(v) for v in values if str(v).strip()})))
        .reset_index()
    )
    summary = summary.drop(columns=["importance_source"], errors="ignore").merge(
        sources,
        on="feature",
        how="left",
        validate="one_to_one",
    )
    signed_available = pd.to_numeric(summary["signed_available_count"], errors="coerce").fillna(0)
    max_signed_side = pd.concat(
        [
            pd.to_numeric(summary["positive_version_count"], errors="coerce").fillna(0),
            pd.to_numeric(summary["negative_version_count"], errors="coerce").fillna(0),
        ],
        axis=1,
    ).max(axis=1)
    summary["sign_stability"] = np.where(
        signed_available > 0,
        max_signed_side / signed_available,
        np.nan,
    )
    summary = summary.drop(columns=["signed_available_count"], errors="ignore")
    return summary.sort_values("feature", kind="mergesort").reset_index(drop=True)


def _combine_feature_importance_ledger_frames(
    frames: list[pd.DataFrame],
    *,
    save_ledger: bool,
) -> pd.DataFrame:
    columns = [
        "model_family",
        "model_version",
        "fit_date",
        "trained_until",
        "feature",
        "signed_importance",
        "abs_importance",
        "normalized_share",
        "rank",
        "importance_source",
        "permutation_sampled",
        "permutation_sample_rows",
        "permutation_n_repeats",
        "permutation_guardrail_reason",
    ]
    if not save_ledger or not frames:
        return pd.DataFrame(columns=columns)
    ledger = pd.concat(frames, ignore_index=True)
    for column in columns:
        if column not in ledger.columns:
            ledger[column] = pd.NA
    return (
        ledger.loc[:, columns]
        .sort_values(["model_version", "rank", "feature"], kind="mergesort")
        .reset_index(drop=True)
    )


def _build_model_diagnostics(
    *,
    config: ModelFactorBuildConfig,
    training_log_df: pd.DataFrame,
    training_metrics_df: pd.DataFrame,
    feature_importance_df: pd.DataFrame,
    feature_oos_ic_df: pd.DataFrame,
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
        "target_price_column": config.target_price_column,
        "max_abs_forward_return": (
            float(config.max_abs_forward_return)
            if config.max_abs_forward_return is not None
            else None
        ),
        "purged_train_gap_dates": max(int(config.target_horizon) - 1, 0),
        "label_winsorize_zscore": (
            float(label_winsorize_zscore) if label_winsorize_zscore is not None else None
        ),
        "label_winsor_clipped_rows": int(label_winsor_clipped_rows),
        "trained_model_versions": int(training_log_df["model_version"].dropna().nunique()),
        "training_metrics_rows": int(len(training_metrics_df)),
        "feature_oos_ic_rows": int(len(feature_oos_ic_df)),
        "feature_oos_ic_enabled": bool(config.compute_feature_oos_ic),
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
            "enabled": bool(_feature_importance_enabled(config.feature_importance)),
            "mode": config.feature_importance.mode,
            "method": config.feature_importance.method,
            "save_ledger": bool(config.feature_importance.save_ledger),
            "over_time": {
                "enabled": bool(
                    _feature_importance_over_time_enabled(config.feature_importance)
                ),
                "top_k": int(_feature_importance_over_time_top_k(config.feature_importance)),
                "source": _feature_importance_over_time_source(config.feature_importance),
            },
            "permutation": {
                "enabled": bool(
                    _feature_importance_permutation_enabled(config.feature_importance)
                ),
                "latest_only": bool(
                    _feature_importance_permutation_latest_only(config.feature_importance)
                ),
                "sample_rows": int(
                    _feature_importance_permutation_sample_rows(config.feature_importance)
                ),
                "n_repeats": int(
                    _feature_importance_permutation_n_repeats(config.feature_importance)
                ),
                "top_k_features": _feature_importance_permutation_top_k_features(
                    config.feature_importance
                ),
                "random_state": int(
                    _feature_importance_permutation_random_state(config.feature_importance)
                ),
                "force": bool(
                    _feature_importance_permutation_force(config.feature_importance)
                ),
            },
            "permutation_max_rows": int(
                _feature_importance_permutation_sample_rows(config.feature_importance)
            ),
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


def _object_to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return None


def _object_to_int(value: object, *, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(cast(Any, value))
    except (TypeError, ValueError):
        return default


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
