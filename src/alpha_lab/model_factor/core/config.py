from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast, get_args

import pandas as pd

from alpha_lab.research_integrity.contracts import IntegrityCheckResult

# Cross-module imports (auto-added)
from ._utils import _mapping_bool, _mapping_int, _mapping_text
from .types import (
    _PERMUTATION_IMPORTANCE_MAX_ROWS,
    _RESERVED_FEATURE_COLUMNS,
    CrossSectionalGroupScope,
    CrossSectionalTransform,
    FeatureImportanceMethod,
    FeatureImportanceMode,
    MissingPolicy,
    ModelFamily,
    ModelSelectionMetric,
    ScaleFeatures,
    WindowType,
)


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
    if isinstance(value, int | float | str):
        return int(value)
    return 20


def _feature_importance_permutation_random_state(config: object) -> int:
    permutation = cast(Mapping[str, object] | None, getattr(config, "permutation", None))
    return _mapping_int(permutation, "random_state", 42)


def _feature_importance_permutation_force(config: object) -> bool:
    permutation = cast(Mapping[str, object] | None, getattr(config, "permutation", None))
    return _mapping_bool(permutation, "force", False)


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
            raise ValueError("model_selection.turnover_penalty_lambda must be >= 0 and finite")
        if not isinstance(self.turnover_bucket_quantile, (int, float)):
            raise ValueError("model_selection.turnover_bucket_quantile must be numeric")
        if not 0.0 < float(self.turnover_bucket_quantile) <= 0.5:
            raise ValueError("model_selection.turnover_bucket_quantile must be in (0, 0.5]")
        if not isinstance(self.candidates, tuple):
            raise ValueError("model_selection.candidates must be a tuple of model specs")
        for idx, candidate in enumerate(self.candidates):
            if not isinstance(candidate, ModelSpec):
                raise ValueError(f"model_selection.candidates[{idx}] must be a ModelSpec instance")


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
                "feature_importance.mode must be one of ['disabled', 'latest_only', 'every_fit']"
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
