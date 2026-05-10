"""Model-factor core package.

Public API is re-exported here so existing callers using
``from alpha_lab.model_factor.core import ...`` keep working.
The implementation is split across topic modules for maintainability.
"""

from ._utils import _indices_as_contiguous_slice
from .build import build_model_factor
from .config import (
    FeatureImportanceConfig,
    FeaturePreprocessConfig,
    ModelFactorBuildConfig,
    ModelFactorBuildResult,
    ModelSelectionSpec,
    ModelSpec,
    TrainingSpec,
    list_model_contracts,
)
from .estimator import (
    _build_estimator,
    _build_model_pipeline,
    _fit_model_bundle,
    _prepare_training_matrix,
    resolve_model_spec_params,
)
from .importance import (
    _feature_importance_extractors_for_family,
    _permutation_importance_guardrail_reason,
)
from .preprocess import _normalize_features
from .types import (
    FEATURE_OOS_IC_COLUMNS,
    TRAINING_METRICS_COLUMNS,
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

__all__ = [
    "ModelFamily",
    "MissingPolicy",
    "ScaleFeatures",
    "WindowType",
    "ModelSelectionMetric",
    "FeatureImportanceMode",
    "FeatureImportanceMethod",
    "CrossSectionalTransform",
    "CrossSectionalGroupScope",
    "TRAINING_METRICS_COLUMNS",
    "FEATURE_OOS_IC_COLUMNS",
    "FeaturePreprocessConfig",
    "ModelSpec",
    "ModelSelectionSpec",
    "TrainingSpec",
    "FeatureImportanceConfig",
    "ModelFactorBuildConfig",
    "ModelFactorBuildResult",
    "list_model_contracts",
    "build_model_factor",
    "resolve_model_spec_params",
    "_build_estimator",
    "_build_model_pipeline",
    "_prepare_training_matrix",
    "_fit_model_bundle",
    "_feature_importance_extractors_for_family",
    "_permutation_importance_guardrail_reason",
    "_normalize_features",
    "_indices_as_contiguous_slice",
]
