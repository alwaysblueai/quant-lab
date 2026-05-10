"""Model-factor pipeline package.

Public API is re-exported here so existing callers using
``from alpha_lab.real_cases.model_factor.pipeline import ...`` keep working.
The implementation is split across topic modules for maintainability.
"""

from .cache import _resolve_preparation_cache_dir
from .core import (
    ModelFactorCaseRunResult,
    run_model_factor_case,
)
from .features import _coverage_by_date, _load_features
from .labels import _build_forward_label_cache, _model_factor_price_read_columns

__all__ = [
    "ModelFactorCaseRunResult",
    "run_model_factor_case",
    "_resolve_preparation_cache_dir",
    "_coverage_by_date",
    "_load_features",
    "_build_forward_label_cache",
    "_model_factor_price_read_columns",
]
