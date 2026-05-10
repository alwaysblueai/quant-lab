from __future__ import annotations

from typing import Literal

import numpy as np

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
