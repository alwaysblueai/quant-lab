from __future__ import annotations

from collections import Counter

import pandas as pd

from alpha_lab.research_integrity.contracts import IntegrityCheckResult

# Cross-module imports (auto-added)
from ._utils import _finite_or_none
from .config import (
    ModelFactorBuildConfig,
    _feature_importance_enabled,
    _feature_importance_over_time_enabled,
    _feature_importance_over_time_source,
    _feature_importance_over_time_top_k,
    _feature_importance_permutation_enabled,
    _feature_importance_permutation_force,
    _feature_importance_permutation_latest_only,
    _feature_importance_permutation_n_repeats,
    _feature_importance_permutation_random_state,
    _feature_importance_permutation_sample_rows,
    _feature_importance_permutation_top_k_features,
)
from .selection import _selection_candidates


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
                "enabled": bool(_feature_importance_over_time_enabled(config.feature_importance)),
                "top_k": int(_feature_importance_over_time_top_k(config.feature_importance)),
                "source": _feature_importance_over_time_source(config.feature_importance),
            },
            "permutation": {
                "enabled": bool(_feature_importance_permutation_enabled(config.feature_importance)),
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
                "force": bool(_feature_importance_permutation_force(config.feature_importance)),
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
