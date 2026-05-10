from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

# Cross-module imports (auto-added)
from .config import (
    FeatureImportanceConfig,
    _feature_importance_permutation_enabled,
    _feature_importance_permutation_force,
    _feature_importance_permutation_latest_only,
    _feature_importance_permutation_n_repeats,
    _feature_importance_permutation_random_state,
    _feature_importance_permutation_sample_rows,
    _feature_importance_permutation_sample_rows_configured,
    _feature_importance_permutation_top_k_features,
)
from .types import (
    _MODEL_FAMILY_IMPORTANCE_EXTRACTORS,
    _PERMUTATION_IMPORTANCE_MAX_PREDICT_CALLS,
    _PERMUTATION_IMPORTANCE_RANDOM_SEED,
)


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
