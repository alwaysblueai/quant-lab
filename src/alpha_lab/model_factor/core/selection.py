from __future__ import annotations

import numpy as np
import pandas as pd

from alpha_lab.validation.purged_kfold import purged_kfold_split

# Cross-module imports (auto-added)
from ._utils import _finite_or_none, _object_to_float, _object_to_int
from .config import ModelFactorBuildConfig, ModelSpec
from .estimator import _fit_model_bundle, _prepare_training_matrix
from .internals import _FittedModelBundle, _ModelSelectionOutcome, _PreparedModelArrays
from .training_arrays import _row_selection_to_indices, _training_matrix_from_selection
from .types import (
    FEATURE_OOS_IC_COLUMNS,
    TRAINING_METRICS_COLUMNS,
    ModelSelectionMetric,
    _RowSelection,
)


def _selection_candidates(config: ModelFactorBuildConfig) -> tuple[ModelSpec, ...]:
    if config.model_selection.candidates:
        return config.model_selection.candidates
    return (config.model,)


def _selection_has_mlp_early_stopping(config: ModelFactorBuildConfig) -> bool:
    return any(
        _mlp_early_stopping_enabled(candidate) for candidate in _selection_candidates(config)
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
        if group["prediction"].nunique(dropna=True) < 2 or group["label"].nunique(dropna=True) < 2:
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
        "n_oos_obs": _object_to_int(left.get("n_oos_obs")) + _object_to_int(right.get("n_oos_obs")),
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
