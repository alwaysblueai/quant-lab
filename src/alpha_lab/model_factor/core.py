from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabDataError
from alpha_lab.interfaces import validate_factor_output
from alpha_lab.labels import forward_return
from alpha_lab.research_contracts import validate_prices_table
from alpha_lab.research_integrity.contracts import IntegrityCheckResult
from alpha_lab.research_integrity.leakage_checks import check_no_future_dates_in_input

ModelFamily = Literal["linear", "ridge", "lasso", "elastic_net", "gbdt", "mlp"]
MissingPolicy = Literal["median_impute"]
ScaleFeatures = Literal["auto", "standard", "none"]
WindowType = Literal["rolling", "expanding"]

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


@dataclass(frozen=True)
class FeaturePreprocessConfig:
    """Training-time feature preprocessing controls."""

    missing_policy: MissingPolicy = "median_impute"
    scale_features: ScaleFeatures = "auto"

    def __post_init__(self) -> None:
        if self.missing_policy != "median_impute":
            raise ValueError(
                "feature_preprocess.missing_policy currently supports only "
                "'median_impute'"
            )
        if self.scale_features not in {"auto", "standard", "none"}:
            raise ValueError(
                "feature_preprocess.scale_features must be one of ['auto', 'standard', 'none']"
            )


@dataclass(frozen=True)
class ModelSpec:
    """Estimator family and fixed hyperparameters."""

    family: ModelFamily = "ridge"
    params: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in {"linear", "ridge", "lasso", "elastic_net", "gbdt", "mlp"}:
            raise ValueError(
                "model.family must be one of ['linear', 'ridge', 'lasso', "
                "'elastic_net', 'gbdt', 'mlp']"
            )
        if not isinstance(self.params, dict):
            raise ValueError("model.params must be an object")
        for key in self.params:
            if not isinstance(key, str) or not key.strip():
                raise ValueError("model.params keys must be non-empty strings")


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
class ModelFactorBuildConfig:
    """Configuration for converting research features into a factor table."""

    factor_name: str
    feature_columns: tuple[str, ...]
    target_horizon: int
    feature_preprocess: FeaturePreprocessConfig = FeaturePreprocessConfig()
    model: ModelSpec = ModelSpec()
    training: TrainingSpec = TrainingSpec()
    known_at_col: str | None = None

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
                raise ValueError(
                    f"feature_columns may not include reserved column {column!r}"
                )
            if column in seen:
                raise ValueError(f"feature_columns must be unique; duplicate: {column!r}")
            seen.add(column)
        if self.target_horizon <= 0:
            raise ValueError("target_horizon must be > 0")
        if self.known_at_col is not None and not self.known_at_col.strip():
            raise ValueError("known_at_col must be non-empty when provided")


@dataclass(frozen=True)
class ModelFactorBuildResult:
    """Canonical factor output plus training diagnostics."""

    factor_df: pd.DataFrame
    training_log_df: pd.DataFrame
    feature_importance_df: pd.DataFrame
    model_diagnostics: dict[str, object]
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


def build_model_factor(
    features_df: pd.DataFrame,
    prices_df: pd.DataFrame,
    config: ModelFactorBuildConfig,
) -> ModelFactorBuildResult:
    """Train walk-forward models and emit canonical `[date, asset, factor, value]` output."""

    prices = _normalize_prices(prices_df)
    features = _normalize_features(features_df, config=config)

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

    label_df = forward_return(prices, horizon=config.target_horizon)
    label_name = f"forward_return_{config.target_horizon}"
    labels = label_df[label_df["factor"] == label_name][["date", "asset", "value"]].rename(
        columns={"value": "label"}
    )

    merged = features.merge(labels, on=["date", "asset"], how="left", validate="one_to_one")
    score_dates = list(
        pd.Index(features["date"].drop_duplicates()).sort_values().to_pydatetime().tolist()
    )

    factor_rows: list[dict[str, object]] = []
    training_log_rows: list[dict[str, object]] = []
    per_fit_importance_frames: list[pd.DataFrame] = []
    current_bundle: _FittedModelBundle | None = None
    last_fit_score_idx: int | None = None
    model_version = 0

    for score_idx, raw_score_date in enumerate(score_dates):
        score_date = pd.Timestamp(raw_score_date)
        score_slice = features[features["date"] == score_date].copy()
        n_score_assets = int(score_slice["asset"].nunique())
        train_slice = _training_slice(merged, score_date=score_date, training=config.training)
        train_dates = pd.Index(train_slice["date"].drop_duplicates()).sort_values()
        n_train_dates = int(len(train_dates))
        train_labeled = train_slice[train_slice["label"].notna()].copy()
        n_train_rows = int(len(train_labeled))

        status = "reused_scored"
        skip_reason: str | None = None
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
            model_version += 1
            fitted = _fit_model_bundle(
                train_slice=train_labeled,
                config=config,
                model_version=model_version,
            )
            current_bundle = fitted
            last_fit_score_idx = score_idx
            status = "fit_scored"
            per_fit_importance_frames.append(
                _feature_importance_frame(
                    fitted.pipeline,
                    config=config,
                    model_version=fitted.model_version,
                    trained_until=fitted.train_end,
                )
            )
        elif current_bundle is None:
            status = "skipped"
            skip_reason = "model_not_ready"

        if current_bundle is not None and status != "skipped":
            score_features = score_slice.loc[:, list(config.feature_columns)]
            predictions = current_bundle.pipeline.predict(score_features)
            for asset, value in zip(score_slice["asset"], predictions, strict=True):
                factor_rows.append(
                    {
                        "date": score_date,
                        "asset": str(asset),
                        "factor": config.factor_name,
                        "value": float(value),
                    }
                )
        elif skip_reason is None and current_bundle is None:
            skip_reason = "model_not_ready"

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
                "model_family": config.model.family,
                "scale_mode": current_bundle.scale_mode if current_bundle is not None else "N/A",
            }
        )

    factor_df = pd.DataFrame(factor_rows, columns=["date", "asset", "factor", "value"])
    if factor_df.empty:
        raise ValueError("model factor build produced no scored rows")
    factor_df = factor_df.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)
    validate_factor_output(factor_df)

    training_log_df = pd.DataFrame(
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
        ],
    ).sort_values("score_date", kind="mergesort").reset_index(drop=True)

    feature_importance_df = _combine_feature_importance_frames(
        per_fit_importance_frames,
        feature_columns=config.feature_columns,
    )
    model_diagnostics = _build_model_diagnostics(
        config=config,
        training_log_df=training_log_df,
        feature_importance_df=feature_importance_df,
    )

    return ModelFactorBuildResult(
        factor_df=factor_df,
        training_log_df=training_log_df,
        feature_importance_df=feature_importance_df,
        model_diagnostics=model_diagnostics,
        integrity_checks=tuple(integrity_checks),
    )


def _normalize_prices(prices_df: pd.DataFrame) -> pd.DataFrame:
    prices = prices_df.copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices = prices.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
    validate_prices_table(prices)
    return prices


def _normalize_features(
    features_df: pd.DataFrame,
    *,
    config: ModelFactorBuildConfig,
) -> pd.DataFrame:
    required = {"date", "asset", *config.feature_columns}
    if config.known_at_col is not None:
        required.add(config.known_at_col)
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
    frame = features_df.loc[:, selected_columns].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    if frame["date"].isna().any():
        raise ValueError("features_df.date contains invalid timestamps")
    if frame["asset"].isna().any() or (frame["asset"].astype(str).str.strip() == "").any():
        raise ValueError("features_df.asset contains null or empty values")
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
    return frame.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def _training_slice(
    merged: pd.DataFrame,
    *,
    score_date: pd.Timestamp,
    training: TrainingSpec,
) -> pd.DataFrame:
    history = merged[merged["date"] < score_date].copy()
    if history.empty:
        return history
    if training.window_type == "expanding":
        return history

    dates = pd.Index(history["date"].drop_duplicates()).sort_values()
    tail_dates = dates[-int(training.train_window_n_dates or 0) :]
    return history[history["date"].isin(tail_dates)].copy()


def _fit_model_bundle(
    *,
    train_slice: pd.DataFrame,
    config: ModelFactorBuildConfig,
    model_version: int,
) -> _FittedModelBundle:
    feature_columns = list(config.feature_columns)
    estimator = _build_estimator(config.model)
    scale_mode = _resolve_scale_mode(config.feature_preprocess.scale_features, config.model.family)
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
    pipeline.fit(train_slice.loc[:, feature_columns], train_slice["label"])
    train_dates = pd.Index(train_slice["date"].drop_duplicates()).sort_values()
    return _FittedModelBundle(
        pipeline=pipeline,
        model_version=model_version,
        train_start=pd.Timestamp(train_dates.min()),
        train_end=pd.Timestamp(train_dates.max()),
        n_train_dates=int(len(train_dates)),
        n_train_rows=int(len(train_slice)),
        scale_mode=scale_mode,
    )


def _build_estimator(spec: ModelSpec) -> object:
    params = dict(spec.params)
    family = spec.family
    if family == "linear":
        return LinearRegression(**params)
    if family == "ridge":
        return Ridge(**params)
    if family == "lasso":
        params.setdefault("random_state", 0)
        return Lasso(**params)
    if family == "elastic_net":
        params.setdefault("random_state", 0)
        return ElasticNet(**params)
    if family == "gbdt":
        params.setdefault("random_state", 0)
        return HistGradientBoostingRegressor(**params)
    params.setdefault("random_state", 0)
    params.setdefault("max_iter", 200)
    return MLPRegressor(**params)


def _resolve_scale_mode(scale_features: ScaleFeatures, model_family: ModelFamily) -> str:
    if scale_features == "standard":
        return "standard"
    if scale_features == "none":
        return "none"
    if model_family in {"linear", "ridge", "lasso", "elastic_net", "mlp"}:
        return "standard"
    return "none"


def _feature_importance_frame(
    pipeline: Pipeline,
    *,
    config: ModelFactorBuildConfig,
    model_version: int,
    trained_until: pd.Timestamp,
) -> pd.DataFrame:
    estimator = pipeline.named_steps["model"]
    importance_source = "unsupported"
    signed: np.ndarray | None = None

    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float).reshape(-1)  # type: ignore[attr-defined]
        signed = coef
        importance_source = "coef"
    elif hasattr(estimator, "feature_importances_"):
        signed = np.asarray(
            estimator.feature_importances_,  # type: ignore[attr-defined]
            dtype=float,
        ).reshape(-1)
        importance_source = "feature_importances"

    if signed is None or signed.size != len(config.feature_columns):
        signed = np.full(len(config.feature_columns), np.nan, dtype=float)

    frame = pd.DataFrame(
        {
            "model_version": model_version,
            "trained_until": trained_until,
            "feature": list(config.feature_columns),
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
) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(
            {
                "feature": list(feature_columns),
                "mean_abs_importance": [float("nan")] * len(feature_columns),
                "latest_importance": [float("nan")] * len(feature_columns),
                "importance_source": ["unsupported"] * len(feature_columns),
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
) -> dict[str, object]:
    trained_rows = training_log_df[training_log_df["status"] != "skipped"].copy()
    skip_counts = Counter(
        str(value).strip()
        for value in training_log_df["skip_reason"]
        if isinstance(value, str) and value.strip()
    )
    return {
        "factor_name": config.factor_name,
        "model_family": config.model.family,
        "feature_columns": list(config.feature_columns),
        "feature_count": len(config.feature_columns),
        "target_horizon": config.target_horizon,
        "trained_model_versions": int(
            training_log_df["model_version"].dropna().nunique()
        ),
        "n_score_dates_total": int(len(training_log_df)),
        "n_score_dates_scored": int(len(trained_rows)),
        "mean_train_rows": _finite_or_none(
            trained_rows["n_train_rows"].mean() if not trained_rows.empty else float("nan")
        ),
        "mean_score_assets": _finite_or_none(
            trained_rows["n_score_assets"].mean() if not trained_rows.empty else float("nan")
        ),
        "skip_reason_counts": dict(skip_counts),
        "top_features": (
            feature_importance_df.sort_values(
                ["mean_abs_importance", "feature"],
                ascending=[False, True],
                kind="mergesort",
            )
            .head(5)["feature"]
            .tolist()
        ),
    }


def _raise_on_integrity_failures(checks: list[IntegrityCheckResult]) -> None:
    failures = [check for check in checks if check.status == "fail"]
    if failures:
        first = failures[0]
        raise ValueError(first.message)


def _finite_or_none(value: float) -> float | None:
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
