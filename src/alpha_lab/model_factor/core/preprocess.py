from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from alpha_lab.research_contracts import validate_prices_table

# Cross-module imports (auto-added)
from ._utils import _sample_float_or_none
from .config import ModelFactorBuildConfig
from .types import _MODEL_FEATURE_DTYPE, CrossSectionalGroupScope, CrossSectionalTransform


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
    summary["filtered_count"] = (summary["universe_count"] - summary["eligible_count"]).clip(
        lower=0
    )
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


def _prices_for_target_labels(
    prices: pd.DataFrame,
    *,
    price_column: str,
    execution_price_mode: str = "close",
) -> pd.DataFrame:
    column = str(price_column or "").strip()
    if not column:
        raise ValueError("target_price_column must be non-empty")
    if column not in prices.columns:
        raise ValueError(f"target price column {column!r} is missing from prices")
    mode = str(execution_price_mode or "close").strip().lower()
    if mode not in {"close", "next_open"}:
        raise ValueError("target_execution_price_mode must be one of ['close', 'next_open']")
    required = ["date", "asset", column]
    if mode == "next_open":
        if "open" not in prices.columns:
            raise ValueError(
                "target_execution_price_mode='next_open' requires an 'open' column in prices"
            )
        required.append("open")
    frame = prices.loc[:, required].copy()
    if column != "close":
        frame = frame.rename(columns={column: "close"})
    out_columns = ["date", "asset", "close"]
    if mode == "next_open":
        out_columns.append("open")
    return frame.loc[:, out_columns]


def _apply_forward_return_extreme_filter(
    forward_label_df: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    label_prices: pd.DataFrame,
    target_price_column: str,
    target_execution_price_mode: str,
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
        "target_execution_price_mode": str(target_execution_price_mode),
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
        execution_price_mode=target_execution_price_mode,
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
    execution_price_mode: str,
    horizon: int,
    limit: int,
) -> list[dict[str, object]]:
    if outlier_rows.empty:
        return []
    price_frame = label_prices.loc[:, ["date", "asset", "close"]].copy()
    mode = str(execution_price_mode or "close").strip().lower()
    if mode == "next_open" and "open" in label_prices.columns:
        price_frame["open"] = label_prices["open"]
    price_frame["date"] = pd.to_datetime(price_frame["date"])
    price_frame = price_frame.sort_values(["asset", "date"], kind="mergesort")
    close_price = pd.to_numeric(price_frame["close"], errors="coerce")
    if mode == "next_open" and "open" in price_frame.columns:
        open_price = pd.to_numeric(price_frame["open"], errors="coerce")
        price_frame["entry_price"] = open_price.groupby(price_frame["asset"], sort=False).shift(-1)
    else:
        price_frame["entry_price"] = close_price
    price_frame["exit_price"] = close_price.groupby(price_frame["asset"], sort=False).shift(
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


def _target_diagnostics_from_data_health(data_health: dict[str, object]) -> dict[str, object]:
    keys = (
        "target_price_column",
        "target_execution_price_mode",
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
