from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from alpha_lab.splits import TimeSeriesSplitContract


def _with_split_phase(
    frame: pd.DataFrame,
    contract: TimeSeriesSplitContract | None,
    *,
    drop_embargo: bool,
) -> pd.DataFrame:
    """Annotate report rows with IS/OOS phase from the strict split contract."""

    if contract is None or frame.empty or "date" not in frame.columns:
        return frame.copy()

    out = frame.copy()
    dates = pd.to_datetime(out["date"], errors="coerce")
    is_end = pd.Timestamp(contract.is_end)
    oos_start = pd.Timestamp(contract.oos_start)
    out["split_phase"] = np.select(
        [dates <= is_end, dates >= oos_start],
        ["IS", "OOS"],
        default="EMBARGO",
    )
    if drop_embargo:
        out = out.loc[out["split_phase"] != "EMBARGO"]
    return out.reset_index(drop=True)


def _build_effective_coverage_by_date(
    *,
    prices: pd.DataFrame,
    factor_df: pd.DataFrame,
    label_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute coverage against the eligible universe, not the scored subset."""

    columns = [
        "date",
        "eligible_count",
        "valid_score_count",
        "valid_forward_return_count",
        "valid_sample_count",
        "asset_coverage",
        "forward_return_coverage",
        "sample_coverage",
        "coverage",
        "missingness",
        "n_assets",
        "n_non_null",
        "missing_score_count",
        "missing_forward_return_count",
        "invalid_sample_count",
        "universe_count",
        "feature_row_count",
        "complete_feature_count",
        "feature_nan_row_count",
        "label_available_count",
        "scored_count",
        "scored_evaluable_count",
        "missing_feature_count",
        "missing_label_count",
        "filtered_count",
        "score_coverage",
        "universe_coverage",
    ]
    if prices.empty or not {"date", "asset"}.issubset(prices.columns):
        return pd.DataFrame(columns=columns)

    eligible = prices[["date", "asset"]].copy()
    eligible["date"] = pd.to_datetime(eligible["date"], errors="coerce")
    eligible = eligible.dropna(subset=["date", "asset"]).drop_duplicates()
    if eligible.empty:
        return pd.DataFrame(columns=columns)
    eligible["date"] = eligible["date"].dt.normalize()
    eligible["asset"] = eligible["asset"].astype(str)

    def _valid_flags(frame: pd.DataFrame, flag_col: str) -> pd.DataFrame:
        if frame.empty or not {"date", "asset", "value"}.issubset(frame.columns):
            return pd.DataFrame(columns=["date", "asset", flag_col])
        out = frame[["date", "asset", "value"]].copy()
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date", "asset"])
        if out.empty:
            return pd.DataFrame(columns=["date", "asset", flag_col])
        out["date"] = out["date"].dt.normalize()
        out["asset"] = out["asset"].astype(str)
        out[flag_col] = pd.to_numeric(out["value"], errors="coerce").notna()
        return out.groupby(["date", "asset"], as_index=False)[flag_col].max().reset_index(drop=True)

    score_flags = _valid_flags(factor_df, "has_valid_score")
    label_flags = _valid_flags(label_df, "has_valid_forward_return")
    merged = eligible.merge(score_flags, on=["date", "asset"], how="left")
    merged = merged.merge(label_flags, on=["date", "asset"], how="left")
    merged["has_valid_score"] = merged["has_valid_score"].fillna(False).astype(bool)
    merged["has_valid_forward_return"] = (
        merged["has_valid_forward_return"].fillna(False).astype(bool)
    )
    merged["has_valid_sample"] = merged["has_valid_score"] & merged["has_valid_forward_return"]

    summary = merged.groupby("date", sort=True).agg(
        eligible_count=("asset", "nunique"),
        valid_score_count=("has_valid_score", "sum"),
        valid_forward_return_count=("has_valid_forward_return", "sum"),
        valid_sample_count=("has_valid_sample", "sum"),
    )
    count_cols = [
        "eligible_count",
        "valid_score_count",
        "valid_forward_return_count",
        "valid_sample_count",
    ]
    for col in count_cols:
        summary[col] = summary[col].astype(int)
    denominator = summary["eligible_count"].replace(0, pd.NA)
    summary["asset_coverage"] = summary["valid_score_count"] / denominator
    summary["forward_return_coverage"] = summary["valid_forward_return_count"] / denominator
    summary["sample_coverage"] = summary["valid_sample_count"] / denominator
    summary["coverage"] = summary["asset_coverage"]
    summary["missingness"] = 1.0 - summary["asset_coverage"]
    summary["n_assets"] = summary["eligible_count"]
    summary["n_non_null"] = summary["valid_score_count"]
    summary["missing_score_count"] = summary["eligible_count"] - summary["valid_score_count"]
    summary["missing_forward_return_count"] = (
        summary["eligible_count"] - summary["valid_forward_return_count"]
    )
    summary["invalid_sample_count"] = summary["eligible_count"] - summary["valid_sample_count"]
    summary["universe_count"] = summary["eligible_count"]
    summary["feature_row_count"] = summary["eligible_count"]
    summary["complete_feature_count"] = summary["eligible_count"]
    summary["feature_nan_row_count"] = 0
    summary["label_available_count"] = summary["valid_forward_return_count"]
    summary["scored_count"] = summary["valid_score_count"]
    summary["scored_evaluable_count"] = summary["valid_sample_count"]
    summary["missing_feature_count"] = 0
    summary["missing_label_count"] = summary["missing_forward_return_count"]
    summary["filtered_count"] = summary["missing_forward_return_count"]
    summary["score_coverage"] = summary["asset_coverage"]
    summary["universe_coverage"] = summary["asset_coverage"]
    return summary.reset_index()[columns]


def _merge_supplied_coverage_details(
    effective: pd.DataFrame,
    supplied: pd.DataFrame | None,
) -> pd.DataFrame:
    if effective.empty or supplied is None or supplied.empty or "date" not in supplied.columns:
        return effective
    detail_columns = [
        "universe_count",
        "feature_row_count",
        "complete_feature_count",
        "feature_nan_row_count",
        "label_available_count",
        "scored_count",
        "scored_evaluable_count",
        "missing_feature_count",
        "missing_label_count",
        "filtered_count",
        "score_coverage",
        "universe_coverage",
    ]
    available = [column for column in detail_columns if column in supplied.columns]
    if not available:
        return effective

    left = effective.copy()
    left["date"] = pd.to_datetime(left["date"], errors="coerce")
    right = supplied.loc[:, ["date", *available]].copy()
    right["date"] = pd.to_datetime(right["date"], errors="coerce")
    right = right.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    merged = left.merge(
        right,
        on="date",
        how="left",
        suffixes=("", "_supplied"),
        validate="one_to_one",
    )
    for column in available:
        supplied_column = f"{column}_supplied"
        if supplied_column not in merged.columns:
            continue
        supplied_values = pd.to_numeric(merged[supplied_column], errors="coerce")
        base_values = pd.to_numeric(merged[column], errors="coerce")
        merged[column] = supplied_values.combine_first(base_values)
        merged = merged.drop(columns=[supplied_column])
    return merged


def _annotate_coverage_warmup(coverage_by_date: pd.DataFrame) -> pd.DataFrame:
    """Mark leading no-score dates as warmup-only diagnostics.

    Model-factor runs can legitimately emit zero scored assets before the first
    walk-forward model is fitted. Those leading dates should stay visible in
    coverage.csv, but they should not drive promotion coverage blockers.
    """

    out = coverage_by_date.copy()
    if out.empty:
        out["is_warmup"] = pd.Series(dtype=bool)
        out["coverage_eval_included"] = pd.Series(dtype=bool)
        return out
    if "date" in out.columns:
        out["_coverage_date_sort"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.sort_values("_coverage_date_sort", kind="mergesort").drop(
            columns=["_coverage_date_sort"]
        )

    score_count = pd.Series(0.0, index=out.index, dtype=float)
    for column in ("valid_score_count", "scored_count", "n_non_null"):
        if column in out.columns:
            values = pd.to_numeric(out[column], errors="coerce")
            score_count = values.combine_first(score_count)
            break

    has_score = score_count.fillna(0.0) > 0
    if bool(has_score.any()):
        first_scored_position = int(np.flatnonzero(has_score.to_numpy())[0])
        warmup_mask = pd.Series(False, index=out.index)
        if first_scored_position > 0:
            warmup_mask.iloc[:first_scored_position] = True
    else:
        warmup_mask = pd.Series(False, index=out.index)

    out["is_warmup"] = warmup_mask.astype(bool)
    out["coverage_eval_included"] = ~out["is_warmup"]
    return out


def _coverage_decision_frame(coverage_by_date: pd.DataFrame) -> pd.DataFrame:
    if coverage_by_date.empty or "coverage_eval_included" not in coverage_by_date.columns:
        return coverage_by_date
    included = coverage_by_date["coverage_eval_included"].fillna(True).astype(bool)
    return coverage_by_date.loc[included].reset_index(drop=True)


def _coverage_warmup_summary(coverage_by_date: pd.DataFrame) -> dict[str, object]:
    if coverage_by_date.empty or "is_warmup" not in coverage_by_date.columns:
        return {
            "coverage_warmup_excluded_days": 0,
            "coverage_warmup_start": None,
            "coverage_warmup_end": None,
        }
    warmup = coverage_by_date.loc[coverage_by_date["is_warmup"].fillna(False).astype(bool)]
    if warmup.empty:
        return {
            "coverage_warmup_excluded_days": 0,
            "coverage_warmup_start": None,
            "coverage_warmup_end": None,
        }
    dates = (
        pd.to_datetime(warmup["date"], errors="coerce")
        if "date" in warmup.columns
        else pd.Series(dtype="datetime64[ns]")
    )
    dates = dates.dropna()
    return {
        "coverage_warmup_excluded_days": int(len(warmup)),
        "coverage_warmup_start": dates.min().strftime("%Y-%m-%d") if not dates.empty else None,
        "coverage_warmup_end": dates.max().strftime("%Y-%m-%d") if not dates.empty else None,
    }


def _summarise_effective_coverage(coverage_by_date: pd.DataFrame) -> dict[str, float | int]:
    if coverage_by_date.empty:
        return {
            "n_dates": 0,
            "n_valid_dates": 0,
            "date_coverage": float("nan"),
            "mean_asset_coverage": float("nan"),
            "median_asset_coverage": float("nan"),
            "min_asset_coverage": float("nan"),
            "max_asset_coverage": float("nan"),
            "overall_sample_coverage": float("nan"),
            "avg_assets": float("nan"),
            "avg_valid_score_assets": float("nan"),
            "avg_valid_forward_return_assets": float("nan"),
        }

    def _numeric(col: str) -> pd.Series:
        if col not in coverage_by_date.columns:
            return pd.Series(dtype=float)
        return pd.to_numeric(coverage_by_date[col], errors="coerce")

    eligible = _numeric("eligible_count")
    valid_scores = _numeric("valid_score_count")
    valid_labels = _numeric("valid_forward_return_count")
    valid_samples = _numeric("valid_sample_count")
    asset_coverage = _numeric("asset_coverage").replace([np.inf, -np.inf], np.nan)
    n_dates = (
        int(coverage_by_date["date"].nunique())
        if "date" in coverage_by_date.columns
        else int(len(coverage_by_date))
    )
    n_valid_dates = int((valid_samples.fillna(0) > 0).sum())
    total_eligible = float(eligible.sum())
    total_valid_samples = float(valid_samples.sum())
    return {
        "n_dates": n_dates,
        "n_valid_dates": n_valid_dates,
        "date_coverage": float(n_valid_dates / n_dates) if n_dates else float("nan"),
        "mean_asset_coverage": float(asset_coverage.mean()),
        "median_asset_coverage": float(asset_coverage.median()),
        "min_asset_coverage": float(asset_coverage.min()),
        "max_asset_coverage": float(asset_coverage.max()),
        "overall_sample_coverage": (
            float(total_valid_samples / total_eligible) if total_eligible > 0 else float("nan")
        ),
        "avg_assets": float(eligible.mean()),
        "avg_valid_score_assets": float(valid_scores.mean()),
        "avg_valid_forward_return_assets": float(valid_labels.mean()),
    }


def _count_coverage_break_days(
    coverage_by_date: pd.DataFrame,
    *,
    threshold: float,
    drop_threshold: float,
) -> int:
    if coverage_by_date.empty or "coverage" not in coverage_by_date.columns:
        return 0
    coverage = pd.to_numeric(coverage_by_date["coverage"], errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )
    low = coverage < float(threshold)
    abrupt_drop = coverage.diff() <= -abs(float(drop_threshold))
    return int((low | abrupt_drop).fillna(False).sum())


def _build_coverage_summary(
    *,
    coverage_stats: Mapping[str, float | int],
    warmup_summary: Mapping[str, object] | None = None,
) -> str:
    def _as_float(key: str) -> float:
        value = coverage_stats.get(key)
        if value is None:
            return float("nan")
        return float(value)

    n_dates = int(coverage_stats.get("n_dates") or 0)
    avg_assets = _as_float("avg_assets")
    avg_asset_coverage = _as_float("mean_asset_coverage")
    overall_sample_coverage = _as_float("overall_sample_coverage")
    if not np.isfinite(avg_assets) or not np.isfinite(avg_asset_coverage):
        return f"n_dates={n_dates}"
    summary = (
        f"n_dates={n_dates}; avg_assets={avg_assets:.1f}; "
        f"asset_coverage={avg_asset_coverage:.1%}; "
        f"sample_coverage={overall_sample_coverage:.1%}"
    )
    raw_warmup_days = (warmup_summary or {}).get("coverage_warmup_excluded_days") or 0
    warmup_days = int(raw_warmup_days) if isinstance(raw_warmup_days, int | float) else 0
    if warmup_days > 0:
        summary = f"{summary}; warmup_excluded_days={warmup_days}"
    return summary
