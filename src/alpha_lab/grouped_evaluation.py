from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from alpha_lab.exceptions import AlphaLabDataError
from alpha_lab.interfaces import validate_factor_output


def compute_ic_by_group(
    factor_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    group_df: pd.DataFrame,
    group_col: str = "sector",
) -> pd.DataFrame:
    """Compute per-date cross-sectional IC by group."""
    validate_factor_output(factor_df)
    validate_factor_output(labels_df)
    _require_columns(group_df, ("date", "asset", group_col), "group_df")

    factor_name = _extract_single_name(factor_df, "factor_df")
    label_name = _extract_single_name(labels_df, "labels_df")
    if group_df.duplicated(subset=["date", "asset"]).any():
        raise AlphaLabDataError("group_df contains duplicate (date, asset) rows")

    merged = (
        factor_df[["date", "asset", "value"]]
        .rename(columns={"value": "factor_value"})
        .merge(
            labels_df[["date", "asset", "value"]].rename(columns={"value": "label_value"}),
            on=["date", "asset"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            group_df[["date", "asset", group_col]],
            on=["date", "asset"],
            how="left",
            validate="many_to_one",
        )
    )
    if merged.empty:
        return pd.DataFrame(columns=["date", "group", "factor", "label", "ic"])

    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged["factor_value"] = pd.to_numeric(merged["factor_value"], errors="coerce")
    merged["label_value"] = pd.to_numeric(merged["label_value"], errors="coerce")
    merged = merged.dropna(subset=[group_col]).copy()
    if merged.empty:
        return pd.DataFrame(columns=["date", "group", "factor", "label", "ic"])

    rows: list[dict[str, object]] = []
    for (date, group), block in merged.groupby(["date", group_col], sort=True, dropna=False):
        valid = block.dropna(subset=["factor_value", "label_value"])
        if len(valid) < 2:
            ic = float("nan")
        else:
            x = valid["factor_value"]
            y = valid["label_value"]
            if x.nunique() < 2 or y.nunique() < 2:
                ic = float("nan")
            else:
                ic = float(x.corr(y, method="pearson"))
        rows.append(
            {
                "date": date,
                "group": group,
                "factor": factor_name,
                "label": label_name,
                "ic": ic,
            }
        )

    return (
        pd.DataFrame(rows, columns=["date", "group", "factor", "label", "ic"])
        .sort_values(
            ["date", "group"],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def compute_ic_by_size_bucket(
    factor_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    market_cap_df: pd.DataFrame,
    n_buckets: int = 3,
) -> pd.DataFrame:
    """Compute IC by per-date market-cap quantile bucket."""
    if n_buckets < 2:
        raise ValueError("n_buckets must be >= 2")
    _require_columns(market_cap_df, ("date", "asset"), "market_cap_df")

    cap_col = _resolve_cap_col(market_cap_df)
    cap = market_cap_df[["date", "asset", cap_col]].copy()
    cap["date"] = pd.to_datetime(cap["date"], errors="coerce")
    cap[cap_col] = pd.to_numeric(cap[cap_col], errors="coerce")

    labels = [f"Q{i + 1}" for i in range(int(n_buckets))]

    def _assign_bucket(series: pd.Series) -> pd.Series:
        out = pd.Series(pd.NA, index=series.index, dtype="object")
        valid = series.notna()
        if int(valid.sum()) < int(n_buckets):
            return out
        ranks = series.loc[valid].rank(method="first")
        out.loc[valid] = pd.qcut(ranks, q=n_buckets, labels=labels, duplicates="drop").astype(str)
        return out

    cap["size_bucket"] = cap.groupby("date", sort=False)[cap_col].transform(_assign_bucket)
    by_size = compute_ic_by_group(factor_df, labels_df, cap, group_col="size_bucket")
    return by_size.rename(columns={"group": "size_bucket"})[
        ["date", "size_bucket", "factor", "label", "ic"]
    ]


def conditional_ic_by_factor_magnitude(
    factor_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    *,
    n_buckets: int = 5,
) -> pd.DataFrame:
    """Summarize IC behavior by per-date |factor| quintile."""
    if n_buckets < 2:
        raise ValueError("n_buckets must be >= 2")

    merged = _merge_factor_and_label_values(factor_df, labels_df)
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "magnitude_quintile",
                "mean_ic",
                "mean_rank_ic",
                "ic_positive_rate",
                "rank_ic_positive_rate",
                "n_dates_used",
                "mean_assets_per_date",
            ]
        )

    labels = [f"Q{i + 1}" for i in range(int(n_buckets))]
    merged["magnitude_quintile"] = merged.groupby("date", sort=False)["factor_abs"].transform(
        lambda series: _assign_quantile_labels(series, labels=labels)
    )

    summary = _summarize_conditional_metric(
        merged=merged,
        group_col="magnitude_quintile",
        group_values=labels,
    )
    return summary[
        [
            "magnitude_quintile",
            "mean_ic",
            "mean_rank_ic",
            "ic_positive_rate",
            "rank_ic_positive_rate",
            "n_dates_used",
            "mean_assets_per_date",
        ]
    ]


def conditional_ic_by_cross_section_size(
    factor_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize IC behavior on dates with small vs large valid cross-sections."""
    merged = _merge_factor_and_label_values(factor_df, labels_df)
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "cross_section_bucket",
                "median_valid_assets_threshold",
                "mean_valid_assets",
                "mean_ic",
                "mean_rank_ic",
                "ic_positive_rate",
                "rank_ic_positive_rate",
                "n_dates_used",
            ]
        )

    valid_counts = (
        merged.groupby("date", sort=True)["asset"].nunique().rename("valid_assets_per_date")
    )
    if valid_counts.empty:
        return pd.DataFrame(
            columns=[
                "cross_section_bucket",
                "median_valid_assets_threshold",
                "mean_valid_assets",
                "mean_ic",
                "mean_rank_ic",
                "ic_positive_rate",
                "rank_ic_positive_rate",
                "n_dates_used",
            ]
        )

    median_threshold = float(valid_counts.median())
    bucket_by_date = valid_counts.reset_index()
    bucket_by_date["cross_section_bucket"] = np.where(
        bucket_by_date["valid_assets_per_date"] < median_threshold,
        "small_cross_section",
        "large_cross_section",
    )
    merged = merged.merge(
        bucket_by_date,
        on="date",
        how="left",
        validate="many_to_one",
    )

    summary = _summarize_conditional_metric(
        merged=merged,
        group_col="cross_section_bucket",
        group_values=("small_cross_section", "large_cross_section"),
        extra_group_stats={
            "median_valid_assets_threshold": lambda block: median_threshold,
            "mean_valid_assets": lambda block: (
                float(block["valid_assets_per_date"].mean()) if not block.empty else float("nan")
            ),
        },
    )
    summary["median_valid_assets_threshold"] = median_threshold
    return summary[
        [
            "cross_section_bucket",
            "median_valid_assets_threshold",
            "mean_valid_assets",
            "mean_ic",
            "mean_rank_ic",
            "ic_positive_rate",
            "rank_ic_positive_rate",
            "n_dates_used",
        ]
    ]


BASIC_CONDITIONAL_BUCKET_COLUMNS: list[str] = ["date", "bucket", "ic", "rank_ic", "n"]


def conditional_ic_by_bucket(
    factor_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    *,
    group_col: str = "bucket",
    bucket_col: str | None = None,
    min_assets: int = 3,
) -> pd.DataFrame:
    """Compute per-date IC by a date-level or asset-level bucket.

    ``bucket_df`` may be ``[date, bucket_col]`` for market-wide regimes or
    ``[date, asset, bucket_col]`` for cross-sectional buckets.
    """
    if min_assets < 2:
        raise ValueError("min_assets must be >= 2")

    resolved_bucket_col = bucket_col if bucket_col is not None else group_col

    merged = _merge_factor_and_label_values(factor_df, labels_df)
    if merged.empty:
        return pd.DataFrame(columns=BASIC_CONDITIONAL_BUCKET_COLUMNS)

    bucket = _normalize_bucket_frame(bucket_df, bucket_col=resolved_bucket_col)
    if bucket.empty:
        return pd.DataFrame(columns=BASIC_CONDITIONAL_BUCKET_COLUMNS)

    join_cols = ["date", "asset"] if "asset" in bucket.columns else ["date"]
    merged = merged.merge(bucket, on=join_cols, how="left", validate="many_to_one")
    merged = merged.dropna(subset=[resolved_bucket_col, "factor_value", "label_value"]).copy()
    if merged.empty:
        return pd.DataFrame(columns=BASIC_CONDITIONAL_BUCKET_COLUMNS)

    merged[resolved_bucket_col] = merged[resolved_bucket_col].astype(str)

    rows: list[dict[str, object]] = []
    for (date, bucket_value), block in merged.groupby(
        ["date", resolved_bucket_col],
        sort=True,
        dropna=False,
    ):
        factor_values = block["factor_value"].to_numpy(dtype=float)
        label_values = block["label_value"].to_numpy(dtype=float)
        if len(block) < int(min_assets):
            ic = float("nan")
            rank_ic = float("nan")
        else:
            ic = _pearson_corr(factor_values, label_values)
            rank_ic = _pearson_corr(
                scipy_stats.rankdata(factor_values, method="average"),
                scipy_stats.rankdata(label_values, method="average"),
            )
        rows.append(
            {
                "date": date,
                "bucket": str(bucket_value),
                "ic": ic,
                "rank_ic": rank_ic,
                "n": int(len(block)),
            }
        )

    return (
        pd.DataFrame(rows, columns=BASIC_CONDITIONAL_BUCKET_COLUMNS)
        .sort_values(["date", "bucket"], kind="mergesort")
        .reset_index(drop=True)
    )


def _extract_single_name(df: pd.DataFrame, name: str) -> str:
    names = pd.unique(df["factor"])
    if len(names) != 1:
        raise AlphaLabDataError(f"{name} must contain exactly one factor name")
    return str(names[0])


def _resolve_cap_col(frame: pd.DataFrame) -> str:
    for col in ("market_cap", "circ_mv", "total_mv", "value"):
        if col in frame.columns:
            return col
    raise AlphaLabDataError(
        "market_cap_df must contain one of: market_cap, circ_mv, total_mv, value"
    )


def _require_columns(frame: pd.DataFrame, cols: tuple[str, ...], name: str) -> None:
    missing = set(cols) - set(frame.columns)
    if missing:
        raise AlphaLabDataError(f"{name} missing required columns: {sorted(missing)}")


def _merge_factor_and_label_values(
    factor_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> pd.DataFrame:
    validate_factor_output(factor_df)
    validate_factor_output(labels_df)
    _extract_single_name(factor_df, "factor_df")
    _extract_single_name(labels_df, "labels_df")

    merged = (
        factor_df[["date", "asset", "factor", "value"]]
        .rename(columns={"factor": "factor_name", "value": "factor_value"})
        .merge(
            labels_df[["date", "asset", "factor", "value"]].rename(
                columns={"factor": "label_name", "value": "label_value"}
            ),
            on=["date", "asset"],
            how="inner",
            validate="one_to_one",
        )
    )
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "asset",
                "factor_name",
                "label_name",
                "factor_value",
                "label_value",
                "factor_abs",
            ]
        )

    merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
    merged["factor_value"] = pd.to_numeric(merged["factor_value"], errors="coerce")
    merged["label_value"] = pd.to_numeric(merged["label_value"], errors="coerce")
    merged = merged.dropna(subset=["date", "factor_value", "label_value"]).copy()
    merged["factor_abs"] = merged["factor_value"].abs()
    return merged


def _normalize_bucket_frame(bucket_df: pd.DataFrame, *, bucket_col: str) -> pd.DataFrame:
    required = (
        ("date", "asset", bucket_col) if "asset" in bucket_df.columns else ("date", bucket_col)
    )
    _require_columns(bucket_df, required, "bucket_df")
    out = bucket_df.loc[:, list(required)].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).copy()
    if "asset" in out.columns:
        out = out.dropna(subset=["asset"]).copy()
        key_cols = ["date", "asset"]
    else:
        key_cols = ["date"]
    if out.duplicated(subset=key_cols).any():
        raise AlphaLabDataError(f"bucket_df contains duplicate {tuple(key_cols)} rows")
    return out


def _assign_quantile_labels(series: pd.Series, *, labels: list[str]) -> pd.Series:
    out = pd.Series(pd.NA, index=series.index, dtype="object")
    valid = series.notna()
    if int(valid.sum()) < len(labels):
        return out
    ranks = series.loc[valid].rank(method="first")
    buckets = pd.qcut(ranks, q=len(labels), labels=labels, duplicates="drop")
    out.loc[valid] = buckets.astype(str)
    return out


def _summarize_conditional_metric(
    *,
    merged: pd.DataFrame,
    group_col: str,
    group_values: list[str] | tuple[str, ...],
    extra_group_stats: dict[str, object] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    extras = extra_group_stats or {}
    grouped_metrics = _conditional_corr_by_date_and_group(merged, group_col=group_col)

    for group_value in group_values:
        group_block = merged.loc[merged[group_col] == group_value]
        group_metric_block = grouped_metrics.loc[grouped_metrics[group_col] == group_value]
        if group_block.empty:
            row: dict[str, object] = {
                group_col: group_value,
                "mean_ic": float("nan"),
                "mean_rank_ic": float("nan"),
                "ic_positive_rate": float("nan"),
                "rank_ic_positive_rate": float("nan"),
                "n_dates_used": 0,
            }
            for key in extras:
                row[key] = float("nan")
            if group_col == "magnitude_quintile":
                row["mean_assets_per_date"] = float("nan")
            rows.append(row)
            continue

        ic_vals = pd.to_numeric(group_metric_block["ic"], errors="coerce").dropna()
        rank_ic_vals = pd.to_numeric(group_metric_block["rank_ic"], errors="coerce").dropna()

        row = {
            group_col: group_value,
            "mean_ic": float(ic_vals.mean()) if len(ic_vals) > 0 else float("nan"),
            "mean_rank_ic": (float(rank_ic_vals.mean()) if len(rank_ic_vals) > 0 else float("nan")),
            "ic_positive_rate": (float((ic_vals > 0).mean()) if len(ic_vals) > 0 else float("nan")),
            "rank_ic_positive_rate": (
                float((rank_ic_vals > 0).mean()) if len(rank_ic_vals) > 0 else float("nan")
            ),
            "n_dates_used": int(
                group_metric_block.loc[group_metric_block["ic"].notna(), "date"].nunique()
            )
            if not group_metric_block.empty
            else 0,
        }
        if group_col == "magnitude_quintile":
            row["mean_assets_per_date"] = float(
                group_block.groupby("date", sort=False)["asset"].nunique().mean()
            )
        for key, func in extras.items():
            row[key] = func(group_block) if callable(func) else func
        rows.append(row)

    return pd.DataFrame(rows)


def _conditional_corr_by_date_and_group(
    merged: pd.DataFrame,
    *,
    group_col: str,
) -> pd.DataFrame:
    values = merged.loc[:, ["date", group_col, "factor_value", "label_value"]].copy()
    values["date"] = pd.to_datetime(values["date"], errors="coerce")
    values["factor_value"] = pd.to_numeric(values["factor_value"], errors="coerce")
    values["label_value"] = pd.to_numeric(values["label_value"], errors="coerce")
    values = values.dropna(subset=["date", group_col, "factor_value", "label_value"])
    if values.empty:
        return pd.DataFrame(columns=[group_col, "date", "ic", "rank_ic"])

    rows: list[dict[str, object]] = []
    for (group_value, date), block in values.groupby([group_col, "date"], sort=True):
        factor_values = block["factor_value"].to_numpy(dtype=float)
        label_values = block["label_value"].to_numpy(dtype=float)
        ic = _pearson_corr(factor_values, label_values)
        if len(factor_values) < 2:
            rank_ic = float("nan")
        else:
            rank_ic = _pearson_corr(
                scipy_stats.rankdata(factor_values, method="average"),
                scipy_stats.rankdata(label_values, method="average"),
            )
        rows.append(
            {
                group_col: group_value,
                "date": date,
                "ic": ic,
                "rank_ic": rank_ic,
            }
        )
    return pd.DataFrame(rows, columns=[group_col, "date", "ic", "rank_ic"])


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_dev = x - x.mean()
    y_dev = y - y.mean()
    denom = float(np.sqrt(np.dot(x_dev, x_dev) * np.dot(y_dev, y_dev)))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(np.dot(x_dev, y_dev) / denom)
