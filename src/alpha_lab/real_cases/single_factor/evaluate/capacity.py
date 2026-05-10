from __future__ import annotations

import numpy as np
import pandas as pd

# Cross-module imports (auto-added by split)
from ._utils import _finite_or_nan, _parse_optional_float


def _build_capacity_estimation(
    *,
    prices: pd.DataFrame,
    labels_df: pd.DataFrame,
    quantile_assignments_df: pd.DataFrame,
    long_short_df: pd.DataFrame,
    mean_long_short_turnover: float,
    n_quantiles: int,
    rebalance_step: int,
    enabled: bool,
    participation_rate: float,
    adv_lookback: int,
) -> pd.DataFrame:
    summary = _empty_capacity_summary()
    summary["capacity_enabled"] = bool(enabled)
    summary["capacity_participation_rate"] = float(participation_rate)
    summary["capacity_adv_lookback"] = int(adv_lookback)
    if not enabled:
        summary["capacity_status"] = "disabled"
        summary["capacity_notes"] = "capacity estimation disabled by case spec"
        return pd.DataFrame([summary])

    cap_col = _resolve_market_cap_col(prices)
    has_amount = "amount" in prices.columns
    if cap_col is None and not has_amount:
        summary["capacity_status"] = "unavailable"
        summary["capacity_notes"] = "missing explicit market-cap column and amount column"
        return pd.DataFrame([summary])

    summary["equal_weight_mean_long_short_return"] = _finite_or_nan(
        _parse_optional_float(
            long_short_df["long_short_return"].mean() if not long_short_df.empty else float("nan")
        )
    )

    if cap_col is not None:
        weighted_mean = _compute_market_cap_weighted_long_short_return(
            prices=prices,
            labels_df=labels_df,
            quantile_assignments_df=quantile_assignments_df,
            n_quantiles=n_quantiles,
            cap_col=cap_col,
        )
        summary["capacity_market_cap_column"] = cap_col
        summary["market_cap_weighted_mean_long_short_return"] = weighted_mean
        equal_weight_mean = _parse_optional_float(
            summary.get("equal_weight_mean_long_short_return")
        )
        if np.isfinite(weighted_mean) and equal_weight_mean is not None:
            summary["market_cap_vs_equal_weight_return_delta"] = float(
                weighted_mean - equal_weight_mean
            )

    if has_amount:
        mean_traded_adv = _compute_mean_traded_adv(
            prices=prices,
            quantile_assignments_df=quantile_assignments_df,
            n_quantiles=n_quantiles,
            rebalance_step=rebalance_step,
            adv_lookback=adv_lookback,
        )
        summary["mean_traded_adv"] = mean_traded_adv

    turnover = _parse_optional_float(mean_long_short_turnover)
    traded_adv = _parse_optional_float(summary.get("mean_traded_adv"))
    if traded_adv is not None and turnover is not None and turnover > 0.0:
        summary["estimated_capacity_upper_bound"] = float(
            traded_adv * float(participation_rate) / turnover
        )

    unavailable_parts: list[str] = []
    if cap_col is None:
        unavailable_parts.append("missing explicit market-cap column")
    if not has_amount:
        unavailable_parts.append("missing amount column for ADV")
    if _parse_optional_float(summary.get("estimated_capacity_upper_bound")) is None:
        if traded_adv is None:
            unavailable_parts.append("mean traded ADV unavailable")
        if turnover is None or turnover <= 0.0:
            unavailable_parts.append("turnover unavailable for capacity upper bound")

    if unavailable_parts:
        if cap_col is not None or has_amount:
            summary["capacity_status"] = "partial"
        else:
            summary["capacity_status"] = "unavailable"
        summary["capacity_notes"] = "; ".join(unavailable_parts)
    else:
        summary["capacity_status"] = "available"
        summary["capacity_notes"] = "capacity diagnostics available"

    return pd.DataFrame([summary])


def _compute_market_cap_weighted_long_short_return(
    *,
    prices: pd.DataFrame,
    labels_df: pd.DataFrame,
    quantile_assignments_df: pd.DataFrame,
    n_quantiles: int,
    cap_col: str,
) -> float:
    working = quantile_assignments_df.merge(
        labels_df[["date", "asset", "value"]].rename(columns={"value": "label_value"}),
        on=["date", "asset"],
        how="inner",
        validate="one_to_one",
    ).merge(
        prices[["date", "asset", cap_col]].rename(columns={cap_col: "market_cap_value"}),
        on=["date", "asset"],
        how="left",
        validate="many_to_one",
    )
    if working.empty:
        return float("nan")

    working["label_value"] = pd.to_numeric(working["label_value"], errors="coerce")
    working["market_cap_value"] = pd.to_numeric(working["market_cap_value"], errors="coerce")
    working = working.dropna(subset=["label_value", "market_cap_value"]).copy()
    working = working.loc[working["market_cap_value"] > 0.0].copy()
    if working.empty:
        return float("nan")

    rows: list[float] = []
    for _, block in working.groupby("date", sort=True):
        long_block = block.loc[block["quantile"] == n_quantiles]
        short_block = block.loc[block["quantile"] == 1]
        if long_block.empty or short_block.empty:
            continue
        long_weights = long_block["market_cap_value"] / long_block["market_cap_value"].sum()
        short_weights = short_block["market_cap_value"] / short_block["market_cap_value"].sum()
        rows.append(
            float((long_block["label_value"] * long_weights).sum())
            - float((short_block["label_value"] * short_weights).sum())
        )
    return float(np.mean(rows)) if rows else float("nan")


def _compute_mean_traded_adv(
    *,
    prices: pd.DataFrame,
    quantile_assignments_df: pd.DataFrame,
    n_quantiles: int,
    rebalance_step: int,
    adv_lookback: int,
) -> float:
    if "amount" not in prices.columns or quantile_assignments_df.empty:
        return float("nan")

    adv_frame = prices[["date", "asset", "amount"]].copy()
    adv_frame["date"] = pd.to_datetime(adv_frame["date"], errors="coerce")
    adv_frame["amount"] = pd.to_numeric(adv_frame["amount"], errors="coerce")
    adv_frame = adv_frame.dropna(subset=["date", "asset", "amount"]).sort_values(
        ["asset", "date"],
        kind="mergesort",
    )
    if adv_frame.empty:
        return float("nan")

    min_periods = min(int(adv_lookback), max(3, int(adv_lookback) // 2))
    adv_frame["adv"] = adv_frame.groupby("asset", sort=False)["amount"].transform(
        lambda series: series.rolling(adv_lookback, min_periods=min_periods).mean()
    )
    adv_lookup = adv_frame[["date", "asset", "adv"]]

    assignments = quantile_assignments_df.copy()
    assignments["date"] = pd.to_datetime(assignments["date"], errors="coerce")
    assignments = assignments.sort_values(["date", "asset"], kind="mergesort")
    active_dates = (
        assignments["date"].drop_duplicates().sort_values().iloc[:: max(1, rebalance_step)]
    )
    if len(active_dates) < 2:
        return float("nan")

    traded_adv_values: list[float] = []
    for prev_date, curr_date in zip(active_dates[:-1], active_dates[1:], strict=False):
        prev_block = assignments.loc[assignments["date"] == prev_date]
        curr_block = assignments.loc[assignments["date"] == curr_date]
        traded_assets = _traded_assets_between_assignments(
            prev_block=prev_block,
            curr_block=curr_block,
            n_quantiles=n_quantiles,
        )
        if not traded_assets:
            continue
        adv_rows = adv_lookup.loc[
            (adv_lookup["date"] == curr_date) & (adv_lookup["asset"].isin(traded_assets)),
            "adv",
        ].dropna()
        if len(adv_rows) == 0:
            continue
        traded_adv_values.append(float(adv_rows.mean()))
    return float(np.mean(traded_adv_values)) if traded_adv_values else float("nan")


def _traded_assets_between_assignments(
    *,
    prev_block: pd.DataFrame,
    curr_block: pd.DataFrame,
    n_quantiles: int,
) -> set[str]:
    prev_long = set(prev_block.loc[prev_block["quantile"] == n_quantiles, "asset"].astype(str))
    prev_short = set(prev_block.loc[prev_block["quantile"] == 1, "asset"].astype(str))
    curr_long = set(curr_block.loc[curr_block["quantile"] == n_quantiles, "asset"].astype(str))
    curr_short = set(curr_block.loc[curr_block["quantile"] == 1, "asset"].astype(str))
    return (prev_long ^ curr_long) | (prev_short ^ curr_short)


def _build_conditional_ic_summary(
    *,
    conditional_by_magnitude: pd.DataFrame,
    conditional_by_cross_section: pd.DataFrame,
) -> dict[str, object]:
    q1_mean_ic = _conditional_metric_value(
        conditional_by_magnitude,
        group_col="magnitude_quintile",
        group_value="Q1",
        value_col="mean_ic",
    )
    q5_mean_ic = _conditional_metric_value(
        conditional_by_magnitude,
        group_col="magnitude_quintile",
        group_value="Q5",
        value_col="mean_ic",
    )
    small_mean_ic = _conditional_metric_value(
        conditional_by_cross_section,
        group_col="cross_section_bucket",
        group_value="small_cross_section",
        value_col="mean_ic",
    )
    large_mean_ic = _conditional_metric_value(
        conditional_by_cross_section,
        group_col="cross_section_bucket",
        group_value="large_cross_section",
        value_col="mean_ic",
    )
    return {
        "conditional_ic_q1_mean_ic": q1_mean_ic,
        "conditional_ic_q5_mean_ic": q5_mean_ic,
        "conditional_ic_extreme_minus_base_ic": (
            float(q5_mean_ic - q1_mean_ic)
            if np.isfinite(q5_mean_ic) and np.isfinite(q1_mean_ic)
            else float("nan")
        ),
        "conditional_ic_small_cross_section_mean_ic": small_mean_ic,
        "conditional_ic_large_cross_section_mean_ic": large_mean_ic,
    }


def _conditional_metric_value(
    frame: pd.DataFrame,
    *,
    group_col: str,
    group_value: str,
    value_col: str,
) -> float:
    if frame.empty or group_col not in frame.columns or value_col not in frame.columns:
        return float("nan")
    values = pd.to_numeric(
        frame.loc[frame[group_col] == group_value, value_col],
        errors="coerce",
    ).dropna()
    return float(values.iloc[0]) if len(values) > 0 else float("nan")


def _resolve_market_cap_col(prices: pd.DataFrame) -> str | None:
    for col in ("market_cap", "circ_mv", "total_mv", "value"):
        if col in prices.columns:
            return col
    return None


def _empty_capacity_summary() -> dict[str, object]:
    return {
        "capacity_enabled": False,
        "capacity_status": "unavailable",
        "capacity_notes": "capacity diagnostics unavailable",
        "capacity_market_cap_column": "",
        "capacity_participation_rate": float("nan"),
        "capacity_adv_lookback": float("nan"),
        "equal_weight_mean_long_short_return": float("nan"),
        "market_cap_weighted_mean_long_short_return": float("nan"),
        "market_cap_vs_equal_weight_return_delta": float("nan"),
        "mean_traded_adv": float("nan"),
        "estimated_capacity_upper_bound": float("nan"),
    }
