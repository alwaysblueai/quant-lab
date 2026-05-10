from __future__ import annotations

import pandas as pd

# Cross-module imports (auto-added by split)
from .capacity import _empty_capacity_summary


def _empty_ic_decay_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "horizon",
            "mean_ic",
            "mean_rank_ic",
            "ic_ir",
            "t_stat",
            "p_value",
            "n_dates",
        ]
    )


def _empty_factor_autocorrelation_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["lag", "mean_autocorr", "std_autocorr", "n_dates"])


def _empty_conditional_ic_by_magnitude_frame() -> pd.DataFrame:
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


def _empty_conditional_ic_by_cross_section_size_frame() -> pd.DataFrame:
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


def _empty_capacity_estimation_frame(
    *,
    enabled: bool,
    participation_rate: float,
    adv_lookback: int,
) -> pd.DataFrame:
    summary = _empty_capacity_summary()
    summary["capacity_enabled"] = bool(enabled)
    summary["capacity_status"] = "skipped"
    summary["capacity_notes"] = "capacity diagnostics skipped by evaluation profile"
    summary["capacity_participation_rate"] = float(participation_rate)
    summary["capacity_adv_lookback"] = int(adv_lookback)
    return pd.DataFrame([summary])
