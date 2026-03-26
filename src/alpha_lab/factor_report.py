from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.evaluation import compute_ic, compute_rank_ic
from alpha_lab.labels import forward_return
from alpha_lab.quantile import long_short_return, quantile_assignments, quantile_returns
from alpha_lab.turnover import long_short_turnover, quantile_turnover


@dataclass(frozen=True)
class FactorReport:
    """Reusable, audit-friendly diagnostics object for one factor study."""

    factor_name: str
    label_name: str
    horizon: int
    n_quantiles: int
    ic_df: pd.DataFrame
    rank_ic_df: pd.DataFrame
    ic_summary_df: pd.DataFrame
    rolling_rank_ic_df: pd.DataFrame
    coverage_df: pd.DataFrame
    quantile_returns_df: pd.DataFrame
    monotonicity_df: pd.DataFrame
    long_short_df: pd.DataFrame
    long_short_turnover_df: pd.DataFrame
    decay_profile_df: pd.DataFrame
    half_life_periods: float

    def to_dict(self) -> dict[str, object]:
        return {
            "factor_name": self.factor_name,
            "label_name": self.label_name,
            "horizon": self.horizon,
            "n_quantiles": self.n_quantiles,
            "half_life_periods": self.half_life_periods,
        }


def compute_ic_summary(ic_df: pd.DataFrame, *, value_col: str) -> pd.DataFrame:
    """Return mean/std/ICIR/t-stat summary for an IC-like series."""
    if value_col not in ic_df.columns:
        raise ValueError(f"ic_df missing value_col {value_col!r}")

    vals = ic_df[value_col].dropna()
    n_obs = int(len(vals))
    mean = float(vals.mean()) if n_obs > 0 else float("nan")
    std = float(vals.std(ddof=1)) if n_obs > 1 else float("nan")

    if np.isnan(std) or std == 0.0:
        icir = float("nan")
        t_stat = float("nan")
    else:
        icir = mean / std
        t_stat = mean / (std / np.sqrt(n_obs))

    return pd.DataFrame(
        [
            {
                "metric": value_col,
                "mean": mean,
                "std": std,
                "ic_ir": icir,
                "t_stat": t_stat,
                "n_obs": n_obs,
            }
        ]
    )


def compute_rolling_ic(
    ic_df: pd.DataFrame,
    *,
    value_col: str,
    window: int = 63,
) -> pd.DataFrame:
    if window <= 1:
        raise ValueError("window must be > 1")
    if value_col not in ic_df.columns:
        raise ValueError(f"ic_df missing value_col {value_col!r}")
    if "date" not in ic_df.columns:
        raise ValueError("ic_df must contain 'date'")

    if ic_df.empty:
        return pd.DataFrame(columns=["date", f"rolling_{value_col}"])

    tmp = ic_df[["date", value_col]].copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp = tmp.sort_values("date").reset_index(drop=True)
    tmp[f"rolling_{value_col}"] = tmp[value_col].rolling(window=window, min_periods=5).mean()
    return tmp[["date", f"rolling_{value_col}"]]


def coverage_by_date(factors: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Compute date-level coverage diagnostics for factors/labels/merge."""
    if factors.empty or labels.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "n_factor_assets",
                "n_label_assets",
                "n_overlap_assets",
                "coverage_overlap_vs_factor",
            ]
        )

    f = factors[["date", "asset", "value"]].copy()
    lbl = labels[["date", "asset", "value"]].copy()
    f = f.rename(columns={"value": "factor_value"})
    lbl = lbl.rename(columns={"value": "label_value"})

    f_nonnull = f[f["factor_value"].notna()]
    l_nonnull = lbl[lbl["label_value"].notna()]
    overlap = f_nonnull.merge(l_nonnull, on=["date", "asset"], how="inner")

    f_count = f_nonnull.groupby("date", sort=True)["asset"].nunique().rename("n_factor_assets")
    l_count = l_nonnull.groupby("date", sort=True)["asset"].nunique().rename("n_label_assets")
    o_count = overlap.groupby("date", sort=True)["asset"].nunique().rename("n_overlap_assets")
    out = pd.concat([f_count, l_count, o_count], axis=1, sort=False).fillna(0).reset_index()
    out["n_factor_assets"] = out["n_factor_assets"].astype(int)
    out["n_label_assets"] = out["n_label_assets"].astype(int)
    out["n_overlap_assets"] = out["n_overlap_assets"].astype(int)
    out["coverage_overlap_vs_factor"] = out["n_overlap_assets"] / out["n_factor_assets"].replace(
        0, np.nan
    )
    return out


def quantile_monotonicity(quantile_ret: pd.DataFrame) -> pd.DataFrame:
    """Compute date-level monotonicity diagnostics for quantile returns."""
    if quantile_ret.empty:
        return pd.DataFrame(
            columns=["date", "n_quantiles_present", "rank_corr_quantile_return", "is_monotonic"]
        )

    required = {"date", "quantile", "mean_return"}
    missing = required - set(quantile_ret.columns)
    if missing:
        raise ValueError(f"quantile_ret missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for date, group in quantile_ret.groupby("date", sort=True):
        g = group.sort_values("quantile")
        n_q = int(g["quantile"].nunique())
        if n_q < 2:
            corr = float("nan")
            mono = False
        else:
            corr = float(g["quantile"].corr(g["mean_return"], method="spearman"))
            diffs = g["mean_return"].diff().dropna()
            mono = bool((diffs >= 0).all() or (diffs <= 0).all())
        rows.append(
            {
                "date": pd.Timestamp(date),
                "n_quantiles_present": n_q,
                "rank_corr_quantile_return": corr,
                "is_monotonic": mono,
            }
        )
    return pd.DataFrame(rows)


def signal_decay_profile(
    factor_df: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    horizons: Sequence[int] = (1, 2, 5, 10, 20),
) -> pd.DataFrame:
    """Compute horizon-wise rank-IC profile for decay diagnostics."""
    if factor_df.empty:
        return pd.DataFrame(columns=["horizon", "mean_rank_ic", "n_obs"])

    rows: list[dict[str, object]] = []
    for h in horizons:
        if h <= 0:
            raise ValueError(f"horizons must be positive, got {h}")
        labels = forward_return(prices, horizon=int(h))
        rank_ic = compute_rank_ic(factor_df, labels)
        vals = rank_ic["rank_ic"].dropna() if not rank_ic.empty else pd.Series(dtype=float)
        rows.append(
            {
                "horizon": int(h),
                "mean_rank_ic": float(vals.mean()) if len(vals) > 0 else float("nan"),
                "n_obs": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def estimate_half_life(decay_profile: pd.DataFrame) -> float:
    """Estimate half-life from horizon profile using log-linear fit on |IC|."""
    if decay_profile.empty:
        return float("nan")

    required = {"horizon", "mean_rank_ic"}
    missing = required - set(decay_profile.columns)
    if missing:
        raise ValueError(f"decay_profile missing required columns: {sorted(missing)}")

    tmp = decay_profile[["horizon", "mean_rank_ic"]].dropna().copy()
    tmp["abs_ic"] = tmp["mean_rank_ic"].abs()
    tmp = tmp[tmp["abs_ic"] > 0]
    if len(tmp) < 2:
        return float("nan")

    x = tmp["horizon"].to_numpy(dtype=float)
    y = np.log(tmp["abs_ic"].to_numpy(dtype=float))
    slope, _intercept = np.polyfit(x, y, deg=1)
    lam = -float(slope)
    if not np.isfinite(lam) or lam <= 0:
        return float("nan")
    return float(np.log(2.0) / lam)


def build_factor_report(
    *,
    prices: pd.DataFrame,
    factors: pd.DataFrame,
    horizon: int,
    n_quantiles: int = 5,
    rolling_window: int = 63,
    decay_horizons: Sequence[int] = (1, 2, 5, 10, 20),
) -> FactorReport:
    """Build a complete diagnostics report from canonical research inputs."""
    labels = forward_return(prices, horizon=horizon)
    ic_df = compute_ic(factors, labels)
    rank_ic_df = compute_rank_ic(factors, labels)
    ic_summary_df = pd.concat(
        [
            compute_ic_summary(ic_df, value_col="ic"),
            compute_ic_summary(rank_ic_df, value_col="rank_ic"),
        ],
        ignore_index=True,
    )
    rolling_rank_ic_df = compute_rolling_ic(rank_ic_df, value_col="rank_ic", window=rolling_window)
    coverage_df = coverage_by_date(factors, labels)
    qret_df = quantile_returns(factors, labels, n_quantiles=n_quantiles)
    mono_df = quantile_monotonicity(qret_df)
    ls_df = long_short_return(qret_df)

    assignments = quantile_assignments(factors, n_quantiles=n_quantiles)
    qto_df = quantile_turnover(assignments)
    lsto_df = long_short_turnover(qto_df)

    decay_df = signal_decay_profile(factors, prices, horizons=decay_horizons)
    half_life = estimate_half_life(decay_df)

    factor_name = str(factors["factor"].iloc[0]) if not factors.empty else "unknown"
    label_name = str(labels["factor"].iloc[0]) if not labels.empty else "unknown"

    return FactorReport(
        factor_name=factor_name,
        label_name=label_name,
        horizon=horizon,
        n_quantiles=n_quantiles,
        ic_df=ic_df,
        rank_ic_df=rank_ic_df,
        ic_summary_df=ic_summary_df,
        rolling_rank_ic_df=rolling_rank_ic_df,
        coverage_df=coverage_df,
        quantile_returns_df=qret_df,
        monotonicity_df=mono_df,
        long_short_df=ls_df,
        long_short_turnover_df=lsto_df,
        decay_profile_df=decay_df,
        half_life_periods=half_life,
    )
