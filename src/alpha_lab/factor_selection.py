from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.evaluation import compute_rank_ic
from alpha_lab.interfaces import FACTOR_OUTPUT_COLUMNS


@dataclass(frozen=True)
class FactorSelectionReport:
    """Typed factor-screening output for governance decisions."""

    summary: pd.DataFrame
    decisions: pd.DataFrame
    pairwise_correlation: pd.DataFrame
    vif: pd.DataFrame
    marginal_contribution: pd.DataFrame


def screen_factors(
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    n_quantiles: int = 5,
    min_coverage: float = 0.6,
    min_abs_monotonicity: float = 0.1,
    max_pairwise_corr: float = 0.9,
    max_vif: float = 10.0,
) -> FactorSelectionReport:
    """Run univariate + redundancy + marginal-contribution screening."""
    if min_coverage <= 0 or min_coverage > 1:
        raise ValueError("min_coverage must be in (0, 1]")
    if min_abs_monotonicity < 0 or min_abs_monotonicity > 1:
        raise ValueError("min_abs_monotonicity must be in [0, 1]")
    if max_pairwise_corr <= 0 or max_pairwise_corr > 1:
        raise ValueError("max_pairwise_corr must be in (0, 1]")
    if max_vif <= 1:
        raise ValueError("max_vif must be > 1")

    _validate_canonical(factors, table_name="factors")
    _validate_canonical(labels, table_name="labels")

    summary = univariate_factor_stats(factors, labels, n_quantiles=n_quantiles)
    corr_pairs, corr_max = pairwise_redundancy(factors)
    vif_df = vif_diagnostics(factors)
    marginal = marginal_ic_contribution(factors, labels)

    merged = summary.merge(corr_max, on="factor", how="left").merge(
        vif_df,
        on="factor",
        how="left",
    )
    merged["max_abs_corr"] = merged["max_abs_corr"].fillna(0.0)
    merged["vif"] = merged["vif"].fillna(1.0)
    merged["coverage_pass"] = merged["coverage"] >= min_coverage
    merged["monotonicity_pass"] = merged["abs_monotonicity"] >= min_abs_monotonicity
    merged["redundancy_pass"] = (merged["max_abs_corr"] <= max_pairwise_corr) & (
        merged["vif"] <= max_vif
    )

    verdict = []
    reason = []
    for _, row in merged.iterrows():
        if pd.isna(row["rank_ic_mean"]) or pd.isna(row["abs_monotonicity"]):
            verdict.append("needs_review")
            reason.append("missing_ic_or_monotonicity")
            continue
        if not row["coverage_pass"] or not row["monotonicity_pass"]:
            verdict.append("weak_factor")
            reason.append("coverage_or_monotonicity_gate_failed")
            continue
        if not row["redundancy_pass"]:
            verdict.append("redundant_factor")
            reason.append("correlation_or_vif_gate_failed")
            continue
        verdict.append("candidate_factor")
        reason.append("all_primary_gates_passed")

    decisions = merged.copy()
    decisions["decision"] = verdict
    decisions["decision_reason"] = reason
    return FactorSelectionReport(
        summary=summary,
        decisions=decisions.sort_values("factor", kind="mergesort").reset_index(drop=True),
        pairwise_correlation=corr_pairs,
        vif=vif_df,
        marginal_contribution=marginal,
    )


def univariate_factor_stats(
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """Compute per-factor univariate quality and monotonicity statistics."""
    factors = factors.copy()
    labels = labels.copy()
    factors["date"] = pd.to_datetime(factors["date"], errors="coerce")
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce")

    rows: list[dict[str, object]] = []
    for factor_name in sorted(pd.unique(factors["factor"].astype(str)).tolist()):
        f = factors[factors["factor"].astype(str) == factor_name].copy()
        label_frame = labels.copy()

        coverage = float(f["value"].notna().mean()) if len(f) > 0 else float("nan")
        rank_ic = compute_rank_ic(f, label_frame)
        vals = rank_ic["rank_ic"].dropna() if not rank_ic.empty else pd.Series(dtype=float)
        ic_mean = float(vals.mean()) if len(vals) > 0 else float("nan")
        ic_std = float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")
        ic_ir = ic_mean / ic_std if np.isfinite(ic_std) and ic_std > 0 else float("nan")
        ic_t = (
            ic_mean / (ic_std / np.sqrt(len(vals)))
            if np.isfinite(ic_std) and ic_std > 0
            else float("nan")
        )

        mono_corr, spread = _monotonicity_stats(
            f,
            label_frame,
            n_quantiles=n_quantiles,
        )
        rows.append(
            {
                "factor": factor_name,
                "coverage": coverage,
                "rank_ic_mean": ic_mean,
                "rank_ic_std": ic_std,
                "rank_ic_ir": ic_ir,
                "rank_ic_t_stat": ic_t,
                "monotonicity_corr": mono_corr,
                "abs_monotonicity": abs(mono_corr) if pd.notna(mono_corr) else float("nan"),
                "long_short_spread": spread,
            }
        )
    return pd.DataFrame(rows).sort_values("factor", kind="mergesort").reset_index(drop=True)


def pairwise_redundancy(factors: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return pairwise factor correlations and per-factor max |corr|."""
    wide = _factor_wide_matrix(factors)
    if wide.shape[1] <= 1:
        pairs = pd.DataFrame(columns=["factor_a", "factor_b", "corr"])
        max_corr = pd.DataFrame(
            {
                "factor": wide.columns.tolist(),
                "max_abs_corr": [0.0] * wide.shape[1],
            }
        )
        return pairs, max_corr

    corr = wide.corr(method="pearson")
    rows: list[dict[str, object]] = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rows.append(
                {
                    "factor_a": cols[i],
                    "factor_b": cols[j],
                    "corr": float(corr.iloc[i, j]),
                }
            )
    pairs = pd.DataFrame(rows).sort_values(
        ["factor_a", "factor_b"],
        kind="mergesort",
    ).reset_index(drop=True)
    max_corr = (
        corr.abs()
        .where(~np.eye(len(corr), dtype=bool), other=np.nan)
        .max(axis=1)
        .rename("max_abs_corr")
        .reset_index()
        .rename(columns={"index": "factor"})
    )
    return pairs, max_corr.sort_values("factor", kind="mergesort").reset_index(drop=True)


def vif_diagnostics(factors: pd.DataFrame) -> pd.DataFrame:
    """Compute simple VIF diagnostics on factor-wide design matrix."""
    wide = _factor_wide_matrix(factors).dropna(axis=0, how="any")
    if wide.shape[1] == 0:
        return pd.DataFrame(columns=["factor", "vif"])
    if wide.shape[1] == 1:
        return pd.DataFrame({"factor": wide.columns.tolist(), "vif": [1.0]})
    if len(wide) < wide.shape[1] + 1:
        return pd.DataFrame(
            {
                "factor": wide.columns.tolist(),
                "vif": [float("nan")] * wide.shape[1],
            }
        )

    vals = wide.to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for i, factor in enumerate(wide.columns):
        y = vals[:, i]
        x = np.delete(vals, i, axis=1)
        x = np.column_stack([np.ones(len(x), dtype=float), x])
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        y_hat = x @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot <= 0:
            r2 = float("nan")
        else:
            r2 = 1.0 - ss_res / ss_tot
        if not np.isfinite(r2):
            vif = float("nan")
        elif r2 >= 0.999999:
            vif = float("inf")
        else:
            vif = float(1.0 / (1.0 - r2))
        rows.append({"factor": factor, "vif": vif})
    return pd.DataFrame(rows).sort_values("factor", kind="mergesort").reset_index(drop=True)


def marginal_ic_contribution(factors: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Approximate marginal contribution via equal-weight pool ablation."""
    wide = _factor_wide_matrix(factors)
    if wide.shape[1] <= 1:
        return pd.DataFrame(
            {
                "factor": wide.columns.tolist(),
                "pool_rank_ic_full": [float("nan")] * wide.shape[1],
                "pool_rank_ic_without_factor": [float("nan")] * wide.shape[1],
                "marginal_delta": [float("nan")] * wide.shape[1],
            }
        )

    full = _equal_weight_composite(wide)
    full_ic = _composite_rank_ic(full, labels, name="pool_full")
    rows: list[dict[str, object]] = []
    for factor in wide.columns:
        other_cols = [c for c in wide.columns if c != factor]
        without = _equal_weight_composite(wide[other_cols])
        without_ic = _composite_rank_ic(without, labels, name=f"pool_wo_{factor}")
        rows.append(
            {
                "factor": factor,
                "pool_rank_ic_full": full_ic,
                "pool_rank_ic_without_factor": without_ic,
                "marginal_delta": (
                    full_ic - without_ic
                    if np.isfinite(full_ic) and np.isfinite(without_ic)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("factor", kind="mergesort").reset_index(drop=True)


def _composite_rank_ic(composite: pd.Series, labels: pd.DataFrame, *, name: str) -> float:
    frame = composite.reset_index().rename(columns={0: "value"})
    frame["factor"] = name
    frame = frame[["date", "asset", "factor", "value"]]
    rank_ic = compute_rank_ic(frame, labels)
    vals = rank_ic["rank_ic"].dropna() if not rank_ic.empty else pd.Series(dtype=float)
    return float(vals.mean()) if len(vals) > 0 else float("nan")


def _equal_weight_composite(wide: pd.DataFrame) -> pd.Series:
    # Cross-sectional z-score per date then equal-weight average across factors.
    indexed = wide.copy()
    date_index = indexed.index.get_level_values("date")
    z = indexed.groupby(date_index, sort=True).transform(_zscore_series)
    return z.mean(axis=1)


def _zscore_series(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    mu = vals.mean()
    sd = vals.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=series.index, dtype=float)
    return (vals - mu) / sd


def _factor_wide_matrix(factors: pd.DataFrame) -> pd.DataFrame:
    _validate_canonical(factors, table_name="factors")
    tmp = factors.copy()
    tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
    wide = tmp.pivot(index=["date", "asset"], columns="factor", values="value")
    wide = wide.sort_index(kind="mergesort")
    wide = wide.reindex(sorted(wide.columns), axis=1)
    return wide


def _monotonicity_stats(
    factor_df: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    n_quantiles: int,
) -> tuple[float, float]:
    merged = factor_df.merge(
        labels[["date", "asset", "value"]],
        on=["date", "asset"],
        how="inner",
        suffixes=("_factor", "_label"),
    )
    merged = merged.dropna(subset=["value_factor", "value_label"])
    if merged.empty:
        return float("nan"), float("nan")

    def _assign(group: pd.DataFrame) -> pd.Series:
        vals = group["value_factor"]
        if vals.nunique() < 2:
            return pd.Series(np.nan, index=group.index, dtype=float)
        q = min(n_quantiles, int(vals.notna().sum()))
        if q < 2:
            return pd.Series(np.nan, index=group.index, dtype=float)
        ranked = vals.rank(method="first")
        return pd.qcut(ranked, q=q, labels=False, duplicates="drop").astype(float) + 1.0

    merged["quantile"] = merged.groupby("date", sort=True, group_keys=False).apply(
        _assign,
        include_groups=False,
    )
    agg = (
        merged.dropna(subset=["quantile"])
        .groupby("quantile", sort=True)["value_label"]
        .mean()
        .rename("mean_return")
        .reset_index()
    )
    if len(agg) < 2:
        return float("nan"), float("nan")
    corr = float(agg["quantile"].corr(agg["mean_return"], method="spearman"))
    spread = float(agg["mean_return"].iloc[-1] - agg["mean_return"].iloc[0])
    return corr, spread


def _validate_canonical(df: pd.DataFrame, *, table_name: str) -> None:
    missing = set(FACTOR_OUTPUT_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{table_name} missing required columns: {sorted(missing)}")
