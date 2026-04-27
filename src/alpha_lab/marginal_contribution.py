"""Marginal contribution analysis for multi-factor contexts.

Provides Fama-MacBeth cross-sectional regression and spanning tests to
determine whether a candidate factor contributes incremental predictive
power beyond existing factors.  This is a Level 2 gate diagnostic —
a factor that is fully spanned by existing factors should not advance.

All computations use only data available at each evaluation date (no
lookahead).  The Fama-MacBeth procedure runs independent cross-sectional
regressions per date and then aggregates coefficient estimates over time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


@dataclass(frozen=True)
class FamaMacBethResult:
    """Time-series aggregated Fama-MacBeth regression output."""

    factor_name: str
    """Name of the candidate factor."""

    mean_coefficient: float
    """Time-series mean of per-date cross-sectional regression coefficients."""

    t_statistic: float
    """Newey-West style t-statistic: mean_coef / (std_coef / sqrt(n))."""

    p_value: float
    """Two-sided p-value for the t-statistic (Student-t, df=n-1)."""

    n_dates: int
    """Number of cross-sectional regressions with a valid coefficient."""

    mean_r_squared: float
    """Mean cross-sectional R² across evaluation dates."""


@dataclass(frozen=True)
class SpanningTestResult:
    """Result of a factor spanning test."""

    candidate_factor: str
    """Name of the candidate factor."""

    is_spanned: bool
    """True if the candidate is fully spanned by existing factors
    (intercept not significantly different from zero)."""

    intercept_t_stat: float
    """t-statistic for the intercept in the spanning regression."""

    intercept_p_value: float
    """Two-sided p-value for the intercept t-statistic."""

    mean_intercept: float
    """Mean intercept from per-date regressions."""

    r_squared_increment: float
    """Mean R² increment from adding the candidate factor."""


@dataclass(frozen=True)
class MarginalContributionSummary:
    """Full marginal contribution analysis output."""

    fama_macbeth: FamaMacBethResult | None
    """Fama-MacBeth regression result for the candidate factor.
    None if insufficient data."""

    spanning_test: SpanningTestResult | None
    """Spanning test result.  None if no existing factors provided."""

    marginal_flags: tuple[str, ...]
    """Diagnostic flags for marginal contribution weakness."""


def fama_macbeth_regression(
    candidate_factor_df: pd.DataFrame,
    label_df: pd.DataFrame,
    existing_factor_dfs: list[pd.DataFrame] | None = None,
) -> FamaMacBethResult | None:
    """Run Fama-MacBeth cross-sectional regressions.

    For each evaluation date, regress forward returns on candidate factor
    values (and existing factors if provided).  Then aggregate the candidate
    factor's coefficient over time.

    Parameters
    ----------
    candidate_factor_df:
        ``[date, asset, factor, value]`` for the candidate factor.
    label_df:
        ``[date, asset, value]`` forward-return labels.
    existing_factor_dfs:
        Optional list of ``[date, asset, factor, value]`` DataFrames for
        existing/control factors.

    Returns
    -------
    FamaMacBethResult or None if insufficient data.
    """
    if candidate_factor_df.empty or label_df.empty:
        return None

    candidate_name = str(candidate_factor_df["factor"].iloc[0])

    # Build per-date cross-sectional data
    cand = candidate_factor_df[["date", "asset", "value"]].copy()
    cand = cand.rename(columns={"value": candidate_name})
    cand["date"] = pd.to_datetime(cand["date"])

    labels = label_df[["date", "asset", "value"]].copy()
    labels = labels.rename(columns={"value": "_label"})
    labels["date"] = pd.to_datetime(labels["date"])

    merged = cand.merge(labels, on=["date", "asset"], how="inner")

    factor_cols = [candidate_name]
    if existing_factor_dfs:
        for i, edf in enumerate(existing_factor_dfs):
            if edf.empty:
                continue
            ename = str(edf["factor"].iloc[0]) if "factor" in edf.columns else f"existing_{i}"
            ef = edf[["date", "asset", "value"]].copy()
            ef = ef.rename(columns={"value": ename})
            ef["date"] = pd.to_datetime(ef["date"])
            merged = merged.merge(ef, on=["date", "asset"], how="inner")
            if ename not in factor_cols:
                factor_cols.append(ename)

    coefficients: list[float] = []
    r_squareds: list[float] = []

    for _, group in merged.groupby("date"):
        group = group.dropna(subset=[candidate_name, "_label"] + factor_cols)
        if len(group) < max(5, len(factor_cols) + 2):
            continue

        y = group["_label"].to_numpy(dtype=float)
        X = group[factor_cols].to_numpy(dtype=float)

        # Add intercept
        X_with_const = np.column_stack([np.ones(len(y)), X])

        try:
            beta, residuals, rank, sv = np.linalg.lstsq(X_with_const, y, rcond=None)
        except np.linalg.LinAlgError:
            continue

        if rank < X_with_const.shape[1]:
            continue

        # Candidate factor coefficient is at index 1 (after intercept)
        coef = float(beta[1])
        coefficients.append(coef)

        # R²
        y_hat = X_with_const @ beta
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        r_squareds.append(r2)

    if len(coefficients) < 2:
        return None

    coefs = np.array(coefficients, dtype=float)
    n = len(coefs)
    mean_coef = float(np.mean(coefs))
    std_coef = float(np.std(coefs, ddof=1))

    if std_coef == 0 or not np.isfinite(std_coef):
        t_stat = float("nan")
        p_val = float("nan")
    else:
        t_stat = mean_coef / (std_coef / np.sqrt(n))
        p_val = float(2.0 * scipy_stats.t.sf(abs(t_stat), df=n - 1))

    valid_r2 = [r for r in r_squareds if np.isfinite(r)]
    mean_r2 = float(np.mean(valid_r2)) if valid_r2 else float("nan")

    return FamaMacBethResult(
        factor_name=candidate_name,
        mean_coefficient=mean_coef,
        t_statistic=t_stat,
        p_value=p_val,
        n_dates=n,
        mean_r_squared=mean_r2,
    )


def spanning_test(
    candidate_factor_df: pd.DataFrame,
    label_df: pd.DataFrame,
    existing_factor_dfs: list[pd.DataFrame],
) -> SpanningTestResult | None:
    """Test whether a candidate factor is spanned by existing factors.

    Runs per-date cross-sectional regressions of the candidate factor's
    contribution to forward returns, controlling for existing factors.
    The intercept captures unexplained alpha.

    Parameters
    ----------
    candidate_factor_df:
        ``[date, asset, factor, value]`` for the candidate.
    label_df:
        ``[date, asset, value]`` forward-return labels.
    existing_factor_dfs:
        List of ``[date, asset, factor, value]`` DataFrames for existing
        factors.  Must contain at least one factor.

    Returns
    -------
    SpanningTestResult or None if insufficient data.
    """
    if not existing_factor_dfs or candidate_factor_df.empty or label_df.empty:
        return None

    candidate_name = str(candidate_factor_df["factor"].iloc[0])

    # Build per-date data
    cand = candidate_factor_df[["date", "asset", "value"]].copy()
    cand = cand.rename(columns={"value": "_candidate"})
    cand["date"] = pd.to_datetime(cand["date"])

    labels = label_df[["date", "asset", "value"]].copy()
    labels = labels.rename(columns={"value": "_label"})
    labels["date"] = pd.to_datetime(labels["date"])

    merged = cand.merge(labels, on=["date", "asset"], how="inner")

    existing_cols: list[str] = []
    for i, edf in enumerate(existing_factor_dfs):
        if edf.empty:
            continue
        ename = str(edf["factor"].iloc[0]) if "factor" in edf.columns else f"existing_{i}"
        ef = edf[["date", "asset", "value"]].copy()
        ef = ef.rename(columns={"value": ename})
        ef["date"] = pd.to_datetime(ef["date"])
        merged = merged.merge(ef, on=["date", "asset"], how="inner")
        if ename not in existing_cols:
            existing_cols.append(ename)

    if not existing_cols:
        return None

    all_cols = existing_cols + ["_candidate"]
    intercepts: list[float] = []
    r2_existing_only: list[float] = []
    r2_with_candidate: list[float] = []

    for _, group in merged.groupby("date"):
        group = group.dropna(subset=["_label", "_candidate"] + existing_cols)
        if len(group) < max(5, len(all_cols) + 2):
            continue

        y = group["_label"].to_numpy(dtype=float)

        # Regression with existing only
        X_existing = np.column_stack([np.ones(len(y)), group[existing_cols].to_numpy(dtype=float)])
        try:
            beta_e, _, rank_e, _ = np.linalg.lstsq(X_existing, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if rank_e < X_existing.shape[1]:
            continue
        y_hat_e = X_existing @ beta_e
        ss_res_e = float(np.sum((y - y_hat_e) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2_e = 1.0 - ss_res_e / ss_tot if ss_tot > 0 else float("nan")

        # Regression with all factors
        X_all = np.column_stack([np.ones(len(y)), group[all_cols].to_numpy(dtype=float)])
        try:
            beta_a, _, rank_a, _ = np.linalg.lstsq(X_all, y, rcond=None)
        except np.linalg.LinAlgError:
            continue
        if rank_a < X_all.shape[1]:
            continue
        y_hat_a = X_all @ beta_a
        ss_res_a = float(np.sum((y - y_hat_a) ** 2))
        r2_a = 1.0 - ss_res_a / ss_tot if ss_tot > 0 else float("nan")

        # Intercept is beta_a[0]; candidate coef is last (after existing)
        intercepts.append(float(beta_a[0]))
        r2_existing_only.append(r2_e)
        r2_with_candidate.append(r2_a)

    if len(intercepts) < 2:
        return None

    intercept_arr = np.array(intercepts, dtype=float)
    n = len(intercept_arr)
    mean_intercept = float(np.mean(intercept_arr))
    std_intercept = float(np.std(intercept_arr, ddof=1))

    if std_intercept == 0 or not np.isfinite(std_intercept):
        intercept_t = float("nan")
        intercept_p = float("nan")
    else:
        intercept_t = mean_intercept / (std_intercept / np.sqrt(n))
        intercept_p = float(2.0 * scipy_stats.t.sf(abs(intercept_t), df=n - 1))

    valid_r2_e = [r for r in r2_existing_only if np.isfinite(r)]
    valid_r2_a = [r for r in r2_with_candidate if np.isfinite(r)]
    mean_r2_e = float(np.mean(valid_r2_e)) if valid_r2_e else 0.0
    mean_r2_a = float(np.mean(valid_r2_a)) if valid_r2_a else 0.0
    r2_increment = mean_r2_a - mean_r2_e

    # Spanned if intercept is not significant at 5% level
    is_spanned = bool(np.isfinite(intercept_p) and intercept_p > 0.05)

    return SpanningTestResult(
        candidate_factor=candidate_name,
        is_spanned=is_spanned,
        intercept_t_stat=intercept_t,
        intercept_p_value=intercept_p,
        mean_intercept=mean_intercept,
        r_squared_increment=r2_increment,
    )


def compute_marginal_contribution(
    candidate_factor_df: pd.DataFrame,
    label_df: pd.DataFrame,
    existing_factor_dfs: list[pd.DataFrame] | None = None,
    *,
    significance_level: float = 0.05,
) -> MarginalContributionSummary:
    """Compute full marginal contribution analysis.

    Parameters
    ----------
    candidate_factor_df:
        ``[date, asset, factor, value]`` for the candidate.
    label_df:
        ``[date, asset, value]`` forward-return labels.
    existing_factor_dfs:
        Optional list of existing factor DataFrames.  When provided, a
        spanning test is also performed.
    significance_level:
        Threshold for flagging non-significant contributions.

    Returns
    -------
    MarginalContributionSummary
    """
    fm_result = fama_macbeth_regression(candidate_factor_df, label_df, existing_factor_dfs)

    span_result: SpanningTestResult | None = None
    if existing_factor_dfs:
        span_result = spanning_test(candidate_factor_df, label_df, existing_factor_dfs)

    flags: list[str] = []

    if fm_result is not None:
        if np.isfinite(fm_result.p_value) and fm_result.p_value > significance_level:
            flags.append("fama_macbeth_not_significant")
        if np.isfinite(fm_result.mean_coefficient) and fm_result.mean_coefficient <= 0:
            flags.append("fama_macbeth_negative_coefficient")

    if span_result is not None:
        if span_result.is_spanned:
            flags.append("candidate_is_spanned")
        if np.isfinite(span_result.r_squared_increment) and span_result.r_squared_increment <= 0:
            flags.append("no_incremental_r_squared")

    if flags:
        flags.append("marginal_contribution_weak")

    return MarginalContributionSummary(
        fama_macbeth=fm_result,
        spanning_test=span_result,
        marginal_flags=tuple(flags),
    )
