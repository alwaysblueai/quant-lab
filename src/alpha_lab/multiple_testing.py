from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MultipleTestingResult:
    """Machine-readable multiple-testing adjustment output."""

    method: str
    alpha: float
    n_tests: float
    adjusted_alpha: float
    reject: np.ndarray
    adjusted_pvalues: np.ndarray


def bonferroni_threshold(alpha: float, n_tests: float) -> float:
    """Bonferroni-adjusted family-wise threshold."""
    _validate_alpha_n(alpha, n_tests)
    return float(alpha / n_tests)


def sidak_threshold(alpha: float, n_tests: float) -> float:
    """Sidak-adjusted family-wise threshold."""
    _validate_alpha_n(alpha, n_tests)
    return float(1.0 - (1.0 - alpha) ** (1.0 / n_tests))


def effective_trial_count(corr: pd.DataFrame) -> float:
    """Heuristic effective trial count from a correlation matrix spectrum."""
    if corr.empty:
        raise ValueError("corr must be non-empty")
    if corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be square")
    vals = corr.to_numpy(dtype=float)
    if not np.all(np.isfinite(vals)):
        raise ValueError("corr contains non-finite values")
    eigvals = np.linalg.eigvalsh(vals)
    eigvals = np.clip(eigvals, 0.0, None)
    total = float(eigvals.sum())
    if total <= 0:
        return 1.0
    neff = (total**2) / float(np.sum(eigvals**2))
    return float(min(max(neff, 1.0), vals.shape[0]))


def adjust_pvalues(
    p_values: pd.Series | np.ndarray | list[float],
    *,
    alpha: float = 0.05,
    method: str = "bonferroni",
    n_tests: float | None = None,
) -> MultipleTestingResult:
    """Adjust p-values under FWER control."""
    p = np.asarray(pd.to_numeric(pd.Series(p_values), errors="coerce"), dtype=float)
    if np.isnan(p).any():
        raise ValueError("p_values contains NaN or non-numeric values")
    if (p < 0).any() or (p > 1).any():
        raise ValueError("p_values must be in [0, 1]")
    resolved_n = float(n_tests) if n_tests is not None else float(len(p))
    _validate_alpha_n(alpha, resolved_n)

    method_l = method.lower()
    if method_l == "bonferroni":
        adj_alpha = bonferroni_threshold(alpha, resolved_n)
        adj_p = np.minimum(p * resolved_n, 1.0)
    elif method_l == "sidak":
        adj_alpha = sidak_threshold(alpha, resolved_n)
        adj_p = np.minimum(1.0 - (1.0 - p) ** resolved_n, 1.0)
    else:
        raise ValueError("method must be one of {'bonferroni', 'sidak'}")

    reject = adj_p <= alpha
    return MultipleTestingResult(
        method=method_l,
        alpha=float(alpha),
        n_tests=float(resolved_n),
        adjusted_alpha=float(adj_alpha),
        reject=reject.astype(bool),
        adjusted_pvalues=adj_p.astype(float),
    )


def apply_multiple_testing_to_trial_log(
    trial_log: pd.DataFrame,
    *,
    pvalue_col: str = "p_value",
    alpha: float = 0.05,
    method: str = "bonferroni",
    n_tests: float | None = None,
) -> pd.DataFrame:
    """Attach adjusted significance fields to a trial log table."""
    if pvalue_col not in trial_log.columns:
        raise ValueError(f"trial_log missing pvalue_col {pvalue_col!r}")
    result = adjust_pvalues(
        trial_log[pvalue_col].to_numpy(dtype=float),
        alpha=alpha,
        method=method,
        n_tests=n_tests,
    )
    out = trial_log.copy()
    out["mt_method"] = result.method
    out["mt_alpha"] = result.alpha
    out["mt_n_tests"] = result.n_tests
    out["mt_adjusted_alpha"] = result.adjusted_alpha
    out["mt_adjusted_pvalue"] = result.adjusted_pvalues
    out["mt_reject"] = result.reject
    return out


def _validate_alpha_n(alpha: float, n_tests: float) -> None:
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0, 1)")
    if n_tests <= 0:
        raise ValueError("n_tests must be > 0")
