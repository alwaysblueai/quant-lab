from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlphaPoolDiagnostics:
    """Alpha-pool breadth/diversification diagnostics."""

    correlation_matrix: pd.DataFrame
    pairwise: pd.DataFrame
    breadth_summary: pd.DataFrame
    clusters: pd.DataFrame


def alpha_pool_diagnostics(
    alpha_returns: pd.DataFrame,
    *,
    cluster_threshold: float = 0.8,
) -> AlphaPoolDiagnostics:
    """Compute pairwise correlation, breadth, and cluster diagnostics."""
    if alpha_returns.empty:
        raise ValueError("alpha_returns is empty")
    if cluster_threshold <= 0 or cluster_threshold > 1:
        raise ValueError("cluster_threshold must be in (0, 1]")

    corr = alpha_returns.corr(method="pearson")
    pairwise = pairwise_correlation_table(corr)
    breadth = breadth_summary(corr)
    clusters = cluster_summary(corr, threshold=cluster_threshold)
    return AlphaPoolDiagnostics(
        correlation_matrix=corr,
        pairwise=pairwise,
        breadth_summary=breadth,
        clusters=clusters,
    )


def pairwise_correlation_table(corr: pd.DataFrame) -> pd.DataFrame:
    """Flatten correlation matrix into pairwise table."""
    if corr.shape[0] != corr.shape[1]:
        raise ValueError("corr must be square")
    cols = corr.columns.tolist()
    rows: list[dict[str, object]] = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            rows.append(
                {
                    "alpha_a": cols[i],
                    "alpha_b": cols[j],
                    "corr": float(corr.iloc[i, j]),
                    "abs_corr": abs(float(corr.iloc[i, j])),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["alpha_a", "alpha_b", "corr", "abs_corr"])
    return out.sort_values(["alpha_a", "alpha_b"], kind="mergesort").reset_index(drop=True)


def effective_breadth(corr: pd.DataFrame) -> float:
    """Estimate effective breadth from correlation eigen-spectrum."""
    if corr.empty:
        return float("nan")
    vals = corr.to_numpy(dtype=float)
    eigvals = np.linalg.eigvalsh(vals)
    eigvals = np.clip(eigvals, 0.0, None)
    total = float(eigvals.sum())
    if total <= 0:
        return 1.0
    neff = (total**2) / float(np.sum(eigvals**2))
    return float(min(max(neff, 1.0), vals.shape[0]))


def breadth_summary(corr: pd.DataFrame) -> pd.DataFrame:
    """Summarize average redundancy and implied diversification limit."""
    n = corr.shape[0]
    if n == 0:
        return pd.DataFrame(
            columns=[
                "n_alphas",
                "avg_abs_corr",
                "effective_breadth",
                "breadth_ratio",
                "ir_multiplier_bound",
            ]
        )

    upper = corr.to_numpy(dtype=float)[np.triu_indices(n, k=1)]
    avg_abs = float(np.nanmean(np.abs(upper))) if len(upper) > 0 else 0.0
    neff = effective_breadth(corr)
    ratio = neff / n if n > 0 else float("nan")
    ir_mult = float(np.sqrt(neff)) if np.isfinite(neff) else float("nan")
    return pd.DataFrame(
        [
            {
                "n_alphas": n,
                "avg_abs_corr": avg_abs,
                "effective_breadth": neff,
                "breadth_ratio": ratio,
                "ir_multiplier_bound": ir_mult,
            }
        ]
    )


def cluster_summary(corr: pd.DataFrame, *, threshold: float = 0.8) -> pd.DataFrame:
    """Connected-component summary from correlation threshold graph."""
    if threshold <= 0 or threshold > 1:
        raise ValueError("threshold must be in (0, 1]")
    cols = corr.columns.tolist()
    if len(cols) == 0:
        return pd.DataFrame(columns=["cluster_id", "n_alphas", "members"])

    abs_corr = corr.abs().fillna(0.0)
    visited: set[str] = set()
    rows: list[dict[str, object]] = []
    cid = 0
    for alpha in cols:
        if alpha in visited:
            continue
        stack = [alpha]
        members: list[str] = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            members.append(cur)
            neigh = abs_corr.index[abs_corr.loc[cur] >= threshold].tolist()
            for nxt in neigh:
                if nxt not in visited:
                    stack.append(nxt)
        members = sorted(members)
        rows.append(
            {
                "cluster_id": cid,
                "n_alphas": len(members),
                "members": "|".join(members),
            }
        )
        cid += 1
    return pd.DataFrame(rows).sort_values("cluster_id", kind="mergesort").reset_index(drop=True)
