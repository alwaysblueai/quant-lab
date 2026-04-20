from __future__ import annotations

import pandas as pd
from statsmodels.stats.multitest import multipletests


def apply_multiple_testing_correction(
    p_values: dict[str, float],
    method: str = "bh",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Apply multiple-testing correction to a factor -> p-value mapping."""
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0, 1)")
    if not p_values:
        return pd.DataFrame(columns=["factor", "p_value", "corrected_p_value", "reject_null"])

    method_key = method.strip().lower()
    method_map = {
        "bonferroni": "bonferroni",
        "bh": "fdr_bh",
        "holm": "holm",
    }
    if method_key not in method_map:
        raise ValueError("method must be one of: bonferroni, bh, holm")

    factors = list(p_values.keys())
    raw = pd.Series([p_values[f] for f in factors], dtype=float)
    reject, corrected, _, _ = multipletests(
        raw.to_numpy(dtype=float),
        alpha=float(alpha),
        method=method_map[method_key],
    )
    out = pd.DataFrame(
        {
            "factor": factors,
            "p_value": raw.to_numpy(dtype=float),
            "corrected_p_value": corrected.astype(float),
            "reject_null": reject.astype(bool),
        }
    )
    return out.sort_values("factor", kind="mergesort").reset_index(drop=True)
