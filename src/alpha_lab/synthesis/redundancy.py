"""Factor correlation and redundancy analysis.

Identifies redundant factor pairs and tracks correlation stability
over time to support factor selection and ensemble construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_factor_correlation_matrix(
    factor_dict: dict[str, pd.DataFrame],
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute cross-sectional correlation matrix between factors.

    For each pair of factors, computes the mean cross-sectional correlation
    across all dates where both factors have at least 3 non-NaN assets.

    Parameters
    ----------
    factor_dict:
        Mapping of ``{factor_name: DataFrame[date, asset, factor, value]}``.
    method:
        Correlation method: ``"spearman"`` (default) or ``"pearson"``.

    Returns
    -------
    pd.DataFrame
        Square correlation matrix indexed by factor names.
    """
    names = sorted(factor_dict.keys())
    n = len(names)
    corr_matrix = np.eye(n)

    # Build wide DataFrames per factor, keyed by (date, asset)
    wide: dict[str, pd.DataFrame] = {}
    for name in names:
        df = factor_dict[name][["date", "asset", "value"]].copy()
        df = df.rename(columns={"value": name})
        wide[name] = df

    for i in range(n):
        for j in range(i + 1, n):
            merged = wide[names[i]].merge(
                wide[names[j]],
                on=["date", "asset"],
                how="inner",
            )
            if merged.empty:
                corr_matrix[i, j] = corr_matrix[j, i] = float("nan")
                continue

            date_corrs: list[float] = []
            for _, group in merged.groupby("date"):
                valid = group[[names[i], names[j]]].dropna()
                if len(valid) < 3:
                    continue
                a = valid[names[i]]
                b = valid[names[j]]
                if method == "spearman":
                    a = a.rank(method="average")
                    b = b.rank(method="average")
                if a.std() == 0 or b.std() == 0:
                    continue
                date_corrs.append(float(a.corr(b)))

            if date_corrs:
                mean_corr = float(np.mean(date_corrs))
            else:
                pooled = merged[[names[i], names[j]]].dropna()
                pooled_valid = (
                    len(pooled) >= 3
                    and pooled[names[i]].nunique() >= 2
                    and pooled[names[j]].nunique() >= 2
                )
                if pooled_valid:
                    a = pooled[names[i]]
                    b = pooled[names[j]]
                    if method == "spearman":
                        a = a.rank(method="average")
                        b = b.rank(method="average")
                    mean_corr = float(a.corr(b))
                else:
                    mean_corr = float("nan")
            corr_matrix[i, j] = corr_matrix[j, i] = mean_corr

    return pd.DataFrame(corr_matrix, index=names, columns=names)


def detect_redundant_factors(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7,
) -> list[tuple[str, str, float]]:
    """Detect factor pairs with absolute correlation above threshold.

    Parameters
    ----------
    corr_matrix:
        Square correlation matrix from :func:`compute_factor_correlation_matrix`.
    threshold:
        Absolute correlation threshold for redundancy.

    Returns
    -------
    list of (factor_a, factor_b, correlation) tuples, sorted by |correlation| descending.
    """
    names = list(corr_matrix.index)
    n = len(names)
    pairs: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            val = corr_matrix.iloc[i, j]
            if np.isfinite(val) and abs(val) >= threshold:
                pairs.append((names[i], names[j], float(val)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs


def compute_rolling_factor_correlation(
    factor_dict: dict[str, pd.DataFrame],
    window: int = 60,
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute rolling cross-sectional correlation between factor pairs.

    For each pair and each date, computes the mean cross-sectional
    correlation over the trailing ``window`` dates.

    Parameters
    ----------
    factor_dict:
        Mapping of ``{factor_name: DataFrame[date, asset, factor, value]}``.
    window:
        Rolling window size in number of dates.
    method:
        Correlation method: ``"spearman"`` or ``"pearson"``.

    Returns
    -------
    pd.DataFrame
        Columns: ``[date, factor_a, factor_b, correlation]``.
    """
    names = sorted(factor_dict.keys())
    if len(names) < 2:
        return pd.DataFrame(columns=["date", "factor_a", "factor_b", "correlation"])

    # Compute per-date cross-sectional correlations for all pairs
    wide: dict[str, pd.DataFrame] = {}
    for name in names:
        df = factor_dict[name][["date", "asset", "value"]].copy()
        df = df.rename(columns={"value": name})
        wide[name] = df

    rows: list[dict[str, object]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            merged = wide[names[i]].merge(
                wide[names[j]],
                on=["date", "asset"],
                how="inner",
            )
            if merged.empty:
                continue

            # Per-date correlation
            date_corrs: list[tuple[pd.Timestamp, float]] = []
            for dt, group in merged.groupby("date"):
                valid = group[[names[i], names[j]]].dropna()
                if len(valid) < 3:
                    date_corrs.append((dt, float("nan")))
                    continue
                a = valid[names[i]]
                b = valid[names[j]]
                if method == "spearman":
                    a = a.rank(method="average")
                    b = b.rank(method="average")
                if a.std() == 0 or b.std() == 0:
                    date_corrs.append((dt, float("nan")))
                    continue
                date_corrs.append((dt, float(a.corr(b))))

            if not date_corrs:
                continue

            corr_series = pd.Series(
                [c for _, c in date_corrs],
                index=pd.DatetimeIndex([d for d, _ in date_corrs]),
            ).sort_index()

            rolling_mean = corr_series.rolling(window=window, min_periods=1).mean()
            for dt, val in rolling_mean.items():
                rows.append(
                    {
                        "date": dt,
                        "factor_a": names[i],
                        "factor_b": names[j],
                        "correlation": float(val),
                    }
                )

    return pd.DataFrame(rows)
