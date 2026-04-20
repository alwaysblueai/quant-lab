"""Combinatorially Symmetric Cross-Validation PBO for parameter-variant studies.

:func:`alpha_lab.validation.compute_pbo` already accepts pre-computed IS/OOS
score matrices.  This module is the thin convenience wrapper that turns a
matrix of per-variant return series into those matrices using the canonical
CSCV procedure (López de Prado, *The Probability of Backtest Overfitting*,
2014).

Typical usage: you sweep a factor's hyperparameter grid, collect a return
series per variant, and call :func:`compute_cscv_pbo` to estimate the
probability that picking the "best IS" variant systematically fails OOS.

For a single variant this function is not meaningful — PBO is an
overfitting-from-selection metric.  Use
:func:`alpha_lab.validation.haircut_sharpe_ratio` instead for single-variant
studies.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha_lab.validation.cpcv import compute_pbo


@dataclass(frozen=True)
class CscvPboResult:
    """Output of :func:`compute_cscv_pbo`."""

    pbo: float
    """Estimated probability of backtest overfitting in [0, 1].  Values
    ≥ 0.5 are diagnostic of systematic overfit (the IS-best variant lands
    in the OOS bottom half more often than not)."""

    n_splits: int
    """Number of partitions the return axis was divided into (even number)."""

    n_variants: int
    """Number of parameter variants compared."""

    n_combinations: int
    """Number of IS/OOS combinations that contributed to the estimate."""


def compute_cscv_pbo(
    variant_returns: pd.DataFrame,
    *,
    n_splits: int = 16,
) -> CscvPboResult:
    """Estimate PBO from a ``[dates × variants]`` return matrix.

    Parameters
    ----------
    variant_returns:
        Rows are time observations (typically dates), columns are parameter
        variants.  Each cell is a per-period return for that variant.  NaNs
        are allowed; cells that are NaN in *any* variant are dropped from
        the participating time axis to keep splits comparable.
    n_splits:
        Partition count ``S``.  Must be even.  CSCV considers every way to
        choose ``S/2`` partitions as IS and the remainder as OOS.

    Notes
    -----
    The ranking criterion used for IS selection is the Sharpe ratio
    (mean / std) on the IS partition.  OOS ranking uses the same metric on
    the OOS partition.  This matches the López de Prado reference
    implementation.
    """
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValueError("n_splits must be an even integer >= 2")
    if variant_returns.shape[1] < 2:
        return CscvPboResult(
            pbo=float("nan"),
            n_splits=int(n_splits),
            n_variants=int(variant_returns.shape[1]),
            n_combinations=0,
        )

    frame = variant_returns.dropna(axis=0, how="any")
    n_obs = int(len(frame))
    if n_obs < n_splits:
        return CscvPboResult(
            pbo=float("nan"),
            n_splits=int(n_splits),
            n_variants=int(variant_returns.shape[1]),
            n_combinations=0,
        )

    # Equal-sized partitions (any remainder goes to the tail partition).
    fold_size = n_obs // n_splits
    partitions: list[np.ndarray] = []
    for i in range(n_splits):
        start = i * fold_size
        end = start + fold_size if i < n_splits - 1 else n_obs
        partitions.append(np.arange(start, end, dtype=int))

    half = n_splits // 2
    data = frame.to_numpy(dtype=float)
    n_variants = data.shape[1]

    is_rows: list[np.ndarray] = []
    oos_rows: list[np.ndarray] = []

    for is_combo in itertools.combinations(range(n_splits), half):
        is_idx = np.concatenate([partitions[j] for j in is_combo])
        oos_set = [j for j in range(n_splits) if j not in is_combo]
        oos_idx = np.concatenate([partitions[j] for j in oos_set])

        is_block = data[is_idx, :]
        oos_block = data[oos_idx, :]

        is_scores = _sharpe_per_column(is_block)
        oos_scores = _sharpe_per_column(oos_block)

        if not np.all(np.isfinite(is_scores)) or not np.all(np.isfinite(oos_scores)):
            continue

        is_rows.append(is_scores)
        oos_rows.append(oos_scores)

    if not is_rows:
        return CscvPboResult(
            pbo=float("nan"),
            n_splits=int(n_splits),
            n_variants=int(n_variants),
            n_combinations=0,
        )

    is_matrix = pd.DataFrame(np.vstack(is_rows))
    oos_matrix = pd.DataFrame(np.vstack(oos_rows))
    pbo_value = compute_pbo(is_matrix, oos_matrix)

    return CscvPboResult(
        pbo=float(pbo_value),
        n_splits=int(n_splits),
        n_variants=int(n_variants),
        n_combinations=int(len(is_rows)),
    )


def _sharpe_per_column(block: np.ndarray) -> np.ndarray:
    if block.size == 0:
        return np.full(block.shape[1] if block.ndim > 1 else 0, np.nan)
    means = np.nanmean(block, axis=0)
    stds = np.nanstd(block, axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(
            (stds > 0.0) & np.isfinite(stds),
            means / stds,
            np.nan,
        )
    out = np.where(np.isfinite(means), out, np.nan)
    # Guard against the rare case of a zero-variance column producing a
    # finite mean — that would still yield inf here, so coerce to NaN.
    out = np.where(np.isfinite(out), out, np.nan)
    if math.isnan(float(np.nansum(out))) and not np.all(np.isnan(out)):
        # Preserve partial finite scores.
        pass
    return out
