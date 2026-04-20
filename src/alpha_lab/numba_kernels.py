from __future__ import annotations

import os
from typing import cast

import numpy as np

_NUMBA_DISABLED_BY_ENV = os.getenv("ALPHA_LAB_DISABLE_NUMBA", "0") == "1"

try:
    if _NUMBA_DISABLED_BY_ENV:
        raise ImportError("numba disabled by ALPHA_LAB_DISABLE_NUMBA=1")
    from numba import njit

    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - runtime optional dependency
    NUMBA_AVAILABLE = False
    njit = None  # type: ignore[assignment]


def numba_enabled() -> bool:
    return bool(NUMBA_AVAILABLE)


if NUMBA_AVAILABLE:

    @njit(cache=False)
    def _pearson_corr_numba(x: np.ndarray, y: np.ndarray) -> float:
        n = x.size
        if n < 2:
            return np.nan

        mean_x = 0.0
        mean_y = 0.0
        for i in range(n):
            mean_x += x[i]
            mean_y += y[i]
        mean_x /= n
        mean_y /= n

        cov = 0.0
        var_x = 0.0
        var_y = 0.0
        for i in range(n):
            dx = x[i] - mean_x
            dy = y[i] - mean_y
            cov += dx * dy
            var_x += dx * dx
            var_y += dy * dy

        if var_x <= 0.0 or var_y <= 0.0:
            return np.nan
        return float(cov / np.sqrt(var_x * var_y))

    @njit(cache=False)
    def _average_rank_numba(values: np.ndarray) -> np.ndarray:
        n = values.size
        ranks = np.empty(n, dtype=np.float64)
        if n == 0:
            return ranks

        order = np.argsort(values)
        i = 0
        while i < n:
            j = i + 1
            v = values[order[i]]
            while j < n and values[order[j]] == v:
                j += 1
            # 1-based average rank for tie block [i, j).
            avg_rank = 0.5 * (i + (j - 1)) + 1.0
            for k in range(i, j):
                ranks[order[k]] = avg_rank
            i = j
        return ranks

    @njit(cache=False)
    def _cross_sectional_corr_by_group_numba(
        x: np.ndarray,
        y: np.ndarray,
        start_idx: np.ndarray,
        end_idx: np.ndarray,
        method_code: int,
    ) -> np.ndarray:
        n_groups = start_idx.size
        out = np.empty(n_groups, dtype=np.float64)
        out[:] = np.nan
        for i in range(n_groups):
            begin = start_idx[i]
            end = end_idx[i]
            if end - begin < 2:
                out[i] = np.nan
                continue
            xg = x[begin:end]
            yg = y[begin:end]
            if method_code == 1:
                xr = _average_rank_numba(xg)
                yr = _average_rank_numba(yg)
                out[i] = _pearson_corr_numba(xr, yr)
            else:
                out[i] = _pearson_corr_numba(xg, yg)
        return out


def cross_sectional_corr_by_group_numba(
    x: np.ndarray,
    y: np.ndarray,
    start_idx: np.ndarray,
    end_idx: np.ndarray,
    *,
    method: str,
) -> np.ndarray | None:
    if not NUMBA_AVAILABLE:
        return None

    method_code = 1 if method == "spearman" else 0
    result = _cross_sectional_corr_by_group_numba(
        np.ascontiguousarray(x, dtype=np.float64),
        np.ascontiguousarray(y, dtype=np.float64),
        np.ascontiguousarray(start_idx, dtype=np.int64),
        np.ascontiguousarray(end_idx, dtype=np.int64),
        method_code,
    )
    return cast(np.ndarray, result)
