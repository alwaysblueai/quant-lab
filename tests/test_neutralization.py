from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.neutralization import _mean_abs_corr


def _mean_abs_corr_reference(y: pd.Series, x_frame: pd.DataFrame) -> float:
    yv = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    corrs: list[float] = []
    for _, col in x_frame.items():
        xv = pd.to_numeric(col, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(yv) & np.isfinite(xv)
        if int(valid.sum()) < 2:
            continue
        y_sub = yv[valid]
        x_sub = xv[valid]
        if np.nanstd(y_sub) == 0 or np.nanstd(x_sub) == 0:
            continue
        corr = np.corrcoef(y_sub, x_sub)[0, 1]
        if np.isfinite(corr):
            corrs.append(float(abs(corr)))
    if not corrs:
        return float("nan")
    return float(np.mean(corrs))


def test_mean_abs_corr_vectorized_equivalence() -> None:
    rng = np.random.default_rng(20260424)
    y = pd.Series(rng.normal(size=50), name="signal")
    x_frame = pd.DataFrame(
        rng.normal(size=(50, 5)),
        columns=[f"exposure_{idx}" for idx in range(5)],
    )
    y.iloc[[3, 11]] = np.nan
    x_frame.iloc[[5, 7], 1] = np.nan
    x_frame.iloc[[2, 17, 23], 3] = np.nan
    x_frame["constant"] = 1.0
    x_frame["all_missing"] = np.nan

    actual = _mean_abs_corr(y, x_frame)
    expected = _mean_abs_corr_reference(y, x_frame)

    assert actual == pytest.approx(expected, abs=1e-12)


def test_mean_abs_corr_returns_nan_when_no_usable_exposure() -> None:
    y = pd.Series([1.0, 2.0, np.nan])
    x_frame = pd.DataFrame({"constant": [1.0, 1.0, 1.0], "missing": [np.nan] * 3})

    assert np.isnan(_mean_abs_corr(y, x_frame))
