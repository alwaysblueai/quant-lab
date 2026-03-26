from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha_lab.composite_signals import compose_signals


def _make_signals(seed: int = 11) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=25, freq="B")
    assets = [f"A{i:02d}" for i in range(20)]
    rows_s: list[dict[str, object]] = []
    rows_l: list[dict[str, object]] = []
    for date in dates:
        latent = rng.normal(0, 1, size=len(assets))
        label = latent + rng.normal(0, 0.5, size=len(assets))
        good = latent + rng.normal(0, 0.2, size=len(assets))
        bad = rng.normal(0, 1, size=len(assets))
        for i, asset in enumerate(assets):
            rows_s.append({"date": date, "asset": asset, "factor": "good", "value": float(good[i])})
            rows_s.append({"date": date, "asset": asset, "factor": "bad", "value": float(bad[i])})
            rows_l.append(
                {
                    "date": date,
                    "asset": asset,
                    "factor": "label",
                    "value": float(label[i]),
                }
            )
    return pd.DataFrame(rows_s), pd.DataFrame(rows_l)


def test_compose_signals_equal_weights() -> None:
    signals, labels = _make_signals()
    result = compose_signals(signals, method="equal", labels=labels)
    assert {"date", "asset", "factor", "value"} == set(result.composite.columns)
    by_date = result.weights.groupby("date", sort=True)["weight"].sum()
    assert np.allclose(by_date.to_numpy(dtype=float), 1.0)


def test_compose_signals_ic_weights_favor_predictive_factor() -> None:
    signals, labels = _make_signals()
    result = compose_signals(signals, method="ic", labels=labels, lookback=10, min_history=5)
    w = result.weights.groupby("factor", sort=True)["weight"].mean()
    assert w["good"] > w["bad"]


def test_compose_signals_icir_weights_sum_to_one_by_date() -> None:
    signals, labels = _make_signals()
    result = compose_signals(signals, method="icir", labels=labels, lookback=10, min_history=5)
    by_date = result.weights.groupby("date", sort=True)["weight"].sum()
    assert np.allclose(by_date.to_numpy(dtype=float), 1.0)


def test_compose_signals_requires_labels_for_ic_methods() -> None:
    signals, _ = _make_signals()
    with pytest.raises(ValueError, match="labels is required"):
        compose_signals(signals, method="ic")
