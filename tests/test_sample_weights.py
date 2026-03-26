from __future__ import annotations

import math

import pandas as pd
import pytest

from alpha_lab.sample_weights import (
    build_sample_weights,
    combine_weight_components,
    concurrency_by_date,
    confidence_weights,
    return_magnitude_weights,
    time_decay_weights,
    uniqueness_weights,
)


def _events() -> pd.DataFrame:
    d = pd.date_range("2024-01-01", periods=6, freq="B")
    return pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "date": [d[0], d[1], d[4]],
            "event_start": [d[0], d[1], d[4]],
            "event_end": [d[2], d[2], d[4]],
            "label_value": [0.10, 0.02, 0.05],
            "confidence": [0.9, 0.3, 0.7],
        }
    )


def test_concurrency_by_date_counts_active_events() -> None:
    out = concurrency_by_date(_events())
    peak = out["concurrency"].max()
    assert peak >= 2


def test_uniqueness_weights_favor_less_overlapped_event() -> None:
    w = uniqueness_weights(_events(), sample_id_col="sample_id")
    assert math.isclose(float(w.sum()), 1.0)
    assert w["c"] > w["a"]
    assert w["c"] > w["b"]


def test_return_magnitude_weights_normalized_and_clipped() -> None:
    vals = pd.Series([0.01, 0.02, 10.0], index=["x", "y", "z"])
    w = return_magnitude_weights(vals, clip_quantile=0.8)
    assert math.isclose(float(w.sum()), 1.0)
    assert w["z"] < 0.999


def test_time_decay_weights_assign_higher_weight_to_recent_rows() -> None:
    d = pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-05"])
    w = time_decay_weights(pd.Series(d), half_life_periods=1.0)
    assert w.iloc[-1] > w.iloc[0]
    assert math.isclose(float(w.sum()), 1.0)


def test_confidence_weights_in_bounds() -> None:
    s = pd.Series([0.2, 0.5, 1.0], index=["a", "b", "c"])
    w = confidence_weights(s, min_weight=0.2, max_weight=1.0)
    assert (w >= 0).all()
    assert math.isclose(float(w.sum()), 1.0)


def test_combine_weight_components_multiplies_and_normalizes() -> None:
    c1 = pd.Series([0.5, 0.5], index=["a", "b"])
    c2 = pd.Series([0.2, 0.8], index=["a", "b"])
    out = combine_weight_components({"c1": c1, "c2": c2})
    assert math.isclose(float(out.sum()), 1.0)
    assert out["b"] > out["a"]


def test_build_sample_weights_includes_components_and_total_weight() -> None:
    result = build_sample_weights(
        _events(),
        sample_id_col="sample_id",
        return_col="label_value",
        confidence_col="confidence",
        half_life_periods=2.0,
    )
    cols = set(result.weights.columns)
    assert "sample_weight" in cols
    assert "weight_uniqueness" in cols
    assert "weight_return_magnitude" in cols
    assert "weight_confidence" in cols
    assert "weight_time_decay" in cols
    assert math.isclose(float(result.weights["sample_weight"].sum()), 1.0)


def test_build_sample_weights_rejects_duplicate_sample_id() -> None:
    bad = _events().copy()
    bad.loc[1, "sample_id"] = "a"
    with pytest.raises(ValueError, match="duplicate"):
        build_sample_weights(bad, sample_id_col="sample_id")
