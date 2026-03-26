from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.labels import (
    UNIFIED_LABEL_COLUMNS,
    rankpct_label,
    regression_forward_label,
    trend_scanning_labels,
    triple_barrier_labels,
    validate_unified_label_table,
)


def _make_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=12, freq="B")
    rows: list[dict[str, object]] = []
    for asset, path in {
        "A": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        "B": [100, 99, 98, 97, 96, 97, 98, 99, 100, 101, 102, 103],
    }.items():
        for i, date in enumerate(dates):
            rows.append({"date": date, "asset": asset, "close": float(path[i])})
    return pd.DataFrame(rows)


def test_regression_forward_label_schema() -> None:
    result = regression_forward_label(_make_prices(), horizon=2)
    assert tuple(result.labels.columns) == UNIFIED_LABEL_COLUMNS
    assert result.metadata["label_type"] == "regression"
    validate_unified_label_table(result.labels)


def test_rankpct_label_range() -> None:
    result = rankpct_label(_make_prices(), horizon=1)
    vals = result.labels["label_value"].dropna()
    assert (vals >= 0).all()
    assert (vals <= 1).all()
    assert result.metadata["label_type"] == "ranking"


def test_triple_barrier_labels_have_valid_triggers() -> None:
    result = triple_barrier_labels(
        _make_prices(),
        horizon=3,
        pt_mult=0.5,
        sl_mult=0.5,
        volatility_lookback=3,
    )
    triggers = set(result.labels["trigger"].dropna().astype(str).unique())
    assert (
        "upper_barrier" in triggers
        or "lower_barrier" in triggers
        or "vertical_barrier" in triggers
    )
    assert result.metadata["label_type"] == "event_classification"


def test_trend_scanning_labels_emit_confidence() -> None:
    result = trend_scanning_labels(
        _make_prices(),
        min_horizon=2,
        max_horizon=4,
    )
    conf = result.labels["confidence"].dropna()
    assert len(conf) > 0
    assert (conf >= 0).all()


def test_validate_unified_label_table_rejects_duplicates() -> None:
    labels = regression_forward_label(_make_prices(), horizon=1).labels
    bad = pd.concat([labels, labels.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_unified_label_table(bad)


def test_trend_scanning_invalid_horizon_raises() -> None:
    with pytest.raises(ValueError, match="max_horizon"):
        trend_scanning_labels(_make_prices(), min_horizon=4, max_horizon=3)


def test_triple_barrier_invalid_parameters_raise() -> None:
    with pytest.raises(ValueError, match="pt_mult"):
        triple_barrier_labels(_make_prices(), horizon=5, pt_mult=0.0, sl_mult=1.0)


def test_rankpct_label_all_nan_when_horizon_too_large() -> None:
    result = rankpct_label(_make_prices(), horizon=100)
    assert result.labels["label_value"].isna().all()
