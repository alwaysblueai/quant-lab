from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.validation_scaffold import (
    WalkForwardValidationSpec,
    fold_windows_from_summary,
    purged_validation_summary,
)


def test_walk_forward_validation_spec_rejects_non_positive_sizes() -> None:
    with pytest.raises(ValueError, match="train_size"):
        WalkForwardValidationSpec(train_size=0, test_size=5, step=1)


def test_walk_forward_validation_spec_to_dict() -> None:
    spec = WalkForwardValidationSpec(
        train_size=60,
        test_size=20,
        step=20,
        val_size=5,
        purge_periods=2,
        embargo_periods=1,
    )
    payload = spec.to_dict()
    assert payload["scheme"] == "walk_forward"
    assert payload["train_size"] == 60
    assert payload["test_size"] == 20
    assert payload["step"] == 20
    assert payload["val_size"] == 5


def test_fold_windows_from_summary_maps_core_columns() -> None:
    summary = pd.DataFrame(
        [
            {
                "fold_id": 0,
                "train_start": pd.Timestamp("2024-01-01"),
                "train_end": pd.Timestamp("2024-02-01"),
                "start_date": pd.Timestamp("2024-02-02"),
                "end_date": pd.Timestamp("2024-03-01"),
            }
        ]
    )
    out = fold_windows_from_summary(summary)
    assert list(out.columns) == [
        "fold_id",
        "train_start",
        "train_end",
        "val_start",
        "val_end",
        "test_start",
        "test_end",
    ]
    assert out["test_start"].iloc[0] == pd.Timestamp("2024-02-02")
    assert out["test_end"].iloc[0] == pd.Timestamp("2024-03-01")


def test_purged_validation_summary_returns_fold_rows() -> None:
    dates = pd.date_range("2024-01-01", periods=8, freq="B")
    samples = pd.DataFrame(
        {
            "date": dates,
            "event_start": dates,
            "event_end": dates,
        }
    )
    out = purged_validation_summary(samples, n_splits=4, embargo_periods=1)
    assert len(out) == 4
    assert {"fold_id", "n_purged", "n_embargoed"}.issubset(out.columns)
