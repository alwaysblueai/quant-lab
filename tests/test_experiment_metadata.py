from __future__ import annotations

import pandas as pd
import pytest

from alpha_lab.experiment_metadata import ExperimentMetadata, ValidationMetadata
from alpha_lab.timing import DelaySpec


def test_validation_metadata_to_dict_serialises_timestamps() -> None:
    vm = ValidationMetadata(
        scheme="time_split",
        train_end=pd.Timestamp("2024-01-31"),
        test_start=pd.Timestamp("2024-02-01"),
        val_start=pd.Timestamp("2024-01-20"),
        purge_periods=2,
        embargo_periods=1,
    )
    payload = vm.to_dict()
    assert payload["scheme"] == "time_split"
    assert payload["train_end"] == "2024-01-31T00:00:00"
    assert payload["test_start"] == "2024-02-01T00:00:00"
    assert payload["val_start"] == "2024-01-20T00:00:00"
    assert payload["purge_periods"] == 2
    assert payload["embargo_periods"] == 1


def test_experiment_metadata_rejects_non_positive_trial_count() -> None:
    with pytest.raises(ValueError, match="trial_count"):
        ExperimentMetadata(trial_count=0)


def test_experiment_metadata_to_dict_contains_validation_and_delay() -> None:
    vm = ValidationMetadata(scheme="full_sample")
    delay = DelaySpec.for_horizon(5)
    md = ExperimentMetadata(
        hypothesis="value factor should mean-revert",
        validation=vm,
        delay=delay,
        assumptions=("no survivorship bias",),
    )
    payload = md.to_dict()
    assert payload["hypothesis"] == "value factor should mean-revert"
    validation = payload["validation"]
    assert isinstance(validation, dict)
    assert validation["scheme"] == "full_sample"
    delay_payload = payload["delay"]
    assert isinstance(delay_payload, dict)
    assert delay_payload["return_horizon_periods"] == 5
