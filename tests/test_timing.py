from __future__ import annotations

import pytest

from alpha_lab.timing import DelaySpec, LabelMetadata


def test_delay_spec_for_horizon_sets_label_window() -> None:
    spec = DelaySpec.for_horizon(5, purge_periods=2, embargo_periods=1)
    assert spec.return_horizon_periods == 5
    assert spec.label_start_offset_periods == 0
    assert spec.label_end_offset_periods == 5
    assert spec.purge_periods == 2
    assert spec.embargo_periods == 1


def test_delay_spec_rejects_invalid_label_offsets() -> None:
    with pytest.raises(ValueError, match="label_end_offset_periods"):
        DelaySpec(label_start_offset_periods=2, label_end_offset_periods=2)


def test_label_metadata_to_dict_contains_nested_delay() -> None:
    delay = DelaySpec.for_horizon(3)
    meta = LabelMetadata(label_name="forward_return_3", horizon_periods=3, delay=delay)
    as_dict = meta.to_dict()
    assert as_dict["label_name"] == "forward_return_3"
    assert as_dict["horizon_periods"] == 3
    delay_dict = as_dict["delay"]
    assert isinstance(delay_dict, dict)
    assert delay_dict["return_horizon_periods"] == 3
