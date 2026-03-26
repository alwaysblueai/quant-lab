from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alpha_lab.real_cases.single_factor.spec import (
    load_single_factor_case_spec,
    single_factor_case_spec_from_mapping,
)
from tests.single_factor_case_helpers import write_demo_single_factor_case


def test_load_single_factor_spec_resolves_relative_paths(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "name": "spec_single_factor_case",
        "factor_name": "bp",
        "factor_path": "bp.csv",
        "prices_path": "prices.csv",
        "rebalance_frequency": "M",
        "n_quantiles": 5,
        "direction": "short",
        "target": {"kind": "forward_return", "horizon": 5},
        "output": {"root_dir": "outputs"},
    }

    spec_path = case_dir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    spec = load_single_factor_case_spec(spec_path)
    assert Path(spec.factor_path).is_absolute()
    assert Path(spec.prices_path).is_absolute()
    assert spec.direction == "short"


def test_single_factor_spec_rejects_invalid_direction() -> None:
    payload = {
        "name": "bad_direction_case",
        "factor_name": "bp",
        "factor_path": "bp.csv",
        "prices_path": "prices.csv",
        "rebalance_frequency": "M",
        "direction": "invalid",
    }
    with pytest.raises(ValueError, match="direction"):
        single_factor_case_spec_from_mapping(payload)


def test_demo_single_factor_spec_fixture_is_loadable(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="roe_ttm")
    spec = load_single_factor_case_spec(spec_path)
    assert spec.factor_name == "roe_ttm"
    assert spec.target.horizon == 5
    assert spec.n_quantiles == 5
