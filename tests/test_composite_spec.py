from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alpha_lab.real_cases.composite.spec import (
    composite_case_spec_from_mapping,
    load_composite_case_spec,
)
from tests.composite_case_helpers import write_demo_composite_case


def test_load_composite_spec_resolves_relative_paths(tmp_path: Path) -> None:
    data_dir = tmp_path / "case"
    data_dir.mkdir(parents=True, exist_ok=True)

    spec_payload = {
        "name": "spec_resolve_case",
        "prices_path": "prices.csv",
        "rebalance_frequency": "M",
        "n_quantiles": 5,
        "target": {"kind": "forward_return", "horizon": 5},
        "components": [
            {
                "name": "bp",
                "path": "bp.csv",
                "weight": 0.5,
                "direction": "positive",
                "transform": "zscore",
            },
            {
                "name": "roe_ttm",
                "path": "roe_ttm.csv",
                "weight": 0.5,
                "direction": -1,
                "transform": "rank",
            },
        ],
        "output": {"root_dir": "outputs"},
    }
    spec_path = data_dir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_payload, sort_keys=False), encoding="utf-8")

    spec = load_composite_case_spec(spec_path)

    assert Path(spec.prices_path).is_absolute()
    assert Path(spec.components[0].path).is_absolute()
    assert spec.components[1].direction == "negative"
    assert spec.components[1].transform == "rank"


def test_spec_rejects_duplicate_component_names() -> None:
    payload = {
        "name": "dup_component_case",
        "prices_path": "prices.csv",
        "rebalance_frequency": "W",
        "n_quantiles": 5,
        "target": {"kind": "forward_return", "horizon": 5},
        "components": [
            {
                "name": "bp",
                "path": "bp.csv",
                "weight": 0.5,
                "direction": "positive",
                "transform": "zscore",
            },
            {
                "name": "bp",
                "path": "roe.csv",
                "weight": 0.5,
                "direction": "positive",
                "transform": "zscore",
            },
        ],
    }
    with pytest.raises(ValueError, match="unique"):
        composite_case_spec_from_mapping(payload)


def test_demo_spec_fixture_is_loadable(tmp_path: Path) -> None:
    spec_path = write_demo_composite_case(tmp_path)
    spec = load_composite_case_spec(spec_path)
    assert spec.name == "demo_value_quality_lowvol"
    assert len(spec.components) == 3
    assert spec.target.horizon == 5
