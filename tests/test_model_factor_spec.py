from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from alpha_lab.real_cases.model_factor.spec import (
    load_model_factor_case_spec,
    model_factor_case_spec_from_mapping,
)
from tests.model_factor_case_helpers import write_demo_model_factor_case


def test_load_model_factor_spec_resolves_relative_paths(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    case_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "name": "spec_model_factor_case",
        "factor_name": "ml_score",
        "features_path": "features.csv",
        "feature_columns": ["x1", "x2"],
        "prices_path": "prices.csv",
        "rebalance_frequency": "M",
        "n_quantiles": 5,
        "direction": "short",
        "model": {"family": "ridge", "params": {"alpha": 1.0}},
        "training": {
            "window_type": "rolling",
            "train_window_n_dates": 20,
            "min_train_dates": 10,
            "min_train_rows": 30,
            "retrain_every_n_dates": 2,
            "min_score_assets": 5,
        },
        "target": {"kind": "forward_return", "horizon": 5},
        "output": {"root_dir": "outputs"},
    }

    spec_path = case_dir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    spec = load_model_factor_case_spec(spec_path)
    assert Path(spec.features_path).is_absolute()
    assert Path(spec.prices_path).is_absolute()
    assert spec.direction == "short"


def test_model_factor_spec_rejects_reserved_feature_column() -> None:
    payload = {
        "name": "bad_feature_case",
        "factor_name": "ml_score",
        "features_path": "features.csv",
        "feature_columns": ["date", "x2"],
        "prices_path": "prices.csv",
        "rebalance_frequency": "M",
        "model": {"family": "ridge"},
        "training": {
            "window_type": "rolling",
            "train_window_n_dates": 20,
            "min_train_dates": 10,
            "min_train_rows": 30,
            "retrain_every_n_dates": 2,
            "min_score_assets": 5,
        },
    }
    with pytest.raises(ValueError, match="reserved"):
        model_factor_case_spec_from_mapping(payload)


def test_demo_model_factor_spec_fixture_is_loadable(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)
    assert spec.factor_name == "ml_score"
    assert spec.target.horizon == 5
    assert spec.n_quantiles == 5
    assert spec.model.family == "ridge"
