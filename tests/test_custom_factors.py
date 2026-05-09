from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from alpha_lab.custom_factors import (
    compile_custom_factor,
    custom_factor_meta_path,
    load_persisted_custom_factors,
    sha256_text,
)
from alpha_lab.factor_recipe import factor_registry
from alpha_lab.real_cases.single_factor.pipeline import run_single_factor_case
from tests.single_factor_case_helpers import write_demo_single_factor_case

CUSTOM_FACTOR_CODE = """
def builder(prices, *, window=5, skip_recent=0, min_periods=None, **kwargs):
    import pandas as pd

    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
    value = frame.groupby("asset", sort=False)["close"].pct_change(window)
    if skip_recent:
        value = value.groupby(frame["asset"], sort=False).shift(skip_recent)
    out = frame[["date", "asset"]].copy()
    out["value"] = value
    return out
""".strip()

BUILD_FACTOR_CODE = """
def build_factor(frame):
    import numpy as np
    import pandas as pd

    required = {"close", "volume"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    data = frame.copy()
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["asset", "date"], kind="mergesort")
    ret = data.groupby("asset", sort=False)["close"].pct_change()
    vol = data.groupby("asset", sort=False)["volume"].pct_change()
    value = (ret - vol).replace([np.inf, -np.inf], np.nan)
    return value.reindex(frame.index)
""".strip()


def test_shared_custom_factor_loader_registers_nested_factor(tmp_path: Path) -> None:
    factor_dir = tmp_path / "custom_factors" / "research" / "shared_ret"
    factor_dir.mkdir(parents=True)
    factor_json = factor_dir / "factor.json"
    factor_json.write_text(
        json.dumps(
            {
                "name": "shared_ret",
                "description": "shared loader smoke factor",
                "code": CUSTOM_FACTOR_CODE,
                "required_columns": ["date", "asset", "close"],
                "frequency": "daily",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    sources = load_persisted_custom_factors(tmp_path)

    assert "shared_ret" in factor_registry
    assert sources["shared_ret"].code_sha256 == sha256_text(CUSTOM_FACTOR_CODE)
    assert sources["shared_ret"].required_columns == ("date", "asset", "close")
    assert custom_factor_meta_path(tmp_path, "shared_ret") == factor_json.resolve()

    factor_registry._builders.pop("shared_ret", None)


def test_custom_factor_compile_rejects_missing_builder() -> None:
    try:
        compile_custom_factor("bad_factor", "x = 1")
    except ValueError as exc:
        assert "must define a callable named 'build_factor' or 'builder'" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("compile_custom_factor should reject code without builder")


def test_custom_factor_compile_accepts_build_factor_series_contract() -> None:
    builder = compile_custom_factor("series_contract", BUILD_FACTOR_CODE)
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=4).tolist() * 2,
            "asset": ["A"] * 4 + ["B"] * 4,
            "close": [10, 11, 12, 13, 20, 21, 22, 23],
            "volume": [100, 105, 103, 110, 200, 198, 205, 208],
        }
    )

    out = builder(frame)

    assert list(out.columns) == ["date", "asset", "value"]
    assert len(out) == len(frame)
    assert out["value"].notna().sum() > 0


def test_single_factor_pipeline_audits_custom_factor_source(tmp_path: Path) -> None:
    factor_dir = tmp_path / "custom_factors" / "research" / "custom_ret"
    factor_dir.mkdir(parents=True)
    factor_json = factor_dir / "factor.json"
    factor_json.write_text(
        json.dumps(
            {
                "name": "custom_ret",
                "description": "custom return draft",
                "code": CUSTOM_FACTOR_CODE,
                "required_columns": ["date", "asset", "close"],
                "optional_columns": ["amount"],
                "unavailable_data_policy": "daily close-only proxy; no intraday fields",
                "pit_assumption": "uses trailing close-to-close returns only",
                "frequency": "daily",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="custom_ret")
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    payload["factor_input"] = {
        "mode": "recipe",
        "disable_pipeline_preprocess": True,
        "recipe": {
            "base": {
                "method": "custom_ret",
                "window": 5,
            },
            "preprocess": {
                "winsorize": {"enabled": True, "lower": 0.01, "upper": 0.99},
                "standardization": {"method": "zscore", "min_group_size": 3},
                "min_coverage": 0.2,
            },
        },
    }
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = run_single_factor_case(
        spec_path,
        evaluation_profile="exploratory_screening",
        vault_export_mode="skip",
    )

    expected_code_hash = sha256_text(CUSTOM_FACTOR_CODE)
    manifest = json.loads(result.artifact_paths["run_manifest"].read_text(encoding="utf-8"))
    factor_definition = json.loads(
        result.artifact_paths["factor_definition_json"].read_text(encoding="utf-8")
    )

    assert result.custom_factor_source is not None
    assert manifest["custom_factor_source"]["code_sha256"] == expected_code_hash
    assert manifest["inputs"]["custom_factor_source"]["factor_json_sha256"] == sha256_text(
        factor_json.read_text(encoding="utf-8")
    )
    assert factor_definition["custom_factor_source"]["path"] == str(factor_json.resolve())
    assert factor_definition["custom_factor_source"]["required_columns"] == [
        "date",
        "asset",
        "close",
    ]

    factor_registry._builders.pop("custom_ret", None)
