from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from alpha_lab.custom_factors import (
    BrokenCustomFactorWarning,
    compile_custom_factor,
    custom_factor_meta_path,
    get_custom_factor_source,
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


# ---------------------------------------------------------------------------
# Broken-metadata observability (OPT-P0-3)
#
# The loader intentionally skips broken factor.json files so an unrelated
# draft cannot block a research run, but the skip must be observable.
# ---------------------------------------------------------------------------


def _write_factor_json(dir_path: Path, name: str, payload: object) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    factor_json = dir_path / "factor.json"
    if isinstance(payload, str):
        factor_json.write_text(payload, encoding="utf-8")
    else:
        factor_json.write_text(json.dumps(payload), encoding="utf-8")
    return factor_json


def test_get_custom_factor_source_warns_on_broken_metadata(tmp_path: Path) -> None:
    # One valid factor + one broken JSON file in the same workspace.
    valid_dir = tmp_path / "custom_factors" / "research" / "good_one"
    _write_factor_json(
        valid_dir,
        "good_one",
        {"name": "good_one", "code": CUSTOM_FACTOR_CODE},
    )
    broken_dir = tmp_path / "custom_factors" / "research" / "broken_one"
    _write_factor_json(broken_dir, "broken_one", "this is { not valid json")

    with pytest.warns(BrokenCustomFactorWarning, match="broken_one"):
        source = get_custom_factor_source("good_one", workspace_root=tmp_path)
    assert source is not None
    assert source.name == "good_one"


def test_load_persisted_custom_factors_warns_and_skips_broken(tmp_path: Path) -> None:
    valid_dir = tmp_path / "custom_factors" / "research" / "valid_factor_for_warning_test"
    _write_factor_json(
        valid_dir,
        "valid_factor_for_warning_test",
        {"name": "valid_factor_for_warning_test", "code": CUSTOM_FACTOR_CODE},
    )
    broken_dir = tmp_path / "custom_factors" / "research" / "broken_for_warning_test"
    _write_factor_json(broken_dir, "broken_for_warning_test", "{ not json")

    with pytest.warns(BrokenCustomFactorWarning, match="broken_for_warning_test"):
        sources = load_persisted_custom_factors(tmp_path)

    assert "valid_factor_for_warning_test" in sources
    assert "broken_for_warning_test" not in sources

    factor_registry._builders.pop("valid_factor_for_warning_test", None)


def test_load_persisted_custom_factors_strict_mode_still_raises(tmp_path: Path) -> None:
    broken_dir = tmp_path / "custom_factors" / "research" / "strict_broken"
    _write_factor_json(broken_dir, "strict_broken", "not json")

    with pytest.raises(json.JSONDecodeError):
        load_persisted_custom_factors(tmp_path, ignore_errors=False)


def test_custom_factor_meta_path_warns_on_broken_neighbor(tmp_path: Path) -> None:
    broken_dir = tmp_path / "custom_factors" / "research" / "neighbor_broken"
    _write_factor_json(broken_dir, "neighbor_broken", "not json")

    # Asking for an unrelated name still scans every factor.json on disk;
    # the broken neighbor must produce a warning before we fall back to the
    # default write path.
    with pytest.warns(BrokenCustomFactorWarning, match="neighbor_broken"):
        path = custom_factor_meta_path(tmp_path, "unrelated_factor_name")

    # The function still falls back to a normal write path on miss.
    assert path.name == "factor.json"
