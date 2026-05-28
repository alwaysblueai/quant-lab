"""Tests for memory-aware price loading in the single-factor pipeline.

Covers the column-projection contract (file vs recipe mode), profile-aware
trailing-return precompute, and the bundle-reuse compatibility guard that keeps
a narrowly projected (file-mode) bundle from being reused for a recipe spec.
"""

from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

import alpha_lab.real_cases.single_factor.pipeline as sf_pipeline
from alpha_lab.custom_factors import CustomFactorSource
from alpha_lab.exceptions import AlphaLabConfigError
from alpha_lab.real_cases.single_factor.pipeline import (
    _ensure_bundle_compatible,
    _resolve_prices_optional_columns,
    load_standard_inputs,
)
from alpha_lab.real_cases.single_factor.spec import (
    FactorInputSpec,
    load_single_factor_case_spec,
)
from tests.single_factor_case_helpers import write_demo_single_factor_case

# A representative intraday-derived column and a column outside every projection
# allowlist; neither should survive a file-mode load.
_INTRADAY_COLUMN = "rv_5m"
_UNUSED_WIDE_COLUMN = "totally_unused_wide_feature"


def _recipe_spec(spec):
    """A recipe-mode variant of ``spec`` (recipe body is irrelevant here)."""
    return replace(spec, factor_input=FactorInputSpec(mode="recipe", recipe={}))


def _recipe_spec_with_method(spec, method: str):
    """A recipe-mode variant whose base method names a (custom-draft) factor."""
    return replace(
        spec,
        factor_input=FactorInputSpec(mode="recipe", recipe={"base": {"method": method}}),
    )


def _fake_custom_factor_source(
    name: str,
    *,
    required_columns: tuple[str, ...],
    optional_columns: tuple[str, ...] = (),
) -> CustomFactorSource:
    return CustomFactorSource(
        name=name,
        scope="research",
        path=Path("/tmp") / name / "factor.json",
        code="def build(prices):\n    return prices",
        code_sha256="0" * 64,
        factor_json_sha256="1" * 64,
        required_columns=required_columns,
        optional_columns=optional_columns,
    )


def test_resolve_prices_optional_columns_file_mode_excludes_intraday(tmp_path: Path) -> None:
    spec = load_single_factor_case_spec(
        write_demo_single_factor_case(tmp_path, factor_name="bp")
    )

    optional = _resolve_prices_optional_columns(spec)

    # Daily columns + market-cap candidates are projected; intraday features are not.
    assert "open" in optional
    assert "amount" in optional
    assert "total_mv" in optional
    assert _INTRADAY_COLUMN not in optional
    assert "ret_morning" not in optional


def test_resolve_prices_optional_columns_recipe_mode_includes_intraday(tmp_path: Path) -> None:
    spec = load_single_factor_case_spec(
        write_demo_single_factor_case(tmp_path, factor_name="bp")
    )

    optional = _resolve_prices_optional_columns(_recipe_spec(spec))

    # A recipe may reference intraday inputs, so they must be loaded.
    assert _INTRADAY_COLUMN in optional
    assert "amount" in optional
    assert "total_mv" in optional


def test_load_standard_inputs_projects_away_unused_wide_columns(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")
    spec = load_single_factor_case_spec(spec_path)
    prices_csv = Path(spec.prices_path)

    # Widen the panel with an intraday feature and an unrelated column, then make a
    # fresh parquet sibling so the loader reads parquet (where pruning truly skips IO).
    prices = pd.read_csv(prices_csv)
    prices[_INTRADAY_COLUMN] = 0.5
    prices[_UNUSED_WIDE_COLUMN] = 1.0
    prices_parquet = prices_csv.with_suffix(".parquet")
    prices.to_parquet(prices_parquet, index=False)
    now = time.time()
    os.utime(prices_csv, (now, now))
    os.utime(prices_parquet, (now + 5.0, now + 5.0))

    bundle = load_standard_inputs(spec)

    panel_columns = set(bundle.prices_panel.columns)
    # The columns the backtest actually reads survive projection.
    assert {"date", "asset", "close", "open", "amount", "total_mv"}.issubset(panel_columns)
    # The wide intraday/unused columns are dropped (file mode).
    assert _INTRADAY_COLUMN not in panel_columns
    assert _UNUSED_WIDE_COLUMN not in panel_columns
    assert _INTRADAY_COLUMN not in set(bundle.prices_optional_columns)


def test_exploratory_profile_skips_trailing_returns(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")

    default_bundle = load_standard_inputs(spec_path, evaluation_profile="default_research")
    exploratory_bundle = load_standard_inputs(
        spec_path, evaluation_profile="exploratory_screening"
    )

    # Default research keeps the trailing-return columns it always has.
    assert default_bundle.base_feature_cache.trailing_return_columns == (
        "ret_1d",
        "ret_5d",
        "ret_10d",
        "ret_20d",
        "ret_60d",
    )
    assert {"ret_1d", "ret_60d"}.issubset(set(default_bundle.prices_panel.columns))

    # Exploratory screening skips them (no consumer in the evaluation path).
    assert exploratory_bundle.base_feature_cache.trailing_return_columns == ()
    assert "ret_1d" not in set(exploratory_bundle.prices_panel.columns)
    assert "ret_60d" not in set(exploratory_bundle.prices_panel.columns)

    # Forward labels are unaffected by profile (IC decay needs them under all profiles).
    assert {1, 5, 10, 20}.issubset(
        set(exploratory_bundle.base_feature_cache.forward_labels_by_horizon)
    )


def test_file_mode_bundle_rejected_for_recipe_spec(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")
    spec = load_single_factor_case_spec(spec_path)

    file_bundle = load_standard_inputs(spec)
    # A file-mode bundle lacks the intraday columns a recipe spec may need, so
    # reusing it for a recipe spec on the same prices must be rejected early.
    with pytest.raises(AlphaLabConfigError, match="prices_optional_columns"):
        _ensure_bundle_compatible(file_bundle, spec=_recipe_spec(spec))


def test_recipe_bundle_accepted_for_file_spec(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")
    spec = load_single_factor_case_spec(spec_path)

    # A recipe bundle is a superset projection, so a file spec can safely reuse it.
    recipe_bundle = load_standard_inputs(_recipe_spec(spec))
    _ensure_bundle_compatible(recipe_bundle, spec=spec)


# --- P0.5: recipe-mode precise projection from factor.json -------------------


def test_recipe_projection_precise_from_resolved_custom_factor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_single_factor_case_spec(
        write_demo_single_factor_case(tmp_path, factor_name="bp")
    )
    method = "my_intraday_factor"
    source = _fake_custom_factor_source(
        method,
        required_columns=("ret_last30", "amihud_intraday", "is_limit_down"),
        optional_columns=("is_suspended",),
    )
    monkeypatch.setattr(
        sf_pipeline, "load_persisted_custom_factors", lambda *a, **k: {method: source}
    )

    optional = set(_resolve_prices_optional_columns(_recipe_spec_with_method(spec, method)))

    # Declared factor columns are projected.
    assert {"ret_last30", "amihud_intraday", "is_limit_down", "is_suspended"}.issubset(optional)
    # Backtest/capacity daily columns + market-cap candidates are always projected.
    assert {"open", "vwap", "amount", "total_mv"}.issubset(optional)
    # Undeclared intraday columns are NOT pulled in (this is the memory win).
    assert "ret_morning" not in optional
    assert "rv_5m" not in optional
    assert "amount_share_close30" not in optional


def test_recipe_projection_falls_back_wide_when_factor_unresolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_single_factor_case_spec(
        write_demo_single_factor_case(tmp_path, factor_name="bp")
    )
    # No matching custom factor in the workspace -> conservative wide projection.
    monkeypatch.setattr(sf_pipeline, "load_persisted_custom_factors", lambda *a, **k: {})

    optional = set(
        _resolve_prices_optional_columns(_recipe_spec_with_method(spec, "unknown_method"))
    )

    # Fallback keeps the full intraday set so an unresolved recipe never breaks.
    assert {"ret_morning", "rv_5m", "amount_share_close30"}.issubset(optional)


def test_recipe_precise_vs_fallback_get_distinct_bundle_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_single_factor_case_spec(
        write_demo_single_factor_case(tmp_path, factor_name="bp")
    )
    method = "my_intraday_factor"
    source = _fake_custom_factor_source(method, required_columns=("ret_last30",))
    monkeypatch.setattr(
        sf_pipeline, "load_persisted_custom_factors", lambda *a, **k: {method: source}
    )

    precise = sf_pipeline._resolved_input_bundle_key(_recipe_spec_with_method(spec, method))
    fallback = sf_pipeline._resolved_input_bundle_key(
        _recipe_spec_with_method(spec, "some_other_unresolved")
    )
    # Different projected column sets must not share a reusable input bundle.
    assert precise != fallback


def test_load_standard_inputs_warns_on_recipe_projection_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = load_single_factor_case_spec(
        write_demo_single_factor_case(tmp_path, factor_name="bp")
    )
    monkeypatch.setattr(sf_pipeline, "load_persisted_custom_factors", lambda *a, **k: {})

    with pytest.warns(UserWarning, match="precise column pruning"):
        load_standard_inputs(_recipe_spec_with_method(spec, "unregistered_factor"))
