"""Regression tests for target.execution_price_mode plumbing.

The signed_jump_t style (after-close intraday-derived) factor is the
motivating case: signed_jump is only known after close of day ``t``, so the
realistic entry price is open[t+1] and the label must be built with
``execution_price_mode="next_open"``.  These tests pin down that the case
spec, the base feature cache, the experiment runner, and the evaluation
metrics all see and agree on the chosen mode.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import alpha_lab.real_cases.single_factor.evaluate.core as evaluate_core
from alpha_lab.exceptions import AlphaLabConfigError
from alpha_lab.labels import forward_return
from alpha_lab.real_cases.common_spec import TargetSpec
from alpha_lab.real_cases.single_factor.pipeline import (
    prepare_base_features,
)
from alpha_lab.real_cases.single_factor.spec import (
    single_factor_case_spec_from_mapping,
)

# ---------------------------------------------------------------------------
# Fixtures: a small panel with open + close so next_open semantics are valid.
# ---------------------------------------------------------------------------


def _toy_panel(*, n_days: int = 8) -> pd.DataFrame:
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    rows: list[dict[str, Any]] = []
    for asset_idx, asset in enumerate(("A", "B")):
        close = 100.0 + asset_idx
        for t, date in enumerate(dates):
            # Distinct open and close so that "close[t+h]/close[t]" and
            # "close[t+h]/open[t+1]" are numerically different.
            close = close * (1.0 + 0.01 * np.sin(asset_idx + t))
            open_price = close * 0.995
            rows.append(
                {"date": date, "asset": asset, "open": float(open_price), "close": float(close)}
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. forward_return semantics (the canonical formula).
# ---------------------------------------------------------------------------


def test_forward_return_next_open_semantics_pins_canonical_formula() -> None:
    """Verify the exact formula:

    - close mode, horizon=1: close[t+1] / close[t] - 1
    - next_open, horizon=1: close[t+1] / open[t+1] - 1
    - next_open, horizon=5: close[t+5] / open[t+1] - 1
    """
    dates = pd.date_range("2024-01-02", periods=7, freq="B")
    df = pd.DataFrame(
        {
            "date": dates,
            "asset": ["X"] * 7,
            "open": [99.0, 101.0, 111.0, 124.0, 130.0, 140.0, 150.0],
            "close": [100.0, 110.0, 121.0, 133.1, 144.5, 155.0, 168.0],
        }
    )

    close_h1 = forward_return(df, horizon=1, execution_price_mode="close")
    next_open_h1 = forward_return(df, horizon=1, execution_price_mode="next_open")
    next_open_h5 = forward_return(df, horizon=5, execution_price_mode="next_open")

    # close mode horizon=1: close[1]/close[0] - 1 = 110/100 - 1
    assert close_h1["value"].iloc[0] == pytest.approx(110.0 / 100.0 - 1.0)
    # next_open horizon=1: close[1]/open[1] - 1 = 110/101 - 1
    assert next_open_h1["value"].iloc[0] == pytest.approx(110.0 / 101.0 - 1.0)
    # next_open horizon=5: close[5]/open[1] - 1 = 155.0/101.0 - 1
    assert next_open_h5["value"].iloc[0] == pytest.approx(155.0 / 101.0 - 1.0)


# ---------------------------------------------------------------------------
# 2. TargetSpec validation: default close, allowed values, fail-fast on bad.
# ---------------------------------------------------------------------------


def test_default_execution_price_mode_is_close() -> None:
    """No-arg TargetSpec keeps legacy close-mode behaviour."""
    spec = TargetSpec()
    assert spec.execution_price_mode == "close"


def test_target_spec_accepts_next_open() -> None:
    spec = TargetSpec(execution_price_mode="next_open")
    assert spec.execution_price_mode == "next_open"


def test_target_spec_normalizes_case() -> None:
    spec = TargetSpec(execution_price_mode="Next_Open")
    assert spec.execution_price_mode == "next_open"


def test_invalid_execution_price_mode_fails_fast() -> None:
    """Unknown mode must raise immediately, not silently fall back."""
    with pytest.raises(AlphaLabConfigError, match="execution_price_mode"):
        TargetSpec(execution_price_mode="same_close_after_intraday")


def test_case_spec_parses_execution_price_mode_from_mapping() -> None:
    payload = {
        "name": "next_open_case",
        "factor_name": "f",
        "factor_path": "f.csv",
        "prices_path": "p.csv",
        "rebalance_frequency": "W",
        "n_quantiles": 5,
        "direction": "long",
        "target": {
            "kind": "forward_return",
            "horizon": 5,
            "execution_price_mode": "next_open",
        },
        "output": {"root_dir": "outputs"},
    }
    spec = single_factor_case_spec_from_mapping(payload)
    assert spec.target.execution_price_mode == "next_open"


def test_case_spec_default_is_close_when_field_absent() -> None:
    payload = {
        "name": "default_case",
        "factor_name": "f",
        "factor_path": "f.csv",
        "prices_path": "p.csv",
        "rebalance_frequency": "W",
        "n_quantiles": 5,
        "direction": "long",
        "target": {"kind": "forward_return", "horizon": 5},
        "output": {"root_dir": "outputs"},
    }
    spec = single_factor_case_spec_from_mapping(payload)
    assert spec.target.execution_price_mode == "close"


def test_case_spec_rejects_invalid_execution_price_mode() -> None:
    payload = {
        "name": "bad_case",
        "factor_name": "f",
        "factor_path": "f.csv",
        "prices_path": "p.csv",
        "rebalance_frequency": "W",
        "n_quantiles": 5,
        "direction": "long",
        "target": {
            "kind": "forward_return",
            "horizon": 5,
            "execution_price_mode": "same_close_after_intraday",
        },
        "output": {"root_dir": "outputs"},
    }
    with pytest.raises(AlphaLabConfigError, match="execution_price_mode"):
        single_factor_case_spec_from_mapping(payload)


# ---------------------------------------------------------------------------
# 3. prepare_base_features threads execution_price_mode into the label cache.
# ---------------------------------------------------------------------------


def test_prepare_base_features_uses_next_open_when_requested() -> None:
    panel = _toy_panel(n_days=8)

    cache_close = prepare_base_features(
        panel,
        trailing_return_horizons=(1,),
        forward_label_horizons=(1, 5),
        execution_price_mode="close",
    )
    cache_next_open = prepare_base_features(
        panel,
        trailing_return_horizons=(1,),
        forward_label_horizons=(1, 5),
        execution_price_mode="next_open",
    )

    expected_close = forward_return(panel, horizon=1, execution_price_mode="close")
    expected_next_open = forward_return(panel, horizon=1, execution_price_mode="next_open")

    pd.testing.assert_frame_equal(
        cache_close.forward_labels_by_horizon[1].reset_index(drop=True),
        expected_close.reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        cache_next_open.forward_labels_by_horizon[1].reset_index(drop=True),
        expected_next_open.reset_index(drop=True),
    )

    # Factor naming differs by mode — proves the underlying values are not
    # just numerically coincidental, they were built with different formulas.
    assert set(cache_close.forward_labels_by_horizon[1]["factor"]) == {"forward_return_1"}
    assert set(cache_next_open.forward_labels_by_horizon[1]["factor"]) == {
        "forward_return_1_next_open"
    }


# ---------------------------------------------------------------------------
# 4. evaluate_single_factor_case routes execution_price_mode into run_factor_experiment.
# ---------------------------------------------------------------------------


def _build_spec_for_routing_test(execution_price_mode: str | None) -> Any:
    target_payload: dict[str, Any] = {"kind": "forward_return", "horizon": 5}
    if execution_price_mode is not None:
        target_payload["execution_price_mode"] = execution_price_mode
    return single_factor_case_spec_from_mapping(
        {
            "name": "routing_case",
            "factor_name": "f",
            "factor_path": "f.csv",
            "prices_path": "p.csv",
            "rebalance_frequency": "W",
            "n_quantiles": 5,
            "direction": "long",
            "target": target_payload,
            "output": {"root_dir": "outputs"},
        }
    )


def test_single_factor_pipeline_passes_execution_price_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """evaluate_single_factor_case must forward target.execution_price_mode
    into every run_factor_experiment call (headline + dual-scope + raw).
    Silent fallback to close would mask the after-close intraday timing bug.
    """
    spec = _build_spec_for_routing_test("next_open")

    captured: list[dict[str, Any]] = []

    def fake_run_factor_experiment(*args: Any, **kwargs: Any) -> Any:
        captured.append(dict(kwargs))
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(evaluate_core, "run_factor_experiment", fake_run_factor_experiment)

    with pytest.raises(RuntimeError, match="stop after capture"):
        evaluate_core.evaluate_single_factor_case(
            prices=pd.DataFrame({"date": [], "asset": [], "close": []}),
            factor_df=pd.DataFrame({"date": [], "asset": [], "factor": [], "value": []}),
            raw_factor_df=None,
            spec=spec,
            coverage_by_date=pd.DataFrame(),
            neutralization_summary=None,
        )

    assert captured, "run_factor_experiment was never called"
    assert captured[0].get("execution_price_mode") == "next_open"


def test_single_factor_pipeline_default_close_when_not_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = _build_spec_for_routing_test(None)

    captured: list[dict[str, Any]] = []

    def fake_run_factor_experiment(*args: Any, **kwargs: Any) -> Any:
        captured.append(dict(kwargs))
        raise RuntimeError("stop after capture")

    monkeypatch.setattr(evaluate_core, "run_factor_experiment", fake_run_factor_experiment)

    with pytest.raises(RuntimeError, match="stop after capture"):
        evaluate_core.evaluate_single_factor_case(
            prices=pd.DataFrame({"date": [], "asset": [], "close": []}),
            factor_df=pd.DataFrame({"date": [], "asset": [], "factor": [], "value": []}),
            raw_factor_df=None,
            spec=spec,
            coverage_by_date=pd.DataFrame(),
            neutralization_summary=None,
        )

    assert captured[0].get("execution_price_mode") == "close"


# ---------------------------------------------------------------------------
# 5. No factor-value shift under next_open (the failure mode we want to avoid).
# ---------------------------------------------------------------------------


def test_next_open_does_not_shift_factor_values() -> None:
    """Scheme A (this fix) only changes the LABEL entry price.  It must not
    additionally shift the factor itself — doing both would be double lag.

    We verify that ``forward_return(..., next_open)`` reads from the price
    panel directly and does not mutate the (date, asset) index of the
    factor input.  The factor path is independent of execution_price_mode.
    """
    dates = pd.date_range("2024-01-02", periods=5, freq="B")
    factor_df = pd.DataFrame(
        {
            "date": dates,
            "asset": ["A"] * 5,
            "factor": ["signed_jump_neg_5d"] * 5,
            "value": [0.1, -0.2, 0.3, -0.4, 0.5],
        }
    )
    panel = pd.DataFrame(
        {
            "date": dates,
            "asset": ["A"] * 5,
            "open": [99.0, 101.0, 111.0, 124.0, 130.0],
            "close": [100.0, 110.0, 121.0, 133.1, 144.5],
        }
    )

    label_close = forward_return(panel, horizon=1, execution_price_mode="close")
    label_next_open = forward_return(panel, horizon=1, execution_price_mode="next_open")

    # The date index of the label is identical across modes: the label is
    # always stored AT date t (representing a position taken at the relevant
    # entry price).  Only the value differs.
    pd.testing.assert_index_equal(
        pd.Index(label_close["date"]),
        pd.Index(label_next_open["date"]),
    )

    # Merging factor with either label should hit the same rows of factor:
    # next_open does not skip / shift any factor date.
    merged_close = factor_df.merge(
        label_close[["date", "asset", "value"]].rename(columns={"value": "label"}),
        on=["date", "asset"],
        how="inner",
    )
    merged_next_open = factor_df.merge(
        label_next_open[["date", "asset", "value"]].rename(columns={"value": "label"}),
        on=["date", "asset"],
        how="inner",
    )
    pd.testing.assert_series_equal(
        merged_close["value"].reset_index(drop=True),
        merged_next_open["value"].reset_index(drop=True),
        check_names=False,
    )


# ---------------------------------------------------------------------------
# 6. End-to-end smoke through the pipeline using a synthetic case with open.
# ---------------------------------------------------------------------------


def _write_next_open_capable_case(
    tmp_path: Path,
    *,
    execution_price_mode: str = "next_open",
) -> Path:
    """Synthetic single-factor case with open column so next_open is valid."""
    import yaml

    n_days = 200
    n_assets = 6
    rng = np.random.default_rng(20260519)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    assets = [f"S{i:02d}" for i in range(n_assets)]

    price_rows: list[dict[str, Any]] = []
    factor_rows: list[dict[str, Any]] = []
    universe_rows: list[dict[str, Any]] = []
    for i, asset in enumerate(assets):
        close = 50.0 + i
        for _t, date in enumerate(dates):
            ret = rng.normal(0.0, 0.01)
            close = max(close * (1.0 + ret), 1.0)
            open_price = close * (1.0 + rng.normal(0.0, 0.002))
            price_rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "open": float(open_price),
                    "close": float(close),
                    "amount": 1_000_000.0 + i * 1_000.0,
                    "total_mv": 50_000_000.0 + i * 100_000.0,
                }
            )
            factor_rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "factor": "synthetic_jump",
                    "value": float(rng.normal(0.0, 1.0)),
                }
            )
            universe_rows.append(
                {"date": date, "asset": asset, "in_universe": True}
            )

    data_dir = tmp_path / "inputs"
    data_dir.mkdir(parents=True, exist_ok=True)
    prices_path = data_dir / "prices.csv"
    factor_path = data_dir / "synthetic_jump.csv"
    universe_path = data_dir / "universe.csv"
    pd.DataFrame(price_rows).to_csv(prices_path, index=False)
    pd.DataFrame(factor_rows).to_csv(factor_path, index=False)
    pd.DataFrame(universe_rows).to_csv(universe_path, index=False)

    spec_payload: dict[str, Any] = {
        "name": f"synthetic_jump_{execution_price_mode}",
        "factor_name": "synthetic_jump",
        "factor_path": str(factor_path),
        "prices_path": str(prices_path),
        "rebalance_frequency": "W",
        "n_quantiles": 5,
        "direction": "long",
        "universe": {
            "name": "syn_universe",
            "path": str(universe_path),
            "in_universe_column": "in_universe",
        },
        "target": {
            "kind": "forward_return",
            "horizon": 5,
            "execution_price_mode": execution_price_mode,
        },
        "preprocess": {
            "winsorize": True,
            "winsorize_lower": 0.01,
            "winsorize_upper": 0.99,
            "standardization": "zscore",
            "min_group_size": 3,
        },
        "capacity": {"enabled": False, "participation_rate": 0.05, "adv_lookback": 20},
        "transaction_cost": {"one_way_rate": 0.0},
        "output": {"root_dir": str(tmp_path / "outputs")},
    }
    spec_path = tmp_path / f"case_{execution_price_mode}.yaml"
    spec_path.write_text(yaml.safe_dump(spec_payload, sort_keys=False), encoding="utf-8")
    return spec_path


def test_pipeline_smoke_next_open_metrics_record_label_mode(tmp_path: Path) -> None:
    """End-to-end: pipeline runs with next_open and metrics record the mode."""
    from alpha_lab.real_cases.single_factor.pipeline import run_single_factor_case

    spec_path = _write_next_open_capable_case(tmp_path, execution_price_mode="next_open")
    result = run_single_factor_case(spec_path)
    metrics: Mapping[str, Any] = result.evaluation_result.metrics

    assert metrics["target_execution_price_mode"] == "next_open"
    assert metrics["label_mode"] == "next_open"
    assert "next open" in str(metrics["label_entry_assumption"]).lower()


def test_pipeline_smoke_default_close_metrics_record_label_mode(tmp_path: Path) -> None:
    """End-to-end: default close mode keeps the legacy entry convention."""
    from alpha_lab.real_cases.single_factor.pipeline import run_single_factor_case

    spec_path = _write_next_open_capable_case(tmp_path, execution_price_mode="close")
    result = run_single_factor_case(spec_path)
    metrics: Mapping[str, Any] = result.evaluation_result.metrics

    assert metrics["target_execution_price_mode"] == "close"
    assert metrics["label_mode"] == "close"
