"""Tests for the fast_screen package: Tier-1 contracts, gating, artifacts, Tier-2 runner."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import alpha_lab.evaluation as evaluation_module
import alpha_lab.experiment as experiment_module
from alpha_lab.factors.momentum import momentum
from alpha_lab.fast_screen import (
    CORE_CHART_KEYS,
    CORE_METRIC_KEYS,
    TIER2_MODULES,
    MetricStatus,
    Tier1Inputs,
    evaluate_gates,
    load_tier1_result,
    load_tier2_index,
    run_tier1,
    run_tier2_modules,
    save_tier1_result,
)
from alpha_lab.fast_screen.cli import _prepare_inputs
from alpha_lab.fast_screen.contracts import (
    FastScreenResult,
    MetricCard,
    Verdict,
)
from alpha_lab.real_cases.single_factor.spec import load_single_factor_case_spec
from tests.single_factor_case_helpers import write_demo_single_factor_case


def _prices(n_assets: int = 8, n_days: int = 60, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-02", periods=n_days, freq="B")
    assets = [f"A{i}" for i in range(n_assets)]
    rows = []
    for asset in assets:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


def _momentum_factor(prices: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    return momentum(prices, window=window)


def _tier1_inputs(**overrides) -> Tier1Inputs:
    prices = _prices()
    factor_df = _momentum_factor(prices)
    base = dict(
        factor_name="momentum",
        factor_df=factor_df,
        prices=prices,
        horizon=1,
        n_quantiles=5,
        cost_rate=0.001,
        universe="synth",
        frequency="daily",
    )
    base.update(overrides)
    return Tier1Inputs(**base)


def test_tier1_produces_ten_metrics_four_charts_in_canonical_order():
    result = run_tier1(_tier1_inputs())
    assert [m.key for m in result.metrics] == list(CORE_METRIC_KEYS)
    assert [c.key for c in result.charts] == list(CORE_CHART_KEYS)
    assert result.verdict.status in {"pass", "warn", "fail"}
    assert result.inputs_hash
    assert result.window["start"] <= result.window["end"]


def test_metric_status_is_enum_not_raw_zero():
    result = run_tier1(_tier1_inputs())
    for card in result.metrics:
        assert isinstance(card.status, MetricStatus)
        if card.status is MetricStatus.COMPUTED:
            # Computed metric must have a real number or structured secondary.
            assert card.value is not None or card.secondary


def test_gates_fail_on_weak_ic_and_low_sharpe():
    weak = FastScreenResult(
        factor_name="weak",
        run_id="r1",
        universe="u",
        frequency="daily",
        window={"start": "2024-01-02", "end": "2024-03-28"},
        metrics=[
            MetricCard("mean_rank_ic", "Mean RankIC", 0.001, MetricStatus.COMPUTED),
            MetricCard("rank_ic_ir", "RankIC IR", 0.05, MetricStatus.COMPUTED),
            MetricCard("ic_positive_ratio", "pos", 0.5, MetricStatus.COMPUTED),
            MetricCard(
                "group_monotonicity",
                "mono",
                0.0,
                MetricStatus.COMPUTED,
                secondary={"kendall_tau": 0.0},
            ),
            MetricCard("ic_half_life", "hl", 2.0, MetricStatus.COMPUTED),
            MetricCard("turnover", "trn", 0.3, MetricStatus.COMPUTED),
            MetricCard(
                "coverage",
                "cov",
                100.0,
                MetricStatus.PARTIAL,
                secondary={"effective_days": 100, "avg_n_assets": 5.0},
            ),
            MetricCard("ls_sharpe_net", "shp", 0.1, MetricStatus.COMPUTED),
            MetricCard("ic_t_stat", "tstat", 0.4, MetricStatus.COMPUTED),
            MetricCard("max_drawdown", "mdd", -0.3, MetricStatus.COMPUTED),
        ],
        charts=[],
        verdict=Verdict(status="pass", triggered_rules=[], next_step=""),
        inputs_hash="h",
        generated_at="2026-04-19T00:00:00+00:00",
    )
    v = evaluate_gates(weak)
    assert v.status == "fail"
    # At least three failure conditions should fire; only three surfaced.
    assert len(v.triggered_rules) == 3


def test_gates_pass_on_strong_factor():
    strong = FastScreenResult(
        factor_name="strong",
        run_id="r1",
        universe="u",
        frequency="daily",
        window={"start": "2024-01-02", "end": "2025-06-01"},
        metrics=[
            MetricCard("mean_rank_ic", "", 0.08, MetricStatus.COMPUTED),
            MetricCard("rank_ic_ir", "", 0.9, MetricStatus.COMPUTED),
            MetricCard("ic_positive_ratio", "", 0.62, MetricStatus.COMPUTED),
            MetricCard(
                "group_monotonicity",
                "",
                0.04,
                MetricStatus.COMPUTED,
                secondary={"kendall_tau": 0.9},
            ),
            MetricCard("ic_half_life", "", 5.0, MetricStatus.COMPUTED),
            MetricCard("turnover", "", 0.3, MetricStatus.COMPUTED),
            MetricCard(
                "coverage",
                "",
                400.0,
                MetricStatus.COMPUTED,
                secondary={"effective_days": 400, "avg_n_assets": 500},
            ),
            MetricCard("ls_sharpe_net", "", 1.8, MetricStatus.COMPUTED),
            MetricCard("ic_t_stat", "", 3.2, MetricStatus.COMPUTED),
            MetricCard("max_drawdown", "", -0.12, MetricStatus.COMPUTED),
        ],
        charts=[],
        verdict=Verdict(status="pass", triggered_rules=[], next_step=""),
        inputs_hash="h",
        generated_at="now",
    )
    v = evaluate_gates(strong)
    assert v.status == "pass"
    assert v.triggered_rules == []


def test_gate_integrity_failure_forces_fail():
    result = run_tier1(_tier1_inputs())
    # Replace verdict via a manual evaluate_gates call with integrity False.
    v = evaluate_gates(result, integrity_passed=False)
    assert v.status == "fail"
    assert any("integrity" in r for r in v.triggered_rules)


def test_artifact_roundtrip(tmp_path: Path):
    result = run_tier1(_tier1_inputs())
    paths = save_tier1_result(tmp_path, result)
    assert paths.tier1_result.exists()
    loaded = load_tier1_result(tmp_path, result.factor_name, result.run_id)
    assert [m.key for m in loaded.metrics] == list(CORE_METRIC_KEYS)
    assert loaded.verdict.status == result.verdict.status
    assert loaded.inputs_hash == result.inputs_hash


def test_tier2_registry_has_expected_keys():
    keys = {m.key for m in TIER2_MODULES}
    assert {"conditional_ic", "coverage_ts", "turnover_ts", "integrity_full"}.issubset(keys)


def test_tier2_unknown_module_records_failure(tmp_path: Path):
    tier1 = run_tier1(_tier1_inputs())
    save_tier1_result(tmp_path, tier1)
    statuses = run_tier2_modules(
        _tier1_inputs(),
        artifact_root=tmp_path,
        factor_name=tier1.factor_name,
        run_id=tier1.run_id,
        modules=["does_not_exist"],
        inputs_hash=tier1.inputs_hash,
    )
    assert statuses["does_not_exist"].status is MetricStatus.FAILED
    index = load_tier2_index(tmp_path, tier1.factor_name, tier1.run_id)
    assert "does_not_exist" in index


def test_tier2_coverage_ts_roundtrip(tmp_path: Path):
    tier1 = run_tier1(_tier1_inputs())
    save_tier1_result(tmp_path, tier1)
    statuses = run_tier2_modules(
        _tier1_inputs(),
        artifact_root=tmp_path,
        factor_name=tier1.factor_name,
        run_id=tier1.run_id,
        modules=["coverage_ts"],
        inputs_hash=tier1.inputs_hash,
    )
    assert statuses["coverage_ts"].status is MetricStatus.COMPUTED
    result_path = (
        tmp_path / tier1.factor_name / tier1.run_id / "tier2" / "coverage_ts" / "result.json"
    )
    payload = json.loads(result_path.read_text())
    assert "coverage_ts" in payload
    assert len(payload["coverage_ts"]) > 0


def test_tier2_turnover_ts_does_not_rerun_full_experiment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    tier1 = run_tier1(_tier1_inputs())
    save_tier1_result(tmp_path, tier1)

    def _fail_run_factor_experiment(*args, **kwargs):
        del args, kwargs
        raise AssertionError("turnover_ts should not call run_factor_experiment")

    monkeypatch.setattr(experiment_module, "run_factor_experiment", _fail_run_factor_experiment)
    statuses = run_tier2_modules(
        _tier1_inputs(),
        artifact_root=tmp_path,
        factor_name=tier1.factor_name,
        run_id=tier1.run_id,
        modules=["turnover_ts"],
        inputs_hash=tier1.inputs_hash,
    )
    assert statuses["turnover_ts"].status is MetricStatus.COMPUTED
    result_path = (
        tmp_path / tier1.factor_name / tier1.run_id / "tier2" / "turnover_ts" / "result.json"
    )
    payload = json.loads(result_path.read_text())
    assert "turnover_ts" in payload
    assert isinstance(payload["turnover_ts"], list)
    assert len(payload["turnover_ts"]) > 0


def test_tier2_random_null_roundtrip(tmp_path: Path):
    tier1 = run_tier1(_tier1_inputs())
    save_tier1_result(tmp_path, tier1)
    statuses = run_tier2_modules(
        _tier1_inputs(),
        artifact_root=tmp_path,
        factor_name=tier1.factor_name,
        run_id=tier1.run_id,
        modules=["random_null"],
        inputs_hash=tier1.inputs_hash,
    )
    assert statuses["random_null"].status is MetricStatus.COMPUTED
    result_path = (
        tmp_path / tier1.factor_name / tier1.run_id / "tier2" / "random_null" / "result.json"
    )
    payload = json.loads(result_path.read_text())
    assert int(payload["n_trials"]) == 200
    samples = payload["null_mean_rank_ic_samples"]
    assert isinstance(samples, list)
    assert len(samples) > 0


def test_tier2_random_null_one_sided_p_value_respects_negative_tail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tier1 = run_tier1(_tier1_inputs())
    save_tier1_result(tmp_path, tier1)

    def _fake_permutation_null(*args, **kwargs):
        del args, kwargs
        return -0.20, np.asarray([-0.30, -0.20, -0.10, 0.00], dtype=float)

    monkeypatch.setattr(
        evaluation_module,
        "compute_mean_rank_ic_permutation_null",
        _fake_permutation_null,
    )
    statuses = run_tier2_modules(
        _tier1_inputs(),
        artifact_root=tmp_path,
        factor_name=tier1.factor_name,
        run_id=tier1.run_id,
        modules=["random_null"],
        inputs_hash=tier1.inputs_hash,
    )
    assert statuses["random_null"].status is MetricStatus.COMPUTED

    result_path = (
        tmp_path / tier1.factor_name / tier1.run_id / "tier2" / "random_null" / "result.json"
    )
    payload = json.loads(result_path.read_text())
    assert payload["actual_mean_rank_ic"] == pytest.approx(-0.20)
    assert payload["p_value_one_sided"] == pytest.approx(0.5)


def test_sanitize_paths_accept_unusual_factor_names(tmp_path: Path):
    result = run_tier1(_tier1_inputs(factor_name="mom/5d test"))
    paths = save_tier1_result(tmp_path, result)
    assert paths.tier1_result.exists()
    # The sanitised directory name must not contain path separators.
    assert "/" not in paths.run_dir.name


def test_prepare_inputs_accepts_parquet_factor_and_exposures(tmp_path: Path):
    spec_path = write_demo_single_factor_case(
        tmp_path,
        factor_name="bp",
        enable_neutralization=True,
    )
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    prices_csv = Path(payload["prices_path"])
    factor_csv = Path(payload["factor_path"])
    universe_csv = Path(payload["universe"]["path"])
    exposures_csv = Path(payload["neutralization"]["exposures_path"])

    prices_parquet = prices_csv.with_suffix(".parquet")
    factor_parquet = factor_csv.with_suffix(".parquet")
    universe_parquet = universe_csv.with_suffix(".parquet")
    exposures_parquet = exposures_csv.with_suffix(".parquet")

    pd.read_csv(prices_csv).to_parquet(prices_parquet, index=False)
    pd.read_csv(factor_csv).to_parquet(factor_parquet, index=False)
    pd.read_csv(universe_csv).to_parquet(universe_parquet, index=False)
    pd.read_csv(exposures_csv).to_parquet(exposures_parquet, index=False)

    payload["prices_path"] = str(prices_parquet)
    payload["factor_path"] = str(factor_parquet)
    payload["universe"]["path"] = str(universe_parquet)
    payload["neutralization"]["exposures_path"] = str(exposures_parquet)
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    spec = load_single_factor_case_spec(spec_path)
    inputs = _prepare_inputs(spec)

    assert not inputs.prices.empty
    assert not inputs.factor_df.empty
    assert {"date", "asset", "close"}.issubset(inputs.prices.columns)
    assert {"date", "asset", "factor", "value"}.issubset(inputs.factor_df.columns)
