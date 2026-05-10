from __future__ import annotations

import os
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pandas as pd
import pytest
import yaml

import alpha_lab.decay as decay_module
import alpha_lab.experiment as experiment
import alpha_lab.real_cases.single_factor.evaluate as sf_evaluate_module
import alpha_lab.real_cases.single_factor.pipeline as sf_pipeline
from alpha_lab.exceptions import AlphaLabConfigError
from alpha_lab.factors.reversal import reversal
from alpha_lab.real_cases.single_factor.pipeline import (
    SingleFactorArtifactPaths,
    SingleFactorBatchParallelConfig,
    SingleFactorCaseRunResult,
    SingleFactorEvaluationResult,
    load_standard_inputs,
    run_single_factor_batch,
    run_single_factor_case,
    run_single_factor_cases,
)
from alpha_lab.real_cases.single_factor.spec import load_single_factor_case_spec
from alpha_lab.research_integrity.contracts import IntegrityReport
from tests.single_factor_case_helpers import write_demo_single_factor_case


def _core_metrics(run) -> dict[str, float]:
    metrics = run.evaluation_result.metrics
    return {
        "mean_ic": float(metrics["mean_ic"]),
        "mean_rank_ic": float(metrics["mean_rank_ic"]),
        "long_short_ir": float(metrics["long_short_ir"]),
        "mean_long_short_return": float(metrics["mean_long_short_return"]),
        "eval_coverage_ratio_mean": float(metrics["eval_coverage_ratio_mean"]),
    }


def _build_reversal_spec_from_base(base_spec_path: Path, *, out_dir: Path) -> Path:
    payload = yaml.safe_load(base_spec_path.read_text(encoding="utf-8"))
    prices = pd.read_csv(payload["prices_path"])
    rev = reversal(prices[["date", "asset", "close"]], window=5).copy()
    rev["factor"] = "negative_past_5d_return"

    factor_path = out_dir / "inputs" / "negative_past_5d_return.csv"
    factor_path.parent.mkdir(parents=True, exist_ok=True)
    rev.to_csv(factor_path, index=False)

    payload["name"] = "demo_negative_past_5d_return_single_factor"
    payload["factor_name"] = "negative_past_5d_return"
    payload["factor_path"] = str(factor_path)
    payload["output"]["root_dir"] = str(out_dir / "outputs_reversal")

    spec_path = out_dir / "negative_past_5d_return_single_factor_case.yaml"
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return spec_path


def test_single_factor_input_bundle_keeps_results_unchanged(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(
        tmp_path,
        factor_name="bp",
        enable_neutralization=True,
    )

    baseline = run_single_factor_case(
        spec_path,
        evaluation_profile="exploratory_screening",
    )
    bundle = load_standard_inputs(spec_path)
    with_bundle = run_single_factor_case(
        spec_path,
        evaluation_profile="exploratory_screening",
        input_bundle=bundle,
    )

    assert _core_metrics(with_bundle) == pytest.approx(_core_metrics(baseline), rel=0.0, abs=1e-12)
    assert bundle.prices_panel is not None
    assert bundle.prices_all is not None
    assert {"ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d"}.issubset(
        set(bundle.prices_panel.columns)
    )
    assert {5, 10, 20}.issubset(set(bundle.base_feature_cache.forward_labels_by_horizon))


def test_load_standard_inputs_prefers_parquet_slice_and_bundle_reuses_parquet_spec(
    tmp_path: Path,
) -> None:
    spec_path = write_demo_single_factor_case(
        tmp_path,
        factor_name="bp",
        enable_neutralization=False,
    )
    spec_csv = load_single_factor_case_spec(spec_path)
    assert spec_csv.universe.path is not None

    prices_csv = Path(spec_csv.prices_path)
    universe_csv = Path(spec_csv.universe.path)
    prices_parquet = prices_csv.with_suffix(".parquet")
    universe_parquet = universe_csv.with_suffix(".parquet")
    pd.read_csv(prices_csv).to_parquet(prices_parquet, index=False)
    pd.read_csv(universe_csv).to_parquet(universe_parquet, index=False)

    now = time.time()
    os.utime(prices_csv, (now, now))
    os.utime(universe_csv, (now, now))
    os.utime(prices_parquet, (now + 5.0, now + 5.0))
    os.utime(universe_parquet, (now + 5.0, now + 5.0))

    bundle = load_standard_inputs(spec_csv)
    assert bundle.prices_path == str(prices_parquet)
    assert bundle.universe_path == str(universe_parquet)

    spec_parquet = replace(
        spec_csv,
        prices_path=str(prices_parquet),
        universe=replace(spec_csv.universe, path=str(universe_parquet)),
    )
    run = run_single_factor_case(
        spec_parquet,
        evaluation_profile="exploratory_screening",
        input_bundle=bundle,
    )
    assert not run.factor_df.empty


def test_single_factor_case_passes_precomputed_labels_to_experiment_runner(
    tmp_path: Path,
) -> None:
    spec_path = write_demo_single_factor_case(
        tmp_path,
        factor_name="bp",
        enable_neutralization=False,
    )
    captured_precomputed: list[dict[int, pd.DataFrame] | None] = []
    original_runner = sf_evaluate_module.core.run_factor_experiment

    def _wrapped_runner(*args, **kwargs):
        captured_precomputed.append(kwargs.get("precomputed_forward_labels"))
        return original_runner(*args, **kwargs)

    with patch.object(sf_evaluate_module.core, "run_factor_experiment", _wrapped_runner):
        run_single_factor_case(
            spec_path,
            evaluation_profile="exploratory_screening",
        )

    assert captured_precomputed
    cache = captured_precomputed[0]
    assert cache is not None
    assert 1 in cache
    assert 5 in cache


def test_load_standard_inputs_uses_dividend_adjusted_close_for_cached_labels(
    tmp_path: Path,
) -> None:
    spec_path = write_demo_single_factor_case(
        tmp_path,
        factor_name="bp",
        enable_neutralization=False,
    )
    spec = load_single_factor_case_spec(spec_path)
    prices_path = Path(spec.prices_path)

    prices = pd.read_csv(prices_path)
    prices["dividend_per_share"] = 0.0
    first_asset = str(prices.loc[0, "asset"])
    first_asset_panel = (
        prices[prices["asset"] == first_asset]
        .copy()
        .assign(date=lambda x: pd.to_datetime(x["date"], errors="coerce"))
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )
    assert len(first_asset_panel) >= 4

    ex_date = pd.Timestamp(first_asset_panel.loc[2, "date"])
    div_value = float(first_asset_panel.loc[1, "close"]) * 0.1
    ex_mask = (prices["asset"] == first_asset) & (
        pd.to_datetime(prices["date"], errors="coerce") == ex_date
    )
    prices.loc[ex_mask, "dividend_per_share"] = div_value
    prices.to_csv(prices_path, index=False)

    spec_no_universe = replace(spec, universe=replace(spec.universe, path=None))
    bundle = load_standard_inputs(spec_no_universe)
    adjusted = (
        bundle.prices_panel[bundle.prices_panel["asset"] == first_asset]
        .sort_values("date", kind="mergesort")
        .reset_index(drop=True)
    )
    raw = first_asset_panel

    ratio = 1.0 - div_value / float(raw.loc[1, "close"])
    assert adjusted.loc[0, "close"] == pytest.approx(float(raw.loc[0, "close"]) * ratio)
    assert adjusted.loc[1, "close"] == pytest.approx(float(raw.loc[1, "close"]) * ratio)
    assert adjusted.loc[2, "close"] == pytest.approx(float(raw.loc[2, "close"]))

    labels_1d = bundle.base_feature_cache.forward_labels_by_horizon[1]
    first_date = pd.Timestamp(raw.loc[0, "date"])
    first_label = labels_1d[
        (labels_1d["asset"] == first_asset) & (labels_1d["date"] == first_date)
    ]
    assert not first_label.empty
    expected_ret = float(adjusted.loc[1, "close"] / adjusted.loc[0, "close"] - 1.0)
    assert float(first_label.iloc[0]["value"]) == pytest.approx(expected_ret)


@pytest.mark.parametrize(
    "batch_parallel_config",
    [
        None,
        SingleFactorBatchParallelConfig(mode="thread", max_workers=2, factors_per_worker=1),
        SingleFactorBatchParallelConfig(mode="thread", max_workers=2, factors_per_worker=2),
    ],
)
def test_run_single_factor_cases_reuses_prices_and_universe_loads(
    tmp_path: Path,
    batch_parallel_config: SingleFactorBatchParallelConfig | None,
) -> None:
    spec_a = write_demo_single_factor_case(
        tmp_path / "case_a",
        factor_name="past_60d_return_skip_5d",
        enable_neutralization=False,
    )
    spec_b = _build_reversal_spec_from_base(spec_a, out_dir=tmp_path / "case_b")

    counters = {"prices": 0, "universe": 0}
    orig_load_prices = sf_pipeline.load_prices
    orig_load_universe = sf_pipeline.load_universe_mask

    def wrap_load_prices(*args, **kwargs):
        counters["prices"] += 1
        return orig_load_prices(*args, **kwargs)

    def wrap_load_universe(*args, **kwargs):
        counters["universe"] += 1
        return orig_load_universe(*args, **kwargs)

    with (
        patch.object(sf_pipeline, "load_prices", wrap_load_prices),
        patch.object(sf_pipeline, "load_universe_mask", wrap_load_universe),
    ):
        runs = run_single_factor_cases(
            [spec_a, spec_b],
            evaluation_profile="exploratory_screening",
            batch_parallel_config=batch_parallel_config,
        )

    assert len(runs) == 2
    assert counters["prices"] == 1
    assert counters["universe"] == 1


def test_run_single_factor_batch_keeps_metrics_unchanged_under_parallel_modes(
    tmp_path: Path,
) -> None:
    spec_a = write_demo_single_factor_case(
        tmp_path / "case_a",
        factor_name="past_60d_return_skip_5d",
        enable_neutralization=False,
    )
    spec_b = _build_reversal_spec_from_base(spec_a, out_dir=tmp_path / "case_b")
    base_spec = load_single_factor_case_spec(spec_a)
    reversal_spec = load_single_factor_case_spec(spec_b)

    factor_defs = [
        {
            "factor_name": base_spec.factor_name,
            "case_name": "batch_momentum_case",
        },
        {
            "factor_name": reversal_spec.factor_name,
            "factor_path": reversal_spec.factor_path,
            "case_name": "batch_reversal_case",
        },
    ]

    serial_runs = run_single_factor_batch(
        base_spec,
        factor_defs,
        output_root_dir=tmp_path / "batch_serial_outputs",
        evaluation_profile="exploratory_screening",
        batch_parallel_config=SingleFactorBatchParallelConfig(mode="serial"),
    )
    thread_runs = run_single_factor_batch(
        base_spec,
        factor_defs,
        output_root_dir=tmp_path / "batch_thread_outputs",
        evaluation_profile="exploratory_screening",
        batch_parallel_config=SingleFactorBatchParallelConfig(
            mode="thread",
            max_workers=2,
            factors_per_worker=1,
        ),
    )
    process_runs = run_single_factor_batch(
        base_spec,
        factor_defs,
        output_root_dir=tmp_path / "batch_process_outputs",
        evaluation_profile="exploratory_screening",
        vault_root="",
        vault_export_mode="skip",
        batch_parallel_config=SingleFactorBatchParallelConfig(
            mode="process",
            max_workers=2,
            factors_per_worker=1,
        ),
    )

    serial_by_name = {run.spec.name: _core_metrics(run) for run in serial_runs}
    thread_by_name = {run.spec.name: _core_metrics(run) for run in thread_runs}
    process_by_name = {run.spec.name: _core_metrics(run) for run in process_runs}
    assert set(serial_by_name) == set(thread_by_name)
    assert set(serial_by_name) == set(process_by_name)
    for case_name, metrics in serial_by_name.items():
        assert thread_by_name[case_name] == pytest.approx(metrics, rel=0.0, abs=1e-12)
        assert process_by_name[case_name] == pytest.approx(metrics, rel=0.0, abs=1e-12)


def test_run_single_factor_batch_reuse_toggle_keeps_metrics_unchanged(
    tmp_path: Path,
) -> None:
    spec_a = write_demo_single_factor_case(
        tmp_path / "case_a",
        factor_name="past_60d_return_skip_5d",
        enable_neutralization=False,
    )
    spec_b = _build_reversal_spec_from_base(spec_a, out_dir=tmp_path / "case_b")
    base_spec = load_single_factor_case_spec(spec_a)
    reversal_spec = load_single_factor_case_spec(spec_b)

    factor_defs = [
        {"factor_name": base_spec.factor_name, "case_name": "reuse_on_momentum"},
        {
            "factor_name": reversal_spec.factor_name,
            "factor_path": reversal_spec.factor_path,
            "case_name": "reuse_on_reversal",
        },
    ]

    runs_reuse = run_single_factor_batch(
        base_spec,
        factor_defs,
        output_root_dir=tmp_path / "reuse_on_outputs",
        evaluation_profile="exploratory_screening",
        reuse_input_bundle=True,
        batch_parallel_config=SingleFactorBatchParallelConfig(mode="serial"),
    )
    runs_no_reuse = run_single_factor_batch(
        base_spec,
        factor_defs,
        output_root_dir=tmp_path / "reuse_off_outputs",
        evaluation_profile="exploratory_screening",
        reuse_input_bundle=False,
        batch_parallel_config=SingleFactorBatchParallelConfig(mode="serial"),
    )

    by_name_reuse = {run.spec.name: _core_metrics(run) for run in runs_reuse}
    by_name_no_reuse = {run.spec.name: _core_metrics(run) for run in runs_no_reuse}
    assert set(by_name_reuse) == set(by_name_no_reuse)
    for case_name, metrics in by_name_reuse.items():
        assert by_name_no_reuse[case_name] == pytest.approx(metrics, rel=0.0, abs=1e-12)


def test_precomputed_label_cache_reduces_forward_return_rebuilds(tmp_path: Path) -> None:
    spec_a = write_demo_single_factor_case(
        tmp_path / "case_a",
        factor_name="past_60d_return_skip_5d",
        enable_neutralization=False,
    )
    spec_b = _build_reversal_spec_from_base(spec_a, out_dir=tmp_path / "case_b")

    full_bundle = load_standard_inputs(spec_a)
    base = full_bundle.base_feature_cache
    plain_prices = base.prices_enriched.drop(
        columns=list(base.trailing_return_columns),
        errors="ignore",
    ).copy()
    plain_bundle = replace(
        full_bundle,
        prices_panel=plain_prices,
        base_feature_cache=replace(
            base,
            prices_enriched=plain_prices,
            trailing_return_columns=(),
            forward_labels_by_horizon={},
        ),
    )

    original_forward_return = experiment.forward_return
    original_decay_forward_return = decay_module.forward_return
    counters = {
        "before_experiment": 0,
        "before_decay": 0,
        "after_experiment": 0,
        "after_decay": 0,
    }

    def wrap_before_experiment(*args, **kwargs):
        counters["before_experiment"] += 1
        return original_forward_return(*args, **kwargs)

    def wrap_before_decay(*args, **kwargs):
        counters["before_decay"] += 1
        return original_decay_forward_return(*args, **kwargs)

    with (
        patch.object(experiment, "forward_return", wrap_before_experiment),
        patch.object(decay_module, "forward_return", wrap_before_decay),
    ):
        run_single_factor_case(
            spec_a,
            evaluation_profile="exploratory_screening",
            input_bundle=plain_bundle,
        )
        run_single_factor_case(
            spec_b,
            evaluation_profile="exploratory_screening",
            input_bundle=plain_bundle,
        )

    def wrap_after_experiment(*args, **kwargs):
        counters["after_experiment"] += 1
        return original_forward_return(*args, **kwargs)

    def wrap_after_decay(*args, **kwargs):
        counters["after_decay"] += 1
        return original_decay_forward_return(*args, **kwargs)

    with (
        patch.object(experiment, "forward_return", wrap_after_experiment),
        patch.object(decay_module, "forward_return", wrap_after_decay),
    ):
        run_single_factor_case(
            spec_a,
            evaluation_profile="exploratory_screening",
            input_bundle=full_bundle,
        )
        run_single_factor_case(
            spec_b,
            evaluation_profile="exploratory_screening",
            input_bundle=full_bundle,
        )

    assert counters["before_experiment"] >= 2
    assert counters["before_decay"] >= 2
    assert counters["after_experiment"] == 0
    assert counters["after_decay"] == 0


def test_input_bundle_rejects_incompatible_spec(tmp_path: Path) -> None:
    spec_a = write_demo_single_factor_case(
        tmp_path / "case_a",
        factor_name="bp",
        enable_neutralization=False,
    )
    spec_b = write_demo_single_factor_case(
        tmp_path / "case_b",
        factor_name="bp",
        enable_neutralization=False,
    )

    bundle = load_standard_inputs(spec_a)
    with pytest.raises(AlphaLabConfigError):
        run_single_factor_case(
            spec_b,
            evaluation_profile="exploratory_screening",
            input_bundle=bundle,
        )


def test_process_batch_mode_rejects_custom_factor_loader(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(
        tmp_path,
        factor_name="bp",
        enable_neutralization=False,
    )
    with pytest.raises(AlphaLabConfigError):
        run_single_factor_cases(
            [spec_path],
            evaluation_profile="exploratory_screening",
            batch_parallel_config=SingleFactorBatchParallelConfig(
                mode="process",
                max_workers=2,
            ),
            factor_loader=lambda spec: pd.read_csv(spec.factor_path),
        )


def test_run_single_factor_cases_without_bundle_reuse_falls_back_to_per_case_loads(
    tmp_path: Path,
) -> None:
    spec_a = write_demo_single_factor_case(
        tmp_path / "case_a",
        factor_name="past_60d_return_skip_5d",
        enable_neutralization=False,
    )
    spec_b = _build_reversal_spec_from_base(spec_a, out_dir=tmp_path / "case_b")

    counters = {"prices": 0, "universe": 0}
    orig_load_prices = sf_pipeline.load_prices
    orig_load_universe = sf_pipeline.load_universe_mask

    def wrap_load_prices(*args, **kwargs):
        counters["prices"] += 1
        return orig_load_prices(*args, **kwargs)

    def wrap_load_universe(*args, **kwargs):
        counters["universe"] += 1
        return orig_load_universe(*args, **kwargs)

    with (
        patch.object(sf_pipeline, "load_prices", wrap_load_prices),
        patch.object(sf_pipeline, "load_universe_mask", wrap_load_universe),
    ):
        runs = run_single_factor_cases(
            [spec_a, spec_b],
            evaluation_profile="exploratory_screening",
            batch_parallel_config=SingleFactorBatchParallelConfig(mode="serial"),
            reuse_input_bundle=False,
        )

    assert len(runs) == 2
    assert counters["prices"] == 2
    assert counters["universe"] == 2


def test_batch_parallel_performance_guard_thread_mode(tmp_path: Path) -> None:
    """A lightweight wall-time guard that catches obvious thread scheduling regressions."""
    base_spec_path = write_demo_single_factor_case(
        tmp_path,
        factor_name="bp",
        enable_neutralization=False,
    )
    base_spec = load_single_factor_case_spec(base_spec_path)
    specs = [replace(base_spec, name=f"{base_spec.name}_guard_{i}") for i in range(6)]

    def fake_run_single_factor_case(
        spec_or_path: object,
        *,
        output_root_dir: object = None,
        factor_loader: object = None,
        evaluation_profile: str = "default_research",
        vault_root: object = None,
        vault_export_mode: str = "versioned",
        progress_callback: object = None,
        fast_screen_artifact_root: object = None,
        fast_screen_run_id: object = None,
        input_bundle: object = None,
    ) -> SingleFactorCaseRunResult:
        del (
            output_root_dir,
            factor_loader,
            evaluation_profile,
            vault_root,
            vault_export_mode,
            progress_callback,
            fast_screen_artifact_root,
            fast_screen_run_id,
            input_bundle,
        )
        time.sleep(0.08)
        spec = cast(sf_pipeline.SingleFactorCaseSpec, spec_or_path)
        return SingleFactorCaseRunResult(
            spec=spec,
            output_dir=Path("/tmp"),
            factor_df=pd.DataFrame(columns=["date", "asset", "factor", "value"]),
            evaluation_result=cast(
                SingleFactorEvaluationResult,
                SimpleNamespace(metrics={}),
            ),
            artifact_paths=cast(SingleFactorArtifactPaths, {}),
            integrity_report=cast(IntegrityReport, SimpleNamespace()),
        )

    with patch.object(sf_pipeline, "run_single_factor_case", fake_run_single_factor_case):
        t0 = time.perf_counter()
        run_single_factor_cases(
            specs,
            batch_parallel_config=SingleFactorBatchParallelConfig(mode="serial"),
            reuse_input_bundle=False,
        )
        serial_seconds = time.perf_counter() - t0

        t1 = time.perf_counter()
        run_single_factor_cases(
            specs,
            batch_parallel_config=SingleFactorBatchParallelConfig(
                mode="thread",
                max_workers=3,
                factors_per_worker=1,
            ),
            reuse_input_bundle=False,
        )
        thread_seconds = time.perf_counter() - t1

    assert thread_seconds < serial_seconds * 0.70
