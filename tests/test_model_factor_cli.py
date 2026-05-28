from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

import alpha_lab.real_cases.model_factor.cli as model_factor_cli
from alpha_lab.model_factor import FeaturePreprocessConfig
from alpha_lab.real_cases.model_factor.pipeline import (
    _build_forward_label_cache,
    _coverage_by_date,
    _load_features,
    _model_factor_price_read_columns,
    _resolve_preparation_cache_dir,
    run_model_factor_case,
)
from alpha_lab.real_cases.model_factor.spec import (
    FeatureAvailabilitySpec,
    load_model_factor_case_spec,
)
from alpha_lab.research_evaluation_config import get_research_evaluation_config
from tests.model_factor_case_helpers import write_demo_model_factor_case


def test_model_factor_cli_help_mentions_profile_and_level12_workflow() -> None:
    parser = model_factor_cli.build_parser()
    run_help = next(
        action.choices["run"].format_help()
        for action in parser._actions
        if (
            hasattr(action, "choices")
            and isinstance(action.choices, dict)
            and "run" in action.choices
        )
    )
    assert "--evaluation-profile" in run_help
    assert "--disable-preparation-cache" in run_help
    assert "--screening-retrain-every-n-dates" in run_help
    assert "exploratory_screening" in run_help
    assert "--cache-root-dir" in run_help
    benchmark_help = next(
        action.choices["benchmark"].format_help()
        for action in parser._actions
        if (
            hasattr(action, "choices")
            and isinstance(action.choices, dict)
            and "benchmark" in action.choices
        )
    )
    assert "--benchmark-output-dir" in benchmark_help
    assert "--memory-sample-interval-seconds" in benchmark_help
    assert "--memory-limit-gb" in benchmark_help
    assert "--memory-profile" in benchmark_help
    assert "--tracemalloc-profile" in benchmark_help
    assert "--cache-root-dir" in benchmark_help
    assert "Level 1/2" in parser.format_help()
    assert "canonical factor" in parser.format_help()


def test_resolve_preparation_cache_dir_default_uses_output_parent(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "case"
    cache_dir = _resolve_preparation_cache_dir(output_dir)
    assert cache_dir == (tmp_path / "outputs" / "_model_factor_cache").resolve()


def test_resolve_preparation_cache_dir_override_routes_under_cache_root(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "_web_runs" / "run-a" / "case"
    shared_root = tmp_path / "outputs" / "_model_factor_shared_cache"
    cache_dir = _resolve_preparation_cache_dir(output_dir, cache_root_dir=shared_root)
    assert cache_dir == (shared_root / "_model_factor_cache").resolve()
    other_output_dir = tmp_path / "outputs" / "_web_runs" / "run-b" / "case"
    other_cache_dir = _resolve_preparation_cache_dir(
        other_output_dir, cache_root_dir=shared_root
    )
    assert other_cache_dir == cache_dir


def test_resolve_preparation_cache_dir_rejects_web_runs_fallback(tmp_path: Path) -> None:
    """A web-launcher run must always pass cache_root_dir.

    Falling back to ``output_dir.parent`` for an output under ``_web_runs/``
    would silently duplicate ~3-4GB of feature matrices per submission (the
    leak fixed in the Phase 2 web_unified hardening). Reaching this branch
    means the launcher contract was violated, so we now refuse to proceed
    instead of warning.
    """

    output_dir = tmp_path / "outputs" / "_web_runs" / "run-a" / "case"
    with pytest.raises(ValueError, match="_web_runs"):
        _resolve_preparation_cache_dir(output_dir)


def test_preparation_cache_key_is_invariant_to_output_dir(tmp_path: Path) -> None:
    """Two web submissions of the same case must produce the same cache key.

    This is the regression guard for the original ``_web_runs/<run_id>/
    _model_factor_cache/`` leak: if the cache key were sensitive to the
    per-run output directory the dataset cache would miss every time and
    rebuild ~3-4GB per submission.
    """

    from dataclasses import asdict

    from alpha_lab.model_factor.dataset_cache import ModelFactorDatasetCache

    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)
    shared_cache_root = tmp_path / "outputs" / "_model_factor_shared_cache"

    def _key_for(run_id: str) -> str:
        output_dir = tmp_path / "outputs" / "_web_runs" / run_id / spec.name
        cache_dir = _resolve_preparation_cache_dir(
            output_dir, cache_root_dir=shared_cache_root
        )
        ds_cache = ModelFactorDatasetCache(cache_dir)
        payload = {
            "features_path": ds_cache.file_signature(Path(spec.features_path)),
            "prices_path": ds_cache.file_signature(Path(spec.prices_path)),
            "feature_columns": list(spec.feature_columns),
            "feature_availability": asdict(spec.feature_availability),
            "feature_preprocess": asdict(spec.feature_preprocess),
            "target": asdict(spec.target),
            "evaluation_profile": "default_research",
        }
        return ds_cache.build_key(payload)

    key_a = _key_for("run-a")
    key_b = _key_for("run-b")
    assert key_a == key_b, (
        "preparation cache key must not depend on the per-run output_dir; "
        "if it does, web runs will repeatedly rebuild the dataset cache"
    )


def test_model_factor_coverage_uses_eligible_denominator_and_final_scores() -> None:
    base = pd.DataFrame(
        [
            {
                "date": "2024-01-01",
                "universe_count": 4,
                "feature_row_count": 3,
                "complete_feature_count": 2,
                "feature_nan_row_count": 1,
                "label_available_count": 2,
                "eligible_count": 2,
                "missing_feature_count": 1,
                "missing_label_count": 1,
                "filtered_count": 2,
            },
            {
                "date": "2024-01-02",
                "universe_count": 4,
                "feature_row_count": 3,
                "complete_feature_count": 3,
                "feature_nan_row_count": 0,
                "label_available_count": 2,
                "eligible_count": 2,
                "missing_feature_count": 1,
                "missing_label_count": 1,
                "filtered_count": 2,
            },
        ]
    )
    factor = pd.DataFrame(
        [
            {"date": "2024-01-01", "asset": "A", "factor": "m", "value": 0.1},
            {"date": "2024-01-01", "asset": "B", "factor": "m", "value": float("nan")},
            {"date": "2024-01-02", "asset": "A", "factor": "m", "value": 0.2},
        ]
    )
    labels = pd.DataFrame(
        [
            {"date": "2024-01-01", "asset": "A", "factor": "forward_return_5", "value": 0.01},
            {"date": "2024-01-01", "asset": "B", "factor": "forward_return_5", "value": 0.02},
            {"date": "2024-01-02", "asset": "A", "factor": "forward_return_5", "value": pd.NA},
            {"date": "2024-01-02", "asset": "B", "factor": "forward_return_5", "value": 0.03},
        ]
    )

    coverage = _coverage_by_date(factor, coverage_base_df=base, target_label_df=labels)

    first = coverage.loc[pd.to_datetime(coverage["date"]) == pd.Timestamp("2024-01-01")].iloc[0]
    assert int(first["universe_count"]) == 4
    assert int(first["eligible_count"]) == 2
    assert int(first["scored_count"]) == 1
    assert int(first["scored_evaluable_count"]) == 1
    assert int(first["missing_score_count"]) == 1
    assert first["coverage"] == pytest.approx(0.5)
    assert first["universe_coverage"] == pytest.approx(0.25)

    second = coverage.loc[pd.to_datetime(coverage["date"]) == pd.Timestamp("2024-01-02")].iloc[
        0
    ]
    assert int(second["scored_count"]) == 1
    assert int(second["scored_evaluable_count"]) == 0
    assert int(second["missing_score_count"]) == 2
    assert second["coverage"] == pytest.approx(0.0)


def test_run_model_factor_case_writes_bundle(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)

    progress_events: list[tuple[str, int]] = []

    result = run_model_factor_case(
        spec_path,
        progress_callback=lambda message, percent: progress_events.append((message, percent)),
    )
    case_dir = Path(spec.output.root_dir) / spec.name

    assert result.output_dir == case_dir.resolve()
    assert (case_dir / "run_manifest.json").exists()
    assert (case_dir / "metrics.json").exists()
    assert (case_dir / "model_definition.json").exists()
    assert (case_dir / "feature_manifest.json").exists()
    assert (case_dir / "diagnostics.json").exists()
    assert (case_dir / "research_tearsheet.json").exists()
    assert (case_dir / "research_tearsheet.pdf").exists()
    assert (case_dir / "training_log.csv").exists()
    assert (case_dir / "training_metrics.csv").exists()
    assert (case_dir / "feature_importance.csv").exists()
    assert (case_dir / "feature_oos_ic.csv").exists()
    assert (case_dir / "ic_decay.csv").exists()
    assert (case_dir / "purged_kfold_summary.json").exists()
    assert (case_dir / "purged_kfold_folds.csv").exists()
    assert (case_dir / "purged_kfold_fold_daily.csv").exists()
    assert (case_dir / "model_selection.json").exists()
    coverage_df = pd.read_csv(case_dir / "coverage.csv")
    assert {
        "universe_count",
        "eligible_count",
        "scored_count",
        "scored_evaluable_count",
        "missing_score_count",
        "filtered_count",
        "score_coverage",
        "universe_coverage",
    }.issubset(set(coverage_df.columns))
    assert (coverage_df["eligible_count"] >= coverage_df["scored_evaluable_count"]).all()

    manifest = json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["workflow"] == "real_case_model_factor"
    assert "ic_decay.csv" in manifest["required_bundle_files"]
    assert "diagnostics.json" in manifest["required_bundle_files"]
    assert "research_tearsheet.json" in manifest["required_bundle_files"]
    assert "research_tearsheet.pdf" in manifest["required_bundle_files"]
    assert "model_selection.json" in manifest["required_bundle_files"]
    assert "training_metrics.csv" in manifest["required_bundle_files"]
    assert "feature_oos_ic.csv" in manifest["required_bundle_files"]
    assert "purged_kfold_fold_daily.csv" in manifest["required_bundle_files"]
    assert "ic_decay" in manifest["outputs"]
    assert "diagnostics" in manifest["outputs"]
    assert "training_metrics" in manifest["outputs"]
    assert "feature_oos_ic" in manifest["outputs"]
    assert "purged_kfold_fold_daily" in manifest["outputs"]
    manifest_split = manifest["split_contract"]
    assert manifest_split["source"] == "model_factor_pipeline"
    assert manifest_split["n_oos_dates"] >= manifest_split["min_oos_dates"]
    assert manifest["research_tearsheet"]["status"] == "emitted"
    assert (
        manifest["research_tearsheet"]["split_contract"]["oos_start"]
        == manifest_split["oos_start"]
    )
    feature_preprocess_inputs = manifest["inputs"]["feature_preprocess"]
    assert feature_preprocess_inputs["cross_sectional_transform"] == "winsorize_zscore"
    assert feature_preprocess_inputs["cross_sectional_transform_default_applied"] is True

    model_definition_payload = json.loads(
        (case_dir / "model_definition.json").read_text(encoding="utf-8")
    )
    feature_preprocess_definition = model_definition_payload["feature_preprocess"]
    assert feature_preprocess_definition["cross_sectional_transform"] == "winsorize_zscore"
    assert feature_preprocess_definition["cross_sectional_transform_default_applied"] is True
    resolved_model_params = model_definition_payload["resolved_model_params"]
    assert resolved_model_params["configured_model"]["family"] == "ridge"
    assert isinstance(resolved_model_params["configured_model"]["params"], dict)
    label_temporal_contract = model_definition_payload["label_temporal_contract"]
    assert label_temporal_contract["target_horizon"] == int(spec.target.horizon)
    assert label_temporal_contract["purged_train_gap_dates"] == max(
        int(spec.target.horizon) - 1,
        0,
    )
    model_selection_outcome = model_definition_payload["model_selection_outcome"]
    assert model_selection_outcome["enabled"] is False
    assert isinstance(model_selection_outcome["n_fit_events"], int)
    assert isinstance(model_selection_outcome["n_reuse_events"], int)
    assert "training_metrics_path" in model_definition_payload["source_artifacts"]
    assert "feature_oos_ic_path" in model_definition_payload["source_artifacts"]

    backtest_payload = json.loads((case_dir / "backtest_result.json").read_text(encoding="utf-8"))
    assert backtest_payload["target_horizon"] == int(spec.target.horizon)
    assert backtest_payload["split_contract"]["oos_start"] == manifest_split["oos_start"]
    assert backtest_payload["oos_start"] == manifest_split["oos_start"]
    assert backtest_payload["summary"]["label_horizon"] == int(spec.target.horizon)
    assert backtest_payload["summary"]["nav_rebalance_step"] == max(
        1,
        int(spec.target.horizon),
    )

    diagnostics_payload = json.loads((case_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics_payload["artifact_type"] == "alpha_lab_model_run_diagnostics"
    assert isinstance(diagnostics_payload["stages"], list)
    assert isinstance(diagnostics_payload["stage_timings"], dict)
    assert isinstance(diagnostics_payload["events"], list)
    assert isinstance(diagnostics_payload["warnings"], list)
    assert isinstance(diagnostics_payload["data_health"], dict)
    artifact_export_stage = next(
        (
            item
            for item in diagnostics_payload["stages"]
            if str(item.get("name")) == "artifact_export"
        ),
        None,
    )
    assert artifact_export_stage is not None
    assert str(artifact_export_stage.get("status")) == "success"
    summary_md = (case_dir / "summary.md").read_text(encoding="utf-8")
    assert "default_applied=true" in summary_md

    metrics_payload = json.loads((case_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics = metrics_payload["metrics"]
    assert metrics["model_family"] == "ridge"
    assert metrics["feature_count"] == 3
    assert metrics["split_contract"]["oos_start"] == manifest_split["oos_start"]
    assert metrics["oos_start"] == manifest_split["oos_start"]
    assert metrics["metric_scope"] == "oos"
    assert metrics["report_metric_scope"] == "full_sample_with_oos_parentheses"
    assert metrics["report_timeseries_scope"] == "full_path_split_by_phase"
    assert "mean_rank_ic_full" in metrics
    assert "mean_rank_ic_is" in metrics
    assert "mean_rank_ic_oos" in metrics
    assert "mean_rank_ic_oos_decay_ratio" in metrics
    assert metrics["mean_rank_ic_oos"] == pytest.approx(metrics["mean_rank_ic"])
    assert "max_drawdown_full" in metrics
    assert "max_drawdown_is" in metrics
    assert "max_drawdown_oos" in metrics
    assert metrics["split_semantics"] == "model_training_prediction_holdout"
    assert "Model-Lab" in metrics["split_semantics_label"]
    assert "mean_mutual_information" in metrics
    assert "mutual_information_ir" in metrics
    training_metrics = pd.read_csv(case_dir / "training_metrics.csv")
    assert {
        "model_version",
        "train_rank_ic",
        "oos_rank_ic",
        "train_loss",
        "oos_loss",
    }.issubset(set(training_metrics.columns))
    assert not training_metrics.empty
    feature_oos_ic = pd.read_csv(case_dir / "feature_oos_ic.csv")
    assert {"feature", "window_start", "window_end", "rank_ic", "n_obs"}.issubset(
        set(feature_oos_ic.columns)
    )
    assert set(spec.feature_columns).issubset(set(feature_oos_ic["feature"]))
    purged_fold_daily = pd.read_csv(case_dir / "purged_kfold_fold_daily.csv")
    assert {"fold", "date", "ic", "rank_ic"}.issubset(set(purged_fold_daily.columns))
    assert not purged_fold_daily.empty
    ic_timeseries = pd.read_csv(case_dir / "ic_timeseries.csv")
    assert {"IS", "OOS"}.issubset(set(ic_timeseries["split_phase"]))
    assert pd.to_datetime(ic_timeseries["date"]).min() < pd.Timestamp(
        manifest_split["oos_start"]
    )
    group_returns = pd.read_csv(case_dir / "group_returns.csv")
    assert {"IS", "OOS"}.issubset(set(group_returns["split_phase"]))
    data_load_stage = next(
        item
        for item in diagnostics_payload["stages"]
        if str(item.get("name")) == "data_load"
    )
    data_load_result = data_load_stage["result"]
    assert data_load_result["prices_requested_columns"] == ["date", "asset", "close", "open"]
    assert data_load_result["preparation_cache_enabled"] is True
    assert data_load_result["preparation_cache_hit"] is False
    assert data_load_result["split_contract"]["oos_start"] == manifest_split["oos_start"]
    assert "large_unused_price_payload" not in data_load_result["prices_loaded_columns"]
    assert result.stage_timings["model_fit_count"] > 0
    assert "evaluate_total" in result.evaluation_result.stage_timings
    assert "core_backtest.ic" in result.evaluation_result.stage_timings
    assert "core_backtest.quantile" in result.evaluation_result.stage_timings

    model_selection_payload = json.loads(
        (case_dir / "model_selection.json").read_text(encoding="utf-8")
    )
    assert model_selection_payload["artifact_type"] == "alpha_lab_model_selection"
    assert model_selection_payload["status"] == "disabled"
    tearsheet_payload = json.loads(
        (case_dir / "research_tearsheet.json").read_text(encoding="utf-8")
    )
    assert tearsheet_payload["meta"]["split_contract"]["oos_start"] == manifest_split["oos_start"]
    assert tearsheet_payload["meta"]["split_semantics"] == "model_training_prediction_holdout"
    assert (case_dir / "research_tearsheet.pdf").stat().st_size > 0
    training_progress = [
        percent
        for message, percent in progress_events
        if message.startswith("训练模型生成因子：第 ")
    ]
    assert training_progress
    assert min(training_progress) >= 30
    assert max(training_progress) <= 68
    assert max(training_progress) > 30


def test_run_model_factor_case_reuses_preparation_cache_on_second_run(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)
    output_root = tmp_path / "cached_runs"

    first = run_model_factor_case(spec_path, output_root_dir=output_root)
    second = run_model_factor_case(spec_path, output_root_dir=output_root)

    assert len(first.factor_df) == len(second.factor_df)
    case_dir = output_root / spec.name
    diagnostics_payload = json.loads((case_dir / "diagnostics.json").read_text(encoding="utf-8"))
    data_load_stage = next(
        item
        for item in diagnostics_payload["stages"]
        if str(item.get("name")) == "data_load"
    )
    feature_stage = next(
        item
        for item in diagnostics_payload["stages"]
        if str(item.get("name")) == "feature_validate"
    )
    target_stage = next(
        item
        for item in diagnostics_payload["stages"]
        if str(item.get("name")) == "target_build"
    )
    assert data_load_stage["result"]["preparation_cache_hit"] is True
    assert feature_stage["result"]["cache_hit"] is True
    assert target_stage["result"]["cache_hit"] is True


def test_run_model_factor_case_skips_feature_reload_for_screening_prepared_cache(
    tmp_path: Path,
) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    output_root = tmp_path / "screening_cached_runs"

    first = run_model_factor_case(
        spec_path,
        output_root_dir=output_root,
        evaluation_profile="exploratory_screening",
    )
    first_diagnostics = json.loads(
        (first.output_dir / "diagnostics.json").read_text(encoding="utf-8")
    )
    first_data_load_stage = next(
        item
        for item in first_diagnostics["stages"]
        if str(item.get("name")) == "data_load"
    )
    assert first_data_load_stage["result"]["preparation_cache_hit"] is False
    assert first_data_load_stage["result"]["features_loaded_for_data_load"] is True
    assert first_data_load_stage["result"]["features_skipped_due_to_prepared_cache"] is False
    assert not pd.read_csv(first.output_dir / "training_metrics.csv").empty
    assert pd.read_csv(first.output_dir / "feature_oos_ic.csv").empty

    second = run_model_factor_case(
        spec_path,
        output_root_dir=output_root,
        evaluation_profile="exploratory_screening",
    )

    assert len(first.factor_df) == len(second.factor_df)
    diagnostics_payload = json.loads(
        (second.output_dir / "diagnostics.json").read_text(encoding="utf-8")
    )
    data_load_stage = next(
        item
        for item in diagnostics_payload["stages"]
        if str(item.get("name")) == "data_load"
    )
    data_load_result = data_load_stage["result"]
    assert data_load_result["preparation_cache_hit"] is True
    assert data_load_result["prepared_inputs_cache_hit"] is True
    assert data_load_result["features_loaded_for_data_load"] is False
    assert data_load_result["features_skipped_due_to_prepared_cache"] is True

    feature_manifest = json.loads(
        (second.output_dir / "feature_manifest.json").read_text(encoding="utf-8")
    )
    assert feature_manifest["manifest_source"] == "cache_metadata"
    assert feature_manifest["features"][0]["non_null_ratio"] is not None


def test_run_model_factor_case_reuses_preparation_cache_across_models(
    tmp_path: Path,
) -> None:
    ridge_spec_path = write_demo_model_factor_case(tmp_path, factor_name="ridge_score")
    lasso_spec_path = tmp_path / "lasso_model_factor_case.yaml"
    payload = yaml.safe_load(ridge_spec_path.read_text(encoding="utf-8"))
    payload["name"] = "lasso_model_factor_case"
    payload["factor_name"] = "lasso_score"
    payload["model"] = {"family": "lasso", "params": {"alpha": 0.001, "max_iter": 5000}}
    lasso_spec_path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    output_root = tmp_path / "cross_model_cached_runs"

    ridge = run_model_factor_case(ridge_spec_path, output_root_dir=output_root)
    lasso = run_model_factor_case(lasso_spec_path, output_root_dir=output_root)

    assert not ridge.factor_df.empty
    assert not lasso.factor_df.empty
    ridge_diag = json.loads(
        (ridge.output_dir / "diagnostics.json").read_text(encoding="utf-8")
    )
    lasso_diag = json.loads(
        (lasso.output_dir / "diagnostics.json").read_text(encoding="utf-8")
    )
    ridge_data_load = next(
        item for item in ridge_diag["stages"] if str(item.get("name")) == "data_load"
    )
    lasso_data_load = next(
        item for item in lasso_diag["stages"] if str(item.get("name")) == "data_load"
    )
    lasso_feature_stage = next(
        item for item in lasso_diag["stages"] if str(item.get("name")) == "feature_validate"
    )
    lasso_target_stage = next(
        item for item in lasso_diag["stages"] if str(item.get("name")) == "target_build"
    )
    assert ridge_data_load["result"]["preparation_cache_hit"] is False
    assert lasso_data_load["result"]["preparation_cache_hit"] is True
    assert (
        ridge_data_load["result"]["preparation_cache_key"]
        == lasso_data_load["result"]["preparation_cache_key"]
    )
    assert lasso_feature_stage["result"]["cache_hit"] is True
    assert lasso_target_stage["result"]["cache_hit"] is True


def test_run_model_factor_case_applies_screening_retrain_override(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    output_root = tmp_path / "screening_override"

    result = run_model_factor_case(
        spec_path,
        output_root_dir=output_root,
        evaluation_profile="exploratory_screening",
        screening_retrain_every_n_dates=60,
    )

    assert result.spec.training.retrain_every_n_dates == 60
    diagnostics_payload = json.loads(
        (result.output_dir / "diagnostics.json").read_text(encoding="utf-8")
    )
    spec_load_stage = next(
        item
        for item in diagnostics_payload["stages"]
        if str(item.get("name")) == "spec_load"
    )
    assert spec_load_stage["result"]["evaluation_profile"] == "exploratory_screening"
    assert spec_load_stage["result"]["screening_retrain_every_n_dates"] == 60
    assert spec_load_stage["result"]["training_retrain_every_n_dates_effective"] == 60
    metrics_payload = json.loads((result.output_dir / "metrics.json").read_text(encoding="utf-8"))
    metrics = metrics_payload["metrics"]
    assert metrics["retrain_density_warning"] is True
    assert metrics["training_retrain_every_n_dates_effective"] == 60


def test_run_model_factor_case_ignores_screening_retrain_override_outside_screening(
    tmp_path: Path,
) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)

    result = run_model_factor_case(
        spec_path,
        output_root_dir=tmp_path / "default_research_override_ignored",
        evaluation_profile="default_research",
        screening_retrain_every_n_dates=60,
    )

    assert result.spec.training.retrain_every_n_dates == spec.training.retrain_every_n_dates
    diagnostics_payload = json.loads(
        (result.output_dir / "diagnostics.json").read_text(encoding="utf-8")
    )
    spec_load_stage = next(
        item
        for item in diagnostics_payload["stages"]
        if str(item.get("name")) == "spec_load"
    )
    assert (
        spec_load_stage["result"]["training_retrain_every_n_dates_effective"]
        == spec.training.retrain_every_n_dates
    )
    warnings = diagnostics_payload.get("warnings")
    assert isinstance(warnings, list)
    assert any("筛选重训间隔覆盖未生效" in str(item.get("title")) for item in warnings)


def test_model_factor_cli_run_executes_and_writes_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)

    out_root = tmp_path / "cli_out"
    rc = model_factor_cli.main(["run", str(spec_path), "--output-root-dir", str(out_root)])
    assert rc == 0

    captured = capsys.readouterr()
    assert "real-case-model-factor" in captured.out
    assert "Status   : success" in captured.out
    assert "Evaluation Profile" in captured.out
    assert "Level 2 Promotion" in captured.out

    case_dir = out_root / spec.name
    assert (case_dir / "run_manifest.json").exists()
    assert (case_dir / "metrics.json").exists()
    assert (case_dir / "model_definition.json").exists()
    assert not (case_dir / "case_report.md").exists()

    manifest = json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["render_status"] == "skipped"
    assert manifest["rendered_report"] is False
    assert manifest["rendered_report_path"] is None


def test_model_factor_cli_benchmark_writes_recorder_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    out_root = tmp_path / "benchmark_case_out"
    benchmark_dir = tmp_path / "benchmark_records"

    rc = model_factor_cli.main(
        [
            "benchmark",
            str(spec_path),
            "--output-root-dir",
            str(out_root),
            "--benchmark-output-dir",
            str(benchmark_dir),
            "--run-id",
            "demo_benchmark",
            "--evaluation-profile",
            "exploratory_screening",
            "--screening-retrain-every-n-dates",
            "60",
            "--memory-sample-interval-seconds",
            "0.01",
            "--memory-profile",
        ]
    )

    assert rc == 0
    captured = capsys.readouterr()
    assert "real-case-model-factor-benchmark" in captured.out
    record_path = benchmark_dir / "demo_benchmark.json"
    assert record_path.exists()
    assert (benchmark_dir / "latest.json").exists()
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["artifact_type"] == "alpha_lab_model_factor_benchmark_record"
    assert record["status"] == "success"
    assert record["run_id"] == "demo_benchmark"
    assert isinstance(record["config_hash"], str)
    assert record["config"]["runtime"]["evaluation_profile"] == "exploratory_screening"
    assert record["config"]["runtime"]["memory_limit_gb"] == 24.0
    assert record["config"]["runtime"]["memory_profile"] is True
    assert record["memory"]["resource_limits"]["address_space_limit_enabled"] is True
    assert Path(record["memory"]["profile_path"]).exists()
    assert record["training"]["fit_count"] >= 1
    assert record["memory"]["peak_rss_kb"] > 0
    assert "stage_timings" in record["timings"]
    assert "evaluation_stage_timings" in record["timings"]
    assert "preparation_cache_hit" in record["cache_lineage"]
    assert Path(record["artifacts"]["diagnostics"]).exists()


def test_model_factor_cli_run_batch_expands_patterns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    spec_a = tmp_path / "a.yaml"
    spec_b = tmp_path / "b.yaml"
    spec_a.write_text("name: a\n", encoding="utf-8")
    spec_b.write_text("name: b\n", encoding="utf-8")
    calls: list[Path] = []

    def _fake_run_model_factor_case(spec_path: Path, **_kwargs: object) -> SimpleNamespace:
        calls.append(spec_path)
        stem = spec_path.stem
        return SimpleNamespace(
            spec=SimpleNamespace(name=stem),
            output_dir=tmp_path / stem,
            artifact_paths={
                "run_manifest": tmp_path / f"{stem}_manifest.json",
                "experiment_card": tmp_path / f"{stem}_experiment_card.md",
                "summary": tmp_path / f"{stem}_summary.md",
            },
        )

    monkeypatch.setattr(model_factor_cli, "run_model_factor_case", _fake_run_model_factor_case)

    rc = model_factor_cli.main(
        ["run-batch", str(tmp_path / "*.yaml"), "--vault-export-mode", "skip"]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert calls == [spec_a.resolve(), spec_b.resolve()]
    assert "real-case-model-factor-batch" in captured.out
    assert "Cases    : 2" in captured.out


def test_model_factor_cli_render_report_writes_case_report(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)

    out_root = tmp_path / "cli_out"
    rc = model_factor_cli.main(
        [
            "run",
            str(spec_path),
            "--output-root-dir",
            str(out_root),
            "--render-report",
        ]
    )
    assert rc == 0

    case_dir = out_root / spec.name
    report_path = case_dir / "case_report.md"
    assert report_path.exists()

    manifest = json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["render_status"] == "success"
    assert manifest["rendered_report"] is True
    assert manifest["rendered_report_path"] == str(report_path.resolve())


def test_run_model_factor_case_accepts_parquet_features(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec_payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    features_csv = Path(spec_payload["features_path"])
    features_parquet = features_csv.with_suffix(".parquet")
    spec_payload["features_path"] = str(features_parquet)

    pd.read_csv(features_csv).to_parquet(features_parquet, index=False)

    parquet_spec_path = tmp_path / "ml_score_model_factor_case_parquet.yaml"
    parquet_spec_path.write_text(
        yaml.safe_dump(spec_payload, sort_keys=False),
        encoding="utf-8",
    )

    result = run_model_factor_case(parquet_spec_path)
    assert not result.factor_df.empty
    assert result.model_factor_result.training_log_df["status"].isin(
        ["fit_scored", "reused_scored", "skipped"]
    ).all()


def test_model_factor_feature_loader_reads_only_required_parquet_columns(tmp_path: Path) -> None:
    features_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "date": ["2024-01-02"],
            "asset": ["000001.SZ"],
            "known_at": ["2024-01-02"],
            "feature_momentum": [1.0],
            "feature_quality": [2.0],
            "large_unused_payload": ["not_loaded"],
        }
    ).to_parquet(features_path, index=False)

    features = _load_features(
        str(features_path),
        feature_columns=("feature_momentum", "feature_quality"),
        feature_availability=FeatureAvailabilitySpec(mode="required_timestamp"),
        feature_preprocess=FeaturePreprocessConfig(),
    )

    assert features.columns.tolist() == [
        "date",
        "asset",
        "feature_momentum",
        "feature_quality",
        "known_at",
    ]
    assert "large_unused_payload" not in features.columns


def test_model_factor_price_read_columns_include_profile_driven_optional_columns() -> None:
    evaluation_config = get_research_evaluation_config("default_research")

    required, optional = _model_factor_price_read_columns(evaluation_config)

    assert required == ("date", "asset", "close")
    assert "volume" in optional
    assert {"open", "high", "low"}.issubset(set(optional))
    assert {"amount", "market_cap", "circ_mv", "total_mv"}.issubset(set(optional))
    assert {"ret_5d", "ret_20d"}.issubset(set(optional))


def test_model_factor_price_read_columns_include_target_price_column() -> None:
    evaluation_config = get_research_evaluation_config("default_research")

    required, _ = _model_factor_price_read_columns(
        evaluation_config,
        target_price_column="close_qfq",
    )

    assert required == ("date", "asset", "close", "close_qfq")


def test_model_factor_price_read_columns_require_open_for_next_open_target() -> None:
    evaluation_config = get_research_evaluation_config("default_research")

    required, optional = _model_factor_price_read_columns(
        evaluation_config,
        target_price_column="close_qfq",
        execution_price_mode="next_open",
    )

    assert required == ("date", "asset", "close", "close_qfq", "open")
    assert "open" not in optional


def test_model_factor_forward_label_cache_precomputes_decay_horizons(
    tmp_path: Path,
) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec = load_model_factor_case_spec(spec_path)
    prices = pd.read_csv(spec.prices_path)
    target_labels = pd.DataFrame(
        {
            "date": pd.to_datetime(prices["date"]),
            "asset": prices["asset"],
            "factor": f"forward_return_{spec.target.horizon}_next_open",
            "value": 0.0,
        }
    )

    cache = _build_forward_label_cache(
        prices=prices,
        target_horizon=int(spec.target.horizon),
        target_label_df=target_labels,
        target_price_column=spec.target.price_column,
        execution_price_mode=spec.target.execution_price_mode,
        max_abs_forward_return=spec.target.max_abs_forward_return,
        evaluation_config=get_research_evaluation_config("default_research"),
    )

    assert set(cache) == {1, 2, 3, 5, 10, 20}
    assert cache[int(spec.target.horizon)]["value"].eq(0.0).all()


def test_run_model_factor_case_materializes_parquet_features_from_csv(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec_payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    features_csv = Path(spec_payload["features_path"])
    features_parquet = features_csv.with_suffix(".parquet")

    assert features_csv.exists()
    assert not features_parquet.exists()

    result = run_model_factor_case(spec_path)

    assert features_parquet.exists()
    diagnostics_payload = json.loads(
        (result.output_dir / "diagnostics.json").read_text(encoding="utf-8")
    )
    data_load_stage = next(
        item
        for item in diagnostics_payload["stages"]
        if str(item.get("name")) == "data_load"
    )
    stage_result = data_load_stage["result"]
    assert stage_result["features_requested_path"] == str(features_csv)
    assert stage_result["features_storage_path"] == str(features_parquet)
    assert stage_result["features_storage_format"] == "parquet"
    assert stage_result["features_parquet_materialized"] is True


def test_run_model_factor_case_failure_still_flushes_diagnostics(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec_payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec_payload["feature_columns"] = ["missing_feature_column"]
    bad_spec_path = tmp_path / "ml_score_model_factor_case_bad.yaml"
    bad_spec_path.write_text(
        yaml.safe_dump(spec_payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(Exception):  # noqa: B017
        run_model_factor_case(bad_spec_path)

    case_dir = Path(spec_payload["output"]["root_dir"]) / spec_payload["name"]
    diagnostics_path = case_dir / "diagnostics.json"
    assert diagnostics_path.exists()
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert payload["run_meta"]["status"] == "failed"
    assert any(str(item.get("status")) == "failed" for item in payload["stages"])


def test_run_model_factor_case_requires_feature_timestamp_by_default(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(
        tmp_path,
        factor_name="ml_score",
        include_known_at=False,
    )
    with pytest.raises(ValueError, match="feature_availability.mode='required_timestamp'"):
        run_model_factor_case(spec_path)


def test_run_model_factor_case_accepts_safety_lag_when_timestamp_missing(tmp_path: Path) -> None:
    spec_path = write_demo_model_factor_case(
        tmp_path,
        factor_name="ml_score",
        include_known_at=False,
    )
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    payload["feature_availability"] = {"mode": "safety_lag", "safety_lag_days": 1}
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = run_model_factor_case(spec_path)
    feature_manifest = json.loads(
        (result.output_dir / "feature_manifest.json").read_text(encoding="utf-8")
    )
    assert feature_manifest["feature_availability"]["mode"] == "safety_lag"
    assert feature_manifest["feature_availability"]["safety_lag_days"] == 1


def test_run_model_factor_case_warns_on_fundamental_features_without_safety_lag(
    tmp_path: Path,
) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    features_path = Path(payload["features_path"])
    features = pd.read_csv(features_path)
    features = features.rename(columns={"feature_quality": "roe_ttm"})
    features.to_csv(features_path, index=False)
    payload["feature_columns"] = ["feature_momentum", "roe_ttm", "feature_noise"]
    payload["feature_availability"] = {"mode": "required_timestamp", "column": "known_at"}
    spec_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    result = run_model_factor_case(spec_path)
    diagnostics_payload = json.loads(
        (result.output_dir / "diagnostics.json").read_text(encoding="utf-8")
    )
    warnings = diagnostics_payload.get("warnings")
    assert isinstance(warnings, list)
    assert any("基本面特征可用性风险" in str(item.get("title")) for item in warnings)


def test_model_factor_artifacts_export_cheap_feature_importance_ledger_for_gbdt(
    tmp_path: Path,
) -> None:
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    spec_payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    spec_payload["model"] = {
        "family": "gbdt",
        "params": {"max_iter": 60, "max_leaf_nodes": 15},
    }
    spec_path.write_text(
        yaml.safe_dump(spec_payload, sort_keys=False),
        encoding="utf-8",
    )

    result = run_model_factor_case(spec_path)
    case_dir = result.output_dir

    feature_importance_df = pd.read_csv(
        case_dir / "feature_importance.csv",
        keep_default_na=False,
    )
    assert "missing_value_reason" in feature_importance_df.columns
    assert "importance_source" in feature_importance_df.columns
    assert "latest_abs_importance" in feature_importance_df.columns
    assert "sign_stability" in feature_importance_df.columns
    assert (feature_importance_df["mean_abs_importance"].astype(str).str.strip() != "").all()
    assert (feature_importance_df["latest_importance"].astype(str).str.strip() != "").all()
    assert (feature_importance_df["missing_value_reason"].astype(str).str.strip() != "").all()
    assert (feature_importance_df["importance_source"] == "built_in_unavailable").all()
    assert (
        feature_importance_df["missing_value_reason"]
        .astype(str)
        .str.contains("permutation fallback", regex=False)
        .all()
    )
    ledger_df = pd.read_csv(case_dir / "feature_importance_ledger.csv", keep_default_na=False)
    assert not ledger_df.empty
    assert {
        "run_id",
        "case",
        "factor",
        "model_family",
        "model_version",
        "fit_date",
        "feature",
        "signed_importance",
        "abs_importance",
        "normalized_share",
        "rank",
        "importance_source",
    }.issubset(set(ledger_df.columns))
    assert (ledger_df["importance_source"] == "built_in_unavailable").all()

    manifest = json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert "feature_importance_ledger.csv" in manifest["required_bundle_files"]
    assert "feature_importance_ledger" in manifest["outputs"]
    notes = manifest.get("artifact_missing_value_notes")
    if isinstance(notes, dict):
        details = notes.get("details")
        if isinstance(details, list):
            assert any(
                str(item.get("artifact")) == "feature_importance.csv"
                for item in details
            )


def test_run_model_factor_case_writes_resource_usage(tmp_path: Path) -> None:
    """P0: the model-factor pipeline emits resource_usage.json on a successful run,
    mirroring the single-factor memory contract (telemetry only)."""
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="ml_score")
    output_root = tmp_path / "out"

    result = run_model_factor_case(spec_path, output_root_dir=output_root)

    resource_path = result.output_dir / "resource_usage.json"
    assert resource_path.exists()
    snapshot = json.loads(resource_path.read_text(encoding="utf-8"))
    assert snapshot["artifact_type"] == "alpha_lab_resource_usage"
    for field in ("peak_rss_mb", "stage_rss_mb", "max_rss_mb_budget"):
        assert field in snapshot
    stage_rss = snapshot["stage_rss_mb"]
    assert isinstance(stage_rss, dict)
    # Stage keys only populate when RSS sampling is available (psutil present).
    if snapshot["monitor_available"]:
        assert "run_start" in stage_rss
        assert "artifacts_exported" in stage_rss
