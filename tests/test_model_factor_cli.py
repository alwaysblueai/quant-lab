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
    _load_features,
    _model_factor_price_read_columns,
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
    assert "exploratory_screening" in run_help
    assert "Level 1/2" in parser.format_help()
    assert "canonical factor" in parser.format_help()


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
    assert (case_dir / "training_log.csv").exists()
    assert (case_dir / "feature_importance.csv").exists()
    assert (case_dir / "ic_decay.csv").exists()
    assert (case_dir / "purged_kfold_summary.json").exists()
    assert (case_dir / "purged_kfold_folds.csv").exists()
    assert (case_dir / "model_selection.json").exists()

    manifest = json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["workflow"] == "real_case_model_factor"
    assert "ic_decay.csv" in manifest["required_bundle_files"]
    assert "diagnostics.json" in manifest["required_bundle_files"]
    assert "model_selection.json" in manifest["required_bundle_files"]
    assert "ic_decay" in manifest["outputs"]
    assert "diagnostics" in manifest["outputs"]
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

    backtest_payload = json.loads((case_dir / "backtest_result.json").read_text(encoding="utf-8"))
    assert backtest_payload["target_horizon"] == int(spec.target.horizon)
    assert backtest_payload["summary"]["label_horizon"] == int(spec.target.horizon)
    assert backtest_payload["summary"]["nav_rebalance_step"] == max(
        1,
        int(spec.target.horizon),
    )

    diagnostics_payload = json.loads((case_dir / "diagnostics.json").read_text(encoding="utf-8"))
    assert diagnostics_payload["artifact_type"] == "alpha_lab_model_run_diagnostics"
    assert isinstance(diagnostics_payload["stages"], list)
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
    assert "mean_mutual_information" in metrics
    assert "mutual_information_ir" in metrics
    data_load_stage = next(
        item
        for item in diagnostics_payload["stages"]
        if str(item.get("name")) == "data_load"
    )
    data_load_result = data_load_stage["result"]
    assert data_load_result["prices_requested_columns"] == ["date", "asset", "close"]
    assert "large_unused_price_payload" not in data_load_result["prices_loaded_columns"]

    model_selection_payload = json.loads(
        (case_dir / "model_selection.json").read_text(encoding="utf-8")
    )
    assert model_selection_payload["artifact_type"] == "alpha_lab_model_selection"
    assert model_selection_payload["status"] == "disabled"
    training_progress = [
        percent
        for message, percent in progress_events
        if message.startswith("训练模型生成因子：第 ")
    ]
    assert training_progress
    assert min(training_progress) >= 30
    assert max(training_progress) <= 68
    assert max(training_progress) > 30


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
        return SimpleNamespace(
            spec=SimpleNamespace(name=spec_path.stem),
            output_dir=tmp_path / spec_path.stem,
            artifact_paths={"run_manifest": tmp_path / f"{spec_path.stem}_manifest.json"},
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
            "factor": f"forward_return_{spec.target.horizon}",
            "value": 0.0,
        }
    )

    cache = _build_forward_label_cache(
        prices=prices,
        target_horizon=int(spec.target.horizon),
        target_label_df=target_labels,
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


def test_model_factor_artifacts_export_permutation_feature_importance_for_gbdt(
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
    assert (feature_importance_df["mean_abs_importance"].astype(str).str.strip() != "").all()
    assert (feature_importance_df["latest_importance"].astype(str).str.strip() != "").all()
    assert (feature_importance_df["missing_value_reason"].astype(str).str.strip() != "").all()
    assert (feature_importance_df["importance_source"] == "permutation").all()
    assert (feature_importance_df["missing_value_reason"] == "无缺失").all()
    assert pd.to_numeric(
        feature_importance_df["mean_abs_importance"],
        errors="coerce",
    ).notna().all()
    assert pd.to_numeric(
        feature_importance_df["latest_importance"],
        errors="coerce",
    ).notna().all()

    manifest = json.loads((case_dir / "run_manifest.json").read_text(encoding="utf-8"))
    notes = manifest.get("artifact_missing_value_notes")
    if isinstance(notes, dict):
        details = notes.get("details")
        if isinstance(details, list):
            assert not any(
                str(item.get("artifact")) == "feature_importance.csv"
                for item in details
            )
