from __future__ import annotations

import difflib
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
import pytest
import yaml

from alpha_lab.artifact_contracts import validate_level12_artifact_payload

pytestmark = [pytest.mark.slow, pytest.mark.golden]
from alpha_lab.examples.profile_aware_campaign_level12 import (
    run_profile_aware_campaign_level12_example,
)
from alpha_lab.real_cases.model_factor.pipeline import run_model_factor_case
from alpha_lab.real_cases.single_factor.pipeline import run_single_factor_case
from alpha_lab.reporting.research_validation_package import (
    build_research_validation_package,
    export_research_validation_package,
)
from tests.model_factor_case_helpers import write_demo_model_factor_case
from tests.single_factor_case_helpers import write_demo_single_factor_case

GOLDEN_ROOT = Path(__file__).resolve().parent / "goldens" / "artifact_regression"
UPDATE_GOLDENS_ENV = "ALPHA_LAB_UPDATE_GOLDENS"
SCRUBBED_TIMESTAMP_TOKEN = "<SCRUBBED_TIMESTAMP>"
VOLATILE_JSON_KEYS = frozenset(
    {
        "created_at_utc",
        "generated_at_utc",
        "run_timestamp_utc",
    }
)
SINGLE_FACTOR_CSV_SNAPSHOT_KEYS = (
    "ic_timeseries",
    "ic_decay",
    "factor_autocorrelation",
    "capacity_estimation",
    "conditional_ic_by_magnitude",
    "conditional_ic_by_cross_section_size",
    "rolling_stability",
    "group_returns",
    "group_nav",
    "turnover",
    "coverage",
    "lag_sensitivity",
    "random_baseline_null",
    "daily_pnl_attribution",
    "quantile_membership",
    "quantile_equal_weights",
    "portfolio_weights",
)
SINGLE_FACTOR_JSON_GOLDEN_KEYS = (
    ("run_manifest", Path("single_factor/run_manifest.json")),
    ("factor_definition_json", Path("single_factor/factor_definition.json")),
    ("signal_validation_json", Path("single_factor/signal_validation.json")),
    ("portfolio_recipe_json", Path("single_factor/portfolio_recipe.json")),
    ("backtest_result_json", Path("single_factor/backtest_result.json")),
    ("research_tearsheet", Path("single_factor/research_tearsheet.json")),
)
MODEL_FACTOR_CSV_SNAPSHOT_KEYS = (
    "training_log",
    "training_metrics",
    "feature_importance",
    "feature_importance_ledger",
    "feature_oos_ic",
    "purged_kfold_folds",
    "purged_kfold_fold_daily",
    "ic_timeseries",
    "ic_decay",
    "rolling_stability",
    "group_returns",
    "group_nav",
    "turnover",
    "coverage",
)
MODEL_FACTOR_JSON_GOLDEN_KEYS = (
    ("run_manifest", Path("model_factor/run_manifest.json")),
    ("metrics", Path("model_factor/metrics.json")),
    ("factor_definition_json", Path("model_factor/factor_definition.json")),
    ("signal_validation_json", Path("model_factor/signal_validation.json")),
    ("portfolio_recipe_json", Path("model_factor/portfolio_recipe.json")),
    ("backtest_result_json", Path("model_factor/backtest_result.json")),
    ("model_selection_json", Path("model_factor/model_selection.json")),
    ("model_definition_json", Path("model_factor/model_definition.json")),
    ("feature_manifest_json", Path("model_factor/feature_manifest.json")),
    ("research_tearsheet", Path("model_factor/research_tearsheet.json")),
    (
        "portfolio_validation_summary",
        Path("model_factor/level2_portfolio_validation/portfolio_validation_summary.json"),
    ),
    (
        "portfolio_validation_package",
        Path("model_factor/level2_portfolio_validation/portfolio_validation_package.json"),
    ),
)


def test_single_factor_level12_core_artifacts_match_golden(tmp_path: Path) -> None:
    """Golden coverage for one deterministic Level 1/2 single-factor run."""
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")
    output_root = tmp_path / "single_factor_level12"
    result = run_single_factor_case(
        spec_path,
        output_root_dir=output_root,
        vault_export_mode="skip",
    )

    replacements = {
        str(tmp_path.resolve()): "<TMP_ROOT>",
        str(output_root.resolve()): "<OUTPUT_ROOT>",
    }

    for artifact_key, golden_relative_path in SINGLE_FACTOR_JSON_GOLDEN_KEYS:
        _assert_json_matches_golden(
            artifact_path=result.artifact_paths[artifact_key],
            golden_relative_path=golden_relative_path,
            replacements=replacements,
        )
    _assert_json_matches_golden(
        artifact_path=result.artifact_paths["metrics"],
        golden_relative_path=Path("single_factor/metrics.json"),
        replacements=replacements,
    )
    _assert_text_matches_golden(
        golden_relative_path=Path("single_factor/summary.md"),
        actual_text=(result.artifact_paths["summary"]).read_text(encoding="utf-8"),
        replacements=replacements,
    )
    _assert_json_matches_golden(
        artifact_path=result.artifact_paths["portfolio_validation_summary"],
        golden_relative_path=Path(
            "single_factor/level2_portfolio_validation/portfolio_validation_summary.json"
        ),
        replacements=replacements,
    )
    _assert_json_matches_golden(
        artifact_path=result.artifact_paths["portfolio_validation_package"],
        golden_relative_path=Path(
            "single_factor/level2_portfolio_validation/portfolio_validation_package.json"
        ),
        replacements=replacements,
    )
    _assert_csv_snapshot_matches_golden(
        artifact_paths=result.artifact_paths,
        artifact_keys=SINGLE_FACTOR_CSV_SNAPSHOT_KEYS,
        golden_relative_path=Path("single_factor/csv_snapshot.json"),
    )
    _write_research_validation_workflow_summary(
        output_dir=result.output_dir,
        case_name=result.spec.name,
        spec_path=spec_path,
        metrics_path=result.artifact_paths["metrics"],
        portfolio_validation_summary_path=result.artifact_paths["portfolio_validation_summary"],
        portfolio_validation_metrics_path=result.artifact_paths["portfolio_validation_metrics"],
        portfolio_validation_package_path=result.artifact_paths["portfolio_validation_package"],
    )
    package = build_research_validation_package(
        result.output_dir,
        case_id=result.spec.name,
        case_name=result.spec.name,
    )
    exported_package = export_research_validation_package(
        package,
        output_root / "research_validation_package",
    )
    package_payload = json.loads(exported_package["json"].read_text(encoding="utf-8"))
    if not isinstance(package_payload, dict):
        raise AssertionError(f"{exported_package['json']} must contain a JSON object")
    validate_level12_artifact_payload(
        package_payload,
        artifact_name=exported_package["json"].name,
        source=exported_package["json"],
    )
    _assert_text_matches_golden(
        golden_relative_path=Path("single_factor/research_validation_package.md"),
        actual_text=exported_package["markdown"].read_text(encoding="utf-8"),
        replacements=replacements,
    )


def test_single_factor_constant_small_cross_section_artifacts_match_golden(
    tmp_path: Path,
) -> None:
    """Golden coverage for the small-cross-section/constant-factor anomaly case."""
    spec_path = _write_constant_small_cross_section_case(tmp_path)
    output_root = tmp_path / "single_factor_constant_small_cross_section"
    result = run_single_factor_case(
        spec_path,
        output_root_dir=output_root,
        vault_export_mode="skip",
    )

    replacements = {
        str(tmp_path.resolve()): "<TMP_ROOT>",
        str(output_root.resolve()): "<OUTPUT_ROOT>",
    }

    for artifact_key, golden_relative_path in (
        ("metrics", Path("single_factor_constant_small_cross_section/metrics.json")),
        (
            "backtest_result_json",
            Path("single_factor_constant_small_cross_section/backtest_result.json"),
        ),
        (
            "research_tearsheet",
            Path("single_factor_constant_small_cross_section/research_tearsheet.json"),
        ),
    ):
        _assert_json_matches_golden(
            artifact_path=result.artifact_paths[artifact_key],
            golden_relative_path=golden_relative_path,
            replacements=replacements,
        )
    _assert_csv_snapshot_matches_golden(
        artifact_paths=result.artifact_paths,
        artifact_keys=SINGLE_FACTOR_CSV_SNAPSHOT_KEYS,
        golden_relative_path=Path("single_factor_constant_small_cross_section/csv_snapshot.json"),
    )


def test_single_factor_high_turnover_fee_artifacts_match_golden(tmp_path: Path) -> None:
    """Golden coverage for a deterministic high-turnover cost attribution case."""
    spec_path = _write_rank_flip_high_turnover_case(tmp_path)
    output_root = tmp_path / "single_factor_high_turnover_fee"
    result = run_single_factor_case(
        spec_path,
        output_root_dir=output_root,
        vault_export_mode="skip",
    )

    turnover = pd.read_csv(result.artifact_paths["turnover"])
    daily_pnl = pd.read_csv(result.artifact_paths["daily_pnl_attribution"])
    turnover_values = pd.to_numeric(turnover["turnover"], errors="coerce").dropna()
    cost_drag_values = pd.to_numeric(daily_pnl["cost_drag"], errors="coerce").dropna()
    net_values = pd.to_numeric(daily_pnl["net"], errors="coerce").dropna()

    assert not turnover_values.empty
    assert float(turnover_values.mean()) > 0.50
    assert not cost_drag_values.empty
    assert float(cost_drag_values.mean()) > 0.0
    assert len(net_values) == len(cost_drag_values)

    replacements = {
        str(tmp_path.resolve()): "<TMP_ROOT>",
        str(output_root.resolve()): "<OUTPUT_ROOT>",
    }

    for artifact_key, golden_relative_path in (
        ("metrics", Path("single_factor_high_turnover_fee/metrics.json")),
        (
            "backtest_result_json",
            Path("single_factor_high_turnover_fee/backtest_result.json"),
        ),
        (
            "research_tearsheet",
            Path("single_factor_high_turnover_fee/research_tearsheet.json"),
        ),
    ):
        _assert_json_matches_golden(
            artifact_path=result.artifact_paths[artifact_key],
            golden_relative_path=golden_relative_path,
            replacements=replacements,
        )
    _assert_csv_snapshot_matches_golden(
        artifact_paths=result.artifact_paths,
        artifact_keys=SINGLE_FACTOR_CSV_SNAPSHOT_KEYS,
        golden_relative_path=Path("single_factor_high_turnover_fee/csv_snapshot.json"),
    )


def test_model_factor_level12_core_artifacts_match_golden(tmp_path: Path) -> None:
    """Golden coverage for one deterministic Level 1/2 model-factor run."""
    spec_path = write_demo_model_factor_case(tmp_path, factor_name="model_alpha")
    output_root = tmp_path / "model_factor_level12"
    result = run_model_factor_case(
        spec_path,
        output_root_dir=output_root,
        vault_export_mode="skip",
    )

    replacements = {
        str(tmp_path.resolve()): "<TMP_ROOT>",
        str(output_root.resolve()): "<OUTPUT_ROOT>",
    }

    metrics_payload = json.loads(result.artifact_paths["metrics"].read_text(encoding="utf-8"))
    metrics = metrics_payload.get("metrics")
    if not isinstance(metrics, dict):
        raise AssertionError("model-factor metrics.json must contain a metrics object")
    assert metrics["split_semantics"] == "model_training_prediction_holdout"
    assert metrics["metric_scope"] == "oos"

    ic_timeseries = pd.read_csv(result.artifact_paths["ic_timeseries"])
    group_returns = pd.read_csv(result.artifact_paths["group_returns"])
    assert {"IS", "OOS"}.issubset(set(ic_timeseries["split_phase"]))
    assert {"IS", "OOS"}.issubset(set(group_returns["split_phase"]))

    # Purged k-fold diagnostics must use the configured label horizon as the
    # purge gap. Defaulting label_horizon to 1 silently allows train rows
    # whose forward-return label overlaps the test fold to leak in.
    target_horizon = int(metrics["target_horizon"])
    purged_summary = json.loads(
        result.artifact_paths["purged_kfold_summary"].read_text(encoding="utf-8")
    )
    assert int(purged_summary["label_horizon"]) == target_horizon
    assert int(purged_summary["purge_days"]) == target_horizon

    # Model selection ↔ metrics consistency: when selection is disabled, the
    # configured family is what gets trained; when enabled, the latest selected
    # candidate's family must match. Either way, metrics.json model_family
    # must agree with model_selection.json.
    selection_payload = json.loads(
        result.artifact_paths["model_selection_json"].read_text(encoding="utf-8")
    )
    metrics_family = str(metrics["model_family"])
    if str(selection_payload.get("status")) in {"disabled", "no_candidates"}:
        configured = selection_payload.get("configured_model") or {}
        assert str(configured.get("family")) == metrics_family
    else:
        latest_id = (selection_payload.get("summary") or {}).get(
            "latest_selected_candidate_id"
        )
        rows = [
            row
            for row in selection_payload.get("selection_rows") or []
            if row.get("selected_candidate_id") == latest_id
        ]
        assert rows, (
            "model_selection.json claims selection ran but no row matches "
            f"latest_selected_candidate_id={latest_id!r}"
        )
        winner_family = rows[-1].get("selected_candidate_family") or rows[-1].get("family")
        assert str(winner_family) == metrics_family

    for artifact_key, golden_relative_path in MODEL_FACTOR_JSON_GOLDEN_KEYS:
        _assert_json_matches_golden(
            artifact_path=result.artifact_paths[artifact_key],
            golden_relative_path=golden_relative_path,
            replacements=replacements,
        )
    _assert_csv_snapshot_matches_golden(
        artifact_paths=result.artifact_paths,
        artifact_keys=MODEL_FACTOR_CSV_SNAPSHOT_KEYS,
        golden_relative_path=Path("model_factor/csv_snapshot.json"),
    )


def test_campaign_profile_comparison_artifacts_match_golden(tmp_path: Path) -> None:
    """Golden coverage for deterministic Level 1/2 campaign profile comparison outputs."""
    output_root = tmp_path / "campaign_profile_comparison"
    result = run_profile_aware_campaign_level12_example(
        output_root_dir=output_root,
        profiles=("exploratory_screening", "default_research"),
        render_report=False,
        clean_output=True,
    )

    replacements = {
        str(output_root.resolve()): "<OUTPUT_ROOT>",
    }

    _assert_json_matches_golden(
        artifact_path=result.comparison_json_path,
        golden_relative_path=Path("campaign_profile_comparison/campaign_profile_comparison.json"),
        replacements=replacements,
    )
    _assert_text_matches_golden(
        golden_relative_path=Path("campaign_profile_comparison/campaign_profile_comparison.md"),
        actual_text=result.comparison_markdown_path.read_text(encoding="utf-8"),
        replacements=replacements,
    )
    _assert_text_matches_golden(
        golden_relative_path=Path("campaign_profile_comparison/campaign_profile_case_matrix.csv"),
        actual_text=result.comparison_csv_path.read_text(encoding="utf-8"),
        replacements=replacements,
    )


def _write_constant_small_cross_section_case(tmp_path: Path) -> Path:
    factor_name = "constant_small_cross_section"
    spec_path = write_demo_single_factor_case(tmp_path, factor_name=factor_name, n_days=160)
    spec_payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec_payload, dict):
        raise AssertionError(f"{spec_path} must contain a YAML object")
    spec_payload["preprocess"] = {
        "winsorize": False,
        "winsorize_lower": 0.01,
        "winsorize_upper": 0.99,
        "standardization": "none",
        "min_group_size": 1,
        "min_coverage": None,
    }
    spec_path.write_text(yaml.safe_dump(spec_payload, sort_keys=False), encoding="utf-8")

    factor_path = Path(str(spec_payload["factor_path"]))
    factors = pd.read_csv(factor_path, parse_dates=["date"])
    dates = pd.Index(pd.to_datetime(factors["date"]).drop_duplicates()).sort_values()
    asset_order = {asset: idx for idx, asset in enumerate(sorted(factors["asset"].unique()))}
    date_order = {date: idx for idx, date in enumerate(dates)}
    keep_counts = [1, 2, 4]
    factors["_date_idx"] = pd.to_datetime(factors["date"]).map(date_order).astype(int)
    factors["_asset_idx"] = factors["asset"].map(asset_order).astype(int)
    factors["value"] = 1.0
    keep_mask = factors["_asset_idx"] < factors["_date_idx"].map(
        lambda idx: keep_counts[int(idx) % len(keep_counts)]
    )
    factors.loc[~keep_mask, "value"] = float("nan")
    factors[["date", "asset", "factor", "value"]].to_csv(factor_path, index=False)
    return spec_path


def _write_rank_flip_high_turnover_case(tmp_path: Path) -> Path:
    factor_name = "rank_flip_high_turnover"
    spec_path = write_demo_single_factor_case(tmp_path, factor_name=factor_name, n_days=160)
    spec_payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(spec_payload, dict):
        raise AssertionError(f"{spec_path} must contain a YAML object")

    factor_path = Path(str(spec_payload["factor_path"]))
    factors = pd.read_csv(factor_path, parse_dates=["date"])
    dates = pd.Index(pd.to_datetime(factors["date"]).drop_duplicates()).sort_values()
    asset_order = {asset: idx for idx, asset in enumerate(sorted(factors["asset"].unique()))}
    date_order = {date: idx for idx, date in enumerate(dates)}
    factors["_date_idx"] = pd.to_datetime(factors["date"]).map(date_order).astype(int)
    factors["_asset_idx"] = factors["asset"].map(asset_order).astype(int)
    direction = factors["_date_idx"].map(lambda idx: 1.0 if int(idx) % 2 == 0 else -1.0)
    factors["value"] = direction * (factors["_asset_idx"].astype(float) + 1.0)
    factors[["date", "asset", "factor", "value"]].to_csv(factor_path, index=False)
    return spec_path


def _assert_csv_snapshot_matches_golden(
    *,
    artifact_paths: Mapping[str, Path],
    artifact_keys: tuple[str, ...],
    golden_relative_path: Path,
) -> None:
    snapshot: dict[str, object] = {}
    for key in artifact_keys:
        path = artifact_paths[key]
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
        header = text.splitlines()[0] if text.splitlines() else ""
        row_count = max(len(text.splitlines()) - 1, 0)
        snapshot[key] = {
            "file": path.name,
            "columns": header.split(",") if header else [],
            "rows": row_count,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        }
    snapshot_text = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    _assert_text_matches_golden(
        golden_relative_path=golden_relative_path,
        actual_text=snapshot_text,
        replacements={},
    )


def _assert_json_matches_golden(
    *,
    artifact_path: Path,
    golden_relative_path: Path,
    replacements: dict[str, str],
) -> None:
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{artifact_path} must contain a JSON object")
    validate_level12_artifact_payload(
        payload,
        artifact_name=artifact_path.name,
        source=artifact_path,
    )
    normalized_payload = _normalize_json(payload, replacements=replacements)
    normalized_text = (
        json.dumps(
            normalized_payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    _assert_text_matches_golden(
        golden_relative_path=golden_relative_path,
        actual_text=normalized_text,
        replacements={},
    )


def _normalize_json(value: object, *, replacements: dict[str, str]) -> object:
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, raw in value.items():
            key_text = str(key)
            if key_text in VOLATILE_JSON_KEYS:
                normalized[key_text] = SCRUBBED_TIMESTAMP_TOKEN
                continue
            normalized[key_text] = _normalize_json(raw, replacements=replacements)
        return normalized
    if isinstance(value, list):
        return [_normalize_json(item, replacements=replacements) for item in value]
    if isinstance(value, str):
        return _normalize_text(value, replacements=replacements)
    return value


def _assert_text_matches_golden(
    *,
    golden_relative_path: Path,
    actual_text: str,
    replacements: dict[str, str],
) -> None:
    golden_path = GOLDEN_ROOT / golden_relative_path
    normalized_actual = _normalize_text(actual_text, replacements=replacements)
    if not normalized_actual.endswith("\n"):
        normalized_actual = normalized_actual + "\n"

    if _update_goldens_enabled():
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(normalized_actual, encoding="utf-8")
        return

    if not golden_path.exists():
        raise AssertionError(
            f"missing golden file: {golden_path}. "
            f"Set {UPDATE_GOLDENS_ENV}=1 to create/update baselines intentionally."
        )

    expected = golden_path.read_text(encoding="utf-8")
    if normalized_actual == expected:
        return

    diff = "".join(
        difflib.unified_diff(
            expected.splitlines(keepends=True),
            normalized_actual.splitlines(keepends=True),
            fromfile=f"expected/{golden_relative_path}",
            tofile=f"actual/{golden_relative_path}",
        )
    )
    raise AssertionError(
        "artifact drift detected for "
        f"{golden_relative_path}. "
        f"If intentional, re-run with {UPDATE_GOLDENS_ENV}=1.\n{diff}"
    )


def _normalize_text(text: str, *, replacements: dict[str, str]) -> str:
    normalized = text.replace("\r\n", "\n")
    for old, new in sorted(replacements.items(), key=lambda row: len(row[0]), reverse=True):
        normalized = normalized.replace(old, new)
    return normalized


def _write_research_validation_workflow_summary(
    *,
    output_dir: Path,
    case_name: str,
    spec_path: Path,
    metrics_path: Path,
    portfolio_validation_summary_path: Path,
    portfolio_validation_metrics_path: Path,
    portfolio_validation_package_path: Path,
) -> Path:
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise AssertionError(f"{metrics_path} must contain a JSON object")
    key_metrics = payload.get("metrics")
    if not isinstance(key_metrics, dict):
        raise AssertionError(f"{metrics_path}.metrics must be a JSON object")

    summary_payload = {
        "workflow": "run-single-factor",
        "experiment_name": case_name,
        "status": "success",
        "config_path": str(spec_path),
        "key_metrics": key_metrics,
        "promotion_decision": {
            "verdict": key_metrics.get("promotion_decision"),
            "reasons": key_metrics.get("promotion_reasons"),
            "blockers": key_metrics.get("promotion_blockers"),
            "source": "level2_promotion_gate",
        },
        "outputs": {
            "portfolio_validation_summary": str(portfolio_validation_summary_path),
            "portfolio_validation_metrics": str(portfolio_validation_metrics_path),
            "portfolio_validation_package": str(portfolio_validation_package_path),
        },
    }
    summary_path = output_dir / f"{case_name}_workflow_summary.json"
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    return summary_path


def _update_goldens_enabled() -> bool:
    raw = os.getenv(UPDATE_GOLDENS_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}
