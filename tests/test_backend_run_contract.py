"""Library-level tests for ``alpha_lab.backend_run_contract``.

Commit 1: library + doc only — no ``real-case`` wiring exists yet, so these
tests synthesize minimal-valid run directories and exercise the audit / receipt
/ comparison / manifest-attach surfaces directly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from alpha_lab.backend_run_contract import (
    BACKEND_RUN_CONTRACT_VERSION,
    BACKEND_RUN_RECEIPT_NAME,
    COMPARISON_SUMMARY_NAME,
    attach_backend_contract_to_manifest,
    audit_backend_run_artifacts,
    build_backend_run_receipt,
    build_comparison_summary,
    detect_research_draft_run,
    finalize_backend_contract,
    write_backend_run_receipt,
    write_comparison_summary,
)
from alpha_lab.custom_factors import read_custom_factor_source
from alpha_lab.custom_models import read_draft_model_source

_FACTOR_CODE = (
    "def build_factor(frame):\n"
    "    return frame['close']\n"
)


# ----------------------------------------------------------------------------
# Fixture builders
# ----------------------------------------------------------------------------


def _write_factor_draft(tmp_path: Path, name: str = "demo_factor") -> Path:
    factor_dir = tmp_path / "custom_factors" / "research" / name
    factor_dir.mkdir(parents=True)
    factor_json = factor_dir / "factor.json"
    factor_json.write_text(
        json.dumps(
            {
                "name": name,
                "description": "backend contract library test factor",
                "required_columns": ["date", "asset", "close"],
                "frequency": "daily",
                "unavailable_data_policy": "return_nan",
                "pit_assumption": "uses trailing close only",
                "code": _FACTOR_CODE,
                "provenance": {
                    "idea_id": "20260521T000000Z__library-test",
                    "stage2_payload_sha256": "a" * 64,
                    "audience_chain": ["claude", "codex", "web_gpt_stage2"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return factor_json


def _write_model_candidate(tmp_path: Path, name: str = "demo_model") -> Path:
    candidate_dir = tmp_path / "custom_models" / "research" / name
    candidate_dir.mkdir(parents=True)
    candidate_json = candidate_dir / "model_candidate.json"
    case_spec_payload = {
        "name": f"{name}_case",
        "factor_name": name,
        "feature_columns": ["feat_a", "feat_b"],
        "feature_availability": {"mode": "required_timestamp", "column": "known_at"},
        "model": {"family": "ridge"},
    }
    candidate_json.write_text(
        json.dumps(
            {
                "contract_version": "stage2_model_candidate_v1",
                "candidate_name": name,
                "implementation_status": "draft_for_stage3",
                "implementation_type": "spec_variant",
                "case_spec_payload": case_spec_payload,
                "provenance": {
                    "idea_id": "20260521T010000Z__library-test",
                    "stage2_payload_sha256": "b" * 64,
                    "audience_chain": ["claude", "codex", "web_gpt_stage2"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return candidate_json


def _make_single_factor_run(
    tmp_path: Path,
    *,
    factor_json: Path,
    case_name: str = "demo_factor_case",
) -> Path:
    source_audit = read_custom_factor_source(factor_json).to_audit_dict()
    run_dir = tmp_path / "outputs" / case_name
    (run_dir / "level2_portfolio_validation").mkdir(parents=True)

    _write_presence_stubs(run_dir, _COMMON_STUBS)
    _write_json(run_dir / "run_manifest.json", _run_manifest(case_name, source_audit, "factor"))
    _write_json(run_dir / "metrics.json", _metrics_payload())
    _write_json(
        run_dir / "factor_definition.json",
        _factor_definition(case_name, source_audit),
    )
    _write_json(run_dir / "signal_validation.json", _signal_validation(case_name))
    _write_json(run_dir / "portfolio_recipe.json", _portfolio_recipe(case_name))
    _write_json(run_dir / "backtest_result.json", _backtest_result(case_name))
    _write_json(run_dir / "research_tearsheet.json", {"placeholder": True})
    _write_json(run_dir / "integrity_report.json", {"placeholder": True})
    for stub in _LEVEL2_JSON_STUBS:
        _write_json(run_dir / stub, {"placeholder": True})
    _write_json(run_dir / "factor_definition.yaml", {"placeholder": True})
    return run_dir


def _make_model_factor_run(
    tmp_path: Path,
    *,
    candidate_json: Path,
    case_name: str = "demo_model_case",
) -> Path:
    source_audit = read_draft_model_source(candidate_json).to_audit_dict()
    run_dir = tmp_path / "outputs" / case_name
    (run_dir / "level2_portfolio_validation").mkdir(parents=True)

    _write_presence_stubs(run_dir, _COMMON_STUBS + _MODEL_STUBS)
    _write_json(run_dir / "run_manifest.json", _run_manifest(case_name, source_audit, "model"))
    _write_json(run_dir / "metrics.json", _metrics_payload())
    _write_json(
        run_dir / "factor_definition.json",
        _factor_definition(case_name, source_audit, package_type="single_factor"),
    )
    _write_json(run_dir / "signal_validation.json", _signal_validation(case_name))
    _write_json(run_dir / "portfolio_recipe.json", _portfolio_recipe(case_name))
    _write_json(run_dir / "backtest_result.json", _backtest_result(case_name))
    _write_json(run_dir / "research_tearsheet.json", {"placeholder": True})
    _write_json(run_dir / "integrity_report.json", {"placeholder": True})
    for stub in _LEVEL2_JSON_STUBS:
        _write_json(run_dir / stub, {"placeholder": True})
    _write_json(
        run_dir / "model_definition.json",
        {
            "artifact_type": "alpha_lab_model_definition",
            "case_name": case_name,
            "factor_name": source_audit["factor_name"],
            "model_family": source_audit.get("model_family") or "ridge",
            "source_artifacts": {"feature_manifest": "feature_manifest.json"},
            "draft_model_source": source_audit,
        },
    )
    _write_json(
        run_dir / "feature_manifest.json",
        {
            "artifact_type": "alpha_lab_feature_manifest",
            "case_name": case_name,
            "factor_name": source_audit["factor_name"],
            "model_family": source_audit.get("model_family") or "ridge",
            "feature_columns": list(
                cast(list[object], source_audit.get("feature_columns") or [])
            ),
            "feature_availability": {"mode": "required_timestamp", "column": "known_at"},
            "draft_model_source": source_audit,
        },
    )
    return run_dir


_COMMON_STUBS: tuple[str, ...] = (
    "summary.md",
    "integrity_report.md",
    "experiment_card.md",
    "case_report.md",
    "research_tearsheet.pdf",
    "level2_portfolio_validation/portfolio_validation_package.md",
)

_MODEL_STUBS: tuple[str, ...] = (
    "model_factor_definition.yaml",
    "model_selection.json",
    "diagnostics.json",
    "training_log.csv",
    "training_metrics.csv",
    "feature_importance.csv",
    "feature_importance_ledger.csv",
    "feature_oos_ic.csv",
)

_LEVEL2_JSON_STUBS: tuple[str, ...] = (
    "level2_portfolio_validation/portfolio_validation_summary.json",
    "level2_portfolio_validation/portfolio_validation_metrics.json",
    "level2_portfolio_validation/portfolio_validation_package.json",
)


def _write_presence_stubs(run_dir: Path, paths: tuple[str, ...]) -> None:
    for relative in paths:
        target = run_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("stub", encoding="utf-8")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _issue_codes(audit: Mapping[str, object]) -> set[str]:
    raw = audit.get("issues")
    assert isinstance(raw, list)
    return {str(cast(Mapping[str, object], item)["code"]) for item in raw}


def _run_manifest(
    case_name: str,
    source_audit: Mapping[str, object],
    kind: str,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "artifact_type": "real_case_single_factor_bundle",
        "run_timestamp_utc": "2026-05-21T00:00:00+00:00",
        "case_name": case_name,
        "outputs": {"factor_path": "factor.parquet"},
        "required_bundle_files": ["factor.parquet"],
        "integrity_summary": {"status": "ok"},
        "evaluation_standard": {
            "profile_name": "exploratory_screening",
            "snapshot": {"profile": "exploratory_screening"},
        },
    }
    if kind == "factor":
        payload["custom_factor_source"] = dict(source_audit)
        payload["inputs"] = {"custom_factor_source": dict(source_audit)}
    elif kind == "model":
        payload["draft_model_source"] = dict(source_audit)
        payload["inputs"] = {"draft_model_source": dict(source_audit)}
    return payload


def _metrics_payload() -> dict[str, object]:
    return {
        "metrics": {
            "research_evaluation_profile": "exploratory_screening",
            "campaign_triage": "pending_review",
            "promotion_decision": "needs_review",
            "ic_mean": 0.04,
            "rank_ic_mean": 0.05,
            "turnover_mean": 0.18,
        },
        "coverage_by_date_summary": {"n_dates": 100, "mean_coverage": 0.95, "min_coverage": 0.8},
    }


def _factor_definition(
    case_name: str,
    source_audit: Mapping[str, object],
    *,
    package_type: str = "single_factor",
) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_factor_definition",
        "case_name": case_name,
        "package_type": package_type,
        "factor_name": str(source_audit.get("name") or "demo"),
        "spec": {"placeholder": True},
        "source_artifacts": {"factor": "factor.parquet"},
        "custom_factor_source": dict(source_audit),
    }


def _signal_validation(case_name: str) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_signal_validation",
        "case_name": case_name,
        "package_type": "single_factor",
        "metrics": {
            "research_evaluation_profile": "exploratory_screening",
            "campaign_triage": "pending_review",
            "promotion_decision": "needs_review",
        },
        "coverage_by_date_summary": {"n_dates": 100},
        "source_artifacts": {"factor": "factor.parquet"},
    }


def _portfolio_recipe(case_name: str) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_portfolio_recipe",
        "case_name": case_name,
        "package_type": "single_factor",
        "recipe_context": {"placeholder": True},
        "portfolio_validation_summary": {
            "validation_status": "passed",
            "promotion_decision": "needs_review",
            "recommendation": "continue investigation",
        },
        "portfolio_validation_metrics": {
            "protocol_settings": {"placeholder": True},
            "scenario_metrics": [{"weighting_method": "equal", "holding_period": 5}],
            "holding_period_sensitivity": [],
            "weighting_sensitivity": [],
            "turnover_summary": {"placeholder": True},
            "transaction_cost_sensitivity": {"placeholder": True},
            "benchmark_relative_evaluation": {"placeholder": True},
        },
        "portfolio_validation_package": {
            "schema_version": "1.0.0",
            "package_type": "alpha_lab_level2_portfolio_validation_package",
            "created_at_utc": "2026-05-21T00:00:00+00:00",
            "input_case_identity": {"placeholder": True},
            "promotion_decision_context": {"placeholder": True},
            "portfolio_validation_settings": {"placeholder": True},
            "key_portfolio_results": {"placeholder": True},
            "recommendation": {"label": "continue investigation"},
        },
        "source_artifacts": {"factor": "factor.parquet"},
    }


def _backtest_result(case_name: str) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_backtest_result",
        "case_name": case_name,
        "package_type": "single_factor",
        "rebalance_frequency": "W",
        "summary": {
            "sharpe": 0.6,
            "max_drawdown": -0.12,
            "annualized_return": 0.08,
            "nav_points": [],
            "monthly_return_table": [],
            "drawdown_table": [],
        },
        "source_artifacts": {"factor": "factor.parquet"},
    }


# ----------------------------------------------------------------------------
# audit_backend_run_artifacts
# ----------------------------------------------------------------------------


def test_audit_passes_with_complete_single_factor_run(tmp_path: Path) -> None:
    factor_json = _write_factor_draft(tmp_path)
    run_dir = _make_single_factor_run(tmp_path, factor_json=factor_json)
    write_comparison_summary(run_dir, workflow="single_factor")

    audit = audit_backend_run_artifacts(
        run_dir,
        workflow="single_factor",
        draft_source_path=factor_json,
    )

    assert audit["ok"] is True
    assert audit["missing_files"] == []
    assert audit["contract_version"] == BACKEND_RUN_CONTRACT_VERSION
    assert audit["issues"] == []


def test_audit_passes_with_complete_model_factor_run(tmp_path: Path) -> None:
    candidate_json = _write_model_candidate(tmp_path)
    run_dir = _make_model_factor_run(tmp_path, candidate_json=candidate_json)
    write_comparison_summary(run_dir, workflow="model_factor")

    audit = audit_backend_run_artifacts(
        run_dir,
        workflow="model_factor",
        draft_source_path=candidate_json,
    )

    assert audit["ok"] is True
    assert audit["missing_files"] == []
    assert audit["issues"] == []


def test_audit_flags_missing_required_artifact(tmp_path: Path) -> None:
    factor_json = _write_factor_draft(tmp_path)
    run_dir = _make_single_factor_run(tmp_path, factor_json=factor_json)
    write_comparison_summary(run_dir, workflow="single_factor")
    (run_dir / "research_tearsheet.pdf").unlink()

    audit = audit_backend_run_artifacts(
        run_dir,
        workflow="single_factor",
        draft_source_path=factor_json,
    )

    assert audit["ok"] is False
    codes = _issue_codes(audit)
    assert "missing_required_artifact" in codes


def test_audit_flags_factor_source_hash_mismatch(tmp_path: Path) -> None:
    factor_json = _write_factor_draft(tmp_path)
    run_dir = _make_single_factor_run(tmp_path, factor_json=factor_json)
    write_comparison_summary(run_dir, workflow="single_factor")

    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    block = dict(cast(Mapping[str, object], manifest["custom_factor_source"]))
    block["code_sha256"] = "0" * 64
    manifest["custom_factor_source"] = block
    _write_json(manifest_path, manifest)

    audit = audit_backend_run_artifacts(
        run_dir,
        workflow="single_factor",
        draft_source_path=factor_json,
    )

    assert audit["ok"] is False
    codes = _issue_codes(audit)
    assert "custom_factor_source_hash_mismatch" in codes


def test_audit_flags_missing_source_audit_fields(tmp_path: Path) -> None:
    factor_json = _write_factor_draft(tmp_path)
    run_dir = _make_single_factor_run(tmp_path, factor_json=factor_json)
    write_comparison_summary(run_dir, workflow="single_factor")

    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["custom_factor_source"] = {}
    _write_json(manifest_path, manifest)

    audit = audit_backend_run_artifacts(
        run_dir,
        workflow="single_factor",
        draft_source_path=factor_json,
    )

    assert audit["ok"] is False
    codes = _issue_codes(audit)
    assert "custom_factor_source_missing_fields" in codes


def test_audit_flags_model_source_hash_mismatch(tmp_path: Path) -> None:
    candidate_json = _write_model_candidate(tmp_path)
    run_dir = _make_model_factor_run(tmp_path, candidate_json=candidate_json)
    write_comparison_summary(run_dir, workflow="model_factor")

    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    block = dict(cast(Mapping[str, object], manifest["draft_model_source"]))
    block["candidate_json_sha256"] = "0" * 64
    manifest["draft_model_source"] = block
    _write_json(manifest_path, manifest)

    audit = audit_backend_run_artifacts(
        run_dir,
        workflow="model_factor",
        draft_source_path=candidate_json,
    )

    assert audit["ok"] is False
    codes = _issue_codes(audit)
    assert "draft_model_source_hash_mismatch" in codes


# ----------------------------------------------------------------------------
# build_comparison_summary
# ----------------------------------------------------------------------------


def test_build_comparison_summary_without_baseline(tmp_path: Path) -> None:
    factor_json = _write_factor_draft(tmp_path)
    run_dir = _make_single_factor_run(tmp_path, factor_json=factor_json)

    payload = build_comparison_summary(run_dir, workflow="single_factor")

    assert payload["status"] == "not_available"
    assert payload["metric_deltas"] == {}
    assert payload["contract_version"] == BACKEND_RUN_CONTRACT_VERSION


def test_build_comparison_summary_with_baseline_emits_metric_deltas(tmp_path: Path) -> None:
    factor_json = _write_factor_draft(tmp_path)
    current = _make_single_factor_run(
        tmp_path, factor_json=factor_json, case_name="current_case"
    )
    baseline = _make_single_factor_run(
        tmp_path, factor_json=factor_json, case_name="baseline_case"
    )
    baseline_metrics = _read_json(baseline / "metrics.json")
    inner_metrics = dict(cast(Mapping[str, object], baseline_metrics["metrics"]))
    inner_metrics["ic_mean"] = 0.01
    baseline_metrics["metrics"] = inner_metrics
    _write_json(baseline / "metrics.json", baseline_metrics)

    payload = build_comparison_summary(
        current,
        workflow="single_factor",
        compare_to_output_dir=baseline,
    )

    assert payload["status"] == "available"
    deltas = payload["metric_deltas"]
    assert isinstance(deltas, dict)
    ic_entry = deltas["ic_mean"]
    assert isinstance(ic_entry, dict)
    assert ic_entry["current"] == 0.04
    assert ic_entry["baseline"] == 0.01
    assert ic_entry["delta"] == 0.04 - 0.01


# ----------------------------------------------------------------------------
# receipt + manifest attach
# ----------------------------------------------------------------------------


def test_build_and_write_backend_run_receipt(tmp_path: Path) -> None:
    factor_json = _write_factor_draft(tmp_path)
    run_dir = _make_single_factor_run(tmp_path, factor_json=factor_json)
    case_spec = tmp_path / "configs" / "real_cases" / "single_factor" / "demo.yaml"
    case_spec.parent.mkdir(parents=True)
    case_spec.write_text("name: demo\n", encoding="utf-8")
    write_comparison_summary(run_dir, workflow="single_factor")
    audit = audit_backend_run_artifacts(
        run_dir, workflow="single_factor", draft_source_path=factor_json
    )

    receipt = build_backend_run_receipt(
        workflow="single_factor",
        output_dir=run_dir,
        case_spec_path=case_spec,
        draft_source_path=factor_json,
        validation_payload={"ok": True},
        artifact_audit=audit,
        evaluation_profile="exploratory_screening",
        comparison_summary_path=run_dir / COMPARISON_SUMMARY_NAME,
        command=("alpha-lab", "real-case", "single-factor", "run", str(case_spec)),
    )
    path = write_backend_run_receipt(run_dir, receipt)

    assert path.name == BACKEND_RUN_RECEIPT_NAME
    assert path.exists()
    written = _read_json(path)
    assert written["status"] == "success"
    assert written["workflow"] == "single_factor"
    assert written["contract_version"] == BACKEND_RUN_CONTRACT_VERSION
    assert written["evaluation_profile"] == "exploratory_screening"
    assert written["validation"] == {"ok": True}
    command = written["command"]
    assert isinstance(command, list)
    assert command[0] == "alpha-lab"


def test_attach_backend_contract_to_manifest_records_sidecars(tmp_path: Path) -> None:
    factor_json = _write_factor_draft(tmp_path)
    run_dir = _make_single_factor_run(tmp_path, factor_json=factor_json)
    write_comparison_summary(run_dir, workflow="single_factor")
    audit = audit_backend_run_artifacts(
        run_dir, workflow="single_factor", draft_source_path=factor_json
    )
    receipt_path = run_dir / BACKEND_RUN_RECEIPT_NAME
    receipt_path.write_text("{}\n", encoding="utf-8")

    attach_backend_contract_to_manifest(
        run_dir,
        receipt_path=receipt_path,
        comparison_summary_path=run_dir / COMPARISON_SUMMARY_NAME,
        artifact_audit=audit,
    )

    manifest = _read_json(run_dir / "run_manifest.json")
    contract_block = manifest["backend_run_contract"]
    assert isinstance(contract_block, dict)
    assert contract_block["status"] == "passed"
    assert contract_block["contract_version"] == BACKEND_RUN_CONTRACT_VERSION
    outputs = manifest["outputs"]
    assert isinstance(outputs, dict)
    assert outputs["backend_run_receipt"].endswith(BACKEND_RUN_RECEIPT_NAME)  # type: ignore[union-attr]
    assert outputs["comparison_summary"].endswith(COMPARISON_SUMMARY_NAME)  # type: ignore[union-attr]


# ----------------------------------------------------------------------------
# finalize_backend_contract (end-to-end library wrapper)
# ----------------------------------------------------------------------------


def test_finalize_backend_contract_writes_sidecars_and_manifest_block(
    tmp_path: Path,
) -> None:
    factor_json = _write_factor_draft(tmp_path)
    run_dir = _make_single_factor_run(tmp_path, factor_json=factor_json)
    case_spec = tmp_path / "configs" / "real_cases" / "single_factor" / "demo.yaml"
    case_spec.parent.mkdir(parents=True)
    case_spec.write_text("name: demo\n", encoding="utf-8")

    receipt = finalize_backend_contract(
        run_dir,
        workflow="single_factor",
        draft_source_path=factor_json,
        case_spec_path=case_spec,
        evaluation_profile="exploratory_screening",
        command=("alpha-lab", "real-case", "single-factor", "run", str(case_spec)),
    )

    assert receipt["status"] == "success"
    assert (run_dir / BACKEND_RUN_RECEIPT_NAME).exists()
    assert (run_dir / COMPARISON_SUMMARY_NAME).exists()
    manifest = _read_json(run_dir / "run_manifest.json")
    contract = manifest["backend_run_contract"]
    assert isinstance(contract, dict)
    assert contract["status"] == "passed"
    assert contract["issue_count"] == 0


def test_finalize_marks_failure_when_required_artifact_missing(tmp_path: Path) -> None:
    factor_json = _write_factor_draft(tmp_path)
    run_dir = _make_single_factor_run(tmp_path, factor_json=factor_json)
    (run_dir / "experiment_card.md").unlink()
    case_spec = tmp_path / "configs" / "real_cases" / "single_factor" / "demo.yaml"
    case_spec.parent.mkdir(parents=True)
    case_spec.write_text("name: demo\n", encoding="utf-8")

    receipt = finalize_backend_contract(
        run_dir,
        workflow="single_factor",
        draft_source_path=factor_json,
        case_spec_path=case_spec,
        evaluation_profile="exploratory_screening",
    )

    assert receipt["status"] == "failed"
    audit = receipt["artifact_audit"]
    assert isinstance(audit, dict)
    assert audit["ok"] is False
    assert (run_dir / BACKEND_RUN_RECEIPT_NAME).exists()
    assert (run_dir / COMPARISON_SUMMARY_NAME).exists()
    manifest = _read_json(run_dir / "run_manifest.json")
    contract = manifest["backend_run_contract"]
    assert isinstance(contract, dict)
    assert contract["status"] == "failed"
    assert int(contract["issue_count"]) >= 1  # type: ignore[arg-type]


# ----------------------------------------------------------------------------
# detect_research_draft_run
# ----------------------------------------------------------------------------


def test_detect_research_draft_run_recognizes_research_factor(tmp_path: Path) -> None:
    factor_json = _write_factor_draft(tmp_path)
    run_dir = _make_single_factor_run(tmp_path, factor_json=factor_json)

    detected = detect_research_draft_run(run_dir, workflow="single_factor")

    assert detected is not None
    assert str(detected).endswith("factor.json")


def test_detect_research_draft_run_recognizes_research_model(tmp_path: Path) -> None:
    candidate_json = _write_model_candidate(tmp_path)
    run_dir = _make_model_factor_run(tmp_path, candidate_json=candidate_json)

    detected = detect_research_draft_run(run_dir, workflow="model_factor")

    assert detected is not None
    assert str(detected).endswith("model_candidate.json")


def test_detect_research_draft_run_ignores_non_research_path(tmp_path: Path) -> None:
    factor_json = _write_factor_draft(tmp_path)
    run_dir = _make_single_factor_run(tmp_path, factor_json=factor_json)
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    block = dict(cast(Mapping[str, object], manifest["custom_factor_source"]))
    block["path"] = str(tmp_path / "custom_factors" / "promoted" / "demo" / "factor.json")
    manifest["custom_factor_source"] = block
    _write_json(manifest_path, manifest)

    assert detect_research_draft_run(run_dir, workflow="single_factor") is None


def test_detect_research_draft_run_ignores_incomplete_source_block(tmp_path: Path) -> None:
    factor_json = _write_factor_draft(tmp_path)
    run_dir = _make_single_factor_run(tmp_path, factor_json=factor_json)
    manifest_path = run_dir / "run_manifest.json"
    manifest = _read_json(manifest_path)
    block = dict(cast(Mapping[str, object], manifest["custom_factor_source"]))
    block.pop("code_sha256")
    manifest["custom_factor_source"] = block
    _write_json(manifest_path, manifest)

    assert detect_research_draft_run(run_dir, workflow="single_factor") is None


def test_detect_research_draft_run_returns_none_when_manifest_missing(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "empty_run"
    run_dir.mkdir()

    assert detect_research_draft_run(run_dir, workflow="single_factor") is None
