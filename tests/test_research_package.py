from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.alpha_registry import ALPHA_REGISTRY_COLUMNS
from alpha_lab.research_package import (
    build_campaign_summary,
    build_research_package,
    export_research_package,
)
from alpha_lab.trial_log import TRIAL_LOG_COLUMNS


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]], *, columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(columns)).to_csv(path, index=False)


def _create_case(
    tmp_path: Path,
    *,
    case_name: str,
    promotion_verdict: str = "candidate_for_registry",
    promotion_blocking: tuple[str, ...] = (),
    include_trial_registry: bool = True,
    include_handoff: bool = True,
    include_replays: bool = True,
    include_execution_impact: bool = True,
    execution_flag_triggered: bool = False,
) -> Path:
    case_dir = tmp_path / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    trial_log_path = case_dir / "trial_log.csv"
    alpha_registry_path = case_dir / "alpha_registry.csv"
    handoff_bundle_path = case_dir / "handoff" / f"{case_name}_bundle"

    if include_trial_registry:
        _write_csv(
            trial_log_path,
            [
                {
                    "experiment_name": case_name,
                    "trial_id": f"{case_name}-trial-1",
                    "trial_count": 5,
                    "factor_name": "momentum_20d",
                    "label_name": "forward_return_5",
                    "dataset_id": "dataset-a",
                    "dataset_hash": "dataset-hash-a",
                    "validation_scheme": "purged_kfold",
                    "verdict": promotion_verdict,
                    "mean_ic": 0.015,
                    "mean_rank_ic": 0.021,
                    "ic_ir": 0.31,
                    "mean_long_short_return": 0.012,
                    "mean_long_short_turnover": 0.24,
                    "run_timestamp_utc": "2026-03-25T00:00:00+00:00",
                    "git_commit": "abc123",
                }
            ],
            columns=TRIAL_LOG_COLUMNS,
        )
        _write_csv(
            alpha_registry_path,
            [
                {
                    "alpha_id": case_name,
                    "taxonomy": "momentum",
                    "hypothesis": "short-term reversal alpha",
                    "economic_rationale": "liquidity overreaction",
                    "neutralization_status": "size_industry_beta",
                    "validation_status": "purged_kfold",
                    "ic_mean": 0.021,
                    "ic_ir": 0.31,
                    "decay_half_life": 6.0,
                    "lifecycle_stage": "candidate",
                    "tags": "momentum|short_horizon",
                    "notes": "good rank_ic",
                    "provenance": "abc123",
                    "updated_at_utc": "2026-03-25T00:00:00+00:00",
                }
            ],
            columns=ALPHA_REGISTRY_COLUMNS,
        )

    if include_handoff:
        handoff_bundle_path.mkdir(parents=True, exist_ok=True)
        _write_json(
            handoff_bundle_path / "manifest.json",
            {
                "schema_version": "2.0.0",
                "dataset_fingerprint": "fingerprint-001",
            },
        )
        _write_json(
            handoff_bundle_path / "dataset_fingerprint.json",
            {"fingerprint": "fingerprint-001"},
        )
        _write_json(
            handoff_bundle_path / "portfolio_construction.json",
            {
                "schema_version": "1.0.0",
                "construction_method": "top_bottom_k",
                "weight_method": "rank",
                "long_short": True,
                "top_k": 20,
                "bottom_k": 20,
                "gross_limit": 1.0,
                "net_limit": 0.0,
                "cash_buffer": 0.0,
                "rebalance_frequency": 1,
            },
        )
        _write_json(
            handoff_bundle_path / "execution_assumptions.json",
            {
                "schema_version": "1.0.0",
                "fill_price_rule": "next_open",
                "execution_delay_bars": 1,
                "commission_model": "bps",
                "slippage_model": "fixed_bps",
                "lot_size_rule": "none",
                "suspension_policy": "skip_trade",
                "price_limit_policy": "skip_trade",
                "trade_when_not_tradable": False,
                "allow_same_day_reentry": False,
            },
        )
        _write_json(
            handoff_bundle_path / "timing.json",
            {
                "delay_spec": {
                    "decision_timestamp": "close",
                    "execution_delay_periods": 1,
                    "return_horizon_periods": 5,
                    "label_start_offset_periods": 1,
                    "label_end_offset_periods": 6,
                }
            },
        )
        _write_json(
            handoff_bundle_path / "experiment_metadata.json",
            {
                "experiment_metadata": {
                    "hypothesis": "short-term reversal alpha",
                    "research_question": "can reversal survive costs?",
                    "factor_spec": "momentum_20d",
                    "dataset_id": "dataset-a",
                    "dataset_hash": "dataset-hash-a",
                    "trial_id": f"{case_name}-trial-1",
                    "trial_count": 5,
                    "validation": {"scheme": "purged_kfold"},
                }
            },
        )
        _write_json(
            handoff_bundle_path / "validation_context.json",
            {"mode": "purged_kfold", "n_splits": 5},
        )

    if include_replays:
        for engine, total_return in (("vectorbt", 0.11), ("backtrader", 0.09)):
            run_dir = case_dir / "replays" / f"{engine}_run"
            run_dir.mkdir(parents=True, exist_ok=True)
            _write_json(
                run_dir / "backtest_summary.json",
                {
                    "engine": engine,
                    "summary": {
                        "total_return": total_return,
                        "sharpe_annualized": 1.1,
                        "max_drawdown": -0.08,
                        "mean_turnover": 0.22,
                        "n_periods": 120,
                    },
                    "warnings": [],
                },
            )
            _write_json(
                run_dir / "adapter_run_metadata.json",
                {
                    "engine": engine,
                    "adapter_version": "1.0.0",
                    "warnings": [],
                },
            )

    if include_execution_impact:
        _write_json(
            case_dir / "execution_impact_report.json",
            {
                "dominant_execution_blocker": "min_adv_filter",
                "reason_summary": [
                    {
                        "reason_code": "min_adv_filter",
                        "skipped_order_count": 3,
                        "skipped_order_ratio": 0.6,
                    }
                ],
                "execution_deviation_summary": {
                    "mean_abs_weight_diff": 0.012 if not execution_flag_triggered else 0.09,
                    "max_abs_weight_diff": 0.14,
                    "gross_abs_diff_mean": 0.04,
                    "gross_abs_diff_max": 0.12,
                },
                "turnover_effect_summary": {
                    "skipped_order_ratio": 0.18,
                    "n_orders": 20,
                    "n_skipped_orders": 4,
                },
                "flags": [
                    {
                        "name": "high_execution_deviation",
                        "triggered": execution_flag_triggered,
                        "observed": 0.09 if execution_flag_triggered else 0.012,
                        "threshold": 0.02,
                    },
                    {
                        "name": "severe_tradability_constraints",
                        "triggered": False,
                        "observed": 0.18,
                        "threshold": 0.20,
                    },
                ],
                "warnings": [],
                "unavailable_metrics": [],
                "missing_artifacts": [],
            },
        )

    workflow_outputs: dict[str, object] = {
        "summary_json": str(case_dir / f"{case_name}_single_factor_workflow_summary.json"),
        "trial_log": str(trial_log_path) if include_trial_registry else None,
        "alpha_registry": str(alpha_registry_path) if include_trial_registry else None,
        "handoff_artifact": str(handoff_bundle_path) if include_handoff else None,
        "handoff_manifest": str(handoff_bundle_path / "manifest.json") if include_handoff else None,
    }
    _write_json(
        case_dir / f"{case_name}_single_factor_workflow_summary.json",
        {
            "workflow": "run-single-factor",
            "status": "success",
            "experiment_name": case_name,
            "promotion_decision": {
                "verdict": promotion_verdict,
                "reasons": ["rank_ic_passed"],
                "blocking_issues": list(promotion_blocking),
                "warnings": [],
                "metrics": {"mean_rank_ic": 0.021, "ic_ir": 0.31},
            },
            "key_metrics": {
                "mean_ic": 0.014,
                "mean_rank_ic": 0.021,
                "ic_ir": 0.31,
                "mean_long_short_return": 0.012,
            },
            "outputs": workflow_outputs,
        },
    )
    return case_dir


def test_build_research_package_from_complete_case(tmp_path: Path) -> None:
    case_dir = _create_case(tmp_path, case_name="case_complete")

    package = build_research_package(
        case_dir,
        created_at_utc="2026-03-25T00:00:00+00:00",
    )

    assert package.case_id == "case_complete"
    assert package.workflow_type == "run-single-factor"
    assert package.identity["dataset_fingerprint"] == "fingerprint-001"
    assert package.execution_impact is not None
    engines = {item.engine for item in package.replay_results}
    assert engines == {"vectorbt", "backtrader"}
    assert package.verdict.verdict == "candidate_for_registry"
    assert package.missing_artifacts == ()


def test_build_research_package_graceful_degradation(tmp_path: Path) -> None:
    case_dir = _create_case(
        tmp_path,
        case_name="case_missing",
        include_trial_registry=False,
        include_handoff=False,
        include_replays=False,
        include_execution_impact=False,
        promotion_verdict="candidate_for_external_backtest",
    )

    package = build_research_package(
        case_dir,
        created_at_utc="2026-03-25T00:00:00+00:00",
    )

    assert package.replay_results == ()
    assert package.execution_impact is None
    assert "replay_outputs" in package.missing_artifacts
    assert "execution_impact_report_json" in package.missing_artifacts
    assert package.verdict.verdict == "candidate_for_external_backtest"


def test_package_verdict_reflects_execution_flags(tmp_path: Path) -> None:
    case_dir = _create_case(
        tmp_path,
        case_name="case_execution_flag",
        execution_flag_triggered=True,
        promotion_verdict="candidate_for_registry",
    )

    package = build_research_package(
        case_dir,
        created_at_utc="2026-03-25T00:00:00+00:00",
    )

    assert package.verdict.verdict == "needs_review"
    assert "high_execution_deviation" in package.verdict.blocking_issues
    assert "execution_flag:high_execution_deviation" in package.verdict.execution_verdict_basis


def test_export_research_package_is_deterministic(tmp_path: Path) -> None:
    case_dir = _create_case(tmp_path, case_name="case_export")
    package = build_research_package(
        case_dir,
        created_at_utc="2026-03-25T00:00:00+00:00",
    )

    output_dir = tmp_path / "package_output"
    files_first = export_research_package(
        package,
        output_dir=output_dir,
        export_artifact_index=True,
    )
    json_first = (output_dir / "research_package.json").read_text(encoding="utf-8")
    markdown_first = (output_dir / "research_package.md").read_text(encoding="utf-8")
    index_first = (output_dir / "artifact_index.json").read_text(encoding="utf-8")

    files_second = export_research_package(
        package,
        output_dir=output_dir,
        export_artifact_index=True,
    )
    json_second = (output_dir / "research_package.json").read_text(encoding="utf-8")
    markdown_second = (output_dir / "research_package.md").read_text(encoding="utf-8")
    index_second = (output_dir / "artifact_index.json").read_text(encoding="utf-8")

    assert files_first["package_json"] == output_dir / "research_package.json"
    assert files_second["package_json"] == output_dir / "research_package.json"
    assert json_first == json_second
    assert markdown_first == markdown_second
    assert index_first == index_second

    artifact_index = json.loads(index_first)
    assert artifact_index["case_id"] == "case_export"
    assert any(
        row["name"] == "workflow_summary" and row["exists"]
        for row in artifact_index["artifacts"]
    )


def test_campaign_summary_aggregates_multiple_packages(tmp_path: Path) -> None:
    case_a = _create_case(
        tmp_path,
        case_name="campaign_case_a",
        promotion_verdict="candidate_for_registry",
    )
    case_b = _create_case(
        tmp_path,
        case_name="campaign_case_b",
        promotion_verdict="reject",
        include_replays=False,
        include_execution_impact=False,
    )
    package_a = build_research_package(case_a, created_at_utc="2026-03-25T00:00:00+00:00")
    package_b = build_research_package(case_b, created_at_utc="2026-03-25T00:00:00+00:00")

    campaign = build_campaign_summary(
        [package_b, package_a],
        campaign_id="campaign-1",
        generated_at_utc="2026-03-25T01:00:00+00:00",
    )

    assert campaign.case_ids == ("campaign_case_a", "campaign_case_b")
    assert campaign.verdict_distribution["candidate_for_registry"] == 1
    assert campaign.verdict_distribution["reject"] == 1
    assert campaign.top_candidates == ("campaign_case_a",)


def test_build_research_package_rejects_ambiguous_workflow_summary(tmp_path: Path) -> None:
    case_dir = _create_case(tmp_path, case_name="case_ambiguous_workflow")
    _write_json(
        case_dir / "zzz_single_factor_workflow_summary.json",
        {
            "workflow": "run-single-factor",
            "status": "success",
            "experiment_name": "zzz",
            "outputs": {},
        },
    )

    with pytest.raises(ValueError, match="multiple workflow summary files found"):
        build_research_package(case_dir, created_at_utc="2026-03-25T00:00:00+00:00")


def test_build_research_package_ignores_noncanonical_replay_dirs(tmp_path: Path) -> None:
    case_dir = _create_case(
        tmp_path,
        case_name="case_canonical_replay_only",
        include_replays=False,
        include_execution_impact=False,
    )
    for engine in ("vectorbt", "backtrader"):
        run_dir = case_dir / "replay_compare" / engine
        _write_json(
            run_dir / "backtest_summary.json",
            {
                "engine": engine,
                "summary": {
                    "total_return": 0.1,
                    "sharpe_annualized": 1.0,
                    "max_drawdown": -0.1,
                    "mean_turnover": 0.2,
                    "n_periods": 10,
                },
                "warnings": [],
            },
        )
        _write_json(
            run_dir / "adapter_run_metadata.json",
            {
                "engine": engine,
                "adapter_version": "1.0.0",
                "warnings": [],
            },
        )

    _write_json(
        case_dir / "replay_compare" / "stale_engine" / "backtest_summary.json",
        {
            "engine": "stale_engine",
            "summary": {
                "total_return": 999.0,
                "sharpe_annualized": 999.0,
                "max_drawdown": -0.99,
                "mean_turnover": 9.9,
                "n_periods": 1,
            },
            "warnings": [],
        },
    )

    package = build_research_package(case_dir, created_at_utc="2026-03-25T00:00:00+00:00")
    engines = {item.engine for item in package.replay_results}
    assert engines == {"vectorbt", "backtrader"}
