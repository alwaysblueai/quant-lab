from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alpha_lab.execution_impact_report import (
    ExecutionImpactThresholds,
    build_execution_impact_report,
    export_execution_impact_report,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _build_sample_run(
    tmp_path: Path,
    *,
    run_name: str,
    engine: str,
    include_weights: bool = True,
    include_turnover: bool = True,
    include_orders: bool = True,
    include_skipped: bool = True,
    warning_code: str = "adapter_warning",
) -> Path:
    run_path = tmp_path / run_name
    run_path.mkdir(parents=True, exist_ok=True)
    _write_json(
        run_path / "backtest_summary.json",
        {
            "engine": engine,
            "summary": {
                "total_return": 0.12,
                "sharpe_annualized": 1.1,
                "max_drawdown": -0.08,
                "mean_turnover": 0.25,
                "n_periods": 2,
            },
            "warnings": [{"code": warning_code, "message": "example warning"}],
        },
    )
    _write_json(
        run_path / "adapter_run_metadata.json",
        {
            "engine": engine,
            "warnings": [{"code": f"{warning_code}_metadata", "message": "metadata warning"}],
        },
    )

    if include_weights:
        _write_csv(
            run_path / "target_weights.csv",
            [
                {"date": "2024-01-02", "asset": "A", "target_weight": 0.5},
                {"date": "2024-01-02", "asset": "B", "target_weight": -0.5},
                {"date": "2024-01-03", "asset": "A", "target_weight": 0.5},
                {"date": "2024-01-03", "asset": "B", "target_weight": -0.5},
            ],
        )
        _write_csv(
            run_path / "executed_weights.csv",
            [
                {"date": "2024-01-02", "asset": "A", "target_weight": 0.5},
                {"date": "2024-01-02", "asset": "B", "target_weight": -0.5},
                {"date": "2024-01-03", "asset": "A", "target_weight": 0.1},
                {"date": "2024-01-03", "asset": "B", "target_weight": -0.5},
            ],
        )
    if include_turnover:
        _write_csv(
            run_path / "turnover.csv",
            [
                {"date": "2024-01-02", "turnover": 0.0},
                {"date": "2024-01-03", "turnover": 0.2},
            ],
        )
    if include_orders:
        _write_csv(
            run_path / "orders.csv",
            [
                {"date": "2024-01-02", "asset": "A", "size": 10},
                {"date": "2024-01-03", "asset": "B", "size": -8},
            ],
        )
    if include_skipped:
        _write_csv(
            run_path / "skipped_orders.csv",
            [
                {
                    "date": "2024-01-03",
                    "asset": "A",
                    "reason_code": "min_adv_filter",
                    "source_reason": "min_adv_filter",
                },
                {
                    "date": "2024-01-03",
                    "asset": "C",
                    "reason_code": "min_adv_filter",
                    "source_reason": "min_adv_filter",
                },
                {
                    "date": "2024-01-03",
                    "asset": "D",
                    "reason_code": "price_limit_locked",
                    "source_reason": "limit_locked_non_executable",
                },
            ],
        )
    return run_path


def _flags_by_name(report) -> dict[str, bool | None]:
    return {flag.name: flag.triggered for flag in report.flags}


def test_reason_aggregation_and_dominant_blocker(tmp_path: Path) -> None:
    run_path = _build_sample_run(tmp_path, run_name="run_reason", engine="backtrader")
    report = build_execution_impact_report(run_path)

    assert report.dominant_execution_blocker == "min_adv_filter"
    summary = report.reason_summary_df
    first = summary.iloc[0]
    assert str(first["reason_code"]) == "min_adv_filter"
    assert int(first["skipped_order_count"]) == 2
    assert float(first["skipped_order_ratio"]) == 2.0 / 3.0


def test_missing_artifacts_degrade_gracefully(tmp_path: Path) -> None:
    run_path = _build_sample_run(
        tmp_path,
        run_name="run_missing",
        engine="backtrader",
        include_weights=False,
        include_turnover=False,
        include_orders=False,
        include_skipped=False,
    )
    report = build_execution_impact_report(run_path)

    missing = set(report.missing_artifacts)
    assert "target_weights.csv" in missing
    assert "executed_weights.csv" in missing
    unavailable_metrics = {item.metric for item in report.unavailable_metrics}
    assert "execution_deviation" in unavailable_metrics
    assert "target_turnover" in unavailable_metrics
    assert report.execution_deviation_summary["mean_abs_weight_diff"] is None


def test_flag_logic_uses_explicit_thresholds(tmp_path: Path) -> None:
    run_path = _build_sample_run(tmp_path, run_name="run_flags", engine="backtrader")
    report = build_execution_impact_report(
        run_path,
        thresholds=ExecutionImpactThresholds(
            high_execution_deviation_mean_abs=0.01,
            severe_tradability_skipped_ratio=0.50,
            price_limit_reason_ratio=0.30,
            liquidity_reason_ratio=0.50,
            reentry_reason_ratio=0.01,
            material_turnover_reduction_ratio=0.05,
        ),
    )
    flags = _flags_by_name(report)
    assert flags["dominant_execution_blocker"] is True
    assert flags["high_execution_deviation"] is True
    assert flags["severe_tradability_constraints"] is True
    assert flags["price_limit_sensitive"] is True
    assert flags["liquidity_sensitive"] is True
    assert flags["reentry_constraint_sensitive"] is False


def test_comparison_mode_reports_vectorbt_vs_backtrader(tmp_path: Path) -> None:
    primary = _build_sample_run(
        tmp_path,
        run_name="primary_bt",
        engine="backtrader",
        warning_code="bt_warning",
    )
    comparison = _build_sample_run(
        tmp_path,
        run_name="comparison_vbt",
        engine="vectorbt",
        warning_code="vbt_warning",
    )
    report = build_execution_impact_report(primary, comparison_run_path=comparison)

    assert report.comparison_summary is not None
    comparison_summary = report.comparison_summary
    assert comparison_summary["target_weights_equal"] is True
    assert comparison_summary["primary_engine"] == "backtrader"
    assert comparison_summary["comparison_engine"] == "vectorbt"
    notes = comparison_summary["notes"]
    assert isinstance(notes, list)
    assert len(notes) == 1


def test_export_is_deterministic_and_writes_csvs(tmp_path: Path) -> None:
    run_path = _build_sample_run(tmp_path, run_name="run_export", engine="backtrader")
    report = build_execution_impact_report(run_path)
    out_dir = tmp_path / "exported"

    files_first = export_execution_impact_report(report, output_dir=out_dir)
    json_first = (out_dir / "execution_impact_report.json").read_text(encoding="utf-8")
    files_second = export_execution_impact_report(report, output_dir=out_dir)
    json_second = (out_dir / "execution_impact_report.json").read_text(encoding="utf-8")

    assert json_first == json_second
    assert files_first["report_json"] == out_dir / "execution_impact_report.json"
    assert files_second["report_json"] == out_dir / "execution_impact_report.json"
    assert (out_dir / "execution_impact_by_reason.csv").exists()
    assert (out_dir / "execution_impact_timeseries.csv").exists()
