from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from alpha_lab.backtest_adapter.backtrader_adapter import run_backtrader_backtest
from alpha_lab.backtest_adapter.loader import load_backtest_input_bundle
from alpha_lab.backtest_adapter.schema import BacktestRunConfig
from alpha_lab.backtest_adapter.target_weights import build_target_weights
from alpha_lab.execution_impact_report import (
    build_execution_impact_report,
    export_execution_impact_report,
)
from alpha_lab.research_package import build_research_package, export_research_package


def _write_minimal_bundle(base: Path) -> Path:
    bundle = base / "handoff" / "demo_single_factor"
    bundle.mkdir(parents=True)

    dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    assets = ["A", "B", "C"]

    signal_rows = []
    universe_rows = []
    tradable_rows = []
    label_rows = []
    for i, date in enumerate(dates):
        for asset, signal in zip(
            assets, [0.5 - i * 0.1, 0.0 + i * 0.05, -0.4 + i * 0.05], strict=False
        ):
            signal_rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "signal_name": "reversal_5d",
                    "signal_value": signal,
                }
            )
            universe_rows.append({"date": date, "asset": asset, "in_universe": True})
            tradable_rows.append(
                {"date": date, "asset": asset, "is_tradable": not (asset == "C" and i == 1)}
            )
            label_rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "label_name": "forward_return_1",
                    "label_value": signal * 0.01,
                }
            )

    pd.DataFrame(signal_rows).to_csv(bundle / "signal_snapshot.csv", index=False)
    pd.DataFrame(universe_rows).to_csv(bundle / "universe_mask.csv", index=False)
    pd.DataFrame(tradable_rows).to_csv(bundle / "tradability_mask.csv", index=False)
    pd.DataFrame(label_rows).to_csv(bundle / "label_snapshot.csv", index=False)
    pd.DataFrame(
        [
            {
                "date": dates[1],
                "asset": "C",
                "reason": "min_adv_filter",
                "detail": "synthetic",
            }
        ]
    ).to_csv(bundle / "exclusion_reasons.csv", index=False)

    (bundle / "timing.json").write_text(
        json.dumps(
            {
                "schema_version": "2.0.0",
                "delay_spec": {"execution_delay_periods": 1},
                "label_metadata": {"label_name": "forward_return_1"},
            }
        ),
        encoding="utf-8",
    )
    (bundle / "experiment_metadata.json").write_text(
        json.dumps({"schema_version": "2.0.0", "experiment_id": "demo"}), encoding="utf-8"
    )
    (bundle / "validation_context.json").write_text(
        json.dumps({"schema_version": "2.0.0"}), encoding="utf-8"
    )
    (bundle / "dataset_fingerprint.json").write_text(
        json.dumps({"schema_version": "2.0.0", "fingerprint": "abc"}), encoding="utf-8"
    )
    (bundle / "portfolio_construction.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "construction_method": "top_bottom_k",
                "signal_name": "reversal_5d",
                "rebalance_frequency": 1,
                "rebalance_calendar": "business_day",
                "long_short": True,
                "top_k": 1,
                "bottom_k": 1,
                "weight_method": "rank",
                "max_weight": 1.0,
                "gross_limit": 1.0,
                "net_limit": 0.0,
                "cash_buffer": 0.0,
                "neutralization_required": False,
                "post_construction_constraints": [],
            }
        ),
        encoding="utf-8",
    )
    (bundle / "execution_assumptions.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "fill_price_rule": "next_close",
                "execution_delay_bars": 1,
                "commission_model": "bps",
                "slippage_model": "fixed_bps",
                "lot_size_rule": "none",
                "lot_size": None,
                "cash_buffer": 0.0,
                "partial_fill_policy": "allow_partial",
                "suspension_policy": "skip_trade",
                "price_limit_policy": "skip_trade",
                "trade_when_not_tradable": False,
                "allow_same_day_reentry": False,
            }
        ),
        encoding="utf-8",
    )
    (bundle / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_name": "demo_single_factor",
                "experiment_id": "demo_single_factor",
                "schema_version": "2.0.0",
                "dataset_fingerprint": "abc",
            }
        ),
        encoding="utf-8",
    )
    return bundle


def _minimal_prices() -> pd.DataFrame:
    rows = []
    closes = {
        "A": [10.0, 10.2, 10.1, 10.3],
        "B": [20.0, 20.1, 20.3, 20.2],
        "C": [30.0, 29.8, 29.9, 30.1],
    }
    for i, date in enumerate(
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    ):
        for asset in ["A", "B", "C"]:
            rows.append({"date": date, "asset": asset, "close": closes[asset][i]})
    return pd.DataFrame(rows)


def test_load_bundle_and_target_weights(tmp_path: Path) -> None:
    bundle_path = _write_minimal_bundle(tmp_path)
    bundle = load_backtest_input_bundle(bundle_path)
    intent = build_target_weights(bundle)

    assert set(intent.target_weights_df.columns) == {
        "date",
        "asset",
        "signal_name",
        "signal_value",
        "in_universe",
        "is_tradable",
        "is_executable",
        "target_weight",
    }
    gross = intent.target_weights_df.groupby("date")["target_weight"].apply(lambda s: s.abs().sum())
    assert (gross <= 1.0 + 1e-9).all()


def test_backtrader_adapter_exports(tmp_path: Path) -> None:
    bundle_path = _write_minimal_bundle(tmp_path)
    bundle = load_backtest_input_bundle(bundle_path)
    out_dir = tmp_path / "case" / "replay_compare" / "backtrader"
    result = run_backtrader_backtest(
        bundle,
        BacktestRunConfig(
            price_df=_minimal_prices(),
            engine="backtrader",
            output_dir=out_dir,
            export_summary=True,
            export_target_weights=True,
            export_series=True,
        ),
    )
    assert result.output_files
    assert (out_dir / "backtest_summary.json").exists()
    assert (out_dir / "portfolio_returns.csv").exists()
    assert (out_dir / "target_weights.csv").exists()


def test_execution_impact_report_and_export(tmp_path: Path) -> None:
    bundle_path = _write_minimal_bundle(tmp_path)
    bundle = load_backtest_input_bundle(bundle_path)

    run_a = tmp_path / "case" / "replay_compare" / "backtrader"
    run_b = tmp_path / "case" / "replay_compare" / "vectorbt_proxy"
    run_backtrader_backtest(
        bundle,
        BacktestRunConfig(price_df=_minimal_prices(), engine="backtrader", output_dir=run_a),
    )
    run_backtrader_backtest(
        bundle,
        BacktestRunConfig(
            price_df=_minimal_prices(),
            engine="backtrader",
            output_dir=run_b,
            commission_bps=5.0,
        ),
    )

    report = build_execution_impact_report(run_path=run_a, comparison_run_path=run_b)
    files = export_execution_impact_report(report, tmp_path / "case" / "execution_impact")
    assert files["report_json"].exists()
    assert report.execution_deviation_summary


def test_build_and_export_research_package(tmp_path: Path) -> None:
    case_dir = tmp_path / "case"
    bundle_path = _write_minimal_bundle(case_dir)
    prices = _minimal_prices()

    replay_bt = case_dir / "replay_compare" / "backtrader"
    replay_vbt = case_dir / "replay_compare" / "vectorbt"
    bundle = load_backtest_input_bundle(bundle_path)
    run_backtrader_backtest(
        bundle,
        BacktestRunConfig(price_df=prices, engine="backtrader", output_dir=replay_bt),
    )
    run_backtrader_backtest(
        bundle,
        BacktestRunConfig(
            price_df=prices, engine="backtrader", output_dir=replay_vbt, slippage_bps=5.0
        ),
    )

    report = build_execution_impact_report(run_path=replay_bt, comparison_run_path=replay_vbt)
    export_execution_impact_report(report, case_dir / "execution_impact")

    workflow = {
        "workflow": "run-single-factor",
        "experiment_name": "demo_single_factor",
        "status": "success",
        "outputs": {"handoff_artifact": str(bundle_path)},
        "promotion_decision": {"verdict": "candidate_for_external_backtest"},
    }
    (case_dir / "demo_single_factor_workflow_summary.json").write_text(
        json.dumps(workflow),
        encoding="utf-8",
    )

    package = build_research_package(
        case_dir,
        case_id="demo_case",
        case_name="Demo Case",
    )
    files = export_research_package(package, case_dir / "research_package")
    assert files["json"].exists()
    assert files["markdown"].exists()
    assert package.verdict.verdict == "candidate_for_external_backtest"
