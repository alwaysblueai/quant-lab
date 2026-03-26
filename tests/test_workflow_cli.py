from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.cli import main


def _make_prices(
    *,
    n_assets: int = 10,
    n_days: int = 110,
    seed: int = 11,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows: list[dict[str, object]] = []
    for i in range(n_assets):
        asset = f"A{i:03d}"
        price = 50.0 + i
        for date in dates:
            ret = rng.normal(0.0002 * (i % 3 - 1), 0.01)
            price = max(price * (1.0 + ret), 1.0)
            volume = int(rng.integers(100_000, 250_000))
            rows.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "asset": asset,
                    "close": float(price),
                    "volume": volume,
                    "dollar_volume": float(price) * float(volume),
                }
            )
    return pd.DataFrame(rows)


def _write_inputs(tmp_path: Path) -> dict[str, Path]:
    prices = _make_prices()
    prices_path = tmp_path / "prices.csv"
    prices.to_csv(prices_path, index=False)

    listing_date = pd.Timestamp("2022-01-01")
    assets = sorted(prices["asset"].astype(str).unique().tolist())
    asset_metadata = pd.DataFrame(
        {
            "asset": assets,
            "listing_date": [listing_date] * len(assets),
            "is_st": [False] * len(assets),
        }
    )
    asset_metadata_path = tmp_path / "asset_metadata.csv"
    asset_metadata.to_csv(asset_metadata_path, index=False)

    market_state = prices[["date", "asset"]].drop_duplicates().copy()
    market_state["is_halted"] = False
    market_state["is_limit_locked"] = False
    market_state["is_st"] = False
    market_state_path = tmp_path / "market_state.csv"
    market_state.to_csv(market_state_path, index=False)

    return {
        "prices_path": prices_path,
        "asset_metadata_path": asset_metadata_path,
        "market_state_path": market_state_path,
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_workflow_cli_single_factor_run_writes_summary_and_side_effects(tmp_path: Path) -> None:
    inputs = _write_inputs(tmp_path)
    config_path = _write_json(
        tmp_path / "single_config.json",
        {
            "data": {
                "prices_path": inputs["prices_path"].name,
                "asset_metadata_path": inputs["asset_metadata_path"].name,
                "market_state_path": inputs["market_state_path"].name,
            },
            "factor": {"name": "momentum", "params": {"window": 10}},
            "spec": {
                "experiment_name": "single_cli_workflow",
                "horizon": 5,
                "n_quantiles": 5,
                "validation_mode": "purged_kfold",
                "purged_n_splits": 4,
                "append_trial_log": False,
                "update_registry": False,
                "export_handoff": False,
            },
        },
    )

    output_dir = tmp_path / "single_out"
    rc = main(
        [
            "run-single-factor",
            "--config-path",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--write-trial-log",
            "--update-registry",
            "--export-handoff",
        ]
    )
    assert rc == 0

    summary_path = output_dir / "single_cli_workflow_single_factor_workflow_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["workflow"] == "run-single-factor"
    assert payload["status"] == "success"
    assert payload["promotion_decision"]["verdict"] in {
        "reject",
        "needs_review",
        "candidate_for_registry",
        "candidate_for_external_backtest",
    }
    assert (output_dir / "trial_log.csv").exists()
    assert (output_dir / "alpha_registry.csv").exists()
    assert (output_dir / "handoff").exists()


def test_workflow_cli_composite_run_writes_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    inputs = _write_inputs(tmp_path)
    config_path = _write_json(
        tmp_path / "composite_config.json",
        {
            "data": {
                "prices_path": inputs["prices_path"].name,
                "asset_metadata_path": inputs["asset_metadata_path"].name,
                "market_state_path": inputs["market_state_path"].name,
            },
            "factors": [
                {"name": "momentum", "params": {"window": 10}},
                {"name": "reversal", "params": {"window": 5}},
                {"name": "low_volatility", "params": {"window": 15}},
            ],
            "spec": {
                "experiment_name": "composite_cli_workflow",
                "horizon": 5,
                "n_quantiles": 5,
                "append_trial_log": False,
                "update_registry": False,
                "export_handoff": False,
            },
        },
    )
    output_dir = tmp_path / "composite_out"

    rc = main(
        [
            "run-composite",
            "--config-path",
            str(config_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert rc == 0

    captured = capsys.readouterr()
    assert "PromotionDecision" in captured.out
    assert "Summary JSON" in captured.out

    summary_path = output_dir / "composite_cli_workflow_composite_workflow_summary.json"
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["workflow"] == "run-composite"
    assert payload["status"] == "success"
    assert "selected_factor_count" in payload["key_metrics"]


def test_workflow_cli_invalid_json_config_exits(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.json"
    bad_config.write_text("{not-valid-json}", encoding="utf-8")
    with pytest.raises(SystemExit):
        main(
            [
                "run-single-factor",
                "--config-path",
                str(bad_config),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )


def test_workflow_cli_missing_required_input_exits(tmp_path: Path) -> None:
    prices = _make_prices()
    prices_path = tmp_path / "prices.csv"
    prices.to_csv(prices_path, index=False)
    config_path = _write_json(
        tmp_path / "missing_asset_meta.json",
        {
            "data": {
                "prices_path": prices_path.name,
            },
            "factor": {"name": "momentum", "params": {"window": 10}},
            "spec": {"experiment_name": "missing_inputs"},
        },
    )
    with pytest.raises(SystemExit):
        main(
            [
                "run-single-factor",
                "--config-path",
                str(config_path),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )


def test_workflow_cli_empty_factor_output_error_is_actionable(
    tmp_path: Path,
    capsys: pytest.CaptureFixture,
) -> None:
    inputs = _write_inputs(tmp_path)
    config_path = _write_json(
        tmp_path / "empty_factor_output.json",
        {
            "data": {
                "prices_path": inputs["prices_path"].name,
                "asset_metadata_path": inputs["asset_metadata_path"].name,
                "market_state_path": inputs["market_state_path"].name,
            },
            "factor": {"name": "momentum", "params": {"window": 10}},
            "spec": {
                "experiment_name": "empty_factor_case",
                "horizon": 5,
                "n_quantiles": 5,
                "universe_rules": {
                    "min_listing_age_days": 120,
                    "min_adv": 1_000_000_000.0,
                    "adv_window": 20,
                },
            },
        },
    )
    with pytest.raises(SystemExit):
        main(
            [
                "run-single-factor",
                "--config-path",
                str(config_path),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
    captured = capsys.readouterr()
    assert "min_adv" in captured.err
