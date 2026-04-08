from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from alpha_lab.experiment import run_factor_experiment
from alpha_lab.factors.momentum import momentum
from alpha_lab.neutralization import neutralize_signal
from alpha_lab.real_cases.single_factor.pipeline import run_single_factor_case
from alpha_lab.research_integrity.exceptions import IntegrityHardFailure
from alpha_lab.walk_forward import run_walk_forward_experiment
from tests.single_factor_case_helpers import write_demo_single_factor_case


def _make_prices(n_assets: int = 4, n_days: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows: list[dict[str, object]] = []
    for i in range(n_assets):
        asset = f"A{i:03d}"
        price = 100.0 + i
        for date in dates:
            price *= 1.001
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


def test_experiment_fails_fast_on_obvious_future_factor_dates() -> None:
    prices = _make_prices()

    def leaky_factor_fn(input_prices: pd.DataFrame) -> pd.DataFrame:
        factor = momentum(input_prices, window=5)
        factor = factor.copy()
        factor["date"] = pd.to_datetime(factor["date"]) + pd.Timedelta(days=45)
        return factor

    with pytest.raises(IntegrityHardFailure):
        run_factor_experiment(prices, leaky_factor_fn, horizon=5, n_quantiles=5)


def test_walk_forward_builds_fold_and_aggregate_integrity_reports() -> None:
    prices = _make_prices(n_days=80)

    result = run_walk_forward_experiment(
        prices,
        lambda p: momentum(p, window=5),
        train_size=30,
        test_size=10,
        step=10,
        horizon=5,
    )

    assert result.aggregate_integrity_report is not None
    assert len(result.fold_integrity_reports) == result.aggregate_summary.n_folds
    assert result.aggregate_integrity_report.summary.n_checks > 0


def test_walk_forward_raises_when_fold_factor_has_future_dates() -> None:
    prices = _make_prices(n_days=80)

    def leaky_factor_fn(input_prices: pd.DataFrame) -> pd.DataFrame:
        factor = momentum(input_prices, window=5)
        factor = factor.copy()
        factor["date"] = pd.to_datetime(factor["date"]) + pd.Timedelta(days=45)
        return factor

    with pytest.raises(IntegrityHardFailure):
        run_walk_forward_experiment(
            prices,
            leaky_factor_fn,
            train_size=30,
            test_size=10,
            step=10,
            horizon=5,
        )


def test_single_factor_pipeline_writes_integrity_artifacts(tmp_path: Path) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")
    result = run_single_factor_case(spec_path)

    assert (result.output_dir / "integrity_report.json").exists()
    assert (result.output_dir / "integrity_report.md").exists()
    assert "integrity_report_json" in result.artifact_paths
    assert "integrity_report_markdown" in result.artifact_paths


def test_single_factor_pipeline_catches_scope_violations_from_universe_alignment(
    tmp_path: Path,
) -> None:
    spec_path = write_demo_single_factor_case(tmp_path, factor_name="bp")

    # Inject one extra investable pair not present in prices.
    universe_path = tmp_path / "inputs" / "universe.csv"
    universe = pd.read_csv(universe_path)
    extra = universe.iloc[[0]].copy()
    extra["asset"] = "X999"
    extra["in_universe"] = True
    universe = pd.concat([universe, extra], ignore_index=True)
    universe.to_csv(universe_path, index=False)

    def factor_loader(spec) -> pd.DataFrame:
        frame = pd.read_csv(spec.factor_path)
        extra_factor = frame.iloc[[0]].copy()
        extra_factor["asset"] = "X999"
        extra_factor["factor"] = spec.factor_name
        return pd.concat([frame, extra_factor], ignore_index=True)

    with pytest.raises(IntegrityHardFailure):
        run_single_factor_case(spec_path, factor_loader=factor_loader)


def test_neutralization_rejects_future_known_at_exposures() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-01-05"]),
            "asset": ["AAA", "BBB"],
            "value": [1.2, -0.4],
            "size": [10.0, 11.0],
            "known_at": pd.to_datetime(["2024-01-06", "2024-01-06"]),
        }
    )

    with pytest.raises(ValueError, match="timing failed integrity check"):
        neutralize_signal(
            df,
            value_col="value",
            by="date",
            size_col="size",
            min_obs=1,
            known_at_col="known_at",
            enforce_integrity=True,
        )
