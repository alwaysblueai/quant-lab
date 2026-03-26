from __future__ import annotations

from pathlib import Path

import pandas as pd

from alpha_lab.backtest_adapter.schema import BacktestInputBundle
from alpha_lab.backtest_adapter.target_weights import build_target_weights
from alpha_lab.handoff import ExecutionAssumptionsSpec, PortfolioConstructionSpec


def _base_bundle(
    *,
    portfolio: PortfolioConstructionSpec,
    execution: ExecutionAssumptionsSpec | None = None,
    signal_values: dict[str, float] | None = None,
    tradable: dict[str, bool] | None = None,
) -> BacktestInputBundle:
    date = pd.Timestamp("2024-01-02")
    assets = ["A", "B", "C", "D"]
    values = signal_values or {"A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}
    tradable_flags = tradable or {"A": True, "B": True, "C": True, "D": True}

    signal = pd.DataFrame(
        {
            "date": [date] * len(assets),
            "asset": assets,
            "signal_name": ["sig"] * len(assets),
            "signal_value": [values[a] for a in assets],
        }
    )
    universe = pd.DataFrame(
        {
            "date": [date] * len(assets),
            "asset": assets,
            "in_universe": [True] * len(assets),
        }
    )
    tradability = pd.DataFrame(
        {
            "date": [date] * len(assets),
            "asset": assets,
            "is_tradable": [tradable_flags[a] for a in assets],
        }
    )
    return BacktestInputBundle(
        artifact_path=Path("/tmp/fake_bundle"),
        schema_version="2.0.0",
        manifest={"schema_version": "2.0.0"},
        signal_snapshot_df=signal,
        universe_mask_df=universe,
        tradability_mask_df=tradability,
        timing_payload={"delay_spec": {"execution_delay_periods": 1}},
        experiment_metadata_payload={"experiment_id": "exp"},
        validation_context_payload={},
        dataset_fingerprint_payload={"fingerprint": "fp"},
        portfolio_construction=portfolio,
        execution_assumptions=execution or ExecutionAssumptionsSpec(),
    )


def _weights_by_asset(target_weights_df: pd.DataFrame) -> dict[str, float]:
    return {
        str(row.asset): float(row.target_weight)
        for row in target_weights_df.itertuples(index=False)
    }


def test_topk_equal_long_only_weights() -> None:
    bundle = _base_bundle(
        portfolio=PortfolioConstructionSpec(
            signal_name="sig",
            long_short=False,
            top_k=2,
            bottom_k=None,
            weight_method="equal",
            gross_limit=1.0,
            net_limit=0.0,
            cash_buffer=0.0,
            max_weight=1.0,
        )
    )
    intent = build_target_weights(bundle)
    weights = _weights_by_asset(intent.target_weights_df)
    assert weights["A"] == 0.5
    assert weights["B"] == 0.5
    assert weights["C"] == 0.0
    assert weights["D"] == 0.0


def test_topbottom_equal_long_short_weights() -> None:
    bundle = _base_bundle(
        portfolio=PortfolioConstructionSpec(
            signal_name="sig",
            long_short=True,
            top_k=1,
            bottom_k=1,
            weight_method="equal",
            gross_limit=1.0,
            net_limit=0.0,
            cash_buffer=0.0,
            max_weight=1.0,
        )
    )
    intent = build_target_weights(bundle)
    weights = _weights_by_asset(intent.target_weights_df)
    assert weights["A"] == 0.5
    assert weights["D"] == -0.5
    assert weights["B"] == 0.0
    assert weights["C"] == 0.0


def test_non_tradable_assets_are_masked_when_trade_not_tradable_false() -> None:
    bundle = _base_bundle(
        portfolio=PortfolioConstructionSpec(
            signal_name="sig",
            long_short=False,
            top_k=1,
            bottom_k=None,
            weight_method="equal",
            gross_limit=1.0,
            net_limit=0.0,
            cash_buffer=0.0,
            max_weight=1.0,
        ),
        tradable={"A": False, "B": True, "C": True, "D": True},
    )
    intent = build_target_weights(bundle)
    weights = _weights_by_asset(intent.target_weights_df)
    assert weights["A"] == 0.0
    assert weights["B"] == 1.0


def test_cash_buffer_reduces_gross_exposure() -> None:
    bundle = _base_bundle(
        portfolio=PortfolioConstructionSpec(
            signal_name="sig",
            long_short=True,
            top_k=1,
            bottom_k=1,
            weight_method="equal",
            gross_limit=1.0,
            net_limit=0.0,
            cash_buffer=0.1,
            max_weight=1.0,
        )
    )
    intent = build_target_weights(bundle)
    gross = float(intent.target_weights_df["target_weight"].abs().sum())
    assert gross == 0.9


def test_max_weight_constraint_is_enforced_and_output_is_deterministic() -> None:
    portfolio = PortfolioConstructionSpec(
        signal_name="sig",
        construction_method="full_universe",
        long_short=False,
        top_k=None,
        bottom_k=None,
        weight_method="rank",
        max_weight=0.3,
        gross_limit=1.0,
        net_limit=0.0,
        cash_buffer=0.0,
        weight_clip=0.8,
    )
    bundle = _base_bundle(portfolio=portfolio)
    first = build_target_weights(bundle).target_weights_df
    second = build_target_weights(bundle).target_weights_df
    assert first.equals(second)
    assert bool((first["target_weight"].abs() <= 0.3 + 1e-12).all())

