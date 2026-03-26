from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.factors.low_volatility import low_volatility
from alpha_lab.factors.momentum import momentum
from alpha_lab.factors.reversal import reversal
from alpha_lab.research_templates import (
    CompositeWorkflowSpec,
    NeutralizationSpec,
    SingleFactorWorkflowSpec,
    run_composite_signal_research_workflow,
    run_single_factor_research_workflow,
)


def _make_prices(
    *,
    n_assets: int = 12,
    n_days: int = 120,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows: list[dict[str, object]] = []
    for i, asset in enumerate([f"A{i:03d}" for i in range(n_assets)]):
        price = 100.0 + i
        for date in dates:
            drift = 0.0005 * (i % 4 - 1.5)
            vol = 0.008 + 0.001 * (i % 3)
            ret = drift + rng.normal(0.0, vol)
            price = max(price * (1.0 + ret), 1.0)
            volume = int(rng.integers(80_000, 350_000))
            rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "close": float(price),
                    "volume": volume,
                    "dollar_volume": float(price) * float(volume),
                }
            )
    return pd.DataFrame(rows)


def _asset_metadata(prices: pd.DataFrame) -> pd.DataFrame:
    first = pd.Timestamp(prices["date"].min()) - pd.Timedelta(days=200)
    assets = sorted(pd.unique(prices["asset"].astype(str)).tolist())
    return pd.DataFrame(
        {
            "asset": assets,
            "listing_date": [first] * len(assets),
            "is_st": [False] * len(assets),
        }
    )


def _market_state(prices: pd.DataFrame) -> pd.DataFrame:
    base = prices[["date", "asset"]].drop_duplicates().copy()
    base["is_halted"] = False
    base["is_limit_locked"] = False
    base["is_st"] = False
    # Inject one known non-tradable point for exclusion-reason propagation.
    if len(base) > 0:
        base.loc[0, "is_halted"] = True
    return base.reset_index(drop=True)


def _exposures(prices: pd.DataFrame) -> pd.DataFrame:
    base = prices[["date", "asset", "close"]].copy()
    base["date"] = pd.to_datetime(base["date"], errors="coerce")
    asset_code = (
        base["asset"]
        .astype(str)
        .str.replace("A", "", regex=False)
        .astype(int)
    )
    base["size_exposure"] = np.log(base["close"]) + asset_code * 0.001
    base["beta_exposure"] = 0.8 + (asset_code % 5) * 0.08
    base["industry"] = np.where(
        asset_code % 3 == 0,
        "IND_A",
        np.where(asset_code % 3 == 1, "IND_B", "IND_C"),
    )
    return base[["date", "asset", "size_exposure", "beta_exposure", "industry"]]


def _momentum_fn(prices: pd.DataFrame) -> pd.DataFrame:
    return momentum(prices, window=10)


def test_single_factor_workflow_wires_trial_registry_and_handoff(tmp_path: Path) -> None:
    prices = _make_prices()
    trial_log_path = tmp_path / "trial_log.csv"
    registry_path = tmp_path / "alpha_registry.csv"
    handoff_dir = tmp_path / "handoff"

    spec = SingleFactorWorkflowSpec(
        experiment_name="single_momentum_template",
        factor_fn=_momentum_fn,
        horizon=5,
        n_quantiles=5,
        neutralization=NeutralizationSpec(
            size_col="size_exposure",
            industry_col="industry",
            beta_col="beta_exposure",
            min_obs=5,
        ),
        label_method="rankpct",
        validation_mode="purged_kfold",
        purged_n_splits=4,
        append_trial_log=True,
        trial_log_path=trial_log_path,
        update_registry=True,
        registry_path=registry_path,
        registry_alpha_id="alpha_mom_template",
        export_handoff=True,
        handoff_output_dir=handoff_dir,
        handoff_artifact_name="single_factor_handoff",
        dataset_id="synthetic_v1",
        trial_id="trial-1",
        trial_count=3,
    )

    result = run_single_factor_research_workflow(
        prices,
        spec=spec,
        asset_metadata=_asset_metadata(prices),
        market_state=_market_state(prices),
        neutralization_exposures=_exposures(prices),
    )

    assert result.validation_summary is not None
    assert not result.validation_summary.empty
    assert result.trial_log_row is not None
    assert trial_log_path.exists()
    trial_log_df = pd.read_csv(trial_log_path)
    assert len(trial_log_df) == 1
    assert result.registry_entry is not None
    assert registry_path.exists()
    assert result.handoff_export is not None
    assert result.handoff_export.artifact_path.exists()
    assert result.experiment_result.metadata is not None
    assert result.experiment_result.metadata.verdict == result.decision.verdict


def test_single_factor_workflow_supports_walk_forward_mode() -> None:
    prices = _make_prices(n_days=140)
    spec = SingleFactorWorkflowSpec(
        experiment_name="single_walk_forward",
        factor_fn=_momentum_fn,
        horizon=5,
        n_quantiles=5,
        validation_mode="walk_forward",
        walk_forward_train_size=60,
        walk_forward_test_size=20,
        walk_forward_step=20,
    )

    result = run_single_factor_research_workflow(
        prices,
        spec=spec,
        asset_metadata=_asset_metadata(prices),
        market_state=_market_state(prices),
    )

    assert result.walk_forward_result is not None
    assert result.validation_summary is not None
    assert len(result.walk_forward_result.per_fold_results) > 0


def test_single_factor_workflow_purged_kfold_scales_to_realistic_panel() -> None:
    prices = _make_prices(n_assets=300, n_days=242, seed=19)
    spec = SingleFactorWorkflowSpec(
        experiment_name="single_purged_realistic_panel",
        factor_fn=_momentum_fn,
        horizon=5,
        n_quantiles=5,
        label_method="rankpct",
        validation_mode="purged_kfold",
        purged_n_splits=5,
        purged_embargo_periods=1,
    )

    result = run_single_factor_research_workflow(
        prices,
        spec=spec,
        asset_metadata=_asset_metadata(prices),
        market_state=_market_state(prices),
    )

    assert result.validation_summary is not None
    assert not result.validation_summary.empty
    assert (result.validation_summary["n_test"] > 0).all()
    assert (result.validation_summary["n_train"] > 0).all()


def test_composite_workflow_end_to_end_with_diagnostics(tmp_path: Path) -> None:
    prices = _make_prices()
    trial_log_path = tmp_path / "trial_log_composite.csv"
    registry_path = tmp_path / "alpha_registry_composite.csv"
    handoff_dir = tmp_path / "handoff_composite"

    factor_fns = {
        "momentum_10d": lambda p: momentum(p, window=10),
        "reversal_5d": lambda p: reversal(p, window=5),
        "low_vol_15d": lambda p: low_volatility(p, window=15),
    }
    spec = CompositeWorkflowSpec(
        experiment_name="composite_template",
        horizon=5,
        n_quantiles=5,
        neutralization=NeutralizationSpec(
            size_col="size_exposure",
            beta_col="beta_exposure",
            industry_col="industry",
            min_obs=5,
        ),
        append_trial_log=True,
        trial_log_path=trial_log_path,
        update_registry=True,
        registry_path=registry_path,
        registry_alpha_id="alpha_composite_template",
        export_handoff=True,
        handoff_output_dir=handoff_dir,
        handoff_artifact_name="composite_handoff",
        dataset_id="synthetic_v1",
        trial_id="trial-2",
        trial_count=5,
    )

    result = run_composite_signal_research_workflow(
        prices,
        spec=spec,
        factor_fns=factor_fns,
        asset_metadata=_asset_metadata(prices),
        market_state=_market_state(prices),
        neutralization_exposures=_exposures(prices),
        exposure_data=_exposures(prices),
    )

    assert result.selected_signals["factor"].nunique() >= 2
    assert not result.alpha_pool_diagnostics.breadth_summary.empty
    assert not result.portfolio_weights.empty
    assert result.capacity_diagnostics is not None
    assert result.cost_diagnostics is not None
    assert result.exposure_audit is not None
    assert result.handoff_export is not None
    assert result.handoff_export.artifact_path.exists()
    assert result.trial_log_row is not None
    assert trial_log_path.exists()
    assert result.registry_entry is not None
    assert registry_path.exists()


def test_composite_workflow_requires_exactly_one_signal_source() -> None:
    prices = _make_prices(n_assets=6, n_days=60)
    spec = CompositeWorkflowSpec(experiment_name="invalid_source")

    with pytest.raises(ValueError, match="exactly one"):
        run_composite_signal_research_workflow(prices, spec=spec)

    with pytest.raises(ValueError, match="exactly one"):
        run_composite_signal_research_workflow(
            prices,
            spec=spec,
            factor_fns={"momentum": _momentum_fn},
            candidate_signals=_momentum_fn(prices),
        )
