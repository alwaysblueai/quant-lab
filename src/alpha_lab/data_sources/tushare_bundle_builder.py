from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from alpha_lab.config import PROCESSED_DATA_DIR
from alpha_lab.data_sources.tushare_schemas import RESEARCH_INPUT_TUSHARE_DIRNAME
from alpha_lab.data_sources.tushare_standardize import build_standardized_tushare_tables
from alpha_lab.factors.momentum import momentum
from alpha_lab.research_contracts import DatasetSnapshot, ResearchBundle


@dataclass(frozen=True)
class TushareBundleArtifacts:
    """Workflow-compatible research inputs built from standardized Tushare tables."""

    output_dir: Path
    prices_path: Path
    asset_metadata_path: Path
    market_state_path: Path
    neutralization_exposures_path: Path
    candidate_signals_path: Path
    manifest_path: Path
    bundle: ResearchBundle
    unavailable_inputs: tuple[str, ...]


@dataclass(frozen=True)
class TushareCaseArtifacts:
    """Config files and data paths for the two canonical real-data cases."""

    output_dir: Path
    single_factor_config_path: Path
    composite_config_path: Path
    data_manifest_path: Path


def load_standardized_tushare_tables(standardized_dir: str | Path) -> dict[str, pd.DataFrame]:
    root = Path(standardized_dir).resolve()
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"standardized directory does not exist: {root}")
    return {
        "prices": pd.read_csv(root / "prices.csv", parse_dates=["date"]),
        "asset_metadata": pd.read_csv(root / "asset_metadata.csv", parse_dates=["listing_date"]),
        "trade_calendar": pd.read_csv(
            root / "trade_calendar.csv",
            parse_dates=["date", "pretrade_date"],
        ),
        "market_state": pd.read_csv(root / "market_state.csv", parse_dates=["date"]),
        "daily_fundamentals": pd.read_csv(root / "daily_fundamentals.csv", parse_dates=["date"]),
        "financial_indicators": pd.read_csv(
            root / "financial_indicators.csv",
            parse_dates=["announce_date", "report_period"],
        ),
    }


def build_tushare_research_inputs(
    standardized_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    dataset_id: str | None = None,
) -> TushareBundleArtifacts:
    """Build workflow-compatible research input tables from standardized Tushare data."""

    root = Path(standardized_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"standardized Tushare directory does not exist: {root}")
    tables = load_standardized_tushare_tables(root)
    manifest = _load_manifest(root)
    unavailable_raw = cast(object, manifest.get("unavailable_raw_endpoints", []))
    unavailable_items = (
        unavailable_raw if isinstance(unavailable_raw, list | tuple | set) else []
    )
    unavailable = tuple(sorted({str(item) for item in unavailable_items if str(item).strip()}))

    prices = _build_prices_table(tables["prices"])
    asset_metadata = _build_asset_metadata_table(tables["asset_metadata"])
    market_state = _build_market_state_table(tables["market_state"])
    exposures = _build_neutralization_exposures(
        prices=prices,
        daily_fundamentals=tables["daily_fundamentals"],
        asset_metadata=asset_metadata,
    )
    candidate_signals = _build_candidate_signals(
        prices=prices,
        daily_fundamentals=tables["daily_fundamentals"],
        financial_indicators=tables["financial_indicators"],
    )

    target_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else PROCESSED_DATA_DIR / RESEARCH_INPUT_TUSHARE_DIRNAME / root.name
    )
    target_dir.mkdir(parents=True, exist_ok=True)

    prices_path = target_dir / "prices.csv"
    asset_metadata_path = target_dir / "asset_metadata.csv"
    market_state_path = target_dir / "market_state.csv"
    exposures_path = target_dir / "neutralization_exposures.csv"
    candidate_signals_path = target_dir / "candidate_signals_vqm.csv"

    _write_df(prices, prices_path)
    _write_df(asset_metadata, asset_metadata_path)
    _write_df(market_state, market_state_path)
    _write_df(exposures, exposures_path)
    _write_df(candidate_signals, candidate_signals_path)

    snapshot = DatasetSnapshot(
        dataset_id=dataset_id or root.name,
        source="tushare_pro",
        version=root.name,
        notes="Standardized Tushare bundle for canonical A-share research cases",
    )
    bundle = ResearchBundle(
        prices=prices.copy(),
        factors=candidate_signals.copy(),
        metadata=asset_metadata.copy(),
        snapshot=snapshot,
    )
    bundle.validate()

    manifest_path = target_dir / "manifest.json"
    notes = [
        (
            "single-factor reversal case consumes "
            "prices/asset_metadata/market_state/neutralization_exposures"
        ),
        "composite VQM case additionally consumes candidate_signals",
        (
            "candidate_signals use inverse pb as value proxy and "
            "PIT-aligned profitability as quality proxy"
        ),
    ]
    if "fina_indicator" in unavailable or tables["financial_indicators"].empty:
        notes.append(
            "quality_profitability_proxy is degraded or missing because financial indicators "
            "were unavailable for this snapshot window"
        )

    manifest_payload = {
        "standardized_dir": str(root),
        "research_input_dir": str(target_dir),
        "dataset_id": snapshot.dataset_id,
        "unavailable_inputs": list(unavailable),
        "files": {
            "prices": str(prices_path),
            "asset_metadata": str(asset_metadata_path),
            "market_state": str(market_state_path),
            "neutralization_exposures": str(exposures_path),
            "candidate_signals": str(candidate_signals_path),
        },
        "notes": notes,
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return TushareBundleArtifacts(
        output_dir=target_dir,
        prices_path=prices_path,
        asset_metadata_path=asset_metadata_path,
        market_state_path=market_state_path,
        neutralization_exposures_path=exposures_path,
        candidate_signals_path=candidate_signals_path,
        manifest_path=manifest_path,
        bundle=bundle,
        unavailable_inputs=unavailable,
    )


def export_canonical_tushare_case_configs(
    artifacts: TushareBundleArtifacts,
    *,
    output_dir: str | Path | None = None,
    dataset_id: str | None = None,
) -> TushareCaseArtifacts:
    """Emit workflow-ready config JSON files for the two canonical real-data cases."""

    target_dir = Path(output_dir).resolve() if output_dir is not None else artifacts.output_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    if dataset_id is not None:
        dataset_name = dataset_id
    elif artifacts.bundle.snapshot is not None:
        dataset_name = artifacts.bundle.snapshot.dataset_id
    else:
        dataset_name = "tushare_dataset"

    single_config = {
        "data": {
            "prices_path": str(artifacts.prices_path),
            "asset_metadata_path": str(artifacts.asset_metadata_path),
            "market_state_path": str(artifacts.market_state_path),
            "neutralization_exposures_path": str(artifacts.neutralization_exposures_path),
        },
        "factor": {"name": "reversal", "params": {"window": 5}},
        "spec": {
            "experiment_name": "tushare_single_reversal_liquidity_screened",
            "horizon": 5,
            "n_quantiles": 5,
            "label_method": "rankpct",
            "validation_mode": "purged_kfold",
            "purged_n_splits": 5,
            "purged_embargo_periods": 1,
            "universe_rules": {
                "min_listing_age_days": 120,
                "min_adv": 20_000_000.0,
                "adv_window": 20,
            },
            "preprocess": {
                "apply_winsorize": True,
                "winsorize_lower": 0.02,
                "winsorize_upper": 0.98,
                "apply_zscore": True,
                "apply_rank": False,
                "min_group_size": 10,
                "min_coverage": 0.7,
            },
            "neutralization": {
                "size_col": "size_exposure",
                "industry_col": "industry",
                "beta_col": "beta_exposure",
                "min_obs": 20,
                "ridge": 1e-8,
            },
            "dataset_id": dataset_name,
            "append_trial_log": True,
            "update_registry": False,
            "export_handoff": True,
            "handoff_include_label_snapshot": True,
        },
    }
    composite_config = {
        "data": {
            "prices_path": str(artifacts.prices_path),
            "asset_metadata_path": str(artifacts.asset_metadata_path),
            "market_state_path": str(artifacts.market_state_path),
            "neutralization_exposures_path": str(artifacts.neutralization_exposures_path),
            "exposure_data_path": str(artifacts.neutralization_exposures_path),
            "candidate_signals_path": str(artifacts.candidate_signals_path),
        },
        "spec": {
            "experiment_name": "tushare_composite_value_quality_momentum",
            "horizon": 5,
            "n_quantiles": 5,
            "label_method": "rankpct",
            "universe_rules": {
                "min_listing_age_days": 120,
                "min_adv": 20_000_000.0,
                "adv_window": 20,
            },
            "preprocess": {
                "apply_winsorize": True,
                "winsorize_lower": 0.01,
                "winsorize_upper": 0.99,
                "apply_zscore": True,
                "apply_rank": False,
                "min_group_size": 10,
                "min_coverage": 0.7,
            },
            "neutralization": {
                "size_col": "size_exposure",
                "industry_col": "industry",
                "beta_col": "beta_exposure",
                "min_obs": 20,
                "ridge": 1e-8,
            },
            "screening_min_coverage": 0.7,
            "screening_min_abs_monotonicity": 0.05,
            "screening_max_pairwise_corr": 0.9,
            "screening_max_vif": 15.0,
            "composite_method": "icir",
            "composite_lookback": 63,
            "composite_min_history": 20,
            "composite_factor_name": "vqm_composite",
            "portfolio_top_k": 15,
            "portfolio_bottom_k": 15,
            "portfolio_weighting_method": "rank",
            "portfolio_value": 10_000_000.0,
            "adv_window": 20,
            "capacity_max_adv_participation": 0.05,
            "capacity_concentration_weight_threshold": 0.08,
            "cost_flat_fee_bps": 1.5,
            "cost_spread_bps": 6.0,
            "cost_impact_eta": 0.12,
            "dataset_id": dataset_name,
            "append_trial_log": True,
            "update_registry": True,
            "export_handoff": True,
            "handoff_include_label_snapshot": True,
        },
    }

    single_path = target_dir / "tushare_case_single_reversal.json"
    composite_path = target_dir / "tushare_case_composite_vqm.json"
    data_manifest_path = target_dir / "tushare_case_data_manifest.json"

    single_path.write_text(
        json.dumps(single_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    composite_path.write_text(
        json.dumps(composite_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    data_manifest = {
        "research_input_dir": str(artifacts.output_dir),
        "single_factor_config_path": str(single_path),
        "composite_config_path": str(composite_path),
        "unavailable_inputs": list(artifacts.unavailable_inputs),
    }
    data_manifest_path.write_text(
        json.dumps(data_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return TushareCaseArtifacts(
        output_dir=target_dir,
        single_factor_config_path=single_path,
        composite_config_path=composite_path,
        data_manifest_path=data_manifest_path,
    )


def materialize_tushare_research_case_data(
    snapshot_dir: str | Path,
    *,
    standardized_dir: str | Path | None = None,
    research_input_dir: str | Path | None = None,
) -> tuple[TushareBundleArtifacts, TushareCaseArtifacts]:
    """Convenience wrapper: raw snapshots -> standardized tables -> case configs."""

    standardized = build_standardized_tushare_tables(
        snapshot_dir,
        output_dir=standardized_dir,
    )
    inputs = build_tushare_research_inputs(
        standardized.snapshot_dir,
        output_dir=research_input_dir,
    )
    cases = export_canonical_tushare_case_configs(inputs, output_dir=inputs.output_dir)
    return inputs, cases


def _build_prices_table(prices: pd.DataFrame) -> pd.DataFrame:
    out = prices.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
    return out[["date", "asset", "close", "volume", "dollar_volume"]].copy()


def _build_asset_metadata_table(asset_metadata: pd.DataFrame) -> pd.DataFrame:
    out = asset_metadata.copy()
    out["listing_date"] = pd.to_datetime(out["listing_date"], errors="coerce")
    keep = ["asset", "listing_date", "is_st"]
    if "industry" in out.columns:
        keep.append("industry")
    return out[keep].copy()


def _build_market_state_table(market_state: pd.DataFrame) -> pd.DataFrame:
    out = market_state.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out[["date", "asset", "is_halted", "is_limit_locked", "is_st"]].copy()


def _build_neutralization_exposures(
    *,
    prices: pd.DataFrame,
    daily_fundamentals: pd.DataFrame,
    asset_metadata: pd.DataFrame,
) -> pd.DataFrame:
    size = daily_fundamentals[["date", "asset", "total_mv_yuan"]].copy()
    size["size_exposure"] = np.log(size["total_mv_yuan"].where(size["total_mv_yuan"] > 0))
    size = size.drop(columns=["total_mv_yuan"])

    industry = asset_metadata[["asset", "industry"]].copy()
    out = prices[["date", "asset", "close"]].copy().merge(size, on=["date", "asset"], how="left")
    out = out.merge(industry, on="asset", how="left")
    out["beta_exposure"] = _rolling_market_beta(prices)
    return out[["date", "asset", "size_exposure", "beta_exposure", "industry"]].sort_values(
        ["date", "asset"],
        kind="mergesort",
    ).reset_index(drop=True)


def _rolling_market_beta(prices: pd.DataFrame, *, window: int = 60) -> pd.Series:
    frame = prices[["date", "asset", "close"]].copy()
    wide = frame.pivot(index="date", columns="asset", values="close").sort_index()
    returns = wide.pct_change()
    market_return = returns.mean(axis=1, skipna=True)
    long = (
        returns.rename_axis(index="date", columns="asset")
        .reset_index()
        .melt(id_vars="date", var_name="asset", value_name="asset_return")
    )
    long = long.merge(
        market_return.rename("market_return").reset_index(),
        on="date",
        how="left",
    )
    long = long.sort_values(["asset", "date"], kind="mergesort").reset_index(drop=True)
    betas: list[pd.Series] = []
    for _asset, group in long.groupby("asset", sort=False):
        asset_return = group["asset_return"]
        market = group["market_return"]
        cov = asset_return.rolling(window=window, min_periods=20).cov(market)
        var = market.rolling(window=window, min_periods=20).var()
        betas.append(cov.div(var.replace(0.0, np.nan)))
    beta_series = pd.concat(betas).sort_index()
    return beta_series.reset_index(drop=True)


def _build_candidate_signals(
    *,
    prices: pd.DataFrame,
    daily_fundamentals: pd.DataFrame,
    financial_indicators: pd.DataFrame,
) -> pd.DataFrame:
    momentum_signal = momentum(prices[["date", "asset", "close"]].copy(), window=63)
    momentum_signal["factor"] = "momentum_63d"

    value_signal = daily_fundamentals[["date", "asset", "pb"]].copy()
    value_signal["factor"] = "value_book_to_price_proxy"
    value_signal["value"] = 1.0 / value_signal["pb"].where(value_signal["pb"] > 0)
    value_signal = value_signal[["date", "asset", "factor", "value"]]

    profitability = _pit_quality_panel(prices=prices, financial_indicators=financial_indicators)
    profitability["factor"] = "quality_profitability_proxy"
    profitability = profitability[["date", "asset", "factor", "value"]]

    combined = pd.concat(
        [
            momentum_signal[["date", "asset", "factor", "value"]],
            value_signal,
            profitability,
        ],
        ignore_index=True,
    )
    combined = combined.sort_values(["date", "asset", "factor"], kind="mergesort").reset_index(
        drop=True
    )
    return combined


def _pit_quality_panel(
    *,
    prices: pd.DataFrame,
    financial_indicators: pd.DataFrame,
) -> pd.DataFrame:
    trading_dates = prices[["date", "asset"]].drop_duplicates().sort_values(
        ["asset", "date"],
        kind="mergesort",
    )
    if financial_indicators.empty:
        out = trading_dates.copy()
        out["value"] = np.nan
        return out

    source = financial_indicators.copy().sort_values(
        ["asset", "announce_date", "report_period"],
        kind="mergesort",
    )
    source["quality_value"] = source["roe_dt"]
    source["quality_value"] = source["quality_value"].where(
        source["quality_value"].notna(),
        source["roe"],
    )
    source["quality_value"] = source["quality_value"].where(
        source["quality_value"].notna(),
        source["roa"],
    )
    source = source[["asset", "announce_date", "quality_value"]].copy()
    source = source.dropna(subset=["quality_value"]).reset_index(drop=True)
    if source.empty:
        out = trading_dates.copy()
        out["value"] = np.nan
        return out

    panels: list[pd.DataFrame] = []
    for asset, asset_dates in trading_dates.groupby("asset", sort=False):
        asset_source = source[source["asset"] == asset]
        merged = pd.merge_asof(
            asset_dates.sort_values("date", kind="mergesort"),
            asset_source.sort_values("announce_date", kind="mergesort"),
            left_on="date",
            right_on="announce_date",
            direction="backward",
            allow_exact_matches=True,
        )
        merged = merged.rename(columns={"asset_x": "asset"})
        panels.append(merged[["date", "asset", "quality_value"]])
    out = pd.concat(panels, ignore_index=True).rename(columns={"quality_value": "value"})
    return out.sort_values(["date", "asset"], kind="mergesort").reset_index(drop=True)


def _load_manifest(root: Path) -> dict[str, object]:
    path = root / "manifest.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return payload


def _write_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
