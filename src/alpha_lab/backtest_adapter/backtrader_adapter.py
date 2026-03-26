from __future__ import annotations

from dataclasses import asdict, replace

import pandas as pd

from alpha_lab.backtest_adapter.base import (
    compute_basic_performance_summary,
    export_backtest_result,
)
from alpha_lab.backtest_adapter.schema import (
    BACKTEST_ADAPTER_VERSION,
    AdapterWarning,
    BacktestInputBundle,
    BacktestResult,
    BacktestRunConfig,
)
from alpha_lab.backtest_adapter.target_weights import build_target_weights
from alpha_lab.backtest_adapter.validators import (
    validate_backtest_input_bundle,
    validate_price_inputs,
)

ENGINE_NAME = "backtrader"


def run_backtrader_backtest(
    bundle: BacktestInputBundle,
    config: BacktestRunConfig,
) -> BacktestResult:
    """Run a stricter execution-aware replay path under the Backtrader adapter."""
    if config.engine != ENGINE_NAME:
        raise ValueError(f"run_backtrader_backtest requires config.engine={ENGINE_NAME!r}")

    warnings: list[AdapterWarning] = [
        AdapterWarning(
            code="simplified_fill_model",
            message=(
                "Backtrader adapter v1 uses a deterministic bar-based fill model derived "
                "from bundle contracts; it is stricter than vectorbt v1 but not a full "
                "microstructure simulation"
            ),
        )
    ]
    if config.freq is None:
        warnings.append(
            AdapterWarning(
                code="unspecified_frequency",
                message=(
                    "BacktestRunConfig.freq is not set; annualized metrics use default "
                    "252-business-day scaling"
                ),
            )
        )

    validate_backtest_input_bundle(bundle)
    validate_price_inputs(config.price_df, config.close_column, config.open_column)

    try:
        import backtrader as bt  # type: ignore[import-not-found]

        engine_version = getattr(bt, "__version__", "unknown")
    except ModuleNotFoundError:
        engine_version = "unknown"

    intent = build_target_weights(bundle)
    warnings.extend(intent.warnings)

    target_matrix = _weights_to_matrix(intent.target_weights_df)
    close_matrix = _prices_to_matrix(config.price_df, config.close_column)
    fill_matrix, fill_warnings, price_source = _resolve_fill_price_matrix(
        close_matrix=close_matrix,
        config=config,
        fill_price_rule=bundle.execution_assumptions.fill_price_rule,
    )
    warnings.extend(fill_warnings)

    execution_delay = int(bundle.execution_assumptions.execution_delay_bars)
    delayed_targets = target_matrix.shift(execution_delay).fillna(0.0)

    fill_matrix = _align_price_to_weights(fill_matrix, target_matrix)
    mark_price_matrix = _align_price_to_weights(close_matrix, target_matrix)
    universe_matrix = _universe_matrix(bundle, target_matrix)
    tradability_matrix = _tradability_matrix(bundle, target_matrix)
    exclusion_reason_map = _exclusion_reason_map(bundle)

    replay = _simulate_execution(
        delayed_targets=delayed_targets,
        mark_price_matrix=mark_price_matrix,
        execution_price_matrix=fill_matrix,
        universe_matrix=universe_matrix,
        tradability_matrix=tradability_matrix,
        exclusion_reason_map=exclusion_reason_map,
        initial_cash=float(config.initial_cash),
        commission_rate=_resolve_commission_rate(bundle, config, warnings),
        slippage_rate=_resolve_slippage_rate(bundle, config, warnings),
        allow_same_day_reentry=bundle.execution_assumptions.allow_same_day_reentry,
        trade_when_not_tradable=bundle.execution_assumptions.trade_when_not_tradable,
        suspension_policy=bundle.execution_assumptions.suspension_policy,
        price_limit_policy=bundle.execution_assumptions.price_limit_policy,
    )

    summary = compute_basic_performance_summary(
        replay["returns"],
        replay["equity"],
        replay["turnover"],
    )
    engine_stats = {
        "engine_mode": "deterministic_bar_replay",
        "engine_version": engine_version,
        "ending_cash": float(replay["equity"].iloc[-1]),
        "n_orders": int(len(replay["orders_df"])),
        "n_skipped_orders": int(len(replay["skipped_orders_df"])),
    }

    adapter_run_metadata = {
        "adapter_version": BACKTEST_ADAPTER_VERSION,
        "engine": ENGINE_NAME,
        "engine_version": engine_version,
        "bundle_schema_version": bundle.schema_version,
        "artifact_path": str(bundle.artifact_path),
        "experiment_id": bundle.experiment_id,
        "dataset_fingerprint": bundle.dataset_fingerprint,
        "portfolio_construction": bundle.portfolio_construction.to_dict(),
        "execution_assumptions": bundle.execution_assumptions.to_dict(),
        "mapping_assumptions": {
            "execution_delay_bars_applied": execution_delay,
            "fill_price_source": price_source,
            "commission_rate_used": float(config.commission_bps) / 10_000.0,
            "slippage_rate_used": float(config.slippage_bps) / 10_000.0,
            "lot_size_rule_applied": bundle.execution_assumptions.lot_size_rule,
            "cash_buffer_embedded_in_target_weights": True,
            "same_day_reentry_blocked": not bundle.execution_assumptions.allow_same_day_reentry,
            "trade_when_not_tradable": bundle.execution_assumptions.trade_when_not_tradable,
            "freq": config.freq,
        },
        "warnings": [asdict(w) for w in _dedupe_warnings(warnings)],
    }

    returns_df = pd.DataFrame(
        {"date": replay["returns"].index, "portfolio_return": replay["returns"].values}
    )
    equity_df = pd.DataFrame({"date": replay["equity"].index, "equity": replay["equity"].values})
    turnover_df = pd.DataFrame(
        {"date": replay["turnover"].index, "turnover": replay["turnover"].values}
    )
    executed_df = _matrix_to_weights(replay["realized_weights"])

    result = BacktestResult(
        target_weights_df=intent.target_weights_df,
        executed_weights_df=executed_df,
        returns_df=returns_df,
        equity_curve_df=equity_df,
        turnover_df=turnover_df,
        summary=summary,
        warnings=_dedupe_warnings(warnings),
        engine_stats=engine_stats,
        adapter_run_metadata=adapter_run_metadata,
        orders_df=replay["orders_df"],
        trades_df=replay["trades_df"],
        skipped_orders_df=replay["skipped_orders_df"],
    )

    if config.output_dir is not None and config.export_summary:
        files = export_backtest_result(
            result,
            config.output_dir,
            export_target_weights=config.export_target_weights,
            export_series=config.export_series,
        )
        result = replace(result, output_files=files)
    return result


def _weights_to_matrix(target_weights_df: pd.DataFrame) -> pd.DataFrame:
    return (
        target_weights_df.pivot(index="date", columns="asset", values="target_weight")
        .sort_index()
        .sort_index(axis=1)
        .astype(float)
    )


def _matrix_to_weights(weights_matrix: pd.DataFrame) -> pd.DataFrame:
    return (
        weights_matrix.stack()
        .rename("target_weight")
        .reset_index()
        .sort_values(["date", "asset"], kind="mergesort")
        .reset_index(drop=True)
    )


def _prices_to_matrix(price_df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    panel = price_df.loc[:, ["date", "asset", value_column]].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    if panel["date"].isna().any():
        raise ValueError("price_df contains invalid dates")
    return (
        panel.pivot(index="date", columns="asset", values=value_column)
        .sort_index()
        .sort_index(axis=1)
        .astype(float)
    )


def _resolve_fill_price_matrix(
    *,
    close_matrix: pd.DataFrame,
    config: BacktestRunConfig,
    fill_price_rule: str,
) -> tuple[pd.DataFrame, list[AdapterWarning], str]:
    warnings: list[AdapterWarning] = []
    if fill_price_rule == "next_close":
        return close_matrix, warnings, "close"
    if fill_price_rule == "next_open":
        if config.open_column is None:
            warnings.append(
                AdapterWarning(
                    code="unsupported_fill_price_rule",
                    message=(
                        "fill_price_rule='next_open' requested but open_column is missing; "
                        "close prices are used as a proxy"
                    ),
                )
            )
            return close_matrix, warnings, "close_proxy_for_next_open"
        open_matrix = _prices_to_matrix(config.price_df, config.open_column)
        return open_matrix, warnings, "open"
    warnings.append(
        AdapterWarning(
            code="unsupported_fill_price_rule",
            message=(
                f"fill_price_rule={fill_price_rule!r} is not faithfully representable in "
                "Backtrader v1 adapter; close prices are used as a proxy"
            ),
        )
    )
    return close_matrix, warnings, "close_proxy_for_unsupported_fill_rule"


def _align_price_to_weights(
    price_matrix: pd.DataFrame, target_matrix: pd.DataFrame
) -> pd.DataFrame:
    return price_matrix.reindex(index=target_matrix.index, columns=target_matrix.columns)


def _universe_matrix(bundle: BacktestInputBundle, target_matrix: pd.DataFrame) -> pd.DataFrame:
    return (
        bundle.universe_mask_df.pivot(index="date", columns="asset", values="in_universe")
        .reindex(index=target_matrix.index, columns=target_matrix.columns)
        .fillna(False)
        .astype(bool)
    )


def _tradability_matrix(bundle: BacktestInputBundle, target_matrix: pd.DataFrame) -> pd.DataFrame:
    return (
        bundle.tradability_mask_df.pivot(index="date", columns="asset", values="is_tradable")
        .reindex(index=target_matrix.index, columns=target_matrix.columns)
        .fillna(False)
        .astype(bool)
    )


def _exclusion_reason_map(bundle: BacktestInputBundle) -> dict[tuple[pd.Timestamp, str], str]:
    if bundle.exclusion_reasons_df is None or bundle.exclusion_reasons_df.empty:
        return {}
    rows = bundle.exclusion_reasons_df.copy()
    rows["reason"] = rows["reason"].astype(str)
    return {
        (pd.Timestamp(row.date), str(row.asset)): str(row.reason)
        for row in rows.itertuples(index=False)
    }


def _resolve_commission_rate(
    bundle: BacktestInputBundle,
    config: BacktestRunConfig,
    warnings: list[AdapterWarning],
) -> float:
    model = bundle.execution_assumptions.commission_model
    if model in {"bps", "fixed_bps"}:
        return float(config.commission_bps) / 10_000.0
    if model == "none":
        return 0.0
    warnings.append(
        AdapterWarning(
            code="unsupported_commission_model",
            message=(
                f"commission_model={model!r} is approximated using commission_bps from "
                "BacktestRunConfig"
            ),
        )
    )
    return float(config.commission_bps) / 10_000.0


def _resolve_slippage_rate(
    bundle: BacktestInputBundle,
    config: BacktestRunConfig,
    warnings: list[AdapterWarning],
) -> float:
    model = bundle.execution_assumptions.slippage_model
    if model in {"fixed_bps", "bps"}:
        return float(config.slippage_bps) / 10_000.0
    if model == "none":
        return 0.0
    warnings.append(
        AdapterWarning(
            code="unsupported_slippage_model",
            message=(
                f"slippage_model={model!r} is approximated using slippage_bps from "
                "BacktestRunConfig"
            ),
        )
    )
    return float(config.slippage_bps) / 10_000.0


def _simulate_execution(
    *,
    delayed_targets: pd.DataFrame,
    mark_price_matrix: pd.DataFrame,
    execution_price_matrix: pd.DataFrame,
    universe_matrix: pd.DataFrame,
    tradability_matrix: pd.DataFrame,
    exclusion_reason_map: dict[tuple[pd.Timestamp, str], str],
    initial_cash: float,
    commission_rate: float,
    slippage_rate: float,
    allow_same_day_reentry: bool,
    trade_when_not_tradable: bool,
    suspension_policy: str,
    price_limit_policy: str,
) -> dict[str, object]:
    dates = delayed_targets.index
    assets = delayed_targets.columns

    realized_weights = pd.DataFrame(0.0, index=dates, columns=assets, dtype=float)
    prev = pd.Series(0.0, index=assets, dtype=float)

    orders: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []

    for date in dates:
        desired = delayed_targets.loc[date].copy()
        tradable = tradability_matrix.loc[date] & universe_matrix.loc[date]
        current = prev.copy()
        delta = desired - prev
        changed_assets = delta[delta.abs() > 1e-12].index.tolist()

        for asset in changed_assets:
            orders.append(
                {
                    "date": date,
                    "asset": asset,
                    "prev_weight": float(prev.loc[asset]),
                    "target_weight": float(desired.loc[asset]),
                    "delta_weight": float(delta.loc[asset]),
                    "is_tradable": bool(tradable.loc[asset]),
                }
            )
            if not tradable.loc[asset] and not trade_when_not_tradable:
                reason = exclusion_reason_map.get((pd.Timestamp(date), str(asset)), "not_tradable")
                skipped.append(
                    {
                        "date": date,
                        "asset": asset,
                        "reason_code": str(reason),
                        "source_reason": str(reason),
                        "policy": "skip_trade",
                    }
                )
                continue

            prev_w = float(prev.loc[asset])
            tgt_w = float(desired.loc[asset])
            if (not allow_same_day_reentry) and (prev_w * tgt_w < 0.0):
                current.loc[asset] = 0.0
                skipped.append(
                    {
                        "date": date,
                        "asset": asset,
                        "reason_code": "same_day_reentry_blocked",
                        "source_reason": "same_day_reentry_blocked",
                        "policy": "same_day_reentry_blocked",
                    }
                )
                continue

            current.loc[asset] = tgt_w
            trades.append(
                {
                    "date": date,
                    "asset": asset,
                    "from_weight": prev_w,
                    "to_weight": float(current.loc[asset]),
                    "delta_weight": float(current.loc[asset] - prev_w),
                }
            )

        realized_weights.loc[date] = current
        prev = current

    # Strategy return at date t uses positions formed at t-1, applied to close-to-close move.
    asset_returns = mark_price_matrix.pct_change().fillna(0.0)
    lagged_weights = realized_weights.shift(1).fillna(0.0)
    gross_turnover = realized_weights.diff().abs().sum(axis=1).fillna(0.0) * 0.5
    trading_cost = gross_turnover * (commission_rate + slippage_rate)
    raw_returns = (lagged_weights * asset_returns).sum(axis=1)
    returns = raw_returns - trading_cost

    equity = pd.Series(initial_cash, index=dates, dtype=float)
    for i in range(1, len(dates)):
        equity.iloc[i] = equity.iloc[i - 1] * (1.0 + float(returns.iloc[i]))

    return {
        "realized_weights": realized_weights,
        "returns": returns.astype(float),
        "equity": equity.astype(float),
        "turnover": gross_turnover.astype(float),
        "orders_df": pd.DataFrame(orders),
        "trades_df": pd.DataFrame(trades),
        "skipped_orders_df": pd.DataFrame(skipped),
    }


def _dedupe_warnings(warnings: list[AdapterWarning]) -> tuple[AdapterWarning, ...]:
    seen: set[tuple[str, str]] = set()
    out: list[AdapterWarning] = []
    for warning in warnings:
        key = (warning.code, warning.message)
        if key in seen:
            continue
        seen.add(key)
        out.append(warning)
    return tuple(out)
