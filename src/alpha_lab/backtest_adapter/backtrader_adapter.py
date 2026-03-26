from __future__ import annotations

from dataclasses import asdict
from typing import Any, Literal, Protocol, TypedDict

import numpy as np
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

ENGINE_NAME: Literal["backtrader"] = "backtrader"


class _ExecutionPolicies(Protocol):
    @property
    def suspension_policy(self) -> str: ...

    @property
    def price_limit_policy(self) -> str: ...


class _ReplayOutputs(TypedDict):
    equity: pd.Series
    returns: pd.Series
    turnover: pd.Series
    realized_weights: pd.DataFrame
    orders_df: pd.DataFrame
    trades_df: pd.DataFrame
    skipped_orders_df: pd.DataFrame
    ending_cash: float


def run_backtrader_backtest(
    bundle: BacktestInputBundle,
    *,
    config: BacktestRunConfig,
) -> BacktestResult:
    """Run a stricter execution-aware replay path under the Backtrader adapter."""

    validate_backtest_input_bundle(bundle)
    validate_price_inputs(
        config.price_df,
        close_column=config.close_column,
        open_column=config.open_column,
    )
    if config.engine != ENGINE_NAME:
        raise ValueError(f"run_backtrader_backtest requires config.engine={ENGINE_NAME!r}")

    bt = _import_backtrader()
    warnings: list[AdapterWarning] = []
    warnings.append(
        AdapterWarning(
            code="simplified_fill_model",
            message=(
                "Backtrader adapter v1 uses a deterministic bar-based fill model derived "
                "from bundle contracts; it is stricter than vectorbt v1 but not a full "
                "microstructure simulation"
            ),
        )
    )
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

    intent = build_target_weights(bundle)
    warnings.extend(intent.warnings)
    target_matrix = _weights_to_matrix(intent.target_weights_df)

    mark_price_matrix = _prices_to_matrix(
        config.price_df,
        value_column=config.close_column,
        asset_columns=target_matrix.columns,
    )
    execution_price_matrix, fill_price_source, fill_warnings = _resolve_fill_price_matrix(
        bundle,
        config,
        target_matrix.columns,
    )
    warnings.extend(fill_warnings)

    mark_price_matrix = _align_price_to_weights(mark_price_matrix, target_matrix)
    execution_price_matrix = _align_price_to_weights(execution_price_matrix, target_matrix)

    execution_delay = int(bundle.execution_assumptions.execution_delay_bars)
    delayed_targets = target_matrix.shift(execution_delay).fillna(0.0)
    universe_matrix = _universe_matrix(
        bundle=bundle,
        index=delayed_targets.index,
        columns=delayed_targets.columns,
    )
    tradability_matrix = _tradability_matrix(
        bundle=bundle,
        index=delayed_targets.index,
        columns=delayed_targets.columns,
    )
    exclusion_reason_map = _exclusion_reason_map(bundle.exclusion_reasons_df)

    commission_rate, commission_warning = _resolve_commission_rate(bundle, config=config)
    slippage_rate, slippage_warning = _resolve_slippage_rate(bundle, config=config)
    if commission_warning is not None:
        warnings.append(commission_warning)
    if slippage_warning is not None:
        warnings.append(slippage_warning)
    warnings.extend(_policy_warnings(bundle))

    replay = _simulate_execution(
        delayed_targets=delayed_targets,
        mark_price_matrix=mark_price_matrix,
        execution_price_matrix=execution_price_matrix,
        universe_matrix=universe_matrix,
        tradability_matrix=tradability_matrix,
        exclusion_reason_map=exclusion_reason_map,
        bundle=bundle,
        initial_cash=float(config.initial_cash),
        commission_rate=commission_rate,
        slippage_rate=slippage_rate,
    )

    summary = compute_basic_performance_summary(
        returns=replay["returns"],
        equity=replay["equity"],
        turnover=replay["turnover"],
    )
    engine_stats = {
        "engine_mode": "backtrader_style_replay",
        "engine_version": str(getattr(bt, "__version__", "unknown")),
        "n_orders": int(len(replay["orders_df"])),
        "n_skipped_orders": int(len(replay["skipped_orders_df"])),
        "ending_cash": float(replay["ending_cash"]),
    }

    warning_rows = _dedupe_warnings(warnings)
    result = BacktestResult(
        engine=ENGINE_NAME,
        artifact_path=bundle.artifact_path,
        experiment_id=bundle.experiment_id,
        dataset_fingerprint=bundle.dataset_fingerprint,
        target_weights_df=intent.target_weights_df.copy(),
        executed_weights_df=_matrix_to_weights(replay["realized_weights"]),
        returns_df=pd.DataFrame(
            {
                "date": replay["returns"].index,
                "portfolio_return": replay["returns"].to_numpy(),
            }
        ),
        equity_curve_df=pd.DataFrame(
            {
                "date": replay["equity"].index,
                "equity": replay["equity"].to_numpy(),
            }
        ),
        turnover_df=pd.DataFrame(
            {
                "date": replay["turnover"].index,
                "turnover": replay["turnover"].to_numpy(),
            }
        ),
        summary=summary,
        warnings=tuple(warning_rows),
        engine_stats=engine_stats,
        adapter_run_metadata={
            "adapter_version": BACKTEST_ADAPTER_VERSION,
            "engine": ENGINE_NAME,
            "engine_version": str(getattr(bt, "__version__", "unknown")),
            "bundle_schema_version": bundle.schema_version,
            "artifact_path": str(bundle.artifact_path),
            "experiment_id": bundle.experiment_id,
            "dataset_fingerprint": bundle.dataset_fingerprint,
            "portfolio_construction": bundle.portfolio_construction.to_dict(),
            "execution_assumptions": bundle.execution_assumptions.to_dict(),
            "mapping_assumptions": {
                "execution_delay_bars_applied": execution_delay,
                "fill_price_source": fill_price_source,
                "commission_rate_used": commission_rate,
                "slippage_rate_used": slippage_rate,
                "lot_size_rule_applied": bundle.execution_assumptions.lot_size_rule,
                "cash_buffer_embedded_in_target_weights": True,
                "same_day_reentry_blocked": not bundle.execution_assumptions.allow_same_day_reentry,
                "trade_when_not_tradable": bundle.execution_assumptions.trade_when_not_tradable,
                "freq": config.freq,
            },
            "warnings": [asdict(w) for w in warning_rows],
        },
        orders_df=replay["orders_df"],
        trades_df=replay["trades_df"],
        skipped_orders_df=replay["skipped_orders_df"],
        output_files={},
    )

    if config.output_dir is not None and config.export_summary:
        result.output_files = export_backtest_result(
            result,
            output_dir=config.output_dir,
            export_target_weights=config.export_target_weights,
            export_series=config.export_series,
        )
    return result


def _import_backtrader() -> Any:
    try:
        import backtrader as bt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "backtrader is not installed. Install it in your environment to run "
            "the Backtrader adapter path."
        ) from exc
    return bt


def _resolve_fill_price_matrix(
    bundle: BacktestInputBundle,
    config: BacktestRunConfig,
    asset_columns: pd.Index,
) -> tuple[pd.DataFrame, str, list[AdapterWarning]]:
    warnings: list[AdapterWarning] = []
    close_matrix = _prices_to_matrix(
        config.price_df,
        value_column=config.close_column,
        asset_columns=asset_columns,
    )
    fill_price_rule = bundle.execution_assumptions.fill_price_rule
    if fill_price_rule == "next_close":
        return close_matrix, "close", warnings
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
            return close_matrix, "close_proxy_for_next_open", warnings
        return (
            _prices_to_matrix(
                config.price_df,
                value_column=config.open_column,
                asset_columns=asset_columns,
            ),
            "open",
            warnings,
        )
    warnings.append(
        AdapterWarning(
            code="unsupported_fill_price_rule",
            message=(
                f"fill_price_rule={fill_price_rule!r} is not faithfully representable in "
                "Backtrader v1 adapter; close prices are used as a proxy"
            ),
        )
    )
    return close_matrix, "close_proxy_for_unsupported_fill_rule", warnings


def _resolve_commission_rate(
    bundle: BacktestInputBundle,
    *,
    config: BacktestRunConfig,
) -> tuple[float, AdapterWarning | None]:
    model = bundle.execution_assumptions.commission_model
    if model == "bps":
        return float(config.commission_bps) / 10_000.0, None
    return (
        float(config.commission_bps) / 10_000.0,
        AdapterWarning(
            code="unsupported_commission_model",
            message=(
                f"commission_model={model!r} is approximated using commission_bps from "
                "BacktestRunConfig"
            ),
        ),
    )


def _resolve_slippage_rate(
    bundle: BacktestInputBundle,
    *,
    config: BacktestRunConfig,
) -> tuple[float, AdapterWarning | None]:
    model = bundle.execution_assumptions.slippage_model
    if model == "none":
        return 0.0, None
    if model == "fixed_bps":
        return float(config.slippage_bps) / 10_000.0, None
    return (
        float(config.slippage_bps) / 10_000.0,
        AdapterWarning(
            code="unsupported_slippage_model",
            message=(
                f"slippage_model={model!r} is approximated using slippage_bps from "
                "BacktestRunConfig"
            ),
        ),
    )


def _policy_warnings(bundle: BacktestInputBundle) -> list[AdapterWarning]:
    warnings: list[AdapterWarning] = []
    execution = bundle.execution_assumptions
    if execution.partial_fill_policy != "allow_partial":
        warnings.append(
            AdapterWarning(
                code="unsupported_partial_fill_policy",
                message=(
                    "partial_fill_policy is not explicitly modeled in Backtrader v1 adapter; "
                    "full fill / skip behavior is applied"
                ),
            )
        )
    if execution.suspension_policy == "defer_trade":
        warnings.append(
            AdapterWarning(
                code="defer_policy_approximated_as_skip",
                message=(
                    "suspension_policy='defer_trade' is approximated as skip in Backtrader "
                    "v1 adapter"
                ),
            )
        )
    if execution.price_limit_policy == "defer_trade":
        warnings.append(
            AdapterWarning(
                code="defer_policy_approximated_as_skip",
                message=(
                    "price_limit_policy='defer_trade' is approximated as skip in Backtrader "
                    "v1 adapter"
                ),
            )
        )
    return warnings


def _simulate_execution(
    *,
    delayed_targets: pd.DataFrame,
    mark_price_matrix: pd.DataFrame,
    execution_price_matrix: pd.DataFrame,
    universe_matrix: pd.DataFrame,
    tradability_matrix: pd.DataFrame,
    exclusion_reason_map: dict[tuple[pd.Timestamp, str], str],
    bundle: BacktestInputBundle,
    initial_cash: float,
    commission_rate: float,
    slippage_rate: float,
) -> _ReplayOutputs:
    assets = delayed_targets.columns.tolist()
    positions = {asset: 0.0 for asset in assets}
    cash = float(initial_cash)
    equity_rows: list[dict[str, object]] = []
    realized_weight_rows: list[dict[str, object]] = []
    order_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []
    execution = bundle.execution_assumptions
    for date in delayed_targets.index:
        mark_prices = mark_price_matrix.loc[date]
        fill_prices = execution_price_matrix.loc[date]
        equity_pre = cash + _positions_market_value(positions, mark_prices)
        if not np.isfinite(equity_pre):
            raise ValueError(f"non-finite equity encountered on {date}")

        date_targets = delayed_targets.loc[date]
        date_universe = universe_matrix.loc[date]
        date_tradability = tradability_matrix.loc[date]
        for asset in sorted(assets):
            target_weight = float(date_targets[asset])
            current_size = float(positions[asset])
            in_universe = bool(date_universe[asset])
            tradable = bool(date_tradability[asset])
            reason = exclusion_reason_map.get((pd.Timestamp(date), str(asset)))
            reason_kind = _classify_exclusion_reason(reason)
            policy = _resolve_non_tradable_policy(reason_kind=reason_kind, execution=execution)

            if not tradable and not execution.trade_when_not_tradable:
                # Do not count a skipped order when the portfolio is already flat and
                # target weight is also zero (no actionable order was intended).
                needs_rebalance = (abs(current_size) > 1e-12) or (abs(target_weight) > 1e-12)
                if not needs_rebalance:
                    continue
                if policy == "error":
                    raise ValueError(
                        "encountered non-tradable asset with execution policy 'error': "
                        f"date={date}, asset={asset}, reason={reason!r}"
                    )
                source_reason = (
                    reason
                    if reason is not None
                    else _fallback_non_tradable_reason(in_universe=in_universe)
                )
                reason_code = _normalize_reason_code(source_reason)
                skipped_rows.append(
                    {
                        "date": date,
                        "asset": asset,
                        "requested_weight": target_weight,
                        "policy": policy,
                        "reason": reason_code,
                        "reason_code": reason_code,
                        "source_reason": source_reason,
                        "action": "skip",
                    }
                )
                continue

            fill_price = float(fill_prices[asset])
            mark_price = float(mark_prices[asset])
            if not np.isfinite(fill_price) or fill_price <= 0.0:
                skipped_rows.append(
                    {
                        "date": date,
                        "asset": asset,
                        "requested_weight": target_weight,
                        "policy": "invalid_price",
                        "reason": "invalid_price",
                        "reason_code": "invalid_price",
                        "source_reason": "non_positive_or_nan_fill_price",
                        "action": "skip",
                    }
                )
                continue

            target_value = target_weight * equity_pre
            raw_target_size = target_value / fill_price
            target_size = _round_target_size(
                raw_target_size,
                lot_size_rule=execution.lot_size_rule,
                lot_size=execution.lot_size,
            )

            if (not execution.allow_same_day_reentry) and current_size * target_size < 0.0:
                skipped_rows.append(
                    {
                        "date": date,
                        "asset": asset,
                        "requested_weight": target_weight,
                        "policy": "same_day_reentry_blocked",
                        "reason": "same_day_reentry_blocked",
                        "reason_code": "same_day_reentry_blocked",
                        "source_reason": "position_sign_flip",
                        "action": "close_only",
                    }
                )
                target_size = 0.0

            delta_size = target_size - current_size
            if abs(delta_size) < 1e-12:
                continue

            signed_notional = delta_size * fill_price
            cost = abs(signed_notional) * (commission_rate + slippage_rate)
            cash -= signed_notional + cost
            positions[asset] = target_size

            order_rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "side": "buy" if delta_size > 0 else "sell",
                    "size": float(delta_size),
                    "price": fill_price,
                    "notional": signed_notional,
                    "commission_and_slippage": cost,
                    "target_weight": target_weight,
                    "target_size": target_size,
                }
            )

        equity_post = cash + _positions_market_value(positions, mark_prices)
        equity_rows.append({"date": date, "equity": float(equity_post), "cash": float(cash)})
        for asset in sorted(assets):
            mark_price = float(mark_prices[asset])
            asset_value = positions[asset] * mark_price
            weight = 0.0 if abs(equity_post) < 1e-12 else asset_value / equity_post
            realized_weight_rows.append(
                {
                    "date": date,
                    "asset": asset,
                    "target_weight": float(weight),
                }
            )

    equity_df = pd.DataFrame(equity_rows).sort_values("date").reset_index(drop=True)
    equity_series = pd.Series(
        equity_df["equity"].to_numpy(dtype=float),
        index=pd.to_datetime(equity_df["date"]),
        dtype=float,
    )
    returns_series = equity_series.pct_change().fillna(0.0).astype(float)
    realized_weight_matrix = (
        pd.DataFrame(realized_weight_rows)
        .pivot(index="date", columns="asset", values="target_weight")
        .sort_index()
        .sort_index(axis=1)
        .fillna(0.0)
    )
    turnover = (realized_weight_matrix.diff().abs().sum(axis=1).fillna(0.0) / 2.0).astype(float)

    if order_rows:
        orders_df = pd.DataFrame(order_rows).sort_values(["date", "asset"]).reset_index(drop=True)
    else:
        orders_df = pd.DataFrame(
            columns=[
                "date",
                "asset",
                "side",
                "size",
                "price",
                "notional",
                "commission_and_slippage",
                "target_weight",
                "target_size",
            ]
        )
    if skipped_rows:
        skipped_df = (
            pd.DataFrame(skipped_rows).sort_values(["date", "asset"]).reset_index(drop=True)
        )
    else:
        skipped_df = pd.DataFrame(
            columns=[
                "date",
                "asset",
                "requested_weight",
                "policy",
                "reason",
                "reason_code",
                "source_reason",
                "action",
            ]
        )

    trades_df = orders_df.copy()
    if not trades_df.empty:
        trades_df = trades_df.rename(columns={"size": "executed_size", "price": "executed_price"})

    return {
        "equity": equity_series,
        "returns": returns_series,
        "turnover": turnover,
        "realized_weights": realized_weight_matrix,
        "orders_df": orders_df,
        "trades_df": trades_df,
        "skipped_orders_df": skipped_df,
        "ending_cash": float(cash),
    }


def _resolve_non_tradable_policy(
    *,
    reason_kind: str,
    execution: _ExecutionPolicies,
) -> str:
    if reason_kind == "price_limit":
        return str(execution.price_limit_policy)
    return str(execution.suspension_policy)


def _classify_exclusion_reason(reason: str | None) -> str:
    if reason is None:
        return "unknown"
    raw = reason.strip().lower()
    if "limit" in raw:
        return "price_limit"
    if "suspend" in raw or "halt" in raw:
        return "suspension"
    return "unknown"


def _fallback_non_tradable_reason(*, in_universe: bool) -> str:
    if not in_universe:
        return "universe_excluded"
    return "tradability_excluded"


def _normalize_reason_code(reason: str) -> str:
    mapping = {
        "listing_age_or_missing_listing_date": "ipo_age_filter",
        "st_filter": "st_filter",
        "halted_trading": "halted_trading",
        "suspension": "suspension",
        "limit_locked_non_executable": "price_limit_locked",
        "price_limit_lock": "price_limit_locked",
        "min_adv_filter": "min_adv_filter",
        "manual_exclusion": "manual_exclusion",
        "position_sign_flip": "position_sign_flip",
        "same_day_reentry_blocked": "same_day_reentry_blocked",
        "non_positive_or_nan_fill_price": "invalid_price",
        "universe_excluded": "universe_excluded",
        "tradability_excluded": "tradability_excluded",
    }
    tokens = [item.strip().lower() for item in str(reason).split(";") if item.strip()]
    if not tokens:
        return "unclassified"
    normalized = [mapping.get(token, token) for token in tokens]
    return ";".join(sorted(set(normalized)))


def _round_target_size(
    raw_target_size: float,
    *,
    lot_size_rule: str,
    lot_size: int | None,
) -> float:
    if lot_size_rule == "none":
        return float(raw_target_size)
    if lot_size is None or lot_size <= 0:
        raise ValueError("lot_size must be > 0 when lot_size_rule is enabled")
    sign = 1.0 if raw_target_size >= 0.0 else -1.0
    rounded_abs = np.floor(abs(raw_target_size) / float(lot_size)) * float(lot_size)
    return sign * float(rounded_abs)


def _positions_market_value(positions: dict[str, float], mark_prices: pd.Series) -> float:
    total = 0.0
    for asset, size in positions.items():
        price = float(mark_prices[asset])
        if not np.isfinite(price):
            continue
        total += size * price
    return float(total)


def _tradability_matrix(
    *,
    bundle: BacktestInputBundle,
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    table = bundle.tradability_mask_df.copy()
    table["date"] = pd.to_datetime(table["date"], errors="coerce")
    wide = (
        table.pivot(index="date", columns="asset", values="is_tradable")
        .sort_index()
        .sort_index(axis=1)
    )
    wide = wide.reindex(index=index, columns=columns)
    wide = wide.astype("boolean").fillna(False).astype(bool)
    return wide


def _universe_matrix(
    *,
    bundle: BacktestInputBundle,
    index: pd.Index,
    columns: pd.Index,
) -> pd.DataFrame:
    table = bundle.universe_mask_df.copy()
    table["date"] = pd.to_datetime(table["date"], errors="coerce")
    wide = (
        table.pivot(index="date", columns="asset", values="in_universe")
        .sort_index()
        .sort_index(axis=1)
    )
    wide = wide.reindex(index=index, columns=columns)
    wide = wide.astype("boolean").fillna(False).astype(bool)
    return wide


def _exclusion_reason_map(
    exclusion_reasons_df: pd.DataFrame | None,
) -> dict[tuple[pd.Timestamp, str], str]:
    if exclusion_reasons_df is None or exclusion_reasons_df.empty:
        return {}
    table = exclusion_reasons_df.copy()
    table["date"] = pd.to_datetime(table["date"], errors="coerce")
    grouped = (
        table.groupby(["date", "asset"], sort=True)["reason"]
        .apply(lambda s: ";".join(sorted({str(v) for v in s if str(v).strip()})))
        .reset_index()
    )
    out: dict[tuple[pd.Timestamp, str], str] = {}
    for row in grouped.itertuples(index=False):
        out[(pd.Timestamp(row.date), str(row.asset))] = str(row.reason)
    return out


def _prices_to_matrix(
    price_df: pd.DataFrame,
    *,
    value_column: str,
    asset_columns: pd.Index,
) -> pd.DataFrame:
    panel = price_df[["date", "asset", value_column]].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    if panel["date"].isna().any():
        raise ValueError("price_df contains invalid dates")
    pivot = panel.pivot(index="date", columns="asset", values=value_column)
    pivot = pivot.sort_index().sort_index(axis=1)
    pivot = pivot.reindex(columns=asset_columns)
    return pivot.astype(float)


def _weights_to_matrix(target_weights_df: pd.DataFrame) -> pd.DataFrame:
    matrix = target_weights_df.pivot(
        index="date",
        columns="asset",
        values="target_weight",
    )
    matrix = matrix.sort_index().sort_index(axis=1).fillna(0.0)
    return matrix.astype(float)


def _matrix_to_weights(weights_matrix: pd.DataFrame) -> pd.DataFrame:
    long_df = (
        weights_matrix.stack()
        .rename("target_weight")
        .reset_index()
        .rename(columns={"level_0": "date", "level_1": "asset"})
    )
    return long_df.sort_values(["date", "asset"]).reset_index(drop=True)


def _align_price_to_weights(price_matrix: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    aligned = price_matrix.reindex(index=weights.index, columns=weights.columns)
    if aligned.isna().any().any():
        missing = int(aligned.isna().sum().sum())
        raise ValueError(
            "price data does not fully cover the bundle weight grid; "
            f"missing values={missing}"
        )
    return aligned


def _dedupe_warnings(warnings: list[AdapterWarning]) -> list[AdapterWarning]:
    seen: set[tuple[str, str]] = set()
    out: list[AdapterWarning] = []
    for warning in warnings:
        key = (warning.code, warning.message)
        if key in seen:
            continue
        seen.add(key)
        out.append(AdapterWarning(**asdict(warning)))
    return out
