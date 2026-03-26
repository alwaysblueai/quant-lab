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

ENGINE_NAME = "vectorbt"


def run_vectorbt_backtest(
    bundle: BacktestInputBundle,
    config: BacktestRunConfig,
) -> BacktestResult:
    """Run one thin external replay using vectorbt."""
    if config.engine != ENGINE_NAME:
        raise ValueError(f"run_vectorbt_backtest requires config.engine={ENGINE_NAME!r}")

    warnings: list[AdapterWarning] = []
    if config.freq is None:
        warnings.append(
            AdapterWarning(
                code="unspecified_frequency",
                message=(
                    "BacktestRunConfig.freq is not set; engine-level annualized "
                    "statistics may be incomplete or use defaults"
                ),
            )
        )

    validate_backtest_input_bundle(bundle)
    validate_price_inputs(config.price_df, config.close_column, config.open_column)

    portfolio_intent = build_target_weights(bundle)
    warnings.extend(list(portfolio_intent.warnings))

    weight_matrix = _weights_to_matrix(portfolio_intent.target_weights_df)
    price_matrix, price_source, price_warnings = _resolve_price_matrix(
        bundle=bundle,
        config=config,
        weight_matrix=weight_matrix,
    )
    warnings.extend(price_warnings)
    price_matrix, alignment_warnings = _align_price_to_weights(price_matrix, weight_matrix)
    warnings.extend(alignment_warnings)

    execution_delay = int(bundle.execution_assumptions.execution_delay_bars)
    executed_weights = weight_matrix.shift(execution_delay).fillna(0.0)

    fee_rate, fee_warning = _map_commission(bundle, config)
    if fee_warning is not None:
        warnings.append(fee_warning)
    slippage_rate, slippage_warning = _map_slippage(bundle, config)
    if slippage_warning is not None:
        warnings.append(slippage_warning)
    warnings.extend(_execution_semantics_warnings(bundle))

    vbt = _import_vectorbt()
    portfolio = _build_vectorbt_portfolio(
        vbt=vbt,
        price_matrix=price_matrix,
        executed_weights=executed_weights,
        initial_cash=float(config.initial_cash),
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        freq=config.freq,
    )

    returns = _to_series(portfolio.returns(), fallback_index=weight_matrix.index)
    equity = _to_series(portfolio.value(), fallback_index=weight_matrix.index)
    turnover = _turnover_from_weights(executed_weights)

    returns_df = pd.DataFrame(
        {"date": returns.index, "portfolio_return": returns.to_numpy(dtype=float)}
    )
    equity_curve_df = pd.DataFrame(
        {"date": equity.index, "equity": equity.to_numpy(dtype=float)}
    )
    turnover_df = pd.DataFrame(
        {"date": turnover.index, "turnover": turnover.to_numpy(dtype=float)}
    )
    executed_weights_df = _matrix_to_weights(executed_weights)

    summary = compute_basic_performance_summary(returns, equity, turnover)
    engine_stats = _portfolio_stats(portfolio)
    engine_version = getattr(vbt, "__version__", "unknown")

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
            "portfolio_construction_engine_input": "target_weights",
            "execution_delay_bars_applied": execution_delay,
            "fill_price_source": price_source,
            "commission_rate_used": fee_rate,
            "slippage_rate_used": slippage_rate,
            "cash_buffer_embedded_in_target_weights": True,
            "freq": config.freq,
        },
        "warnings": [asdict(w) for w in _dedupe_warnings(warnings)],
    }

    result = BacktestResult(
        target_weights_df=portfolio_intent.target_weights_df,
        executed_weights_df=executed_weights_df,
        returns_df=returns_df,
        equity_curve_df=equity_curve_df,
        turnover_df=turnover_df,
        summary=summary,
        warnings=_dedupe_warnings(warnings),
        engine_stats=engine_stats,
        adapter_run_metadata=adapter_run_metadata,
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


def _import_vectorbt():
    try:
        import vectorbt as vbt  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "vectorbt is not installed. Install it in your environment to run "
            "the vectorbt adapter path."
        ) from exc
    return vbt


def _build_vectorbt_portfolio(
    *,
    vbt,
    price_matrix: pd.DataFrame,
    executed_weights: pd.DataFrame,
    initial_cash: float,
    fee_rate: float,
    slippage_rate: float,
    freq: str | None,
):
    portfolio_cls = vbt.Portfolio
    if hasattr(portfolio_cls, "from_weights"):
        return portfolio_cls.from_weights(
            close=price_matrix,
            weights=executed_weights,
            init_cash=initial_cash,
            fees=fee_rate,
            slippage=slippage_rate,
            cash_sharing=True,
            freq=freq,
        )
    if hasattr(portfolio_cls, "from_orders"):
        return portfolio_cls.from_orders(
            close=price_matrix,
            size=executed_weights,
            size_type="targetpercent",
            init_cash=initial_cash,
            fees=fee_rate,
            slippage=slippage_rate,
            cash_sharing=True,
            freq=freq,
        )
    raise ValueError("vectorbt Portfolio class lacks both from_weights and from_orders")


def _weights_to_matrix(target_weights_df: pd.DataFrame) -> pd.DataFrame:
    weight_matrix = (
        target_weights_df.pivot(index="date", columns="asset", values="target_weight")
        .sort_index()
        .sort_index(axis=1)
        .astype(float)
    )
    return weight_matrix


def _matrix_to_weights(weights_matrix: pd.DataFrame) -> pd.DataFrame:
    long_df = (
        weights_matrix.stack()
        .rename("target_weight")
        .reset_index()
        .sort_values(["date", "asset"], kind="mergesort")
        .reset_index(drop=True)
    )
    return long_df


def _resolve_price_matrix(
    *,
    bundle: BacktestInputBundle,
    config: BacktestRunConfig,
    weight_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, str, list[AdapterWarning]]:
    warnings: list[AdapterWarning] = []
    fill_price_rule = bundle.execution_assumptions.fill_price_rule
    close_matrix = _prices_to_matrix(config.price_df, config.close_column)
    if fill_price_rule == "next_close":
        return close_matrix, "close", warnings
    if fill_price_rule == "next_open":
        if config.open_column is None:
            warnings.append(
                AdapterWarning(
                    code="unsupported_fill_price_rule",
                    message=(
                        "fill_price_rule='next_open' requested but BacktestRunConfig.open_column "
                        "is not provided; adapter falls back to close prices"
                    ),
                )
            )
            return close_matrix, "close_proxy_for_next_open", warnings
        return _prices_to_matrix(config.price_df, config.open_column), "open", warnings
    if fill_price_rule == "vwap_next_bar":
        warnings.append(
            AdapterWarning(
                code="unsupported_fill_price_rule",
                message=(
                    "fill_price_rule='vwap_next_bar' cannot be represented faithfully in "
                    "vectorbt v1 adapter; close prices are used as an approximation"
                ),
            )
        )
        return close_matrix, "close_proxy_for_vwap", warnings
    warnings.append(
        AdapterWarning(
            code="unsupported_fill_price_rule",
            message=(
                f"fill_price_rule={fill_price_rule!r} is unsupported by vectorbt v1 adapter; "
                "close prices are used as fallback"
            ),
        )
    )
    return close_matrix, "close_proxy_for_unknown_fill_rule", warnings


def _prices_to_matrix(price_df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    panel = price_df.loc[:, ["date", "asset", value_column]].copy()
    panel["date"] = pd.to_datetime(panel["date"], errors="coerce")
    if panel["date"].isna().any():
        raise ValueError("price_df contains invalid dates")
    matrix = (
        panel.pivot(index="date", columns="asset", values=value_column)
        .sort_index()
        .sort_index(axis=1)
        .astype(float)
    )
    return matrix


def _align_price_to_weights(
    price_matrix: pd.DataFrame,
    weight_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, list[AdapterWarning]]:
    aligned = price_matrix.reindex(index=weight_matrix.index, columns=weight_matrix.columns)
    missing = int(aligned.isna().sum().sum())
    if missing > 0:
        return aligned, [
            AdapterWarning(
                code="price_alignment_missing_values",
                message=(
                    "price data does not fully cover the bundle weight grid; "
                    f"missing values={missing}"
                ),
            )
        ]
    return aligned, []


def _map_commission(
    bundle: BacktestInputBundle,
    config: BacktestRunConfig,
) -> tuple[float, AdapterWarning | None]:
    model = bundle.execution_assumptions.commission_model
    if model in {"bps", "fixed_bps"}:
        return float(config.commission_bps) / 10_000.0, None
    if model == "none":
        return 0.0, None
    return (
        0.0,
        AdapterWarning(
            code="unsupported_commission_model",
            message=(
                f"commission_model={model!r} cannot be represented faithfully in vectorbt "
                "v1 adapter; commission is set to 0"
            ),
        ),
    )


def _map_slippage(
    bundle: BacktestInputBundle,
    config: BacktestRunConfig,
) -> tuple[float, AdapterWarning | None]:
    model = bundle.execution_assumptions.slippage_model
    if model in {"fixed_bps", "bps"}:
        return float(config.slippage_bps) / 10_000.0, None
    if model == "none":
        return 0.0, None
    return (
        float(config.slippage_bps) / 10_000.0,
        AdapterWarning(
            code="unsupported_slippage_model",
            message=(
                f"slippage_model={model!r} cannot be represented faithfully in vectorbt v1 "
                "adapter; fixed bps proxy from BacktestRunConfig.slippage_bps is used"
            ),
        ),
    )


def _execution_semantics_warnings(bundle: BacktestInputBundle) -> list[AdapterWarning]:
    warnings: list[AdapterWarning] = []
    assumptions = bundle.execution_assumptions
    if assumptions.lot_size_rule != "none":
        warnings.append(
            AdapterWarning(
                code="unsupported_lot_size_rule",
                message=(
                    "lot_size_rule semantics are not modeled in vectorbt v1 adapter; "
                    "weights are treated as fractional targets"
                ),
            )
        )
    if assumptions.partial_fill_policy != "allow_partial":
        warnings.append(
            AdapterWarning(
                code="unsupported_partial_fill_policy",
                message=(
                    "partial_fill_policy is not represented in vectorbt v1 adapter; "
                    "full target rebalancing is assumed"
                ),
            )
        )
    if assumptions.suspension_policy != "skip_trade":
        warnings.append(
            AdapterWarning(
                code="unsupported_suspension_policy",
                message=(
                    "suspension_policy beyond pre-trade tradability masking is not modeled "
                    "in vectorbt v1 adapter"
                ),
            )
        )
    if assumptions.price_limit_policy != "skip_trade":
        warnings.append(
            AdapterWarning(
                code="unsupported_price_limit_policy",
                message=(
                    "price_limit_policy beyond pre-trade tradability masking is not modeled "
                    "in vectorbt v1 adapter"
                ),
            )
        )
    if assumptions.allow_same_day_reentry:
        warnings.append(
            AdapterWarning(
                code="unsupported_same_day_reentry",
                message=(
                    "allow_same_day_reentry is not represented in vectorbt v1 adapter and "
                    "may diverge from strict engine behavior"
                ),
            )
        )
    return warnings


def _to_series(value, fallback_index: pd.Index) -> pd.Series:
    if isinstance(value, pd.Series):
        out = value.copy()
    else:
        out = pd.Series(value, index=fallback_index)
    out.index = pd.to_datetime(out.index, errors="coerce")
    return out.sort_index()


def _turnover_from_weights(weight_matrix: pd.DataFrame) -> pd.Series:
    diff = weight_matrix.diff().abs().sum(axis=1) * 0.5
    return diff.fillna(0.0)


def _portfolio_stats(portfolio) -> dict[str, object] | None:
    if hasattr(portfolio, "stats"):
        stats = portfolio.stats()
        if isinstance(stats, pd.Series):
            return {str(k): _safe_scalar(v) for k, v in stats.to_dict().items()}
    return None


def _safe_scalar(value):
    if pd.isna(value):
        return None
    if isinstance(value, (float, int, bool, str)):
        return value
    return str(value)


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
