from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

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

ENGINE_NAME: Literal["vectorbt"] = "vectorbt"


def run_vectorbt_backtest(
    bundle: BacktestInputBundle,
    *,
    config: BacktestRunConfig,
) -> BacktestResult:
    """Run one thin external replay using vectorbt."""

    validate_backtest_input_bundle(bundle)
    validate_price_inputs(
        config.price_df,
        close_column=config.close_column,
        open_column=config.open_column,
    )
    if config.engine != ENGINE_NAME:
        raise ValueError(f"run_vectorbt_backtest requires config.engine={ENGINE_NAME!r}")

    portfolio_intent = build_target_weights(bundle)
    warnings: list[AdapterWarning] = list(portfolio_intent.warnings)
    if config.freq is None:
        warnings.append(
            AdapterWarning(
                code="unspecified_frequency",
                message=(
                    "BacktestRunConfig.freq is not set; engine-level annualized statistics may "
                    "be incomplete or use defaults"
                ),
            )
        )

    weight_matrix = _weights_to_matrix(portfolio_intent.target_weights_df)
    price_matrix, price_source, price_warnings = _resolve_price_matrix(
        bundle,
        config,
        weight_matrix.columns,
    )
    warnings.extend(price_warnings)
    price_matrix = _align_price_to_weights(price_matrix, weight_matrix)

    execution_delay = int(bundle.execution_assumptions.execution_delay_bars)
    executed_weights = weight_matrix.shift(execution_delay).fillna(0.0)

    fee_rate, fee_warning = _map_commission(bundle=bundle, config=config)
    slippage_rate, slippage_warning = _map_slippage(bundle=bundle, config=config)
    if fee_warning is not None:
        warnings.append(fee_warning)
    if slippage_warning is not None:
        warnings.append(slippage_warning)
    warnings.extend(_execution_semantics_warnings(bundle))

    vbt = _import_vectorbt()
    portfolio, portfolio_build_mode = _build_vectorbt_portfolio(
        vbt=vbt,
        price_matrix=price_matrix,
        executed_weights=executed_weights,
        initial_cash=float(config.initial_cash),
        fee_rate=fee_rate,
        slippage_rate=slippage_rate,
        freq=config.freq,
    )

    returns = _to_series(portfolio.returns(), fallback_index=price_matrix.index)
    equity = _to_series(portfolio.value(), fallback_index=price_matrix.index)
    turnover = _turnover_from_weights(executed_weights)

    returns_df = pd.DataFrame({"date": returns.index, "portfolio_return": returns.to_numpy()})
    equity_df = pd.DataFrame({"date": equity.index, "equity": equity.to_numpy()})
    turnover_df = pd.DataFrame({"date": turnover.index, "turnover": turnover.to_numpy()})
    summary = compute_basic_performance_summary(
        returns=returns,
        equity=equity,
        turnover=turnover,
    )

    engine_stats = _portfolio_stats(portfolio)
    result = BacktestResult(
        engine=ENGINE_NAME,
        artifact_path=bundle.artifact_path,
        experiment_id=bundle.experiment_id,
        dataset_fingerprint=bundle.dataset_fingerprint,
        target_weights_df=portfolio_intent.target_weights_df.copy(),
        executed_weights_df=_matrix_to_weights(executed_weights),
        returns_df=returns_df,
        equity_curve_df=equity_df,
        turnover_df=turnover_df,
        summary=summary,
        warnings=tuple(_dedupe_warnings(warnings)),
        engine_stats=engine_stats,
        adapter_run_metadata={
            "adapter_version": BACKTEST_ADAPTER_VERSION,
            "engine": ENGINE_NAME,
            "engine_version": str(getattr(vbt, "__version__", "unknown")),
            "bundle_schema_version": bundle.schema_version,
            "artifact_path": str(bundle.artifact_path),
            "experiment_id": bundle.experiment_id,
            "dataset_fingerprint": bundle.dataset_fingerprint,
            "portfolio_construction": bundle.portfolio_construction.to_dict(),
            "execution_assumptions": bundle.execution_assumptions.to_dict(),
            "mapping_assumptions": {
                "portfolio_construction_engine_input": portfolio_build_mode,
                "execution_delay_bars_applied": execution_delay,
                "fill_price_source": price_source,
                "commission_rate_used": fee_rate,
                "slippage_rate_used": slippage_rate,
                "cash_buffer_embedded_in_target_weights": True,
                "freq": config.freq,
            },
            "warnings": [asdict(w) for w in _dedupe_warnings(warnings)],
        },
        output_files={},
    )

    if config.output_dir is not None and config.export_summary:
        output_files = export_backtest_result(
            result,
            output_dir=config.output_dir,
            export_target_weights=config.export_target_weights,
            export_series=config.export_series,
        )
        result.output_files = output_files
    return result


def _import_vectorbt() -> Any:
    try:
        import vectorbt as vbt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "vectorbt is not installed. Install it in your environment to run "
            "the vectorbt adapter path."
        ) from exc
    return vbt


def _build_vectorbt_portfolio(
    *,
    vbt: Any,
    price_matrix: pd.DataFrame,
    executed_weights: pd.DataFrame,
    initial_cash: float,
    fee_rate: float,
    slippage_rate: float,
    freq: str | None,
) -> tuple[Any, str]:
    portfolio_cls = vbt.Portfolio
    if hasattr(portfolio_cls, "from_weights"):
        return (
            portfolio_cls.from_weights(
                close=price_matrix,
                weights=executed_weights,
                init_cash=initial_cash,
                fees=fee_rate,
                slippage=slippage_rate,
                cash_sharing=True,
                freq=freq,
            ),
            "from_weights",
        )
    if hasattr(portfolio_cls, "from_orders"):
        return (
            portfolio_cls.from_orders(
                close=price_matrix,
                size=executed_weights,
                size_type="targetpercent",
                init_cash=initial_cash,
                fees=fee_rate,
                slippage=slippage_rate,
                cash_sharing=True,
                freq=freq,
            ),
            "from_orders_targetpercent",
        )
    raise ValueError("vectorbt Portfolio class lacks both from_weights and from_orders")


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


def _resolve_price_matrix(
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
                        "fill_price_rule='next_open' requested but BacktestRunConfig.open_column "
                        "is not provided; adapter falls back to close prices"
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


def _align_price_to_weights(price_matrix: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    aligned = price_matrix.reindex(index=weights.index, columns=weights.columns)
    if aligned.isna().any().any():
        missing = int(aligned.isna().sum().sum())
        raise ValueError(
            "price data does not fully cover the bundle weight grid; "
            f"missing values={missing}"
        )
    return aligned


def _map_commission(
    *,
    bundle: BacktestInputBundle,
    config: BacktestRunConfig,
) -> tuple[float, AdapterWarning | None]:
    model = bundle.execution_assumptions.commission_model
    if model == "bps":
        return float(config.commission_bps) / 10_000.0, None
    warning = AdapterWarning(
        code="unsupported_commission_model",
        message=(
            f"commission_model={model!r} cannot be represented faithfully in vectorbt v1 "
            "adapter; commission is set to 0"
        ),
    )
    return 0.0, warning


def _map_slippage(
    *,
    bundle: BacktestInputBundle,
    config: BacktestRunConfig,
) -> tuple[float, AdapterWarning | None]:
    model = bundle.execution_assumptions.slippage_model
    if model == "none":
        return 0.0, None
    if model == "fixed_bps":
        return float(config.slippage_bps) / 10_000.0, None
    warning = AdapterWarning(
        code="unsupported_slippage_model",
        message=(
            f"slippage_model={model!r} cannot be represented faithfully in vectorbt v1 "
            "adapter; fixed bps proxy from BacktestRunConfig.slippage_bps is used"
        ),
    )
    return float(config.slippage_bps) / 10_000.0, warning


def _execution_semantics_warnings(bundle: BacktestInputBundle) -> list[AdapterWarning]:
    execution = bundle.execution_assumptions
    warnings: list[AdapterWarning] = []

    if execution.lot_size_rule != "none":
        warnings.append(
            AdapterWarning(
                code="unsupported_lot_size_rule",
                message=(
                    "lot_size_rule semantics are not modeled in vectorbt v1 adapter; "
                    "weights are treated as fractional targets"
                ),
            )
        )
    if execution.partial_fill_policy != "allow_partial":
        warnings.append(
            AdapterWarning(
                code="unsupported_partial_fill_policy",
                message=(
                    "partial_fill_policy is not represented in vectorbt v1 adapter; "
                    "full target rebalancing is assumed"
                ),
            )
        )
    if execution.suspension_policy != "skip_trade":
        warnings.append(
            AdapterWarning(
                code="unsupported_suspension_policy",
                message=(
                    "suspension_policy beyond pre-trade tradability masking is not modeled "
                    "in vectorbt v1 adapter"
                ),
            )
        )
    if execution.price_limit_policy != "skip_trade":
        warnings.append(
            AdapterWarning(
                code="unsupported_price_limit_policy",
                message=(
                    "price_limit_policy beyond pre-trade tradability masking is not modeled "
                    "in vectorbt v1 adapter"
                ),
            )
        )
    if execution.allow_same_day_reentry:
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


def _turnover_from_weights(weights: pd.DataFrame) -> pd.Series:
    return (weights.diff().abs().sum(axis=1).fillna(0.0) / 2.0).astype(float)


def _to_series(obj: object, *, fallback_index: pd.Index) -> pd.Series:
    if isinstance(obj, pd.Series):
        return obj.astype(float)
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 1:
            return obj.iloc[:, 0].astype(float)
        return obj.mean(axis=1).astype(float)
    try:
        arr = np.asarray(obj, dtype=float)
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise ValueError(
            f"cannot convert object of type {type(obj).__name__} to pd.Series"
        ) from exc
    if arr.ndim != 1:
        raise ValueError(f"expected 1-D series-like object, got shape={arr.shape}")
    if len(arr) != len(fallback_index):
        raise ValueError("series-like object length does not match expected index length")
    return pd.Series(arr, index=fallback_index, dtype=float)


def _portfolio_stats(portfolio: object) -> dict[str, object] | None:
    if not hasattr(portfolio, "stats"):
        return None
    try:
        stats = portfolio.stats(silence_warnings=True)
    except TypeError:
        stats = portfolio.stats()
    if isinstance(stats, pd.Series):
        return {
            str(key): _to_jsonable(value)
            for key, value in stats.to_dict().items()
        }
    if isinstance(stats, dict):
        return {str(key): _to_jsonable(value) for key, value in stats.items()}
    return {"raw_stats_repr": repr(stats)}


def _to_jsonable(value: object) -> object:
    if isinstance(value, (float, int, str, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


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
