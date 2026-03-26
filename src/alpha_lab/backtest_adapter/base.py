from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from alpha_lab.backtest_adapter.schema import (
    BacktestInputBundle,
    BacktestResult,
    BacktestRunConfig,
)


class ExternalBacktestAdapter(Protocol):
    """Engine adapter protocol for future pluggable adapters."""

    engine_name: str

    def run(self, bundle: BacktestInputBundle, config: BacktestRunConfig) -> BacktestResult:
        """Run one replay and return a normalized backtest result."""


def run_external_backtest(
    bundle: BacktestInputBundle,
    *,
    config: BacktestRunConfig,
) -> BacktestResult:
    """Dispatch to the requested engine adapter."""

    if config.engine == "vectorbt":
        from alpha_lab.backtest_adapter.vectorbt_adapter import run_vectorbt_backtest

        return run_vectorbt_backtest(bundle, config=config)
    if config.engine == "backtrader":
        from alpha_lab.backtest_adapter.backtrader_adapter import run_backtrader_backtest

        return run_backtrader_backtest(bundle, config=config)
    raise ValueError(f"unsupported engine {config.engine!r}")


def compute_basic_performance_summary(
    *,
    returns: pd.Series,
    equity: pd.Series,
    turnover: pd.Series,
) -> dict[str, object]:
    """Compute engine-agnostic, deterministic summary metrics."""

    clean_returns = returns.dropna()
    if clean_returns.empty:
        mean_return = float("nan")
        vol = float("nan")
        sharpe = float("nan")
    else:
        mean_return = float(clean_returns.mean())
        vol = float(clean_returns.std(ddof=1))
        sharpe = float(np.sqrt(252.0) * mean_return / vol) if vol > 0.0 else float("nan")

    if equity.empty:
        total_return = float("nan")
        max_drawdown = float("nan")
    else:
        start = float(equity.iloc[0])
        end = float(equity.iloc[-1])
        total_return = float(end / start - 1.0) if start != 0.0 else float("nan")
        running_max = equity.cummax()
        drawdown = (equity / running_max) - 1.0
        max_drawdown = float(drawdown.min()) if not drawdown.empty else float("nan")

    return {
        "n_periods": int(len(returns)),
        "mean_return": mean_return,
        "volatility": vol,
        "sharpe_annualized": sharpe,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "mean_turnover": float(turnover.mean()) if not turnover.empty else float("nan"),
    }


def export_backtest_result(
    result: BacktestResult,
    *,
    output_dir: str | Path,
    export_target_weights: bool,
    export_series: bool,
) -> dict[str, Path]:
    """Write a compact set of replay artifacts for inspection/audit."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}

    summary_path = out_dir / "backtest_summary.json"
    summary_payload = {
        "engine": result.engine,
        "artifact_path": str(result.artifact_path),
        "experiment_id": result.experiment_id,
        "dataset_fingerprint": result.dataset_fingerprint,
        "summary": result.summary,
        "warnings": [asdict(w) for w in result.warnings],
        "engine_stats": result.engine_stats,
        "adapter_run_metadata": result.adapter_run_metadata,
    }
    summary_path.write_text(
        json.dumps(summary_payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    files["summary"] = summary_path

    if result.adapter_run_metadata is not None:
        metadata_path = out_dir / "adapter_run_metadata.json"
        metadata_path.write_text(
            json.dumps(result.adapter_run_metadata, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        files["adapter_run_metadata"] = metadata_path

    if export_series:
        returns_path = out_dir / "portfolio_returns.csv"
        equity_path = out_dir / "equity_curve.csv"
        turnover_path = out_dir / "turnover.csv"
        result.returns_df.to_csv(returns_path, index=False)
        result.equity_curve_df.to_csv(equity_path, index=False)
        result.turnover_df.to_csv(turnover_path, index=False)
        files["returns"] = returns_path
        files["equity_curve"] = equity_path
        files["turnover"] = turnover_path

    if export_target_weights:
        target_path = out_dir / "target_weights.csv"
        executed_path = out_dir / "executed_weights.csv"
        result.target_weights_df.to_csv(target_path, index=False)
        result.executed_weights_df.to_csv(executed_path, index=False)
        files["target_weights"] = target_path
        files["executed_weights"] = executed_path

    if result.orders_df is not None:
        orders_path = out_dir / "orders.csv"
        result.orders_df.to_csv(orders_path, index=False)
        files["orders"] = orders_path
    if result.trades_df is not None:
        trades_path = out_dir / "trades.csv"
        result.trades_df.to_csv(trades_path, index=False)
        files["trades"] = trades_path
    if result.skipped_orders_df is not None:
        skipped_path = out_dir / "skipped_orders.csv"
        result.skipped_orders_df.to_csv(skipped_path, index=False)
        files["skipped_orders"] = skipped_path

    return files
