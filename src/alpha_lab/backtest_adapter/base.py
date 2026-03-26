from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.backtest_adapter.schema import (
    BacktestInputBundle,
    BacktestResult,
    BacktestRunConfig,
)


def run_external_backtest(
    bundle: BacktestInputBundle,
    config: BacktestRunConfig,
) -> BacktestResult:
    """Dispatch to the requested engine adapter."""
    if config.engine == "vectorbt":
        from alpha_lab.backtest_adapter.vectorbt_adapter import run_vectorbt_backtest

        return run_vectorbt_backtest(bundle, config)
    if config.engine == "backtrader":
        from alpha_lab.backtest_adapter.backtrader_adapter import run_backtrader_backtest

        return run_backtrader_backtest(bundle, config)
    raise ValueError(f"unsupported engine {config.engine!r}")


def compute_basic_performance_summary(
    returns: pd.Series,
    equity: pd.Series,
    turnover: pd.Series,
) -> dict[str, float]:
    """Compute engine-agnostic, deterministic summary metrics."""
    clean_returns = returns.dropna()
    if clean_returns.empty:
        return {
            "n_periods": 0.0,
            "mean_return": float("nan"),
            "volatility": float("nan"),
            "sharpe_annualized": float("nan"),
            "total_return": float("nan"),
            "max_drawdown": float("nan"),
            "mean_turnover": float("nan"),
        }

    mean_return = float(clean_returns.mean())
    volatility = float(clean_returns.std(ddof=0))
    sharpe = (
        (mean_return / volatility) * float(np.sqrt(252.0))
        if volatility > 0
        else float("nan")
    )
    start = float(equity.iloc[0])
    total_return = float(equity.iloc[-1] / start - 1.0) if start > 0 else float("nan")
    running_max = equity.cummax()
    drawdowns = equity / running_max - 1.0
    max_drawdown = float(drawdowns.min())
    mean_turnover = float(turnover.dropna().mean()) if not turnover.empty else float("nan")

    return {
        "n_periods": float(len(clean_returns)),
        "mean_return": mean_return,
        "volatility": volatility,
        "sharpe_annualized": sharpe,
        "total_return": total_return,
        "max_drawdown": max_drawdown,
        "mean_turnover": mean_turnover,
    }


def export_backtest_result(
    result: BacktestResult,
    output_dir: str | Path,
    *,
    export_target_weights: bool,
    export_series: bool,
) -> dict[str, Path]:
    """Write a compact set of replay artifacts for inspection/audit."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}

    summary_path = out_dir / "backtest_summary.json"
    summary_payload = {
        "artifact_path": result.adapter_run_metadata["artifact_path"],
        "experiment_id": result.adapter_run_metadata["experiment_id"],
        "dataset_fingerprint": result.adapter_run_metadata["dataset_fingerprint"],
        "engine": result.adapter_run_metadata["engine"],
        "summary": result.summary,
        "warnings": [w.__dict__ for w in result.warnings],
        "engine_stats": result.engine_stats,
        "adapter_run_metadata": result.adapter_run_metadata,
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    files["summary"] = summary_path

    metadata_path = out_dir / "adapter_run_metadata.json"
    metadata_path.write_text(
        json.dumps(result.adapter_run_metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    files["adapter_run_metadata"] = metadata_path

    if export_series:
        returns_path = out_dir / "portfolio_returns.csv"
        result.returns_df.to_csv(returns_path, index=False)
        files["returns"] = returns_path

        equity_path = out_dir / "equity_curve.csv"
        result.equity_curve_df.to_csv(equity_path, index=False)
        files["equity_curve"] = equity_path

        turnover_path = out_dir / "turnover.csv"
        result.turnover_df.to_csv(turnover_path, index=False)
        files["turnover"] = turnover_path

    if export_target_weights:
        target_path = out_dir / "target_weights.csv"
        result.target_weights_df.to_csv(target_path, index=False)
        files["target_weights"] = target_path

        executed_path = out_dir / "executed_weights.csv"
        result.executed_weights_df.to_csv(executed_path, index=False)
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
