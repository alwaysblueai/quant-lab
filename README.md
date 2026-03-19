# Alpha Lab

Minimal quantitative research workspace for factor prototyping.

## Scope

This repository is intentionally small. It supports:

- reusable factor code under `src/alpha_lab`
- tests, linting, and type checking for local development
- documented data conventions for auditable research

It does not currently provide a full backtesting engine, production data ingestion
pipeline, or portfolio simulation framework.

## Setup

Requirements:

- Python 3.12
- `uv`

Install the environment:

```bash
uv sync --all-extras
```

Run the local checks:

```bash
make check
```

If `uv` cache permissions are restricted in WSL or a sandboxed environment, set a
writable cache directory:

```bash
UV_CACHE_DIR=/tmp/uv-cache make check
```

## Project Structure

- `src/alpha_lab`: reusable research code
- `tests`: unit and regression tests
- `docs`: repository conventions and lightweight documentation
- `data/raw`: raw immutable datasets
- `data/processed`: derived intermediate datasets
- `scripts`: one-off scripts
- `notebooks`: exploratory work

## Workflow

1. Add or modify reusable logic under `src/alpha_lab`.
2. Add tests for every reusable function and every known leakage or alignment risk.
3. Run:

```bash
make lint
make typecheck
make test
```

Or:

```bash
make check
```

## Canonical Factor Output Schema

All reusable factors must return long-form output with exactly these columns:

- `date`: observation timestamp for the factor value
- `asset`: asset identifier
- `factor`: factor name
- `value`: numeric factor value

Rules:

- one row per `(date, asset, factor)`
- features at `date=t` may only use information available at or before `t`
- labels and forward returns belong in separate tables
- merges must be explicit on `("date", "asset")`, and include `factor` when stacking factors

Example:

| date | asset | factor | value |
|------|-------|--------|-------|
| 2024-01-02 | AAPL | momentum_20d | 0.031 |
| 2024-01-02 | MSFT | momentum_20d | -0.008 |

## Data Conventions

See [docs/data_conventions.md](/home/yukun_zhao/quant/projects/alpha-lab/docs/data_conventions.md)
for the canonical timestamp, merge, and storage rules.

## Current Reusable Components

- `alpha_lab.factors.momentum.momentum`
- `alpha_lab.labels.forward_return`
- `alpha_lab.evaluation.compute_ic`
- `alpha_lab.evaluation.compute_rank_ic`
- `alpha_lab.quantile.quantile_returns`
- `alpha_lab.quantile.long_short_return`
- `alpha_lab.splits.time_split`
- `alpha_lab.splits.walk_forward_split`
- `alpha_lab.experiment.run_factor_experiment`
- `alpha_lab.reporting.summarise_experiment_result`
- `alpha_lab.reporting.export_summary_csv`
- `alpha_lab.reporting.to_obsidian_markdown`
- `alpha_lab.quantile.quantile_assignments`
- `alpha_lab.turnover.quantile_turnover`
- `alpha_lab.turnover.long_short_turnover`
- `alpha_lab.costs.apply_linear_cost`
- `alpha_lab.costs.cost_adjusted_long_short`
- `alpha_lab.preprocess.winsorize_series`
- `alpha_lab.preprocess.zscore_series`
- `alpha_lab.interfaces.validate_factor_output`
- `alpha_lab.comparison.compare_experiments`
- `alpha_lab.comparison.rank_experiments`
- `alpha_lab.registry.register_experiment`
- `alpha_lab.registry.load_registry`
- `alpha_lab.registry.append_to_registry`

## Running an Experiment

`run_factor_experiment` connects all evaluation modules into a single call:

```python
import pandas as pd
from alpha_lab.experiment import run_factor_experiment
from alpha_lab.factors.momentum import momentum

prices: pd.DataFrame  # long-form [date, asset, close]

result = run_factor_experiment(
    prices,
    lambda p: momentum(p, window=20),
    horizon=5,          # forward-return look-ahead in per-asset rows
    n_quantiles=5,
    train_end="2022-12-31",
    test_start="2023-01-01",
)

print(result.summary)
# ExperimentSummary(mean_ic=..., mean_rank_ic=..., ic_ir=...,
#                   mean_long_short_return=..., long_short_hit_rate=..., n_dates=...)
```

`result.factor_df` and `result.label_df` always cover the full sample.
`result.ic_df`, `result.rank_ic_df`, `result.quantile_returns_df`, and
`result.long_short_df` are restricted to the evaluation period.

The split is date-based: every row sharing a test-period date enters evaluation,
while train-period rows are excluded. Labels at test date `t` still use strictly
future prices (`close[t+horizon]/close[t]-1`) — that is by construction, not
lookahead, because the label value is stored at `t` for alignment with factor
values observed at `t`.

## Reporting

Turn any `ExperimentResult` into a summary record, CSV, or Obsidian note:

```python
from alpha_lab.reporting import (
    export_summary_csv,
    summarise_experiment_result,
    to_obsidian_markdown,
)

# One-row summary DataFrame (stackable across experiments)
# n_quantiles, train_end, and test_start are carried on result automatically
summary = summarise_experiment_result(result)

# Export to CSV (parent directories created automatically)
export_summary_csv(summary, "output/reports/momentum_5d.csv")

# Obsidian-friendly markdown note
md = to_obsidian_markdown(result, title="Momentum 5d — OOS", notes="Needs decay analysis.")
```

## Turnover and Cost Estimation

`ExperimentResult` now includes portfolio turnover outputs computed alongside
the IC and quantile-return metrics:

```python
# Turnover is already computed inside run_factor_experiment
result.quantile_turnover_df      # (date, factor, quantile, turnover)
result.long_short_turnover_df    # (date, factor, long_short_turnover)
result.summary.mean_long_short_turnover

# Apply a cost rate manually
from alpha_lab.costs import cost_adjusted_long_short
adj = cost_adjusted_long_short(
    result.long_short_df,
    result.long_short_turnover_df,
    cost_rate=0.001,  # 10 bps one-way
)

# Or include in the summary / markdown report
summary = summarise_experiment_result(result, cost_rate=0.001)
md = to_obsidian_markdown(result, cost_rate=0.001)
```

**Important:** This is a minimal research friction estimate only.  Turnover
uses a one-way entry-rate definition on calendar-rebalance portfolios.  The
cost model is `adjusted_return = return - cost_rate × turnover` with a
user-supplied flat one-way rate.  It does not model market impact, bid-ask
spread variation, short-borrow fees, or execution timing.

## Comparison and Registry

Run multiple experiments, compare them side-by-side, and persist results to a
lightweight CSV registry:

```python
from alpha_lab.comparison import compare_experiments, rank_experiments
from alpha_lab.registry import load_registry, register_experiment
from alpha_lab.reporting import summarise_experiment_result

# --- 1. Run experiments and summarise ---
result_a = run_factor_experiment(prices, lambda p: momentum(p, window=20), horizon=5)
result_b = run_factor_experiment(prices, lambda p: momentum(p, window=5), horizon=5)

summary_a = summarise_experiment_result(result_a)
summary_b = summarise_experiment_result(result_b)

# --- 2. Compare side-by-side ---
comparison = compare_experiments([summary_a, summary_b])
ranked = rank_experiments(comparison, metric="ic_ir")

# --- 3. Register to the CSV log ---
register_experiment("momentum_20d_5h", summary_a)
register_experiment("momentum_5d_5h",  summary_b)

# --- 4. Reload the registry ---
registry = load_registry()
```

The registry is stored at `data/processed/experiment_registry.csv` by default.
Each call to `register_experiment` appends one row; the file is created on
first use.  The registry is an append-only log — duplicate experiment names
are permitted.  Schema consistency is checked on every append and load.

## Current Limitations

- no portfolio construction or backtest engine
- no execution simulation or realistic transaction-cost model
- no database, dashboard, or experiment tracking framework
