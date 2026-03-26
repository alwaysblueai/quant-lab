# Alpha Lab

Minimal quantitative research workspace for factor prototyping.

## Scope

This repository is intentionally small. It supports:

- reusable factor code under `src/alpha_lab`
- tests, linting, and type checking for local development
- documented data conventions for auditable research

It does not currently provide a full backtesting engine, production data ingestion
pipeline, realistic execution simulation, or broker integration.

## Architecture

The repository is structured around three research layers.  Each layer has a
well-defined scope and does **not** model execution, order routing, or position
accounting.

```
Factor Research Layer
  alpha_lab.factors.*          — factor computation (e.g. momentum)
  alpha_lab.labels             — forward-return label generation
  alpha_lab.evaluation         — IC / Rank-IC computation
  alpha_lab.quantile           — quantile bucket returns and long-short
  alpha_lab.turnover           — quantile / long-short turnover
  alpha_lab.preprocess         — winsorize, z-score

Strategy Construction Intent Layer
  alpha_lab.strategy.StrategySpec   — explicit portfolio construction spec:
                                      long_top_k, short_bottom_k,
                                      weighting_method, holding_period,
                                      rebalance_frequency
                                      (n_quantiles is a factor-evaluation
                                       param, not part of StrategySpec)

Portfolio Research Layer
  alpha_lab.portfolio_research — portfolio_weights, simulate_portfolio_returns,
                                 portfolio_turnover,
                                 portfolio_cost_adjusted_returns

Orchestration
  alpha_lab.experiment         — run_factor_experiment (single split)
  alpha_lab.walk_forward       — run_walk_forward_experiment (rolling OOS)
```

`StrategySpec` is the explicit boundary between the factor research layer and
the portfolio research layer.  It answers: which assets to include in each leg,
how to weight them, and how often to rebalance.  Passing a `StrategySpec` to
`run_factor_experiment` or `run_walk_forward_experiment` makes all portfolio
construction intent visible in one place rather than spread across call sites.

**This is still NOT a full trading system or execution simulator.**  There is no
order routing, market impact model, position accounting, broker integration, or
portfolio optimiser.  All portfolio outputs are research-level approximations.

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

## Documentation

- [docs/architecture.md](docs/architecture.md) — layer contracts, data flow, path/config
- [docs/system_manual.md](docs/system_manual.md) — API reference and usage patterns
- [docs/developer_guide.md](docs/developer_guide.md) — how to extend the codebase
- [docs/data_conventions.md](docs/data_conventions.md) — canonical timestamp, merge, and storage rules

## Current Reusable Components

- `alpha_lab.strategy.StrategySpec`
- `alpha_lab.strategy.portfolio_weights_from_strategy`
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

## Strategy Construction Intent

Use `StrategySpec` to make portfolio construction intent explicit before
passing it to the experiment runner:

```python
from alpha_lab.strategy import StrategySpec

# Long-only: top 10 assets, rank-weighted, rebalance every date, hold 1 period
spec = StrategySpec(
    long_top_k=10,
    weighting_method="rank",
    holding_period=1,
    rebalance_frequency=1,
)

# Long-short: top 5 long / bottom 5 short, equal-weighted
ls_spec = StrategySpec(
    long_top_k=5,
    short_bottom_k=5,
    weighting_method="equal",
    holding_period=2,
    rebalance_frequency=1,
)

result = run_factor_experiment(
    prices,
    lambda p: momentum(p, window=20),
    horizon=5,
    strategy=spec,
    portfolio_cost_rate=0.001,
)
print(result.portfolio_summary)
```

When `strategy` is provided it overrides `holding_period`, `rebalance_frequency`,
and `weighting_method` (a `UserWarning` is raised if those are also passed
explicitly).  `n_quantiles` governs the factor-evaluation path (IC, quantile
returns) and is **not** part of `StrategySpec` — pass it directly to
`run_factor_experiment`.  `portfolio_cost_rate` is intentionally not part of
`StrategySpec` — it is a cost assumption, not a construction decision.

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

Experiment card export to quant-knowledge:

```python
import os
from alpha_lab.reporting import export_experiment_card

# Option A: set environment variable once
os.environ["OBSIDIAN_VAULT_PATH"] = "/path/to/quant-knowledge"
path = export_experiment_card(result, name="momentum-5d-Ashare")

# Option B: pass vault_path explicitly
path = export_experiment_card(
    result,
    name="momentum-5d-Ashare",
    vault_path="/path/to/quant-knowledge",
)
```

If neither `vault_path` nor `OBSIDIAN_VAULT_PATH` is provided, export raises
`ValueError`.

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

## CLI

A thin command-line wrapper over the existing pipeline lives at
`scripts/run_experiment.py`.  It does not redesign the pipeline — it parses
arguments and delegates to the same modules used in notebook workflows.

**Input CSV** must contain at least the columns `date`, `asset`, and `close`.
Extra columns are ignored.

```bash
# Minimal run — writes a summary CSV to output/
uv run python scripts/run_experiment.py \
    --input-path data/raw/prices.csv \
    --factor momentum \
    --label-horizon 5 \
    --quantiles 5

# Full run: split, cost rate, Obsidian note, registry entry
uv run python scripts/run_experiment.py \
    --input-path data/raw/prices.csv \
    --factor momentum \
    --momentum-window 20 \
    --label-horizon 5 \
    --quantiles 5 \
    --train-end 2022-12-31 \
    --test-start 2023-01-01 \
    --cost-rate 0.001 \
    --experiment-name momentum_20d_5q_oos_2023 \
    --output-dir output/reports \
    --obsidian-markdown-path notes/momentum_20d_5q_oos_2023.md \
    --append-registry

# Write the note into a directory — filename is auto-generated as
# YYYY-MM-DD_{experiment_name}.md
uv run python scripts/run_experiment.py \
    --input-path data/raw/prices.csv \
    --factor momentum \
    --label-horizon 5 \
    --quantiles 5 \
    --obsidian-markdown-path notes/
```

`--experiment-name` must contain only letters, digits, hyphens, underscores,
and dots — path separators are rejected to prevent accidental file writes
outside `--output-dir`.

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

## Walk-Forward Evaluation and Portfolio Research

### Walk-Forward Evaluation

A single train/test split can overfit to the test period: the researcher may
consciously or unconsciously choose factor parameters that happen to look good
on that one window.  Walk-forward evaluation forces every evaluation date to
be strictly out-of-sample by rolling the train and test windows forward
through time.

`run_walk_forward_experiment` wraps `run_factor_experiment` over all folds
produced by `walk_forward_split`.  Each fold receives only the prices visible
up to its own test-end date — no future data can leak into the factor
computation.  Evaluation metrics (IC, L/S return, turnover) are computed on
the test window only.

```python
from alpha_lab.walk_forward import run_walk_forward_experiment
from alpha_lab.factors.momentum import momentum

wf = run_walk_forward_experiment(
    prices,
    lambda p: momentum(p, window=20),
    train_size=252,   # 1-year training window (trading days)
    test_size=63,     # 1-quarter test window
    step=63,          # advance by one quarter between folds
    horizon=5,
    n_quantiles=5,
    cost_rate=0.001,
)

print(wf.aggregate_summary)
# WalkForwardAggregate(n_folds=4, mean_ic=..., std_ic=...,
#   pooled_ic_mean=..., pooled_ic_ir=...,
#   mean_portfolio_return=..., pooled_portfolio_return_mean=...,
#   pooled_cost_adjusted_return_mean=..., ...)

# Per-fold breakdown
print(wf.fold_summary_df[["fold_id", "start_date", "end_date", "mean_ic", "ic_ir"]])

# Each fold's full ExperimentResult is also available
first_fold = wf.per_fold_results[0]

# Pooled OOS observation DataFrames (all folds concatenated, test window only)
wf.pooled_ic_df                              # [fold_id, date, ic]
wf.pooled_portfolio_return_df               # [fold_id, date, portfolio_return]
wf.pooled_portfolio_turnover_df             # [fold_id, date, portfolio_turnover]
wf.pooled_cost_adjusted_portfolio_return_df # [fold_id, date, portfolio_return, adjusted_return]
```

**Why walk-forward reduces overfitting risk**: the aggregate `mean_ic` and
`std_ic` reflect the factor's consistency across multiple independent test
windows.  A high `std_ic` relative to `mean_ic` signals that performance is
unstable and may not generalise.  The `best_fold` / `worst_fold` fields
identify the most and least favourable periods for deeper investigation.

**Single split vs. walk-forward**: `run_factor_experiment` with a single
`train_end` / `test_start` evaluates on one contiguous test window.
Walk-forward provides multiple independent windows of the same total span,
giving a distribution of outcomes rather than a point estimate.

### Portfolio Research Layer

`alpha_lab.portfolio_research` provides research-level portfolio construction
and simulation tools.  These are designed for signal evaluation, not live
execution.

#### Computing weights

```python
from alpha_lab.portfolio_research import portfolio_weights

# Long-only: top 20 assets, weights proportional to factor rank
weights = portfolio_weights(
    factor_df,
    method="rank",   # or "equal", "score"
    top_k=20,
)

# Long-short: top 10 long (+weight sums to 1),
#             bottom 10 short (weight sums to -1, net = 0)
ls_weights = portfolio_weights(
    factor_df,
    method="equal",
    top_k=10,
    bottom_k=10,
)
```

Weight methods:
- `"equal"`: uniform weight across selected assets.
- `"rank"`: weight proportional to cross-sectional factor rank.
- `"score"`: weight proportional to `value − min(value)` across the selection.

#### Simulating returns with overlapping holdings

```python
from alpha_lab.portfolio_research import simulate_portfolio_returns
from alpha_lab.labels import forward_return

# 1-period returns (pass result.label_df or compute fresh)
labels = forward_return(prices, horizon=1)

port_returns = simulate_portfolio_returns(
    weights,
    labels,
    holding_period=5,       # hold each position for 5 rebalance periods
    rebalance_frequency=1,  # rebalance at every available date
)
# Returns: DataFrame[date, portfolio_return]
```

When `holding_period > rebalance_frequency`, multiple overlapping positions
are active simultaneously.  The portfolio return on each date is the mean
across all currently active positions — the standard staggered-portfolio
model used in academic factor research.

#### Portfolio turnover

```python
from alpha_lab.portfolio_research import portfolio_turnover

to = portfolio_turnover(weights)
# Returns: DataFrame[date, portfolio_turnover]
# turnover(t) = 0.5 × Σ|w_new_i − w_old_i|  (two-way, fraction traded)
# First date is always NaN (no prior state).
```

#### Integrating into run_factor_experiment

Pass `holding_period` and `rebalance_frequency` to attach portfolio
outputs to the standard `ExperimentResult`:

```python
result = run_factor_experiment(
    prices,
    lambda p: momentum(p, window=20),
    horizon=5,
    n_quantiles=5,
    holding_period=1,
    rebalance_frequency=1,
    weighting_method="rank",
)

result.portfolio_weights_df   # DataFrame[date, asset, weight]
result.portfolio_return_df    # DataFrame[date, portfolio_return]
```

**Research disclaimer**: this is a minimal friction estimate for signal
evaluation only.  It does not model market impact, intraday slippage,
short-borrow costs, execution timing, or partial fills.

## Current Limitations

- no full backtesting engine or realistic execution simulation
- no transaction-cost model beyond linear flat-rate research approximation
- no database, dashboard, or experiment tracking framework
