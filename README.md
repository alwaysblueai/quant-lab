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
- `alpha_lab.preprocess.winsorize_series`
- `alpha_lab.preprocess.zscore_series`
- `alpha_lab.interfaces.validate_factor_output`

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

## Current Limitations

- no transaction-cost or slippage model implementation yet
- no portfolio construction or backtest engine
