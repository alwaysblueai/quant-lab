# System Manual

Reference guide for working with Alpha Lab.  For higher-level architectural
context see [architecture.md](architecture.md).  For contributor guidance see
[developer_guide.md](developer_guide.md).

---

## Setup

**Requirements**: Python 3.12, `uv`.

```bash
uv sync --all-extras
```

**Path behaviour**: the package resolves the project root via
`src/alpha_lab/config.py`.  For editable installs (the standard workflow)
this works automatically.  For non-editable installs, set:

```bash
export ALPHA_LAB_PROJECT_ROOT=/path/to/alpha-lab
```

A `RuntimeError` is raised at import time if the resolved root does not contain
`pyproject.toml`, preventing silent artifact misplacement.

Run all checks:

```bash
make check   # lint + typecheck + test
```

In WSL or sandboxed environments with restricted cache directories:

```bash
UV_CACHE_DIR=/tmp/uv-cache make check
```

---

## Raw Input Validation

`alpha_lab.data_validation.validate_price_panel(df)` is called automatically at
`run_factor_experiment()` and at the CLI entry point.  It raises `ValueError`
(or `SystemExit` at the CLI) on the first violation:

| Check | Error trigger |
|---|---|
| Required columns | `date`, `asset`, `close` missing |
| Non-empty | zero rows |
| Valid dates | NaT or unparseable values in `date` |
| Non-null asset | null or empty-string in `asset` |
| No duplicates | duplicate `(date, asset)` pairs |
| NaN close | any NaN in `close` |
| Positive close | any `close <= 0` |

You can also call it directly before passing data to any pipeline function:

```python
from alpha_lab.data_validation import validate_price_panel
validate_price_panel(your_prices_df)  # raises ValueError on violation
```

---

## Canonical Data Contract

All factor outputs must conform to the long-form schema:

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Observation timestamp |
| `asset` | str | Asset identifier |
| `factor` | str | Factor name |
| `value` | float | Numeric factor value |

Rules:
- At most one row per `(date, asset, factor)`.
- Factor values at `date=t` may only use information available at or before `t`.
- Labels and forward returns must be stored in **separate** tables.

---

## Core API

### run_factor_experiment

Connects all evaluation modules (IC, quantile returns, long-short, turnover,
and optionally portfolio simulation) into a single call.

```python
from alpha_lab.experiment import run_factor_experiment
from alpha_lab.factors.momentum import momentum

result = run_factor_experiment(
    prices,                      # long-form [date, asset, close]
    lambda p: momentum(p, window=20),
    horizon=5,                   # forward-return look-ahead in rows
    n_quantiles=5,               # quantile buckets for IC/quantile eval
    train_end="2022-12-31",      # omit for full-sample evaluation
    test_start="2023-01-01",
)

# Core outputs (always present)
result.summary               # ExperimentSummary scalar metrics
result.ic_df                 # [date, factor, ic]
result.rank_ic_df            # [date, factor, rank_ic]
result.quantile_returns_df   # [date, factor, quantile, quantile_return]
result.long_short_df         # [date, factor, long_short_return]
result.factor_df             # full-sample factor values
result.label_df              # full-sample forward-return labels
```

**Portfolio simulation** (optional):

```python
result = run_factor_experiment(
    prices,
    lambda p: momentum(p, window=20),
    horizon=5,
    holding_period=1,
    rebalance_frequency=1,
    weighting_method="rank",    # "equal", "rank", or "score"
    portfolio_cost_rate=0.001,  # 10 bps one-way; omit for no cost adjustment
)

result.portfolio_weights_df             # [date, asset, weight]
result.portfolio_return_df              # [date, portfolio_return]
result.portfolio_turnover_df            # [date, portfolio_turnover] (active rebalance dates)
result.portfolio_cost_adjusted_return_df  # [date, portfolio_return, adjusted_return]
result.portfolio_summary                # PortfolioSummary scalars
```

**Using StrategySpec** (makes construction intent explicit):

```python
from alpha_lab.strategy import StrategySpec

spec = StrategySpec(
    long_top_k=10,
    weighting_method="rank",
    holding_period=1,
    rebalance_frequency=1,
)

result = run_factor_experiment(
    prices,
    lambda p: momentum(p, window=20),
    horizon=5,
    n_quantiles=5,         # factor-eval param; not part of StrategySpec
    strategy=spec,
    portfolio_cost_rate=0.001,
)
```

When `strategy` is provided, `holding_period`, `rebalance_frequency`, and
`weighting_method` are taken from the spec.  Passing them explicitly alongside
`strategy` raises a `UserWarning` (spec values win).

---

### run_walk_forward_experiment

Rolls `run_factor_experiment` across non-overlapping test windows.  Every
evaluation date is strictly out-of-sample.

```python
from alpha_lab.walk_forward import run_walk_forward_experiment

wf = run_walk_forward_experiment(
    prices,
    lambda p: momentum(p, window=20),
    train_size=252,    # unique dates in each training window
    test_size=63,      # unique dates in each test window
    step=63,           # advance by this many dates between folds
    horizon=5,
    n_quantiles=5,
    cost_rate=0.001,   # long-short cost-adjusted return (separate from portfolio path)
)

# Fold-level summary
wf.fold_summary_df          # one row per fold
wf.per_fold_results         # list of ExperimentResult, one per fold

# Aggregate statistics
agg = wf.aggregate_summary  # WalkForwardAggregate
agg.n_folds
agg.pooled_ic_mean          # mean IC across all OOS observations
agg.pooled_ic_ir            # IC-IR from pooled series
agg.best_fold               # fold_id with highest mean_ic

# Pooled OOS DataFrames
wf.pooled_ic_df                              # [fold_id, date, ic]
wf.pooled_portfolio_return_df               # [fold_id, date, portfolio_return]
wf.pooled_portfolio_turnover_df             # [fold_id, date, portfolio_turnover]
wf.pooled_cost_adjusted_portfolio_return_df # [fold_id, date, portfolio_return, adjusted_return]
```

**`val_size`** is a fold-construction parameter only.  It reserves trailing
training-window dates as a gap between training and test windows.  No
validation-period outputs are produced — the validation dates are excluded from
both training and test evaluation.

---

### StrategySpec

Frozen dataclass that is the explicit boundary between the factor research
layer and the portfolio research layer.

```python
from alpha_lab.strategy import StrategySpec

# Long-only
spec = StrategySpec(
    long_top_k=10,            # None = all assets
    weighting_method="rank",  # "equal", "rank", or "score"
    holding_period=1,
    rebalance_frequency=1,
)

# Long-short (net-zero)
ls_spec = StrategySpec(
    long_top_k=5,
    short_bottom_k=5,
    weighting_method="equal",
    holding_period=2,
    rebalance_frequency=1,
)

spec.is_long_short   # True when short_bottom_k is not None
```

`n_quantiles` is **not** a field of `StrategySpec`.  It governs the
factor-evaluation path (IC, quantile returns) and is passed directly to the
experiment runner.

---

### Reporting

```python
from alpha_lab.reporting import (
    summarise_experiment_result,
    export_summary_csv,
    to_obsidian_markdown,
)

summary = summarise_experiment_result(result, cost_rate=0.001)
export_summary_csv(summary, "output/reports/momentum_5d.csv")
md = to_obsidian_markdown(result, title="Momentum 5d OOS", cost_rate=0.001)
```

---

### Registry

Append-only CSV log of experiment results.  Default location:
`<project_root>/data/processed/experiment_registry.csv`
(anchored by `alpha_lab.config.PROCESSED_DATA_DIR`, not the current working
directory).

```python
from alpha_lab.registry import register_experiment, load_registry

register_experiment("momentum_20d_5q_oos_2023", summary)
registry = load_registry()
```

Schema is validated on every append and load; mismatches raise `ValueError`.

---

### Comparison

```python
from alpha_lab.comparison import compare_experiments, rank_experiments

comparison = compare_experiments([summary_a, summary_b])
ranked = rank_experiments(comparison, metric="ic_ir")
```

---

## Timestamp Discipline

- `forward_return(prices, horizon=h)` stores `close[t+h]/close[t]-1` **at
  row `t`** so it can be merged with factor values observed at `t` without
  lookahead.  The label value uses strictly future prices.
- Portfolio simulation inside `_run_portfolio_block` always uses
  `forward_return(prices, horizon=1)` (one-period step returns), **not** the
  H-period evaluation labels, to avoid compounding mismatch in the
  staggered-portfolio model.

---

## Cost Model

```
adjusted_return(t) = portfolio_return(t) − cost_rate × turnover(t)
```

- Applied only on **active rebalance dates** (every `rebalance_frequency`-th
  weight date).
- First active rebalance date is always `NaN` (no prior portfolio state).
- Non-rebalance evaluation dates receive `adjusted_return = portfolio_return`
  (zero incremental cost).
- `cost_rate` is one-way, flat-rate.  It does not model market impact,
  bid-ask spread variation, short-borrow fees, or execution timing.

---

## Provenance and Diagnostics

Every `ExperimentResult` carries a `provenance` field (`ExperimentProvenance`)
and three diagnostic counts:

```python
result.provenance.factor_name        # e.g. "momentum_5d"
result.provenance.horizon            # forward-return horizon used
result.provenance.n_quantiles        # quantile buckets used
result.provenance.run_timestamp_utc  # ISO-8601 UTC run time
result.provenance.git_commit         # short commit hash or None
result.provenance.portfolio_cost_rate
result.provenance.strategy_repr      # repr(spec) or None

result.n_eval_dates       # distinct dates in the evaluation period
result.n_eval_assets      # distinct assets in the evaluation period
result.n_label_nan_dates  # eval dates with no valid forward return label
                          # (= horizon for full-sample runs)
```

`n_label_nan_dates` tells you how many trailing dates were excluded from IC and
quantile-return computation because the forward-return horizon extended beyond
the available price history.

---

## Parameter Misuse Warnings

Two `UserWarning`s are raised for clearly no-op parameter combinations:

1. **`portfolio_cost_rate` without portfolio mode** — if `portfolio_cost_rate`
   is supplied but neither `holding_period`/`rebalance_frequency` nor a
   `StrategySpec` is provided, the rate would be silently dropped.  A warning
   is raised in both `run_factor_experiment` and `run_walk_forward_experiment`.

2. **`holding_period`/`rebalance_frequency` alongside `strategy`** — the spec
   values override; the explicit arguments are warned and ignored.

---

## Scope Limitations

- No full backtesting engine or realistic execution simulation.
- No position accounting or broker integration.
- No market impact or intraday slippage model.
- No database, dashboard, or streaming experiment tracking.
- Cost model is a minimal research friction estimate only.
