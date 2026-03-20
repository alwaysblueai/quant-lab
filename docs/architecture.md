# Architecture

## Overview

Alpha Lab is a minimal quantitative research workspace.  It is organised into
three layers with explicit contracts between them.  No layer models execution,
order routing, position accounting, or broker integration.

```
┌─────────────────────────────────────────────────────────────────┐
│ Factor Research Layer                                           │
│   alpha_lab.factors.*     factor computation (e.g. momentum)   │
│   alpha_lab.labels        forward-return label generation       │
│   alpha_lab.evaluation    IC / Rank-IC computation             │
│   alpha_lab.quantile      quantile bucket returns, long-short   │
│   alpha_lab.turnover      quantile / long-short turnover        │
│   alpha_lab.preprocess    winsorize, z-score                   │
├─────────────────────────────────────────────────────────────────┤
│ Strategy Construction Intent Layer                              │
│   alpha_lab.strategy.StrategySpec   portfolio construction spec │
│                                     (explicit boundary object)  │
├─────────────────────────────────────────────────────────────────┤
│ Portfolio Research Layer                                        │
│   alpha_lab.portfolio_research      portfolio_weights,          │
│                                     simulate_portfolio_returns, │
│                                     portfolio_turnover,         │
│                                     portfolio_cost_adjusted_    │
│                                     returns                     │
├─────────────────────────────────────────────────────────────────┤
│ Orchestration                                                   │
│   alpha_lab.experiment      run_factor_experiment (one split)  │
│   alpha_lab.walk_forward    run_walk_forward_experiment (OOS)  │
├─────────────────────────────────────────────────────────────────┤
│ Support                                                         │
│   alpha_lab.splits       time_split, walk_forward_split        │
│   alpha_lab.reporting    summarise, export CSV, Obsidian note  │
│   alpha_lab.registry     append-only CSV experiment log        │
│   alpha_lab.comparison   compare_experiments, rank_experiments │
│   alpha_lab.costs        cost_adjusted_long_short              │
│   alpha_lab.config       project-root-relative path constants  │
│   alpha_lab.interfaces   validate_factor_output schema guard   │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Contracts

### Factor Research Layer → Strategy Layer

**Input**: long-form `[date, asset, factor, value]` DataFrame (one row per
`(date, asset, factor)`).  Factor values at date `t` may only use information
available at or before `t`.

**Output** consumed by Strategy Layer: the same factor DataFrame, which
`portfolio_weights_from_strategy` uses to rank assets and assign weights.

**`n_quantiles`** lives in this layer (passed to `run_factor_experiment` as a
standalone parameter).  It governs IC and quantile bucket evaluation — not
portfolio weight construction.

### Strategy Layer → Portfolio Research Layer

**`StrategySpec`** is the explicit boundary object.  It answers only
portfolio-construction questions:

| Field | Purpose |
|---|---|
| `long_top_k` | how many top-ranked assets enter the long leg |
| `short_bottom_k` | how many bottom-ranked assets enter the short leg (None = long-only) |
| `weighting_method` | `"equal"`, `"rank"`, or `"score"` |
| `holding_period` | periods to hold each position |
| `rebalance_frequency` | dates between rebalances |

`n_quantiles` and `portfolio_cost_rate` are **not** part of `StrategySpec`.
They belong to the orchestration caller.

### Portfolio Research Layer → Orchestration

Portfolio Research functions return typed DataFrames with stable column
contracts (`_WEIGHT_COLUMNS`, `_RETURN_COLUMNS`, etc.).  Orchestration
attaches these to `ExperimentResult` optional fields.

## Data Flow (single experiment)

```
prices (long-form)
    │
    ▼
factor_fn(prices)  →  factor_df  [date, asset, factor, value]
    │
    ├──► forward_return(prices, horizon)  →  label_df
    │
    ├──► eval period mask (time_split or full sample)
    │
    ├──── Factor Eval Path ─────────────────────────────────────────
    │     compute_ic, compute_rank_ic, quantile_returns,
    │     long_short_return, quantile_assignments,
    │     quantile_turnover, long_short_turnover
    │     → ic_df, rank_ic_df, quantile_returns_df, long_short_df, …
    │
    └──── Portfolio Path (optional, requires holding_period) ───────
          portfolio_weights / portfolio_weights_from_strategy
              → weights_df  [date, asset, weight]
          simulate_portfolio_returns (1-period step returns)
              → return_df   [date, portfolio_return]
          portfolio_turnover (active rebalance dates only)
              → turnover_df [date, portfolio_turnover]
          portfolio_cost_adjusted_returns (if cost_rate supplied)
              → cost_adj_df [date, portfolio_return, adjusted_return]
```

## Walk-Forward Evaluation

`run_walk_forward_experiment` wraps `run_factor_experiment` over rolling folds
produced by `walk_forward_split`.  Each fold:

1. Receives prices filtered to `date ≤ test_end` so factor_fn cannot access
   future data beyond the fold's test period.
2. Evaluates on its own non-overlapping test window only.
3. Contributes one row to `fold_summary_df` and one slice to each pooled
   observation DataFrame.

**Pooled OOS DataFrames** (all folds concatenated, test window only):

| Field | Columns |
|---|---|
| `pooled_ic_df` | `fold_id, date, ic` |
| `pooled_portfolio_return_df` | `fold_id, date, portfolio_return` |
| `pooled_portfolio_turnover_df` | `fold_id, date, portfolio_turnover` |
| `pooled_cost_adjusted_portfolio_return_df` | `fold_id, date, portfolio_return, adjusted_return` |

Pooled series are statistically sounder than mean-of-fold-means when fold
sizes differ.

## Path / Config

`alpha_lab.config` defines project-root-relative path constants anchored to the
location of the installed package:

```python
PROJECT_ROOT       = Path(__file__).resolve().parents[2]
DATA_DIR           = PROJECT_ROOT / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
```

All modules that write or read project-relative paths (e.g. `registry.py`)
import from `config` rather than constructing CWD-relative `Path()` literals.

## Entrypoint

The CLI entry point is `scripts/run_experiment.py`, which delegates to
`alpha_lab.cli`.  There is no `main.py`.  Notebook and script workflows import
from `alpha_lab` directly.
