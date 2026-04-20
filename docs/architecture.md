# Architecture

## Overview

Alpha Lab is a Level 1/2 quantitative research system:

- Level 1: factor discovery
- Level 2: portfolio construction validation

It is a Level 1/2 research system and does not provide execution replay.

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer A — Research Integrity                                   │
│   alpha_lab.research_integrity.asof         PIT/as-of checks    │
│   alpha_lab.research_integrity.leakage_checks anti-leakage      │
│   alpha_lab.research_integrity.reporting     integrity artifacts │
├─────────────────────────────────────────────────────────────────┤
│ Layer B — Factor Research                                      │
│   alpha_lab.factors.*     factor computation (e.g. momentum)    │
│   alpha_lab.model_factor  features -> model score -> factor     │
│   alpha_lab.labels        forward-return label generation        │
│   alpha_lab.evaluation    IC / Rank-IC computation              │
│   alpha_lab.quantile      quantile bucket returns, long-short   │
│   alpha_lab.turnover      quantile / long-short turnover        │
│   alpha_lab.preprocess    winsorize, z-score                    │
├─────────────────────────────────────────────────────────────────┤
│ Layer C — Portfolio Research                                   │
│   alpha_lab.strategy.StrategySpec   portfolio construction spec │
│   alpha_lab.portfolio_research      research-level approximation│
│                                     portfolio weights/returns   │
├─────────────────────────────────────────────────────────────────┤
│ Layer D — Reporting / Registry / Knowledge Export              │
│   alpha_lab.reporting    summaries + experiment-card export     │
│   alpha_lab.reporting.research_validation_package               │
│   alpha_lab.registry     append-only experiment registry        │
│   alpha_lab.comparison   experiment comparison/ranking          │
├─────────────────────────────────────────────────────────────────┤
│ Orchestration (Level 1/2)                                      │
│   alpha_lab.experiment      run_factor_experiment (one split)  │
│   alpha_lab.walk_forward    run_walk_forward_experiment (OOS)  │
├─────────────────────────────────────────────────────────────────┤
│ Support                                                         │
│   alpha_lab.splits       time_split, walk_forward_split        │
│   alpha_lab.costs        cost_adjusted_long_short              │
│   alpha_lab.config       project-root-relative path constants  │
│   alpha_lab.interfaces   validate_factor_output schema guard   │
└─────────────────────────────────────────────────────────────────┘
```

## Layer Contracts

### Layer B (Factor Research) → Layer C (Portfolio Research)

**Input**: long-form `[date, asset, factor, value]` DataFrame (one row per
`(date, asset, factor)`).  Factor values at date `t` may only use information
available at or before `t`.

**Output** consumed by Strategy Layer: the same factor DataFrame, which
`portfolio_weights_from_strategy` uses to rank assets and assign weights.

**`n_quantiles`** lives in this layer (passed to `run_factor_experiment` as a
standalone parameter).  It governs IC and quantile bucket evaluation — not
portfolio weight construction.

### Strategy Boundary Inside Layer C

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

### Layer C → Orchestration

Portfolio Research functions return typed DataFrames with stable column
contracts (`_WEIGHT_COLUMNS`, `_RETURN_COLUMNS`, etc.).  Orchestration
attaches these to `ExperimentResult` optional fields.

## Core vs Experimental Boundary

| Module group | Classification | Role |
|---|---|---|
| `alpha_lab.research_integrity.asof`, `leakage_checks`, `reporting` | Core (Level 1/2) | Research temporal correctness and leakage control |
| `alpha_lab.factors`, `model_factor`, `labels`, `evaluation`, `quantile`, `turnover`, `neutralization` | Core (Level 1/2) | Factor discovery and robustness diagnostics |
| `alpha_lab.strategy`, `portfolio_research`, `experiment`, `walk_forward` | Core (Level 2) | Portfolio construction validation with research approximations |
| `alpha_lab.reporting`, `reporting.research_validation_package`, `registry`, `comparison`, `vault_export` | Core (Level 1/2) | Reproducible reporting and knowledge export |

## Data Flow (single experiment)

```
prices (long-form) / features (wide)
    │
    ├──── Hand-crafted factor path ───────────────► factor_fn(prices)
    │
    └──── Model-factor path ──────────────────────► build_model_factor(features, prices)
                                                     (rolling / expanding training)
                                                     → factor_df [date, asset, factor, value]
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

`alpha_lab.config` defines project-root-relative path constants:

```python
# Env-var override (required for non-editable installs):
PROJECT_ROOT = Path(os.environ["ALPHA_LAB_PROJECT_ROOT"]).resolve()
# Editable-install default (src/alpha_lab/config.py → parents[2]):
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR           = PROJECT_ROOT / "data"
RAW_DATA_DIR       = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
```

**Integrity check**: if `PROJECT_ROOT / "pyproject.toml"` does not exist, a
`RuntimeError` is raised immediately — preventing silent artifact misplacement.

**Env var override**: set `ALPHA_LAB_PROJECT_ROOT` to the project root directory
for non-editable installs or when running from unusual working directories.

All modules that write or read project-relative paths (e.g. `registry.py`,
CLI default `--output-dir`) import from `config` rather than constructing
CWD-relative `Path()` literals.

## Raw Input Validation

`alpha_lab.data_validation.validate_price_panel(df)` enforces the raw price
panel contract at every system entrypoint (CLI, `run_factor_experiment`):

- required columns: `date`, `asset`, `close`
- no empty DataFrame
- no NaT or unparseable dates
- no null/empty asset strings
- no duplicate `(date, asset)` rows
- no NaN close values
- no non-positive close values

`alpha_lab.interfaces.validate_factor_output(df)` enforces the canonical factor
output contract after every `factor_fn` call:

- required columns: `date`, `asset`, `factor`, `value`
- no NaT dates
- no null/empty asset or factor strings
- no duplicate `(date, asset, factor)` rows
- no all-NaN value column

## Entrypoint

The installed CLI entry point is `alpha-lab` (`alpha_lab.cli:main`), with
explicit Level 1/2 routing:

- `alpha-lab run ...` (legacy single-experiment route)
- `alpha-lab real-case ...`
- `alpha-lab real-case model-factor ...`
- `alpha-lab campaign ...`
- `alpha-lab profiles` (evaluation-profile discovery)

`scripts/run_experiment.py` remains as a thin wrapper for legacy single-run
usage. There is no `main.py`. Notebook and script workflows import from
`alpha_lab` directly.
