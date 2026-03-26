# Architecture

## Overview

Alpha Lab is a minimal quantitative research workspace.  It is organised into
three layers with explicit contracts between them.  No layer models execution,
order routing, position accounting, or broker integration.

```
┌────────────────────────────────────────────────────────────────────┐
│ Research Contract Layer                                            │
│   alpha_lab.data_validation      raw price-panel guards            │
│   alpha_lab.interfaces           canonical factor-output guards     │
│   alpha_lab.research_contracts   typed research bundle contracts    │
│   alpha_lab.research_universe    PIT universe/tradability builder   │
│   alpha_lab.timing               DelaySpec / LabelMetadata          │
│   alpha_lab.experiment_metadata  governance metadata schema         │
│   alpha_lab.validation_scaffold  walk-forward validation metadata   │
│   alpha_lab.purged_validation    purged/embargo split logic         │
│   alpha_lab.sample_weights       concurrency/uniqueness/decay       │
├────────────────────────────────────────────────────────────────────┤
│ Factor Research Layer                                               │
│   alpha_lab.factors.*          factor computation                  │
│   alpha_lab.signal_transforms  winsorize/zscore/rank/neutralize    │
│   alpha_lab.labels             forward/rankpct/event labels         │
│   alpha_lab.neutralization     residual neutralization              │
│   alpha_lab.evaluation         IC / Rank-IC                        │
│   alpha_lab.quantile           quantile returns / long-short        │
│   alpha_lab.turnover           quantile / long-short turnover       │
│   alpha_lab.factor_report      rich diagnostics bundle              │
│   alpha_lab.factor_selection   screening / redundancy gates         │
│   alpha_lab.multiple_testing   significance inflation controls       │
│   alpha_lab.feature_importance MDI/MDA/SFI scaffolds               │
├────────────────────────────────────────────────────────────────────┤
│ Strategy + Portfolio Research Layer                                 │
│   alpha_lab.strategy             StrategySpec boundary object       │
│   alpha_lab.portfolio_research   research-level weight/return path  │
├────────────────────────────────────────────────────────────────────┤
│ Orchestration and Governance                                        │
│   alpha_lab.experiment           run_factor_experiment              │
│   alpha_lab.walk_forward         run_walk_forward_experiment        │
│   alpha_lab.research_templates   end-to-end research workflows      │
│   alpha_lab.reporting            summaries + card export            │
│   alpha_lab.data_sources         vendor ingestion + standardization │
│   alpha_lab.handoff              external backtest handoff export   │
│   alpha_lab.backtest_adapter     external replay adapter layer       │
│   alpha_lab.research_package     case-level archival package layer   │
│   alpha_lab.trial_log            append-only trial accounting       │
│   alpha_lab.registry             summary registry                   │
│   alpha_lab.alpha_registry       alpha lifecycle registry           │
│   alpha_lab.composite_signals    IC/ICIR/equal blend layer          │
│   alpha_lab.alpha_pool_diagnostics breadth/diversification checks   │
│   alpha_lab.rebalance_recommendation decay-aware cadence helper     │
│   alpha_lab.capacity_diagnostics  ADV/capacity warning layer        │
│   alpha_lab.exposure_audit        industry/style exposure audit      │
│   alpha_lab.research_costs        layered friction proxies           │
└────────────────────────────────────────────────────────────────────┘
```

## Layer Contracts

### Vendor Data Layer → Research Contract Layer

`alpha_lab.data_sources` is the only vendor-facing boundary in the repository.
It isolates Tushare extraction from research workflows:

- raw snapshots preserve vendor schemas and extraction params
- standardization converts Tushare fields into canonical internal tables
- bundle building emits workflow-compatible `prices`, `asset_metadata`,
  `market_state`, `neutralization_exposures`, and candidate signal tables

The core workflow architecture does not call Tushare APIs directly.

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
    ├── validate_price_panel
    │
    ├── optional research sample construction
    │   └── universe / tradability / exclusion reasons
    │
    ├── factor_fn(prices) -> factor_df
    │   └── validate_factor_output
    │
    ├── label generation -> label_df
    │   ├── forward_return / rankpct / event labels
    │   └── LabelMetadata + DelaySpec
    │
    ├── optional sample weights
    │   └── uniqueness / return magnitude / decay / confidence
    │
    ├── evaluation split mask (time_split or full sample)
    │
    ├── factor diagnostics path
    │   ├── IC / Rank-IC / quantile / long-short / turnover
    │   └── optional FactorReport (rolling IC, coverage, monotonicity, decay)
    │
    ├── optional governance diagnostics
    │   ├── neutralization / screening / multiple-testing
    │   ├── feature importance / alpha-pool breadth
    │   └── capacity / exposure / friction diagnostics
    │
    ├── optional portfolio research path
    │   └── weights / returns / turnover / cost-adjusted returns
    │
    └── ExperimentResult
        ├── provenance
        ├── ExperimentMetadata
        ├── DelaySpec + LabelMetadata
        └── optional FactorReport
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

Walk-forward also surfaces explicit validation scaffolding:
- `validation_spec`: global walk-forward split assumptions
- `fold_windows_df`: fold-level train/val/test timestamp windows

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

The CLI entry point is `scripts/run_experiment.py`, which delegates to
`alpha_lab.cli`.  There is no `main.py`.  Notebook and script workflows import
from `alpha_lab` directly.

## External Replay Boundary

`alpha_lab.backtest_adapter` consumes handoff bundles and translates them into
engine inputs (vectorbt + backtrader in v1).  This remains adapter-only functionality:

- no internal strict execution simulator
- no broker/OMS/EMS/live trading stack
- unsupported engine semantics are surfaced as explicit warnings in
  `BacktestResult`

`alpha_lab.research_package` is a post-run archival layer.  It reads existing
workflow/replay/execution outputs and emits deterministic package artifacts
(`research_package.json`, `research_package.md`, optional `artifact_index.json`)
without rerunning research or replay logic.

## Tushare Data Boundary

The disciplined Tushare path is:

```text
Tushare Pro API
  -> raw snapshots
  -> standardized internal research tables
  -> workflow-compatible research inputs
  -> existing workflows / handoff / replay / research package
```

This keeps three concerns separate:

- vendor provenance and refresh timing
- internal canonical schemas
- research execution and evaluation

For details see [tushare_integration.md](tushare_integration.md).
