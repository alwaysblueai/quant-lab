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

If you use the Tushare ingestion path, also provide a Tushare Pro token via
`TUSHARE_TOKEN` or `--token`.  This is only required for the extraction step,
not for downstream research workflows.

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

For full research-input contracts (prices/factors/labels/universe/tradability),
see `alpha_lab.research_contracts` and `ResearchBundle.validate()`.

---

## Tushare Real-Data Ingestion

Alpha Lab supports an offline-first Tushare ingestion path for A-share data.
The design intent is to fetch once, standardize once, and then run research on
stored files.

Pipeline:

```text
fetch raw snapshots -> standardize vendor tables -> build workflow inputs -> run existing workflows
```

Script entrypoint:

```bash
uv run python scripts/tushare_pipeline.py <command> ...
```

Commands:

- `fetch-snapshots`
- `build-standardized`
- `build-cases`

The resulting case configs feed the existing `scripts/run_research_workflow.py`
entrypoints. They do not introduce a separate workflow engine.

See [tushare_integration.md](tushare_integration.md) for endpoint coverage,
fallback behavior, and PIT notes.

---

## Core API

### run_factor_experiment

Connects all evaluation modules (IC, quantile returns, long-short, turnover,
and optionally portfolio simulation) into a single call. It also attaches
timing and governance metadata to the `ExperimentResult`.

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
result.ic_df                 # [date, ic]
result.rank_ic_df            # [date, rank_ic]
result.quantile_returns_df   # [date, quantile, mean_return]
result.long_short_df         # [date, long_short_return]
result.factor_df             # full-sample factor values
result.label_df              # full-sample forward-return labels
result.delay_spec            # DelaySpec alignment assumptions
result.label_metadata        # LabelMetadata serializable label window
result.metadata              # ExperimentMetadata governance object
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

**Richer research diagnostics** (optional):

```python
result = run_factor_experiment(
    prices,
    lambda p: momentum(p, window=20),
    horizon=5,
    generate_factor_report=True,
)

result.factor_report  # FactorReport (rolling IC, coverage, monotonicity, decay)
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
    purge_periods=1,   # metadata-only validation/timing controls
    embargo_periods=1,
)

# Fold-level summary
wf.fold_summary_df          # one row per fold
wf.per_fold_results         # list of ExperimentResult, one per fold
wf.validation_spec          # WalkForwardValidationSpec
wf.fold_windows_df          # explicit fold windows (train/val/test boundaries)

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

### Advanced research modules

PIT sample construction:

```python
from alpha_lab.research_universe import construct_research_universe, ResearchUniverseRules

u = construct_research_universe(
    prices,
    asset_metadata=asset_metadata_df,
    market_state=market_state_df,
    rules=ResearchUniverseRules(min_listing_age_days=60, min_adv=5_000_000),
)
```

Rich labels and sample weights:

```python
from alpha_lab.labels import triple_barrier_labels
from alpha_lab.sample_weights import build_sample_weights

lbl = triple_barrier_labels(prices, horizon=10, pt_mult=1.0, sl_mult=1.0)
w = build_sample_weights(
    lbl.labels,
    sample_id_col="asset",
    decision_col="date",
    return_col="label_value",
    confidence_col="confidence",
    half_life_periods=20,
)
```

Purged validation:

```python
from alpha_lab.purged_validation import purged_kfold_split, purged_fold_summary

folds = purged_kfold_split(events_df, n_splits=5, embargo_periods=2)
diag = purged_fold_summary(folds)
```

Factor governance:

```python
from alpha_lab.neutralization import neutralize_signal
from alpha_lab.factor_selection import screen_factors
from alpha_lab.multiple_testing import apply_multiple_testing_to_trial_log
from alpha_lab.feature_importance import build_feature_importance_report
```

Alpha-pool and executability diagnostics:

```python
from alpha_lab.composite_signals import compose_signals
from alpha_lab.alpha_pool_diagnostics import alpha_pool_diagnostics
from alpha_lab.capacity_diagnostics import run_capacity_diagnostics
from alpha_lab.exposure_audit import run_exposure_audit
from alpha_lab.research_costs import layered_research_costs
```

---

### End-to-end research templates

`alpha_lab.research_templates` exposes canonical orchestration entrypoints for
repeatable daily research campaigns.

```python
from alpha_lab.factors.momentum import momentum
from alpha_lab.research_templates import (
    SingleFactorWorkflowSpec,
    run_single_factor_research_workflow,
)

result = run_single_factor_research_workflow(
    prices,
    spec=SingleFactorWorkflowSpec(
        experiment_name="momentum_template",
        factor_fn=lambda p: momentum(p, window=20),
        validation_mode="purged_kfold",
        append_trial_log=True,
        export_handoff=True,
        handoff_output_dir="data/processed/handoff",
    ),
    asset_metadata=asset_metadata_df,
    market_state=market_state_df,
)

result.decision.verdict
# reject / needs_review / candidate_for_registry / candidate_for_external_backtest
```

For the full single-factor and composite-template contract, see
`docs/research_templates.md`.

---

### Research package (canonical end-product)

`alpha_lab.research_package` is a post-processing layer that turns completed
case outputs into one standardized package artifact.

```python
from alpha_lab.research_package import build_research_package, export_research_package

package = build_research_package("data/processed/research_cases/case1_single_reversal")
export_research_package(package, output_dir="data/processed/research_cases/case1_single_reversal")
```

Package outputs:

- `research_package.json` (complete machine-readable payload)
- `research_package.md` (concise review-oriented summary)
- optional `artifact_index.json` (explicit artifact path map)

Package verdict logic is explicit and transparent. It is assembled from:

- workflow `PromotionDecision`
- replay warnings/limitations
- execution-impact flags
- explicit missing-artifact records

See `docs/research_package.md` for field-level details.

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
    export_experiment_card,
)

summary = summarise_experiment_result(result, cost_rate=0.001)
export_summary_csv(summary, "output/reports/momentum_5d.csv")
md = to_obsidian_markdown(result, title="Momentum 5d OOS", cost_rate=0.001)
```

#### export_experiment_card

Writes a structured experiment note to `{vault}/50_experiments/Exp - YYYYMM - {name}.md`
using the quant-knowledge frontmatter schema.

```python
path = export_experiment_card(result, name="momentum-5d-Ashare")
# vault defaults to OBSIDIAN_VAULT_PATH from config / env var
# returns the resolved Path of the written file

# Explicit vault and overwrite:
path = export_experiment_card(
    result,
    name="momentum-5d-Ashare",
    vault_path="/path/to/quant-knowledge",
    overwrite=True,
)
```

**Behaviour:**
- The vault root must already exist (`FileNotFoundError` if not).
- The `50_experiments/` subdir is created automatically if absent.
- Default is safe: raises `FileExistsError` if the card already exists.
  Pass `overwrite=True` to replace an existing card intentionally.
- `name` must be non-empty and must not contain path separators.
- Returns the resolved `Path` of the written file.

**Generated vs manual sections:**
Setup, Results, and YAML frontmatter are auto-generated from the
`ExperimentResult` and must not be edited manually.  The note includes a
visible notice to that effect.  Interpretation, Next Steps, Open Questions,
and Notes are placeholders for researcher completion.

When metadata is available, exported cards include:
- `dataset_id`, `dataset_hash`, and `trial_id` in frontmatter
- timing assumptions (`DelaySpec`) in Setup
- validation scheme and runtime versions in Setup

**Vault path resolution order:**
`vault_path` argument → `OBSIDIAN_VAULT_PATH` env var → config default
(`/mnt/c/quant/vault/quant-knowledge`).  An empty or whitespace env var is
treated as "not configured" and falls through to the default.

---

### Trial Log

Append-only per-trial accounting for anti-p-hacking hygiene:

```python
from alpha_lab.trial_log import append_trial_log, trial_row_from_result

row = trial_row_from_result(result, experiment_name="momentum_h5")
append_trial_log(row)  # default: data/processed/trial_log.csv
```

---

### External Backtest Handoff

Use `alpha_lab.handoff` to export deterministic research-to-backtest artifacts:

```python
from alpha_lab.handoff import export_handoff_artifact
from alpha_lab.handoff import PortfolioConstructionSpec, ExecutionAssumptionsSpec

export = export_handoff_artifact(
    result,
    output_dir="data/processed/handoff",
    universe_df=universe_df,
    tradability_df=tradability_df,
    include_label_snapshot=True,
    portfolio_construction=PortfolioConstructionSpec(
        signal_name="momentum_20d",
        construction_method="top_bottom_k",
        top_k=20,
        bottom_k=20,
        weight_method="rank",
    ),
    execution_assumptions=ExecutionAssumptionsSpec(
        fill_price_rule="next_open",
        execution_delay_bars=1,
    ),
)

export.artifact_path
export.manifest_path
export.dataset_fingerprint
```

Walk-forward fold export:

```python
from alpha_lab.handoff import export_walk_forward_handoff_artifacts

exports = export_walk_forward_handoff_artifacts(
    wf_result,
    output_dir="data/processed/handoff",
    universe_df=universe_df,
    tradability_df=tradability_df,
    fold_ids=[0, 1, 2],
)
```

See [handoff_artifact.md](handoff_artifact.md) for schema and versioning policy.

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
result.provenance.python_version
result.provenance.pandas_version
result.provenance.numpy_version

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
