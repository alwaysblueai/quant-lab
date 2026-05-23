# Data Conventions

## Core Principles

1. No future data usage.
2. All timestamps must be explicit.
3. All datasets must be reproducible.
4. Dataset contracts must be stable and testable.

## Canonical Table Formats

### Research Features

Reusable research features for the model-factor path should be stored as a
wide table with:

| date | asset | feature_1 | feature_2 | ... |
|------|-------|-----------|-----------|-----|

- `date`: signal/feature date
- `asset`: unique asset identifier
- feature columns: numeric research inputs used by the model
- optional `known_at`: when the feature row becomes available for use

Features are inputs only. They are **not** reusable factor outputs until the
model layer converts them into canonical factor form.

For model-factor lab runs, large feature tables should be consumed as Parquet.
Specs may still point at a CSV source for convenience, but the pipeline treats
that CSV as a one-time materialization input: it creates or refreshes a sibling
`.parquet` file when needed and loads the run from the parquet path.
When the input is Parquet, the model-factor pipeline reads only the required
identity, availability, preprocessing, and selected feature columns instead of
materializing the full wide feature table.
Model-factor price panels follow the same rule: the runner always reads
`date`, `asset`, and `close`. Because the default target execution convention
is `next_open`, default model-factor runs also require `open`; additional
price columns are loaded only when the active evaluation profile can use them,
such as `high/low/volume` for tradability, `amount` or market-cap columns for
capacity diagnostics, and cached return columns for baseline comparisons.

Model-factor feature-importance diagnostics are deliberately throttled. By
default the pipeline computes importance only for the latest fitted model
version, and permutation importance is sampled with `permutation_max_rows`.
Specs can set `feature_importance.mode` to `disabled`, `latest_only`, or
`every_fit` when a run needs a different diagnostics/cost tradeoff.
The raw forward-return label frame built for model training is also retained.
For model-factor evaluation, the runner precomputes the active IC-decay
horizons and passes them as a label cache, avoiding repeated full
`forward_return` passes in core evaluation and IC-decay diagnostics.
Training-window diagnostics are aggregated at the run level: the pipeline
records one window-index summary and keeps per-date details in
`training_log.csv`, rather than emitting a diagnostics stage for every score
date.

### Factor Output

Reusable factor outputs must be long-form with:

| date | asset | factor | value |
|------|-------|--------|-------|

- `date`: observation timestamp for the factor value
- `asset`: unique asset identifier
- `factor`: factor name
- `value`: numeric factor value

There must be at most one row per (`date`, `asset`, `factor`).

### Labels / Forward Returns

Labels must be stored separately from feature outputs, but should use the same
canonical long-form schema:

| date | asset | factor | value |
|------|-------|--------|-------|

This keeps merge and validation rules consistent while still preventing
accidental leakage from mixing features and targets in the same reusable table.

### Group Returns vs. Group NAV

Two artifacts coexist for quantile portfolio analysis. They have different
semantics — pick the one that matches your question:

| File | Schema | One row = | Compoundable? |
|------|--------|-----------|---------------|
| `group_returns.csv` | `date, factor, group, group_return` | the **H-day forward return** of the bucket formed at `date` (where `H = target.horizon`) | **No.** Daily-grid rows overlap by `H-1` days; naïve `cumprod(1+x)` over consecutive rows multiplies each return ~H times. |
| `group_nav.csv` | `date, group, period_return, nav, sample_step, rebalance_step, label_horizon` | a non-overlapping NAV checkpoint (`sample_step = max(rebalance_step, label_horizon)`) | **Yes.** This is the canonical NAV path; `nav` is already the running cumprod within each group. |

Rules:

- Diagnostics that need every available observation (IC, RankIC, distributional
  plots, conditional analyses) read `group_returns.csv`.
- Anything that compounds returns (NAV curves, Sharpe/Sortino/Calmar, rolling
  drawdown, monthly aggregation) reads `group_nav.csv` — or, when only
  `group_returns.csv` is available, samples it at
  `max(rebalance_step, label_horizon)` first. The Python helper
  `alpha_lab.real_cases.artifact_enrichment.build_group_nav_table` produces the
  canonical sampled frame.
- `backtest_result.json → source_artifacts.group_nav_path` points consumers at
  the canonical file; `summary.label_horizon` and `summary.nav_rebalance_step`
  expose the sampling parameters for fallback consumers.
- Never `cumprod(1 + group_return)` over consecutive rows of
  `group_returns.csv` when `target.horizon > 1`. The artificial Q5 NAV of
  ~10⁷–10¹¹ × seen on 10-year daily panels with `H=5` is the canonical
  symptom of this mistake.

## Time Alignment Rules

- Factor values at time `t` may only use information available at or before `t`.
- For model-generated factors, all training rows used to score date `t` must satisfy `train_date < t`.
- If a feature table includes `known_at`, then `known_at <= date` must hold for every row.
- Labels must be strictly after features.
- Row-based lookbacks must be defined explicitly.
- If a factor uses per-asset history, the implementation must operate on each asset's own ordered observations.
- Never rely on union-calendar alignment unless the strategy explicitly requires it and the choice is documented.

## Missing Data

- Never silently forward-fill research inputs.
- Explicitly document:
  - fill method
  - dropped rows
  - interpolation
- Missing observations for one asset must not change the lookback definition for another asset.

## Factor Construction Rules

Every factor must specify:

- hypothesis
- lookback window
- horizon of intended use
- whether the computation is cross-sectional or time-series
- timestamp alignment
- leakage risk

For the model-factor path, this still resolves to one canonical factor output:

- data layer: prepares research features
- model layer: maps features to a score
- evaluation layer: treats the score exactly like any other factor

## Merge Rules

- Always merge explicitly on (`date`, `asset`).
- Include `factor` when combining stacked factor outputs.
- Never rely on index alignment.
- Always check row counts before and after merges.

## Storage

- raw data -> `data/raw/`
- processed data -> `data/processed/`
- never overwrite raw data

## Anti-Patterns

- using future returns in features
- mixing features and labels in the same reusable factor table
- mixing different frequencies without documented alignment
- implicit timezone conversion
- silent NaN filling

## Daily Price-Volume Workflow Scope

The default research workflow is **daily price-volume single-factor research**.
The following invariants hold for that workflow; changing any of them is an
opt-in decision that must be documented on the case spec.

### Ingest scope

`alpha-lab data ingest tushare core` and `alpha-lab data update tushare core`
both default to `--mode daily`, which ingests only:

- `daily_bars` (OHLCV, amount, pre_close)
- `adj_factor` (for qfq/hfq reconstruction)
- `stk_limit`, `suspend_status`, `st_name_events` (tradability signals)
- `index_membership` (universe construction)
- `moneyflow`, `daily_basic` (PE/PB/turnover_rate — PV-adjacent)
- `instruments` (reference)

Skipped under `--mode daily`: `financial_indicator` (ROE/TTM fundamentals),
`industry_classification`. Use `--mode fundamental` or `--mode full` only when
a case requires them.

### Price adjustment invariant

The slice presets export prices with `adjustment="qfq"` by default. Case
pipelines consume qfq-adjusted prices as-is; they do **not** call
`adjust_for_splits` or `adjust_for_dividends` at runtime. The runtime guard is
`detect_unadjusted_splits`, which flags day-over-day moves beyond A-share
daily limits. If a case deliberately slices with `adjustment="raw"`, that case
is responsible for wiring its own adjustment step.

### Neutralization is opt-in

`NeutralizationSpec.enabled` defaults to `False`. A PV-only case should leave
it off unless the research question requires size/industry neutralization; in
that case the case also opts into the fundamental/industry ingest scope above
and supplies an `exposures_path`.

### Out-of-scope modules for PV-only research

The following subsystems exist but are not registered into any default
workflow; they are invoked only by their dedicated CLIs:

- `real_cases.composite` (composite/multi-factor case CLI)
- `real_cases.model_factor` (ML-trained factor CLI)
- `data_quality.pit_fundamentals` (report-date PIT alignment — fundamental-only)
- `risk_model.barra` (risk attribution / pure-alpha extraction)

A PV-only session should not touch these paths.
