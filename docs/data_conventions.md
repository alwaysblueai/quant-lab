# Data Conventions

## Core Principles

1. No future data usage.
2. All timestamps must be explicit.
3. All datasets must be reproducible.
4. Dataset contracts must be stable and testable.

## Canonical Table Formats

### Research Bundle Components

`alpha_lab.research_contracts.ResearchBundle` separates research inputs into:
- `prices`
- `factors`
- `labels`
- `universe`
- `tradability`
- `metadata`
- optional `snapshot` descriptor (`dataset_id`, hash/version notes)

Validation rules enforced by contract utilities:
- required columns and dtypes
- uniqueness of canonical keys
- parseable timestamps
- monotonic price history per asset (when required)
- alignment of `(date, asset)` support between `universe` and `tradability`

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

For event-style labels, the repository also supports a unified label schema:

| date | asset | label_name | label_type | label_value | event_start | event_end | trigger | realized_horizon | confidence |
|------|-------|------------|------------|-------------|-------------|-----------|---------|------------------|------------|

Used by:
- `regression_forward_label`
- `rankpct_label`
- `triple_barrier_labels`
- `trend_scanning_labels`

`label_value` semantics depend on `label_type`:
- regression: continuous forward-return style target
- ranking: percentile rank target
- event_classification: discrete event class (e.g., -1/0/1)

## Time Alignment Rules

- Factor values at time `t` may only use information available at or before `t`.
- Labels must be strictly after features.
- Row-based lookbacks must be defined explicitly.
- If a factor uses per-asset history, the implementation must operate on each asset's own ordered observations.
- Never rely on union-calendar alignment unless the strategy explicitly requires it and the choice is documented.

### Explicit Timing Metadata

Every experiment should record a timing contract via `DelaySpec`:
- `decision_timestamp`
- `execution_delay_periods`
- `return_horizon_periods`
- `label_start_offset_periods`
- `label_end_offset_periods`
- optional `purge_periods` and `embargo_periods`

This metadata is audit metadata, not an execution simulator.

## Universe / Tradability / Exclusions

Research-sample construction is explicit and PIT-safe:
- `universe`: `[date, asset, in_universe]`
- `tradability`: `[date, asset, is_tradable]`
- `exclusion_reasons`: `[date, asset, reason, detail]`

Typical exclusion reasons include:
- listing age / missing listing date
- ST filter
- halted trading
- limit-locked non-executable day
- minimum ADV filter

Universe and tradability must share the same `(date, asset)` support.

## Sample Weights

Sample weights are stored separately from factors/labels:

| date | asset | sample_weight |
|------|-------|---------------|

Optional component columns may be tracked in analysis artifacts:
- uniqueness weight
- return-magnitude weight
- time-decay weight
- confidence weight

Weights are non-negative and typically normalized.

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
