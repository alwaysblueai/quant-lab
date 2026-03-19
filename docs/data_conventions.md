# Data Conventions

## Core Principles

1. No future data usage.
2. All timestamps must be explicit.
3. All datasets must be reproducible.
4. Dataset contracts must be stable and testable.

## Canonical Table Formats

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

## Time Alignment Rules

- Factor values at time `t` may only use information available at or before `t`.
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
