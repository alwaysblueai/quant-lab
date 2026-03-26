# External Backtest Handoff Bundle

## Purpose

The handoff bundle is a deterministic, audit-friendly export package from
alpha-lab research outputs to an external strict backtesting engine.

Schema `2.0.0` upgrades the old research-only artifact into a full
three-part protocol bundle:

1. research artifact
2. portfolio construction contract
3. execution assumptions contract

This module does **not** run internal strict backtesting and does **not**
simulate execution.

## Schema Version

Current export schema version: `2.0.0`
(`alpha_lab.handoff.HANDOFF_SCHEMA_VERSION`).

Backward compatibility:
- `validate_handoff_artifact(...)` accepts both `1.0.0` and `2.0.0` bundles.
- `export_handoff_artifact(...)` always writes `2.0.0`.

Versioning policy:
- patch: non-breaking metadata changes
- minor: backward-compatible additions
- major: protocol-breaking layout/contract changes

Consumers should always check `manifest.json.schema_version`.

## Bundle Layout (Schema 2.0.0)

Required files:
- `manifest.json`
- `signal_snapshot.csv`
- `universe_mask.csv`
- `tradability_mask.csv`
- `timing.json`
- `experiment_metadata.json`
- `validation_context.json`
- `dataset_fingerprint.json`
- `portfolio_construction.json`
- `execution_assumptions.json`

Optional files:
- `label_snapshot.csv`
- `exclusion_reasons.csv`

## Three-Part Contract

### 1. Research Artifact

Research artifact files define what was produced and under what research context:
- signal snapshot, universe mask, tradability mask
- timing metadata
- experiment and validation metadata
- dataset fingerprint

Guaranteed:
- deterministic ordering and serialization
- explicit timing assumptions
- reproducible dataset fingerprint

### 2. Portfolio Construction Contract

`portfolio_construction.json` is represented by
`PortfolioConstructionSpec` and includes:
- `construction_method`
- `signal_name`
- `rebalance_frequency`, `rebalance_calendar`
- `long_short`, `top_k`, `bottom_k`
- `weight_method`, `weight_clip`
- `max_weight`, `min_weight`
- `gross_limit`, `net_limit`
- `cash_buffer`
- `neutralization_required`
- `post_construction_constraints`

### 3. Execution Assumptions Contract

`execution_assumptions.json` is represented by
`ExecutionAssumptionsSpec` and includes:
- `fill_price_rule`
- `execution_delay_bars`
- `commission_model`
- `slippage_model`
- `lot_size_rule`, `lot_size`
- `cash_buffer`
- `partial_fill_policy`
- `suspension_policy`
- `price_limit_policy`
- `trade_when_not_tradable`
- `allow_same_day_reentry`

## Manifest (Schema 2.0.0)

`manifest.json` now includes:
- `bundle_type = "alpha_lab_handoff_bundle"`
- `schema_version`
- identifiers (`artifact_name`, `experiment_id`, `fold_id`)
- `dataset_fingerprint`
- `bundle_components` (research/portfolio/execution component registry)
- export environment metadata
- per-file hash registry (`files`)

## Validation Behavior

`validate_handoff_artifact(path)` verifies:
- required files by schema version
- manifest file registry and per-file hash integrity
- core table columns and universe/tradability key alignment
- timing payload completeness
- dataset fingerprint consistency
- portfolio and execution spec validation
- cross-file consistency checks:
  - `portfolio_construction.signal_name` vs `signal_snapshot.csv`
  - non-contradictory delay assumptions
  - cash buffer consistency between portfolio/execution contracts

## External Adapter Consumption Pattern

1. Load `manifest.json` and check schema compatibility.
2. Run bundle validation (`validate_handoff_artifact` equivalent in adapter).
3. Read research artifact tables and metadata.
4. Apply portfolio construction rules from `portfolio_construction.json`.
5. Apply execution assumptions from `execution_assumptions.json`.
6. Replay in the external strict backtest engine.
