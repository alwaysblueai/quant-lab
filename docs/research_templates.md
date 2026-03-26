# Research Templates

`alpha_lab.research_templates` provides two canonical end-to-end workflows:

1. `run_single_factor_research_workflow(...)`
2. `run_composite_signal_research_workflow(...)`

These templates do not add new modeling engines. They orchestrate existing
modules into repeatable, auditable research campaigns.

## Single-Factor Template

### What it does

- constructs PIT-style universe / tradability / exclusion reasons
- generates one factor and applies preprocessing
- optionally applies residual neutralization
- builds labels (`forward_return`, `rankpct`, `triple_barrier`, or `trend_scanning`)
- runs factor experiment and report generation
- runs either:
  - `single_split` metadata mode
  - `purged_kfold` summary
  - `walk_forward` summary
- runs factor screening
- builds machine-readable promotion decision
- optionally writes:
  - trial log row
  - alpha registry entry
  - external handoff package

### Core inputs

- `prices`: long-form `date/asset/close` (plus optional volume fields)
- `SingleFactorWorkflowSpec`
- optional:
  - `asset_metadata`
  - `market_state`
  - `neutralization_exposures`

### Core outputs

- `SingleFactorWorkflowResult` containing:
  - processed factor and labels
  - `ExperimentResult`, `FactorReport`, `FactorSelectionReport`
  - validation summary artifacts
  - `PromotionDecision`
  - optional governance side-effect artifacts

## Composite-Signal Template

### What it does

- loads candidate signals (either from factor callables or a prebuilt long-form table)
- applies tradability masking, preprocessing, and optional neutralization
- runs redundancy/screening gates
- composes signals (`equal`, `ic`, or `icir`)
- evaluates the composite as a canonical experiment result
- computes alpha-pool diagnostics (correlation, effective breadth, clustering)
- computes research diagnostics:
  - exposure audit (if exposures provided)
  - capacity diagnostics (if trade liquidity fields available)
  - layered cost diagnostics (if trade liquidity fields available)
- builds machine-readable promotion decision
- optionally writes trial log / registry / handoff

### Core inputs

- `prices`
- `CompositeWorkflowSpec`
- exactly one of:
  - `factor_fns`
  - `candidate_signals`
- optional:
  - `asset_metadata`
  - `market_state`
  - `neutralization_exposures`
  - `exposure_data`

### Core outputs

- `CompositeWorkflowResult` containing:
  - selected/composite signals
  - screening, pool, exposure/capacity/cost diagnostics
  - `PromotionDecision`
  - optional governance side-effect artifacts

## Promotion Decision Contract

Both templates output:

- `verdict`: one of
  - `reject`
  - `needs_review`
  - `candidate_for_registry`
  - `candidate_for_external_backtest`
- `reasons`
- `blocking_issues`
- `warnings`
- `metrics` (numeric gate context)

Decision logic is explicit and threshold-driven (no hidden heuristic model).

## What is Guaranteed vs Not Guaranteed

Guaranteed:

- deterministic orchestration from explicit inputs
- explicit gate outputs with reasons and blockers
- optional artifact writing only when requested in spec flags

Not guaranteed:

- internal strict execution simulation
- OMS/EMS behavior
- broker/live trading integration
- optimization-grade portfolio engine features

The templates are research orchestration and governance layers, designed for
handoff into an external strict backtesting engine.

## Workflow CLI

Alpha Lab also provides a thin CLI wrapper over these templates:

- `run-single-factor`
- `run-composite`

Entry script:

```bash
uv run python scripts/run_research_workflow.py <command> --config-path <json> --output-dir <dir>
```

Examples:

```bash
uv run python scripts/run_research_workflow.py run-single-factor \
  --config-path examples/workflows/single_factor_workflow.json \
  --output-dir data/processed/workflows/single

uv run python scripts/run_research_workflow.py run-composite \
  --config-path examples/workflows/composite_workflow.json \
  --output-dir data/processed/workflows/composite
```

Optional side-effect overrides:

- `--write-trial-log` / `--no-write-trial-log`
- `--update-registry` / `--no-update-registry`
- `--export-handoff` / `--no-export-handoff`

### Config shape

Single-factor config:

- `data.prices_path` (required)
- `data.asset_metadata_path` (required)
- `data.market_state_path` (optional)
- `data.neutralization_exposures_path` (optional)
- `factor.name` + `factor.params` (required; built-in: `momentum`, `reversal`, `low_volatility`)
- `spec` mapped to `SingleFactorWorkflowSpec`

Composite config:

- `data.prices_path` (required)
- `data.asset_metadata_path` (required)
- `data.market_state_path` (optional)
- `data.neutralization_exposures_path` (optional)
- `data.exposure_data_path` (optional)
- exactly one of:
  - `factors` (built-in factor list)
  - `data.candidate_signals_path` (prebuilt canonical signal table)
- `spec` mapped to `CompositeWorkflowSpec`

### Produced artifacts

Each run writes one deterministic summary JSON under `--output-dir`:

- single-factor: `<experiment_name>_single_factor_workflow_summary.json`
- composite: `<experiment_name>_composite_workflow_summary.json`

Summary includes:

- workflow status
- key metrics
- `PromotionDecision`
- resolved output locations (trial log / registry / handoff when enabled)
