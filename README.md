# Alpha Lab

Alpha Lab is a research-grade factor research system for cross-sectional equity alpha studies.
It is intentionally scoped to research workflows and diagnostics, not to trading execution.

## Scope

Implemented:
- canonical research data contracts (`prices`, `factors`, `labels`, `universe`, `tradability`)
- PIT-safe research universe construction with tradability masks and exclusion reasons
- richer labeling (`forward`, `rankpct`, `triple_barrier`, `trend_scanning`)
- purged/embargoed validation utilities and sample-weight framework
- factor experiments with explicit timing metadata
- walk-forward OOS evaluation
- IC/Rank-IC, quantile, turnover, decay, and coverage diagnostics
- residual neutralization (size/industry/beta) and factor-screening diagnostics
- multiple-testing utilities (Bonferroni/Sidak/effective trial count heuristics)
- feature-importance scaffolds (MDI/MDA/SFI + correlation clusters)
- experiment metadata, provenance, and trial logging
- alpha-pool governance primitives (alpha registry, composite signals, breadth diagnostics)
- research-side rebalance cadence recommendation
- research executability diagnostics (capacity, exposure audit, layered friction proxies)
- deterministic external-backtest handoff artifact export
- external backtest adapter layer (`backtest_adapter`) with vectorbt + backtrader v1 paths
- disciplined Tushare ingestion layer (`data_sources`) with raw snapshots, standardized tables, and workflow-compatible A-share bundles
- quant-knowledge experiment card export (`50_experiments/`)
- canonical research package export (`research_package.json` + concise markdown summary)

Explicit non-goals:
- event-driven backtesting engine
- OMS/EMS, broker adapters, paper/live trading
- production orchestration and cloud infrastructure

## Quick Start

Requirements:
- Python 3.12+
- `uv`

Install:

```bash
uv sync --all-extras
```

Run checks:

```bash
make check
```

If cache permissions are restricted:

```bash
UV_CACHE_DIR=/tmp/uv-cache make check
```

## Repository Layout

- `src/alpha_lab`: reusable research modules
- `tests`: unit/regression tests
- `docs`: architecture/contracts/manual
- `data/raw`: immutable raw inputs
- `data/processed`: derived artifacts (summaries, registries, trial logs)
- `scripts`: CLI entrypoints
- `notebooks`: exploratory work

## Core Contracts

Price panel:
- required columns: `date`, `asset`, `close`
- no duplicate `(date, asset)`
- `close > 0`, no NaN prices

Canonical long-form signal tables (`factors` and `labels`):
- required columns: `date`, `asset`, `factor`, `value`
- unique key: `(date, asset, factor)`
- `value` numeric

Reusable factor rule:
- factor value at timestamp `t` may only use information available at or before `t`

Labels rule:
- labels are stored separately from factors
- forward labels are generated explicitly from price data and horizon

## Timing and Validation Metadata

`DelaySpec` records alignment assumptions:
- decision timestamp
- execution delay
- return horizon
- label window offsets
- purge/embargo metadata

`ExperimentMetadata` records governance fields:
- hypothesis / question / factor spec
- dataset id/hash
- trial id / trial count
- validation scheme metadata
- assumptions, caveats, warnings, verdict
- runtime environment metadata

## Main APIs

Single experiment:

```python
from alpha_lab.experiment import run_factor_experiment

result = run_factor_experiment(
    prices,
    factor_fn,
    horizon=5,
    n_quantiles=5,
    train_end="2024-12-31",
    test_start="2025-01-01",
    generate_factor_report=True,
)
```

Walk-forward experiment:

```python
from alpha_lab.walk_forward import run_walk_forward_experiment

wf = run_walk_forward_experiment(
    prices,
    factor_fn,
    train_size=252,
    test_size=63,
    step=63,
    val_size=21,
    purge_periods=1,
    embargo_periods=1,
)
```

Canonical end-to-end templates:

```python
from alpha_lab.factors.momentum import momentum
from alpha_lab.research_templates import (
    SingleFactorWorkflowSpec,
    run_single_factor_research_workflow,
)

wf = run_single_factor_research_workflow(
    prices,
    spec=SingleFactorWorkflowSpec(
        experiment_name="momentum_template",
        factor_fn=lambda p: momentum(p, window=20),
        validation_mode="purged_kfold",
        export_handoff=True,
        handoff_output_dir="data/processed/handoff",
    ),
    asset_metadata=asset_metadata_df,
    market_state=market_state_df,
)
```

Workflow CLI (thin wrapper over the same templates):

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

Tushare ingestion/build path:

```bash
uv run python scripts/tushare_pipeline.py fetch-snapshots \
  --snapshot-name ashare_202401_202412 \
  --start-date 20240101 \
  --end-date 20241231

uv run python scripts/tushare_pipeline.py build-standardized \
  --snapshot-dir data/raw/tushare/ashare_202401_202412

uv run python scripts/tushare_pipeline.py build-cases \
  --standardized-dir data/processed/tushare_standardized/ashare_202401_202412
```

Financial indicator extraction is endpoint-specific:
`fina_indicator_vip(period=...)` is preferred, with automatic per-stock
`fina_indicator(ts_code=..., start_date=..., end_date=...)` fallback when VIP
is unavailable. If both fail, the pipeline records graceful degradation and the
single-factor reversal case remains runnable.

This produces workflow-compatible research inputs for the canonical real-data
cases:
- liquidity-screened short-term reversal
- value + quality + momentum composite

## Research Diagnostics

`FactorReport` includes:
- IC and Rank-IC series
- IC mean/std/ICIR/t-stat summaries
- rolling IC
- coverage diagnostics
- quantile monotonicity diagnostics
- turnover diagnostics
- horizon decay profile and half-life estimate

Additional diagnostics/governance modules:
- `research_universe.construct_research_universe`
- `purged_validation.purged_kfold_split`
- `sample_weights.build_sample_weights`
- `neutralization.neutralize_signal`
- `factor_selection.screen_factors`
- `composite_signals.compose_signals`
- `alpha_pool_diagnostics.alpha_pool_diagnostics`
- `capacity_diagnostics.run_capacity_diagnostics`
- `exposure_audit.run_exposure_audit`
- `research_costs.layered_research_costs`

## Governance and Reporting

- Experiment card export to quant-knowledge:

```python
from alpha_lab.reporting import export_experiment_card

path = export_experiment_card(result, name="momentum-5d-Ashare")
```

- Trial logging:

```python
from alpha_lab.trial_log import trial_row_from_result, append_trial_log

row = trial_row_from_result(result, experiment_name="momentum_h5")
append_trial_log(row)
```

## External Backtest Handoff

```python
from alpha_lab.handoff import export_handoff_artifact

export = export_handoff_artifact(
    result,
    output_dir="data/processed/handoff",
    universe_df=universe_df,
    tradability_df=tradability_df,
    include_label_snapshot=True,
)
```

This writes a deterministic package (`manifest.json`, signal/universe/tradability
snapshots, timing metadata, experiment metadata, validation context, dataset
fingerprint, `portfolio_construction.json`, and `execution_assumptions.json`)
for strict external backtest replay.

External adapter replay (v1):

```python
from alpha_lab.backtest_adapter import (
    BacktestRunConfig,
    load_backtest_input_bundle,
    run_external_backtest,
)

bundle = load_backtest_input_bundle("data/processed/handoff/my_bundle")
result = run_external_backtest(bundle, config=BacktestRunConfig(price_df=prices_df))
```

Use `engine="vectorbt"` for fast replay and protocol checks, or
`engine="backtrader"` for stricter execution-aware semantics (lot-size and
tradability/policy gating) with explicit warnings for remaining approximations.

Execution impact diagnostics from replay artifacts:

```python
from alpha_lab.execution_impact_report import build_execution_impact_report

report = build_execution_impact_report("data/processed/backtest_runs/my_bundle")
print(report.dominant_execution_blocker)
```

Research package build from completed outputs:

```python
from alpha_lab.research_package import build_research_package, export_research_package

package = build_research_package("data/processed/research_cases/case1_single_reversal")
export_research_package(package, output_dir="data/processed/research_cases/case1_single_reversal")
```

Or CLI:

```bash
uv run python scripts/build_research_package.py \
  --case-output-dir data/processed/research_cases/case1_single_reversal
```

The execution impact report is research-facing and descriptive. It explains
execution constraints and target-vs-realized deviations; it is not a trading
performance optimization report.

## Documentation

- [Architecture](docs/architecture.md)
- [Data Conventions](docs/data_conventions.md)
- [System Manual](docs/system_manual.md)
- [Developer Guide](docs/developer_guide.md)
- [Handoff Artifact](docs/handoff_artifact.md)
- [External Backtest Integration](docs/external_backtest_integration.md)
- [Research Package](docs/research_package.md)
- [Research Templates](docs/research_templates.md)
- [Canonical Research Cases](docs/research_case_examples.md)
