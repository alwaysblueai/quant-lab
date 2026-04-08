# Research Integrity Layer

## Purpose

`alpha_lab.research_integrity` is a core Level 1/2 layer for research
correctness. It enforces:

- point-in-time (PIT) / as-of discipline
- anti-leakage constraints
- cross-timeframe timestamp correctness
- explicit factor/label temporal ordering
- incomplete-bar and early-visibility checks

This layer exists to protect research validity, not to simulate live execution.

## Core Scope (Level 1/2)

Core modules:

- `asof.py`
  - `asof_join_frame(...)`
  - `validate_asof_monotonicity(...)`
  - `validate_forward_fill_lag(...)`
  - metadata helpers (`TimeSemanticsMetadata`, attach/read helpers)
- `leakage_checks.py`
  - `check_no_future_dates_in_input(...)`
  - `check_factor_label_temporal_order(...)`
  - `check_asof_inputs_not_after_signal_date(...)`
  - `check_cross_section_transform_scope(...)`
  - bar/cross-timeframe guards:
    - `check_closed_bar_required_before_signal_use(...)`
    - `check_incomplete_bar_not_used(...)`
    - `check_bar_close_known_before_order_submission(...)`
    - `check_higher_timeframe_feature_not_available_early(...)`
    - `check_multitimeframe_alignment(...)`
    - `check_intraday_to_daily_alignment(...)`
    - `check_daily_feature_asof_intraday(...)`
    - `check_same_bar_close_execution_conflict(...)`
    - `check_signal_execution_gap_is_respected(...)`
- `contracts.py`
  - `IntegrityCheckResult`, `IntegrityReport`, `IntegrityReportSummary`
- `reporting.py`
  - `build_integrity_report(...)`
  - report export/render helpers
- `exceptions.py`
  - `raise_on_hard_failures(...)`

Status taxonomy:

- `pass` / `info`: check passed
- `warn` / `warning`: degraded semantics or limited coverage
- `fail` / `error`: hard violation

## Pipeline Integration

Current core integrations:

- `run_factor_experiment` runs core integrity checks and attaches:
  - `integrity_checks`
  - `integrity_report`
- `run_walk_forward_experiment` returns:
  - fold-level integrity reports
  - one aggregate integrity report
- real-case single-factor/composite pipelines export:
  - `integrity_report.json`
  - `integrity_report.md`

## Explicit Non-Goals

This core layer is not:

- an order-fill simulator
- a market microstructure simulator
- an execution replay engine
- implementability certification

## Experimental Future Level 3

Replay semantic-audit code is preserved but demoted from core defaults:

- `alpha_lab.research_integrity.semantic_consistency`
- replay/adapter integrations in `alpha_lab.experimental_level3`

Use these only when running explicit future-facing Level 3 experiments.
