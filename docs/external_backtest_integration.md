# External Backtest Adapter (v1)

## Purpose

`alpha_lab.backtest_adapter` is a thin translation layer from the handoff bundle
(`schema_version = 2.0.0`) to an external backtesting engine input path.

This layer:
- loads and validates handoff bundles
- builds deterministic target weights from bundle contracts
- maps execution assumptions into engine-specific parameters
- runs a replay path via `vectorbt` and `backtrader`
- exports replay artifacts including `adapter_run_metadata.json` for audit

This layer does **not** implement an internal strict backtesting engine.

## Module Responsibilities

- `schema.py`: typed adapter objects (`BacktestInputBundle`, `BacktestRunConfig`, `BacktestResult`)
- `loader.py`: bundle loading + parsing via existing `validate_handoff_artifact(...)`
- `validators.py`: in-memory bundle and price-panel checks
- `target_weights.py`: engine-agnostic target-weight construction
- `vectorbt_adapter.py`: fast replay translation with explicit approximations
- `backtrader_adapter.py`: stricter execution-aware translation (lot/tradability/policy gates)
- `base.py`: engine dispatch + basic summary/export helpers

The target-weight logic is intentionally outside engine adapters. Both vectorbt
and Backtrader adapters reuse the same engine-agnostic target-weight layer.

## Supported v1 Semantics

### Target-weight construction

Supported intent methods:
- `rank_topk_equal` (long-only top-k equal weight)
- `rank_topbottom_equal` (long-short top/bottom equal weight)
- `zscore_proportional` (rank/score weighting path)

Supported controls:
- long-only and long-short
- `top_k` / `bottom_k`
- `max_weight`
- `gross_limit` / `net_limit` normalization
- `cash_buffer`
- tradability masking (`trade_when_not_tradable=False` is respected)

### Execution mapping (vectorbt v1)

Implemented:
- execution delay via shifted target weights
- commission mapping for `commission_model="bps"` using `BacktestRunConfig.commission_bps`
- slippage mapping for `slippage_model="fixed_bps"` using `BacktestRunConfig.slippage_bps`

### Execution mapping (Backtrader v1)

Backtrader adapter focuses on stricter execution-aware handling:
- execution delay in bars
- lot-size rounding (`round_to_lot`)
- non-tradable/suspension/price-limit order gating
- same-day reentry block when `allow_same_day_reentry=false`
- target-weight to order-size translation with deterministic ordering

## Explicit Warning Policy

If an engine adapter cannot represent an execution assumption faithfully, the adapter
does not silently hide this. It emits explicit `AdapterWarning` entries in
`BacktestResult.warnings`.

Examples that generate warnings:
- unsupported/approximated fill-price rules (`vwap_next_bar`, missing open prices for `next_open`)
- unsupported commission model (`flat`, `per_share`)
- unsupported slippage model (`spread_plus_impact_proxy` uses fixed-bps proxy)
- policy approximations such as `defer_trade` treated as skip in v1
- missing exclusion detail when tradability policy requires reason-level interpretation

Consumers should treat these warnings as replay limitations and document them in
research conclusions.

## Replay Output Artifacts

When `BacktestRunConfig.output_dir` is set and `export_summary=True`, the adapter
writes:
- `backtest_summary.json`
- `adapter_run_metadata.json`
- `portfolio_returns.csv`, `equity_curve.csv`, `turnover.csv` (if `export_series=True`)
- `target_weights.csv`, `executed_weights.csv` (if `export_target_weights=True`)

`adapter_run_metadata.json` includes:
- adapter version
- engine and engine version
- bundle schema/fingerprint references
- portfolio + execution contracts consumed
- mapped replay assumptions actually used
- emitted warning list

## Execution Impact Reporting

`alpha_lab.execution_impact_report` provides a research-facing post-replay
diagnostic layer. It summarizes how execution constraints changed realized
replay behavior relative to target portfolio intent.

Core APIs:
- `load_execution_artifacts(run_path)`
- `build_execution_impact_report(run_path, comparison_run_path=None)`
- `export_execution_impact_report(report, output_dir=None)`

Primary report outputs:
- `execution_impact_report.json`
- optional `execution_impact_by_reason.csv`
- optional `execution_impact_timeseries.csv`

Coverage:
- skipped-order reason distribution (normalized reason taxonomy)
- dominant execution blocker identification
- target-vs-realized weight deviation summary
- target turnover vs realized turnover context (when artifacts are available)
- machine-readable flags (`high_execution_deviation`, `price_limit_sensitive`, etc.)
- optional vectorbt vs Backtrader comparison summary

Graceful degradation policy:
- missing optional replay artifacts do not hard-fail report generation
- unavailable metrics are explicitly listed in `unavailable_metrics`
- `missing_artifacts` is recorded in the JSON report
- metrics are not silently dropped

Interpretation boundary:
- this report is descriptive research diagnostics
- it is not causal attribution and not a strategy optimization report

## Example

```python
from alpha_lab.backtest_adapter import (
    BacktestRunConfig,
    load_backtest_input_bundle,
    run_external_backtest,
)

bundle = load_backtest_input_bundle("data/processed/handoff/my_bundle")
result = run_external_backtest(
    bundle,
    config=BacktestRunConfig(
        engine="backtrader",
        price_df=prices_df,  # long-form: date, asset, close (and optional open)
        close_column="close",
        open_column="open",
        commission_bps=10.0,
        slippage_bps=5.0,
        output_dir="data/processed/backtest_runs/my_bundle",
    ),
)

print(result.summary)
for warning in result.warnings:
    print(warning.code, warning.message)

from alpha_lab.execution_impact_report import (
    build_execution_impact_report,
    export_execution_impact_report,
)

impact = build_execution_impact_report("data/processed/backtest_runs/my_bundle")
export_execution_impact_report(impact)
```

## Adapter Comparison

- `vectorbt` adapter:
  - faster replay and protocol sanity checks
  - more assumptions approximated, with explicit warnings
- `backtrader` adapter:
  - stricter execution-aware translation for lot/tradability/policy handling
  - still v1: not a full market microstructure simulator

## Extension Path

- Keep engine-agnostic transformation logic in `target_weights.py` and `base.py`.
- Extend stricter execution semantics in `backtrader_adapter.py` only.
- Do not move research/factor/portfolio-construction logic into engine adapters.
