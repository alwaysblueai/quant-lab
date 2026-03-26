# Canonical Research Cases

This note defines two realistic end-to-end workflow cases used to validate
daily research usability of the existing template + CLI layer.

## Why these 2 cases

- **Case 1 (single-factor)** focuses on a classic short-horizon cross-sectional
  signal under realistic tradability constraints.
- **Case 2 (composite)** validates the full multi-factor governance path
  (screening, composition, pool diagnostics, exposure/capacity/cost checks).

Together they test the repository as a practical research system without
expanding into internal strict backtesting.

## Case 1: Liquidity-screened short-term reversal

- Config:
  - `examples/research_cases/configs/case1_single_reversal_liquidity_screened.json`
- Factor definition:
  - `reversal(window=5)` (negative 5-day momentum)
  - universe constraints include listing-age and ADV filters
  - preprocessing + size/industry/beta neutralization enabled
- Run:

```bash
uv run python scripts/run_research_workflow.py run-single-factor \
  --config-path examples/research_cases/configs/case1_single_reversal_liquidity_screened.json \
  --output-dir data/processed/research_cases/case1_single_reversal
```

## Case 2: Value + Quality + Momentum composite

- Config:
  - `examples/research_cases/configs/case2_composite_value_quality_momentum.json`
- Factor definitions:
  - `value_book_to_price_proxy`
  - `quality_profitability_proxy`
  - `momentum_63d`
  - ICIR-weighted composite construction
- Run:

```bash
uv run python scripts/run_research_workflow.py run-composite \
  --config-path examples/research_cases/configs/case2_composite_value_quality_momentum.json \
  --output-dir data/processed/research_cases/case2_composite_vqm
```

## What to inspect in outputs

For each case, inspect:

- `<output_dir>/*_workflow_summary.json`
  - key metrics
  - `promotion_decision.verdict`
  - blockers and warnings
- `<output_dir>/trial_log.csv` (if enabled)
- `<output_dir>/alpha_registry.csv` (if enabled)
- `<output_dir>/handoff/<artifact_name>/manifest.json` (if enabled)

Input panel for these examples is deterministic and stored under
`examples/research_cases/data/`.
