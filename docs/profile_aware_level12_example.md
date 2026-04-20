# Profile-Aware Level 1/2 Example Workflow

This example is a compact end-to-end Level 1/2 workflow that runs one
single-factor case through:

1. Level 1 evaluation (diagnostics, verdict, uncertainty, rolling stability)
2. Campaign triage
3. Level 2 promotion gate
4. Level 2 portfolio validation

The same case is run under two profiles to make profile effects explicit:
`exploratory_screening` and `default_research`.

## Commands

List available Level 1/2 profiles:

```bash
alpha-lab profiles
```

Run the profile-aware example:

```bash
uv run --no-sync --frozen python scripts/run_profile_aware_level12_example.py \
  --output-root-dir dist/examples/profile_aware_level12 \
  --profiles exploratory_screening default_research
```

If your environment requires writable uv cache (for example sandbox/WSL), use:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen python scripts/run_profile_aware_level12_example.py \
  --output-root-dir dist/examples/profile_aware_level12 \
  --profiles exploratory_screening default_research
```

## Output Locations

Example root:

- `dist/examples/profile_aware_level12/profile_aware_single_factor_case.json`
- `dist/examples/profile_aware_level12/profile_comparison.md`
- `dist/examples/profile_aware_level12/profile_comparison.json`

Per-profile run outputs:

- `dist/examples/profile_aware_level12/runs/exploratory_screening/profile_aware_bp_single_factor/`
- `dist/examples/profile_aware_level12/runs/default_research/profile_aware_bp_single_factor/`

Each run directory contains standard Level 1/2 artifacts:

- `run_manifest.json`
- `metrics.json`
- `summary.md`
- `experiment_card.md`
- `case_report.md`
- `level2_portfolio_validation/portfolio_validation_summary.json`
- `level2_portfolio_validation/portfolio_validation_metrics.json`
- `level2_portfolio_validation/portfolio_validation_package.json`

## What To Inspect First

1. `profile_comparison.md` for a side-by-side profile summary.
2. `metrics.json` in each profile run for core decision fields:
   - `factor_verdict`
   - `campaign_triage`
   - `promotion_decision`
   - `portfolio_validation_status`
   - `portfolio_validation_recommendation`
3. `level2_portfolio_validation/portfolio_validation_summary.json` for
   portfolio-level diagnostics and risk notes.

## Interpreting Profile Differences

For this deterministic example input:

- `exploratory_screening` is more permissive in triage decisions.
- `default_research` applies stricter baseline standards.
- Portfolio-validation behavior can differ even when promotion does not pass,
  because profile policy controls whether non-promoted cases are still
  evaluated at Level 2.

Use `profile_comparison.json` when you need machine-readable deltas for tests,
CI logs, or downstream reporting.

## How To Check Level 2 Promotion

Promotion is explicit in each profile run `metrics.json`:

- Promoted: `promotion_decision == "Promote to Level 2"`
- Not promoted: any other `promotion_decision` label

`portfolio_validation_status` is related but not identical to promotion:
some profiles can still run Level 2 portfolio validation for non-promoted cases.
