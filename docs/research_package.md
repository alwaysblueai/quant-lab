# Research Package

## Purpose

`alpha_lab.research_package` is the canonical post-run packaging layer for one
completed research case.

It does not rerun research, modeling, or replay engines. It loads existing
artifacts and exports a standardized package for archival, review, comparison,
and downstream ingestion.

## What It Collects

When available, one package includes:

- workflow summary (`*_workflow_summary.json`)
- promotion decision and key workflow diagnostics
- trial-log and alpha-registry references
- handoff bundle references and contract summaries
- replay summaries (vectorbt / backtrader)
- execution impact report summary
- explicit artifact index with concrete paths
- package-level final verdict with transparent basis fields

## Exports

For one case:

- `research_package.json`
  - complete machine-readable payload for tooling and ingestion
- `research_package.md`
  - concise, review-oriented summary for human inspection
- `artifact_index.json` (optional)
  - explicit artifact path map for automation and audit checks

Design rule:
- markdown is intentionally concise and review-first
- JSON is complete and deterministic

## Missing Artifacts and Degradation

Package construction is resilient to missing optional outputs.

- Missing artifacts are surfaced in `missing_artifacts` explicitly.
- Missing sections are represented as `null`/empty collections, not silently dropped.
- Corrupt JSON artifacts still raise explicit errors.

This keeps package generation audit-friendly while preserving strictness for
malformed files.

## Verdict Interpretation

Package verdict assembly is explicit and transparent:

- base from workflow `PromotionDecision`
- adjusted by replay limitations / missing replay artifacts
- adjusted by execution-impact flags and blockers
- emits:
  - `verdict`
  - `package_readiness`
  - `research_verdict_basis`
  - `replay_verdict_basis`
  - `execution_verdict_basis`
  - `blocking_issues`
  - `warnings`
  - `next_recommended_action`

No opaque score is used.

## API

Core APIs:

- `build_research_package(...)`
- `export_research_package(...)`
- `load_research_case_outputs(...)`
- `summarize_replay_outputs(...)`
- `summarize_execution_impact(...)`

Optional campaign-level helper:

- `build_campaign_summary(...)`

## Optional CLI

Use the standalone script as a post-processing step:

```bash
uv run python scripts/build_research_package.py \
  --case-output-dir data/processed/research_cases/case1_single_reversal \
  --output-dir data/processed/research_cases/case1_single_reversal/package
```
