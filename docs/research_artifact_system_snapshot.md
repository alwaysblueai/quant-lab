# Research Artifact System Snapshot (Level 1/2)

Implementation-grounded architecture snapshot and operator guide for the current artifact-backed research workbench.

## 1) System Overview

The artifact system is the persistence boundary for the Level 1/2 research workflow:

- Level 1: factor discovery and signal validation
- Level 2: portfolio construction validation and backtest review

Core purpose:

- make research outputs reproducible and auditable as typed JSON artifacts
- keep dashboard/report rendering artifact-backed instead of ad-hoc dataframe reconstruction
- surface artifact load quality as structured diagnostics (not only free text warnings)

Why three object families exist:

- Canonical objects (`factor_definition.json`, `signal_validation.json`, `portfolio_recipe.json`, `backtest_result.json`) persist case-level L1/L2 state.
- Workflow-closure objects (`factor_set_result.json`, `candidate_recipe_generation.json`, `winner_selection.json`, `next_step_recommendations.json`) persist deterministic dashboard decision layers that are derived from campaign comparison data.
- Diagnostics object (`artifact_load_diagnostics.json`) persists loader governance state, fallback usage, and strict/permissive execution context.

Primary implementation anchors:

- `src/alpha_lab/real_cases/single_factor/artifacts.py`
- `src/alpha_lab/real_cases/composite/artifacts.py`
- `src/alpha_lab/campaigns/research_campaign_1.py`
- `src/alpha_lab/campaigns/profile_comparison.py`
- `src/alpha_lab/reporting/renderers/campaign_profile_dashboard.py`
- `src/alpha_lab/reporting/renderers/research_dashboard_schema.py`
- `src/alpha_lab/artifact_contracts.py`

## 2) Artifact Graph (First-Class Types)

| Artifact file | `artifact_type` | Layer | Producer | Primary consumers |
|---|---|---|---|---|
| `factor_definition.json` | `alpha_lab_factor_definition` | Canonical | `export_artifact_bundle()` in `real_cases/*/artifacts.py` | Dashboard case loader, lineage registry, profile comparison payload pointers |
| `signal_validation.json` | `alpha_lab_signal_validation` | Canonical | `export_artifact_bundle()` in `real_cases/*/artifacts.py` | Dashboard case loader, factor metrics canonicalization |
| `portfolio_recipe.json` | `alpha_lab_portfolio_recipe` | Canonical | `export_artifact_bundle()` in `real_cases/*/artifacts.py` | Dashboard recipe builder, portfolio controls view |
| `backtest_result.json` | `alpha_lab_backtest_result` | Canonical | `export_artifact_bundle()` in `real_cases/*/artifacts.py` | Dashboard backtest summary, recipe comparison |
| `factor_set_result.json` | `alpha_lab_factor_set_result` | Workflow closure | `persist_workflow_closure_artifacts()` in `campaign_profile_dashboard.py` | Dashboard workflow loader, lineage workflow links |
| `candidate_recipe_generation.json` | `alpha_lab_candidate_recipe_generation` | Workflow closure | `persist_workflow_closure_artifacts()` in `campaign_profile_dashboard.py` | Dashboard workflow loader, winner selection inputs |
| `winner_selection.json` | `alpha_lab_winner_selection` | Workflow closure | `persist_workflow_closure_artifacts()` in `campaign_profile_dashboard.py` | Dashboard workflow loader, recommendation layer |
| `next_step_recommendations.json` | `alpha_lab_next_step_recommendations` | Workflow closure | `persist_workflow_closure_artifacts()` in `campaign_profile_dashboard.py` | Dashboard workflow loader, operator action panel |
| `artifact_load_diagnostics.json` | `alpha_lab_artifact_load_diagnostics` | Governance / diagnostics | `persist_workflow_closure_artifacts()` and `write_campaign_profile_dashboard_html()` in `campaign_profile_dashboard.py` | Operators, strict-mode troubleshooting, lineage workflow link tail |

Contract validation for all above is centralized in `validate_level12_artifact_payload()` (`src/alpha_lab/artifact_contracts.py`).

## 3) Loading Semantics

Entry point: `_build_research_dashboard_data()` in `campaign_profile_dashboard.py`.

### Canonical-first behavior

Case load path:

1. `_load_case_artifacts()` reads `profiles[*].artifact_paths` from `campaign_profile_comparison.json`.
2. For each canonical artifact, `_case_artifact_path()` resolves:
   - explicit path from `artifact_paths.<..._json_path>`, else
   - `output_dir / <artifact_filename>`.
3. `_load_case_artifact_json_payload()` loads and validates each canonical JSON.
4. `metrics_payload` is built from legacy `metrics.json`, then canonical signal/portfolio fields overwrite legacy fields where present.

Result: canonical objects are preferred when present; legacy metrics remain fallback context.

### Workflow persisted-first behavior

Workflow load path:

1. `_workflow_closure_artifact_paths_from_payload()` resolves paths for workflow artifacts from top-level `workflow_closure_artifacts` pointers in comparison payload.
2. If missing pointers, it falls back to sibling files near comparison JSON:
   - `factor_set_result.json`
   - `candidate_recipe_generation.json`
   - `winner_selection.json`
   - `next_step_recommendations.json`
   - `artifact_load_diagnostics.json`
3. Workflow artifacts are loaded with `_load_<type>_artifact()` wrappers and parsed into typed dataclasses.

### Permissive mode (`artifact_load_mode="permissive"`)

Policy (`_build_artifact_load_policy()`):

- `require_canonical_artifacts=False`
- `require_workflow_closure_artifacts=False`
- fallback enabled for canonical legacy surfaces and workflow derived surfaces
- persisted workflow preference defaults to on for dashboard build, but can be disabled internally when regenerating workflow closure artifacts

Behavior:

- missing/invalid artifacts emit warning diagnostics
- fallback objects are used where available
- dashboard build succeeds unless an unexpected runtime error occurs outside policy handling

### Strict mode (`artifact_load_mode="strict"`)

Policy:

- requires canonical artifacts (for successful cases) and workflow closure artifacts
- disables fallback (`allow_legacy_case_fallback=False`, `allow_workflow_fallback=False`)
- forces persisted workflow artifact preference

Behavior:

- canonical artifact requirements are enforced for case rows with `status="success"`
- any required artifact issue becomes an error diagnostic
- build appends `STRICT_LOAD_ABORTED` and raises `ArtifactLoadRuntimeError`
- exception carries structured diagnostics via `ArtifactLoadRuntimeError.diagnostics`

### Failure / fallback summary

- Missing canonical/workflow path -> diagnostic (`MISSING_*`), optional fallback in permissive.
- Invalid workflow artifact payload -> diagnostic (`INVALID_WORKFLOW_ARTIFACT`), workflow fallback in permissive.
- Invalid canonical artifact payload -> diagnostic (`INVALID_CANONICAL_ARTIFACT`); strict fails, permissive continues with warning.
- Strict mode with any artifact errors -> aborts with structured diagnostic context.

## 4) Diagnostics Semantics

Structured shape (`ArtifactLoadDiagnostic` in `research_dashboard_schema.py`):

- `code`
- `severity` (`warning` or `error`)
- `artifact_type`
- `object_scope`
- `message`
- `path`
- `case_name`
- `profile_name`
- `mode` (`permissive` or `strict`)
- `fallback_used`
- `remediation_hint`

Diagnostic codes currently emitted:

- `MISSING_CANONICAL_ARTIFACT`
- `INVALID_CANONICAL_ARTIFACT`
- `MISSING_WORKFLOW_ARTIFACT`
- `INVALID_WORKFLOW_ARTIFACT`
- `FALLBACK_USED`
- `STRICT_LOAD_ABORTED`

Text warning derivation:

- `_artifact_diagnostic_to_text()` currently maps each diagnostic to `diagnostic.message`.
- `ResearchDashboardData.artifact_load_warnings` is deduplicated from warning diagnostics.

Strict failure surfacing:

- `_build_research_dashboard_data()` raises `ArtifactLoadRuntimeError("strict artifact load checks failed: ...")`.
- `diagnostics` tuple on exception is the machine-readable failure payload.

Persistence:

- `artifact_load_diagnostics.json` stores:
  - mode and policy summary
  - full diagnostic list (`asdict` serialization)
  - source artifact pointers
- `write_campaign_profile_dashboard_html()` persists diagnostics only on successful dashboard build (strict failures raise before write).

## 5) Payload / Pointer Plumbing

Canonical path emission:

- `research_campaign_1._run_case()` captures canonical paths from case run artifact bundles.
- `research_campaign_1._case_result_to_dict()` and pointer writers persist canonical path keys:
  - `factor_definition_json_path`
  - `signal_validation_json_path`
  - `portfolio_recipe_json_path`
  - `backtest_result_json_path`

Comparison payload propagation:

- `profile_comparison._case_profile_payload()` embeds per-profile `artifact_paths` under each case row.
- `campaign_profile_comparison.json` becomes the pointer hub for dashboard and comparison renderers.

Workflow closure pointer propagation:

- `persist_workflow_closure_artifacts()` writes workflow closure JSONs near comparison output.
- `_update_comparison_workflow_closure_context()` writes `workflow_closure_artifacts` pointers into:
  - top-level comparison payload
  - each `profile_runs[*].campaign_artifacts.workflow_closure_artifacts`
  - `campaign_level_summary.workflow_closure_artifacts`

Consumer lookup behavior:

- Dashboard loader resolves canonical paths from per-case `artifact_paths`, with `output_dir` filename fallback.
- Dashboard workflow loader resolves persisted workflow pointers from `workflow_closure_artifacts`, then sibling-file fallback.
- Campaign comparison markdown uses `case_evidence_index` artifact hints to expose traceable pointers.
- Campaign report loader falls back to `metrics_path` and then `output_dir/metrics.json` when key metrics are absent.

## 6) Lineage / Registry

Lineage object model is assembled in `campaign_profile_dashboard.py`:

- Run registry entries: `ExperimentRegistryEntry`
- Relationship links: `ResearchLineageLink`
- Aggregate registry: `ResearchLineageRegistry`

Canonical chain (per case/profile):

- `factor_definition -> signal_validation -> portfolio_recipe -> backtest_result`
- Implemented via `_build_lineage_links()`.

Workflow chain (default profile context):

- `factor_shortlist -> factor_set_result -> candidate_recipe_generation -> winner_selection -> next_step_recommendations -> artifact_load_diagnostics`
- Implemented in `_build_lineage_registry()` when workflow artifact paths are available.

Diagnostics linkage:

- Diagnostics are both:
  - embedded in `ResearchDashboardData.artifact_load_diagnostics`
  - persisted as first-class artifact and linked in workflow lineage tail

Provenance links:

- `_provenance_links()` extracts `source_artifacts` from canonical payloads into registry entry provenance strings.

## 7) Safe Extension Guidance

When adding a new artifact type, keep one contract pipeline end-to-end:

1. Contract:
   - Add artifact type constant and validator in `src/alpha_lab/artifact_contracts.py`.
   - Register validator in `validate_level12_artifact_payload()` dispatch.
2. Writer:
   - Add one payload builder + writer at the true producer boundary (canonical producer or workflow-closure producer).
   - Always write through validation (`validate_level12_artifact_payload`) before persistence.
3. Payload plumbing:
   - Add pointer keys where artifacts must be discoverable (`artifact_paths` for case-level, `workflow_closure_artifacts` for workflow-level).
   - Keep pointers deterministic and portable.
4. Loader:
   - Add loader/parser function in dashboard loader path.
   - Add diagnostics coverage for missing/invalid/pointer issues.
   - Update strict/permissive behavior intentionally (required vs fallback).
5. Tests:
   - Add/extend contract tests in `tests/test_artifact_contracts.py`.
   - Add loader and strict/permissive tests in `tests/test_campaign_profile_dashboard_renderer.py`.
   - Add payload wiring checks in `tests/test_campaign_profile_comparison.py` if pointer shape changes.

Anti-patterns to avoid:

- duplicating schema validation outside `artifact_contracts.py`
- bypassing canonical objects by reading only legacy `metrics.json` in new consumers
- introducing hidden fallback paths without structured diagnostics

## 8) Known Heuristics and Current Limitations

Current heuristics:

- Factor family inference uses keyword rules (`_FACTOR_FAMILY_RULES`) in dashboard renderer.
- Factor shortlist scoring uses normalized ranges and fixed component weights/thresholds.
- Factor-set construction and candidate recipe generation are deterministic policy heuristics.
- Winner selection uses weighted normalized metrics plus guardrails, with heuristic handling when core metrics are missing.
- Next-step recommendations are deterministic rule outputs over shortlist/set/candidate/winner states.

Permissive fallback surfaces:

- canonical missing artifacts can fall back to legacy metrics/manifest context
- workflow missing/invalid artifacts can fall back to in-memory deterministic builders
- fallback usage is explicitly tracked with `FALLBACK_USED` diagnostics

Limitations:

- Workflow closure artifacts are persisted for one comparison context (`default_profile` anchored), not separate per-profile workflow closure payloads.
- Dashboard case selection is default-profile-first (`_select_profile_payload()`), with fallback to first available profile when default is absent.
- Invalid canonical payloads in permissive mode are tolerated with diagnostics; this favors continuity over hard fail.
- Workflow artifact fallback data is computed at dashboard build time and may differ from previously persisted workflow artifacts if policies/configs evolve.

## 9) Operator Checklist (Strict Mode Formal Review)

Use this sequence for formal artifact-backed review:

1. Generate/refresh comparison and workflow artifacts:
   - `alpha-lab campaign compare-profiles --source <example|campaign> ...`
   - (This calls `persist_workflow_closure_artifacts()` internally.)
2. Render dashboard in strict mode:
   - `alpha-lab campaign render-dashboard --comparison-json <path>/campaign_profile_comparison.json --artifact-load-mode strict --overwrite`
3. If strict mode fails, inspect diagnostics from CLI error and (if present) `artifact_load_diagnostics.json` near comparison outputs.
4. Require zero strict-mode errors before sign-off.
5. Archive as review bundle:
   - `campaign_profile_comparison.json`
   - canonical case artifacts
   - workflow closure artifacts
   - rendered dashboard/report outputs
