# quant-lab AGENT Rules

## 1) Scope Boundary (Permanent)
- Core scope is only:
  - Level 1: factor discovery
  - Level 2: portfolio construction validation
- Experimental only:
  - Future Level 3 replay / implementability
  - execution replay, fill simulation, adapter parity, execution semantics auditing

## 2) Hard Separation Rules
- Never introduce Level 3 semantics into:
  - core/public APIs (`src/alpha_lab` Level 1/2 modules)
  - default CLI workflows/help text
  - default docs narrative
  - default test surface
- Level 3 code must stay isolated under experimental namespaces and opt-in commands.

## 3) Core Engineering Priorities
- Preserve research-integrity guarantees first:
  - no future data usage
  - no label-feature leakage
  - explicit temporal alignment and split discipline
  - PIT/as-of and cross-timeframe correctness
- Keep Tier 1/Tier 2 research boundaries aligned with `docs/research_playbook.md`.
  New diagnostics start in Tier 2 unless they compare a batch of factors and affect
  the continue/stop verdict.
- Prefer minimal, high-confidence edits over broad redesign.
- Keep code and documentation aligned in the same change.
- When adding new features, prioritize:
  - factor diagnostics and robustness analysis
  - portfolio validation and sensitivity analysis
  - reproducible reporting and experiment export
  - not execution realism

## 4) Default vs Experimental Testing
- Default test workflow excludes `experimental_level3` tests.
- Mark Level 3 tests with `@pytest.mark.experimental_level3` under `tests/experimental_level3/`.
- Run experimental tests only when explicitly requested.

## 5) Main Commands
- If `uv` cache permissions are restricted, prefix commands with:
  - `UV_CACHE_DIR=/tmp/uv-cache`
- Full local gate:
  - `make check`
- Individual checks:
  - `make lint`
  - `make typecheck`
  - `make test`
- Direct equivalents:
  - `uv run --no-sync --frozen ruff check .`
  - `uv run --no-sync --frozen mypy src`
  - `uv run --no-sync --frozen pytest`
- Experimental-only tests:
  - `uv run --no-sync --frozen pytest -m experimental_level3 tests/experimental_level3`

### LLM rerank for research_bridge
- `alpha-lab explore-idea` and `alpha-lab model-idea` automatically use Claude
  `claude-sonnet-4-6` to rerank coarse candidates for idea relevance when
  `ANTHROPIC_API_KEY` is present.
- The score is recorded as `score_components.llm_relevance`; missing keys or SDK
  failures silently fall back to deterministic hash-TFIDF ranking.
- Diagnostics are recorded under `retrieval_diagnostics.llm_rerank`.

### V2 mechanism recall for research_bridge
- V2 recall is opt-in: set `ALPHA_LAB_RESEARCH_BRIDGE_V2=1` and
  `ANTHROPIC_API_KEY` to enable query expansion, mechanism-tier retrieval, and
  categorize/compress synthesis.
- Build the offline sidecar before first use:
  `python -m alpha_lab.research_bridge.mechanism_index build --vault <vault>`.
- Sidecar files live under
  `.research_bridge_cache/mechanism_index/<vault_hash>/`; if the flag is unset
  or the API key is missing, the explorer falls back completely to v1 behavior.

## 6) Codex GUI Stage 3 Draft Factor Guardrails
- When the user provides Stage2 output, `factor_json_payload`, or asks to run a
  backend draft factor, treat the task as a Stage 3 backend draft-factor run.
- Follow:
  - `docs/templates/stage3_backend_draft_factor_prompt.md`
  - `docs/templates/codex_gui_stage3_execution_envelope.md`
  - `docs/backend_draft_factor_workflow.md`
- Only `factor_json_payload` and explicit local paths/commands from the user are
  machine facts. If prose conflicts with `factor_json_payload`, use
  `factor_json_payload` and record the conflict in `research_log.md`.
- Allowed writes are limited to:
  - `custom_factors/research/<factor_name>/factor.json`
  - `custom_factors/research/<factor_name>/research_log.md`
  - the matching `configs/real_cases/single_factor/<factor_name>_vN.yaml`
- Do not create one-off scripts, notebooks, scattered `.py` files, promoted
  factors, or frontend registrations during Stage 3 draft runs.
- Before running a case, validate the draft with:
  - `uv run --no-sync --frozen alpha-lab validate-draft-factor custom_factors/research/<factor_name>/factor.json`
- Run experiments only through:
  - `uv run --no-sync --frozen alpha-lab real-case single-factor run <case.yaml> ...`
- After the run, inspect `run_manifest.json` and `factor_definition.json`; the
  run is not acceptable unless `custom_factor_source.code_sha256`,
  `custom_factor_source.factor_json_sha256`, and source path are present.
- If validation, required-column availability, leakage checks, or artifact audit
  fields fail, stop and report the failure instead of rewriting the factor
  outside the contract.

## 6.5) Codex GUI Model-Lab Stage 3 Draft Model Guardrails (v1: spec variants)
- When the user provides Stage2 output, `model_candidate_payload`, or asks to run a
  backend draft model, treat the task as a Stage 3 backend draft-model run.
- Follow:
  - `docs/templates/model_lab_stage3_backend_draft_prompt.md`
  - `docs/templates/codex_gui_model_stage3_execution_envelope.md`
  - `docs/backend_draft_model_workflow.md`
- Only `model_candidate_payload` (and its embedded `case_spec_payload`) and explicit
  local paths/commands from the user are machine facts. If prose conflicts with
  `case_spec_payload`, use `case_spec_payload` and record the conflict in
  `model_candidates/research/<candidate_name>/research_log.md`.
- Allowed writes are limited to:
  - `model_candidates/research/<candidate_name>/model_candidate.json`
  - `model_candidates/research/<candidate_name>/research_log.md`
  - the matching `configs/real_cases/model_factor/<candidate_name>_vN.yaml`
- Do not create one-off scripts, notebooks, scattered `.py` files, promoted
  candidates, frontend registrations, custom feature builders, or custom
  estimator code during Stage 3 draft-model runs. v1 only supports spec-variant
  candidates.
- Before running a case, validate the candidate with:
  - `uv run --no-sync --frozen alpha-lab validate-draft-model model_candidates/research/<candidate_name>/model_candidate.json`
- Run experiments only through:
  - `uv run --no-sync --frozen alpha-lab real-case model-factor run <case.yaml> --draft-model-candidate model_candidates/research/<candidate_name>/model_candidate.json ...`
- If the Web Model Lab is available, the `/model-lab` Draft Candidates panel may
  be used as the orchestrator, but it must still perform the same save ->
  validate -> materialize-spec -> standard run sequence and preserve artifact
  hash auditing.
- After the run, inspect `run_manifest.json`, `model_definition.json`, and
  `feature_manifest.json`; the run is not acceptable unless
  `draft_model_source.candidate_json_sha256`, `case_spec_sha256`,
  `feature_contract_sha256`, and source path are all present.
- If validation, feature-column availability, PIT contract checks, or artifact
  audit fields fail, stop and report the failure instead of rewriting the
  candidate outside the contract.

## 7) Pre-Finish Review Checklist
- Scope: change remains Level 1/2 by default.
- API/CLI/docs: no Level 3 semantics leaked into core/default paths.
- Integrity: temporal/leakage guarantees preserved or strengthened.
- Tests: core tests updated; experimental tests marked and isolated.
- Docs: README/docs match actual behavior.
- Diff quality: minimal, coherent, and reversible.

## 8) Code Layout Registry

The repo intentionally co-locates two parallel pipelines (single-factor and
model-factor / "model lab") under a single Python package. Use this registry
as the single source of truth for "where does <X> live?". The model line
deliberately stays distributed rather than collapsed into one subpackage so
each layer owns its own contract; this table is what makes that layout legible.

### Single-factor line (alpha-lab)

| Concern | Path |
| --- | --- |
| Case spec | `src/alpha_lab/real_cases/single_factor/spec.py` |
| Pipeline driver | `src/alpha_lab/real_cases/single_factor/pipeline.py` |
| Evaluate package | `src/alpha_lab/real_cases/single_factor/evaluate/` (core + coverage + strict_research + capacity + comparisons + pnl_attribution + data_quality + diagnostics + summary_metrics + _utils) |
| Artifacts | `src/alpha_lab/real_cases/single_factor/artifacts.py` |
| Templates | `src/alpha_lab/real_cases/single_factor/templates.py` |
| CLI | `src/alpha_lab/real_cases/single_factor/cli.py` |
| Custom factor loader | `src/alpha_lab/custom_factors.py` |
| Draft validator | `src/alpha_lab/draft_factor_validation.py` |
| Persisted factor defs | `custom_factors/{research,promoted}/<name>/factor.json` |
| Configs | `configs/real_cases/single_factor/*.yaml` |
| Workflow docs | `docs/backend_draft_factor_workflow.md`, `docs/factor_promotion_checklist.md`, `docs/templates/stage3_backend_draft_factor_prompt.md` |

### Model-factor line (model lab)

| Concern | Path |
| --- | --- |
| Core ML package | `src/alpha_lab/model_factor/core/` (types + config + internals + preprocess + training_arrays + selection + estimator + importance + diagnostics_build + build + _utils) |
| Dataset cache | `src/alpha_lab/model_factor/dataset_cache.py` |
| Diagnostics observer | `src/alpha_lab/model_factor/diagnostics.py` |
| Memory helper | `src/alpha_lab/model_factor/_memory.py` |
| Case spec | `src/alpha_lab/real_cases/model_factor/spec.py` |
| Pipeline package | `src/alpha_lab/real_cases/model_factor/pipeline/` (core + cache + features + labels + feature_manifest + diagnostics + _utils) |
| Artifacts package | `src/alpha_lab/real_cases/model_factor/artifacts/` (core + model_selection + backtest_recipe + feature_export + diagnostics + _utils) |
| Templates | `src/alpha_lab/real_cases/model_factor/templates.py` |
| CLI | `src/alpha_lab/real_cases/model_factor/cli.py` |
| Benchmark runner | `src/alpha_lab/real_cases/model_factor/benchmark.py` |
| Candidate loader | `src/alpha_lab/model_candidates.py` |
| Draft validator | `src/alpha_lab/draft_model_validation.py` |
| Persisted candidate defs | `model_candidates/{research,promoted}/<name>/model_candidate.json` |
| Configs | `configs/real_cases/model_factor/*.yaml` |
| Web UI fixtures | `src/alpha_lab/dev_fixtures/model_lab_overview/*.json` |
| Workflow docs | `docs/backend_draft_model_workflow.md`, `docs/model_factor_performance_roadmap.md`, `docs/model_factor_metric_inventory.md`, `docs/templates/model_lab_stage3_backend_draft_prompt.md`, `docs/templates/codex_gui_model_stage3_execution_envelope.md` |

### Cross-cutting

| Concern | Path |
| --- | --- |
| Research bridge (idea exploration) | `src/alpha_lab/research_bridge/` |
| Reporting (manifests, tearsheets, triage) | `src/alpha_lab/reporting/` |
| Research integrity contracts | `src/alpha_lab/research_integrity/` |
| Splits / purged k-fold | `src/alpha_lab/splits.py`, `src/alpha_lab/validation/purged_kfold.py` |
| Web UI server (alpha-lab + model-lab tabs) | `src/alpha_lab/web_unified.py` |
| Frontend metrics dashboard | `frontend/metrics-dashboard/` |
| Vault export / experiment cards | `src/alpha_lab/reporting/__init__.py` (`export_experiment_card`); cards land at `<vault>/50_experiments/` |

### Scripts layout (post-E7 reorganization)

| Bucket | Purpose |
| --- | --- |
| `scripts/bench/` | Performance benchmarks (`bench_*`) |
| `scripts/run/` | Runners — Python and shell (`run_*`) |
| `scripts/data/` | Data ingestion / preparation (adapt, generate, fill, materialize) |
| `scripts/build/` | Sidecar builders (`build_factor_from_recipe`, `build_mechanism_index`) |
| `scripts/smoke/` | Frontend smoke harnesses (`smoke_*`) |
| `scripts/etl/` | 1-min ETL (Phase 1) |
| `scripts/intraday/` | Intraday readiness checks |
| `scripts/verify/` | Cross-checks against vendor parquet |
| `scripts/{ab_compare,gc_web_runs,diagnose_runtime_stability}.{py,sh}` | Top-level miscellaneous tools |

### Archived / opt-out paths

These directories are intentionally outside the active surface — leave them
alone unless you're explicitly working on archival, perf baselining, or
ad-hoc data backfill.

| Path | What it is |
| --- | --- |
| `configs/_archive/single_factor/` | Pre-2026-05 tushare `_qfq`/no-suffix sweep configs. Moved out because vendor `_qfq` is not point-in-time; new single-factor work uses no-suffix + `_bfq` under `configs/real_cases/single_factor/`. History preserved via rename (commit `dbf38a7`). |
| `configs/_archive/model_factor/` | Pre-2026-05 bare `stock_*.yaml` configs targeting the legacy `features_safe.parquet` suite. Superseded by `*_safe_bfq*` variants under `configs/real_cases/model_factor/` once 35-feature safe_bfq became the primary benchmark line (commit `310e7f4`). |
| `tests/perf_baselines/*.json` | Frozen `scripts/bench/bench_pipeline.py` outputs used by `.github/workflows/bench.yml` for informational A/B compare against the current branch. Not pytest fixtures — refreshed by re-running `bench_pipeline.py --size <medium\|small>` when the baseline intentionally moves. |
| `scripts/intraday/_oneshots/` | Gitignored ad-hoc backfill / cleanup scripts (e.g., `strip_year_column.py`, `apply_cutoff_2025.py`, `rebuild_summaries.py`). They run once against a specific local snapshot and are deliberately not version-controlled; do not promote them out without first making them idempotent and contract-checked. |

## 9) Card Language Policy
- Generated cards, summaries, and human-facing research docs should be Chinese-first.
- The main body, section titles, and explanatory text should use Chinese by default.
- Preserve English only where it is necessary for professional terminology, proper nouns, established technical abbreviations, formulas, code symbols, file paths, or quoted source titles.
- Avoid mixed-language prose unless the English phrase is the canonical term or is required for exact technical meaning.
