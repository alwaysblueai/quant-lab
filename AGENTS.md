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

## 6) Pre-Finish Review Checklist
- Scope: change remains Level 1/2 by default.
- API/CLI/docs: no Level 3 semantics leaked into core/default paths.
- Integrity: temporal/leakage guarantees preserved or strengthened.
- Tests: core tests updated; experimental tests marked and isolated.
- Docs: README/docs match actual behavior.
- Diff quality: minimal, coherent, and reversible.

## 7) Card Language Policy
- Generated cards, summaries, and human-facing research docs should be Chinese-first.
- The main body, section titles, and explanatory text should use Chinese by default.
- Preserve English only where it is necessary for professional terminology, proper nouns, established technical abbreviations, formulas, code symbols, file paths, or quoted source titles.
- Avoid mixed-language prose unless the English phrase is the canonical term or is required for exact technical meaning.
