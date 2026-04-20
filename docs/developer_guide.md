# Developer Guide

How to extend and maintain Alpha Lab.  For system-level API reference see
[system_manual.md](system_manual.md).  For layer contracts see
[architecture.md](architecture.md).

---

## Scope Guardrails

- Treat this repository as a Level 1/2 research system.
- Keep defaults focused on research integrity, factor research, and
  portfolio-level validation.
- Do not move replay/implementability abstractions into core APIs.

---

## Evaluation Profiles

- Discover available Level 1/2 profiles with:
  `alpha-lab profiles`
- Profile intent:
  - `default_research`: balanced baseline for routine Level 1/2 work
  - `exploratory_screening`: more permissive candidate discovery
  - `stricter_research`: more conservative evidence standards for Level 2 readiness
- Real-case and campaign workflows should pass `--evaluation-profile` explicitly
  when reproducibility/auditability is required.
- Profile selection changes standards across factor verdicts, campaign triage,
  Level 2 promotion, and Level 2 portfolio-validation guardrails.
- Uncertainty settings are centralized in `UncertaintyConfig`, including
  `method` (`normal`, `bootstrap`, or `block_bootstrap`) and bootstrap controls
  (`bootstrap_resamples`, `bootstrap_confidence_level`, `bootstrap_random_seed`,
  `block_bootstrap_block_length`).
- Keep profile additions centralized in `alpha_lab.research_evaluation_config`;
  avoid per-command ad-hoc threshold flags in default workflows.

---

## Adding a New Factor

For quick Level 1/2 batch validation workflow and cache/parallel options, see
`docs/single_factor_quick_pipeline.md`.

1. Create `src/alpha_lab/factors/<factor_name>.py`.
2. Export one public function with signature:
   ```python
   def <factor_name>(prices: pd.DataFrame, **kwargs) -> pd.DataFrame:
       ...
   ```
3. Return a canonical long-form DataFrame:
   - Columns: `[date, asset, factor, value]`
   - Exactly one row per `(date, asset, factor)`
   - Factor values at `date=t` use only information available at or before `t`
   - Factor name column must be a string literal, not derived from input data
4. Call `validate_factor_output(factor_df)` before returning (or let
   `run_factor_experiment` validate it).
5. Document in the factor's docstring:
   - hypothesis
   - lookback window
   - intended use horizon
   - whether computation is cross-sectional or time-series
   - timestamp alignment
   - leakage risk
6. Add tests in `tests/test_factors_<factor_name>.py`:
   - happy-path output schema
   - edge case: single-asset input
   - edge case: window larger than available history
   - no-future-data assertion (factor at `t` uses only `prices[date <= t]`)

---

## Adding a New Portfolio Weight Method

1. Open `src/alpha_lab/portfolio_research.py`.
2. Add the method name to `_VALID_WEIGHT_METHODS` frozenset.
3. Add a branch inside `portfolio_weights()` that computes per-date weights
   using the new method.
4. Ensure long-leg weights sum to `+1` and short-leg weights sum to `−1` per
   date (or `+1` for long-only).
5. Add the method name to `_VALID_WEIGHT_METHODS` in
   `src/alpha_lab/strategy.py` as well, so `StrategySpec` accepts it.
6. Add tests asserting the sum constraint and column schema.

---

## Extending StrategySpec

`StrategySpec` should only contain **portfolio-construction** parameters —
fields that govern which assets to select, how to weight them, and at what
frequency to rebalance.

Do **not** add to `StrategySpec`:
- `n_quantiles` — this governs the factor-evaluation layer (IC, quantile
  returns), not portfolio construction.
- `horizon` — this governs label generation, not portfolio construction.
- `cost_rate` — this is a cost assumption, not a construction decision.

If you add a new field:
1. Add it to `StrategySpec` with a frozen dataclass default.
2. Add validation in `__post_init__`.
3. Thread it through `portfolio_weights_from_strategy` if it affects weight
   computation.
4. Update `_run_portfolio_block` in `experiment.py` and the strategy override
   block in `run_walk_forward_experiment` in `walk_forward.py`.
5. Update docstrings in `strategy.py`, `experiment.py`, `walk_forward.py`,
   and `docs/architecture.md`.

---

## Adding a New ExperimentResult Field

1. Add the field (with a default of `None`) to `ExperimentResult` **after**
   all required fields and all existing optional fields.  `@dataclass` requires
   fields with defaults to come after required fields.
2. Populate the field in `run_factor_experiment`.
3. If the field is a per-fold output in walk-forward evaluation, add an
   accumulator in the fold loop of `run_walk_forward_experiment`, build a
   pooled DataFrame, add it to `WalkForwardResult`, and add the corresponding
   aggregate statistic to `WalkForwardAggregate`.
4. Add tests for: field is `None` when the feature is not requested; field is
   a DataFrame with the expected columns when the feature is active.

---

## Adding a Walk-Forward Pooled Output

Follow the pattern established by `pooled_ic_df` and
`pooled_portfolio_return_df`:

1. Add a `list[pd.DataFrame]` accumulator before the fold loop.
2. Inside the fold loop, if the relevant `ExperimentResult` field is not
   `None` and not empty, copy the relevant columns and `insert(0, "fold_id",
   fold_id)`.  Append to the accumulator.
3. After the loop, `pd.concat` with `ignore_index=True` if the accumulator is
   non-empty; otherwise return an empty DataFrame with the expected columns.
4. Add the field to `WalkForwardResult`.
5. Pass the pooled DataFrame to `_compute_aggregate`; add the corresponding
   scalar statistic to `WalkForwardAggregate`.
6. Update `docs/architecture.md` pooled-DataFrame table.

---

## Code Quality Checklist

Before opening a PR:

```bash
make lint        # ruff check src tests
make typecheck   # mypy src
make test        # pytest -q
```

Or run all at once:

```bash
make check
```

**Ruff rules enforced**:
- `I` — import sorting
- `E`, `W` — pycodestyle
- `B` — flake8-bugbear
- `UP` — pyupgrade
- Line length 100 (configured in `pyproject.toml`)

**Mypy**: strict defaults are enabled on `src/alpha_lab`, with explicit
`pyproject.toml` overrides where necessary. Keep overrides narrow and documented.

---

## Testing Conventions

- Test files live in `tests/`, named `test_<module>.py`.
- Every public function needs at least:
  - a happy-path test
  - an empty-input test (where applicable)
  - a validation-error test for each `ValueError` guard
- Factor tests must include a no-future-data assertion.
- Use `np.random.default_rng(seed)` for reproducible synthetic data.
- Do not use `pytest.raises(Exception)` — be specific (`ValueError`,
  `AttributeError`, `TypeError`, etc.).
- Golden artifact regression coverage for key Level 1/2 user-facing outputs
  lives in `tests/test_artifact_golden_regression.py`.
  Covered workflows are intentionally compact:
  - deterministic single-factor Level 1/2 run
  - deterministic campaign profile-comparison example output
  Refresh goldens only when output drift is intentional:
  `ALPHA_LAB_UPDATE_GOLDENS=1 uv run --no-sync --frozen pytest -q tests/test_artifact_golden_regression.py`

---

## Path / Config Conventions

- Use `alpha_lab.config` for project-root-relative paths:
  ```python
  from alpha_lab.config import PROCESSED_DATA_DIR
  path = PROCESSED_DATA_DIR / "my_output.csv"
  ```
- Do not construct `Path("data/processed/...")` — this is CWD-relative and
  breaks when the process is started from a different directory.
- `RAW_DATA_DIR` is for immutable raw inputs.  Do not write to it.
- `PROCESSED_DATA_DIR` is for derived outputs (registry, summaries, etc.).
- The project root is verified at import time: `config.py` checks for
  `pyproject.toml` and raises `RuntimeError` immediately if it is missing.
  This prevents silent artifact misplacement in non-editable installs.
- For non-editable installs, set `ALPHA_LAB_PROJECT_ROOT` env var.

## Raw Input Validation

Every new entrypoint that accepts a raw price panel must call
`validate_price_panel(df)` before any computation:

```python
from alpha_lab.data_validation import validate_price_panel
validate_price_panel(prices)  # raises ValueError on violation
```

Do not duplicate these checks in individual pipeline functions —
`validate_price_panel` is the single enforcement point.

## Factor Contract Enforcement

Every factor output must pass `validate_factor_output(df)` from
`alpha_lab.interfaces`.  This is called automatically inside
`run_factor_experiment` after each `factor_fn` call.  When writing
tests for new factors, call `validate_factor_output` directly to
verify the output satisfies the full contract (including NaT dates,
null assets, null factor names).

---

## Entrypoint

The installed CLI entry point is `alpha-lab` (`alpha_lab.cli:main`).
`scripts/run_experiment.py` remains as the legacy single-experiment wrapper.
There is no `main.py`.

To add a new CLI command, extend `alpha_lab.cli` routing first; only add a
script wrapper in `scripts/` when a dedicated standalone entrypoint is still
needed. Do not add top-level `main.py` files.

---

## Unified Research Frontend (`web_unified.py`)

### Overview

`alpha-lab web unified` launches a single-file local HTTP server (port 8766 by default)
integrating four subsystems into one 5-page SPA:

| Page | URL fragment | Purpose |
|---|---|---|
| Dashboard | `#dashboard` | Project count, pending writebacks, run status overview |
| Knowledge Ops | `#knowledge` | Card search, vault inbox, vault stats |
| Bridge Workspace | `#bridge` | Project lifecycle, rounds, GPT export, case scaffolding |
| Validation Console | `#validation` | Run launcher, live run table, artifact viewer |
| Writeback Review | `#writeback` | Draft review, patch, preview, apply |

### Entry point

```
alpha-lab web unified [--host HOST] [--port PORT] [--workspace-root DIR]
                      [--vault-root DIR] [--no-open-browser]
```

Default: `127.0.0.1:8766`. `web cockpit` has been formally deprecated and is now
only a compatibility alias that routes to `web unified` with a deprecation warning.

### Implementation constraints

- **Single-file, no framework**: `web_unified.py` is a self-contained Python module
  using `ThreadingHTTPServer` with inline HTML/CSS/JS. Do not introduce Streamlit,
  Dash, or a JS bundler — the pattern is intentional for portability.
- **File size caps**: `_MAX_TEXT_BYTES = 512 KB` on reads; `_MAX_REQUEST_BODY_BYTES = 2 MB`
  on request bodies. Adjust these constants if legitimate files exceed the limit.
- **Path traversal**: all file reads resolve the path with `.resolve()` and verify it
  starts with the expected root directory. Never bypass this check.
- **Error format**: all JSON error responses must include `"ok": False` and `"error"`.

### Adding a new API endpoint

1. Add the route pattern to `do_GET` / `do_POST` / `do_PATCH` / `do_PUT` in
   `_UnifiedRequestHandler` (follow the existing `parts[]` pattern).
2. Add the service method to `_UnifiedService`.
3. Add path-safety checks if the method reads files from user-supplied names.
4. Add a test in `tests/test_web_unified.py`.

### Adding a new frontend panel

All HTML is in `_index_html()`. Each page is a `<section id="view-{name}">` block.
JS is inline at the bottom of the function. Follow the existing `api()` helper for
all fetch calls — it centralises error handling and JSON parsing.

---
