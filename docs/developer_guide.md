# Developer Guide

How to extend and maintain Alpha Lab.  For system-level API reference see
[system_manual.md](system_manual.md).  For layer contracts see
[architecture.md](architecture.md).

---

## Adding a New Factor

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

**Mypy**: strict mode on `src/alpha_lab`.  All public functions must have
complete type annotations.

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

---

## Entrypoint

The CLI entry point is `scripts/run_experiment.py`, which delegates to
`alpha_lab.cli`.  There is no `main.py`.

To add a new CLI command, extend `alpha_lab.cli` and expose it through a new
script in `scripts/`.  Do not add top-level `main.py` files — the CLI module
is the single entry point.
