# Alpha Lab — Extension Points

This document maps the three seams a researcher is most likely to extend —
**factors**, **metrics**, and **evaluation profiles** — to the minimum code
changes required. Each section cites the exact file and line where the seam
lives so the touch surface is small and auditable.

Of the three, only the **factor** seam is a true pluggable registry today.
Metrics and evaluation profiles currently require an in-source change
(detailed in each section). The document calls out exactly what is and is
not a runtime registration so future readers do not assume an extension
point exists when it is still tracked as drift.

---

## 1. Add a new factor

### Seam

`src/alpha_lab/factor_recipe.py:73` — module-level `factor_registry`
singleton. Every factor recipe referenced by a `single-factor` case spec
is resolved through this registry.

### Minimum change

Write the factor as a pure function
`(prices: pd.DataFrame, **params) -> pd.DataFrame` returning the canonical
long-form `(date, asset, factor, value)` schema, then register it:

```python
# src/alpha_lab/custom_factors/my_factor.py
from alpha_lab.factor_recipe import factor_registry


@factor_registry.register_factor("my_factor")
def my_factor(prices, *, lookback: int = 10):
    """Cross-sectional rank of trailing {lookback}-day return."""
    ...  # compute long-form factor frame
    return frame  # columns: date, asset, factor, value
```

Wire it into a case spec (YAML):

```yaml
factor_input:
  mode: recipe
  recipe:
    method: my_factor
    params: {lookback: 10}
```

### What you do NOT need to touch

- `real_cases/single_factor/pipeline.py` — resolution happens via
  `factor_recipe.build_factor_from_recipe_mapping`, which reads from the
  registry.
- `cli.py` — the `real-case single-factor` subcommand loads specs at runtime
  and never hardcodes a factor name.
- `web_unified.py` — the Validation Console submits case names, not factor
  symbols; the registry is the authority.

### Runtime persistence (optional)

`_UnifiedService._load_persisted_custom_factors`
(`src/alpha_lab/web_unified.py`) auto-registers factor definitions saved via
the Custom Factor Workshop UI, so interactive additions survive restarts.

---

## 2. Add a new metric

### Seam (and what is NOT a seam yet)

`src/alpha_lab/evaluation.py` exposes individual metric functions; the
`ExperimentSummary` dataclass in `src/alpha_lab/experiment.py:78` freezes the
scalar contract. Artifact serialization happens in
`src/alpha_lab/real_cases/*/artifacts.py::export_artifact_bundle` and the
reporting packages under `src/alpha_lab/reporting/`.

**No artifact registry exists today.** Each case type's `artifacts.py`
hardcodes its manifest (item A6 in the drift table). Adding a metric that
surfaces as a new artifact file therefore means editing every `artifacts.py`
that should emit it. This is deferred drift, not a solved seam.

### Recommended path: additive field inside an existing artifact

Adding a field to `ExperimentSummary` is an **additive contract change**
and touches every consumer (artifacts, reporting, goldens). For a new
metric, prefer:

1. Compute the metric in a standalone function under
   `src/alpha_lab/reporting/` or `src/alpha_lab/evaluation.py`.
2. Emit it as a new field inside an **existing** artifact (typically
   `metrics.json`) under a namespaced key, so the existing bundle
   serialization picks it up without new files.
3. Add a unit test that pins the metric's value on a fixture panel.
4. Regenerate goldens with `ALPHA_LAB_UPDATE_GOLDENS=1` only once the field
   stabilizes.

### Example: pooled IC alongside mean-of-daily IC

`mean_ic` on `ExperimentSummary` is the **mean of per-date ICs**
(`experiment.py:85`). If you want pooled IC too — walk-forward already reports
both under `mean_ic` / `pooled_ic_mean` in `walk_forward.py` — compute it in a
helper, emit it to `metrics.json` under a new key, and pin it with a
regression test. Do **not** mutate `ExperimentSummary`; the reviewer note on
Round A explicitly rejects that path.

### What you do NOT need to touch

- Pipeline skeletons in `real_cases/*/pipeline.py` — metrics flow through the
  artifact bundle they already emit.
- Frontend renderers — if the metric goes into `metrics.json`, the existing
  Validation Console already surfaces new scalar keys.

---

## 3. Add a new evaluation profile

### Seam

`src/alpha_lab/research_evaluation_config.py:650` — `_PROFILE_BUILDERS` is a
`dict[str, Callable[[], ResearchEvaluationConfig]]`. Every case runner
dispatches via `get_research_evaluation_config(profile_name)`
(`research_evaluation_config.py:661`).

### Current state — in-source addition, not a public registry

**Important**: `_PROFILE_BUILDERS` is a **private module-level dict**
(leading underscore, see `research_evaluation_config.py:650`). There is no
`register_profile(...)` public API today, and no YAML / plugin loading path
for external profiles. Adding a profile is therefore an **in-source edit**
to `research_evaluation_config.py`, not a runtime registration.

Concretely, to add a profile:

1. Define a builder returning `ResearchEvaluationConfig` inside
   `research_evaluation_config.py` (or a module it imports).
2. Add one line to `_PROFILE_BUILDERS` in the same file.
3. Run the suite — `AVAILABLE_RESEARCH_EVALUATION_PROFILES` is derived
   from the dict, so CLI `--evaluation-profile` help text and
   `/api/evaluation-profiles` update automatically.

```python
# research_evaluation_config.py
def _build_my_strict_profile() -> ResearchEvaluationConfig:
    return ResearchEvaluationConfig(...)


_PROFILE_BUILDERS: dict[str, Callable[[], ResearchEvaluationConfig]] = {
    ...,
    "my_strict": _build_my_strict_profile,
}
```

Once the entry is in the dict, CLI & Web pick it up without further changes:

- `alpha-lab real-case single-factor --evaluation-profile my_strict ...`
- Validation Console `POST /api/projects/<slug>/runs` with
  `evaluation_profile: "my_strict"`.

### What you do NOT need to touch

- `real_cases/single_factor/pipeline.py`, `real_cases/composite/pipeline.py`,
  `real_cases/model_factor/pipeline.py` — all three already call
  `get_research_evaluation_config(evaluation_profile)` at pipeline entry.
- `web_unified.py::submit_run` — passes the profile string through
  verbatim to the pipeline.

### Drift note — deferred, not done

A true **external profile registry** (public `register_profile(...)`,
or YAML/plugin loading at server startup) is tracked as deferred
architectural drift (item A4 in the audit) and is **not implemented** on
this branch. The in-source edit above is the only supported path today.
Mutating `_PROFILE_BUILDERS` from external code works mechanically but is
touching a private symbol and should not be relied on.

---
