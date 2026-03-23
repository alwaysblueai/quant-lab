# Alpha Lab Agent Rules

## Knowledge reference layer

This repo is the **implementation layer**.  The **knowledge layer** lives at:

```
/mnt/c/quant/vault/quant-knowledge
```

Before implementing a factor or designing an experiment, consult:

| Need | Where to look |
|------|---------------|
| Factor definition / hypothesis | `30_factors/Factor - *.md` |
| Algorithm / estimation method | `20_methods/Method - *.md` |
| Pre-flight checklist | `60_playbooks/Playbook - *.md` |
| End-to-end pipeline pattern | `80_pipelines/Pipeline - *.md` |
| Concept definition | `10_concepts/Concept - *.md` |

After an experiment, export the result card back to quant-knowledge:

```python
from alpha_lab.reporting import export_experiment_card
path = export_experiment_card(result, name="momentum-5d-Ashare")
# writes to /mnt/c/quant/vault/quant-knowledge/50_experiments/Exp - YYYYMM - momentum-5d-Ashare.md
```

The vault root must already exist.  The `50_experiments/` subdir is created on demand.

By default the call raises `FileExistsError` if the card already exists.
Pass `overwrite=True` to replace an existing card intentionally.

The generated card marks Setup and Results as **auto-generated** (do not edit
manually) and leaves Interpretation, Next Steps, Open Questions, and Notes as
**manual sections** for researcher completion.

**Rule**: quant-knowledge is read-only from this repo's perspective except for
`50_experiments/` which alpha-lab writes to via `export_experiment_card`.

---

## Core research constraints
- Never use future data.
- Never leak labels into features.
- Always state the timestamp alignment explicitly.
- Always specify train/validation/test split.
- Always state transaction cost and slippage assumptions.
- Prefer auditable, modular code over clever code.

## Canonical data contracts
- Reusable factor outputs must be long-form with columns: `date`, `asset`, `factor`, `value`.
- There must be at most one row per (`date`, `asset`, `factor`).
- Factor values at timestamp `t` may only use information available at or before `t`.
- Labels and forward returns must be stored separately from factor outputs.

## Project structure
- Reusable code goes under `src/alpha_lab`.
- Tests go under `tests`.
- One-off scripts go under `scripts`.
- Exploratory work goes under `notebooks`.

## Coding expectations
- Write small functions with explicit inputs and outputs.
- Add at least one test for every reusable function.
- Add regression tests for known leakage or alignment risks.
- When editing code, explain what changed and why.
- When proposing factors, state hypothesis, horizon, and risk of leakage.

## Remediation rules
- Prefer the smallest coherent fix over broad redesign.
- Before editing, identify the canonical schema and enforce it consistently.
- Any factor implementation must be tested against sparse per-asset date histories.
- README must describe only implemented capabilities, not aspirational ones.
- Do not leave critical framework files untracked once they are part of the claimed repository surface.
