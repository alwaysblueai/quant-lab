# Alpha Lab Instructions

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
Default behavior is safe: raises `FileExistsError` if the card already exists.
Pass `overwrite=True` to replace an existing card intentionally.

**Rule**: quant-knowledge is read-only from this repo's perspective except for
`50_experiments/` which alpha-lab writes to via `export_experiment_card`.

---

## Research rules
- Never use future data.
- Never leak labels into features.
- Always state timestamp alignment explicitly.
- Always state train/validation/test split explicitly.
- Always state transaction cost and slippage assumptions.
- Prefer small, auditable functions over long scripts.
- Add tests for edge cases and empty data.

## Canonical data contracts
- Reusable factor outputs must use the long-form schema: `date`, `asset`, `factor`, `value`.
- There must be at most one row per (`date`, `asset`, `factor`).
- Factor values at `t` may only use information available at or before `t`.
- Labels and forward returns must be separate from factor outputs.

## Coding style
- Use Python 3.12.
- Prefer clear type-safe code.
- Keep research code modular.
- Put reusable code under src/alpha_lab.
- Put one-off scripts under scripts.
- Put exploratory work under notebooks.
