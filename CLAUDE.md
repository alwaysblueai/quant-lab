# Alpha Lab Instructions

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
