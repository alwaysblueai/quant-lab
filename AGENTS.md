# Alpha Lab Agent Rules

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
