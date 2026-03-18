# Alpha Lab Agent Rules

## Core research constraints
- Never use future data.
- Never leak labels into features.
- Always state the timestamp alignment explicitly.
- Always specify train/validation/test split.
- Always state transaction cost and slippage assumptions.
- Prefer auditable, modular code over clever code.

## Project structure
- Reusable code goes under `src/alpha_lab`.
- Tests go under `tests`.
- One-off scripts go under `scripts`.
- Exploratory work goes under `notebooks`.

## Coding expectations
- Write small functions with explicit inputs and outputs.
- Add at least one test for every reusable function.
- When editing code, explain what changed and why.
- When proposing factors, state hypothesis, horizon, and risk of leakage.
