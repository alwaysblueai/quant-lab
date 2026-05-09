# asym_vol_reversal Deepdive

This folder is Tier 2 research space for `asym_vol_reversal`.

Use it for factor-specific diagnostics that are not part of the stable
single-factor pipeline contract yet:

- regime-specific behavior
- size and liquidity slices
- trailing-return mechanism checks
- exploratory figures and tables

Promotion rule: only move a diagnostic back into Tier 1 when it is useful
across multiple factors and affects the continue/stop verdict.

Import boundary:

- use `alpha_lab.bucket_builders` for stable buckets that may become Tier 1
- use `alpha_lab.research.bucket_builders` for factor-specific or experimental buckets
- use `alpha_lab.research.deepdive_io` for notebook artifact loading and saving
