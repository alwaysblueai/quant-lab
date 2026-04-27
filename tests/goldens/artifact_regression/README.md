# Level 1/2 Artifact Goldens

These goldens provide compact regression coverage for high-value user-facing
Level 1/2 artifacts that schema checks alone cannot protect.

Covered workflows:

- Deterministic single-factor Level 1/2 run
- Deterministic campaign profile comparison output

Snapshot scope (intentional and small):

- Key summary and recommendation JSON outputs
- Key markdown comparison/summary outputs
- Core `research_validation_package.md` output for one deterministic Level 1/2 case
- One comparison CSV surface

Normalization rules used in tests:

- Scrub volatile JSON timestamp fields:
  - `run_timestamp_utc`
  - `created_at_utc`
  - `generated_at_utc`
- Replace temporary absolute output roots with `<OUTPUT_ROOT>`

Refresh intentionally when drift is expected:

```bash
ALPHA_LAB_UPDATE_GOLDENS=1 uv run --no-sync --frozen pytest -q tests/test_artifact_golden_regression.py
```
