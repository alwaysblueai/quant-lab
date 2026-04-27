# Profile-Aware Campaign-Level Level 1/2 Example

This workflow extends the case-level profile-aware example to a compact
campaign-style comparison with multiple candidate cases.

It is strictly Level 1/2 and runs each case through:

1. Level 1 factor verdict
2. Campaign triage
3. Level 2 promotion gate
4. Level 2 portfolio validation recommendation

The deterministic campaign includes three cases:

- `case_stable_promoted`: consistently strong and promoted across profiles
- `case_short_window_sensitive`: strong under looser profile, weaker under stricter profiles
- `case_triage_sensitive`: same Level 1 verdict but different campaign triage outcomes

## Commands

List available profiles:

```bash
alpha-lab profiles
```

Run the campaign-level profile-aware example:

```bash
alpha-lab campaign compare-profiles \
  --source example \
  --output-root-dir dist/examples/profile_aware_campaign_level12 \
  --profiles exploratory_screening default_research stricter_research
```

If your environment requires writable uv cache:

```bash
UV_CACHE_DIR=/tmp/uv-cache alpha-lab campaign compare-profiles \
  --source example \
  --output-root-dir dist/examples/profile_aware_campaign_level12 \
  --profiles exploratory_screening default_research stricter_research
```

To compare an existing campaign definition instead of the built-in lightweight
example:

```bash
alpha-lab campaign compare-profiles \
  --source campaign \
  --campaign-config configs/campaigns/research_campaign_1/campaign.yaml \
  --output-root-dir dist/campaign_profile_comparisons/research_campaign_1 \
  --pair-mode adjacent \
  --artifact-hint-path-mode relative \
  --profiles exploratory_screening default_research stricter_research
```

`--pair-mode` controls profile-pair coverage for transition/reason deltas:

- `adjacent` (default): compare neighboring profile pairs only.
- `all_pairs`: include non-adjacent ordered pairs.

`--artifact-hint-path-mode` controls how artifact hints are rendered:

- `relative` (default): render hints relative to `--output-root-dir` for portability.
- `absolute`: keep full absolute filesystem paths.

Legacy script route remains available for backward compatibility:

```bash
uv run --no-sync --frozen python scripts/run_profile_aware_campaign_level12_example.py \
  --output-root-dir dist/examples/profile_aware_campaign_level12 \
  --profiles exploratory_screening default_research stricter_research
```

Render the local Chinese-first HTML dashboard:

```bash
alpha-lab campaign render-dashboard \
  --comparison-json dist/examples/profile_aware_campaign_level12/campaign_profile_comparison.json \
  --output-html dist/examples/profile_aware_campaign_level12/campaign_profile_dashboard_zh.html \
  --overwrite
```

One-click compare + render + open dashboard:

```bash
bash scripts/run_profile_aware_campaign_dashboard.sh
```

## Output Locations

Primary comparison artifacts:

- `dist/examples/profile_aware_campaign_level12/campaign_profile_comparison.md`
- `dist/examples/profile_aware_campaign_level12/campaign_profile_comparison.json`
- `dist/examples/profile_aware_campaign_level12/campaign_profile_case_matrix.csv`
- `dist/examples/profile_aware_campaign_level12/campaign_profile_dashboard_zh.html` (factor-first dashboard)

Generated case specs and inputs:

- `dist/examples/profile_aware_campaign_level12/cases/<case_name>/single_factor_case.json`
- `dist/examples/profile_aware_campaign_level12/cases/<case_name>/inputs/`

Per-profile case runs:

- `dist/examples/profile_aware_campaign_level12/runs/<profile>/<case_name>/metrics.json`
- `dist/examples/profile_aware_campaign_level12/runs/<profile>/<case_name>/summary.md`
- `dist/examples/profile_aware_campaign_level12/runs/<profile>/<case_name>/case_report.md`

## What To Inspect

1. `campaign_profile_comparison.md` for campaign-level interpretation and case-level deltas.
2. `campaign_profile_comparison.json` for machine-readable differences, including:
   - `field_change_index`
   - `case_comparison[*].changed_fields`
   - `case_comparison[*].profile_sensitivity`
3. `campaign_profile_case_matrix.csv` for quick filtering and spreadsheet comparison.
4. `campaign_profile_dashboard_zh.html` for factor-first local research visualization and drill-down.

## Profile Sensitivity In Practice

A case is treated as profile-sensitive when one or more of these decision fields
changes across profiles:

- `factor_verdict`
- `campaign_triage`
- `promotion_decision`
- `portfolio_validation_recommendation`

Interpretation buckets in the generated summary:

- stable across profiles
- promoted only under looser profiles
- consistently strong
- highly profile-sensitive
