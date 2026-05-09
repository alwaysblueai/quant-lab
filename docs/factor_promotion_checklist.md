# Factor Promotion Checklist

The protocol for promoting a research-stage single-factor into the formally
registered (frontend-visible) factor library. This is the "research → promoted"
gate; it is **separate** from the Tier 1/Tier 2 diagnostic-promotion log in
`docs/promotion_log.md`.

## Where this fits

```
Stage 3 (web GPT mechanism → CLI implementation)
    │
    ├─ exploration   ──→ exploratory_screening profile (fast GO/NO-GO)
    │
    ├─ promising?    ──→ default_research profile      (full PIT/regime/cost)
    │
    └─ pass gate?    ──→ promote → frontend-registered → full image report
```

`stricter_research` profile is **not** required for the gate at this stage —
the thresholds below applied on `default_research` artifacts are sufficient.
Revisit if false-positive rate becomes a problem.

Backend draft-factor runs before this gate are standardized in
[`backend_draft_factor_workflow.md`](backend_draft_factor_workflow.md). Codex GUI
Stage 3 handoffs should use
[`templates/stage3_backend_draft_factor_prompt.md`](templates/stage3_backend_draft_factor_prompt.md)
and the fixed execution envelope in
[`templates/codex_gui_stage3_execution_envelope.md`](templates/codex_gui_stage3_execution_envelope.md).
Research drafts must stay under `custom_factors/research/` and run through
`alpha-lab validate-draft-factor` followed by
`alpha-lab real-case single-factor run`; do not add one-off scripts for factor
iteration.

## On-disk layout

```
custom_factors/
├── research/
│   └── <factor>/
│       ├── factor.json          # builder code + description (workshop-managed)
│       └── research_log.md      # one-line iteration history
└── promoted/
    └── <factor>/
        ├── factor.json
        └── promotion_card.md    # written at promotion time
```

`research_log.md` lines (≤ 80 chars):

```
2026-05-08  v1 baseline                case=foo_v1.yaml  art=outputs/.../foo_v1  RankIC=0.018 IR=0.21
2026-05-09  +industry-neutralize       case=foo_v2.yaml  art=outputs/.../foo_v2  RankIC=0.024 IR=0.34  ← worth default_research
```

Do not write decision documents here — only the trail.

## Promotion thresholds (industry starter values)

These apply to artifacts produced under `--evaluation-profile default_research`
on a multi-year sample. **All must hold simultaneously**; partial passes stay in
research.

| # | Metric | Threshold | Source artifact |
|---|---|---|---|
| 1 | RankIC mean | ≥ **0.02** | `metrics.json` |
| 2 | RankIC IR (mean / std) | ≥ **0.30** | `metrics.json` |
| 3 | PIT scan | **0 violations** (hard) | PIT scan report |
| 4 | Regime stability | RankIC same sign in **≥ 3 of 4** regimes | regime split |
| 5 | Correlation with promoted suite | spearman ≤ **0.60** vs every existing promoted factor | cross-corr |
| 6 | Cost robustness | IR ≥ **0.20** at one-way cost = **5 bps** | cost-stress sweep |
| 7 | Sample length | ≥ **3 years** of in-sample evaluation window | case YAML |

These are starter values — tighten if the promoted suite shows alpha decay,
loosen only with written justification in the promotion card.

## Promotion physical actions

When all thresholds pass:

1. **Copy** the factor (do not move — research log keeps the iteration record):
   ```bash
   cp -r custom_factors/research/<factor>/ custom_factors/promoted/<factor>/
   ```
2. **Write `custom_factors/promoted/<factor>/promotion_card.md`** with:
   - promotion date
   - case YAML path used for the gate run
   - artifact path used for the gate run
   - actual values for thresholds 1–7 (one line each)
   - any deviation from defaults + justification
3. **Append to `docs/factor_promotion_checklist.md` § Promotion log** (table at
   bottom of this doc) — one row per promoted factor.
4. **Frontend registration**: open the workshop UI; the factor will already be
   loaded from `custom_factors/promoted/<factor>/factor.json` by the loader.
   Run a full case from the UI to generate the image-rich report for archival.

## Demotion / parking

If a promoted factor later fails (alpha decay, regime break), do not silently
delete. Move `promoted/<f>/` back to `research/<f>/`, append a "demoted" row to
the promotion log, and add a one-line reason to its `research_log.md`.

## Promotion log

| Date | Factor | Case YAML | Artifact | RankIC | IR | Notes |
|---|---|---|---|---|---|---|
| _none yet_ | | | | | | |
