# Campaign Profile Comparison (Level 1/2)

- Source: `example`
- Output root: `<OUTPUT_ROOT>`
- Profiles: `['exploratory_screening', 'default_research']`

## Campaign Cases

- `case_stable_promoted`: High-signal neutralized case engineered to remain stable and promoted across profiles.
- `case_short_window_sensitive`: Noisy stability-sensitive case that changes Level 1 verdict and promotion behavior across profiles.
- `case_triage_sensitive`: Borderline case with profile-dependent campaign triage and portfolio validation recommendation.

## Per-Profile View

### exploratory_screening

| Rank | Case | Status | Factor Verdict | Campaign Triage | Level 2 Promotion | L1->L2 Transition | Portfolio Validation Recommendation |
|---:|---|---|---|---|---|---|---|
| 1 | case_short_window_sensitive | success | Promising but fragile | Fragile / monitor | Promote to Level 2 | Weakened at portfolio level | N/A |
| 2 | case_stable_promoted | success | Promising but fragile | Fragile / monitor | Promote to Level 2 | Weakened at portfolio level | N/A |
| 3 | case_triage_sensitive | success | Promising but fragile | Fragile / monitor | Blocked from Level 2 | Inconclusive transition | N/A |

### default_research

| Rank | Case | Status | Factor Verdict | Campaign Triage | Level 2 Promotion | L1->L2 Transition | Portfolio Validation Recommendation |
|---:|---|---|---|---|---|---|---|
| 1 | case_short_window_sensitive | success | Promising but fragile | Fragile / monitor | Blocked from Level 2 | Inconclusive transition | Not evaluated (not promoted) |
| 2 | case_stable_promoted | success | Promising but fragile | Fragile / monitor | Blocked from Level 2 | Inconclusive transition | Not evaluated (not promoted) |
| 3 | case_triage_sensitive | success | Promising but fragile | Fragile / monitor | Blocked from Level 2 | Inconclusive transition | Not evaluated (not promoted) |

## Case-Level Profile Comparison

### case_stable_promoted

- Sensitivity: `profile_sensitive`
- L1->L2 transition delta: `transition_weakened_under_stricter_profile`
- L1->L2 transition path: `exploratory_screening`=`Weakened at portfolio level` -> `default_research`=`Inconclusive transition`
- Changed fields: `portfolio_validation_recommendation`, `promotion_decision`
- default_research: status=`success`, verdict=`Promising but fragile`, triage=`Fragile / monitor`, promotion=`Blocked from Level 2`, transition=`Inconclusive transition`, portfolio_validation=`Not evaluated (not promoted)`
- default_research blockers: blocked by sharp IC decay under 1-day execution lag
- exploratory_screening: status=`success`, verdict=`Promising but fragile`, triage=`Fragile / monitor`, promotion=`Promote to Level 2`, transition=`Weakened at portfolio level`, portfolio_validation=`N/A`

### case_short_window_sensitive

- Sensitivity: `profile_sensitive`
- L1->L2 transition delta: `transition_weakened_under_stricter_profile`
- L1->L2 transition path: `exploratory_screening`=`Weakened at portfolio level` -> `default_research`=`Inconclusive transition`
- Changed fields: `portfolio_validation_recommendation`, `promotion_decision`
- default_research: status=`success`, verdict=`Promising but fragile`, triage=`Fragile / monitor`, promotion=`Blocked from Level 2`, transition=`Inconclusive transition`, portfolio_validation=`Not evaluated (not promoted)`
- default_research blockers: blocked by sharp IC decay under 1-day execution lag
- exploratory_screening: status=`success`, verdict=`Promising but fragile`, triage=`Fragile / monitor`, promotion=`Promote to Level 2`, transition=`Weakened at portfolio level`, portfolio_validation=`N/A`

### case_triage_sensitive

- Sensitivity: `profile_sensitive`
- L1->L2 transition delta: `transition_stable`
- L1->L2 transition path: `exploratory_screening`=`Inconclusive transition` -> `default_research`=`Inconclusive transition`
- Changed fields: `portfolio_validation_recommendation`
- default_research: status=`success`, verdict=`Promising but fragile`, triage=`Fragile / monitor`, promotion=`Blocked from Level 2`, transition=`Inconclusive transition`, portfolio_validation=`Not evaluated (not promoted)`
- default_research blockers: blocked by unstable rolling evidence, blocked by high uncertainty overlap, blocked by sharp IC decay under 1-day execution lag
- exploratory_screening: status=`success`, verdict=`Promising but fragile`, triage=`Fragile / monitor`, promotion=`Blocked from Level 2`, transition=`Inconclusive transition`, portfolio_validation=`N/A`
- exploratory_screening blockers: blocked by unstable rolling evidence

## Case Evidence Index

### case_short_window_sensitive

- Profiles observed: `exploratory_screening`, `default_research`
- Sensitivity: `profile_sensitive`; profile_delta_label=`transition_weakened_under_stricter_profile`
- Changed fields: `portfolio_validation_recommendation`, `promotion_decision`
- Factor verdict by profile: default_research=`Promising but fragile`; exploratory_screening=`Promising but fragile`
- Campaign triage by profile: default_research=`Fragile / monitor`; exploratory_screening=`Fragile / monitor`
- Promotion decision by profile: default_research=`Blocked from Level 2`; exploratory_screening=`Promote to Level 2`
- Portfolio robustness by profile: default_research=`Not evaluated (not promoted)`; exploratory_screening=`N/A`
- L1->L2 transition label by profile: default_research=`Inconclusive transition`; exploratory_screening=`Weakened at portfolio level`
- Key reason hints by profile: default_research: campaign triage: Fragile / monitor | promotion decision: Blocked from Level 2 | portfolio recommendation: Not evaluated (not promoted); exploratory_screening: campaign triage: Fragile / monitor | promotion decision: Promote to Level 2 | promotion reason: uncertainty remains supportive
- Artifact hints by profile: default_research: runs/default_research/case_short_window_sensitive/metrics.json | runs/default_research/case_short_window_sensitive/summary.md; exploratory_screening: runs/exploratory_screening/case_short_window_sensitive/metrics.json | runs/exploratory_screening/case_short_window_sensitive/summary.md

### case_stable_promoted

- Profiles observed: `exploratory_screening`, `default_research`
- Sensitivity: `profile_sensitive`; profile_delta_label=`transition_weakened_under_stricter_profile`
- Changed fields: `portfolio_validation_recommendation`, `promotion_decision`
- Factor verdict by profile: default_research=`Promising but fragile`; exploratory_screening=`Promising but fragile`
- Campaign triage by profile: default_research=`Fragile / monitor`; exploratory_screening=`Fragile / monitor`
- Promotion decision by profile: default_research=`Blocked from Level 2`; exploratory_screening=`Promote to Level 2`
- Portfolio robustness by profile: default_research=`Not evaluated (not promoted)`; exploratory_screening=`N/A`
- L1->L2 transition label by profile: default_research=`Inconclusive transition`; exploratory_screening=`Weakened at portfolio level`
- Key reason hints by profile: default_research: campaign triage: Fragile / monitor | promotion decision: Blocked from Level 2 | portfolio recommendation: Not evaluated (not promoted); exploratory_screening: campaign triage: Fragile / monitor | promotion decision: Promote to Level 2 | promotion reason: uncertainty remains supportive
- Artifact hints by profile: default_research: runs/default_research/case_stable_promoted/metrics.json | runs/default_research/case_stable_promoted/summary.md; exploratory_screening: runs/exploratory_screening/case_stable_promoted/metrics.json | runs/exploratory_screening/case_stable_promoted/summary.md

### case_triage_sensitive

- Profiles observed: `exploratory_screening`, `default_research`
- Sensitivity: `profile_sensitive`; profile_delta_label=`transition_stable`
- Changed fields: `portfolio_validation_recommendation`
- Factor verdict by profile: default_research=`Promising but fragile`; exploratory_screening=`Promising but fragile`
- Campaign triage by profile: default_research=`Fragile / monitor`; exploratory_screening=`Fragile / monitor`
- Promotion decision by profile: default_research=`Blocked from Level 2`; exploratory_screening=`Blocked from Level 2`
- Portfolio robustness by profile: default_research=`Not evaluated (not promoted)`; exploratory_screening=`N/A`
- L1->L2 transition label by profile: default_research=`Inconclusive transition`; exploratory_screening=`Inconclusive transition`
- Key reason hints by profile: default_research: campaign triage: Fragile / monitor | promotion decision: Blocked from Level 2 | portfolio recommendation: Not evaluated (not promoted); exploratory_screening: campaign triage: Fragile / monitor | promotion decision: Blocked from Level 2 | promotion reason: blocked by unstable rolling evidence
- Artifact hints by profile: default_research: runs/default_research/case_triage_sensitive/metrics.json | runs/default_research/case_triage_sensitive/summary.md; exploratory_screening: runs/exploratory_screening/case_triage_sensitive/metrics.json | runs/exploratory_screening/case_triage_sensitive/summary.md

## Campaign-Level Interpretation

- Minimum support thresholds: minimum_cases_per_transition_label=2, minimum_cases_per_transition_label_for_reason_shift=2, minimum_cases_with_reasons_per_transition_label=2, minimum_cases_with_transition_label=3, minimum_observed_cases_per_profile_pair=3, minimum_reason_bucket_count_for_dominance=2, minimum_reason_bucket_count_for_shift=2, minimum_transition_labels_with_reason_evidence_per_profile_pair=2
### Compact Comparison Summary

- Transition stability: 1/3 stable (33.3%), sensitive=2; representative=case_triage_sensitive; pointers=case_triage_sensitive [default_research] runs/default_research/case_triage_sensitive/metrics.json; case_triage_sensitive [exploratory_screening] runs/exploratory_screening/case_triage_sensitive/metrics.json; support=tentative due to low support.
- Most profile-sensitive cases: case_short_window_sensitive (changed_fields=2, delta=transition_weakened_under_stricter_profile, pointer=case_short_window_sensitive [default_research] runs/default_research/case_short_window_sensitive/metrics.json); case_stable_promoted (changed_fields=2, delta=transition_weakened_under_stricter_profile, pointer=case_stable_promoted [default_research] runs/default_research/case_stable_promoted/metrics.json); case_triage_sensitive (changed_fields=1, delta=transition_stable, pointer=case_triage_sensitive [default_research] runs/default_research/case_triage_sensitive/metrics.json).
- Strongest profile-pair shift: exploratory_screening -> default_research changed=2/3 (66.7%), reason_shifted_labels=0/2, top_flows=Weakened at portfolio level -> Inconclusive transition (2, cases=case_stable_promoted,case_short_window_sensitive), representative_cases=case_stable_promoted,case_short_window_sensitive,case_triage_sensitive; support=reason shift observed, but only in a small number of cases.
- Stricter profile impact: promotion (promotion_reduction=2, robustness_reduction=0, adjacent_pairs=1); support=tentative due to low support.

- Cases stable across profiles: none
- Cases profile-sensitive: `case_stable_promoted`, `case_short_window_sensitive`, `case_triage_sensitive`
- Cases promoted only under looser profiles: `case_stable_promoted`, `case_short_window_sensitive`
- Cases consistently strong: none
- Cases highly profile-sensitive: none
- Transition-stable cases (L1->L2 labels): `case_triage_sensitive`
- Transition-sensitive cases (L1->L2 labels): `case_stable_promoted`, `case_short_window_sensitive`
- L1->L2 transition distribution by profile:
- exploratory_screening (n=3): Confirmed=0, Weakened=2, Fragile=0, Improved=0, Inconclusive=1; interpretation=Most common transition outcome is `Weakened at portfolio level` (2/3 cases).; support=tentative due to low support
- exploratory_screening representative cases: Weakened at portfolio level: case_stable_promoted, case_short_window_sensitive; Inconclusive transition: case_triage_sensitive
- exploratory_screening artifact hints: Weakened at portfolio level: runs/exploratory_screening/case_stable_promoted/metrics.json; Inconclusive transition: runs/exploratory_screening/case_triage_sensitive/metrics.json
- exploratory_screening dominant transition reasons: Weakened at portfolio level: `campaign triage: Fragile / monitor` (2, 100.0%; cases=case_stable_promoted,case_short_window_sensitive); Inconclusive transition: sparse transition evidence (`blocked by unstable rolling evidence` (1, 100.0%; cases=case_triage_sensitive))
- default_research (n=3): Confirmed=0, Weakened=0, Fragile=0, Improved=0, Inconclusive=3; interpretation=Most common transition outcome is `Inconclusive transition` (3/3 cases).; support=tentative due to low support
- default_research representative cases: Inconclusive transition: case_stable_promoted, case_short_window_sensitive
- default_research artifact hints: Inconclusive transition: runs/default_research/case_stable_promoted/metrics.json
- default_research dominant transition reasons: Inconclusive transition: `campaign triage: Fragile / monitor` (3, 100.0%; cases=case_stable_promoted,case_short_window_sensitive)
- L1->L2 transition profile-delta matrix (adjacent profiles):
- exploratory_screening -> default_research: observed=3, stable=1, changed=2, missing=0; support=tentative due to low support
- exploratory_screening -> default_research representative cases: case_stable_promoted, case_short_window_sensitive, case_triage_sensitive
- exploratory_screening -> default_research artifact hints: case_stable_promoted: exploratory_screening=runs/exploratory_screening/case_stable_promoted/metrics.json; default_research=runs/default_research/case_stable_promoted/metrics.json; case_short_window_sensitive: exploratory_screening=runs/exploratory_screening/case_short_window_sensitive/metrics.json; default_research=runs/default_research/case_short_window_sensitive/metrics.json
- exploratory_screening -> default_research pair counts: Weakened at portfolio level -> Inconclusive transition: 2 (66.7%); cases=case_stable_promoted, case_short_window_sensitive; Inconclusive transition -> Inconclusive transition: 1 (33.3%); cases=case_triage_sensitive
- L1->L2 dominant reason deltas by profile pair (adjacent profiles):
- exploratory_screening -> default_research: observed_labels=2, shifted_labels=0, stable_labels=0, tentative_shifted_labels=2, added=3, removed=3, increased=0, decreased=0; support=reason shift observed, but only in a small number of cases
- exploratory_screening -> default_research representative cases: case_stable_promoted, case_short_window_sensitive, case_triage_sensitive
- exploratory_screening -> default_research artifact hints: runs/exploratory_screening/case_stable_promoted/metrics.json; runs/exploratory_screening/case_short_window_sensitive/metrics.json
- exploratory_screening -> default_research [Weakened at portfolio level]: reason shift observed, but only in a small number of cases; exploratory_screening dominant=`campaign triage: Fragile / monitor` 2/2; cases=case_stable_promoted,case_short_window_sensitive; `promotion decision: Promote to Level 2` 2/2; cases=case_stable_promoted,case_short_window_sensitive; default_research dominant=none; shifts=removed: `campaign triage: Fragile / monitor` 2/2 -> 0/0 (-100.0pp); cases=case_stable_promoted,case_short_window_sensitive; support=reason shift observed, but only in a small number of cases
- exploratory_screening -> default_research [Inconclusive transition]: reason shift observed, but only in a small number of cases; exploratory_screening dominant=none; default_research dominant=`campaign triage: Fragile / monitor` 3/3; cases=case_stable_promoted,case_short_window_sensitive; `portfolio recommendation: Not evaluated (not promoted)` 3/3; cases=case_stable_promoted,case_short_window_sensitive; shifts=added: `campaign triage: Fragile / monitor` 0/1 -> 3/3 (+100.0pp); cases=case_stable_promoted,case_short_window_sensitive; support=reason shift observed, but only in a small number of cases

## Artifacts

- `campaign_profile_comparison.md` (human-readable summary)
- `campaign_profile_comparison.json` (machine-readable profile deltas)
- `campaign_profile_case_matrix.csv` (flat case/profile matrix)

