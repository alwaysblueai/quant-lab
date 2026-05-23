# docs/ — index

This directory is the **implementation-side documentation** for alpha-lab.
The companion **knowledge layer** (Concept / Method / Factor / Playbook /
Pipeline cards) lives at `/mnt/c/quant/vault/quant-knowledge` — see
`CLAUDE.md` for the boundary.

Every `.md` in this directory is listed below. If you add a new doc, add
a link here so the file does not become orphaned.

---

## Foundations & contracts

- [architecture.md](architecture.md) — layer contracts, data flow, paths
- [system_manual.md](system_manual.md) — API reference and usage patterns
- [data_conventions.md](data_conventions.md) — canonical timestamp, merge, storage rules
- [module_classification.md](module_classification.md) — core module map

## Research workflows

- [research_workflow.md](research_workflow.md) — three-stage idea / experiment flow
- [research_playbook.md](research_playbook.md) — Tier 1 / Tier 2 diagnostics boundaries
- [end_to_end_workflow.md](end_to_end_workflow.md) — full pipeline walkthrough
- [single_factor_quick_pipeline.md](single_factor_quick_pipeline.md) — quick-iteration recipe
- [backend_draft_factor_workflow.md](backend_draft_factor_workflow.md) — Stage 3 draft-factor flow
- [backend_draft_model_workflow.md](backend_draft_model_workflow.md) — Stage 3 draft-model flow
- [factor_promotion_checklist.md](factor_promotion_checklist.md) — gates before a factor goes live
- [change_workflow_rules.md](change_workflow_rules.md) — when code changes need re-runs of which tests

## Research integrity & governance

- [research_integrity.md](research_integrity.md) — no-future-data / no-label-leak rules
- [promotion_log.md](promotion_log.md) — audit log of past promotion decisions

## Intraday

- [intraday_etl_contract.md](intraday_etl_contract.md) — Stage A/B/C 1min ETL contract
- [intraday_factor_workflow.md](intraday_factor_workflow.md) — slim-slice + precompute pattern

## Model factor

- [model_factor_performance_roadmap.md](model_factor_performance_roadmap.md) — Qlib-style perf work
- [model_factor_metric_inventory.md](model_factor_metric_inventory.md) — what each metric measures
- [model_factor_metric_curve_review.md](model_factor_metric_curve_review.md) — how to read the curves
- [model_factor_safe_bfq_35_feature_screening.md](model_factor_safe_bfq_35_feature_screening.md) — feature screening notes
- [model_factor_20_lean_experiment_report_2026-04-29.md](model_factor_20_lean_experiment_report_2026-04-29.md) — dated experiment report
- [model_factor_20_lean_vs_25_consensus_report_2026-04-29.md](model_factor_20_lean_vs_25_consensus_report_2026-04-29.md) — dated comparison

## Single factor

- [single_factor_metric_inventory.md](single_factor_metric_inventory.md) — metric definitions
- [single_factor_metric_curve_review.md](single_factor_metric_curve_review.md) — reading the curves

## Examples & walkthroughs

- [profile_aware_level12_example.md](profile_aware_level12_example.md) — single-case profile walkthrough
- [profile_aware_campaign_level12_example.md](profile_aware_campaign_level12_example.md) — multi-case campaign walkthrough

## Reference & operations

- [developer_guide.md](developer_guide.md) — how to extend the codebase
- [extension_points.md](extension_points.md) — public seams (factor registry, etc.)
- [runtime_stability_runbook.md](runtime_stability_runbook.md) — operational guide for long-running jobs
- [knowledge_bridge.md](knowledge_bridge.md) — research-bridge / vault recall internals
- [research_artifact_system_snapshot.md](research_artifact_system_snapshot.md) — artifact-system point-in-time snapshot
