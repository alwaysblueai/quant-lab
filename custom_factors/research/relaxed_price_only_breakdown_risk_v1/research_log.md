# relaxed_price_only_breakdown_risk_v1 research log

idea_id: `20260523T132055Z__30`
stage2_payload_sha256: `f4c6a0ede797860877439f4dd359d214e0c0b4c0e78e735e6d4ae74b8d3fd23f`
audience_chain: claude -> codex -> web_gpt_stage2
primary_mechanism: M_relaxed_price_only_breakdown_risk_v1 (`relaxed_price_only_breakdown_risk`)

## v1 (2026-05-26)

- Source: materialized exactly from `ideas/20260523T132055Z__30/stage2_payload_v1.yaml.factor_json_payload`, except replacing the placeholder `provenance.stage2_payload_sha256`.
- Hash materialization: `sha256(json.dumps(factor_json_payload_with_stage2_payload_sha256_empty, ensure_ascii=False, sort_keys=True).encode("utf-8"))`.
- Stage1 usage: `stage1_reconcile.yaml` was used only as mechanism background; it did not change `code`, `required_columns`, `pit_assumption`, or direction.
- Case: `configs/real_cases/single_factor/relaxed_price_only_breakdown_risk_v1_v1.yaml`.
- Profile: `exploratory_screening`.
2026-05-26 v1 relax case=v1 art=ok RankIC=.009 IR=.111 verdict=drop
