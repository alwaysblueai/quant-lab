# v1_product_breakdown_risk_v1 research log

idea_id: `20260523T132055Z__30`
stage2_payload_sha256: `4c218fb222aabcc9fe5e9f19df327ee8f446769b6197349068954528bf3e6596`
audience_chain: claude -> codex -> web_gpt_stage2
primary_mechanism: M_v1_product_breakdown_risk_v1 (`v1_product_breakdown_risk`)

## v1 (2026-05-26)

- Source: materialized exactly from `ideas/20260523T132055Z__30/stage2_payload_v1.yaml.factor_json_payload`, except replacing the placeholder `provenance.stage2_payload_sha256`.
- Hash materialization: `sha256(json.dumps(factor_json_payload_with_stage2_payload_sha256_empty, ensure_ascii=False, sort_keys=True).encode("utf-8"))`.
- Stage1 usage: `stage1_reconcile.yaml` was used only as mechanism background; it did not change `code`, `required_columns`, `pit_assumption`, or direction.
- Case: `configs/real_cases/single_factor/v1_product_breakdown_risk_v1_v1.yaml`.
- Profile: `exploratory_screening`.
2026-05-26 v1 prod case=v1 art=ok RankIC=-.008 IR=-.094 verdict=drop
