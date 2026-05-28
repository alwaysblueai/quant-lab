# price_only_breakdown_risk_v1 research log

idea_id: `20260523T132055Z__30`
stage2_payload_sha256: `947bb0e4d630c1311b04c3434c9dba1d21aed80411a6fa38ea2ea200e4a1747e`
audience_chain: claude -> codex -> web_gpt_stage2
primary_mechanism: M_price_only_breakdown_risk_v1 (`price_only_breakdown_risk`)

## v1 (2026-05-26)

- Source: materialized exactly from `ideas/20260523T132055Z__30/stage2_payload_v1.yaml.factor_json_payload`, except replacing the placeholder `provenance.stage2_payload_sha256`.
- Hash materialization: `sha256(json.dumps(factor_json_payload_with_stage2_payload_sha256_empty, ensure_ascii=False, sort_keys=True).encode("utf-8"))`.
- Stage1 usage: `stage1_reconcile.yaml` was used only as mechanism background; it did not change `code`, `required_columns`, `pit_assumption`, or direction.
- Case: `configs/real_cases/single_factor/price_only_breakdown_risk_v1_v1.yaml`.
- Profile: `exploratory_screening`.
2026-05-26 v1 price case=v1 art=ok RankIC=.014 IR=.184 verdict=drop
