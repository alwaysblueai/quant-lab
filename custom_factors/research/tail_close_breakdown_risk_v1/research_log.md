# tail_close_breakdown_risk_v1 research log

idea_id: `20260523T132055Z__30`
stage2_payload_sha256: `08804fc172b75ba9e5e235dab03996ea6a90df4f60b68fa9c4557162988df826`
audience_chain: claude -> codex -> web_gpt_stage2
primary_mechanism: M_tail_close_breakdown_risk_v1 (`tail_close_breakdown_risk`)

## v1 (2026-05-26)

- Source: materialized exactly from `ideas/20260523T132055Z__30/stage2_payload_v1.yaml.factor_json_payload`, except replacing the placeholder `provenance.stage2_payload_sha256`.
- Hash materialization: `sha256(json.dumps(factor_json_payload_with_stage2_payload_sha256_empty, ensure_ascii=False, sort_keys=True).encode("utf-8"))`.
- Stage1 usage: `stage1_reconcile.yaml` was used only as mechanism background; it did not change `code`, `required_columns`, `pit_assumption`, or direction.
- Case: `configs/real_cases/single_factor/tail_close_breakdown_risk_v1_v1.yaml`.
- Profile: `exploratory_screening`.
2026-05-26 v1 risk case=v1 art=ok RankIC=.013 IR=.178 verdict=drop
