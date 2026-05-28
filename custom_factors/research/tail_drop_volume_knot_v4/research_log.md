# tail_drop_volume_knot_v4 research log

idea_id: `20260523T132055Z__30`
stage2_payload_sha256: `a43eb822407c9e8ab623b56a79f767c2907a4e76ce692a2daba47352635acac9`
audience_chain: claude -> codex -> web_gpt_stage2
primary_mechanism: M_condition_count_bonus_v4 (`tail_drop_volume_knot`)

## v1 (2026-05-26)

- Source: materialized exactly from `ideas/20260523T132055Z__30/stage2_payload_v1.yaml.factor_json_payload`, except replacing the placeholder `provenance.stage2_payload_sha256`.
- Hash materialization: `sha256(json.dumps(factor_json_payload_with_stage2_payload_sha256_empty, ensure_ascii=False, sort_keys=True).encode("utf-8"))`.
- Stage1 usage: `stage1_reconcile.yaml` was used only as mechanism background; it did not change `code`, `required_columns`, `pit_assumption`, or direction.
- Case: `configs/real_cases/single_factor/tail_drop_volume_knot_v4_v1.yaml`.
- Profile: `exploratory_screening`.
2026-05-26 v1 gate case=v1 art=ok RankIC=-.013 IR=-.178 verdict=drop
2026-05-26 v1 rerun case=v1 art=ok RankIC=-.013 IR=-.178 verdict=drop
