# Stage 2 Input — 用价格、换手率与流动性相关的安全技术特征，对未来短期（约一周）横截面收益排序做轻量预测；偏好低复杂度线性族、成交活跃但波动不过热的股票更可信

idea_id: `20260525T115846Z__idea-f3ca571d`
lab: `model_factor`

## 协议（2026-05-11，docs/end_to_end_workflow.md）
Stage 1 = 两引擎并行执行**同一任务**（generator + reviewer 合一）。
网页 GPT 在 Stage 2 综合两份输出取长补短，输出唯一 `model_candidate_payload`。

## 网页 GPT 项目 'sources' 必须包含（且只包含）
- `docs/templates/model_lab_source_pack.md`
- `docs/templates/model_lab_stage1_reconcile_contract.md`
- `docs/templates/model_lab_stage2_candidate_contract.md`

Stage 3 envelope 是给 Codex GUI 的开场提示，**不**放网页 GPT 项目。

## 输入产物（Stage 1）
- `claude` prompt: `prompt_claude.md`
  output: `stage1_claude.md`
- `codex` prompt: `prompt_codex.md`
  output: `stage1_codex.md`

## 输出产物（Stage 2）
- Step 2.1 reconcile → 写入 `ideas/<idea_id>/stage1_reconcile.yaml`
  - 输入：两份 stage1_<engine>.md（含 Part A 机制 + Part B 评审）
    + retrieval_pack.md
  - 输出 YAML 顶层含 `provenance.idea_id`、`mechanisms[]`、
    合并的 `code_feasibility_review`、`stage2_entry_recommendation`
  - 取两引擎机制并集；implementation_status 冲突时取更保守那方
- Step 2.2 candidate → 写入 `ideas/<idea_id>/stage2_payload_v<n>.json`
  - 输出完整 `model_candidate_payload`（不接受 patch）
  - 必填 `provenance.{idea_id, stage2_payload_sha256, audience_chain}`
  - `audience_chain` 固定为 
    `["claude", "codex", "web_gpt_stage2"]`（两引擎对称协议）

## 迭代回灌（Stage 3 → Stage 2）
Stage 3 每轮跑完后，Codex GUI 直接把摘要段落粘回同一个网页 GPT 项目。
网页 GPT 出 v<n+1> payload，`idea_id` 不变，`stage2_payload_sha256` 更新。
不需要中转文件。

## quality gate
- mechanisms 至少 1 条
- code_feasibility_review 必须覆盖每条机制的 implementation_status
- payload provenance.idea_id 与 `manifest.json::idea_id` 一致
- 不把 `operative_claims` 当作 kill 条件
