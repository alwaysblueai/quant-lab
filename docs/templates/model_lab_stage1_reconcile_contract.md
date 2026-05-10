# Model-Lab Stage1 Reconcile Contract

本文件用于放入网页版 GPT 项目的“来源”。当用户提供 Claude Code 的
**机制候选**（`mechanism_deepdive.md`）和 Codex GUI 的**代码可执行性评审**
（`code_feasibility_review.md`）两份 Stage 1 输出时，网页版 GPT 必须按本合同输出
`model_stage1_reconcile_payload`，不能只做命名、摘要或主观点评。

> 协议变更（2026-05-11）：之前协议把 Claude / Codex 都当 generator（深度组合 vs
> 广度迁移）；新协议下 **Claude = generator（机制候选 ledger），Codex = reviewer
> （代码库可执行性评审）**。reconcile 不再是"两份 ledger 并集"，而是**机制候选
> 矩阵 × 可执行性评审**的合并；详见 `docs/research_workflow.md`。

## 任务边界

当前阶段是 Stage1 mechanism reconcile：

- 合并两个 agent 的机制候选。
- 保留互补机制与 novel synthesis。
- 标注每条机制的可实现性边界。
- 给 Stage2 提供一个可落地的优先入口。

当前不要：

- 不要输出实验结论。
- 不要输出 model 有效性判断。
- 不要生成 `model_candidate_payload`。
- 不要写 case spec。
- 不要引入自定义 feature code、estimator code、sample_weight hook、
  target extension、execution / replay / fill simulation / portfolio construction。

只输出名字、slug、中文名、营销式摘要、单一 best idea，均为不合格。

## 输入材料

用户会提供（来自 `ideas/<idea_id>/`）：

- `mechanism_deepdive.md`（Claude Code generator 输出）：候选机制 ledger。
- `code_feasibility_review.md`（Codex GUI reviewer 输出）：每条候选机制是否在 v1
  `ModelFactorCaseSpec` schema 内可执行；含 `implementation_status`、
  `validator_blockers`、`required_columns_missing`、`spec_fields_touched`。
- `retrieval_pack.md` / `retrieval_log.md`：vault 卡 + 代码库索引快照。
- `manifest.json`：含 `idea_id`、audiences、created_at。
- 当前 base model spec 摘要或路径。
- 当前 feature inventory 摘要。

reconcile 顶层必须保留 `provenance.idea_id`（取自 `manifest.json::idea_id`），
让 Stage 3 artifact 可以反查。

如果缺少 base spec 或 feature inventory，也必须输出 payload，但在
`unresolved_questions` 中标注缺口。如果 reviewer 输出缺失，也必须输出
payload，但 `mechanisms[].implementation_status` 必须保守标注
`needs_extension` 直到下一轮补齐评审。

## 输出格式

必须只输出一个 YAML code block。不要在 code block 外输出正文。

顶层只允许以下键：

```yaml
contract_version: "model_stage1_reconcile_v1"
stage: "stage1_reconcile"
provenance: {}
idea_title: ""
candidate_slug_hint: ""
base_case_spec_path: ""
input_agents: []
code_feasibility_review: {}
mechanisms: []
stage2_entry_recommendation: {}
rejected_as_stage2_primary: []
unresolved_questions: []
quality_gate: {}
```

## YAML schema

```yaml
contract_version: "model_stage1_reconcile_v1"
stage: "stage1_reconcile"
provenance:
  idea_id: "20260511T143000Z__turnover-conditioned-pv"
  audience_chain: ["claude_mechanism", "codex_review"]
  retrieval_pack_sha256: ""        # optional, from ideas/<id>/manifest.json
idea_title: "Turnover-Conditioned Price-Volume Synthesis"
candidate_slug_hint: "turnover_conditioned_pv_synthesis_v1"
base_case_spec_path: "configs/real_cases/model_factor/..."
input_agents:
  - audience: "claude_mechanism"
    role: "mechanism_generator"
    artifact: "ideas/<idea_id>/mechanism_deepdive.md"
  - audience: "codex_review"
    role: "code_feasibility_reviewer"
    artifact: "ideas/<idea_id>/code_feasibility_review.md"
code_feasibility_review:
  reviewer_audience: "codex_review"
  spec_schema_version: ""           # ModelFactorCaseSpec parser version reviewer saw
  validator_rules_seen: []          # validator hard rules reviewer was briefed with
  per_mechanism:
    M1:
      in_v1_contract: true
      implementation_status: "in_contract_spec_variant"
      required_columns_missing: []
      spec_fields_touched: ["feature_columns", "model.family"]
      validator_blockers: []
      reviewer_note: ""
mechanisms:
  - id: "M1"
    name: ""
    family: "feature_interaction | loss_regularization | model_selection | sample_weighting | target_construction | training_window | other"
    implementation_status: "in_contract_spec_variant | partial_in_contract | needs_extension | future_enhancement"
    stage2_priority: "primary_candidate | secondary_candidate | defer"
    source_agents: []
    hypothesis: ""
    agent_data_generating_story: ""
    signal_sketch: ""
    touched_contract_surfaces: []
    allowed_v1_spec_changes: []
    forbidden_v1_changes: []
    data_needs: []
    pit_requirements: []
    artifact_audit_requirements: []
    why_not_parameter_tuning: ""
    inspired_by: []
    fusion_of: []
    novel_delta: ""
    concern: ""
stage2_entry_recommendation:
  primary_mechanism_ids: []
  recommended_candidate_slug: ""
  recommended_implementation_type: "spec_variant"
  reason: ""
  must_preserve_mechanisms_as_context: []
  allowed_changes:
    - "feature_columns"
    - "feature_preprocess"
    - "model"
    - "model_selection"
    - "training"
    - "target"
    - "transaction_cost"
    - "output"
  forbidden_changes:
    - "custom_feature_code"
    - "custom_estimator_code"
    - "sample_weight_hook"
    - "explicit_interaction_feature_builder"
    - "custom_target_code"
    - "portfolio_construction"
    - "execution_replay"
rejected_as_stage2_primary:
  - mechanism_id: ""
    reason: ""
unresolved_questions:
  - ""
quality_gate:
  name_only_response: false
  contains_machine_payload: true
  mechanism_count: 0
  has_primary_stage2_entry: true
  has_in_contract_candidate: true
  preserves_needs_extension_items_as_future: true
```

## 选择规则

- `mechanisms` 建议保留 3-6 条。
- 可以推荐一个 Stage2 primary entry，但不能删除其他机制。
- Stage2 primary entry 优先选择 `in_contract_spec_variant`。
- `needs_extension` 或 `future_enhancement` 机制可以保留为上下文，但不能作为 v1
  Stage2 主候选。
- `mechanisms[].implementation_status` 必须以 `code_feasibility_review.per_mechanism.<id>.implementation_status`
  为准；不要凭机制名字主观判定。
- `mechanisms[].source_agents` 始终为 `["claude_mechanism"]`（机制候选只来自 generator）；
  reviewer 的输入通过 `code_feasibility_review` 反映，不写进 `source_agents`。
- 如果 Claude 提出高创造性机制但 Codex 判断当前合同不可执行，保留为
  `needs_extension`，并把 `validator_blockers` / `required_columns_missing` 写进 reviewer note。
- 如果某条机制在 generator 输出中存在、但在 reviewer 评审中缺失，必须保守降级为
  `needs_extension` 直到补齐评审。

## Model-Lab v1 硬约束

v1 只支持 spec 变体：

- 允许修改 `case_spec_payload` 中现有 ModelFactorCaseSpec 字段。
- 不允许新增 Python feature builder。
- 不允许新增 estimator code。
- 不允许新增 sample_weight 训练入口。
- 不允许新增 target construction code。
- 不允许写 interaction expression，例如 `feature_a * turnover_rate_f`，除非该列已经真实存在于 features 文件。
- 不允许把 Level 3、execution、fill simulation、portfolio construction 写入候选。

## 不合格输出判定

出现以下任一情况，本次输出不合格：

- 只给名字或 slug。
- 只给“建议采用某机制”但没有 YAML payload。
- 没有 `contract_version=model_stage1_reconcile_v1`。
- 没有 `mechanisms`。
- 没有 `stage2_entry_recommendation`。
- 没有区分 `in_contract_spec_variant / needs_extension / future_enhancement`。
- 把 sample_weight、custom target、explicit interaction builder 直接写成 v1 可执行。
- 输出 ranking / best idea 后删除其他机制。
