# Model-Lab Stage1 Reconcile Contract

本文件用于放入网页版 GPT 项目的“来源”。当用户提供 Claude Code 与 Codex GUI
两份 Model-Lab Stage1 机制讨论结果时，网页版 GPT 必须按本合同输出
`model_stage1_reconcile_payload`，不能只做命名、摘要或主观点评。

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

用户会提供：

- Claude Code Stage1 输出。
- Codex GUI Stage1 输出。
- 当前 base model spec 摘要或路径。
- 当前 feature inventory 摘要。
- 已有 model/factor/run 检索上下文。

如果缺少 base spec 或 feature inventory，也必须输出 payload，但在
`unresolved_questions` 中标注缺口。

## 输出格式

必须只输出一个 YAML code block。不要在 code block 外输出正文。

顶层只允许以下键：

```yaml
contract_version: "model_stage1_reconcile_v1"
stage: "stage1_reconcile"
idea_title: ""
candidate_slug_hint: ""
base_case_spec_path: ""
input_agents: []
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
idea_title: "Turnover-Conditioned Price-Volume Synthesis"
candidate_slug_hint: "turnover_conditioned_pv_synthesis_v1"
base_case_spec_path: "configs/real_cases/model_factor/..."
input_agents:
  - agent: "claude"
    role: "mechanism_depth"
  - agent: "codex"
    role: "execution_contract_review"
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
- 如果两个 agent 都提到同一机制，要合并并在 `source_agents` 中保留双方。
- 如果 Claude 提出高创造性机制但 Codex 判断当前合同不可执行，应保留为
  `needs_extension`，并写清楚为什么不能进入 v1。
- 如果 Codex 提出可执行但机制较窄，应保留为 Stage2 primary 的候选。

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
