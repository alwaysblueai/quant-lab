# Single-Factor Stage1 Reconcile Contract

本文件用于放入网页版 GPT 项目的"来源"。当用户提供 Claude Code 的
**机制候选**（`mechanism_deepdive.md`）和 Codex GUI 的**代码可执行性评审**
（`code_feasibility_review.md`）两份单因子 Stage 1 输出时，网页版 GPT 必须按本合同
输出 `single_factor_stage1_reconcile_payload`，不能只做命名、摘要或主观点评。

> 协议（2026-05-11）：Stage 1 角色 = generator + reviewer。Claude 输出机制候选
> ledger，Codex 输出可执行性评审；reconcile 是"机制 × 评审"合并，不是"两份
> ledger 并集"。详见 `docs/research_workflow.md`。

## 任务边界

当前阶段是 Stage1 mechanism reconcile：

- 合并 generator 提出的机制候选 + reviewer 的可执行性评审。
- 保留互补机制与 novel synthesis。
- 标注每条机制的可实现性边界（基于 reviewer 输出）。
- 给 Stage2 提供一个可落地的优先入口。

当前不要：

- 不要输出实验结论。
- 不要输出因子有效性判断。
- 不要生成 `factor_json_payload`。
- 不要写因子 code。
- 不要引入 portfolio construction、execution / replay / fill simulation。

只输出名字、slug、中文名、营销式摘要、单一 best idea，均为不合格。

## 输入材料

用户会提供（来自 `ideas/<idea_id>/`）：

- `mechanism_deepdive.md`（Claude Code generator 输出）：候选机制 ledger。
- `code_feasibility_review.md`（Codex GUI reviewer 输出）：每条候选机制是否在
  `factor.json` schema + validator 硬约束内可执行。
- `retrieval_pack.md` / `retrieval_log.md`：vault 卡 + 代码库索引快照。
- `manifest.json`：含 `idea_id`、audiences、created_at。
- 可用字段列表（必填，单因子需要 prices schema + 已 ETL 的 intraday/daily 列名）。
- 已有 `custom_factors/{promoted,research}/` 列表（防重做）。

reconcile 顶层必须保留 `provenance.idea_id`（取自 `manifest.json::idea_id`）。

如果 reviewer 输出缺失，必须把 `mechanisms[].implementation_status` 保守标
`needs_extension` 直到下一轮补齐评审。

## 输出格式

必须只输出一个 YAML code block。不要在 code block 外输出正文。

顶层只允许以下键：

```yaml
contract_version: "single_factor_stage1_reconcile_v1"
stage: "stage1_reconcile"
provenance: {}
idea_title: ""
candidate_slug_hint: ""
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
contract_version: "single_factor_stage1_reconcile_v1"
stage: "stage1_reconcile"
provenance:
  idea_id: "20260511T143000Z__signed-jump-reversal"
  audience_chain: ["claude_mechanism", "codex_review"]
  retrieval_pack_sha256: ""
idea_title: ""
candidate_slug_hint: ""
input_agents:
  - audience: "claude_mechanism"
    role: "mechanism_generator"
    artifact: "ideas/<idea_id>/mechanism_deepdive.md"
  - audience: "codex_review"
    role: "code_feasibility_reviewer"
    artifact: "ideas/<idea_id>/code_feasibility_review.md"
code_feasibility_review:
  reviewer_audience: "codex_review"
  factor_json_schema_keys: []          # required keys reviewer was briefed with
  validator_rules_seen: []
  per_mechanism:
    M1:
      in_v1_contract: true
      implementation_status: "in_contract_factor_def | partial_in_contract | needs_extension | future_enhancement"
      required_columns_missing: []
      validator_blockers: []           # e.g. ["future_leakage_pattern", "cross_section_full_sample_stat"]
      reviewer_note: ""
mechanisms:
  - id: "M1"
    name: ""
    family: "intraday_microstructure | regime_conditioned | crowding_proxy | anchor_drift | dispersion_term_structure | other"
    implementation_status: "in_contract_factor_def | partial_in_contract | needs_extension | future_enhancement"
    stage2_priority: "primary_candidate | secondary_candidate | defer"
    source_agents: ["claude_mechanism"]
    hypothesis: ""
    agent_data_generating_story: ""
    signal_sketch: ""
    required_columns: []
    pit_requirements: []
    inspired_by: []
    fusion_of: []
    novel_delta: ""
    concern: ""
stage2_entry_recommendation:
  primary_mechanism_ids: []
  recommended_factor_slug: ""
  reason: ""
  must_preserve_mechanisms_as_context: []
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
- `mechanisms[].implementation_status` 必须以 `code_feasibility_review` 为准。
- `mechanisms[].source_agents` 始终为 `["claude_mechanism"]`（机制只来自 generator）。
- Stage2 primary entry 优先 `in_contract_factor_def`：可以用现有 prices /
  intraday / volume / volatility 列直接定义因子，不引入未来收益、不全样本统计。
- `needs_extension` 机制保留为上下文，但不进入 v1 primary。

## Single-factor v1 硬约束

v1 只支持 **可在 `custom_factors/research/<f>/factor.json` 内一次写完** 的因子：

- `code` 必须定义 `build_factor(frame)` 返回与输入 index 对齐的 `pd.Series`。
- 不允许读写文件、网络访问、`subprocess`、`eval`、`exec`。
- 不允许 `shift(-n)` / 负向 `pct_change` / 任何 future label 进入 feature。
- 不允许全样本均值 / 标准差作为 feature 标准化（横截面 demean / cross-section
  rank 例外）。
- rolling / expanding 必须按 `asset` 分组。
- `required_columns` 必须是真实 prices/intraday 列名（reviewer 已校验）。

## 不合格输出判定

- 只给 candidate 名字。
- 只给"建议采用某机制"但没有 YAML payload。
- 没有 `contract_version=single_factor_stage1_reconcile_v1`。
- 没有 `mechanisms`。
- 没有 `code_feasibility_review`。
- 没有 `stage2_entry_recommendation`。
- 没有区分 `in_contract_factor_def / needs_extension / future_enhancement`。
- 输出 ranking / best idea 后删除其他机制。
