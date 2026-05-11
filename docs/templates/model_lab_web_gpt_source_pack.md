# Model-Lab Web GPT Source Pack

本文件用于放入网页版 GPT 项目的“来源”，作为 model-lab 对话的总入口。
它告诉网页版 GPT：不同输入材料应触发哪个固定合同，避免输出只命名、只摘要、
只建议继续讨论的非机器化结果。

## 合同路由

### 输入是两个 Stage1 agent 输出

如果用户提供 Claude Code 与 Codex GUI 对同一个 model idea 的 Stage1 输出，
必须使用：

- `docs/templates/model_lab_stage1_reconcile_contract.md`

输出唯一的 `model_stage1_reconcile_payload`。

不允许只回答：

- candidate 名字
- 中文名
- 文件名
- “我建议使用 X”
- “下一步可以做 Y”

### 输入是 Stage1 reconcile payload + base spec / feature inventory

如果用户要求“生成可执行方案”“进入 Stage2”“给 Codex GUI 跑实验的草案”，
必须使用：

- `docs/templates/model_lab_stage2_candidate_contract.md`

输出唯一的 `model_candidate_payload`，并包含完整 `case_spec_payload`。

### 输入是 Stage2 model_candidate_payload

如果用户要在本地运行、验证、保存 candidate、生成 case YAML 或跑 screening，
必须使用：

- `docs/templates/model_lab_stage3_backend_draft_prompt.md`
- `docs/templates/codex_gui_model_stage3_execution_envelope.md`
- `docs/backend_draft_model_workflow.md`

该阶段由 Codex GUI 或 Web Model-Lab 执行；网页版 GPT 只负责输出结构化 payload。

## 完整性门禁

每次输出前必须自检：

- 当前处于 Stage1、Stage2 还是 Stage3？
- 是否输出了该阶段要求的 `contract_version`？
- 是否输出了机器可提取的 YAML code block？
- 是否避免了只命名、只摘要、只主观建议？
- 是否把 needs-extension 机制排除在 v1 `case_spec_payload` 外？
- 是否把 feature columns 限制为真实已存在字段？
- 是否没有引入自定义 feature code、estimator code、sample_weight、custom target？
- 是否没有 Level 3、execution replay、fill simulation、portfolio construction 语义？

如果不能满足，必须输出：

```yaml
contract_version: "model_lab_output_failed_quality_gate_v1"
stage: "<stage>"
ok: false
failure_reason: ""
missing_inputs: []
required_next_input: []
```

不能用普通自然语言糊过去。

## 对 Turnover-Conditioned PV 的特别约束

当 idea 是“成交活跃但波动不过热时，价量信号更可信”时：

- `turnover_conditioned_pv_synthesis_v1` 可以作为 slug hint，但不能作为最终输出本身。
- Stage1 必须保留机制候选并标注可执行性。
- Stage2 v1 首选 spec variant，不走自定义交互列、sample_weight、cost-shaped target
  或双窗口训练。
- “条件信任 / turnover-conditioned”可以作为机制解释框架，但进入 Stage3 的
  机器事实必须落在 `case_spec_payload` 已支持字段上。

## 来源文件清单（**只放 3 个**）

网页版 GPT 项目"sources" 严格只放：

- `model_lab_web_gpt_source_pack.md`（本文，总入口）
- `model_lab_stage1_reconcile_contract.md`
- `model_lab_stage2_candidate_contract.md`

**不要**把 Stage 3 envelope / backend_draft_model_workflow / stage3_backend_draft_prompt
放进网页 GPT 项目 —— 那些是 Codex GUI 的开场提示，放进 GPT 会让它越界
"自己跑实验"或写代码。

详见 `docs/end_to_end_workflow.md` Stage 2。
