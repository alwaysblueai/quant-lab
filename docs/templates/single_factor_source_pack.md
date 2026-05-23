# Single-Factor Web GPT Source Pack

本文件用于放入网页版 GPT 项目的"来源"，作为 single-factor 对话的总入口。
它告诉网页版 GPT：不同输入材料应触发哪个固定合同，避免输出只命名、只摘要、
只建议继续讨论的非机器化结果。

## 合同路由

### 输入是两个 Stage 1 引擎输出（含 Part A + Part B）

如果用户提供 Claude Code 与 Codex GUI 对同一个 factor idea 的 Stage 1 输出
（每份都含 Part A 机制候选 + Part B 可执行性评审），必须使用：

- `docs/templates/single_factor_stage1_reconcile_contract.md`

输出唯一的 `single_factor_stage1_reconcile_payload`。

不允许只回答：

- candidate 名字
- 中文名
- 文件名
- "我建议使用 X"
- "下一步可以做 Y"

### 输入是 Stage 1 reconcile payload + 可用字段 / 已有 factor 索引

如果用户要求"生成可执行方案""进入 Stage 2""给 Codex GUI 跑实验的草案"，
必须使用：

- `docs/templates/single_factor_stage2_candidate_contract.md`

输出唯一的 `factor_json_payload`（含完整 `code`、`required_columns`、
`pit_assumption`、`provenance`）。

### 输入是 Stage 3 后端实验摘要

如果用户在网页 GPT 项目里直接粘了 Stage 3 跑完的摘要段落（包括 metrics + 失败点
+ 下轮建议），网页 GPT 应该输出 **v<n+1>** 的 `factor_json_payload`：

- 复用上一轮 `factor_json_payload` 的 `provenance.idea_id`（保持不变）
- 更新 `provenance.stage2_payload_sha256`
- 仅在用户允许范围内调整：机制 proxy / 字段 / 参数 / 方向 / 中性化层

不要让网页 GPT 在 Stage 3 阶段自己跑实验或解读数据。

## 完整性门禁

每次输出前必须自检：

- 当前处于 Stage 1、Stage 2 还是 Stage 3 迭代？
- 是否输出了该阶段要求的 `contract_version`？
- 是否输出了机器可提取的 YAML / JSON code block？
- 是否避免了只命名、只摘要、只主观建议？
- 是否把 `needs_extension` 机制排除在 v1 `factor_json_payload` 外？
- 是否把 `required_columns` 限制为 `factor_recipe.py` 已注册列名？
- 是否没有引入 `shift(-n)` / 负向 `pct_change` / 全样本统计标准化？
- `code` 是否定义 `build_factor(frame)` 或兼容旧 `builder(prices, ...)`？
- 是否没有 Level 3 / execution replay / fill simulation / portfolio construction 语义？

如果不能满足，必须输出：

```yaml
contract_version: "single_factor_output_failed_quality_gate_v1"
stage: "<stage>"
ok: false
failure_reason: ""
missing_inputs: []
required_next_input: []
```

不能用普通自然语言糊过去。

## 来源文件清单（**只放 3 个**）

网页版 GPT 项目 "sources" 严格只放：

- `single_factor_source_pack.md`（本文，总入口）
- `single_factor_stage1_reconcile_contract.md`
- `single_factor_stage2_candidate_contract.md`

**不要**把 Stage 3 envelope / backend_draft_factor_workflow / stage3_backend_draft_factor_prompt
放进网页 GPT 项目 —— 那些是 Codex GUI 的开场提示，放进 GPT 会让它越界
"自己跑实验"或写代码。

详见 `docs/end_to_end_workflow.md` Stage 2。
