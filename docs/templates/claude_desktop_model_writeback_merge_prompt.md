# Claude Desktop Model Writeback Merge Prompt

把下面 prompt 粘贴到 Claude Desktop。它是自包含的最终合并 prompt：Claude
Desktop 不需要再打开 `model_lab_iteration_writeback_template.md`，也不需要用户手工
套模板。你只要在 prompt 末尾粘贴三类输入材料，Claude 就应在对话中直接输出完整的
`model_lab_iteration_writeback_v1` Markdown 写回稿文本，不应自行创建或修改任何文件。

```text
你现在要担任 alpha-lab model-factor（spec 变体）优化的最终写回合并编辑器。

你会收到三类材料：
1. Web GPT 模型机制迭代总结：负责初始假设、候选演化、设计取舍、可复用思路。
2. Claude Desktop Stage2.5 迭代内容：负责 vault 加固、推理补强、候选收敛。
3. Codex GUI 实验总结：负责本地 artifact、每轮实验表现、metrics、training_pass_rate、
   feature importance、PIT、三个 sha256 hash、audit。

你的目标：
- 按 `model_lab_iteration_writeback_v1` 写成最终知识库回写稿。
- 直接在对话中输出完整 Markdown 写回稿；不要只输出提纲、建议或待填模板。
- 不要要求用户再手工套用 `model_lab_iteration_writeback_template.md`。
- 不要自行访问、创建、移动、修改或删除 vault / repo 文件；尤其不要直接写入
  `50_experiments/`。
- 输出要适合未来大模型检索、理解、迁移和再创造。
- 不保存原始对话全文，只保留浓缩后的研究经验。
- 把 `emergent_moves` 写成最重要的主回写字段，必须具体、可迁移、落到 case_spec
  字段层（feature_preprocess / training / model.family / target.horizon ...）。
- 把 `operative_claims` 写成弱观察，不能写成硬 kill 规则。
- 把 `negative_constraints` 写清楚，避免未来重复踩坑（含被误当成 v1 spec_variant 的
  future-enhancement 想法）。
- 完整保留「模型专属诊断」段：training_pass_rate / model_selection / overfit_gap /
  feature importance 分布 / 训练稳定性 / resource；这是 model 线区别于单因子的关键证据。

事实边界：
- 机器事实只认 Codex GUI 实验总结及其引用的本地 artifacts。
- 如果 Web GPT 或 Claude Desktop 的机制叙述与 Codex GUI artifact 事实冲突，
  以 Codex GUI 为准。
- 如果 Codex GUI 标注某个指标、hash、artifact 为 missing/unknown，你也必须保留
  missing/unknown，不要猜。
- 你可以改写机制解释，让它更贴合最终实验表现，但不能编造新实验。
- spec_variant only：不引入 Level 3 / execution / portfolio construction / 自定义
  feature/estimator code 语义。
- 中文优先；指标名、代码符号、路径、hash、专业缩写保留英文。
- 本输出是写回草稿内容；正式进入 `50_experiments/` 必须经过用户审批和
  alpha-lab export / apply-writeback 流程。

输入材料会放在本 prompt 末尾，格式如下。如果某类材料缺失，就在最终稿对应字段
标注 `missing` 或 `not_provided`，不要猜：

<WEB_GPT_MECHANISM_SUMMARY>
粘贴 Web GPT 模型机制总结
</WEB_GPT_MECHANISM_SUMMARY>

<CLAUDE_DESKTOP_STAGE2_5_NOTES>
粘贴 Claude Desktop 中间优化记录；如果当前对话本身已经包含，则自行总结。
</CLAUDE_DESKTOP_STAGE2_5_NOTES>

<CODEX_GUI_EXPERIMENT_SUMMARY>
粘贴 Codex GUI 模型实验事实总结
</CODEX_GUI_EXPERIMENT_SUMMARY>

请严格输出下面 Markdown schema 的一份完整成稿。不要输出 schema 外说明，不要输出
"以下是模板"，不要留下空白占位符；确实未知的字段写 `unknown` / `missing` 并保留
原因。

---
type: model_iteration_writeback
schema_version: model_lab_iteration_writeback_v1
project_slug:
idea_id:
candidate_name:
lab: model_factor
market:
frequency: daily
outcome: promoted | keep_iterating | parked | killed
review_status: pending
source_summaries:
  web_gpt_mechanism_summary:
  claude_desktop_stage2_5_summary:
  codex_gui_experiment_summary:
machine_artifacts:
  model_candidate_json:
  case_yaml:
  final_run_root:
  run_manifest:
  model_definition:
  feature_manifest:
  metrics_json:
  summary_md:
  integrity_report:
  comparison_summary:
  resource_usage:
writeback_stage_target_hint: 55_projects/<project_slug>/50_writeback_drafts/
approved_archive_target_hint: 50_experiments/ via alpha-lab export/apply-writeback
manual_vault_write_policy: "do_not_create_or_touch_files_in_50_experiments_directly"
---

# <模型候选中文标题>

## 0. 来源合并说明

- `web_gpt_role`: 模型改进机制推理、候选演化、设计取舍。
- `claude_desktop_role`: Stage2.5 vault 加固、最终合并编辑、经验抽象。
- `codex_gui_role`: 本地 artifact 事实、每轮实验表现、训练稳定性、审计证据。
- `conflict_policy`: 机器事实冲突时以 Codex GUI artifact 总结为准；机制解释冲突时保留更贴合最终实验表现的一版，并在不确定处标注。

## 一句话结论

- `verdict`:
- `why_it_matters`:
- `should_reuse`: yes | conditional | no

## 1. 机制压缩

- `improvement_mechanism`:
- `mechanism_thesis`:
- `spec_realization`:
- `direction`:
- `expected_failure_mode`:
- `not_this`:

## 2. 迭代轨迹

| Version | spec 改动（feature/preprocess/model/training/target） | 改动理由 | 实验反馈 | 决策 |
| --- | --- | --- | --- | --- |
| v1 |  |  |  |  |
| v2 |  |  |  |  |
| final |  |  |  |  |

## 3. 最终模型契约

- `model_candidate_json`:
- `feature_columns`:
- `feature_preprocess`:
- `model.family`:
- `model_selection`:
- `training`:
- `target`:
- `feature_availability`:
- `case_yaml`:
- `evaluation_profile`:
- `sample_window`:

## 4. 证据摘要

| Metric | Value | Gate / Context | Pass | Source |
| --- | ---: | --- | --- | --- |
| RankIC mean |  | Tier L >= 0.01 |  | `metrics.json` |
| RankIC IR |  | Tier L >= 0.15 |  | `metrics.json` |
| PIT scan |  | hard gate: 0 violations |  | `integrity_report.md` |
| Regime same-sign |  | Tier L >= 2/4 |  | regime artifact |
| Cost IR 5bps |  | Tier L >= 0.05 |  | cost artifact |
| Coverage mean |  | case/context dependent |  | coverage artifact |
| Correlation vs promoted suite |  | Tier L <= 0.70 |  | cross-corr artifact |

## 5. 模型专属诊断

- `training_pass_rate`:
- `model_selection`:
- `overfit_gap`:
- `feature_importance_concentration`:
- `feature_importance_top`:
- `training_stability`:
- `resource`:

## 6. 被支持、被否定、仍不确定

### 被支持

- 

### 被否定

- 

### 仍不确定

- 

## 7. emergent_moves（主回写字段）

- `move`:
  `when_to_try`:
  `spec_field_hint`:
  `why_it_helped_or_failed`:

## 8. operative_claims（弱观察）

- 

## 9. negative_constraints（以后不要重复踩）

- 

## 10. future_hypotheses（下一轮可发明方向）

- `hypothesis`:
  `rationale`:
  `required_data_or_schema`:
  `risk`:

## 11. 检索标签和别名

- `mechanism_tags`:
- `feature_tags`:
- `model_family_tags`:
- `risk_tags`:
- `aliases_to_avoid`:
- `related_models_or_cards`:

## 12. 审计引用

- `candidate_json_sha256`:
- `case_spec_sha256`:
- `feature_contract_sha256`:
- `provenance.idea_id`:
- `run_manifest_path`:
- `final_artifact_path`:
```
