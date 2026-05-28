# Single-Factor Iteration Writeback Template

用于一次单因子优化结束后的 Stage 4 写回。目标不是保存完整对话，
而是把「网页版 GPT 的机制迭代总结」「Claude Desktop 的 Stage2.5 优化总结」
和「Codex GUI 的实验事实总结」合并成一份便于后续大模型检索、理解、
迁移和再创造的知识资产。

原则：

- 中文优先；代码符号、指标名、文件路径、专业缩写保留英文。
- 机器事实只来自 `factor.json`、case YAML、run artifacts、manifest 和 metrics。
- 网页版 GPT 的内容只作为机制推理、候选演化、设计取舍和未验证假设。
- Claude Desktop 可作为最终合并编辑器，但不得覆盖 Codex GUI 的 artifact 事实。
- 本模板产物是写回草稿内容，不是直接写入 `50_experiments/` 的指令。
- `50_experiments/` 只作为审批后由 alpha-lab export / apply-writeback 产生的归档目标。
- 不保存原始聊天全文；只保存浓缩后的可复用经验。
- 不引入 Level 3 / execution 语义。

```markdown
---
type: factor_iteration_writeback
schema_version: single_factor_iteration_writeback_v1
project_slug:
idea_id:
factor_name:
lab: single_factor
market:
frequency: daily
outcome: promoted | keep_iterating | parked | killed
review_status: pending
source_summaries:
  web_gpt_mechanism_summary:
  claude_desktop_stage2_5_summary:
  codex_gui_experiment_summary:
machine_artifacts:
  factor_json:
  case_yaml:
  final_run_root:
  run_manifest:
  factor_definition:
  metrics_json:
  summary_md:
  integrity_report:
writeback_stage_target_hint: 55_projects/<project_slug>/50_writeback_drafts/
approved_archive_target_hint: 50_experiments/ via alpha-lab export/apply-writeback
manual_vault_write_policy: "do_not_create_or_touch_files_in_50_experiments_directly"
---

# <因子中文标题>

## 0. 来源合并说明

- `web_gpt_role`: 机制推理、候选演化、设计取舍。
- `claude_desktop_role`: Stage2.5 优化、最终合并编辑、经验抽象。
- `codex_gui_role`: 本地 artifact 事实、每轮实验表现、审计证据。
- `conflict_policy`: 机器事实冲突时以 Codex GUI artifact 总结为准；机制解释冲突时保留更贴合最终实验表现的一版，并在不确定处标注。

## 一句话结论

- `verdict`:
- `why_it_matters`:
- `should_reuse`: yes | conditional | no

## 1. 机制压缩

- `mechanism_thesis`: <最终保留下来的经济/微结构假设>
- `signal_proxy`: <因子实际用什么可观测 proxy 表达该机制>
- `direction`: <high=long / high=short，以及理由>
- `expected_failure_mode`: <机制理论上什么时候容易失效>
- `not_this`: <它不是什么；避免以后误归类或重复发明>

## 2. 迭代轨迹

| Version | 机制/实现改动 | 改动理由 | 实验反馈 | 决策 |
| --- | --- | --- | --- | --- |
| v1 |  |  |  |  |
| v2 |  |  |  |  |
| final |  |  |  |  |

## 3. 最终因子契约

- `factor_json`:
- `required_columns`:
- `optional_columns`:
- `pit_assumption`:
- `unavailable_data_policy`:
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

## 5. 被支持、被否定、仍不确定

### 被支持

- 

### 被否定

- 

### 仍不确定

- 

## 6. emergent_moves（主回写字段）

这些是未来大模型最应该复用的「可迁移动作」。每条必须具体到可执行层，
不能只是“关注量价关系”这种泛化表达。

- `move`:
  `when_to_try`:
  `implementation_hint`:
  `why_it_helped_or_failed`:

## 7. operative_claims（弱观察）

这些是经验性观察，只作为未来探索素材，不作为硬 kill 规则。

- 

## 8. negative_constraints（以后不要重复踩）

- 

## 9. future_hypotheses（下一轮可发明方向）

- `hypothesis`:
  `rationale`:
  `required_data`:
  `risk`:

## 10. 检索标签和别名

- `mechanism_tags`:
- `data_tags`:
- `risk_tags`:
- `aliases_to_avoid`:
- `related_factors_or_cards`:

## 11. 审计引用

- `code_sha256`:
- `factor_json_sha256`:
- `provenance.idea_id`:
- `run_manifest_path`:
- `final_artifact_path`:
```
