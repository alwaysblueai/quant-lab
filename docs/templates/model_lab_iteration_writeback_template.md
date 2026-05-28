# Model-Lab Iteration Writeback Template

用于一次模型（model-factor）优化结束后的 Stage 4 写回。目标不是保存完整对话，
而是把「网页版 GPT 的模型改进机制迭代总结」「Claude Desktop 的 Stage2.5 优化总结」
和「Codex GUI 的实验事实总结」合并成一份便于后续大模型检索、理解、迁移和再创造的
知识资产。

与单因子写回的区别：机器事实不是一个因子公式，而是 `model_candidate.json`
（spec 变体）+ 三类 artifact（`model_definition.json` / `feature_manifest.json` /
`metrics.json`）。emergent_moves 是**模型改进动作**（正则、feature 选择、target
horizon、训练窗 / retrain、sample weighting、model selection），不是因子方向。
model 线还要额外捕捉**训练稳定性 / 特征重要性分布 / 过拟合**这类单因子没有的证据。

原则：

- 中文优先；代码符号、指标名、文件路径、专业缩写保留英文。
- 机器事实只来自 `model_candidate.json`、case YAML、run artifacts、manifest 和 metrics。
- 网页版 GPT 的内容只作为机制推理、候选演化、设计取舍和未验证假设。
- Claude Desktop 可作为最终合并编辑器，但不得覆盖 Codex GUI 的 artifact 事实。
- 本模板产物是写回草稿内容，不是直接写入 `50_experiments/` 的指令。
- `50_experiments/` 只作为审批后由 alpha-lab export / apply-writeback 产生的归档目标。
- 不保存原始聊天全文；只保存浓缩后的可复用经验。
- 不引入 Level 3 / execution / portfolio construction 语义；spec_variant only。

```markdown
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

- `improvement_mechanism`: <最终保留下来的模型改进机制：loss / regularization / feature interaction / target construction / sample weighting / training window / model selection 中的哪一类>
- `mechanism_thesis`: <为什么这个改进在该市场/频率上应当有效的经济或统计假设>
- `spec_realization`: <该机制实际用 case_spec 的哪些字段表达：feature_columns / feature_preprocess / model.family / training / target.horizon ...>
- `direction`: <direction=long / short，以及理由>
- `expected_failure_mode`: <机制理论上什么时候容易失效>
- `not_this`: <它不是什么；避免以后误归类或重复发明>

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
- `training`: <window_type / train_window_n_dates / retrain_every_n_dates / min_rows>
- `target`: <kind / horizon>
- `feature_availability`: <mode / column / safety_lag_days，PIT 假设>
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

## 5. 模型专属诊断（单因子没有的部分）

机器事实只认 Codex GUI artifact；填不出可靠值的写 `unknown`/`missing`，不要猜。

- `training_pass_rate`: <跨 retrain 窗口的训练通过率；不稳就是 blocker 候选>
- `model_selection`: <enabled/disabled；启用了哪个 selection metric>
- `overfit_gap`: <train vs validation 表现差；明显发散记下来>
- `feature_importance_concentration`: <重要性是否落在合理少数列，还是被某个 noise feature 主导>
- `feature_importance_top`: <top 列名 + 权重，从 model_definition/feature_manifest>
- `training_stability`: <跨 seed / 跨窗口稳定性观察>
- `resource`: <peak_rss_mb vs max_rss_mb_budget；是否触发/逼近 OOM>

## 6. 被支持、被否定、仍不确定

### 被支持

- 

### 被否定

- 

### 仍不确定

- 

## 7. emergent_moves（主回写字段）

这些是未来大模型最应该复用的「可迁移模型改进动作」。每条必须具体到 spec 层，
不能只是“关注量价关系”这种泛化表达。

- `move`:
  `when_to_try`:
  `spec_field_hint`: <落到哪个 case_spec 字段，如 feature_preprocess.winsorize / training.train_window_n_dates / model.family>
  `why_it_helped_or_failed`:

## 8. operative_claims（弱观察）

这些是经验性观察，只作为未来探索素材，不作为硬 kill 规则。

- 

## 9. negative_constraints（以后不要重复踩）

- 

## 10. future_hypotheses（下一轮可发明方向）

- `hypothesis`:
  `rationale`:
  `required_data_or_schema`: <需要的新 feature 列 / 需要的 spec schema 扩展，如 sample_weight 支持>
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
