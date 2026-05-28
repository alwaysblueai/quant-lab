# Codex GUI Factor Experiment Summary Prompt

把下面 prompt 粘贴给 Codex GUI。它的任务是读取本地仓库和 artifacts，
总结实验事实、迭代表现和可审计证据。

```text
请为这次 alpha-lab 单因子优化做 Stage 4 实验侧总结。

你可以读取本地仓库和 run artifacts。请只把机器可验证事实当作事实：
- `custom_factors/research/<factor>/factor.json`
- `custom_factors/research/<factor>/research_log.md`
- `configs/real_cases/single_factor/<factor>_vN.yaml`
- 每轮 run 目录中的 `metrics.json`、`summary.md`、`integrity_report.md`
- `run_manifest.json`、`factor_definition.json`
- backend contract sidecar，如 `comparison_summary.json`、`backend_run_receipt.json`

要求：
- 不要把 Codex GUI 原始聊天全文写入总结。
- 不要改 factor、case YAML、promoted 目录或前端注册；本任务只总结。
- 如果某轮 artifact 缺失，就明确列为 missing，不要猜指标。
- 检查最终 run 是否有 `custom_factor_source.code_sha256`、
  `custom_factor_source.factor_json_sha256`、source path。
- 按 Tier L 视角判断是否达到 library admission；没有足够 artifact 就写 blocked/unknown。
- 中文优先；指标名、路径、hash、代码符号保留英文。
- 不引入 Level 3 / execution / fill simulation 语义。

请先读取必要文件，再严格按下面 schema 输出 Markdown：

---
type: codex_gui_experiment_summary
schema_version: codex_gui_experiment_summary_v1
factor_name:
source: codex_gui_local_artifacts
machine_fact_policy: "artifact_facts_only"
---

# Codex GUI 实验总结 - <factor_name>

## 1. 输入和 artifact 范围

- `factor_json`:
- `research_log`:
- `case_yamls`:
- `run_roots_inspected`:
- `final_run_root`:
- `missing_artifacts`:

## 2. 工程契约审计

- `validate_draft_factor`: passed | failed | unknown
- `custom_factor_source.path`:
- `code_sha256`:
- `factor_json_sha256`:
- `provenance.idea_id`:
- `run_manifest_audit`: ok | failed | unknown
- `factor_definition_audit`: ok | failed | unknown
- `pit_or_temporal_alignment`: ok | failed | unknown

## 3. 每轮迭代表现

| Version | Case YAML | Run Root | 关键改动 | RankIC mean | RankIC IR | Coverage | PIT | Verdict |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| v1 |  |  |  |  |  |  |  |  |
| v2 |  |  |  |  |  |  |  |  |
| final |  |  |  |  |  |  |  |  |

## 4. 最终 scorecard

| Metric | Value | Gate / Context | Pass | Source |
| --- | ---: | --- | --- | --- |
| RankIC mean |  | Tier L >= 0.01 |  |  |
| RankIC IR |  | Tier L >= 0.15 |  |  |
| PIT scan |  | hard gate: 0 violations |  |  |
| Regime same-sign |  | Tier L >= 2/4 |  |  |
| Cost IR 5bps |  | Tier L >= 0.05 |  |  |
| Sample length |  | Tier L >= 2 years |  |  |
| Correlation vs promoted suite |  | Tier L <= 0.70 |  |  |

## 5. 数据支持的结论

### 支持

- 

### 阻塞

- 

### 不确定

- 

## 6. 从实验中浮现的可复用经验

- `emergent_moves`:
  - 
- `operative_claims`:
  - 
- `negative_constraints`:
  - 

## 7. 建议写回合并字段

- `outcome`: promoted | keep_iterating | parked | killed
- `one_sentence_verdict`:
- `final_mechanism_thesis_if_supported`:
- `next_action`:
- `recommended_writeback_stage_target`: `55_projects/<project_slug>/50_writeback_drafts/`
- `approved_archive_target`: `50_experiments/` via alpha-lab export/apply-writeback
- `manual_vault_write_policy`: do_not_create_or_touch_files_in_50_experiments_directly
- `source_paths_to_cite`:
  - 
```
