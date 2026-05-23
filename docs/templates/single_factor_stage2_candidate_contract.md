# Single-Factor Stage2 Candidate Contract

本文件用于放入网页版 GPT 项目的"来源"。当用户提供
`single_factor_stage1_reconcile_payload`、可用字段列表、已有 factor 索引、
已有 single-factor run 摘要后，网页版 GPT 必须把 Stage1 机制判断转化为唯一的
`factor_json_payload`。该 payload 是 Codex GUI / Claude Code Stage 3 的机器事实。

## 任务边界

当前阶段是 Stage2 mechanism-to-factor candidate：

- 选择一个 v1 可执行的 mechanism。
- 输出完整 `factor_json_payload`（可直接写入 `custom_factors/research/<f>/factor.json`）。
- 明确 `provenance.idea_id` 和 `provenance.stage2_payload_sha256`。
- 明确 deferred mechanisms 和原因。

当前不要：

- 不要跑实验。
- 不要声称因子有效。
- 不要输出 portfolio construction。
- 不要写 promoted candidate。
- 不要把 future enhancement 写进 `factor_json_payload`。

只输出名字、自然语言建议、下一步讨论方向，均为不合格。

## 输入材料

用户会提供：

- Stage1 reconcile payload（含 `code_feasibility_review`）。
- 可用 prices/intraday/daily 列名（必填）。
- 已有 `custom_factors/{promoted,research}/<name>/factor.json` 列表（防重做）。
- validator 硬约束清单（reviewer 在 Stage1 已看到，Stage2 直接复用）。

如果缺少真实字段列表，不能编造 `required_columns`；必须在 `quality_gate.blockers` 标注。

## 输出格式

必须只输出一个 YAML code block。不要在 code block 外输出正文。

顶层只允许以下键：

```yaml
contract_version: "single_factor_stage2_candidate_output_v1"
stage: "stage2_candidate"
provenance: {}
human_summary: {}
factor_json_payload: {}
deferred_mechanisms: []
stage3_execution_notes: []
quality_gate: {}
```

## YAML schema

```yaml
contract_version: "single_factor_stage2_candidate_output_v1"
stage: "stage2_candidate"
provenance:
  idea_id: ""
  stage2_payload_sha256: ""           # sha256 of canonical-JSON(factor_json_payload), filled by GPT
  audience_chain: ["claude", "codex", "web_gpt_stage2"]
human_summary:
  factor_name: ""
  implemented_mechanism_ids: []
  implementation_summary: ""
  not_implemented_summary: ""
  why_v1_is_in_contract: ""
factor_json_payload:
  name: ""                            # snake_case, must not collide with BUILTIN_FACTOR_NAMES
  description: ""
  required_columns: []
  optional_columns: []
  frequency: "daily"
  unavailable_data_policy: "return_nan"
  pit_assumption: ""
  code: |
    def build_factor(frame):
        ...
        return value
  provenance:
    idea_id: ""
    stage2_payload_sha256: ""
    audience_chain: ["claude", "codex", "web_gpt_stage2"]
deferred_mechanisms:
  - mechanism_id: ""
    status: "needs_extension | future_enhancement | rejected_for_v1"
    reason: ""
stage3_execution_notes:
  - ""
quality_gate:
  name_only_response: false
  contains_factor_json_payload: true
  factor_name_is_snake_case: true
  required_columns_are_real: true
  no_future_leakage: true
  no_full_sample_stats: true
  no_portfolio_or_execution_semantics: true
  blockers: []
```

## v1 允许的实现

允许：

- 使用现有 `prices` 列：`open / high / low / close / volume / amount / vwap` 等。
- 使用现有 intraday-derived 日频列（`factor_recipe.py` 已注册的 ~60 列：rv / jump /
  skew / share / vwap_dev / limit / minutes_at_extremes 等）。
- rolling / expanding 必须按 `asset` 分组。
- 横截面 demean / rank（同一 `date` 内）。
- 已知方向先验时显式标注（不可凭因子名隐含方向）。

禁止：

- `shift(-n)`、负向 `pct_change`、`future_return` / `forward_return` /
  `next_return` 进入 feature。
- 全样本均值 / 标准差作为 feature 标准化。
- 读写文件、网络访问、`subprocess`、`eval`、`exec`、`open()`。
- `import os / pathlib / shutil / requests / urllib / pickle`。

## PIT / required_columns 规则

- `required_columns` 必须是 `factor_recipe.py` 已注册的列名（包括 prices 标准列
  + intraday-derived 列）。
- 如果机制需要的列不在已注册列表中，先放进 `quality_gate.blockers`，让 Stage 3
  补 ETL；不要 mock 列名。
- `pit_assumption` 必须显式说明：哪些 rolling 用的是 t 及之前的数据，最后一个
  bar 是否包含 t 当天的收盘信息。

## 不合格输出判定

- 只给 candidate 名字。
- 只给自然语言建议。
- 没有 `contract_version=single_factor_stage2_candidate_output_v1`。
- 没有 `factor_json_payload`。
- `factor_json_payload.code` 缺 `build_factor(frame)` 函数。
- `required_columns` 包含表达式或不存在字段。
- 把 `needs_extension` 机制写进 v1 `factor_json_payload`。
- 没有 `quality_gate`。
- `provenance.idea_id` 缺失或与 `manifest.json::idea_id` 不一致。
