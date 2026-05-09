# Codex GUI Model-Lab Stage 3 后端草稿模型执行合同

本文件约束 Codex GUI 在 alpha-lab 本地仓库中执行网页版 GPT 输出的
Stage2 `model_candidate_payload`。当前阶段只做 v1 **spec 变体型草稿模型** 的
后端标准验证，不做前端注册，不做晋升，不引入自定义 feature builder 或
estimator code。

## 机器事实来源

只把以下内容作为机器事实：

- Stage2 输出中的 `model_candidate_payload`（`contract_version=stage2_model_candidate_v1`）
- Stage2 输出中的 `case_spec_payload`
- 用户明确指定的本地路径或命令

如果 `human_summary`、`stage3_notes`、`risk_controls` 自由文本与
`case_spec_payload` 冲突：

1. 以 `case_spec_payload` 为准。
2. 在 `model_candidates/research/<candidate_name>/research_log.md` 记录冲突。
3. 不要根据自由文本自行改写 case spec。

## 允许写入

只能写入或更新：

- `model_candidates/research/<candidate_name>/model_candidate.json`
- `model_candidates/research/<candidate_name>/research_log.md`
- `configs/real_cases/model_factor/<candidate_name>_vN.yaml`

禁止写入：

- 临时 Python 脚本
- notebook
- 散落的 `.py` 文件
- `model_candidates/promoted`
- `src/alpha_lab/model_factor/`、`src/alpha_lab/factors/`、其他 core 模块
- 前端正式注册文件
- execution / replay / fill simulation / portfolio construction 相关目录

v1 不接受任何自定义 feature builder code 或自定义 estimator code；只能通过
`case_spec_payload` 中现有的 `model.family` / `feature_columns` /
`feature_preprocess` / `training` 字段进行变体。

## model_candidate.json 合同

`model_candidate_payload` 必须可以直接写入：

```text
model_candidates/research/<candidate_name>/model_candidate.json
```

最小结构（严格遵循）：

```yaml
contract_version: "stage2_model_candidate_v1"
candidate_name: ""
implementation_status: "draft_for_stage3"
implementation_type: "spec_variant"
source_mechanisms: []
base_case_spec_path: ""
expected_horizon: "t_plus_1_or_later"
data_contract:
  prices_required_columns: ["date", "asset", "close"]
  feature_required_columns: []
  feature_optional_columns: []
  feature_availability:
    mode: "required_timestamp"      # 或 "safety_lag"
    column: "known_at"               # required_timestamp 模式必填
    safety_lag_days: null            # safety_lag 模式必填且 > 0
risk_controls:
  feature_availability_pit: ""
  label_leakage: ""
  overfit_complexity: ""
  turnover_cost: ""
  feature_instability: ""
  split_regime_fragility: ""
run_controls:
  evaluation_profile: "exploratory_screening"
  screening_retrain_every_n_dates: 40
  vault_export_mode: "skip"
case_spec_payload:
  name: "<case_name>"
  factor_name: "<candidate_name>"
  features_path: "<absolute or relative path>"
  prices_path: "<absolute or relative path>"
  feature_columns: ["..."]
  rebalance_frequency: "W"
  n_quantiles: 5
  direction: "long"
  universe: { ... }
  target: { kind: "forward_return", horizon: 5 }
  feature_availability: { mode: "required_timestamp", column: "known_at" }
  feature_preprocess: { ... }
  feature_importance: { ... }
  model: { family: "ridge", params: { alpha: 1.0 } }
  model_selection: { enabled: false, ... }
  training: { window_type: "rolling", train_window_n_dates: 60, ... }
  neutralization: { ... }
  transaction_cost: { one_way_rate: 0.001 }
  output: { root_dir: "..." }
stage3_validation_focus: []
```

`case_spec_payload` 必须能直接被
`alpha_lab.real_cases.model_factor.spec.model_factor_case_spec_from_mapping`
解析；含：`features_path`、`prices_path`、`feature_columns`、
`feature_availability`、`feature_preprocess`、`model`、`model_selection`、
`training`、`target`、`transaction_cost`、`output`。

约束：

- `model.family` 必须是当前支持集合内的字符串：`linear / ridge / lasso / elastic_net / gbdt / xgboost / lightgbm / mlp`。
- `feature_columns` 不能包含保留列 `date / asset / factor / value`，且必须在 features 文件表头中存在。
- `feature_availability.mode='required_timestamp'` 时 `safety_lag_days` 必须省略或为 0；`mode='safety_lag'` 时 `column` 必须省略且 `safety_lag_days >= 1`。
- 不得引入 portfolio construction / Level 3 / execution semantics。

## 标准执行步骤

1. 从 Stage2 输出中提取唯一的 `model_candidate_payload`。
2. 写入 `model_candidates/research/<candidate_name>/model_candidate.json`。
3. 创建或更新 `configs/real_cases/model_factor/<candidate_name>_vN.yaml`。
   YAML 内容必须与 `case_spec_payload` 完全一致；不要根据 `human_summary` 调参。
4. 先运行 validator：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab validate-draft-model model_candidates/research/<candidate_name>/model_candidate.json
```

5. validator 通过后，按照 `run_controls` 设定的 profile 跑标准 pipeline。

快速初筛（默认）：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab real-case model-factor run \
  configs/real_cases/model_factor/<candidate_name>_v1.yaml \
  --evaluation-profile exploratory_screening \
  --screening-retrain-every-n-dates 40 \
  --render-report \
  --vault-export-mode skip \
  --draft-model-candidate model_candidates/research/<candidate_name>/model_candidate.json
```

更完整验证（在初筛通过后再跑）：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab real-case model-factor run \
  configs/real_cases/model_factor/<candidate_name>_v1.yaml \
  --evaluation-profile default_research \
  --render-report \
  --vault-export-mode skip \
  --draft-model-candidate model_candidates/research/<candidate_name>/model_candidate.json
```

## Preflight 必查

- `model_candidate.json` 是合法 JSON。
- `contract_version=stage2_model_candidate_v1`、`implementation_status=draft_for_stage3`、`implementation_type=spec_variant`。
- `candidate_name` 是英文 snake_case，长度 3-64。
- 文件位于 `model_candidates/research/<candidate_name>/model_candidate.json`。
- `case_spec_payload` 通过 `model_factor_case_spec_from_mapping` 解析无错。
- `feature_columns` 全部出现在 features 文件表头中。
- `feature_availability` 满足 PIT 合同；疑似基本面特征不允许 `mode='required_timestamp'` 且 `column=null`。
- `model.family` 在支持集合内。
- 没有任何 Level 3 / execution_replay / fill_simulation / portfolio_construction 关键词。

## Artifact 审计

运行后必须读取：

- `run_manifest.json`
- `model_definition.json`
- `feature_manifest.json`
- `metrics.json`
- `summary.md`
- `integrity_report.md`
- 已渲染的 `case_report.md`

必须确认 `run_manifest.json`、`model_definition.json`、`feature_manifest.json`
均包含：

```json
{
  "draft_model_source": {
    "name": "<candidate_name>",
    "scope": "research",
    "path": "...",
    "candidate_json_sha256": "...",
    "case_spec_sha256": "...",
    "feature_contract_sha256": "...",
    "contract_version": "stage2_model_candidate_v1",
    "implementation_status": "draft_for_stage3",
    "implementation_type": "spec_variant",
    "factor_name": "...",
    "feature_columns": [...],
    "feature_availability": {...},
    "model_family": "..."
  }
}
```

如果缺少 `candidate_json_sha256`、`case_spec_sha256`、`feature_contract_sha256`
或 source path，本次 Stage3 视为失败。

## 输出要求

最终用中文输出：

1. 写入了哪些文件（candidate JSON、research_log.md、case YAML）。
2. 实际运行的 validator 命令和实验命令。
3. 实验成功或失败，artifact 输出路径。
4. 是否确认 `candidate_json_sha256`、`case_spec_sha256`、`feature_contract_sha256`、
   source path、feature columns、feature availability、model family、PIT 假设。
5. 初筛结果摘要：coverage、IC / rank IC、ic_decay、turnover、训练通过率、
   model_selection 状态、portfolio validation status。
6. 主要失败点或脆弱点（机制不兼容、特征覆盖率低、训练样本不足、PIT 风险等）。
7. 下一轮 1-3 个具体修改方向（仅限 `case_spec_payload` 字段调整：feature_columns、
   feature_preprocess、model.family、training、target.horizon 等）。

禁止输出：

- 模型已经有效
- 可以实盘
- 买入 / 卖出建议
- portfolio construction 结论
- Level 3 execution / fill simulation / replay 建议
