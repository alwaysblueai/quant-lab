# Model-Lab Stage2 Candidate Contract

本文件用于放入网页版 GPT 项目的“来源”。当用户提供
`model_stage1_reconcile_payload`、base spec、feature inventory、已有 run 摘要后，
网页版 GPT 必须把 Stage1 机制判断转化为唯一的
`model_candidate_payload`。该 payload 是 Codex GUI / Web Model-Lab Stage3 的机器事实。

## 任务边界

当前阶段是 Stage2 signal-to-spec candidate：

- 选择一个 v1 可执行的 spec 变体。
- 输出完整 `model_candidate_payload`。
- `case_spec_payload` 必须是完整 ModelFactorCaseSpec，不是 patch。
- 明确 rejected/deferred mechanisms 和原因。

当前不要：

- 不要跑实验。
- 不要声称模型有效。
- 不要输出 portfolio construction。
- 不要写 promoted candidate。
- 不要引入自定义 feature code 或 estimator code。
- 不要把 future enhancement 写进 `case_spec_payload`。

只输出名字、自然语言建议、下一步讨论方向，均为不合格。

## 输入材料

用户会提供：

- Stage1 reconcile payload。
- base case spec 原文或路径。
- feature inventory：features 文件路径、真实字段名、是否存在 `known_at` /
  `available_at`。
- prices 文件路径。
- 已有 model-factor / single-factor run 摘要。
- 本轮用户选择或约束。

如果缺少真实字段列表，不能编造 `feature_columns`；必须在
`stage3_validation_focus` 和 `quality_gate.blockers` 标注。

## 输出格式

必须只输出一个 YAML code block。不要在 code block 外输出正文。

顶层只允许以下键：

```yaml
contract_version: "model_stage2_candidate_output_v1"
stage: "stage2_candidate"
human_summary: {}
model_candidate_payload: {}
deferred_mechanisms: []
stage3_execution_notes: []
quality_gate: {}
```

## YAML schema

```yaml
contract_version: "model_stage2_candidate_output_v1"
stage: "stage2_candidate"
human_summary:
  candidate_name: ""
  implemented_mechanism_ids: []
  implementation_summary: ""
  not_implemented_summary: ""
  why_v1_is_spec_variant: ""
model_candidate_payload:
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
      mode: "required_timestamp | safety_lag"
      column: "known_at"
      safety_lag_days: null
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
  case_spec_payload: {}
  stage3_validation_focus: []
deferred_mechanisms:
  - mechanism_id: ""
    status: "needs_extension | future_enhancement | rejected_for_v1"
    reason: ""
stage3_execution_notes:
  - ""
quality_gate:
  name_only_response: false
  contains_model_candidate_payload: true
  contains_complete_case_spec_payload: true
  implementation_type_is_spec_variant: true
  feature_columns_are_real_columns: true
  no_custom_code: true
  no_portfolio_or_execution_semantics: true
  blockers: []
```

## `case_spec_payload` 必填字段

`case_spec_payload` 必须完整包含当前 `ModelFactorCaseSpec` 所需字段：

```yaml
case_spec_payload:
  name: ""
  factor_name: ""
  features_path: ""
  prices_path: ""
  feature_columns: []
  rebalance_frequency: "W"
  n_quantiles: 5
  direction: "long"
  universe: {}
  target:
    kind: "forward_return"
    horizon: 5
  feature_availability:
    mode: "required_timestamp | safety_lag"
    column: "known_at"
    safety_lag_days: null
  feature_preprocess: {}
  feature_importance: {}
  model: {}
  model_selection: {}
  training: {}
  neutralization: {}
  transaction_cost: {}
  output: {}
```

不能只输出 patch，例如：

```yaml
model_selection:
  enabled: true
```

这种 patch 不合格。必须输出完整 case spec。

## v1 允许的实现

允许：

- 使用已有 features 文件中的真实字段。
- 切换支持的 `model.family`：`linear / ridge / lasso / elastic_net / gbdt /
  xgboost / lightgbm / mlp`。
- 调整 `feature_columns`。
- 调整 `feature_preprocess`。
- 启用或关闭 `model_selection`。
- 使用当前支持的 selection metric，例如 `rank_ic_minus_turnover_penalty`，
  前提是 spec parser 支持。
- 调整 `training` 窗口、retrain cadence、min rows。
- 调整 `target.horizon`，但仍使用当前支持的 target kind。
- 调整 `transaction_cost.one_way_rate`。

禁止：

- 写 `feature_a * turnover_rate_f` 这类表达式列。
- 假设不存在的 `known_at`、`available_at`。
- 使用未来收益构造 feature。
- 使用自定义 sample_weight。
- 使用自定义 target construction，例如 `forward_return - lambda * turnover`，
  除非该 target 已经是当前 spec schema 的正式字段。
- 使用双窗口 selection/refit，除非当前 spec schema 已正式支持。
- 生成 Python 文件、notebook、feature builder、estimator wrapper。

## 可用性选择规则

优先顺序：

1. 可由完整 `case_spec_payload` 表达的 in-contract spec variant。
2. 低复杂度、可审计、可快速跑 exploratory screening 的变体。
3. 与 Stage1 机制最接近的可验证 proxy。
4. future enhancement 只写入 `deferred_mechanisms`，不进入 `case_spec_payload`。

如果 Stage1 中存在以下机制：

- sample weighting
- cost-shaped target
- explicit interaction builder
- two-window training
- custom estimator

默认标记为 `needs_extension`，除非用户提供当前仓库已支持的 schema 证据。

## PIT / feature availability 规则

- 如果 features 文件含 `known_at` 或 `available_at`，优先使用
  `feature_availability.mode=required_timestamp` 并填写 `column`。
- 如果没有 timestamp 列，但所有特征都是日频收盘后可得技术特征，可使用
  `feature_availability.mode=safety_lag` 和 `safety_lag_days >= 1`。
- 如果包含 `pe_ttm / pb / ps_ttm / dv_ttm` 等基本面字段，不能用
  `column=null` 的 required_timestamp；必须有真实 timestamp 或足够保守的
  safety lag，并在 `risk_controls.feature_availability_pit` 中说明。
- `feature_columns` 必须是 features 文件真实列名，不得编造 safe_bfq_35 字段。

## Turnover-Conditioned PV v1 推荐落地形态

对于 `turnover_conditioned_pv_synthesis_v1` 这类 idea，v1 首选不是 sample_weight、
custom target 或 explicit interaction，而是：

- 使用已有 price-volume / volatility / turnover feature columns。
- 保持 spec variant。
- 优先低复杂度线性族：ridge / lasso / elastic_net。
- 如果启用 `model_selection`，selection metric 优先成本/换手惩罚版本。
- 把“成交活跃但波动不过热”写入 `human_summary` 和 `risk_controls`，作为解释框架；
  不要把它写成不可执行的交互列或训练 hook。

## 不合格输出判定

出现以下任一情况，本次输出不合格：

- 只给 candidate 名字。
- 只给自然语言建议。
- 没有 `contract_version=model_stage2_candidate_output_v1`。
- 没有 `model_candidate_payload`。
- 没有完整 `case_spec_payload`。
- `case_spec_payload` 只是 patch。
- `feature_columns` 包含表达式或不存在字段。
- 把 `needs_extension` 机制写进 v1 case spec。
- 没有 `quality_gate`。
