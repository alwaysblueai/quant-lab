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
2. 在 `custom_models/research/<candidate_name>/research_log.md` 记录冲突。
3. 不要根据自由文本自行改写 case spec。

## 允许写入

只能写入或更新：

- `custom_models/research/<candidate_name>/model_candidate.json`
- `custom_models/research/<candidate_name>/research_log.md`
- `configs/real_cases/model_factor/<candidate_name>_vN.yaml`

禁止写入：

- 临时 Python 脚本
- notebook
- 散落的 `.py` 文件
- `custom_models/promoted`
- `src/alpha_lab/model_factor/`、`src/alpha_lab/factors/`、其他 core 模块
- 前端正式注册文件
- execution / replay / fill simulation / portfolio construction 相关目录

v1 不接受任何自定义 feature builder code 或自定义 estimator code；只能通过
`case_spec_payload` 中现有的 `model.family` / `feature_columns` /
`feature_preprocess` / `training` 字段进行变体。

## model_candidate.json 合同

`model_candidate_payload` 必须可以直接写入：

```text
custom_models/research/<candidate_name>/model_candidate.json
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
2. 写入 `custom_models/research/<candidate_name>/model_candidate.json`。
3. 创建或更新 `configs/real_cases/model_factor/<candidate_name>_vN.yaml`。
   YAML 内容必须与 `case_spec_payload` 完全一致；不要根据 `human_summary` 调参。
4. 先运行 validator：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab validate-draft-model custom_models/research/<candidate_name>/model_candidate.json
```

5. validator 通过后，按照 `run_controls` 设定的 profile 跑标准 pipeline。

快速初筛（Codex GUI 后端默认）：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab real-case model-factor run \
  configs/real_cases/model_factor/<candidate_name>_v1.yaml \
  --evaluation-profile exploratory_screening \
  --screening-retrain-every-n-dates 40 \
  --render-report \
  --vault-export-mode skip \
  --draft-model-candidate custom_models/research/<candidate_name>/model_candidate.json
```

更完整验证（候选成熟后，可在前端 Draft Candidates 触发完整报告）：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab real-case model-factor run \
  configs/real_cases/model_factor/<candidate_name>_v1.yaml \
  --evaluation-profile default_research \
  --render-report \
  --vault-export-mode skip \
  --draft-model-candidate custom_models/research/<candidate_name>/model_candidate.json
```

## Preflight 必查

- `model_candidate.json` 是合法 JSON。
- `contract_version=stage2_model_candidate_v1`、`implementation_status=draft_for_stage3`、`implementation_type=spec_variant`。
- `candidate_name` 是英文 snake_case，长度 3-64。
- 文件位于 `custom_models/research/<candidate_name>/model_candidate.json`。
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

Stage3 的最终交付物是**一份固定 schema 的迭代反馈包**，供网页版 GPT 在同一个
候选模型项目里生成下一版 Stage2 `case_spec_payload`。它有两个落点：

**(a) durable trail —— 追加一行到 `research_log.md`**（≤ 80 字符，只写流水、不写
决策文档）：

```text
2026-05-26  v1 ridge baseline  case=pv_turnover_v1_v1.yaml  art=outputs/.../pv_turnover_v1_v1  RankIC=0.012 IR=0.18  verdict=keep-iterating
```

**(b) feedback pack —— 作为本轮 Stage3 的最终结构化输出（贴进网页版 GPT 的就是
这一块）**。固定 schema 如下；prose 字段用中文，专有名词/代码符号保留英文：

```yaml
model_stage3_feedback_version: model_factor_stage3_feedback_v1
idea_id: <从 model_candidate.json provenance 原样带出>
candidate_name: <candidate_name>
iteration: <本次评估的 payload 版本，如 v1>
model_thesis: <一句话机制假设，逐字保留、除非用户显式重审>
base_case_spec_path: <case_spec_payload.base_case_spec_path>
run_output_dir: <artifact 输出目录>

contract:                       # 工程契约一行带过，不展开审计细节
  validator: passed             # validate-draft-model 结果
  source_hash_audit: ok         # candidate_json / case_spec / feature_contract sha256 + source path
  artifact_audit: ok            # run_manifest + model_definition + feature_manifest draft_model_source
  provenance_idea_id: present

verdict:
  campaign_triage: <如 "Drop for now" / "Keep iterating">
  promotion_tier_L: <pass / blocked>   # 共享指标门槛见 factor_promotion_checklist.md Tier L
  one_line: <一句话定性，如 "契约全过；rank_ic 偏低、训练通过率不稳">

scorecard:                      # 共享指标门槛取 factor_promotion_checklist.md Tier L
  rank_ic_mean:   {value: <num>, gate: ">=0.01", pass: <bool>}
  rank_ic_ir:     {value: <num>, gate: ">=0.15", pass: <bool>}
  cost_ir_5bps:   {value: <num>, gate: ">=0.05", pass: <bool>}
  coverage_mean:  {value: <num>, gate: ">=0.65", pass: <bool>}
  # model 专属诊断（无硬门槛，给值供判断）：
  training_pass_rate:       <num>
  ic_decay_rebalance_ratio: <num>
  model_selection:          <status>
  portfolio_validation:     <status>
  core_gap: <一句话：距 Tier C 核心档还差什么；不到 Tier L 就写 "未到准入档">

blockers:                       # 真正卡住准入的 1-3 条现象（不是动作）
  - <如 特征覆盖率低 / 训练样本不足 / 机制不兼容 / rolling 不稳>

codex_assessment:               # Codex 自主研判：advisory，网页版 GPT 可推翻
  read: |                       # 自由文本，≤ ~150 字：本轮结果说明了什么、为何强/弱
    <训练稳定性 / feature importance 分布 / 过拟合 / 失败模式 / 有效之处；不复述 scorecard 数字>
  recommended_directions:       # ranked，最多 4 条，每条一句理由
    - {move: <如 train_window 拉到 24 个月>, rationale: <为何>, confidence: high}
  lead_pick: <如果只能改一处 case_spec 字段先改哪个 + 一句为什么>

iteration_request:
  preserve: [idea_id, model_thesis, feature_availability PIT 合同, base_case_spec_path, "spec_variant only（不引入自定义 code）"]
  do_not: [新增未注册 feature 列, 自定义 feature builder/estimator code, needs_extension 机制, Level3/portfolio/execution 语义, 换数据/绕 validator]
  try_next:                     # codex_assessment 的 lead_pick + top directions 蒸馏成 case_spec 改动，≤3 条
    - <如 feature_columns 增减 / feature_preprocess / model.family / training.train_window / target.horizon>

resource: <一行；如 "oom: not_triggered, peak_rss vs budget ok"；触发或逼近预算才展开>

history:                        # 读取 research_log.md 重建，保证贴最新块也不丢轨迹
  - {v: v1, change: <一句话>, rank_ic: <num>, verdict: <如 keep-iterating>}
```

规则：

- scorecard 共享指标行的 `gate` 值必须与 `docs/factor_promotion_checklist.md`
  的 **Tier L** 保持一致；改了 checklist 就同步改这里。model 专属诊断行不设硬门槛。
- 填不出可靠值的字段**直接省略**，不要用 `<value>` / `<未知>` 占位凑格式。
- `idea_id`、`model_thesis`、`base_case_spec_path` 必须逐字延续，这是网页版 GPT
  识别"在迭代同一个候选模型"的对齐锚。
- `codex_assessment` 是 advisory，网页版 GPT 可推翻；它**不改** `preserve` /
  `do_not` 硬约束，且 `recommended_directions` 必须全部落在 `do_not` 之外（不准
  建议新增未注册 feature 列 / 自定义 code / `needs_extension` / Level3 / 换数据绕
  validator）。
- `try_next` 必须是 `codex_assessment` 的蒸馏，二者不得矛盾；只能是 `case_spec_payload`
  字段调整，v1 不接受自定义 code。探索阶段不 KILL，最重只能 "drop for now"。

禁止输出：

- 模型已经有效
- 可以实盘
- 买入 / 卖出建议
- portfolio construction 结论
- Level 3 execution / fill simulation / replay 建议
