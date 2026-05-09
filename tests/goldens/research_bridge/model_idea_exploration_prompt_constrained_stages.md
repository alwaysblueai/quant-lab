<!-- stage: mechanism_discovery -->

## 已知漂移模式
- 最近一次回复缺少 early falsifier，不能直接进入 spec patch。

## 上游模型研究产物
- 机制候选：tree interaction + turnover-aware selection。

# Model Lab Idea Explorer Prompt
> Mode: constrained
> Stage: mechanism_discovery (next recommended: signal_mapping)

## Research Idea
用更强的非线性模型刻画成交额加权动量，但必须保持 PIT、行业中性和换手约束。

## System Contracts
- Supported model families: linear, ridge, lightgbm
- Supported preprocess: missing_policy=drop, median_fill; scale=zscore; transform=rank, winsorize_zscore; group_scope=date, date_and_industry
- Supported training windows: rolling, expanding
- Supported selection metrics: rank_ic, rank_ic_ir, turnover_penalized_rank_ic
- Supported feature importance modes: permutation
- Stay inside current model-factor contracts unless explicitly marked as code-change needed.

## Current Spec Context
- Spec name: stock_ridge_momentum
- Factor name: stock_ridge_momentum
- Model family: ridge
- Feature count: 3
- Feature sample: ret_20d, turnover_20d, size
- Target horizon: 5
- Direction / rebalance / quantiles: long_short / W / 5
- Feature availability mode: known_at
- Feature availability column: known_at
- Feature availability safety lag days: 1
- Training window type: rolling
- Training details: {"min_periods": 126, "window": 252}
- Feature preprocess: {"cross_sectional_transform": "rank", "missing_policy": "median_fill"}
- Model selection enabled: True
- Model selection metric: rank_ic_ir
- Model selection details: {"top_k": 2}
- Feature importance: {"mode": "permutation"}
- Neutralization: {"industry": true, "size": true}
- Transaction cost: {"bps": 10}

## Knowledge Context
- [K1] Finance Tree (factor): Known_at + winsorize + turnover-aware model selection.
- handling: enforce known_at/as-of or safety_lag for PIT consistency
- handling: evaluate turnover-aware selection and cost-aware IR together

## Experiment Context
- [F1] run=gbdt_unlagged_fundamentals status=failed outputs=metrics.json reason=known_at missing for restated fundamentals
- [E1] run=ridge_momentum_202604 model=ridge mean_rank_ic=0.041 rank_ic_ir=0.72 ls_ir=0.83 net_ls_ir=0.61 turnover=0.18 coverage=0.97 outputs=metrics.json, summary.md

## Session Memory
- [M1] 2026-04-27T09:00:00Z stage=mechanism_discovery mode=constrained responded=True lint_errors=False idea=Test nonlinear feature interactions under cost limits. patch=candidate uses lightgbm and turnover penalty

## Code Anchors
- src/alpha_lab/model_factor/core.py: ModelFamily and estimator behavior
- src/alpha_lab/real_cases/model_factor/spec.py: spec contracts and feature availability rules

## Candidate Spec Patch Hint
- summary: Switch ridge baseline to turnover-aware LightGBM candidate.
- requires_code_change: False
- patch_fields:
```json
{
  "model": {
    "family": "lightgbm"
  },
  "model_selection": {
    "enabled": true,
    "selection_metric": "turnover_penalized_rank_ic"
  },
  "feature_preprocess": {
    "missing_policy": "median_fill",
    "cross_sectional_transform": "rank"
  }
}
```

## Warnings
- available_data excludes intraday_tick_volume; keep daily-only.

## Task
## 阶段声明
你处于 model-lab 的 mechanism_discovery 阶段。
目标是发现可能提升模型的结构性机制，而不是给出最终模型、最终 spec patch 或单一最佳方案。

## 机制发现规则
1. 只讨论模型改进机制：loss/regularization、feature interaction、target construction、sample weighting、training window、model selection。
2. 每个机制必须写清 touched contract surfaces：model、feature_preprocess、feature_availability、training、model_selection、feature_importance。
3. 必须说明该机制与当前 spec / baseline 的差异：是新机制、现有机制强化，还是只是参数调节。
4. 必须保留不确定性，并写 concern：第一轮数据验证要重点观察什么。
5. 禁止输出最终 spec patch、JSON patch、single best model、推荐版本或完整训练方案。

## Stage 1 ledger 协议
本 prompt 只负责 model-lab 的 Stage 1：用 vault 素材起草模型改进机制。输出必须能转写为 `ledger_v1.yaml` + `retrieval_log.md`；不要给 kill 结论。
- vault 是素材库，不是判决书；`transferable_moves` 是主要生成原料。
- `operative_claims` 只能作为弱上下文 hint，不能触发 precedent kill。
- 每条 model mechanism 至少写 `hypothesis` / `signal_sketch` / `data_needs`，并说明 touched contract surfaces。
- `inspired_by` / `fusion_of` / `cross_domain_jump` 都是可选溯源字段；有来源就写，novel synthesis 无来源也合法。
- 不要强制 lineage 或从来源卡继承 kill 条件。需要担心的点写成 `concern`，留给 Stage 3 数据验证。
6. 优先尝试跨领域迁移：把 daily PV、组合构建、稳健统计或其他 asset class 的 move 搬到模型机制。
7. 尝试至少一个多卡 fusion 候选；弱点只标 concern，不删除。
8. constrained 模式输出 2-4 个机制候选，且每个候选必须说明当前 contracts 内可落地还是 requires_code_change。
9. 如果一个候选只是在调 alpha/lambda/window/depth 等参数，必须改写成真正机制或标 `concern: parameter-only`。

## 输出格式（严格遵守）
[模型机制候选]
### 机制 1
- mechanism family: loss/regularization | feature interaction | target construction | sample weighting | training window | model selection
- agent / data-generating story:
- touched contract surfaces:
- why it is not just parameter tuning:
- evidence anchor: [Kx] / [Ex] / [Fx] / external analogy + transfer cost
- inspired_by: [{card, what_i_took, cross_domain_jump}]（可选）
- fusion_of: [[card_1, card_2]]（可选）
- novel_delta:
- concern:

[实现假设草图]
- 每个候选只写可讨论的实现轮廓，不写最终 patch。
- 标记 in-contract / needs-extension / requires_code_change。

[与当前 spec / baseline 的关系]
- current baseline captured:
- structural difference:
- likely unchanged pieces:

[不确定性与失败路径]
- PIT / label leakage risk:
- overfit / split fragility risk:
- turnover / cost risk:
- feature instability risk:

[ledger_v1.yaml 草案]
- mechanisms: hypothesis / inspired_by（可选）/ fusion_of（可选）/ novel_delta / signal_sketch / data_needs / concern
- retrieval_log: surfaced_cards + transferable_moves + operative_claims weak hints

## 输出自检（系统会用 lint 校验你的输出）
- 必须包含 [模型机制候选] / [实现假设草图] / [与当前 spec / baseline 的关系] / [不确定性与失败路径] 四段。
- 模型机制候选至少 2 个；每个候选必须写清 touched contract surfaces 与 concern。
- 机制发现阶段不得输出最终 spec patch、不得推荐 single best model、不得把方向写成单纯调参。
- 必须至少覆盖两类不同模型机制：loss/regularization、feature interaction、target construction、sample weighting、training window、model selection 中的两类。
- 每个机制候选应能落入 ledger_v1：包含 hypothesis / signal_sketch / data_needs，inspired_by / fusion_of / cross_domain_jump 可选；无来源不算缺口。

---

<!-- stage: signal_mapping -->

## 已知漂移模式
- 最近一次回复缺少 early falsifier，不能直接进入 spec patch。

## 上游模型研究产物
- 机制候选：tree interaction + turnover-aware selection。

# Model Lab Idea Explorer Prompt
> Mode: constrained
> Stage: signal_mapping

## Research Idea
用更强的非线性模型刻画成交额加权动量，但必须保持 PIT、行业中性和换手约束。

## System Contracts
- Supported model families: linear, ridge, lightgbm
- Supported preprocess: missing_policy=drop, median_fill; scale=zscore; transform=rank, winsorize_zscore; group_scope=date, date_and_industry
- Supported training windows: rolling, expanding
- Supported selection metrics: rank_ic, rank_ic_ir, turnover_penalized_rank_ic
- Supported feature importance modes: permutation
- Stay inside current model-factor contracts unless explicitly marked as code-change needed.

## Current Spec Context
- Spec name: stock_ridge_momentum
- Factor name: stock_ridge_momentum
- Model family: ridge
- Feature count: 3
- Feature sample: ret_20d, turnover_20d, size
- Target horizon: 5
- Direction / rebalance / quantiles: long_short / W / 5
- Feature availability mode: known_at
- Feature availability column: known_at
- Feature availability safety lag days: 1
- Training window type: rolling
- Training details: {"min_periods": 126, "window": 252}
- Feature preprocess: {"cross_sectional_transform": "rank", "missing_policy": "median_fill"}
- Model selection enabled: True
- Model selection metric: rank_ic_ir
- Model selection details: {"top_k": 2}
- Feature importance: {"mode": "permutation"}
- Neutralization: {"industry": true, "size": true}
- Transaction cost: {"bps": 10}

## Knowledge Context
- [K1] Finance Tree (factor): Known_at + winsorize + turnover-aware model selection.
- handling: enforce known_at/as-of or safety_lag for PIT consistency
- handling: evaluate turnover-aware selection and cost-aware IR together

## Experiment Context
- [F1] run=gbdt_unlagged_fundamentals status=failed outputs=metrics.json reason=known_at missing for restated fundamentals
- [E1] run=ridge_momentum_202604 model=ridge mean_rank_ic=0.041 rank_ic_ir=0.72 ls_ir=0.83 net_ls_ir=0.61 turnover=0.18 coverage=0.97 outputs=metrics.json, summary.md

## Session Memory
- [M1] 2026-04-27T09:00:00Z stage=mechanism_discovery mode=constrained responded=True lint_errors=False idea=Test nonlinear feature interactions under cost limits. patch=candidate uses lightgbm and turnover penalty

## Code Anchors
- src/alpha_lab/model_factor/core.py: ModelFamily and estimator behavior
- src/alpha_lab/real_cases/model_factor/spec.py: spec contracts and feature availability rules

## Candidate Spec Patch Hint
- summary: Switch ridge baseline to turnover-aware LightGBM candidate.
- requires_code_change: False
- patch_fields:
```json
{
  "model": {
    "family": "lightgbm"
  },
  "model_selection": {
    "enabled": true,
    "selection_metric": "turnover_penalized_rank_ic"
  },
  "feature_preprocess": {
    "missing_policy": "median_fill",
    "cross_sectional_transform": "rank"
  }
}
```

## Warnings
- available_data excludes intraday_tick_volume; keep daily-only.

## Task
## 阶段声明
你处于 model-lab 的 signal_mapping 阶段。
目标是把上游模型机制翻译成可测试 spec/run 版本；不是生成新机制，也不是选择赢家。

## 映射规则
1. 使用 Mechanism -> implication -> spec/data 三段映射；每个字段必须解释为什么必要。
2. 对每个 required field 标注 role: necessary / decorative / risk control。
3. necessary 字段必须写 remove-and-test 理由：删掉它会破坏哪条机制链。
4. 每个版本都要说明控制哪些模型风险，哪些风险暂不控制以及原因。
5. 输出 2-3 个可测试模型版本；禁止推荐 final pick。Stage 3 数据验证 / spec 审计才做最终 KILL / HOLD-FOR-AUDIT。

## Stage 1 ledger 协议
本 prompt 只负责 model-lab 的 Stage 1：用 vault 素材起草模型改进机制。输出必须能转写为 `ledger_v1.yaml` + `retrieval_log.md`；不要给 kill 结论。
- vault 是素材库，不是判决书；`transferable_moves` 是主要生成原料。
- `operative_claims` 只能作为弱上下文 hint，不能触发 precedent kill。
- 每条 model mechanism 至少写 `hypothesis` / `signal_sketch` / `data_needs`，并说明 touched contract surfaces。
- `inspired_by` / `fusion_of` / `cross_domain_jump` 都是可选溯源字段；有来源就写，novel synthesis 无来源也合法。
- 不要强制 lineage 或从来源卡继承 kill 条件。需要担心的点写成 `concern`，留给 Stage 3 数据验证。
6. constrained 模式下，每个版本必须是当前 contracts 内可运行的最小变更，或明确 requires_code_change。
7. 当前实现必须给出 binary alias-tag：是否只是 baseline linear/ridge 或 regularization-only 的换壳。

## 模型风险控制清单
- `feature availability / PIT`: known_at / safety_lag / as-of 对齐
- `label / target leakage`: forward label、重叠窗口、目标构造泄漏
- `overfit / complexity`: 模型复杂度、样本量、超参自由度
- `turnover / cost`: 换手惩罚、成本后 IR、可交易性
- `feature instability`: 特征重要性、top feature 稳定性、冗余特征
- `split / regime fragility`: walk-forward、purged split、年份/市场状态稳定性

## 输出格式（严格遵守）
[Model Mechanism Mapping]
### 机制 1
- mechanism:
- implication:
- required_data_or_spec_fields:
  - field: name | role: necessary/decorative/risk control | remove-and-test reason:
- current-contract fit:

[当前实现解释]
- current implementation captures:
- current implementation misses:
- cannot disambiguate at current data/spec tier:

[模型风险控制]
- `feature availability / PIT`: 规避 / 显式控制 / 压力测试 / 暂不控制 - known_at / safety_lag / as-of 对齐 - one-line reason
- `label / target leakage`: 规避 / 显式控制 / 压力测试 / 暂不控制 - forward label、重叠窗口、目标构造泄漏 - one-line reason
- `overfit / complexity`: 规避 / 显式控制 / 压力测试 / 暂不控制 - 模型复杂度、样本量、超参自由度 - one-line reason
- `turnover / cost`: 规避 / 显式控制 / 压力测试 / 暂不控制 - 换手惩罚、成本后 IR、可交易性 - one-line reason
- `feature instability`: 规避 / 显式控制 / 压力测试 / 暂不控制 - 特征重要性、top feature 稳定性、冗余特征 - one-line reason
- `split / regime fragility`: 规避 / 显式控制 / 压力测试 / 暂不控制 - walk-forward、purged split、年份/市场状态稳定性 - one-line reason

[可测试模型版本]
- v1: mechanism | minimal spec/run delta | controls | residual assumptions
- v2: mechanism | minimal spec/run delta | controls | residual assumptions
- v3: optional; only if structurally different

[ledger_v1.yaml 草案]
- mechanisms: 每条机制保留 hypothesis / inspired_by（可选）/ fusion_of（可选）/ novel_delta / signal_sketch / data_needs / concern
- retrieval_log: surfaced_cards + transferable_moves + operative_claims weak hints

[retrieval_log.md 草案]
- surfaced_cards: 本阶段实际使用的 [Kx] / [Ex] / [Fx] 与用途
- spec_dependency_notes: 当前 contracts、required fields 与 data tier 边界

## 输出自检（系统会用 lint 校验你的输出）
- 必须包含 [Model Mechanism Mapping] / [当前实现解释] / [模型风险控制] / [可测试模型版本] 四段。
- 模型风险控制必须逐项覆盖 feature availability / PIT、label / target leakage、overfit / complexity、turnover / cost、feature instability、split / regime fragility。
- 每个风险项必须给出 {规避 / 显式控制 / 压力测试 / 暂不控制} 之一的处理判定。
- 不得出现“推荐版本”“最优版本”“final pick”“I recommend”等最终选择语言；最终选择属于 Stage 3 数据验证。
- constrained 模式必须输出 2-3 个可测试模型版本，并对每个 necessary 字段写 remove-and-test 理由。
