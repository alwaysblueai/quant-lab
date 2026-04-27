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
4. 必须保留不确定性，并写 early falsifier：第一轮什么证据会让这个机制被放弃。
5. 禁止输出最终 spec patch、JSON patch、single best model、推荐版本或完整训练方案。
6. constrained 模式最多保留 3 个机制候选，且每个候选必须在当前 contracts 内可落地；否则标记 requires_code_change。
7. 如果一个候选只是在调 alpha/lambda/window/depth 等参数，必须删除或改写成真正机制。

## 输出格式（严格遵守）
[模型机制候选]
### 机制 1
- mechanism family: loss/regularization | feature interaction | target construction | sample weighting | training window | model selection
- agent / data-generating story:
- touched contract surfaces:
- why it is not just parameter tuning:
- evidence anchor: [Kx] / [Ex] / [Fx] / external analogy + transfer cost
- early falsifier:

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

## 输出自检（系统会用 lint 校验你的输出）
- 必须包含 [模型机制候选] / [实现假设草图] / [与当前 spec / baseline 的关系] / [不确定性与失败路径] 四段。
- 模型机制候选至少 2 个；每个候选必须写清 touched contract surfaces 与 early falsifier。
- 机制发现阶段不得输出最终 spec patch、不得推荐 single best model、不得把方向写成单纯调参。
- 必须至少覆盖两类不同模型机制：loss/regularization、feature interaction、target construction、sample weighting、training window、model selection 中的两类。

---

<!-- stage: signal_mapping -->

## 已知漂移模式
- 最近一次回复缺少 early falsifier，不能直接进入 spec patch。

## 上游模型研究产物
- 机制候选：tree interaction + turnover-aware selection。

# Model Lab Idea Explorer Prompt
> Mode: constrained
> Stage: signal_mapping (next recommended: validation_kill_tests)

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
5. 输出 2-3 个可测试模型版本；禁止推荐 final pick。
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

## 输出自检（系统会用 lint 校验你的输出）
- 必须包含 [Model Mechanism Mapping] / [当前实现解释] / [模型风险控制] / [可测试模型版本] 四段。
- 模型风险控制必须逐项覆盖 feature availability / PIT、label / target leakage、overfit / complexity、turnover / cost、feature instability、split / regime fragility。
- 每个风险项必须给出 {规避 / 显式控制 / 压力测试 / 暂不控制} 之一的处理判定。
- 不得出现“推荐版本”“最优版本”“final pick”“I recommend”等最终选择语言。
- constrained 模式必须输出 2-3 个可测试模型版本，并对每个 necessary 字段写 remove-and-test 理由。

---

<!-- stage: validation_kill_tests -->

## 已知漂移模式
- 最近一次回复缺少 early falsifier，不能直接进入 spec patch。

## 上游模型研究产物
- 机制候选：tree interaction + turnover-aware selection。

# Model Lab Idea Explorer Prompt
> Mode: constrained
> Stage: validation_kill_tests

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
你处于 model-lab 的 validation_kill_tests 阶段。
目标不是证明模型有效，而是尽快判断它是否只是伪改进、泄漏、过拟合或成本假象。
请用强审计口吻；回避性结论无效。

## Alias / 问题归因靶子
- `baseline linear/ridge`: 是否只是线性基线或 ridge 的轻微调参
- `regularization-only`: 是否只是正则强弱变化而非新模型机制
- `feature-count / complexity`: 是否由更多特征或更高复杂度驱动
- `leakage / PIT`: 是否由未来信息、known_at 或目标泄漏驱动
- `split luck / regime overfit`: 是否只在某个切分或市场状态有效
- `turnover / cost artifact`: 是否被换手、成本或组合再平衡假象驱动

## Kill Test 要求
1. 数据与时间完整性：known_at / safety_lag / target horizon / 重叠窗口 / as-of 对齐。
2. 训练与验证稳健性：walk-forward、purged split、年份/市场状态切分、窗口和 retrain 频率扰动。
3. 特征与解释稳定性：top feature 稳定性、特征数量依赖、feature importance 漂移、冗余特征删除。
4. 成本与组合影响：turnover、交易成本后 IR、行业/市值/流动性桶、组合约束敏感性。
5. 如果 strict/constrained 模式下无法排除 hard kill 条件，只能输出 KILL。
6. constrained 模式必须给出 KILL 或 HOLD-FOR-AUDIT 二值判定。
7. 每个 alias verdict 必须引用 [Kx] / [Ex] / [Fx] 或当前 spec/run 字段作为锚点。

## 输出格式（严格遵守）
[Alias / 问题归因审计]
- `baseline linear/ridge`: 显著风险 / 部分风险 / 不构成风险 - 是否只是线性基线或 ridge 的轻微调参 - anchor:
- `regularization-only`: 显著风险 / 部分风险 / 不构成风险 - 是否只是正则强弱变化而非新模型机制 - anchor:
- `feature-count / complexity`: 显著风险 / 部分风险 / 不构成风险 - 是否由更多特征或更高复杂度驱动 - anchor:
- `leakage / PIT`: 显著风险 / 部分风险 / 不构成风险 - 是否由未来信息、known_at 或目标泄漏驱动 - anchor:
- `split luck / regime overfit`: 显著风险 / 部分风险 / 不构成风险 - 是否只在某个切分或市场状态有效 - anchor:
- `turnover / cost artifact`: 显著风险 / 部分风险 / 不构成风险 - 是否被换手、成本或组合再平衡假象驱动 - anchor:

[数据与时间完整性]
- PIT / known_at:
- target leakage / overlapping label:
- safety lag / as-of:
- missing / survivorship / sample filter:

[训练与验证稳健性]
- split design:
- window / retrain perturbation:
- hyperparameter freedom:
- regime and year stability:

[特征与解释稳定性]
- feature count dependence:
- top feature stability:
- redundancy / remove-and-test:
- feature importance drift:

[成本与组合影响]
- turnover and transaction cost:
- industry/size/liquidity bucket:
- portfolio construction sensitivity:
- capacity or implementation caveat:

[最终判定]
- verdict: KILL / HOLD / ITERATE / HOLD-FOR-AUDIT
- hard kill trigger, if any:
- next experiments if not killed:

## 输出自检（系统会用 lint 校验你的输出）
- 必须包含 [Alias / 问题归因审计] / [数据与时间完整性] / [训练与验证稳健性] / [特征与解释稳定性] / [成本与组合影响] / [最终判定] 六段。
- Alias / 问题归因审计必须逐项覆盖 baseline linear/ridge、regularization-only、feature-count / complexity、leakage / PIT、split luck / regime overfit、turnover / cost artifact。
- 每个 alias 必须给出 {显著风险 / 部分风险 / 不构成风险} 之一的判定。
- 最终判定不得使用“看情况”“需要更多数据”“进一步研究”等回避语。
- constrained 模式最终判定只能是 KILL 或 HOLD-FOR-AUDIT。
