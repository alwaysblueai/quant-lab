# Experiment Feedback Template

在运行完 `alpha-lab` case 并导出机器生成的 experiment card 之后使用这份模板。
它用于记录研究者自己写的解释。请用中文主导的表达来写，只有必要的
技术术语、专有名词、缩写、公式或代码符号才保留英文。

## 基本信息

- `experiment_name`:
- `spec_path`:
- `origin_cards`:
- `exported_card_path`:

## 结论

- `status`: `keep` / `revise` / `drop`
- `one_sentence_verdict`:

## 有效之处

- `signal_quality`:
- `cross_sectional_monotonicity`:
- `stability_or_regime_observations`:

## 失效之处

- `implementation_or_data_issues`:
- `weak_diagnostics`:
- `suspected_leakage_or_bias_checks`:

## 需要记录的关键数字

- `rank_ic_mean`:
- `rank_ic_ir`:
- `top_minus_bottom`:
- `turnover`:
- `max_drawdown_or_risk_note`:

## 解释

- `what_the_result_likely_means`:
- `what_it_does_not_mean`:
- `is_this_additive_to_existing_signals`:

## 下一步动作

- `next_experiment`:
- `parameter_change`:
- `data_change`:
- `should_update_theory_card`:

## Vault 更新规则

- 如果结果只是探索性的，只更新导出的 experiment card，保持 origin card 的生命周期不变。
- 如果结果可重复，并且通过了计划内检查，再把 origin card 从 `theoretical`
  晋升到 `validated-backtest`。
- 不要用没有 experiment 证据支撑的叙事性结论去覆盖 theory card。
