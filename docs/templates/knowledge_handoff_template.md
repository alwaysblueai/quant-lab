# 知识交接模板

在打开 `alpha-lab` 之前使用这份模板。目标是把 vault 里的想法压缩成
一个小而可运行的包。

## 基本信息

- `idea_name`:
- `origin_cards`:
- `owner`:
- `date`:

## 研究假设

- `hypothesis`:
- `why_now_or_why_this_market`:
- `economic_or_behavioral_story`:

## 研究范围

- `market`:
- `asset_universe`:
- `frequency`:
- `holding_horizon`:
- `long_short_or_long_only`:

## 数据要求

- `required_columns`:
- `data_source_candidates`:
- `point_in_time_constraints`:
- `known_data_risks`:

## 因子或信号定义

- `raw_definition`:
- `formula_or_pseudocode`:
- `lookbacks_and_parameters`:
- `expected_sign`:

## 预处理与中性化方案

- `winsorize`:
- `standardize_or_rank`:
- `neutralize`:
- `coverage_rule`:
- `missing_value_rule`:

## 目标与评估

- `target_kind`:
- `target_horizon`:
- `primary_metrics`:
- `robustness_checks`:
- `failure_conditions`:

## alpha-lab 映射

- `factor_input_mode`: `recipe` or `csv`
- `candidate_case_name`:
- `candidate_spec_path`:
- `notes_for_codex_or_claude`:

## 就绪检查

只有在对应项已经明确时才标记 `yes`。

- `hypothesis_is_testable`:
- `data_is_identified`:
- `formula_is_concrete`:
- `target_is_defined`:
- `evaluation_plan_is_defined`:

如果还有任何一项是 `no`，先继续细化想法，再去构建 case spec。
