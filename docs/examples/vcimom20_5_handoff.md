# VCIMom20_5 Handoff

This is the first complete bridge example using the `quant-knowledge -> alpha-lab`
workflow. It binds a compact research handoff to an existing runnable case spec
instead of cloning a duplicate YAML file.

## Identity

- `idea_name`: `VCIMom20_5`
- `origin_cards`:
  - `/mnt/c/quant/vault/quant-knowledge/90_moc/MOC - Expected Returns in A-share Research.md`
  - `/mnt/c/quant/vault/quant-knowledge/90_moc/MOC - Whale-Quant Applied Research.md`
  - `/mnt/c/quant/vault/quant-knowledge/20_methods/signal/Method - A-share High-Frequency Factor Taxonomy (Five-Family Framework).md`
  - `/mnt/c/quant/vault/quant-knowledge/20_methods/validation/Method - Alpha IC Evaluation & Factor Backtest Report.md`
- `owner`: `yukun_zhao`
- `date`: `2026-03-29`

## Research Thesis

- `hypothesis`: A-share daily momentum is more robust when the trend is measured on market-residual returns, confirmed by abnormal trading amount, and penalized when recent gains look like blow-off spikes.
- `why_now_or_why_this_market`: Raw momentum in A-shares is easily polluted by market beta, crowding, and short-horizon reversal. A volume-confirmed residual formulation is a better first-pass daily factor than plain price momentum.
- `economic_or_behavioral_story`: Persistent idiosyncratic demand should appear as residual price drift with supportive participation. Extremely strong positive residual-return and amount shocks are more likely to signal crowding and near-term mean reversion than durable continuation.

## Scope

- `market`: `a_share`
- `asset_universe`: `baostock_all_a`
- `frequency`: `daily`
- `holding_horizon`: `5 trading days`
- `long_short_or_long_only`: `long-only factor ranking validation` with quintile diagnostics

## Data Requirements

- `required_columns`: `date`, `asset`, `close`, and either `amount` or `volume`
- `data_source_candidates`: `BaoStock` first, `Tushare` second
- `point_in_time_constraints`: use point-in-time universe membership and standard suspension / invalid-trading filters from the `alpha-lab` data-source flow
- `known_data_risks`:
  - `amount` is preferred because confirmation is based on trading-amount abnormality
  - the current implementation accepts `volume` if `amount` is absent
  - short windows plus aggressive universe filtering can make the factor sparse

## Factor or Signal Definition

- `raw_definition`: cross-sectional combination of residual momentum, amount-confirmation, and blow-off penalty
- `formula_or_pseudocode`:
  1. compute daily returns from `close`
  2. compute daily cross-sectional market return
  3. estimate rolling beta over `60` days and derive residual return
  4. momentum leg = residual-return sum over days `t-20` to `t-5`
  5. amount leg = rolling z-score of log trading amount over `20` days
  6. confirmation leg = rolling correlation of residual return and amount z-score over `10` days
  7. penalty leg = rolling max of positive residual-return × positive amount-z over `5` days
  8. final signal = `z(momentum_leg) + 0.6 * z(confirm_leg) - 0.4 * z(penalty_leg)`
- `lookbacks_and_parameters`:
  - `residual_window: 60`
  - `momentum_window: 20`
  - `skip_recent: 4`
  - `confirm_window: 10`
  - `penalty_window: 5`
  - `amount_window: 20`
  - `confirm_weight: 0.6`
  - `penalty_weight: 0.4`
- `expected_sign`: higher values should rank names with more attractive forward 5-day returns

## Preprocess and Neutralization Plan

- `winsorize`: cross-sectional `1% / 99%`
- `standardize_or_rank`: cross-sectional `zscore`
- `neutralize`: none in the first validation pass beyond the internal residual-return construction
- `coverage_rule`: `min_coverage = 0.2`
- `missing_value_rule`: keep case-level preprocess as no-op because recipe preprocess is already active

## Target and Evaluation

- `target_kind`: `forward_return`
- `target_horizon`: `5`
- `primary_metrics`:
  - `rank_ic_mean`
  - `rank_ic_ir`
  - quintile monotonicity
  - top-minus-bottom spread
  - turnover
- `robustness_checks`:
  - shorter date window first for runtime control
  - inspect whether confirmation and penalty legs behave sensibly across volatile subperiods
  - compare against simpler residual momentum later
- `failure_conditions`:
  - factor becomes empty after universe filtering
  - rank IC is unstable and non-monotonic
  - turnover is too high relative to the 5-day horizon

## alpha-lab Mapping

- `factor_input_mode`: `recipe`
- `candidate_case_name`: `vcimom20_5_webui_demo`
- `candidate_spec_path`: *(original demo YAML
  `configs/real_cases/single_factor/vcimom20_5_webui_baostock.yaml` was
  retired in the 2026-05 cleanup; copy
  `configs/real_cases/single_factor/_template_exploration.yaml` and
  rewire `factor_input` per the original handoff if you want to re-run.)*
- `notes_for_codex_or_claude`:
  - use `BaoStock`
  - start with a short runtime window such as `2024-01-01` to `2024-03-31`
  - start with `asset_limit = 30`
  - if runtime or sparsity is a problem, debug the universe and factor coverage before changing the factor logic

## Ready Check

- `hypothesis_is_testable`: `yes`
- `data_is_identified`: `yes`
- `formula_is_concrete`: `yes`
- `target_is_defined`: `yes`
- `evaluation_plan_is_defined`: `yes`

## Bound Spec

The original runnable spec (`vcimom20_5_webui_baostock.yaml`) was retired in
the 2026-05 backend cleanup. Re-running this handoff requires copying
`configs/real_cases/single_factor/_template_exploration.yaml` and rewiring
the fields above. This example intentionally reused an existing spec rather
than creating a second equivalent YAML file.
