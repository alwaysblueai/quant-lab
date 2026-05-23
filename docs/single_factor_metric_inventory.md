# 单因子指标 Inventory 与验证路线

本文档是单因子 Level 1/2 评估指标的对账清单。后续新增、删除或改口径时，先更新这里，再更新测试和 golden snapshot。

边界声明：本文只覆盖因子发现和组合构建验证。`next_open` 是单因子与模型合成信号回测的默认标签执行口径；tradability、cost、capacity 等字段仍只作为研究诊断和敏感性分析登记，不改变 Level 1/2 研究边界。

## 总口径

| 项目 | 当前口径 |
| --- | --- |
| 因子输入 | canonical long-form: `date`, `asset`, `factor`, `value`。单次评估要求单一 `factor` 名称。 |
| 标签输入 | canonical long-form: `date`, `asset`, `factor`, `value`。默认由 `forward_return(prices, horizon=h, execution_price_mode="next_open")` 产生。 |
| 默认标签对齐 | 因子 `value(t)` 与标签 `forward_return_h_next_open(t)` 按 `(date, asset)` 内连接；标签值为 `close[t+h] / open[t+1] - 1`，存放在 `t`。 |
| 其他执行价标签 | `close`: `close[t+h] / close[t] - 1`；`vwap`: `close[t+h] / vwap[t+1] - 1`。`close` 仅用于显式 legacy close-to-close 复核。 |
| 单因子预处理 | `single_factor/pipeline.py` 的 `_prepare_factor` 默认先按日 winsorize，再按日 z-score；可选 rank 或 none；`min_coverage` 可将低覆盖日期置为 NaN。 |
| Split 口径 | 有 `TimeSeriesSplitContract` 时，核心评估用 OOS；报告曲线可附 `split_phase`。IC、rolling、group returns、turnover 打包时 drop EMBARGO；coverage 保留 EMBARGO 用于审计。 |
| 截面权重 | IC、RankIC、MI、分组收益均按日期截面计算；汇总时每个有效日期等权。 |
| 分组权重 | `quantile_returns` 是桶内简单平均；capacity 诊断可另算 market-cap weighted 多空均值。 |
| IR/Sharpe | `ic_ir`、`rank_ic_ir`、`long_short_ir` 均是未年化的 `mean / std(ddof=1)`。写 artifact 时的 `annualized_return`、`annualized_volatility`、`sharpe` 由 `artifact_enrichment.py` 基于非重叠长短收益序列另算。 |
| 费用 | `cost_rate` 是单边每期线性费率。`adjusted_return(t) = long_short_return(t) - cost_rate * turnover(t)`。首期 turnover 为 NaN，因此首期 adjusted return 也是 NaN。 |

## 产物地图

| 产物 | 主要字段 | 生产位置 | 口径 |
| --- | --- | --- | --- |
| `ExperimentResult.ic_df` | `date`, `factor`, `label`, `ic` | `evaluation.compute_ic` | 每日 Pearson 截面相关。 |
| `ExperimentResult.rank_ic_df` | `date`, `factor`, `label`, `rank_ic` | `evaluation.compute_rank_ic` | 每日 Spearman RankIC。 |
| `ExperimentResult.mutual_information_df` | `date`, `factor`, `label`, `mutual_information` | `evaluation.compute_mutual_information` | 每日截面离散 MI，默认 `max_bins=10`，实际 bin 数受 `sqrt(n)` 和有效样本限制。 |
| `ExperimentResult.quantile_returns_df` | `date`, `factor`, `quantile`, `mean_return` | `quantile.quantile_returns` | 因子同日截面分桶后，桶内标签均值。 |
| `ExperimentResult.long_short_df` | `date`, `factor`, `long_short_return` | `quantile.long_short_return` | 最高已占用桶减最低已占用桶。 |
| `ExperimentResult.quantile_assignments_df` | `date`, `asset`, `factor`, `quantile` | `quantile.quantile_assignments` | 只看因子，不要求标签可用；末尾无标签日期仍可能出现。 |
| `ExperimentResult.quantile_turnover_df` | `date`, `factor`, `quantile`, `turnover` | `turnover.quantile_turnover` | 桶成员 one-way entry rate；首个日期为 NaN。 |
| `ExperimentResult.long_short_turnover_df` | `date`, `factor`, `long_short_turnover` | `turnover.long_short_turnover` | top/bottom 桶 turnover 均值；任一 leg NaN 则 NaN。 |
| `ExperimentResult.rolling_stability_df` | `date`, rolling mean/positive-rate columns | `experiment._build_rolling_stability_frame` | 对 IC、RankIC、MI、多空收益滚动窗口计算。 |
| `ExperimentResult.summary` | `ExperimentSummary` 标量 | `experiment._summarise` | 核心标量汇总。 |
| `SingleFactorEvaluationResult.metrics` | 研究卡片/JSON 标量 | `single_factor/evaluate.py` | 核心标量加诊断、判定、split scope 字段。 |
| `SingleFactorEvaluationResult.ic_timeseries` | `date`, `ic`, `rank_ic`, `mutual_information`, optional `split_phase` | `_build_ic_timeseries_frame` | 报告用 IC 曲线。 |
| `SingleFactorEvaluationResult.ic_decay` | `horizon`, `mean_ic`, `mean_rank_ic`, `ic_ir`, `t_stat`, `p_value`, `n_dates` | `decay.compute_ic_decay` | 默认 horizons 为 `{1,2,3,5,10,20,target_horizon}`。 |
| `SingleFactorEvaluationResult.factor_autocorrelation` | `lag`, `mean_autocorr`, `std_autocorr`, `n_dates` | `decay.compute_factor_autocorrelation` | 默认 lags 为 `(1,2,3,5,10)`，按截面 rank autocorr。 |
| `SingleFactorEvaluationResult.group_returns` | `date`, `factor`, `group`, `group_return`, optional `split_phase` | `quantile_returns_df` rename | 报告用分组收益曲线。 |
| `SingleFactorEvaluationResult.turnover` | `date`, `factor`, `turnover`, optional `split_phase` | `long_short_turnover_df` rename | 报告用多空 turnover 曲线。 |
| `SingleFactorEvaluationResult.coverage` | per-date coverage count/ratio fields | `_build_effective_coverage_by_date` | eligible universe 对因子和标签的有效覆盖。 |
| `SingleFactorEvaluationResult.capacity_estimation` | capacity summary row | `_build_capacity_estimation` | 可选容量/市值权重诊断。 |
| `SingleFactorEvaluationResult.conditional_ic_by_magnitude` | magnitude bucket summary | `grouped_evaluation.conditional_ic_by_factor_magnitude` | 按每日 `abs(factor)` 分桶后再算条件 IC。 |
| `SingleFactorEvaluationResult.conditional_ic_by_cross_section_size` | cross-section bucket summary | `grouped_evaluation.conditional_ic_by_cross_section_size` | 小/大截面日期的 IC 诊断。 |
| `SingleFactorEvaluationResult.lag_sensitivity` | `lag`, `mean_ic`, `long_short_ir` | `_merge_signal_lag_sensitivity_metrics` | 因子按资产向后滞后 `0,1,2,3` 天后重算。 |
| `SingleFactorEvaluationResult.random_baseline_null` | `permutation`, `mean_ic` | `_merge_random_factor_baseline_metrics` | 每日截面内 permutation 的 RankIC null。 |
| `SingleFactorEvaluationResult.daily_pnl_attribution` | `date`, `long_leg`, `short_leg`, `gross`, `cost_drag`, `net` | `_merge_daily_pnl_attribution_metrics` | 多空日收益分解。 |

## 指标 Inventory

### 截面预测力

| 指标/曲线 | 输入字段 | 对齐 | 计算口径 | 输出位置 |
| --- | --- | --- | --- | --- |
| `ic` | factor `value`, label `value` | 同 `(date, asset)` | 每日 Pearson 截面相关；有效资产小于 2 或常数截面为 NaN。 | `ic_df`, `ic_timeseries` |
| `rank_ic` | factor `value`, label `value` | 同 `(date, asset)` | 每日 Spearman 相关；rank 使用 scipy/pandas 语义。 | `rank_ic_df`, `ic_timeseries` |
| `mutual_information` | factor `value`, label `value` | 同 `(date, asset)` | 每日 rank-quantile 离散 MI，单位 nats。 | `mutual_information_df`, `ic_timeseries` |
| `mean_ic` | `ic` series | eval/OOS 日期 | 有效 `ic` 日期等权均值。 | `summary`, `metrics` |
| `mean_rank_ic` | `rank_ic` series | eval/OOS 日期 | 有效 `rank_ic` 日期等权均值。 | `summary`, `metrics` |
| `mean_mutual_information` | `mutual_information` series | eval/OOS 日期 | 有效 MI 日期等权均值。 | `summary`, `metrics` |
| `ic_ir` | `ic` series | eval/OOS 日期 | `mean_ic / std(ic, ddof=1)`，未年化。 | `summary`, `metrics` |
| `rank_ic_ir` | `rank_ic` series | eval/OOS 日期 | `mean(rank_ic) / std(rank_ic, ddof=1)`，未年化。 | `metrics` |
| `mutual_information_ir` | MI series | eval/OOS 日期 | `mean(MI) / std(MI, ddof=1)`，未年化。 | `summary`, `metrics` |
| `ic_t_stat`, `ic_p_value` | `ic` series | eval/OOS 日期 | `mean / (std / sqrt(n))`；p 值用 Student-t, `df=n-1`。 | `summary`, `metrics` |
| `ic_positive_rate`, `rank_ic_positive_rate`, `mutual_information_positive_rate` | per-date series | eval/OOS 日期 | 有效观测中 `> 0` 的比例。 | `summary`, `metrics` |
| `ic_valid_ratio`, `rank_ic_valid_ratio`, `mutual_information_valid_ratio` | per-date series | eval/OOS 日期 | 非 NaN 行数 / 总行数。 | `summary`, `metrics` |
| `mean_ic_ci_*`, `mean_rank_ic_ci_*` | IC/RankIC series | eval/OOS 日期 | 由 `reporting.uncertainty.compute_core_uncertainty` 计算，可为 normal 或 block bootstrap。 | `summarise_experiment_result`, `metrics` |
| `fama_macbeth_*` | factor, label | 同 `(date, asset)` | 单因子 Fama-MacBeth 系数、t 值、p 值；无 existing factors 时 spanning 字段为 None。 | `metrics` |
| `random_baseline_*` | factor, label | 每日截面内 shuffle | mean RankIC permutation null 的均值、std、分位数、p 值、z-score。 | `metrics`, `random_baseline_null` |

### 分组、多空与收益曲线

| 指标/曲线 | 输入字段 | 对齐 | 计算口径 | 输出位置 |
| --- | --- | --- | --- | --- |
| `quantile` assignment | factor `value` | 按 `date` 截面 | dense rank 后线性映射到 `1..effective_q`；`effective_q=min(n_quantiles,n_valid)`；常数因子全在 1 桶。 | `quantile_assignments_df` |
| `group_return`/`mean_return` | assignment, label `value` | 同 `(date, asset)` | 每日每桶标签均值，桶内等权。 | `quantile_returns_df`, `group_returns` |
| `long_short_return` | per-date quantile returns | 同 date | 最高已占用桶均值减最低已占用桶均值；只有一个桶则 NaN。 | `long_short_df` |
| `mean_long_short_return` | `long_short_return` | eval/OOS 日期 | 有效日期等权均值。 | `summary`, `metrics` |
| `long_short_ir` | `long_short_return` | eval/OOS 日期 | `mean / std(ddof=1)`，未年化。 | `summary`, `metrics` |
| `long_short_hit_rate` | `long_short_return` | eval/OOS 日期 | 有效日期中 `> 0` 的比例。 | `summary`, `metrics` |
| `mean_long_short_return_ci_*` | `long_short_return` | eval/OOS 日期 | 与 core uncertainty 配置一致。 | `summarise_experiment_result`, `metrics` |
| `group_monotonicity_qtop_qbottom` | group returns | eval/OOS 日期 | top 桶平均收益减 bottom 桶平均收益。 | `metrics` |
| `group_monotonicity_share` | group returns | eval/OOS 日期 | 每日分组收益是否随桶号单调的比例。 | `metrics` |
| `daily_pnl_*` | quantile returns, turnover | eval/OOS 日期 | long leg、short leg、gross、cost drag、net 的均值和极值。 | `metrics`, `daily_pnl_attribution` |
| `annualized_return`, `annualized_volatility`, `sharpe`, `sortino`, `calmar` | group returns | artifact write-time | 先取多空收益，再按 `max(rebalance_step, label_horizon)` 非重叠采样；`periods_per_year` 约为 `252 / effective_step`。 | `artifact_enrichment.py` backtest summary |
| `nav_points` | group returns | artifact write-time | 基于非重叠多空收益 `(1+r).cumprod()`。 | artifact backtest summary |

### Turnover、费用与容量

| 指标/曲线 | 输入字段 | 对齐 | 计算口径 | 输出位置 |
| --- | --- | --- | --- | --- |
| `turnover` per quantile | quantile assignments | `t` vs `t-1` | `1 - overlap(curr, prev) / n_curr`；首期 NaN。 | `quantile_turnover_df` |
| `long_short_turnover` | per-quantile turnover | 同 date | top/bottom 桶 turnover 平均；任一 leg NaN 则 NaN。 | `long_short_turnover_df`, `turnover` |
| `mean_long_short_turnover` | `long_short_turnover` | 仅 long-short 有收益的日期 | 有效 turnover 均值；首期通常被 drop。 | `summary`, `metrics` |
| `long_short_return_per_turnover` | mean L/S, mean turnover | eval/OOS 日期 | `mean_long_short_return / mean_long_short_turnover`。 | `summary`, `metrics` |
| `mean_cost_adjusted_long_short_return` | L/S return, L/S turnover | inner join `(date,factor)` | `mean(long_short_return - cost_rate * turnover)`；NaN adjusted return 不参与均值。 | `summarise_experiment_result`, `metrics` |
| `cost_aware_long_short_ir` | adjusted L/S return | inner join `(date,factor)` | adjusted return 的未年化 `mean/std(ddof=1)`；不可得时回退展示 raw `long_short_ir`。 | `metrics` |
| `capacity_*` | prices `amount`/market cap, assignments, labels | date/asset | 市值加权 L/S 均值、traded ADV、`participation_rate * traded_adv / turnover` 上限。 | `capacity_estimation`, `metrics` |

### 时序稳定性

| 指标/曲线 | 输入字段 | 对齐 | 计算口径 | 输出位置 |
| --- | --- | --- | --- | --- |
| `ic_timeseries` | IC, RankIC, MI | date outer merge | 报告用每日序列；strict split 时可附 `split_phase` 并 drop EMBARGO。 | `SingleFactorEvaluationResult.ic_timeseries` |
| `rolling_mean_ic`, `rolling_mean_rank_ic`, `rolling_mean_mutual_information`, `rolling_mean_long_short_return` | per-date series | 按 date 排序 | rolling mean，默认窗口来自 `RollingStabilityConfig.rolling_window_size`，当前默认 20。 | `rolling_stability_df` |
| `rolling_*_positive_rate` | per-date series | 同上 | rolling 窗口内 `>0` 比例。 | `rolling_stability_df` |
| `rolling_*_positive_share`, `rolling_*_min_mean` | rolling mean columns | eval/OOS 日期 | 有效 rolling 窗口中正均值比例、最差 rolling mean。 | `summary`, `metrics` |
| `subperiod_*_positive_share`, `subperiod_*_min_mean` | IC/L/S series | chrono split | 有效序列按时间切成 3 段后算分段均值。 | `summary`, `metrics` |
| `ic_decay` | factor, prices or cached labels | horizon-specific label | 对 horizons `{1,2,3,5,10,20,target}` 逐个重算 IC/RankIC summary。 | `ic_decay`, `metrics` |
| `ic_half_life_*` | `ic_decay` | horizon order | 以 abs mean IC 衰减估计 half-life；可为 estimated/not_reached/unavailable。 | `metrics` |
| `ic_decay_retention_5_over_1` | `ic_decay` | horizon 1 and 5 | `mean_ic@5 / mean_ic@1`。 | `metrics` |
| `factor_autocorrelation` | factor values | `t` vs `t-lag` | 每个 lag 的截面 rank autocorr 均值/std/n。 | `factor_autocorrelation` |
| `lag_sensitivity_*` | shifted factor, labels | lag per asset | 因子值按资产滞后 `0,1,2,3` 后重算 mean IC 和 L/S IR。 | `metrics`, `lag_sensitivity` |
| `strict_*` metrics | full-path RankIC/L/S series | strict profile only | bootstrap IR CI、前后半段、奇偶年份、post-split gap scan、regime segment。 | `metrics` |

### 条件、覆盖与风控诊断

| 指标/曲线 | 输入字段 | 对齐 | 计算口径 | 输出位置 |
| --- | --- | --- | --- | --- |
| `conditional_ic_by_magnitude` | factor, label | 同 `(date, asset)` | 每日按 `abs(factor)` 分 Q1..Q5，再按桶汇总 IC/RankIC 正率和资产数。 | result frame, `metrics` |
| `conditional_ic_by_cross_section_size` | factor, label | 同 `(date, asset)` | 按有效资产数是否低于全样本 median 分 small/large。 | result frame, `metrics` |
| `coverage` per date | prices, factor, label | date/asset | eligible、valid score、valid forward return、valid sample 的计数和比例。 | `coverage`, `metrics` |
| `n_label_nan_dates` | eval factor dates, labels | eval dates | 评估期没有任何有效 forward label 的日期数。 | `ExperimentResult` |
| `ls_max_drawdown`, `ls_var_5`, `ls_cvar_5`, `ls_calmar_ratio` | L/S returns | date order | tail risk 用 compounded NAV；overlap label 时按 `max(horizon, rebalance_frequency)` 采样。 | `summary`, `metrics` |
| `regime_*` | IC/L/S, prices | date | 市场方向/波动 regime 下的 IC 和 L/S 统计。 | `regime_df`, `regime_summary`, `metrics` |
| `tradability_*` | prices, labels | label mask | 检测 limit up/down/suspended 并 mask 不可交易标签后重算 mean IC 和 L/S。 | `metrics` |
| `next_open_*` | prices open, factor | next-open label | 用 `execution_price_mode="next_open"` 重算 mean IC、L/S return、L/S IR 和 delta。 | `metrics` |
| `data_quality_*` | prices, integrity checks | raw panel | suspended/stale/suspected split rows 和 integrity warn/fail counts。 | `metrics` |
| `neutralization_*` | raw/neutralized factor | date/asset | 暴露残差化前后对比、coverage delta、rolling delta。 | `metrics`, `neutralization_summary` |
| `baseline_*` | factor, price baseline | date/asset | 与 20D momentum、5D reversal 的 mean IC、L/S IR、factor rank corr 对比。 | `metrics` |
| `param_sensitivity_*` | factor, labels | variant `n_quantiles` | `n_quantiles` in `{3,5,10}` 的 mean IC/L/S IR min/max/std/range。 | `metrics` |
| `haircut_sharpe_*`, `dsr_pvalue` | L/S IR, n obs | eval/OOS 日期 | 多重检验调整和 Deflated Sharpe p-value；输入 Sharpe/IR 均按未年化口径。 | `metrics` |

### Scope 与判定字段

| 字段组 | 说明 |
| --- | --- |
| `*_full`, `*_is`, `*_oos` | strict split 下报告 full path，同时保留 OOS gate 口径和 IS 对照。 |
| `*_oos_decay_ratio` | `oos / is`，用于观察样本外保持率。 |
| `factor_verdict*` | 因子继续/停止判定与理由，来自 `reporting.factor_verdict`。 |
| `campaign_triage*` | campaign 层筛选/排队信号，属于批量研究辅助。 |
| `level2_promotion*` | Level 2 portfolio validation 推进判定。 |
| `stage_timings` | 性能诊断，不参与研究统计。 |

## Golden Snapshot 清单

建议 golden fixture 固定为 5 个资产 x 30 个交易日，包含 `close`、`open`、`vwap`、`amount`、`market_cap`，并包含一个 strict split contract。snapshot 存 JSON 数值和数组 hash，不存图片。

| Snapshot 项 | 存储建议 |
| --- | --- |
| `metrics` | 对核心标量保留完整 JSON；对浮点用固定精度或 `math.isclose` helper。 |
| `ic_timeseries` | 存 `date`, `ic`, `rank_ic`, `mutual_information`, optional `split_phase` 的截断数组和 hash。 |
| `ic_decay` | 存所有 horizon 行，特别是 target horizon 与 horizon 1 的一致性。 |
| `factor_autocorrelation` | 存所有 lag 行。 |
| `group_returns` | 存 `date/group/group_return` long-form 数组 hash。 |
| `turnover` | 存首期 NaN、后续 turnover 数列。 |
| `rolling_stability` | 存窗口开始前 NaN 和窗口开始后的滚动值。 |
| `coverage` | 存 per-date count/ratio，coverage 不 drop EMBARGO。 |
| `capacity_estimation` | 存一行 summary；对缺少 `amount` 或 cap 的状态也应有 fallback case。 |
| `lag_sensitivity` | 存 `lag=0..3` 的 mean IC 和 L/S IR。 |
| `random_baseline_null` | 固定 seed，存 permutation 数组 hash 和 summary scalar。 |
| `daily_pnl_attribution` | 存 `long_leg/short_leg/gross/cost_drag/net` 数列。 |

## 必须验证的合成数据清单

### P0: 先写的 5 个高风险用例

1. IC 双路径一致性  
   构造 `close[t+h]/close[t]-1` 与因子有闭式相关的面板。同一份 factor/prices 同时跑 `compute_ic` 和 `compute_ic_decay(horizons=(h,))`，要求 `mean_ic`、`mean_rank_ic`、`ic_ir`、`n_dates` 精确一致或在浮点容差内一致。

2. Turnover 首期 NaN 与费后样本数  
   构造两天以上固定分桶。断言 `quantile_turnover` 和 `long_short_turnover` 首期为 NaN；`cost_adjusted_long_short` 首期 adjusted return 为 NaN；`mean_cost_adjusted_long_short_return` 的有效样本数比 raw L/S 少 1，除非 raw 本身有 NaN。

3. `execution_price_mode` 标签对齐  
   用三到五天价格手算 `close`、`next_open`、`vwap` 标签：`close[t+h]/close[t]-1`、`close[t+h]/open[t+1]-1`、`close[t+h]/vwap[t+1]-1`。特别覆盖 `horizon=1`，确认 non-close mode 的 entry 和 exit 都落在 `t+1`，没有 off-by-one。

4. Split EMBARGO drop  
   构造 IS、EMBARGO、OOS 三段，其中 EMBARGO 期信号反向且会显著改变均值。断言核心 OOS metrics 不含 EMBARGO；打包后的 `ic_timeseries`、`rolling_stability`、`group_returns`、`turnover` drop EMBARGO；`coverage` 保留 EMBARGO 且标注 `split_phase`。

5. `n_quantiles=5` 的小截面和 ties  
   覆盖 `n_valid=1`、`n_valid=2`、`n_valid < n_quantiles`、`n_distinct=1`、ties。断言 bucket 1 包含最低值、最高已占用 bucket 包含最高值、常数因子 long-short 为 NaN、两资产时只有 bottom/top 两端有效。

### P1: 闭式合成扩展

| 用例 | 目标 |
| --- | --- |
| 白噪声因子 | mean IC 和 RankIC 接近 0，分组收益接近 0，random baseline p-value 不显著。 |
| 完美单调因子 | RankIC 为 1，top-bottom 为手算 spread。 |
| 反向单调因子 | RankIC 为 -1，top-bottom 为负。 |
| 常数因子 | IC/RankIC、L/S、IR 均为 NaN 或不可用；coverage 不应受影响。 |
| MI 非线性 | 构造 U-shape 标签，使 Pearson IC 接近 0、MI 为正。 |
| rolling/subperiod | 构造前半正、后半负的 IC 序列，验证 rolling min、positive share、subperiod min。 |
| tail risk | 手写 L/S return 序列，闭式验证 max drawdown、duration、VaR、CVaR、Calmar。 |
| factor autocorr | 因子逐日平移或完全不变，验证 autocorr 接近 1；白噪声接近 0。 |
| capacity | 给定 `amount` 和 turnover，验证 `estimated_capacity_upper_bound = mean_traded_adv * participation_rate / turnover`。 |
| preprocessing | 验证 winsorize 分位点、z-score 使用 `ddof=0`、小样本 z-score/rank 置 NaN、winsor 小样本跳过。 |

### 外部对照

外部对照只作为一次性审计，不进默认 CI。推荐选择 `mom_5d` 或 `reversal_5d`：

| 对照项 | 对照实现 |
| --- | --- |
| IC/RankIC | 用一个极简 pandas 实现按 `(date, asset)` merge，再 groupby date 做 Pearson/Spearman。 |
| 分组收益和 L/S | 独立实现 qcut/rank 分桶，并明确记录与本项目 dense-rank linear-map 分桶差异。 |
| 年化 backtest summary | 独立从 `group_returns` 重建 L/S 序列，按 `max(rebalance_step, horizon)` 非重叠采样后年化。 |
| 可选 alphalens 对照 | 只比较方向一致的核心项；若分桶语义不同，记录差异而非强行 assert。 |

## Reference Card 要求

选一个真实因子，建议 `reversal` 或 `asym_vol_reversal`，跑完整单因子报告。卡片正文中文优先，且每个关键指标必须能追溯到本 inventory 的一行：

| 卡片项目 | 要写清楚的内容 |
| --- | --- |
| 标签口径 | horizon、`execution_price_mode`、是否 OOS、是否 drop EMBARGO。 |
| 预处理 | winsorize bounds、standardization、min coverage、neutralization 是否启用。 |
| IC/RankIC | 日期数、有效率、IR 是否未年化、CI 方法。 |
| 分组收益 | 分桶语义、top/bottom 定义、多空方向。 |
| 费用 | `cost_rate`、首期 turnover NaN、费后样本数。 |
| 年化统计 | 是否来自 artifact enrichment，effective step 和 periods/year。 |
| 风险/稳定性 | rolling window、subperiod、tail risk、IC decay horizons、autocorr lags。 |
| 结论 | `factor_verdict`、Level 2 promotion 信号和限制条件。 |

## 当前风险登记

| 风险 | 状态 | 下一步 |
| --- | --- | --- |
| IC/RankIC 双路径漂移 | 高风险 | P0 合成测试 1。 |
| turnover 首期 NaN 影响费后均值 | 高风险 | P0 合成测试 2，并在 reference card 明示样本数。 |
| `next_open`/`vwap` label off-by-one | 高风险 | P0 合成测试 3。 |
| split embargo 混入报告曲线 | 高风险 | P0 合成测试 4。 |
| 小截面分桶边界 | 高风险 | P0 合成测试 5。 |
| 单因子预处理口径被忽略 | 已确认存在 | golden fixture 同时保存 raw/preprocessed 关键值。 |
| 年化 Sharpe 与未年化 IR 混用 | 中风险 | Reference card 必须分别标注来源。 |
| `marginal_contribution`/`haircut_sharpe` 是否落 artifact | 已确认写入 metrics | artifact compact payload 仍需 golden 覆盖。 |
