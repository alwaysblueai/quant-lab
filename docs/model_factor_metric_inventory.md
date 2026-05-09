# 模型因子指标 Inventory 与验证路线

本文档是 model-factor Level 1/2 评估指标的对账清单。新增、删除或改口径时，先更新这里，再更新测试和 golden snapshot。

边界声明：本文只覆盖 **model-factor 独有产物**——训练健康、特征贡献、时序交叉验证、模型选择。与单因子共享的 IC/RankIC/分组收益/turnover/coverage/decay 等核心评估指标按 [single_factor_metric_inventory.md](single_factor_metric_inventory.md) 登记，不在这里复述。下游回测口径（split semantics、年化 Sharpe vs 未年化 IR、`split_phase` drop/保留策略）也跟单因子一致，model-factor 复用同一份 `evaluate_single_factor_case` 与 `artifact_enrichment` 链路。

## 总口径

| 项目 | 当前口径 |
| --- | --- |
| 输入特征 | wide 表，列为 `date`、`asset` 加 N 个特征列。`feature_availability.mode` 决定可用时点：`required_timestamp` 严格按 `known_at`，`safety_lag` 按特征日 + lag 天。 |
| 标签对齐 | `forward_return(prices, horizon=spec.target.horizon)`，与单因子一致。 |
| 训练样本窗口 | `training.training_window_months` 控制样本回溯；`training.retrain_every_n_dates` 控制重训节奏；`exploratory_screening` profile 可通过 `--screening-retrain-every-n-dates` 临时拉宽。 |
| OOS 边界 | 与单因子相同的 `TimeSeriesSplitContract`：`is_end` 之后 + embargo + `oos_start`，metrics 默认在 OOS 窗口汇总，`metric_scope=oos`。 |
| 模型族 | `ridge`/`lasso`/`elastic_net`/`linear`/`mlp`/`gbdt`/`lightgbm`/`xgboost`，由 `spec.model.family` 选定；展示侧 `metrics.json["model_family"]` 必须与 `model_selection.json` 中实际训练的族一致（详见下文一致性 invariant）。 |
| Purge gap | Purged k-fold diagnostics 的 `purge_days = label_horizon`，由 `spec.target.horizon` 注入；不可独立配置，避免 `purge_gap < label_horizon` 导致前视。 |
| Cache 路径 | CLI 默认 `output_dir.parent / _model_factor_cache`，跨 case 共享；web 路径由 `--cache-root-dir` 显式注入 `<base_root>/_model_factor_shared_cache`，跨 web run 共享。 |

## 产物地图

| 产物文件 | 主要字段/列 | 生产位置 | 口径 |
| --- | --- | --- | --- |
| `training_log.csv` | `score_date`, `status`, `skip_reason`, `model_version`, `trained_date_start`, `trained_date_end`, `n_train_dates`, `n_train_rows`, `n_score_assets`, `model_family`, `scale_mode`, `selection_status`, `selection_metric`, `selected_candidate_id`, `selected_candidate_score`, `selected_candidate_turnover` | `model_factor/core.py` 训练循环 | 每个 score_date 一行；`status ∈ {trained, skipped, reused}`，`skipped` 行 `skip_reason` 必有值，`trained_date_*` 与 `model_version` 为 `N/A`；条件不适用字段统一写 `N/A` 或 `not_skipped`。 |
| `training_metrics.csv` | `model_version`, `model_family`, `train_start`, `train_end`, `oos_start`, `oos_end`, `train_ic`, `train_rank_ic`, `train_loss`, `oos_ic`, `oos_rank_ic`, `oos_loss`, `n_train_obs`, `n_train_dates`, `n_oos_obs`, `n_oos_dates`, `selected_candidate_id`, `selected_candidate_score` | `model_factor/core.py` per-fit 指标聚合 | 每个 `model_version` 一行；IC/RankIC 是该 model_version 在 train/OOS 窗口内的有效日期等权均值，**未年化**。`*_loss` 是 family-specific 训练目标值（ridge/lasso 是 MSE，gbdt 是 boosting loss）。 |
| `feature_importance.csv` | `feature`, `mean_abs_importance`, `mean_signed_importance`, `n_model_versions`, `latest_importance`, `latest_abs_importance`, `positive_version_count`, `negative_version_count`, `zero_version_count`, `importance_source`, `sign_stability`, `missing_value_reason` | `model_factor/diagnostics.py` 聚合 ledger | 跨所有训练 model_version 聚合的特征汇总；`importance_source ∈ {coefficient, gain, permutation}`；`sign_stability = max(positive_version_count, negative_version_count) / n_model_versions`。 |
| `feature_importance_ledger.csv` | `run_id`, `case`, `factor`, `model_family`, `model_version`, `fit_date`, `feature`, `signed_importance`, `abs_importance`, `normalized_share`, `rank`, `importance_source`, `permutation_sampled`, `permutation_sample_rows`, `permutation_n_repeats`, `permutation_guardrail_reason` | `model_factor/diagnostics.py` 每次 fit 写一组 | 每次训练每个特征一行；`normalized_share = abs_importance / sum(abs_importance over features)`，同 model_version 内归一；`rank` 1-based。 |
| `feature_oos_ic.csv` | `feature`, `window_start`, `window_end`, `model_version`, `ic`, `rank_ic`, `n_obs`, `n_dates` | `model_factor/diagnostics.py` 单特征 OOS IC | 每个 (feature, model_version) 一行；IC 是把单个特征当作 score、与该 model_version 对应 OOS 窗口内 forward return 的 Pearson/Spearman；用于诊断"模型贡献来自哪个特征"。 |
| `purged_kfold_summary.json` | `status`, `n_eval_dates`, `n_splits_requested`, `n_splits_used`, `label_horizon`, `embargo_pct`, `embargo_days`, `purge_days`, `n_folds`, `fold_metrics_available`, `mean_ic`, `mean_rank_ic`, `mean_sharpe`, `ic_positive_folds`, `rank_ic_positive_folds`, `verdict`, `reasons` | `reporting/purged_kfold_diagnostics.py::build_purged_kfold_diagnostics` | `purge_days = label_horizon`（强绑定，见上文）；`mean_sharpe` 是 fold-level 多空收益的 mean/std(ddof=1)，**未年化**；`verdict ∈ {ok, weak, fragile, not_available}` 由 reasons 数组解释。 |
| `purged_kfold_folds.csv` | `fold_id`, `train_start`, `train_end`, `test_start`, `test_end`, `n_train_dates`, `n_test_dates`, `mean_ic`, `mean_rank_ic`, `long_short_sharpe`, `mean_long_short_return` | 同上 | 每个 fold 一行；`long_short_sharpe` 同样未年化；`n_train_dates` 已扣除 purge/embargo。 |
| `purged_kfold_fold_daily.csv` | `fold`, `date`, `ic`, `rank_ic` | 同上 | 每个 (fold, date) 一行；`ic`/`rank_ic` 是 test 窗口内逐日值，方便绘制 fold-level 时间序列。 |
| `model_selection.json` | `status`, `configured_model`, `configured_model_selection`, `selection_rows`, `summary.latest_selected_candidate_id` | `model_factor/artifacts.py::_build_model_selection_payload` | `status ∈ {disabled, no_candidates, ok}`；`disabled`/`no_candidates` 时 `selection_rows = []`，配置族即训练族；`ok` 时 `selection_rows` 每行 `(score_date, candidate_id, family, params, metric_score, turnover, ...)`，`summary.latest_selected_candidate_id` 指向最近一次入选的 candidate。 |
| `model_definition.json` | `diagnostics.*`、`feature_importance.*` 等元数据 | `model_factor/artifacts.py` | 训练侧元信息聚合（特征列、importance 配置、score 资产数、winsorize 参数、artifact_missing_value_notes）。 |
| `feature_manifest.json` | `feature_columns`, `features[].{feature, mean, std, non_null_ratio}`, `feature_availability.*`, `n_rows`, `n_dates`, `n_assets` | `model_factor/artifacts.py` | 训练前的特征面板统计；`feature_availability` 记录 known_at/safety_lag 应用后的 dropped/shifted 行数。 |

## Indicator Inventory

### 训练健康

| 指标/曲线 | 输入字段 | 对齐 | 计算口径 | 输出位置 |
| --- | --- | --- | --- | --- |
| `n_train_rows` / `n_train_dates` | feature panel | `score_date` | 训练样本数 / 训练日数；`skipped` 行为 N/A | `training_log` |
| `train_ic` / `train_rank_ic` | model_version 内训练样本 | `(date, asset)` | 训练窗口内截面 IC 等权均值 | `training_metrics` |
| `oos_ic` / `oos_rank_ic` | model_version 对应 OOS 窗口 | `(date, asset)` | OOS 窗口内截面 IC 等权均值；用于跨 model_version 训练健康监测，**与单因子 metrics 的 `mean_ic`/`mean_rank_ic` 不是同一汇总粒度** | `training_metrics` |
| `train_loss` / `oos_loss` | family-specific | model_version | 训练目标函数值（ridge/lasso/elastic_net = MSE，gbdt/lightgbm/xgboost = boosting loss，mlp = 训练 loop 末尾 loss） | `training_metrics` |
| `selected_candidate_score` | selection candidate | `score_date` | 配置中 `selection_metric` 在该 candidate 的 CV 得分 | `training_log`, `training_metrics` |

### 特征贡献

| 指标/曲线 | 输入字段 | 对齐 | 计算口径 | 输出位置 |
| --- | --- | --- | --- | --- |
| `signed_importance` / `abs_importance` | per-fit | `(model_version, feature)` | family-specific：`coefficient`（ridge/lasso/elastic_net/linear），`gain`（gbdt/lightgbm/xgboost），`permutation`（任意族手动开启） | `feature_importance_ledger` |
| `mean_abs_importance` | ledger | `feature` | 跨所有 model_version 的 `abs_importance` 等权均值 | `feature_importance` |
| `sign_stability` | ledger | `feature` | `max(positive_version_count, negative_version_count) / n_model_versions`，越接近 1 越稳定 | `feature_importance` |
| `feature_oos_ic` | 单特征 score, label | `(date, asset)` | 把单特征当作 score 算 OOS 窗口内 IC，反映特征独立预测力，区别于 importance（系数大小） | `feature_oos_ic` |

### 时序交叉验证

| 指标/曲线 | 输入字段 | 对齐 | 计算口径 | 输出位置 |
| --- | --- | --- | --- | --- |
| `mean_ic` / `mean_rank_ic` | fold test | `(date, asset)` | 每个 fold test 窗口内 IC 等权均值，再跨 fold 等权均值 | `purged_kfold_summary`, `purged_kfold_folds` |
| `long_short_sharpe` | fold test | `date` | 每个 fold test 窗口内多空收益序列的 `mean/std(ddof=1)`，**未年化** | `purged_kfold_folds` |
| `purge_days` | n/a | n/a | `= label_horizon`，由 `spec.target.horizon` 注入；测试通过 `tests/test_artifact_golden_regression.py::test_model_factor_level12_core_artifacts_match_golden` 锁定 | `purged_kfold_summary` |
| `embargo_days` | n/a | n/a | `ceil(n_eval_dates * embargo_pct)`，默认 `embargo_pct=0.01` | `purged_kfold_summary` |
| `verdict` | fold metrics | n/a | 由 fold 数、IC 正性比、覆盖率决定；`reasons` 数组给出降级原因 | `purged_kfold_summary` |

### 模型选择

| 指标/曲线 | 输入字段 | 对齐 | 计算口径 | 输出位置 |
| --- | --- | --- | --- | --- |
| `selection_metric` | candidate CV scores | `score_date` | 配置项；通常 `rank_ic`/`ic`/`long_short_sharpe`，由 `spec.model.selection.metric` 决定 | `model_selection` |
| `turnover_penalty_lambda` | candidate turnover | n/a | 选择阶段用 `metric_score - λ * turnover` 作为综合分；`λ=0` 等同纯 metric | `model_selection.configured_model_selection` |
| `latest_selected_candidate_id` | selection_rows | n/a | 最近一次入选 candidate；`metrics.json["model_family"]` 必须与该 candidate 的 family 一致（disabled/no_candidates 时与 `configured_model.family` 一致） | `model_selection.summary` |

## 一致性 invariant

下列跨文件不变量由 `tests/test_artifact_golden_regression.py::test_model_factor_level12_core_artifacts_match_golden` 锁定：

- `purged_kfold_summary.json["label_horizon"] == metrics.json["target_horizon"]`
- `purged_kfold_summary.json["purge_days"] == metrics.json["target_horizon"]`
- `model_selection.json["status"] in {"disabled","no_candidates"}` ⇒ `configured_model.family == metrics.json["model_family"]`
- `model_selection.json["status"] == "ok"` ⇒ `selection_rows` 中 `latest_selected_candidate_id` 对应行的 family == `metrics.json["model_family"]`

## 与单因子 inventory 的关系

| 共享产物 | 来源 | 在 model-factor 下的差异 |
| --- | --- | --- |
| `ic_timeseries.csv` | 单因子 inventory | model-factor 把 ML 预测当作 score factor 喂进同一 evaluation；分箱、turnover、coverage 都按单因子口径计算 |
| `group_returns.csv` / `turnover.csv` / `coverage.csv` | 同上 | 同上，含 `split_phase` 列 |
| `ic_decay.csv` / `rolling_stability.csv` | 同上 | 同上 |
| `backtest_result.json` (`nav_points`、`annualized_return`、`sharpe`) | `artifact_enrichment.py` | 与单因子用同一 `build_backtest_summary_payload`；NAV 用非重叠采样、Sharpe 走年化口径 |
| `level2_portfolio_validation/*` | `level2_portfolio_validation` 共享模块 | 与单因子完全一致 |

## 修改流程

1. 改口径前先确认本文件的当前条目，在 PR 描述里点名"哪个 inventory 行被改了"。
2. 改完代码后更新本文件对应行。
3. 同步更新 `tests/test_artifact_golden_regression.py::test_model_factor_level12_core_artifacts_match_golden` 的断言或 golden hash（`tests/goldens/artifact_regression/model_factor/`）。
4. 如果是跨文件不变量（如 `purge_days == label_horizon`），同时在"一致性 invariant"小节里登记。

## 当前未覆盖项

- `feature_oos_ic` 与 `feature_importance` 的相关性诊断（高 importance 但低 OOS IC 的特征是否被报告）尚未做静态对账。
- `training_metrics.train_loss`/`oos_loss` 在不同 family 下的可比性未文档化（不同损失函数无法跨族比较）。
- web overview 的 training health chart 与本文件的 `oos_ic`/`train_ic` 对账尚未做端到端 invariant 测试，目前只有单元级 HTML 不变量测试。
