# 模型因子指标与曲线绘制审查报告

本文基于 [model_factor_metric_inventory.md](model_factor_metric_inventory.md) 与 [single_factor_metric_curve_review.md](single_factor_metric_curve_review.md) 对 model-factor 评估链路做静态审查，覆盖"训练循环 / 指标计算 / 诊断产出 / artifact 写出 / web_model_lab 展示"。审查目标与单因子审查口径一致：确认对齐、时间无 leakage、口径一致、空值/缺失语义保持，并识别真实 bug 与口径漂移。

边界声明：单因子共享路径（`evaluation.py` / `quantile.py` / `decay.py` / `turnover.py` / `costs.py` / `artifact_enrichment.py`）已在单因子审查中通过，本轮只审查 model-factor 独有部分：
- `src/alpha_lab/model_factor/{core,diagnostics,dataset_cache}.py`
- `src/alpha_lab/real_cases/model_factor/{spec,templates,cli,benchmark,pipeline,artifacts}.py`
- `src/alpha_lab/web_model_lab.html` 中的 model-lab 独有展示（training health / feature importance / feature OOS IC / purged k-fold / model selection）

## 总体结论

| 结论项 | 状态 | 说明 |
| --- | --- | --- |
| 训练循环 / IC 口径 | 通过 | `core.py` per-date cross-sectional transform、winsorize、IC/RankIC 都按 date 分组，PIT-safe；purged_kfold_split 内部已含 purge + embargo。 |
| 数据集缓存 | 口径需说明 | `dataset_cache.py` cache_key 已覆盖核心字段，但缺独立 label cache key、写入非原子，存在小概率竞态/部分写。 |
| Pipeline 编排 | 通过 | `pipeline.py` 把 score factor 喂给共享 `evaluate_single_factor_case`，split contract 透传；端到端 (date, asset) 唯一性靠下游 `drop_duplicates` 兜底，建议上游做 invariant 校验。 |
| Artifact 契约 | 部分修复待落 | 列名/状态枚举与 inventory + 现有不变量测试存在不匹配；现行 golden 因 `status=disabled` 没暴露，需要 enabled selection 的 fixture。 |
| Spec / CLI | 口径需说明 | `selection.metric` 允许集与 inventory 文档不一致；`--screening-retrain-every-n-dates` 缺 profile 守卫。 |
| Web UI 指标解析 | 通过 | `fmtNum`/`toNum`/`irOfPoints` 与单因子审查后保持一致；模型选择 / k-fold verdict 颜色映射缺失（功能缺口非 bug）。 |

## 审查矩阵

| 类别 | 状态 | 计算口径核查 | Artifact / 显示核查 |
| --- | --- | --- | --- |
| Per-date cross-sectional transform | 通过 | `_apply_cross_sectional_transform` (`core.py:1802-1835`) 显式 `groupby("date")`，注释 line 1810-1811 标记 PIT-safe。 | feature panel 写出后保持 `(date, asset)` 唯一。 |
| Per-date label winsorize | 通过 | `_winsorize_labels_per_date` (`core.py:1907-1925`) 按 date 分组算 mean/std。 | `label_winsor_clipped_rows` 写入 `label_temporal_contract`。 |
| Train / OOS IC | 通过 | `_score_prediction_cross_sections` (`core.py:2618-2640`) 先 per-date Pearson/Spearman 再跨日均值，未年化；与 inventory 一致。 | `training_metrics.csv` 每个 model_version 一行。 |
| Purged k-fold | 通过 | `purged_kfold_split` (`validation/purged_kfold.py:9-67`) 内部已扣 `[test_start - label_horizon, test_end + embargo]` (line 56-63)，core.py CV 仅按 `train_dates[masks["train"]]` 取行，无额外 leakage 风险。`purge_days = label_horizon` 由 `artifacts.py:546` 注入。 | `purged_kfold_summary.json` 含 `label_horizon`/`purge_days`；不变量测试 `tests/test_artifact_golden_regression.py:306-307` 已锁定。 |
| Sign stability | 口径需说明 | `core.py:3646-3650` 用 `signed_available_count` (signed importance 非 NaN 的版本数) 作为分母；inventory line 25 / 53 写的是 `n_model_versions`。 | 树模型族 `signed=NaN` → sign_stability=NaN（else 分支），coefficient 族两者相等；混合族出现差异。 |
| Feature OOS IC | 口径需说明 | `core.py` `_feature_oos_ic_rows` 不在函数内校验输入是否仅 OOS 段——依赖 caller 切片。 | `feature_oos_ic.csv` 每 (feature, model_version) 一行。 |
| Cache 完整性 | 口径需说明 | `dataset_cache.py:425-518` 顺序写 7 个文件，无 atomic rename；loader 只查文件存在，不校验文件大小/内容；mmap 加载 truncated `.npy` 不会立即失败。 | 罕见的 kill-mid-write 后下次启动可能命中破损 cache。 |
| Cache key 完整性 | 口径需说明 | 核心字段已覆盖（features path/sig、universe、target、feature_availability、feature_preprocess）；缺独立 `forward_label` 键，labels 与 features 共用同一 cache_key。 | 单文件 features 改 + label spec 不变时正常；label 来源（如 `target.price_column`）已在 key 中，普通使用无影响。 |
| Score factor 唯一性 | 通过（依赖兜底） | `pipeline.py:1140` `_coverage_by_date` 用 `drop_duplicates(["date","asset"])` 兜底；上游 model_factor concat 没有显式 invariant 检查。 | 风险只在 retrain windows 重叠时出现，当前 walk-forward 调度上不会。 |
| Model selection 列名 | 不通过 | `_build_model_selection_payload` (`artifacts.py:677-704`) 用 `model_selection_df.to_dict(orient="records")`，列名是 `candidate_id`/`candidate_family`（`core.py:2534-2535`）；不变量测试 `test_artifact_golden_regression.py:327, 333` 读 `selected_candidate_id` / `selected_candidate_family`。 | 现行 golden run 都是 `status="disabled"` (n_rows=0)，`else` 分支没被覆盖。 |
| Model selection 状态枚举 | 不通过 | `artifacts.py:689` 在 enabled 但空表时写 `status="not_available"`，inventory line 31 与测试 line 317 都假设 `{disabled, no_candidates, ok}`，从未生成 `no_candidates`。 | 现行 golden 不会触发；启用 selection 又无候选时 invariant 测试会 KeyError。 |
| Empty CSV fallback | 口径需说明 | `_write_csv` (`artifacts.py:995-1001`) 在写出文件 size=0 时回退 `status,reason\nnot_available,empty_dataframe\n`；这覆盖 `pd.DataFrame()`（无列无行）边角；有列空表正常写 header。 | 0-fit 极端 run 会让 `feature_importance.csv` 等被改写为不规范 header，下游 reader 会失败。 |
| `selection.metric` 可选集 | 口径需说明 | inventory line 70 列出 `{ic, rank_ic, long_short_sharpe}`；`core.ModelSelectionSpec` 只允许 `{ic, rank_ic, ic_minus_turnover_penalty, rank_ic_minus_turnover_penalty}`。 | `long_short_sharpe` 文档存在但代码不接受。 |
| `--screening-retrain-every-n-dates` profile 守卫 | 口径需说明 | `cli.py` 在 `run`/`benchmark` 都暴露此 flag，无 profile 检查；可对非 `exploratory_screening` profile 静默生效。 | inventory line 13 措辞"can through ... 临时 override"，应该限定在 screening profile。 |
| Web 指标解析 | 通过 | `fmtNum`/`toNum` 保留 `""`/`NaN`/`N/A`；`irOfPoints` `std(ddof=1)`；line series 按 x 排序、NaN 不连接（保留单因子审查结论）。 | model-lab 独有 chart card 已通过 `strong_skipped_extreme_nav` fixture smoke。 |
| Web — model selection / k-fold verdict 视觉 | 缺口 | UI 当前只渲染 `candidate_families` 元数据，未高亮 `metrics.model_family` 对应的 selected family；`renderPurgedKfoldSummary` 未按 verdict 着色。 | 功能缺口，不属计算 bug；需要时按 inventory 补 verdict palette。 |

## 关键发现

### 证据索引

| 链路 | 代码位置 | 本轮核查结论 |
| --- | --- | --- |
| Cross-sectional transform PIT-safety | `model_factor/core.py:1802-1835` | 文档化为 per-date；`groupby("date")` 实施，正确。 |
| Label per-date winsorize | `model_factor/core.py:1907-1925` | 按 date 分组算 mu/sd，逐日 clip，正确。 |
| Walk-forward purge | `validation/purged_kfold.py:55-63` | overlap mask + embargo mask 都已落实；core.py CV 直接复用日期掩码无需再过滤。 |
| Sign stability 分母 | `model_factor/core.py:3638-3650` | 用 `signed_available_count`，inventory 写 `n_model_versions`；coefficient-only / gain-only 两端等价，混合族不一致。 |
| Score factor 唯一性兜底 | `real_cases/model_factor/pipeline.py:1140` | `drop_duplicates` 在 coverage 处兜底；上游建议加 invariant 校验。 |
| Model selection 列名 | `real_cases/model_factor/artifacts.py:677-704` + `core.py:2530-2548` | df 列是 `candidate_id`/`candidate_family`；test 期望 `selected_*` 前缀；golden status=disabled 掩盖问题。 |
| Model selection status | `real_cases/model_factor/artifacts.py:688-693` | 缺 `no_candidates` 分支，enabled-but-empty 写 `not_available`，与 inventory + test 不一致。 |
| Empty CSV fallback | `real_cases/model_factor/artifacts.py:995-1001` | 极端空 frame 写出非合法 schema 的 header。 |
| Selection metric 可选集 | `real_cases/model_factor/spec.py:_parse_model_selection_spec` + `model_factor/core.ModelSelectionSpec` | 接受 `{ic, rank_ic, *_minus_turnover_penalty}`，inventory 写 `long_short_sharpe`，文档/代码二选一对齐。 |
| `--screening-retrain-every-n-dates` | `real_cases/model_factor/cli.py` | flag 在所有 subcommand 共享，无 profile 守卫。 |
| Cache atomic write | `model_factor/dataset_cache.py:425-518` | 7 文件顺序写、无 fsync rename。 |
| Cache key 字段 | `real_cases/model_factor/pipeline.py:854-871` | 已含 features/prices/universe/target/feature_availability/feature_preprocess；label 与 feature 共用同 key。 |
| Web 解析口径 | `web_model_lab.html` `toNum`/`fmtNum`/`irOfPoints` | 与单因子审查后状态一致，未发现回归。 |

### 通过

- **PIT 安全**：`_apply_cross_sectional_transform`（line 1802-1835）和 `_winsorize_labels_per_date`（line 1907-1925）都是严格 per-date，函数文档明确这一点。审查用 agent 误读为"全样本 fit 后 leak"，实地核对后排除。
- **Purged CV 已含 purge + embargo**：`validation/purged_kfold.py:55-63` 同时屏蔽 train-overlap 与 embargo；`core.py:2587-2588` 按日期 `isin` 取行无需再加 purge——审查用 agent 标的 P0 leakage 是误报。
- **整数完整性检查**：`_check_feature_known_at_not_after_signal_date`、`check_no_future_dates_in_input` 在训练前对 prices/features 做未来日断言（`core.py:753-778`）。
- **NAV 年化与未年化区分**：复用单因子 `build_backtest_summary_payload`，model-factor `nav_points`/`annualized_*`/`sharpe` 与 IR 分离，未发现 model-factor 单独引入 misnamed Sharpe。
- **不变量测试已落地**：`purged_kfold_summary.label_horizon == metrics.target_horizon`、`purge_days == target_horizon`、`model_family` 一致性（disabled 分支）已被 `test_artifact_golden_regression.py:302-334` 锁定。
- **Web UI 数字解析口径**：`toNum` 把空字符串 / `"N/A"` / `"nan"` / `"none"` 全部映射为 `null`，不再被 `Number("")` 误算成 0；`irOfPoints` 用 `(n-1)` 样本方差与后端 `std(ddof=1)` 一致；line chart 按 x 排序并在非有限点断开。

### 口径需说明

- `sign_stability` 分母用 `signed_available_count` 而非 `n_model_versions`：树模型族 signed=NaN 时整列退到 NaN，coefficient 族两者相等；混合族（同时跑 coefficient + gain）会出现 stability 偏高。如果坚持 inventory 口径，把分母换成 `n_model_versions` 即可；如果坚持代码当前行为，更新 inventory line 25 / 53 措辞。
- `model_selection.status` 没有 `no_candidates`：enabled 但 0 候选 → 当前写 `not_available`，invariant 测试 disabled 分支不接受 `not_available`，会落到 ok 分支然后 KeyError。需要加 `no_candidates` 分支或更新测试枚举。
- `selection_rows` 列名 `candidate_id`/`candidate_family`：与不变量测试期望的 `selected_candidate_id`/`selected_candidate_family` 不一致。要么改 dataframe schema，要么把测试改为读 `candidate_id`，二选一。
- `selection.metric` 文档/代码漂移：inventory line 70 提 `long_short_sharpe`，代码不接受；其它 turnover penalty 变体代码接受但文档没列。
- `--screening-retrain-every-n-dates` 在非 screening profile 也生效：要么加 `profile == "exploratory_screening"` 守卫，要么在 inventory 里去掉 "临时" 措辞、明确这是全 profile flag。
- Cache 缺独立 label key：当前 `target.price_column` / `target.horizon` / `target.winsorize_zscore` 都进 key，常规修改会触发 invalidation；只在 label 计算逻辑变化（如换成动态 vol-target label）时风险显著，需要单独的 label_signature 字段。
- Cache atomic write：罕见 kill-mid-write 可能让下次 run 命中半成品；建议把 metadata.json 留到所有 numpy/parquet 文件 fsync 后再写（loader 已读 metadata.json 校验 cache_key，写最后即成 sentinel）。
- Empty CSV fallback：`_write_csv` 用 `status,reason` header 兜底极端空 frame，会让 `feature_importance.csv` 等违反 schema；建议判断 `frame.shape[1] == 0` 时跳过文件，或写出与 manifest 一致的空 header。
- Score factor `(date, asset)` 唯一性靠 `_coverage_by_date.drop_duplicates`：当前 walk-forward 不会产生跨 model_version 重叠 score，但缺少显式 invariant；建议在 `pipeline.py` 把 `factor_df.duplicated(["date","asset"])` 加成 hard assert，让上游 retrain 重叠立即报错。

### 已闭环

以下是单因子审查链路（已 propagate 到 model-factor）+ 本轮核对继续有效的口径：

- model-factor 共享 `evaluate_single_factor_case`，因此 `(date, asset)` 对齐、`split_phase`、`EMBARGO` drop/保留、未年化 IR vs 年化 Sharpe 分离、非重叠 NAV 年化统计与单因子完全一致。
- `web_model_lab.html` `fmtNum`/`toNum`/`irOfPoints` 已和后端 `std(ddof=1)` 对齐；空 CSV 单元不画为 0。
- `purge_days == label_horizon` 由 `artifacts.py:546` 强绑定，invariant 测试已锁定。
- `model_factor_metric_inventory.md` 已建立产物地图与一致性 invariant；本轮新发现的 schema 漂移正是因为 inventory 维护严格、可被反查到。

## 缺口清单

| 优先级 | 缺口 | 影响面 | 复现 / 触发 | 建议位置 |
| --- | --- | --- | --- | --- |
| P1 | `selection_rows` 列名与不变量测试不一致 | enabled selection 真实 run 会让 invariant assert 失败 | 用 `model_selection.enabled=true` + 多个 candidates 的 spec 跑 golden | `artifacts.py:_build_model_selection_payload` 或测试 |
| P1 | `model_selection.status` 缺 `no_candidates` 分支 | enabled-but-zero-candidates 会触发 invariant test KeyError | 用 enabled selection + 候选都被过滤的 spec | `artifacts.py:688-693` |
| P1 | enabled selection 路径没有 golden 覆盖 | 当前 model-factor golden 全是 `status=disabled`，上面两类 schema bug 没被自动捕获 | 加一个最小 enabled selection fixture | `tests/goldens/artifact_regression/model_factor/` |
| P2 | `sign_stability` 分母与 inventory 不一致 | coefficient 与 gain 族单跑等价；混合族（experiment ensembling）显示偏差 | 同 case 同 spec 跑两族（罕见）| `core.py:3646-3650` |
| P2 | `selection.metric` 文档/代码漂移 | 配置 `long_short_sharpe` 会被静默拒绝 | YAML 写 `selection.metric: long_short_sharpe` | inventory line 70 与 `core.ModelSelectionSpec` |
| P2 | `--screening-retrain-every-n-dates` 缺 profile 守卫 | 非 screening profile 被静默 override retrain 节奏 | `--evaluation-profile default_research --screening-retrain-every-n-dates 60` | `cli.py` argparse 后或 `pipeline.py` |
| P2 | dataset_cache 写入非原子 | kill-mid-write 后命中半成品 | 训练中断 + 立即重启 | `dataset_cache.py:425-518` 改 metadata.json 最后写 |
| P2 | empty CSV fallback 写 status,reason header | 极端 0-fit run 让 `feature_importance.csv` 等违反 schema | 全部 fit 都被 skip 的 case | `artifacts.py:995-1001` |
| P3 | score factor 缺上游唯一性 invariant | 当前不会触发；将来 retrain windows 重叠会被 silently dedup | 让两个 model_version 同时落同一 (date, asset) | `pipeline.py` 在 build factor_df 后加 `assert not duplicated` |
| P3 | model-lab UI 缺 verdict / selected-family 视觉 | 解读 model selection 与 k-fold verdict 需要切到 CSV | 真实 run 打开 web | `web_model_lab.html` k-fold / model selection card |
| P3 | feature OOS IC 函数不强制 OOS 切片 | caller 错误传全 panel → IC 含 train 段 | 改 caller 传错的 panel | `model_factor/diagnostics.py` 或 `core.py` 加 assert |

## 误报记录（agent 提报后核对排除）

为保留审查痕迹，下面记录本轮 agent 提报但实际为误报的发现：

- **`_apply_cross_sectional_transform` 在全样本 fit 导致 leakage**：`core.py:1810-1811` 与 line 1833 `groupby("date")` 表明严格 per-date，每个 cross-section 独立计算 mean/std，PIT-safe。
- **`_winsorize_labels_per_date` 在全样本 fit 导致 leakage**：函数名即说明 per-date；`core.py:1914` 按 `date` 分组算 mu/sd。
- **CV fold 缺 purge 导致 leakage（core.py:2587-2588）**：`purged_kfold_split` 内部已扣 `[test_start - label_horizon, test_end + embargo]`（`validation/purged_kfold.py:55-63`），core.py 用 `train_dates[masks["train"]]` 取行无需再过滤。
- **Web `top3Share` / `top5Share` fallback 链导致 top5 < top3**：`Math.min(idx, length-1)` 在 length<3/5 时回退到最后可用 cumulative_share，由 cumulative 单调性保证 top5 >= top3。

## 本轮验证记录

- 本轮**未运行**任何测试（`make test` / pytest），仅静态审查；上面 P1/P2 都需要后续真实 run + golden 验证。
- 静态对照过的不变量测试：`tests/test_artifact_golden_regression.py:302-334`。
- 未做：enabled selection golden fixture、`no_candidates` 分支端到端 case、UI 视觉 QA、cache atomic write 故障注入。

## 建议测试 / 修复实施顺序

1. **P1 修复（先行）**：让 `_build_model_selection_payload` 在 enabled-but-empty 时输出 `status="no_candidates"`；`selection_rows` 列名加 `selected_` 前缀（或保留 `candidate_id` 同时把测试也对齐成同名）。任选其一即可，关键是 schema 与不变量测试匹配。
2. **P1 covering golden**：加一个 `model_selection.enabled=true` 且至少 2 个 candidate 的 golden case，让 `else` 分支被持续跑过；同时加一个所有 candidate 都被过滤的 minimal case 锁 `no_candidates`。
3. **P2 文档/守卫**：对齐 `selection.metric` 文档与代码、给 `--screening-retrain-every-n-dates` 加 profile 检查、修 `sign_stability` 分母（任一选项）。
4. **P2 cache 加固**：把 `metadata.json` 改成最后一步写入并 fsync；写入时先落 `*.tmp` 再 atomic rename。
5. **P3 上游 invariant**：在 pipeline 把 `factor_df.duplicated(["date","asset"])` 上 assert，避免未来 retrain 重叠静默 dedup。
6. **UI 视觉 enhancement**：补 model selection selected family 高亮与 k-fold verdict palette；归类为非阻塞缺口。

## 文档对账索引

| Inventory 行 | 当前产出 | 当前展示 | 状态 |
| --- | --- | --- | --- |
| `training_log.csv` | `core.py` 训练循环 + `artifacts.py:_prepare_training_log_for_export` | web training health table；status pill 已有，skip_reason 当 `not_skipped` 文本展示 | 通过 |
| `training_metrics.csv` | `core.py` per-fit | web training health curves（IC/RankIC/loss vs score_date） | 通过 |
| `feature_importance.csv` | `core.py:3603-3650` aggregation | web feature importance bar + top1/top3 share + concentration warnings | 口径需说明（sign_stability 分母）|
| `feature_importance_ledger.csv` | per-fit | web 暂以 CSV 诊断为主，feature panel 不画 ledger 时间线 | 通过 |
| `feature_oos_ic.csv` | `diagnostics.py` 单特征 OOS IC | web feature panel | 通过（但函数无 OOS 切片硬约束）|
| `purged_kfold_*` | `purged_kfold_diagnostics` | web k-fold card；verdict 颜色未实现 | 通过（视觉缺口）|
| `model_selection.json` / `_definition.json` | `artifacts.py` | web meta strip；selected family 高亮未实现 | 不通过（schema 漂移 + 视觉缺口）|

## 本轮未执行项

- 未运行真实 enabled selection run 验证 invariant test else 分支。
- 未做 dataset_cache atomic write 故障注入测试。
- 未做 model-lab UI 真实生产 run 视觉复核（仅依赖 `strong_skipped_extreme_nav` fixture smoke）。
- 未补 `no_candidates` / `selected_candidate_*` schema 修复或对应 golden。
- 未补 score factor `(date, asset)` 唯一性 hard assert。
