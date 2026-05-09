# Model-Factor 性能路线记录

这份记录保存 model-factor 路径的性能证据、已排除假设和下一步顺序。默认研究完整性优先：
`default_research` 不弱化检查、不改变 Level 1/2 语义；快速路径和压测工具只用于显式 profile 或 benchmark。

## 当前证据

- `stock_ridge_medium_bfq` medium 数据：
  - 行数：`1,331,053`
  - feature parquet：约 `675MB`
  - cold run：`90.57s`，Peak RSS `5.17GB`，Swap `0`
  - warm run：`42.99s`，Peak RSS `5.63GB`，Swap `0`
- medium memory-profile：
  - cold peak：`model_fit`，约 `5.21GB`
  - `data_load` peak：约 `3.35GB`
  - `target_build` peak：约 `3.32GB`
  - `feature_validate` peak：约 `2.86GB`
  - `predict` peak：约 `2.80GB`
- full safe_bfq 旧全特征压测：
  - `features_safe_bfq.parquet` 约 `4.6GB`；后续训练配置先切到 `features_safe_bfq_45.parquet`，再收敛到中间档 `features_safe_bfq_35.parquet`
  - `features_safe.parquet`：25 个特征，约 `1.15GB`，已知 full 可跑完
  - `features_safe_bfq_35.parquet`：35 个特征，约 `1.63GB`，作为当前 model-factor 默认研究折中档
  - `features_safe_bfq_45.parquet`：45 个特征，约 `2.01GB`，保留为扩展候选池
  - full cold run 曾在 `data_load` cache 写完后 OOM kill
  - Python anon RSS 观测约 `17-19GB`，swap 用尽

## 已落地

- benchmark guarded launcher：
  - `ulimit -v`
  - `MALLOC_ARENA_MAX=2`
  - `PYTHONMALLOC=malloc`
  - benchmark record 写入 memory resource limits
- medium 数据物化：
  - 最近 365 天，全特征，全 universe
  - 用 pyarrow streaming filter 物化，避免先读 full pandas
- memory instrumentation：
  - per-sample RSS/swap timeline
  - current stage
  - stage peak 汇总
  - selected stage 边界 tracemalloc diff
- data_load cache writer：
  - 从 `DataFrame.to_parquet()` 改为 `ParquetWriter`
  - 每 `128k` 行一个 row group
  - medium data_load cache 从 `2` row groups 改成 `11` row groups
- 5a OOM 止血：
  - 无 `model_selection` 的默认训练路径改为在 `training_window_index` 后构造紧凑 numpy 数组。
  - `model_fit` / `predict` 通过 row index 切数组，不再反复 `features.take(...).copy()`。
  - 特征重要性只保留最多 `permutation_max_rows` 的小样本，不挂住完整训练窗口。
  - 大对象阶段边界使用 `release_unused_memory()`，在 `gc.collect()` 后尝试 `malloc_trim(0)` 归还 glibc heap 页。
  - `model_selection=True` 仍保留原 DataFrame / Purged CV 路径，避免改变选模语义。

## 5a 验收结果

当前 5a 还不是最终收口，只是第一轮止血通过。full 数据约 `13M` 行，按 medium warm 当前
`3.67GB / 1.33M rows` 外推仍约 `36GB`，不足以保证 full 在 `24GB` ulimit 内稳定完成。
5a 的退出门槛更新为：

- `stock_ridge_medium_bfq` warm Peak RSS `<= 2.0GB`
  - `2.5GB` 只对应 full 约 `24GB`，几乎没有 sklearn C 层和 glibc 碎片 headroom。
  - `2.0GB` 对应 full 约 `20GB`，给 guarded full run 留约 `4GB` 余量。
- 完成 prepared DataFrame 引用审计，确认 numpy 热路径是替换而不是叠加
- 完成 `copy=False` 与 contiguous window slice 验证
- prepared_v2 warm path 中 fit 窗口应尽量从 `advanced_index` 翻成 `contiguous_slice`
- medium 达标后再跑 full guarded baseline

- `stock_ridge_medium_bfq` cold，`exploratory_screening`，`retrain_every_n_dates=60`：
  - wall：`87.37s`
  - Peak RSS：`4.87GB`
  - cache：`feature_validate=false`，`target_build=false`
  - stage peak：`data_load 3.36GB`，`target_build 3.25GB`，`training_window_index 2.43GB`，`preprocess 3.06GB`，`model_fit 4.87GB`，`predict 2.51GB`，`evaluate 1.33GB`
- `stock_ridge_medium_bfq` warm，同配置：
  - wall：`40.90s`
  - Peak RSS：`4.76GB`
  - cache：`feature_validate=true`，`target_build=true`
  - stage peak：`data_load 2.31GB`，`training_window_index 2.51GB`，`preprocess 2.55GB`，`model_fit 4.76GB`，`predict 2.50GB`，`evaluate 2.32GB`
- `stock_ridge_medium_bfq` warm，引用审计 + `copy=False` + 跳过 screening prepared-cache features reload 后：
  - wall：`40.96s`
  - Peak RSS：`3.67GB`
  - cache：`feature_validate=true`，`target_build=true`
  - stage peak：`training_window_index 1.87GB`，`preprocess 1.94GB`，`model_fit 3.67GB`，`predict 1.67GB`，`evaluate 1.49GB`
  - `data_load`：`0.084s`，memory sampler 未捕捉到有效峰值
  - prepared DataFrame 审计：`source_dataframe_released=true`
  - fit 窗口选择：`advanced_index=3`，`contiguous_slice=0`
  - `feature_manifest.manifest_source=cache_metadata`，快速筛选 warm path 不再为了 manifest 重读 features 全表
- `stock_ridge_medium_bfq` warm，prepared_inputs v2 + compact labeled matrix 后：
  - wall：`39.84s`
  - Peak RSS：`3.39GB`
  - cache：`feature_validate=true`，`target_build=true`
  - stage peak：`training_window_index 0.66GB`，`preprocess 0.63GB`，`model_fit 3.39GB`，`predict 1.50GB`，`evaluate 1.23GB`
  - prepared cache layout：`features.npy` + `labeled_features.npy` + `columns.json` + `index.parquet` + `labeled_index.parquet`
  - prepared cache version：`model_factor_preparation_v2`
  - row order：`date_asset`
  - feature dtype：`float32`
  - `features.npy`：约 `472MB`
  - `labeled_features.npy`：约 `462MB`
  - fit 窗口选择：`contiguous_slice=3`，`advanced_index=0`
- 对比 5a 前 medium：
  - cold peak：约 `5.17GB` -> `4.87GB`
  - warm peak：约 `5.63GB` -> `3.39GB`
  - `predict` warm peak：约 `2.80GB` -> `1.50GB`
  - 旧 prepared DataFrame 已释放；warm 降幅不足不是引用悬挂导致。
  - prepared_v2 已把 fit 窗口从 `advanced_index` 翻成 `contiguous_slice`。
  - 5a 尚未达 `<=2.0GB` 退出门槛；剩余主峰集中在 sklearn model_fit 内部，而不是 prepared pandas 或窗口取数。

## 已排除或校准的假设

- Ridge solver 不是主瓶颈。
  - toy `675k x 94 float32` Ridge `solver=auto`：
    - baseline RSS 约 `428.6MB`
    - fit peak 约 `904.5MB`
    - fit delta 约 `476MB`
  - pipeline `model_fit` peak 约 `5.2GB`，差距主要不在 sklearn solver 内部。
- data_load row group writer 修复有价值，但不是立即降 RSS 的充分条件。
  - cache writer 已保留更细 row group。
  - `data_load` tracemalloc top allocator 仍约 `967MB`，来自 pandas block materialization。
  - 结论：真问题是 reader 仍通过 `pd.read_parquet` 全表物化宽表。
- `prepared_inputs` 单独分区写不是 warm peak 的直接解。
  - warm cache hit 后 peak 仍发生在 `model_fit`。
  - 仅改 prepared cache layout，不改窗口取数/输入形态，收益有限。
- C1/C2 的非 evaluate wall 下降不应归因为 per-fit cached label copy。
  - 低 fit_count A/B 中，临时关闭 C1/C2 后 wall 几乎不变：
    `151.16s` vs `151.20s`。
  - 结论：此前 40-fit benchmark 中的非 evaluate 差额更像运行噪音或运行状态差异。
  - 后续性能结论必须以 stage timing 和可切换 A/B 归因为准，不能只看单次 wall delta。

## 规模感与归因口径

- knife A 是本轮 model-factor 性能工作的主胜利：
  - 35 特征 full path 从 guarded failure 变成可稳定跑完。
  - cold 从失败前约 `1613s` 的长跑崩溃路径收敛到 `484.53s`。
  - warm 路径稳定在约 `228s`、Peak RSS 约 `9.2GB`，无 swap。
- C1/C2/C3 属于 trim-the-edges：
  - C1/C2/C3 真实可解释收益约 `20s`，约 `7%` wall。
  - baseline -> C1/C2 的 `-53.92s` wall 中，只有 evaluate 内 `-16.64s` 可稳定归因；其余差额不再作为优化收益入账。
  - C3 的可归因收益约 `3.43s`。
- 经验：
  - fix-the-crash 与 trim-the-edges 不是同一个量级。
  - 测试 bit-exact 只能说明语义安全，不能说明性能值得收下；全局 `(date, value)` lexsort 就是反例。
  - 未来小于 warm noise floor 的优化不应进入路线主叙事。

## 5a 子顺序与状态

1. Stage 边界释放审计。
   - 目标：确认 `target_build` 退出到 `model_fit` 入口之间仍存活的大对象。
   - 优先检查前阶段 DataFrame 是否被局部变量、diagnostics payload、闭包或列表引用保留。
   - 修复方式优先用 `del`、置空 DataFrame、`gc.collect()`，不改语义。
   - 状态：已完成。
   - 结果：medium warm `model_fit` peak 从约 `5.36GB-5.63GB` 降到 `4.76GB`。

2. Prepared DataFrame 引用审计。
   - 目标：确认 `_PreparedModelArrays` 构造后，旧 prepared/merged DataFrame 没有被 pipeline 局部变量、prepared cache、闭包或 diagnostics payload 持有。
   - 方法：
     - 在数组构造后用 `tracemalloc.take_snapshot()` 看大块 pandas allocator 是否仍活跃。
     - 对明确命名的 prepared/merged DataFrame 做 `gc.get_referrers(...)` 审计。
   - 预期：
     - 若命中，说明 numpy 热路径是“叠加”而非“替换”，可能直接回收 `1GB+`。
   - 状态：已完成。
   - 结果：`source_dataframe_released=true`，旧 prepared/merged DataFrame 没有在数组热路径后继续悬挂。

3. `copy=False` 与 contiguous slice 验证。
   - 目标：消掉训练窗口和 sklearn preprocess 的重复矩阵副本。
   - 检查点：
     - `SimpleImputer(copy=False)` 与 `StandardScaler(copy=False)` 是否能在 float32 C-contiguous 输入上原地工作。
     - `_PreparedModelArrays` 是否已经按 `date, asset` 排序。
     - rolling window 的 row index 是否能表达为 contiguous slice；`X[start:end]` 是 view，`X[idx]` 是 advanced-index copy。
   - 预期：
     - 如果大部分训练窗口可用 contiguous slice，可直接减少一份训练窗口矩阵副本。
   - 状态：已完成第一轮。
   - 结果：
     - `SimpleImputer(copy=False)`、`StandardScaler(copy=False)` 已落地。
     - 线性模型族默认 `copy_X=False`，用户显式参数仍优先。
     - medium warm 3 次 fit 全部为 `advanced_index`，没有命中 contiguous slice。
     - 单独收益较小，`4.76GB -> 4.63GB`，说明主副本不只在 sklearn preprocess copy 参数。

4. data_load reader lazy 化。
   - 用 `pyarrow.dataset.Dataset` / row group scanner 替代 `pd.read_parquet` 全表读。
   - 注意：如果下游立刻 `to_pandas()` 全表，只是平移 materialization 点，不算完成。
   - 状态：未做。
   - 说明：5a 后 cold `data_load` peak 约 `3.36GB`，已不是 medium 主峰；full 仍可能需要 lazy reader。

4a. prepared-cache warm path 跳过 data_load features 全表加载。
   - 状态：已完成，限 `exploratory_screening`。
   - 结果：
     - warm `data_load` 从 `1.5s` 量级降到 `0.084s`。
     - warm Peak RSS 从 `4.63GB` 降到 `3.67GB`。
     - `default_research` 仍保留完整 feature reload / manifest 统计路径，避免弱化默认研究完整性。

5. prepared inputs 改 numpy 形态缓存。
   - 建议形态：
     - `features.npy`
     - `columns.json`
     - `index.parquet`，保存 `date/asset/known_at/label` 等索引与标签列
   - 目标：同时去掉 pandas 中间形式，并让训练窗口尽量成为 numpy view。
   - 关键约束：
     - 写 cache 时固定按 `date, asset` 排序，并保存可验证的 row order metadata。
     - `features.npy` 使用 `float32`，metadata 写入 dtype，warm load 时必须校验。
     - warm path 用 `np.load(mmap_mode="r")`，它省的是 prepared 整段常驻，不会省掉 imputer/scaler 输出矩阵。
     - labeled train rows 需要能映射到连续区间；如果同一 date 内 label 缺失破坏连续性，要考虑把 valid-labeled training mask/compact training matrix 一并写入 sidecar。
     - schema 破坏性变更，cache version bump 到 `model_factor_preparation_v2`，老 v1 cache 失效并给开发者 migration 提示：`prepared cache schema v1 -> v2, please rerun cold once`。
   - 状态：已完成第一版。
   - 结果：
     - `model_factor_preparation_v2` 已落地。
     - warm path 用 `np.load(..., mmap_mode="r")` 读取 `features.npy` 和 `labeled_features.npy`。
     - `labeled_features.npy` 让 3 次 fit 全部命中 `contiguous_slice`。
     - `training_window_index/preprocess` warm peak 降到 `0.6GB` 量级。
     - warm Peak RSS 降到 `3.39GB`，仍未达 `2.0GB` 退出门槛。

5a. prepared_v2 回归测试要求。
   - cold run 仍必须走完整 `data_load -> feature_validate -> target_build -> prepared write`。
   - screening warm fast path 只允许在 prepared cache hit 时跳过 data-load features reload。
   - prepared_v2 warm 需要断言：
     - `features_loaded_for_data_load=false`
     - `prepared_inputs_cache_hit=true`
     - `feature_manifest.manifest_source` 来自 sidecar/cache metadata
     - `model_matrix_selection` 的 `contiguous_slice` 命中率被记录；若仍为 0，需要解释原因。

5b. full 35 特征 guarded 验收。
   - `stock_ridge_safe_bfq_35`，full 数据，`exploratory_screening`，`retrain_every_n_dates=60`，guarded `18GB`。
   - 刀 A 前：
     - cold run 在 `prepare_model_arrays` 失败：
       `MemoryError: Unable to allocate 1.40 GiB for array (10701607, 35)`。
     - 未触发 WSL OOM，swap `0`。
   - 刀 A：
     - prepared cache 写入改为 chunked memmap。
     - cold run 写好 prepared cache 后同轮采用 `numpy_v2_mmap`。
   - cold run：
     - wall `484.53s`，Peak RSS `10.92GB`，swap `0`，fit_count `40`。
     - `feature_validate 77.52s`，`target_build 143.74s`，`model_fit 93.26s`，`evaluate 95.00s`。
     - `prepared_cache_write_succeeded=true`，`prepared_cache_adopted_for_run=true`。
   - warm run：
     - wall `224.96s`，Peak RSS `9.62GB`，swap `0`，fit_count `40`。
     - `feature_validate 0.24s`，`target_build 0.00s`，`model_fit 95.55s`，`evaluate 96.64s`。
     - `feature_validate_cache_hit=true`，`target_build_cache_hit=true`。
   - 结论：
     - 35 特征 full 已能在当前 WSL 配置下用 guarded benchmark 跑完。
     - 后续主要瓶颈转为 `evaluate` 与 `model_fit`，以及 cold path 的 `target_build/feature_validate`。

5c. evaluate drilldown。
   - `stock_ridge_safe_bfq_35`，warm full，`exploratory_screening`，`retrain_every_n_dates=60`，guarded `18GB`。
   - wall `285.46s`，Peak RSS `9.13GB`，swap `0`，fit_count `40`。
   - cache 命中：
     - `feature_validate_cache_hit=true`
     - `target_build_cache_hit=true`
   - case-level：
     - `model_fit 110.79s`
     - `evaluate 119.95s`
     - `predict 9.14s`
   - evaluate 子阶段：
     - `core_backtest 86.09s`
     - `ic_decay 16.15s`
     - `data_quality_summary 3.34s`
     - `evaluate_other 4.35s`
   - `core_backtest` 子阶段：
     - `label 30.10s`
     - `quantile 21.20s`
     - `factor 13.10s`
     - `ic 9.26s`
     - `merge 8.61s`
     - `regime_tail 2.17s`
   - 结论：
     - warm path 的 evaluate 主要不是 exploratory 诊断，而是 core backtest。
     - `factor` 和 `label` 很可能是 full-frame copy 成本；下一刀优先评估减少 `run_factor_experiment` 在 model-factor evaluate 中的重复拷贝，而不是先做 cold-only feature_validate 向量化。
   - 计时契约：
     - `core_backtest.*` 是 `run_factor_experiment.stage_timings` 的下钻视图，不计入 `evaluate_named_children_total`。
     - 因此 `evaluate_named_children_total` 与所有 `evaluation_stage_timings` 子项简单求和不应相等，这不是重复计时 bug。

5d. core_backtest C1/C2 no-copy 热路径。
   - 改动：
     - `LabelCache.forward_return(..., copy=True)` 保持默认安全 copy；仅内部热路径显式传 `copy=False`。
     - cached close label builder 增加 `copy` 参数；model-factor evaluate 的核心回测使用只读浅拷贝 label。
     - core backtest 的 factor 输入从 `factor_df.copy()` 改为只读浅拷贝，避免一开始就复制全表。
     - `_resolve_merged_pairs` 保持旧的 copy 语义；新增 `_borrow_merged_pairs` 用于 IC / RankIC / MI 热路径复用 `merged_eval`。
     - 借用路径对共享列加 array-level write protection，未来若误写共享数组会直接失败。
   - 契约：
     - 公开/默认调用仍是 copy-safe。
     - 只有明确命名的 borrow / `copy=False` 内部路径允许复用共享 frame。
   - `stock_ridge_safe_bfq_35` warm full，同 5c 配置：
     - wall `231.54s`，Peak RSS `9.41GB`，swap `0`，fit_count `40`。
     - case-level：`model_fit 95.95s`，`evaluate 103.53s`，`predict 6.79s`。
     - evaluate 子阶段：`core_backtest 68.35s`，`ic_decay 15.67s`，`data_quality_summary 3.20s`，`evaluate_other 6.07s`。
     - `core_backtest` 子阶段：
       - `label 30.10s -> 21.41s`
       - `factor 13.10s -> 7.42s`
       - `quantile 21.20s -> 16.74s`
       - `merge 8.61s -> 8.56s`
       - `ic 9.26s -> 11.05s`
     - 相对 5c drilldown baseline：
       - wall `285.46s -> 231.54s`，约 `18.9%` 下降。
       - `evaluate_total 109.95s -> 93.31s`。
       - `core_backtest 86.09s -> 68.35s`。
   - 结论：
     - C1 的 factor/label 去 copy 是实打实收益。
     - C2 的 `_resolve_merged_pairs` copy 不是 IC 主瓶颈；IC 下游仍会为数值清洗、排序、numba/numpy 输入构造做 copy。
     - 下一刀不应继续磨 `_resolve_merged_pairs`，而应看 `quantile` 的 groupby/assign 路径，或把 IC 下游的 clean/sort 数组构造和 `merged_eval` 生命周期一起重新设计。

5e. C3 quantile assignment 快路径。
   - 语义校准：
     - `quantile_assignments` 的 universe 是所有因子非 NaN 行，包含尾部“有因子、无 label”的日期，用于换手率。
     - `quantile_returns` 的 universe 是因子和 label 都非 NaN 的子集，用于桶收益。
     - 因此两者不能强行共享单一分桶 universe，否则会改变尾部换手率或桶收益语义。
   - 测试护栏：
     - 新增随机 mixed panel oracle，`_assign_quantiles_by_date` 必须逐元素匹配旧 dense-rank / half-up 语义。
     - 新增 tail-universe 回归测试，锁住 assignments 与 returns 在 label 缺失日期使用不同 universe 的契约。
   - 反证：
     - 全局 `(date, value)` lexsort 向量化尝试是负收益：
       - wall `238.95s`
       - `core_backtest.quantile 23.43s`
       - 原因：旧实现虽然有 per-date loop，但真实截面较小；全表 value sort 成本更高。
     - 该实现未保留为最终路径。
   - 最终改动：
     - `_assign_quantiles_by_date` 保持旧 per-date dense-rank 语义，但在输入日期已单调时跳过全表 `date` 排序。
     - `quantile_returns(merged_pairs=...)` 不再先 rename 出 `value/_label` 副本，而是直接使用 `value_factor/value_label`。
     - `quantile_assignments` 去掉一层守势 `.copy()`。
   - `stock_ridge_safe_bfq_35` warm full，同 5d 配置：
     - wall `228.11s`，Peak RSS `9.19GB`，swap `0`，fit_count `40`。
     - case-level：`model_fit 93.07s`，`evaluate 104.34s`，`predict 6.14s`。
     - evaluate 子阶段：`core_backtest 65.49s`，`ic_decay 18.29s`，`data_quality_summary 3.82s`，`evaluate_other 5.80s`。
     - `core_backtest.quantile 16.74s -> 14.76s`，`core_backtest 68.35s -> 65.49s`。
   - 结论：
     - C3 是安全小收益，不是数量级收益。
     - quantile 残余时间不只在 assignment；若继续压，需要把 `quantile_returns` 的 assignment / groupby mean / assignments / turnover 分开计时。

5f. low-fit C1/C2 因果校准。
   - 目的：
     - 验证 C1/C2 后 40-fit benchmark 中非 evaluate 的额外下降是否来自 per-fit cached label copy。
   - 方法：
     - 使用 `screening_retrain_every_n_dates=240`，fit_count 从 `40` 降到 `10`。
     - 跑当前 C1/C2/C3 代码，再临时关闭 C1/C2 热路径跑同一 warm cache 对照，随后恢复代码。
   - 结果：
     - 当前 C1/C2/C3：wall `151.16s`，Peak RSS `8.62GB`，fit_count `10`。
     - 临时关闭 C1/C2：wall `151.20s`，Peak RSS `8.93GB`，fit_count `10`。
     - C1/C2 对低-fit wall 影响约 `0.04s`，不支持“约 1s/fit 的 label copy 节省”假设。
     - 40-fit 到 10-fit 当前路径：
       - wall `228.11s -> 151.16s`
       - `model_fit 93.07s -> 23.36s`
       - `evaluate 104.34s -> 97.75s`
   - 结论：
     - fit cadence 是确定大杠杆；`model_fit` 基本按 fit_count 线性缩放。
     - C1/C2 的主要可解释收益仍在 core backtest 的 factor/label copy；此前观察到的非 evaluate 差额不应归因给 C1/C2，记为 benchmark 抖动/运行状态差异。
     - 后续做性能结论时，不能只看单次 wall delta；需要用 stage timing 与可切换 A/B 锁定因果。

6. 兜底才做 partition-aware fit/predict。
   - 只有引用审计、`copy=False`、contiguous slice、reader/cache layout 这些小刀后 warm peak 仍不达标，才进入按 date 分区的窗口取数与训练路径重构。
   - 这是较大设计变更，不作为默认第一刀。

7. 暂不优先换 Ridge solver。
   - solver memory 已被 toy benchmark 排除为主因。
   - 如未来遇到不同模型族或更高维特征，再单独评估 solver / incremental fit。

## 保留事项

- 不回滚 data_load row group writer。
  - 它是 lazy reader 后续按 row group 调度的前提。
- `prepared_inputs` schema 若改为 numpy/partitioned layout，需要 bump cache version。
- full baseline 只在 medium 通过并且 guarded benchmark 稳定后再跑。
- 下一轮优先级：
  - 回到原 roadmap：`exploratory_screening + training_sample_policy=train_window_months:12` / medium `<=1.5GB` 仍待开，优先级高于继续打磨 `quantile/IC/merge` 边角。
  - fit cadence 是当前最稳的开发筛选杠杆；可考虑将 `exploratory_screening` 的推荐值从 `60` 分层到更激进的 dev profile（例如 `120/240`），但需明确结果只用于粗筛。
  - benchmark 路径建议分流：
    - perf A/B、内存追踪、noise-floor 测试默认用低 fit_count（例如 `screening_retrain_every_n_dates=240`），减少训练链路噪音。
    - production-equivalent 因子表现评估继续用 `60`，因为 cadence 会改变 OOS 行为，不能当成纯性能参数。
  - 建议新增 benchmark CLI preset，而不是每次手写参数：
    - `perf_debug`：低 fit_count，面向性能 A/B 与内存诊断。
    - `research_equivalent`：当前 `exploratory_screening` 推荐 cadence，面向接近真实研究结论的对比。
  - 若继续压 evaluate，先给 `core_backtest.quantile` 增加子计时：`quantile_returns.assignment` / `quantile_returns.groupby_mean` / `quantile_assignments.assignment` / `turnover`。
  - 若 quantile 子计时显示 groupby mean 是主瓶颈，再考虑 numpy `bincount` 聚合；否则转向 IC 下游的 clean/sort 数组构造。
  - `model_fit` 内部 copy profile 暂作为后续分支；当前 warm wall 的下一根长杆已经转到 evaluate/core_backtest。
- 待测 noise floor：
  - 同一份 cache、同一份代码连续跑 5 次 warm benchmark，记录 wall / stage stddev。
  - 后续所有 “节省 X 秒” 的小优化必须和该 noise floor 对比后再归因。
