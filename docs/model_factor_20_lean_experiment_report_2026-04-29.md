# 20-lean 模型因子实验报告

生成日期：2026-04-29

## 结论摘要

本轮前端运行名称里包含两组重复 case：

- `stock_lightgbm_safe_bfq_20_lean`
- `stock_lasso_safe_bfq_20_lean`

本地 artifacts 目录中实际只找到每个唯一 case 的一份结果。因此本报告按两个唯一模型实验解读，并与对应 `35-core` baseline 对照。

核心结论：

1. `20-lean` 降维方向成立：数据准备与训练耗时明显下降。
2. `LightGBM 20-lean` 的绝对 alpha 强度低于 `LightGBM 35-core`，但 long-short IR 与成本后 IR 明显更好，说明降维后噪音和收益波动被压低。
3. `Lasso 20-lean` 基本保住了 RankIC 排序能力，但 IC 强度和 long-short 收益下降，说明 35-core 中被移除的部分特征对线性收益强度仍有贡献。
4. 两个 20-lean 模型仍然都是 `Strong Level 1 candidate`，但仍为 `Blocked from Level 2`，阻塞原因主要是 rolling evidence 不够稳定。
5. 20-lean 更适合作为下一轮特征筛选与增量因子验证的轻量 baseline，而不是直接替代 35-core 成为最终模型。

## 实验配置一致性

两个 20-lean case 共享以下条件：

- 评估 profile：`exploratory_screening`
- 特征数量：20
- target：5 日 forward return
- 训练窗口：rolling，`train_window_n_dates=120`
- retrain cadence：`retrain_every_n_dates=40`
- 最小训练行数：`min_train_rows=50000`
- 交易成本：`one_way_rate=0.001`
- model selection：未启用
- coverage：`coverage_mean=1.0`，评估覆盖率约 `73.3%`

完整性检查中，两者都有相同 warning：

- `pe_ttm` 使用 `required_timestamp`，仍有基本面特征可用性 caveat。
- factor 输出覆盖少于 raw universe，属于覆盖 gate/有效样本过滤提示。

这两个 warning 不影响 LightGBM 与 Lasso 之间的相对比较，但正式研究仍建议补跑 safety-lag 稳健性。

## 20-lean 结果对比

| 指标 | LightGBM 20-lean | Lasso 20-lean | 解读 |
| --- | ---: | ---: | --- |
| Mean IC | 0.1382 | 0.0716 | LightGBM alpha 强度明显更高 |
| ICIR | 0.8472 | 0.6334 | LightGBM 更强 |
| Mean RankIC | 0.0470 | 0.0729 | Lasso 排序能力更好 |
| RankIC IR | 0.4476 | 0.6122 | Lasso 排序稳定性更好 |
| IC 正比例 | 83.92% | 76.41% | LightGBM 方向一致性更好 |
| RankIC 正比例 | 68.81% | 76.08% | Lasso 排序正向天数更多 |
| Mean long-short | 1.4965% | 1.3355% | LightGBM 收益更高 |
| 成本后 long-short | 1.4313% | 1.3065% | LightGBM 仍更高 |
| Long-short IR | 0.5989 | 0.5096 | LightGBM 更好 |
| 成本后 long-short IR | 0.5725 | 0.4989 | LightGBM 更好 |
| 平均换手 | 0.6480 | 0.2788 | Lasso 换手显著更低 |
| 最大回撤 | 27.67% | 17.43% | Lasso 风险更低 |
| IC decay retention 5/1 | 1.3013 | 1.6899 | Lasso 衰减表现更稳 |
| Level 2 状态 | Blocked | Blocked | 两者都不能直接推进 Level 2 |

读数：

- 如果按 **alpha 强度/收益能力**，LightGBM 20-lean 更强。
- 如果按 **排序稳健性/低换手/低回撤**，Lasso 20-lean 更稳。
- 这和 35-core 阶段的结论一致：LightGBM 是树模型 champion，Lasso 是线性 champion，但两者承担不同研究角色。

## 与 35-core baseline 对照

### LightGBM：20-lean vs 35-core

| 指标 | 20-lean | 35-core | 变化 |
| --- | ---: | ---: | ---: |
| Mean IC | 0.1382 | 0.1537 | -10.1% |
| ICIR | 0.8472 | 0.9218 | -8.1% |
| Mean RankIC | 0.0470 | 0.0502 | -6.2% |
| RankIC IR | 0.4476 | 0.4721 | -5.2% |
| Mean long-short | 1.4965% | 1.8182% | -17.7% |
| 成本后 long-short | 1.4313% | 1.7552% | -18.5% |
| Long-short IR | 0.5989 | 0.2191 | 大幅改善 |
| 成本后 long-short IR | 0.5725 | 0.2115 | 大幅改善 |
| 平均换手 | 0.6480 | 0.6298 | +2.9% |
| 最大回撤 | 27.67% | 27.19% | 小幅变差 |
| Rolling RankIC 正比例 | 88.17% | 85.69% | 小幅改善 |

LightGBM 降维后的判断：

- 删除 15 个特征后，IC 与收益强度确实下降。
- 但 long-short IR 明显改善，说明 20-lean 的收益波动更可控。
- 回撤没有改善，换手略升，这说明它还不是更优组合候选，只是更干净的 Level 1 筛选版本。
- `pe_ttm` 在 20-lean 中仍进入 top features，基本面可用性 caveat 仍需要保留。

### Lasso：20-lean vs 35-core

| 指标 | 20-lean | 35-core | 变化 |
| --- | ---: | ---: | ---: |
| Mean IC | 0.0716 | 0.0863 | -17.0% |
| ICIR | 0.6334 | 0.7157 | -11.5% |
| Mean RankIC | 0.0729 | 0.0732 | -0.4% |
| RankIC IR | 0.6122 | 0.6483 | -5.6% |
| Mean long-short | 1.3355% | 1.5621% | -14.5% |
| 成本后 long-short | 1.3065% | 1.5334% | -14.8% |
| Long-short IR | 0.5096 | 0.1931 | 大幅改善 |
| 成本后 long-short IR | 0.4989 | 0.1896 | 大幅改善 |
| 平均换手 | 0.2788 | 0.2842 | -1.9% |
| 最大回撤 | 17.43% | 17.69% | 小幅改善 |
| Rolling RankIC 正比例 | 91.79% | 91.79% | 持平 |

Lasso 降维后的判断：

- RankIC 几乎完整保留，说明 20-lean 没有破坏 Lasso 的主要排序信息。
- IC 与 long-short 收益下降，说明被删掉的 15 个特征中仍有一些线性收益强度贡献。
- 风险调整后 IR 改善明显，可能来自收益序列波动下降。
- Lasso 20-lean 是一个很好的低冗余线性筛选器，但不是收益最强版本。

## 特征重要性观察

### LightGBM 20-lean Top 10

| 排名 | 特征 | 重要性 |
| ---: | --- | ---: |
| 1 | `updays` | 1817 |
| 2 | `topdays` | 1815 |
| 3 | `lowdays` | 1770 |
| 4 | `downdays` | 1652 |
| 5 | `total_mv` | 1074 |
| 6 | `atr_bfq` | 1033 |
| 7 | `obv_bfq` | 915 |
| 8 | `pe_ttm` | 833 |
| 9 | `turnover_rate` | 830 |
| 10 | `turnover_rate_f` | 818 |

LightGBM 仍高度依赖路径/趋势状态组：`updays/topdays/lowdays/downdays`。这与 35-core 一致，说明这组特征不是偶然噪音。

### Lasso 20-lean 非零特征

| 特征 | 系数方向 | 重要性 |
| --- | ---: | ---: |
| `turnover_rate_f` | 负 | 0.001803 |
| `atr_bfq` | 正 | 0.000365 |
| `turnover_rate` | 负 | 0.000248 |
| `dv_ttm` | 负 | 0.000155 |

Lasso 的非零集合与 35-core 基本一致，说明 20-lean 没有改变线性模型的核心选择逻辑。

## 运行效率

| 阶段 | LightGBM 20-lean | LightGBM 35-core | Lasso 20-lean | Lasso 35-core |
| --- | ---: | ---: | ---: | ---: |
| data_load | 22.75s | 57.75s | 14.30s | 38.11s |
| feature_validate | 39.95s | 71.16s | 41.93s | 81.47s |
| target_build | 82.34s | 145.74s | 91.34s | 154.10s |
| model_fit | 286.63s | 447.73s | 86.84s | 146.94s |
| predict | 33.83s | 37.75s | 4.44s | 4.40s |
| evaluate | 105.10s | 110.25s | 103.85s | 104.14s |

效率结论：

- 20-lean 对数据准备和训练有实质收益。
- LightGBM model_fit 下降约 36%。
- Lasso model_fit 下降约 41%。
- evaluate 基本不随特征数下降，说明后续若继续提速，应优化评估链路，而不是继续砍特征。

## 当前定位

`20-lean` 不是直接替代 `35-core` 的最终版本，而是一个更适合迭代的轻量研究基线：

- 用于快速筛选新增 alpha 是否有增量。
- 用于检查模型是否过度依赖某一类特征。
- 用于和 `25-consensus`、`25-tree-aware` 做三路对照。

当前建议：

1. 保留 `Lasso 20-lean` 作为线性低冗余筛选器。
2. 保留 `LightGBM 20-lean` 作为非线性轻量筛选器。
3. 不建议仅凭本轮结果把 35-core 替换为 20-lean。
4. 下一步优先跑 `25-consensus + Lasso/LightGBM`，验证能否在保留 20-lean 稳定性的同时追回部分 35-core alpha 强度。
5. 若 25-consensus 仍不够，再跑 `25-tree-aware + LightGBM/GBDT`，专门验证树模型非线性信息。

