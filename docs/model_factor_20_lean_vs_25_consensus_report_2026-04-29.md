# 20-lean vs 25-consensus 模型因子实验报告

生成日期：2026-04-29

## 实验范围

本报告基于前端汇总结果，并补充读取本地 artifacts：

- `stock_lightgbm_safe_bfq_20_lean`
- `stock_lightgbm_safe_bfq_25_consensus`
- `stock_lasso_safe_bfq_20_lean`
- `stock_lasso_safe_bfq_25_consensus`

四个实验均成功完成，coverage 均为 `1.0000`。所有结果仍是 Level 1 baseline 代表模型选择与特征降维验证，不是 Level 2 可上线结论。

## 总体结论

1. `20-lean` 比 `25-consensus` 更适合作为当前轻量 baseline。
2. `LightGBM 25-consensus` 虽然 IC 和 RankIC 略高于 `LightGBM 20-lean`，但 long-short IR 从 `0.5989` 降到 `0.2092`，成本后 IR 从 `0.5725` 降到 `0.2015`，最大回撤升到 `32.26%`，并被判为 `Fails basic robustness`。树模型不建议采用 25-consensus。
3. `Lasso 25-consensus` 相比 `Lasso 20-lean` 提升了 IC、RankIC 和平均 long-short return，但 long-short IR 从 `0.5096` 降到 `0.1881`，成本后 IR 从 `0.4989` 降到 `0.1846`，回撤也略升。它更像“收益强度增强但波动显著变差”的版本，不适合作为当前 champion。
4. 当前最稳的代表仍是：
   - 线性筛选器：`Lasso 20-lean`
   - 树模型筛选器：`LightGBM 20-lean`
5. `25-consensus` 暂不淘汰，但应降级为候选观察组，下一步需要定位新增的 5 个特征是否引入了收益路径不稳定或基本面可用性风险。

## 四模型核心指标

| 模型 | Verdict | Mean IC | ICIR | Mean RankIC | RankIC IR | 平均换手 | Long-short IR | 成本后 IR | 最大回撤 | Coverage |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| LightGBM 20-lean | Promising but fragile | 0.1382 | 0.8472 | 0.0470 | 0.4476 | 0.6480 | 0.5989 | 0.5725 | 0.2767 | 1.0000 |
| LightGBM 25-consensus | Fails basic robustness | 0.1414 | 0.8759 | 0.0476 | 0.4518 | 0.6269 | 0.2092 | 0.2015 | 0.3226 | 1.0000 |
| Lasso 20-lean | Promising but fragile | 0.0716 | 0.6334 | 0.0729 | 0.6122 | 0.2788 | 0.5096 | 0.4989 | 0.1743 | 1.0000 |
| Lasso 25-consensus | Promising but fragile | 0.0781 | 0.6786 | 0.0744 | 0.6438 | 0.2717 | 0.1881 | 0.1846 | 0.1828 | 1.0000 |

## LightGBM 组内对比

`LightGBM 25-consensus` 的正面变化：

- Mean IC：`0.1382 -> 0.1414`
- ICIR：`0.8472 -> 0.8759`
- Mean RankIC：`0.0470 -> 0.0476`
- RankIC IR：`0.4476 -> 0.4518`
- 平均换手：`0.6480 -> 0.6269`

这些提升都存在，但幅度很小。

主要负面变化：

- Long-short IR：`0.5989 -> 0.2092`
- 成本后 long-short IR：`0.5725 -> 0.2015`
- 最大回撤：`27.67% -> 32.26%`
- Factor verdict：`Promising but fragile -> Fails basic robustness`
- Campaign triage：从 `Strong Level 1 candidate` 降为 `Drop for now`

解释：

25-consensus 给 LightGBM 增加了 `free_share`、`pb`、`ps_ttm`、`macd_bfq`、`bias2_bfq` 等特征。它确实让 IC 层面稍微更强，但没有转化成更稳的 long-short 收益，反而让收益路径恶化。这通常说明模型捕捉到了一些横截面相关性，但这些相关性在组合分层收益中不够稳定，或集中在不利的尾部/阶段性结构里。

结论：

`LightGBM 20-lean` 明显优于 `LightGBM 25-consensus`。树模型 champion 暂时应保留为 `LightGBM 20-lean`。

## Lasso 组内对比

`Lasso 25-consensus` 的正面变化：

- Mean IC：`0.0716 -> 0.0781`
- ICIR：`0.6334 -> 0.6786`
- Mean RankIC：`0.0729 -> 0.0744`
- RankIC IR：`0.6122 -> 0.6438`
- 平均 long-short return：`1.3355% -> 1.5336%`
- 成本后 long-short return：`1.3065% -> 1.5055%`
- 平均换手：`0.2788 -> 0.2717`

负面变化：

- Long-short IR：`0.5096 -> 0.1881`
- 成本后 long-short IR：`0.4989 -> 0.1846`
- 最大回撤：`17.43% -> 18.28%`

解释：

Lasso 25-consensus 的横截面排序和平均收益都变强了，但收益序列波动大幅上升，导致 IR 反而明显变差。也就是说，新增特征对 Lasso 有信息量，但这些信息更像是“高波动的收益强度”，不是稳定可复用的收益路径。

结论：

`Lasso 25-consensus` 可以作为观察组，但当前不应替代 `Lasso 20-lean`。线性 champion 仍建议定为 `Lasso 20-lean`。

## 特征层解释

`20-lean` 和 `25-consensus` 的差异主要来自新增 5 个特征：

- `free_share`
- `pb`
- `ps_ttm`
- `macd_bfq`
- `bias2_bfq`

从结果看：

- 对 LightGBM：新增特征提升 IC，但显著破坏 long-short 稳定性。
- 对 Lasso：新增特征提升 RankIC 和平均收益，但收益波动大幅放大。
- `pb`、`ps_ttm` 属于基本面/估值类特征，当前 artifacts 仍提示 `feature_availability.mode=required_timestamp` 的基本面可用性 warning，因此正式研究前需要 safety-lag 复核。

初步判断：

这 5 个新增特征不应整包纳入。更好的下一步是做 ablation：

1. `20-lean + free_share`
2. `20-lean + macd_bfq + bias2_bfq`
3. `20-lean + pb + ps_ttm`
4. `20-lean + free_share + macd_bfq + bias2_bfq`

这样可以区分到底是基本面估值列导致不稳定，还是技术类冗余特征导致收益路径恶化。

## 效率对比

| 模型 | stage sum | model_fit | evaluate | target_build | feature_validate |
| --- | ---: | ---: | ---: | ---: | ---: |
| LightGBM 20-lean | 574.46s | 286.63s | 105.10s | 82.34s | 39.95s |
| LightGBM 25-consensus | 720.91s | 353.78s | 140.33s | 116.69s | 52.92s |
| Lasso 20-lean | 346.94s | 86.84s | 103.85s | 91.34s | 41.93s |
| Lasso 25-consensus | 404.85s | 102.65s | 110.87s | 106.54s | 56.76s |

效率结论：

- 25-consensus 比 20-lean 明显更慢。
- 对 LightGBM，25-consensus 增加约 `146s` stage sum。
- 对 Lasso，25-consensus 增加约 `58s` stage sum。
- 在表现没有稳定改善的前提下，25-consensus 的额外成本暂时不划算。

## 风险与完整性提示

四个实验仍有共同 caveat：

1. 仍然都是 `Blocked from Level 2`，不能进入组合上线判断。
2. 基本面特征存在可用性 warning：
   - 20-lean：主要是 `pe_ttm`
   - 25-consensus：`pb`、`pe_ttm`、`ps_ttm`
3. 当前结果来自 `exploratory_screening`，适合作为 Level 1 筛选，不是完整研究结论。

## 建议决策

当前建议：

1. 保留 `Lasso 20-lean` 作为线性 champion。
2. 保留 `LightGBM 20-lean` 作为树模型 champion。
3. 暂缓使用 `25-consensus` 作为主力候选。
4. 把 `25-consensus` 拆成更小的增量实验，优先做新增 5 特征的 ablation。
5. 若要继续扩特征，不要再整包增加；每次新增 2-3 个特征，并要求同时改善：
   - RankIC / ICIR
   - 成本后 long-short return
   - long-short IR
   - max drawdown
   - promotion gate 距离

## 一句话结论

`25-consensus` 没有证明自己比 `20-lean` 更适合作为当前 Level 1 baseline；它提高了部分 IC 指标，但牺牲了收益路径稳定性和运行效率。当前应以 `20-lean` 为主线，围绕新增 5 个特征做拆分验证。

