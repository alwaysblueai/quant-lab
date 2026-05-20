# signed_jump_neg_5d 研究日志

## 机制定义

`signed_jump` 是一个日内 realized-volatility 非对称特征：

```text
signed_jump = (rv_pos_5m - rv_neg_5m) / rv_5m
```

研究假设是：较高的正向 `signed_jump` 往往对应短周期上行跳跃压力或微观结构层面的过度冲击，后续 1-5 个交易日容易出现均值回复。本因子取 `signed_jump` 的 5 日滚动均值并做符号翻转，因此高因子值对应可做多侧。它不是普通日频 OHLC 反转，而是一个由日内分钟特征派生出的短周期微结构反转信号。

## 为什么适合面试展示

- 它来自日内派生列，能展示 Quant Knowledge -> 机制抽取 -> AlphaLab 后端验证的完整路径，比纯日频 OHLC 因子更能体现系统价值。
- PIT 假设清楚：`signed_jump` 只使用当日 09:30-15:00 分钟线，因子在 `t` 日收盘后可用，5 日 forward-return 标签由标准 single-factor pipeline 处理。
- 已经有完整 `default_research` 运行结果，包含 artifact hash、split contract、baseline comparison、IC decay、lag sensitivity、tradability checks、random baseline 等诊断。
- 系统没有盲目 promote：即使 IC 证据较强，promotion gate 仍然因为覆盖率口径和 rolling evidence 保守拦截。这正好体现研究纪律。

## v1 / default_research 证据

Case: `configs/real_cases/single_factor/signed_jump_neg_5d_v1_default.yaml`

Output: `outputs/real_cases/signed_jump_neg_5d_v1_default/`

`metrics.json` 关键证据：

- Full-sample mean RankIC: `0.0502`；OOS mean RankIC: `0.0628`。
- OOS mean IC: `0.0416`；OOS IC t-stat: `6.90`。
- OOS RankIC positive rate: `70.4%`。
- OOS rolling RankIC positive share: `97.7%`。
- OOS mean long-short return: `0.00818`；OOS long-short hit rate: `64.5%`。
- OOS rank-IC IR: `0.442`；OOS long-short IR: `0.274`。
- Random baseline observed z-score: `46.06`；p-value: `0.0196`。
- Baseline suite best mean IC: `0.0395`；factor mean IC advantage: `0.0020`。

主要限制：当前 gate 给出 `Blocked from Level 2`。主要原因不是预测关系不存在，而是 evaluation coverage ratio 和 rolling evidence 仍需更干净的复核。

## v2 精细化目标

现有文件：`factor_signed_jump_neg_5d_v2.parquet`

已检查元信息：

- Rows: `5,373,753`。
- Assets: `5,388`。
- Date range: `2020-01-06` to `2026-04-15`。
- Columns: `date`, `asset`, `factor`, `value`。

v2 不改变因子定义，而是用已经存在的 v2 precomputed factor 文件建立一个更适合展示的标准 case：样本从 2020 年开始，避开早期日内特征面板覆盖较不稳定的阶段。目标不是通过调参过拟合，而是让展示口径更清楚：

1. 保持同一机制和公式。
2. 保持 weekly rebalance 与 5 日 forward-return target。
3. 继续走同一套 `default_research` gate。
4. 对比 v2 是否改善 coverage、rolling stability 和 promotion blockers。

## v2 / default_research 证据

Case: `configs/real_cases/single_factor/signed_jump_neg_5d_v2.yaml`

Output: `outputs/real_cases/signed_jump_neg_5d_v2/`

Run timestamp: `2026-05-19`

验证结果：

- `validate-draft-factor` 通过。
- Warning: `provenance_missing`；这是 pre-protocol research factor，后续如果重新从 Stage 0 -> Stage 2 流程生成候选，应补齐 `provenance`。

Artifact 审计：

- `run_manifest.json` 和 `factor_definition.json` 都包含 `custom_factor_source.path`。
- `factor_json_sha256`: `583ccc6a7b4a5bedca15c9d7ccd72a872afcf7f7b818788202be540821d46294`。
- `custom_factor_source.code_sha256`: `4f23e658afd9300f582b4ec483c23f00bda0e19c0a067cf02d5855b26187c68b`。
- Integrity checks: `10` pass, `3` warn, `0` fail。

`metrics.json` 关键证据：

- Full-sample mean RankIC: `0.0492`；OOS mean RankIC: `0.0628`。
- OOS mean IC: `0.0416`；OOS IC t-stat: `6.90`。
- OOS RankIC positive rate: `70.4%`。
- OOS rolling RankIC positive share: `97.7%`。
- OOS subperiod IC positive share: `100%`。
- OOS subperiod long-short positive share: `100%`。
- OOS mean long-short return: `0.00818`。
- OOS long-short IR: `0.274`。
- Full-sample evaluation coverage ratio 从 v1 `0.581` 提升到 v2 `0.658`。

解释：

v2 改善了展示样本的干净程度，并保留了较强 OOS 证据，但仍然没有通过 Level 2 promotion。这个结果是理想的研究姿态：它是一个 promising Level 1 candidate，而不是已经可以进入组合层的正式因子。剩余 blockers 是 thin coverage、weak single-case verdict 和 conservative gate 下的 unstable rolling evidence。

## 面试展示口径

这个因子适合用来展示三件事：

- 知识抽取：机制不是“LLM 凭空发明 alpha”，而是一个关于 realized-volatility signed jump 非对称的微结构假设。
- 执行纪律：idea 被转化成 custom factor，经过 validator、标准回测、split、PIT/integrity 检查和 hash-backed artifact 审计。
- 研究成熟度：结果有吸引力，但系统仍把它留在 research candidate 阶段。下一步应做 neutralization、coverage 复核和 Level 2 portfolio validation，而不是直接 promote。
