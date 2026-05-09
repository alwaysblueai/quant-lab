# 单因子外部对照审计记录

日期：2026-05-02

## 范围

本记录是一次性审计，不进入默认 CI。目标是用极简 pandas 实现独立复算一个简单 `mom_5d` 因子的 IC、RankIC 与分组收益，并与 `alpha_lab.evaluation` / `alpha_lab.quantile` 的结果做数值对照，降低“实现与测试共用同一错误口径”的风险。

## 对照数据

- 合成面板：8 个资产 x 80 个交易日。
- 因子：`mom_5d = close[t] / close[t-5] - 1`。
- 标签：`forward_return_5 = close[t+5] / close[t] - 1`。
- 分组：`n_quantiles=5`，外部实现按 inventory 记录的 dense-rank 线性映射口径复算。

## 结果

| 项目 | 结果 |
| --- | ---: |
| IC 对照日期数 | 70 |
| RankIC 对照日期数 | 70 |
| 分组收益对照行数 | 350 |
| `mean_ic` alpha_lab | -0.6716026153813786 |
| `mean_ic` external pandas | -0.6716026153813786 |
| `mean_rank_ic` alpha_lab | -0.7040816326530611 |
| `mean_rank_ic` external pandas | -0.7040816326530612 |
| IC 最大绝对差 | 2.220446049250313e-16 |
| RankIC 最大绝对差 | 2.220446049250313e-16 |
| 分组收益最大绝对差 | 0.0 |

## 结论

在该合成 `mom_5d` fixture 上，核心 IC、RankIC 和分组收益与独立 pandas 复算一致，差异处于浮点舍入级别。该对照只证明当前登记口径下的一条简单因子链路一致，不替代默认 CI 中的合成数据层与 golden snapshot。
