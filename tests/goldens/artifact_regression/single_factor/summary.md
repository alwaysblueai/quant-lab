# 实验摘要：demo_bp_single_factor

## 基本信息

- 因子：`bp`
- 方向：`long`
- 股票池：`demo_universe`
- 调仓频率：`W`
- 目标：`forward_return` / horizon=`5`

## 初筛结论

| Metric | Value |
|---|---|
| Evaluation Profile | default_research |
| Factor Verdict | Promising but fragile |
| Campaign Triage | Drop for now |
| Level 2 Promotion | Blocked from Level 2 |
| Level 1->2 Transition | Inconclusive transition |
| Portfolio Validation | skipped_not_promoted (Not evaluated (not promoted)) |
| Mean Rank IC | 0.065094 (OOS: 0.088228) |
| Mean MI | 0.236611 (OOS: 0.236906) |
| ICIR | 0.217827 (OOS: 0.256815) |
| IC Half-Life | 4.03 |
| Decay vs Rebalance | rebalance=5; ratio=1.24 |
| Mean Long-Short Return | 0.005259 (OOS: 0.004856) |
| Mean Turnover | 0.847403 (OOS: 0.843220) |
| Coverage Mean | 0.998387 (OOS: 1.000000) |
| Capacity | available; upper=7063874.77; adv=119128057.53 |
| Conditional IC | Q5-Q1=0.1476; large-small=-0.0816 |
| 主要诊断 | positive IC and RankIC means; IC and RankIC signs are consistently positive; signal weakens materially in some periods; rebalance cadence may be too slow for IC decay; confidence interval overlaps zero: long-short; apparent edge is weak relative to estimation noise |
| 主要阻断项 | blocked by unstable rolling evidence; blocked by underperformance vs simple momentum/reversal baselines; blocked by sharp IC decay under 1-day execution lag |
| 主要风险 | blocked by unstable rolling evidence; blocked by underperformance vs simple momentum/reversal baselines; blocked by sharp IC decay under 1-day execution lag |

## 产物路径

- 输出目录：`<OUTPUT_ROOT>/demo_bp_single_factor`
