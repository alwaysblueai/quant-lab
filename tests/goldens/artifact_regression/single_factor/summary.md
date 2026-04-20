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
| Campaign Triage | Fragile / monitor |
| Level 2 Promotion | Blocked from Level 2 |
| Level 1->2 Transition | — |
| Portfolio Validation | — |
| Mean Rank IC | 0.076923 |
| Mean MI | 0.230997 |
| ICIR | 0.344757 |
| IC Half-Life | 2.59 |
| Decay vs Rebalance | rebalance=5; ratio=1.93 |
| Mean Long-Short Return | 0.003812 |
| Mean Turnover | 0.848214 |
| Coverage Mean | 0.997059 |
| Capacity | available; upper=4951840.69; adv=84004440.33 |
| Conditional IC | Q5-Q1=0.0159; large-small=-0.1917 |
| 主要诊断 | positive IC and RankIC means; IC and RankIC signs are consistently positive; signal weakens materially in some periods; rebalance cadence may be too slow for IC decay; long-short max drawdown is elevated; factor performance is regime-dependent |
| 主要阻断项 | blocked by unstable rolling evidence |
| 主要风险 | none |

## 产物路径

- 输出目录：`<OUTPUT_ROOT>/demo_bp_single_factor`
