# safe_bfq_35 模型因子特征筛选记录

本记录用于 Level 1 baseline 代表模型选择后的特征降维实验。结论只用于研究筛选，不代表 Level 2 可上线组合。

## 输入依据

- 线性 champion：`stock_lasso_safe_bfq_35`
- 树模型 champion：`stock_lightgbm_safe_bfq_35`
- 树模型稳健性 challenger：`stock_gbdt_safe_bfq_35`
- 原始特征集：`features_safe_bfq_35.parquet` 的 35 个特征
- 共同配置：同一特征文件、价格文件、target horizon、训练窗口、交易成本和 evaluation profile

## 筛选原则

1. Lasso 非零系数给强线性票，但不把它视为最终真理。
2. LightGBM importance 给非线性强度票。
3. GBDT permutation importance 给排序稳健性票。
4. 避免直接取 LightGBM Top 20，必须保留经济分组覆盖。
5. `pe_ttm`、`pb`、`ps_ttm`、`dv_ttm` 有基本面可用性 caveat：相对比较可用，正式研究建议再跑 safety-lag 稳健性。

## 关键信号

Lasso 只留下 4 个非零线性特征：

| 特征 | Lasso 系数方向 | 解释 |
| --- | ---: | --- |
| `turnover_rate_f` | 负 | 换手/活跃度的独立线性信息最强 |
| `atr_bfq` | 正 | 波动/振幅信息在线性模型中仍有效 |
| `turnover_rate` | 负 | 与 `turnover_rate_f` 同组，需关注冗余 |
| `dv_ttm` | 负 | 股息/估值类线性票，但有基本面可用性 caveat |

LightGBM 与 GBDT 同时强支持：

- `topdays`
- `updays`
- `downdays`
- `lowdays`
- `turnover_rate_f`
- `total_mv`
- `atr_bfq`
- `macd_dea_bfq`
- `macd_dif_bfq`
- `trix_bfq`

## 25-consensus

定位：主力降维候选。兼顾 Lasso 稀疏线性票、LightGBM 强度、GBDT 稳健性和经济分组覆盖。

```yaml
feature_columns:
  - turnover_rate_f
  - turnover_rate
  - atr_bfq
  - topdays
  - updays
  - downdays
  - lowdays
  - total_mv
  - circ_mv
  - free_share
  - dv_ttm
  - pb
  - pe_ttm
  - ps_ttm
  - macd_dea_bfq
  - macd_dif_bfq
  - macd_bfq
  - trix_bfq
  - obv_bfq
  - emv_bfq
  - rsi_bfq_24
  - wr_bfq
  - dmi_adx_bfq
  - dmi_pdi_bfq
  - bias2_bfq
```

分组覆盖：

- 估值/股息：`dv_ttm`, `pb`, `pe_ttm`, `ps_ttm`
- 规模/股本：`total_mv`, `circ_mv`, `free_share`
- 交易活跃度：`turnover_rate_f`, `turnover_rate`
- 趋势/动量：`macd_dea_bfq`, `macd_dif_bfq`, `macd_bfq`, `trix_bfq`, `bias2_bfq`
- 波动/幅度：`atr_bfq`
- 超买超卖：`rsi_bfq_24`, `wr_bfq`
- 量价确认：`obv_bfq`, `emv_bfq`
- 路径/趋势状态：`topdays`, `updays`, `downdays`, `lowdays`, `dmi_adx_bfq`, `dmi_pdi_bfq`

## 20-lean

定位：更激进的低冗余版本，用于验证“少而稳”的核心信号是否足够。

```yaml
feature_columns:
  - turnover_rate_f
  - turnover_rate
  - atr_bfq
  - dv_ttm
  - topdays
  - updays
  - downdays
  - lowdays
  - total_mv
  - circ_mv
  - pe_ttm
  - macd_dea_bfq
  - macd_dif_bfq
  - trix_bfq
  - obv_bfq
  - emv_bfq
  - rsi_bfq_24
  - wr_bfq
  - dmi_adx_bfq
  - dmi_pdi_bfq
```

取舍：

- 保留 Lasso 全部 4 个非零特征。
- 保留树模型共同最强的路径状态组。
- 规模组只保留 `total_mv` 与 `circ_mv`，暂时放弃 `free_share`。
- 估值组只保留 `dv_ttm` 与 `pe_ttm`，降低基本面可用性风险暴露。
- 舍弃 `pb`、`ps_ttm`、`bias1_bfq`、`bias2_bfq`、`bias3_bfq` 等更偏模型族或冗余的信号。

## 25-tree-aware

定位：偏非线性增强。保留更多 LightGBM 强特征，同时用 GBDT 和 Lasso 做约束。

```yaml
feature_columns:
  - topdays
  - updays
  - lowdays
  - downdays
  - total_mv
  - atr_bfq
  - pb
  - ps_ttm
  - free_share
  - turnover_rate_f
  - obv_bfq
  - trix_bfq
  - circ_mv
  - turnover_rate
  - pe_ttm
  - macd_dea_bfq
  - macd_dif_bfq
  - bias3_bfq
  - macd_bfq
  - bias1_bfq
  - dv_ttm
  - emv_bfq
  - rsi_bfq_24
  - dmi_adx_bfq
  - dmi_pdi_bfq
```

取舍：

- 基本以 LightGBM Top 20 为骨架。
- 强制加入 Lasso 非零的 `dv_ttm`。
- 用 GBDT 较强的 `rsi_bfq_24`、`emv_bfq`、`dmi_adx_bfq`、`dmi_pdi_bfq` 替代纯 LightGBM 但 GBDT 较弱的尾部特征。

## 暂不纳入

这些特征不是永久剔除，只是在本轮降维中优先级较低：

- `volume_ratio`：LightGBM 与 GBDT 排名都靠后，Lasso 为零。
- `cci_bfq`：三方支持都弱。
- `mfi_bfq`：LightGBM 与 GBDT 都偏弱。
- `rsi_bfq_6`：短周期 RSI 在本轮模型中较弱。
- `mtm_bfq`：树模型支持不足，Lasso 为零。
- `dmi_mdi_bfq`：LightGBM 有一定票，但 GBDT permutation 为零，暂不作为优先项。
- `vr_bfq`：可作为量价确认备选，但当前支持不如 `obv_bfq` 与 `emv_bfq`。

## 推荐验证顺序

1. 先跑 `20-lean + Lasso/LightGBM`，看 RankIC、成本后 long-short 和 drawdown 是否明显恶化。
2. 再跑 `25-consensus + Lasso/LightGBM`，作为主候选。
3. 最后跑 `25-tree-aware + LightGBM/GBDT`，验证非线性增强是否稳定。
4. 三套都必须和原始 `35-core` 对照，不直接替换。
5. 若降维后缺口明显，再从已验证单因子中补 5 个左右方向性 alpha。

