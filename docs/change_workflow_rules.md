# 改动流程规则（alpha-lab + model-lab）

本文是加功能 / 改函数 / 改口径前必读的"流程规则"文档。它不重述 [`extension_points.md`](extension_points.md) 与 [`developer_guide.md`](developer_guide.md) 的具体配方，而是规定**改什么先看哪份文档、按什么顺序落地、什么时候可以合 PR**。

适用范围：`src/alpha_lab/` 全部模块，包括单因子（alpha-lab 主线）和模型因子（model-lab，即 `src/alpha_lab/{model_factor,real_cases/model_factor}/`）。

---

## 1. 审查现状（截至 2026-05-02）

两条线已分别完成一文件一份的静态审查，权威产出：

| 链路 | 报告 | 状态 |
| --- | --- | --- |
| 单因子 | [`single_factor_metric_curve_review.md`](single_factor_metric_curve_review.md) | 全部"通过 / 口径需说明"；主单因子 + 小截面 + 高换手三套 golden，覆盖 13 个 CSV + 核心 JSON hash |
| 模型因子（model-lab） | [`model_factor_metric_curve_review.md`](model_factor_metric_curve_review.md) | 共享路径继承单因子；独有部分有 2×P1 schema 漂移 + 数条 P2/P3，记录未修 |

口径权威文档（**改指标前必先动这两份**）：
- [`single_factor_metric_inventory.md`](single_factor_metric_inventory.md) — 单因子产物地图、列对账、状态表
- [`model_factor_metric_inventory.md`](model_factor_metric_inventory.md) — 模型因子独有产物 + 一致性 invariant

---

## 2. 三类扩展的最小改动路径

只走 [`extension_points.md`](extension_points.md) 列出的 seam，不要自己造接口：

| 想做什么 | seam | 改动量 | 不要碰 |
| --- | --- | --- | --- |
| 加新因子 | `factor_recipe.py:73` `factor_registry.register_factor` | 写一个 `(prices, **params) → long-form DataFrame` 函数 + decorator + YAML 引用 | pipeline / CLI / web_unified |
| 加新指标 | `evaluation.py` 加 helper + 写到既有 `metrics.json` 的 namespaced key | 计算函数 + namespaced key + 单元测试 + golden 重生成 | **`ExperimentSummary` dataclass**（额外字段不要加） |
| 加新 evaluation profile | `research_evaluation_config.py:650` 的 `_PROFILE_BUILDERS` | 一个 builder 函数 + 一行字典 entry | pipeline / CLI / web（自动看到） |

> Factor 是真正的 runtime registry；metric 与 profile 都是**源码级编辑**，没有外部插件路径。  
> 如果你的需求不在这三类里（例如改训练循环、改 split 语义、改 artifact schema），跳到第 3 节。

---

## 3. 改动前要读哪份文档

按改动范围对号入座：

| 改动范围 | 必读 → 必改 |
| --- | --- |
| IC / RankIC / 分组收益 / NAV / turnover / coverage 任何**口径** | `single_factor_metric_inventory.md` → 改代码 → 改 golden hash |
| model-factor 训练 / 模型选择 / 诊断口径 | `model_factor_metric_inventory.md` → 改代码 → 改 cross-file invariant 测试 |
| split / embargo / 分桶 | `single_factor_metric_curve_review.md` "现状已闭环"段（避免重做已修过的坑） |
| 加诊断 → Tier 1 / Tier 2 边界 | [`research_playbook.md`](research_playbook.md) — **默认 Tier 2**，除非"比较一批因子并影响 continue/stop verdict" |
| Prompt / research workflow 语义（idea explorer） | [`research_workflow.md`](research_workflow.md)（stage × mode 双正交轴是权威） |
| 性能优化 | 先用 `_stage()` timing 拆瓶颈，区分**计算 / 诊断 / 重复**三类对症；不要按直觉/Qlib 类比排优先级 |
| UI / 前端 | [`single_factor_metric_curve_review.md`](single_factor_metric_curve_review.md) "Dashboard 口径"段 + [`model_factor_metric_curve_review.md`](model_factor_metric_curve_review.md) "Web UI"段 |

---

## 4. 实施顺序（每个 PR 都按这个）

1. **先改 inventory**——口径文档先动，PR 描述里明确"哪一行被改了"。  
   口径文档与代码不同步是过去最大的漂移源；先改文档强迫先想清楚。
2. **再改代码**——遵守 [`AGENTS.md`](../AGENTS.md) §3 研究完整性约束：
   - 无未来数据使用
   - 无 label-feature leak
   - 显式 temporal alignment + split discipline
   - PIT / as-of / 跨频率正确
3. **同步改测试**：
   - artifact 列变更 → 改 `tests/test_artifact_golden_regression.py` 或 golden hash
   - 跨文件不变量（如 `purge_days == label_horizon`、`metrics.model_family == model_selection.*`）→ 在 inventory "一致性 invariant" 小节登记 + 加 assert
4. **跑本地 gate**：
   - `make check`（lint + typecheck + test）必须过
   - UI 改动：`make test` + 双场景 browser smoke（不是只跑 fixture）
5. **golden 重生成**——`ALPHA_LAB_UPDATE_GOLDENS=1` 只在字段稳定后跑一次；不要为修 lint 反复生成。

---

## 5. Hard Rules

来自 `AGENTS.md` 和已落实的 feedback：

- **不引入 Level 3 语义到核心**：execution realism / fill simulation / adapter parity 类的东西放 `experimental_level3` namespace；不进入默认 CLI / 默认 docs / 默认测试。
- **后端先行**：评价 UI 缺数据走 `artifact_contracts.py` + manifest `not_emitted_v1`；**不在 JS 里伪造指标，也不写 TODO 占位**。
- **最小手术**：bug fix 不附带 cleanup；不为假想需求抽象；三行相似 > 早抽象；不写半成品。
- **新诊断默认 Tier 2**：除非满足"比较一批因子 → Tier 1"判定。
- **空值语义保留**：CSV 空单元 / `NaN` / `"N/A"` / `None` 不能在前端被 `Number("")` 解析成 0；`fmtNum` / `toNum` / `irOfPoints` 是过去已修过的痛点。
- **`std(ddof=1)` 一致**：后端用样本标准差，前端 `irOfPoints` 必须用 `(n - 1)` 分母。
- **未年化 IR 不要叫 Sharpe**：年化 Sharpe 走 `artifact_enrichment.build_backtest_summary_payload`；未年化 IR 单独命名（`ic_ir`、`rank_ic_ir`、`long_short_ir`）。

---

## 6. Model-lab 当前已知未修项（下次动 model-lab 时优先做）

来自 [`model_factor_metric_curve_review.md`](model_factor_metric_curve_review.md) "缺口清单"：

- **P1**：`_build_model_selection_payload` 列名是 `candidate_id` / `candidate_family`；不变量测试期望 `selected_candidate_id` / `selected_candidate_family`。golden run 全是 `status=disabled` 把 else 分支掩盖了。
- **P1**：`artifacts.py:689` enabled-but-empty 写 `status="not_available"`，inventory + 测试枚举只接受 `{disabled, no_candidates, ok}`，`no_candidates` 永远不被生成。
- **P1**：缺 enabled selection 的 golden fixture（上面两个 P1 没被自动捕获就因为这个）。
- **P2**：`sign_stability` 分母（`signed_available_count` vs inventory 的 `n_model_versions`）/ `selection.metric` 文档漂移（`long_short_sharpe` 文档存在代码不接受）/ `--screening-retrain-every-n-dates` 缺 profile 守卫 / dataset_cache 写非原子 / empty CSV fallback header 错乱。
- **P3**：score factor `(date, asset)` 唯一性靠 `_coverage_by_date.drop_duplicates` 兜底，缺上游 hard assert / model-lab UI 缺 selected family 高亮 + k-fold verdict 颜色。

---

## 7. 一句话总结

> **改之前先动 inventory，改之后跑 golden + invariant 测试，别绕开 [`extension_points.md`](extension_points.md) 列的三类 seam 自己造新接口。**
