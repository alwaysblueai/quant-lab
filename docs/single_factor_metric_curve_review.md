# 单因子指标与曲线绘制审查报告

本文基于 [single_factor_metric_inventory.md](single_factor_metric_inventory.md) 对单因子评估链路做静态审查，覆盖“指标计算 -> artifact 写出 -> research tearsheet/PDF -> metrics dashboard 展示”。审查目标是确认口径一致、时间对齐可追溯、split/embargo 处理明确，并识别未展示或待测试的缺口。

边界声明：本轮只审查 Level 1/2 单因子评估、报告 artifact、research tearsheet 与 metrics dashboard。`next_open`、`tradability`、`capacity`、`cost` 维持现有研究诊断范围——已存在的 `next_open_*` sensitivity 在本轮做了 split 口径修复，但不新增新的诊断语义（例如 `vwap_*` 未启用）。

## 总体结论

| 结论项 | 状态 | 说明 |
| --- | --- | --- |
| 后端指标计算链路 | 通过 | `evaluation.py`、`quantile.py`、`decay.py`、`turnover.py`、`costs.py` 的核心口径与 inventory 一致；IC/IR 使用 `std(ddof=1)`，分组收益按 `(date, asset)` 对齐。 |
| Artifact 契约 | 通过 | `single_factor/artifacts.py` 会写出 `metrics.json`、IC/decay/autocorr/rolling/group/turnover/coverage/lag sensitivity/random baseline/daily PnL 等 CSV，并已纳入 manifest required list。 |
| Split 与 embargo | 通过 | 后端打包时 IC、rolling、group returns、turnover 会 drop `EMBARGO`，coverage 保留 `EMBARGO` 作为覆盖率审计；dashboard 已优先消费 CSV 内的 `split_phase`。 |
| Dashboard 口径 | 口径需说明 | dashboard 的 OOS summary 仍是展示层重算，但已按 `split_phase`/结构化 split contract 分段，ICIR 使用 `std(ddof=1)`，空 CSV 单元不再误画为 0。 |
| PDF/tearsheet 图表 | 通过 | `research_tearsheet.py` 从 artifact CSV/JSON 构建 NAV、IC、分组累计收益、分组均值、rolling、IC decay、IC 分布、turnover、coverage 图表。 |
| 未展示诊断 | 未展示但 artifact 存在 | dashboard 当前主要展示 RankIC 日线与分组均值，turnover/coverage 仅进 summary strip；decay、autocorr、rolling、daily PnL、capacity 仍主要依赖 CSV/PDF/JSON。 |

## 审查矩阵

| 类别 | 状态 | 计算口径核查 | Artifact/PDF 核查 | Dashboard 核查 |
| --- | --- | --- | --- | --- |
| IC/RankIC/MI | 口径需说明 | `compute_ic_summary` 使用有效日期等权均值与 `std(ddof=1)`；`single_factor/evaluate.py` 将 IC、RankIC、MI 合并进 `ic_timeseries`；缓存式 `merged_pairs` 入口已拒绝重复 `(date, asset)`。 | `ic_timeseries.csv` 写出 `ic`、`rank_ic`、`mutual_information`；PDF 优先绘制 `rank_ic`，无有效 RankIC 时退到 `ic`；常数分布也会生成可渲染 histogram bin。 | dashboard 优先画 RankIC；当 RankIC 无有效点且 IC 有效时，标题、series label 和 summary 第一项会同步退到 IC。 |
| 分组收益 | 通过 | `quantile_returns_df` rename 为 `group_returns`，桶内等权均值；小截面会降级到有效桶数，单桶多空为 NaN；缓存式 `merged_pairs` 入口已拒绝重复 `(date, asset)`。 | `group_returns.csv` 持久化；PDF 同时绘制分组累计收益和分组均值柱状图。 | “Mean group return” 是时间均值柱状图，未冒充累计净值。 |
| 多空与 NAV | 通过 | `long_short_return` 为逐日最高已占用桶减最低已占用桶；IR 未年化。 | `artifact_enrichment.py` 用同一逐日 top-bottom 口径构建 NAV，并按 `max(rebalance_step, label_horizon)` 非重叠采样计算年化统计与 `nav_points`；`backtest_result.json` 提供 NAV。 | dashboard 当前不画 NAV，需在报告中标为未展示但 artifact/PDF 存在。 |
| Rolling stability | 通过 | rolling 稳定性由 `experiment` 产出，窗口前自然 NaN。 | `rolling_stability.csv` 写出；PDF 绘制 rolling IC/RankIC。 | dashboard 未绘制 rolling 曲线。 |
| Turnover 与费后 | 口径需说明 | `long_short_turnover` 首期 NaN；`adjusted_return = long_short_return - cost_rate * turnover`，首期费后收益也为 NaN；费后 merge 已要求 `(date, factor)` 一对一。 | `turnover.csv`、`daily_pnl_attribution.csv` 写出；split 场景下 turnover artifact 已按 IS/OOS 分段内换手拼接，避免跨 EMBARGO 计算 OOS 首期换手；PDF appendix 绘制 turnover 曲线。 | summary 读取 `mean_long_short_turnover`；若有 `split_phase`，OOS 均值按 OOS 行重算并跳过 NaN。 |
| Coverage | 通过 | coverage 按 eligible universe 统计因子与标签有效覆盖。 | `coverage.csv` 保留 `EMBARGO`，用于审计 split 期间覆盖率；PDF appendix 绘制 coverage 曲线。 | summary 读取 `coverage_mean`；若有 `split_phase`，OOS 均值按 OOS 行重算。 |
| IC decay/autocorr | 未展示但 artifact 存在 | decay horizons 为 `{1,2,3,5,10,20,target}`；autocorr lags 为 `(1,2,3,5,10)`；RankIC/autocorr 快路径只在完整有限面板启用，有缺失时回到逐日交集排名。 | `ic_decay.csv`、`factor_autocorrelation.csv` 写出；PDF 绘制 IC decay，autocorr 当前以 CSV 诊断为主。 | dashboard 未绘制 decay/autocorr。 |
| Split 展示 | 通过 | 后端 `_with_split_phase` 已写出 `IS/OOS/EMBARGO`，并按曲线类型决定是否 drop embargo。 | artifact 中有 `split_phase` 列时可直接追溯。 | dashboard 优先使用 CSV `split_phase`，再 fallback 到结构化 split contract/legacy 描述。 |
| 缺失/异常展示 | 通过 | 后端多数函数对空表、常数因子、小截面返回 NaN 或空表。 | writer 会写空 CSV；PDF 图表在无有效点时跳过；IC 图无有效 RankIC 时退到 IC。 | dashboard 解析阶段保留 NaN/空值语义，不把空 CSV 单元误画成 0。 |

## 关键发现

### 证据索引

| 链路 | 代码位置 | 本轮核查结论 |
| --- | --- | --- |
| IC/RankIC/MI 基础计算 | `src/alpha_lab/evaluation.py` | 每日截面计算后用有效日期汇总；`ic_ir` 为 `mean / std(ddof=1)`，未年化。 |
| 分组收益与多空 | `src/alpha_lab/quantile.py` | 因子和标签按 `(date, asset)` 严格合并；分组收益为桶内均值，多空为 top bucket minus bottom bucket。 |
| IC decay 与 autocorr | `src/alpha_lab/decay.py`, `src/alpha_lab/real_cases/single_factor/evaluate.py` | horizons/lags 由单因子 evaluate 固定注入，decay 产出 `ic_decay.csv`，autocorr 产出 `factor_autocorrelation.csv`。 |
| Turnover 与 cost | `src/alpha_lab/turnover.py`, `src/alpha_lab/costs.py` | 首期 turnover 为 NaN；线性费后收益保留 NaN，不把首期成本填 0。 |
| 标签敏感性 | `src/alpha_lab/labels.py` | `next_open`/`vwap` 使用 `close[t+h] / exec_price[t+1] - 1`，属于研究诊断口径。 |
| Split 包装 | `src/alpha_lab/real_cases/single_factor/evaluate.py` | `_with_split_phase` 标注 `IS/OOS/EMBARGO`；coverage 保留 embargo，其他报告曲线 drop embargo。 |
| Artifact 写出 | `src/alpha_lab/real_cases/single_factor/artifacts.py` | 主指标、曲线 CSV、daily PnL、tearsheet JSON/PDF 都有固定路径写出。 |
| Summary/card 文档 | `src/alpha_lab/real_cases/single_factor/templates.py` | `summary.md` 与 `experiment_card.md` 的核心 split 指标使用 full + OOS 并列展示；Coverage 已纳入同一展示口径。 |
| Research validation package | `src/alpha_lab/reporting/research_validation_package.py` | Level 1/2 validation package 以 key/value 方式透传核心研究指标；未发现把未年化 IR 命名为 Sharpe 的新增问题。 |
| PDF 图表 | `src/alpha_lab/reporting/research_tearsheet.py` | 从 artifact CSV/JSON 构建 NAV、IC、分组、rolling、decay、turnover、coverage 图表；空数据跳过图表。 |

### 通过

- 指标计算的主路径保持同 `(date, asset)` 对齐：IC/RankIC/MI、分组收益、多空收益与 turnover 都没有发现跨日期 join。
- 后端 split 包装符合 inventory：`ic_timeseries`、`rolling_stability`、`group_returns`、`turnover` drop `EMBARGO`；`coverage` 保留 `EMBARGO`。
- 年化统计没有和未年化 IR 混用：`ic_ir`、`rank_ic_ir`、`long_short_ir` 保持未年化；`artifact_enrichment.py` 的 `annualized_return`、`annualized_volatility`、`sharpe` 基于非重叠多空收益另算。
- `research_tearsheet.py` 的图表覆盖完整，已能从 artifact 构建 NAV、IC、quantile cumulative、rolling、decay、turnover、coverage。

### 口径需说明

- `ic_timeseries.csv` 同时包含 `ic`、`rank_ic`、`mutual_information`；PDF tearsheet 选择 RankIC 作为主图是合理默认，fallback 到 `ic` 时标题和 series label 已同步变更。
- `capacity_estimation.csv`、`daily_pnl_attribution.csv`、`factor_autocorrelation.csv` 目前更像审计 CSV，未默认进入主图序列。

### 现状已闭环的口径与修复

以下条目是本轮静态审查时核对到的当前状态：部分是这一轮新做的修复，部分是更早已经落地、本轮验证仍然有效的口径。统一作为现状快照记录，不区分"新 fix"与"既有"。

- `artifact_enrichment.py` 的 `backtest_result.json.summary.sharpe` 已改为优先使用年化统计 Sharpe，不再 fallback 到未年化 `long_short_ir`。
- `single_factor/artifacts.py` 的 bundle required list 已补齐 `lag_sensitivity.csv`、`random_baseline_null.csv`、`daily_pnl_attribution.csv`，避免实际写出的审计 artifact 未被 manifest 必备清单覆盖。
- `evaluation.py` 的 mutual information 分箱回填已修复：rank-quantile bins 现在按有效样本位置回填，MI 不再对输入行顺序敏感。
- dashboard 已修复空 CSV 单元被 `Number("")` 解析为 0 的问题，RankIC/IC、turnover、coverage 的缺失值会保持缺失。
- dashboard 的 IC/RankIC 曲线已改为按整条序列选择一个数据列：RankIC 有有效点时只画 RankIC，只有 RankIC 全空才整体退到 IC，避免同一条线混用两种指标；OOS ICIR 只从 `ic` 列重算。
- dashboard 的主指标 summary 已随 IC/RankIC fallback 同步切换：RankIC 全空时显示 Mean IC，并使用 `mean_ic` 作为 overall。
- dashboard 已优先消费 artifact CSV 的 `split_phase`，并在缺失时 fallback 到结构化 split contract/legacy 描述。
- dashboard OOS ICIR 已改为样本标准差口径 `std(ddof=1)`。
- dashboard 的解析/聚合逻辑已抽到 `dashboardSeries.ts`，并补轻量 Node 逻辑测试，覆盖空 CSV 单元、RankIC->IC fallback、`split_phase`/EMBARGO、OOS ICIR 与 group mean 聚合。
- PDF/tearsheet 的 IC 时间序列与分布图已在 RankIC 全空时退到 IC。
- PDF/tearsheet 的 IC 分布图已修复常数序列零宽度 bin，避免 histogram payload 有数据但 PDF 显示“图表数据不足”。
- PDF/tearsheet 的指标 alias fallback 已修复：主字段为 NaN 时会继续 fallback 到 `coverage_by_date_summary` 等备用来源，不再把 NaN 当成有效命中。
- `evaluation.py` 的 supplied `merged_pairs` 入口已补重复 `(date, asset)` 防护，IC、RankIC、MI 与随机 baseline 不再接受会静默放大截面权重的预合并缓存。
- `next_open`/`vwap` 标签已补闭式测试，锁定 `close[t+h] / exec[t+1] - 1` 的对齐口径。
- `next_open` execution-price sensitivity 已修复 split 口径：有 split contract 时只在 OOS 日期重算，并补 IS/EMBARGO 反向、OOS 正向的端到端合成 case。
- reporting 层费后 summary 已补测试，锁定首期 NaN turnover 不参与 `mean_cost_adjusted_long_short_return`。
- `daily_pnl_attribution` 已修复首期 turnover NaN 被当成 0 成本的问题，费后 `net` 与 `daily_pnl_net_mean` 现在和 cost-adjusted return 的有效样本口径一致。
- `turnover.csv` 已修复 split 场景下先算 full-sample turnover 再 drop `EMBARGO` 的口径问题；IS/OOS 现在各自分段计算后拼接，OOS 首期 turnover 保持 NaN，并已和 `daily_pnl_attribution.csv` 的 cost/net 数组做端到端对账。
- `quantile_returns(..., merged_pairs=...)` 已补重复 `(date, asset)` 防护，避免预合并缓存把同一资产重复计入分组均值。
- `cost_adjusted_long_short` 已补必需列校验和重复 `(date, factor)` 防护，避免收益/换手 merge 静默 fan-out。
- `daily_pnl_attribution` 已改为逐日取最高/最低已占用分桶，和 `long_short_return` 的小截面口径一致，不再用全样本最大分桶导致部分日期被丢弃。
- `backtest_result.json` 的 NAV/年化统计已改为逐日取最高/最低已占用分桶，和 `long_short_return`、`daily_pnl_attribution` 的小截面口径一致。
- `decay.py` 的 IC decay RankIC 与 factor autocorr 快路径已收紧为完整有限面板专用；高覆盖但有缺失时回到逐日交集排名，避免“先全截面排名再取交集”的 Spearman 漂移。
- `summary.md` 与 `experiment_card.md` 的 `Coverage Mean` 已改为 full + OOS 并列展示，避免 split run 中只显示裸 `eval_coverage_ratio_mean` 而隐藏 full/OOS 差异。
- `grouped_evaluation.py` 的条件 IC helper 已补单因子/单标签名校验，避免多因子输入在日期/资产不重叠时被静默混成一张条件 IC 摘要。
- fast-screen profile 的 `metrics.json` compact 白名单已补结构化 coverage 标量及 full/IS/OOS 版本，不再只保留 `coverage_summary` 文案。
- `summary.md` 与 `experiment_card.md` 已改为在 portfolio validation 和 Level 1->2 transition 指标写回后渲染，避免卡片显示 `—` 而 `metrics.json` 已有真实状态。
- `experiment.py` 的 `eval_coverage_ratio_*` 已补全截面因子缺失日处理：只要该日期有可用 label，就按 0 覆盖纳入均值/最小值，避免 coverage 被“只统计有有效样本日期”高估。
- 单因子端到端已补 `EMBARGO` 反转合成 case，验证核心 OOS metrics 不含 `EMBARGO`，同时确认 `coverage.csv` 保留 `EMBARGO` 用于覆盖率审计。
- 已补 5 资产 x 30 天 inventory fixture，对齐 `decay.py` target horizon IC decay 行与 `evaluation.py` 的 IC/RankIC/IR/n_dates 直接计算路径，并同时覆盖 cached labels 与 dense fast path。
- 已补 research tearsheet payload 图表输入测试，从 artifact CSV 独立验算 IC/cumulative IC、long-short NAV、group mean bar、turnover、coverage 的 chart input 数组。
- 已补 research tearsheet 完整图表输入 fixture，覆盖当前 PDF 渲染的 9 类图表 payload：NAV、IC+cumulative、分组累计、分组均值、rolling、decay、IC 分布、turnover、coverage（对应下方对账表中已可视化的 7 行 inventory，加 `IC Distribution` 与 `IC + Cumulative IC` 两个组合视图）。
- 单因子 golden regression 已补 canonical JSON artifact 与 `csv_snapshot.json`：锁定 run manifest、factor definition、signal validation、portfolio recipe、backtest result、research tearsheet，以及 13 个 CSV artifact 的列、行数和规范化内容 hash。
- 已补小截面/常数因子端到端 artifact smoke：`n_quantiles=5` 且每日仅 1/2/4 个有效资产时，常数因子只落到单一分组，不产出伪造 long-short、NAV 或 daily PnL。
- 已补小截面/常数因子 artifact golden：固定 `metrics.json`、`backtest_result.json`、`research_tearsheet.json` 与 13 个 CSV artifact hash，覆盖空 daily PnL、空 rolling、单桶 group returns、coverage 保留 split 的异常场景。
- 已补高换手/费后 artifact golden：用日期间排名翻转的 deterministic fixture 触发持续换手，固定 `turnover.csv`、`daily_pnl_attribution.csv` 与核心 JSON/CSV hash，避免 fee/net 数组漂移。
- 已留一次性外部对照审计记录：`docs/audit/single_factor_external_crosscheck.md` 用极简 pandas 复算 `mom_5d` 的 IC/RankIC/分组收益，最大差异为浮点舍入级别。

## 缺口清单

| 优先级 | 缺口 | 影响面 | 复现数据/触发条件 | 建议验证位置 |
| --- | --- | --- | --- | --- |
| P1 | `vwap` execution-price sensitivity metrics 未产出 | `labels.forward_return(..., execution_price_mode="vwap")` 公式已锁定；当前单因子 metrics 只产出 `next_open_*` 诊断，不产出 `vwap_*` 诊断 | 如团队需要 vwap 作为研究诊断，手工设定 `close/open/vwap`，补 `vwap_mean_ic` 等指标与端到端 case | 先记为未产出；若启用，放在 `evaluate.py` execution-price sensitivity tests |
| P1 | dashboard 视觉 QA 仍缺真实 run 验证 | parser/helper 已有轻量逻辑测试；仍需确认真实页面布局和术语不误导 | 带 `split_phase` 的真实 run，包含 RankIC 分段、group bar、summary strip、NaN fallback | 本地 dashboard/PDF 人工验收或 Playwright 视觉 smoke |
| P1 | 更多异常场景 golden 覆盖不足 | 主 deterministic run、小截面/常数因子异常 run、高换手/费后 run 已覆盖 canonical JSON/核心 JSON 和 13 个 CSV artifact hash；全空 CSV、极端缺失等异常仍主要靠 focused smoke/单元测试 | 全空 CSV、极端缺失等场景单独保存核心 artifact hash | `tests/goldens/artifact_regression/` |

## 建议测试实施顺序

1. Artifact golden：主单因子 golden 已锁定 canonical JSON、summary、Level 2 package 与 13 个 CSV 文件的列、行数、内容 hash；小截面/常数因子异常 golden 与高换手/费后 golden 已锁定核心 JSON 与 13 个 CSV hash；后续补全空/极端缺失等异常 fixture。
2. 端到端合成 case：split/embargo 信号反转、`next_open` OOS sensitivity、daily PnL 费后样本数与 turnover OOS 首期 NaN 已覆盖；`vwap` 诊断目前记录为未产出。
3. 图表输入验算：dashboard parser 已覆盖关键 split/NaN/fallback 聚合逻辑；research tearsheet 当前 9 类图表已做 CSV/JSON fixture 输入数组比较，不比较图片像素。
4. 视觉 QA：选一个带 split 的真实 run 打开 dashboard/PDF，人工确认 RankIC 分段、group bar、summary strip、NaN fallback 与术语不误导。
5. 外部对照：`mom_5d` 的 IC/RankIC/分组收益已保存一次性 pandas 对照记录，不加入默认 CI；后续如改分桶口径再重跑。

## 本轮验证记录

- Python 定点回归通过：主单因子 golden、小截面/常数因子 golden、高换手/费后 golden、split/embargo、`next_open` OOS sensitivity、turnover/fee、小截面 smoke、inventory fixture、tearsheet chart input 共 9 项；另有完整 tearsheet chart payload fixture 单独通过。
- Model-Lab1 定点回归通过：`run_model_factor_case` bundle smoke 与 model-factor artifact golden 均通过；golden 固定核心 JSON、Level 2 package、训练日志/训练指标/feature OOS IC/purged k-fold 以及 IC/group/turnover/coverage 等 13 个 CSV artifact hash。
- Model-Lab1 overview 视觉 smoke 通过：复用本地 `http://127.0.0.1:8766` 服务，`scripts/smoke_model_lab_overview_fixtures.mjs` 在 desktop/mobile 视口检查 `strong_skipped_extreme_nav` fixture，未发现 console error、水平溢出、文字溢出、空模块或 `NaN`/`undefined` 占位符。
- `ruff check` 通过，`git diff --check` 通过。
- dashboard parser 逻辑测试已用 Codex bundled Node 跑通，覆盖空 CSV 单元、RankIC->IC fallback、`split_phase`/`EMBARGO`、OOS ICIR 与 group mean 聚合。
- dashboard TypeScript 编译通过；Vite build 因当前 `node_modules` 为 WSL/Linux optional dependency 形态、但可用 Node runtime 为 Windows 版，Rollup 缺 `@rollup/rollup-win32-x64-msvc` 而未完成。该项属于本地工具链架构不匹配，仍需在一致的 Node/node_modules 环境中补跑。

## Model-Lab1 同步审查

Model-Lab1 的模型因子回测后端复用 `evaluate_single_factor_case` 与共享 artifact enrichment，因此本轮单因子链路中关于 `(date, asset)` 对齐、`split_phase`、`EMBARGO` drop/保留策略、非年化 IR 与年化 Sharpe 分离、非重叠 NAV 年化统计等修复，会同步覆盖模型因子 artifact。

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| 后端回测 artifact | 通过 | `src/alpha_lab/real_cases/model_factor/pipeline.py` 调用 `evaluate_single_factor_case`，`model_factor/artifacts.py` 使用共享 `build_backtest_summary_payload` 写 `backtest_result.json`。 |
| Split 展示 | 通过 | model-lab1 artifact 已写出 `split_phase`；`web_model_lab.html` 优先使用行内 `split_phase`，再 fallback 到结构化 split contract。 |
| 空值展示 | 通过 | `web_model_lab.html` 的 `fmtNum`、`toNum` 与 backtest NAV 解析已保留空字符串/`NaN`/`N/A` 缺失语义，不再把空 CSV/JSON 单元画成 0。 |
| OOS IR 重算 | 通过 | `web_model_lab.html` 的 `irOfPoints` 已改为样本方差 `/ (n - 1)`，和后端 `std(ddof=1)` 口径一致。 |
| Model artifact golden | 通过 | `tests/goldens/artifact_regression/model_factor/` 固定 `metrics.json`、`run_manifest.json`、`backtest_result.json`、`research_tearsheet.json`、model definition/selection/feature manifest、Level 2 package，以及 13 个 CSV artifact hash。 |
| Overview 视觉 QA | 通过 | 已用 fixture smoke 覆盖 desktop/mobile、chart card 渲染、文本/布局溢出、占位符与 console error；截图/cache 为本地临时 QA 产物，不纳入版本交付。 |
| Web 路径 dataset cache 复用 | 通过 | `run_model_factor_case` 新增 `cache_root_dir` 参数，CLI 暴露 `--cache-root-dir`；`web_unified.py` 的 model-factor subprocess 现在显式传 `<base_root>/_model_factor_shared_cache`，避免每个 web run 在 `_web_runs/<run_id>/_model_factor_cache/` 下重复落 ~4.5GB 特征矩阵。CLI 默认行为不变（cache 仍在 `output_dir.parent / _model_factor_cache`，跨 case 共享）。短期止血脚本 `scripts/gc_web_runs.py --keep-last/--older-than [--apply]` 仍保留，用于清理已积累的旧 run。 |

剩余风险：model-lab1 已完成 fixture 视觉 QA，但还未用真实生产 run 做人工复核；后续可选一个近期模型因子 run 复看 RankIC 分段、NAV/回撤、coverage strip、training health 和极端 NaN fallback；同时验证一次完整 web → web 同 spec 跑两次确认共享 cache 命中（cold→warm 时间差应与 CLI 路径一致）。

### Model-Lab1 P2/P3 闭环

- **指标 inventory**：新增 [model_factor_metric_inventory.md](model_factor_metric_inventory.md) 覆盖 model-factor 独有产物（training_log/training_metrics/feature_importance/feature_importance_ledger/feature_oos_ic/purged_kfold_*/model_selection/model_definition/feature_manifest），并登记跨文件 invariant（`purge_days == label_horizon`、`metrics.model_family == model_selection.family`）。共享指标继续指向单因子 inventory，避免重复。
- **Purged k-fold purge_gap × label_horizon**：`build_purged_kfold_diagnostics` 既有签名已强制要求 `label_horizon` kwarg，调用方 `model_factor/artifacts.py:546` 已正确传 `int(spec.target.horizon)`，且诊断内部 `purge_days = label_horizon`（结构性绑定，非可独立配置）。`tests/test_artifact_golden_regression.py::test_model_factor_level12_core_artifacts_match_golden` 已新增断言锁定 `purged_kfold_summary.json` 的 `label_horizon` / `purge_days` 等于 `metrics.json["target_horizon"]`。
- **`model_selection.json` ↔ `metrics.json` 一致性**：同一测试加 cross-file 断言：`status ∈ {disabled, no_candidates}` 时 `configured_model.family == metrics.model_family`；`status == "ok"` 时 `selection_rows` 中 `latest_selected_candidate_id` 对应行的 family == `metrics.model_family`。
- **P3 `web_model_lab.html` 抽离 (11k 行)**：本轮显式不动。功能稳定 + HTML 不变量测试已覆盖关键解析口径（`fmtNum`/`toNum`/`irOfPoints` 与 NAV 空点处理），单独抽离 ROI 偏低；建议下次大改 model-lab UI 时顺手把解析/聚合逻辑抽到 `modelLabSeries.ts`，参考单因子 `dashboardSeries.ts` 的拆分。

## 文档对账索引

| Inventory 行 | 当前产出 | 当前展示 | 状态 |
| --- | --- | --- | --- |
| `ic` / `rank_ic` / `mutual_information` | `ic_timeseries.csv`, `metrics.json` | dashboard 主图、PDF IC 图；MI 不单独画 | 口径需说明 |
| `group_return` / quantile cumulative | `group_returns.csv` | dashboard 分组均值、PDF 分组累计与均值 | 通过 |
| `long_short_return` / NAV | `metrics.json`, `backtest_result.json` `nav_points` | PDF NAV；dashboard 未画 | 未展示但 artifact 存在 |
| `rolling_stability` | `rolling_stability.csv` | PDF rolling 图；dashboard 未画 | 未展示但 artifact 存在 |
| `turnover` | `turnover.csv`, `metrics.json` | dashboard summary、PDF appendix | 口径需说明 |
| `coverage` | `coverage.csv`, `metrics.json` | dashboard summary、PDF appendix | 通过 |
| `ic_decay` | `ic_decay.csv` | PDF decay 图；dashboard 未画 | 未展示但 artifact 存在 |
| `factor_autocorrelation` | `factor_autocorrelation.csv` | dashboard/PDF 未画，CSV 诊断 | 未展示但 artifact 存在 |
| `capacity_estimation` | `capacity_estimation.csv` | dashboard/PDF 未画，CSV 诊断 | 未展示但 artifact 存在 |
| `daily_pnl_attribution` | `daily_pnl_attribution.csv`, daily PnL metrics | dashboard/PDF 未画，CSV 诊断 | 未展示但 artifact 存在 |

## 本轮未执行项

- 未运行真实 run 的 dashboard/PDF 视觉 QA。
- 未运行真实生产 run 的 model-lab1 overview 人工复核；fixture visual smoke、HTML 数字解析不变量测试与 model-factor artifact golden 已完成。
- 未在一致的 Node/node_modules 环境中完成 dashboard Vite build；本轮已完成 TypeScript 编译与 parser 逻辑测试。
- 未做 alphalens 对照；已完成极简 pandas 一次性对照。
- 已建立主单因子 golden snapshot、小截面/常数因子异常 golden snapshot、高换手/费后 golden snapshot，覆盖 canonical JSON/核心 JSON、summary、Level 2 package 与 13 个 CSV artifact hash；全空/极端缺失等更多异常 golden 仍待补。
- 已做展示层、artifact 契约、split/embargo、`next_open` sensitivity 与 turnover/fee 样本口径的最小修复；完整视觉 QA 与 alphalens 对照仍未执行。
