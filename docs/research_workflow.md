# Research Workflow（探索协议）

本文档锁定 alpha-lab idea explorer 的研究工作流语义。Prompt 模板与 retrieval 行为应以本文档为准；如发现实现与文档不一致，应改 prompt / retrieval、不改协议。

## 两条正交轴

### `workflow_stage` — 研究动作

研究在哪一步。三档线性推进，但**不强制**：caller 可以显式跳到任意一档。

| Stage | 在做什么 | 不允许做什么 |
| --- | --- | --- |
| `mechanism_discovery` | 拆机制；维护多个互斥候选；保留不确定性 | 提前命名（用现成标签）/ 预设收益方向 / 写公式 / ranking |
| `signal_mapping` | 把机制翻译成最少必要信号组件；解释当前实现捕捉了哪个机制 | 生成新机制 / 做最终选择 |
| `validation_kill_tests` | 审计已有实现；alias / exposure / robustness / sub-sample 四类 kill test | 再生成假设 / 写新公式 / 回避结论 |

默认 `mechanism_discovery`。每次返回的 `retrieval_diagnostics.recommended_next_stage` 给出建议下一步，但不锁定流程。

### `mode` — 约束强度

同一 stage 内部的严格度微调。三档：

- `start` — kickoff 风格，最宽松。在 `mechanism_discovery` 等价于 `free`；在 `signal_mapping` / `validation_kill_tests` 路由到对应 stage 的 open 模板。
- `free` — 默认结构化，要求保留多候选、识别风险，但不要求收敛。
- `constrained` — 严格变体：必须 cite 知识锚点、必须给二值判定、强制候选数量上限等。各 stage 的 strict 规则在该 stage 的 prompt builder 里硬编码。

## Stage × Mode 矩阵

每格的语义都是独立模板，不是 slot 替换：

| | `start` / `free` | `constrained` |
| --- | --- | --- |
| `mechanism_discovery` | 概念禁用约束 + 2-3 个候选 + 必须写差异 | 同上 + 候选 ≤ 3 + 强制证据锚点 + 自检删除塌缩候选 |
| `signal_mapping` | 三段映射 + 当前实现解释 + confound 清单 + 2-3 测试版本 | 同上 + 每条 implication 必须 cite + 当前实现需 binary alias-tag + 版本数 ∈ {2,3} |
| `validation_kill_tests` | 5 alias 审计 + 4 类 kill test + HOLD/ITERATE/KILL 三选一 | 同上 + 每个 alias 判定必须 cite + 二值 KILL/HOLD-FOR-AUDIT + 列出 3-5 follow-up 实证 |

## 同一组标签的三种身份

`reversal / momentum / value / quality / size / skewness / liquidity / sentiment / crowding / low-vol / 反转 / 动量 / ...` 这组 canonical labels 在三个 stage 扮演**完全不同**的角色：

| Label 身份 | Stage | 行为 |
| --- | --- | --- |
| 护栏（forbidden） | `mechanism_discovery` | 禁止用这些词命名候选机制；命名 = 提前归类 = 塌缩假设空间 |
| 控制项（confound） | `signal_mapping` | 必须逐项说明信号版本如何处理：`{包含 / 残差化 / 显式控制 / 不控制（带风险声明）}` |
| 审计靶子（alias target） | `validation_kill_tests` | 必须明确回答"这个因子能否仅用 X 来解释？"，给 `{显著重叠 / 部分重叠 / 不重叠}` |

**不要把这三种身份混起来**。最常见的错误是把发现阶段的"禁止命名"扩展到所有阶段——那会让 audit prompt 没法说出审计靶子的名字。同样反向错误也存在：把验证阶段的"全部要审一遍"灌到发现阶段，结果第一步就把候选机制全压成 reversal/momentum 的变体。

## `available_data` 优先级

数据可用性决定哪些卡片进 prompt 上下文（dependency hard filter，详见 `scoring.data_dependency_score`）。来源优先级：

1. **`explicit`** — 调用方显式传入 `available_data=frozenset({"close", "volume", ...})`。永远赢。
2. **`frequency:<f>`** — 未显式传入时，从 `project.frequency` 自动推断：`daily / weekly / monthly / d / w / m / ashare / a_share` → `DAILY_DATA_INVENTORY`；`intraday / tick / minute / 1min / ... / hft` → `INTRADAY_DATA_INVENTORY`。
3. **`none`** — 不可识别的频率或没有 project 时不启用 hard filter，dependency 分数退化为 0.5（中性），soft signal only。

`retrieval_diagnostics.available_data_source` 字段记录实际命中的来源（`explicit` / `frequency:daily` / `none`），便于排查"为什么 HFT 卡片进/没进上下文"。

## 一个例子：asym-vol（log(downside_vol) − log(upside_vol)）

日频项目研究一个上下行波动不对称因子。无文档时常见漂移：被高频卡片带偏 → 第一步直接命名为"reversal 换壳" → 第二步绕过中性化检查 → 第三步给"看起来不错，需要更多数据"。

按本协议走则：

**1. `stage=mechanism_discovery, mode=free`**（默认）。`project.frequency=daily` 自动启用 `DAILY_DATA_INVENTORY`，HFT 卡片（如 `Intraday Volume Burst`，依赖 `intraday_tick_volume`）在 dependency hard filter 阶段直接出局，不进 prompt。Prompt 里禁止用 reversal / volatility / skewness / liquidity 这些词命名候选。LLM 必须给出 ≥2 个互斥候选——例如：(a) 风险厌恶非对称定价，(b) 负面信息释放速度差异，(c) 流动性冲击的方向不对称——每个候选必须显式写"与最相近的已有标签的差异"。

**2. `stage=signal_mapping, mode=constrained`**。同样的标签列表回来，但身份变成 confound 控制项。LLM 必须对每个候选写：observable implication → required_data → tag 频率（daily sufficient / intraday required）+ 角色（necessary / decorative / confound control）。然后必须回答"`log(downside_vol) − log(upside_vol)`这个当前实现到底捕捉了 (a)(b)(c) 中的哪一个？哪些机制需要 intraday 才能区分但当前是 daily 实现？"。strict 模式还要求当前实现对 `reversal / total volatility` 给一个 binary alias-tag。

**3. `stage=validation_kill_tests, mode=constrained`**。标签列表第三次回来，这次是审计靶子。LLM 必须按 `{显著重叠 / 部分重叠 / 不重叠}` 判定本因子是否只是 reversal / volatility / skewness-downside / liquidity-turnover / size-industry-price 的换壳；必须做行业 / 市值 / 流动性 / 波动率四项中性化测残差 IC；必须走涨跌停 / 停牌 / ST / 复权 / IPO 退市 PIT 检查；必须做 skip_recent / 窗口 / horizon / 预处理稳健性扫描；必须分年份 / regime / 行业 / 市值桶看稳定性。最后输出二值 `KILL / HOLD-FOR-AUDIT`，HOLD 必须列 3-5 个 follow-up 实证。

整条链路里日频/HFT 不会混淆，标签角色不会混淆，结论不会停在"看起来不错"。

## 代码入口

| 关注点 | 路径 |
| --- | --- |
| Stage 常量 + 推进规则 | `src/alpha_lab/research_bridge/scoring.py` (`WORKFLOW_STAGES`, `recommend_next_stage`) |
| Mode 规整 | `src/alpha_lab/research_bridge/service.py` (`_normalize_explore_mode`) |
| Stage-first prompt 派发 | `service.py::_build_factor_recipe_exploration_prompt`；先按 `stage` 选模板族，再由 `mode` 决定该模板内的 strictness |
| `mechanism_discovery` prompt builder | `service.py::_build_factor_recipe_start_prompt` / `_build_factor_recipe_structured_prompt` / `_build_factor_recipe_constrained_prompt` |
| `signal_mapping` / `validation_kill_tests` prompt builder | `service.py::_build_factor_recipe_signal_mapping_prompt` / `_build_factor_recipe_validation_kill_tests_prompt`；这两段不再拆成 mode-specific 函数，`constrained` 只开启 strict 规则 |
| 概念禁用约束（mechanism_discovery） | `service.py::_append_mechanism_discovery_concept_constraints` |
| 共享 canonical 标签 | `scoring.py::FORBIDDEN_FACTOR_LABELS` / `VALIDATION_ALIAS_TARGETS` / `SIGNAL_MAPPING_CONFOUND_CONTROLS` |
| `available_data` 自动推断 | `scoring.py::infer_available_data_from_frequency` |
| Retrieval 多分量评分 | `scoring.py::score_card` + `service.py::_typed_rank_candidates` |
| 诊断输出 | `ExploreIdeaResult.retrieval_diagnostics`（mode / stage / recommended_next_stage / score_components_by_name / dropped_cards / score_weights / query_anchor / available_data_source） |

## Model-lab 对齐

model-lab 使用同一条 `workflow_stage` 轴，但研究对象从“因子机制”变成“模型改进机制”。三段语义如下：

| Stage | model-lab 目标 | 必须避免 |
| --- | --- | --- |
| `mechanism_discovery` | 发现模型改进机制：loss/regularization、feature interaction、target construction、sample weighting、training window、model selection | 直接给最终 spec patch、推荐 single best model、把机制写成单纯调参 |
| `signal_mapping` | 把上游机制映射成可测试 spec/run 版本；每个字段标 role，并说明 remove-and-test 理由 | 生成新机制、选择赢家、忽略 PIT / leakage / overfit / turnover / feature stability |
| `validation_kill_tests` | 审计模型改进是否只是 baseline/ridge、regularization-only、feature-count、leakage、split luck 或 turnover/cost artifact | 用“需要更多数据/进一步研究”等回避语替代 KILL/HOLD 判定 |

model-lab 的 lint / session / UI 闭环：

- 响应 lint：`output_lint.py::lint_model_idea_response`，并通过 `describe_model_lint_contract` 注入 prompt 自检。
- 回灌响应：`model_idea.py::record_model_idea_response`，写入 `response`、`response_sections`、`lint_report`。
- 跨阶段 chaining：`find_upstream_model_idea_session` 与 `render_model_upstream_artifact_header` 把上一阶段结构化产物注入下一阶段 prompt。
- 数据库存推断：未显式传入 `available_data` 时，`model_idea.py::explore_model_idea` 会从当前 spec 的 `rebalance_frequency` 调用 `scoring.py::infer_available_data_from_frequency`，让日频模型和 alpha-lab 一样过滤掉依赖 HFT 数据的知识卡片。
- CLI：`python -m alpha_lab.research_bridge.model_idea record-response --session-id ... --response-text ...`。
- Web：`/api/model-lab/idea-explorer/record-response`，页面里的“回灌响应并 Lint”按钮会调用该接口。
