# Research Workflow（想法探索器协议）

本文档锁定 alpha-lab idea explorer 的研究工作流语义：核心定位、三段流水线、ledger schema、推理引擎编排、回灌闭环。Prompt 模板、retrieval 行为、CLI 设计、vault 卡 schema 都以本文档为准；如发现实现与文档不一致，应改实现、不改协议。

## 核心定位

**vault 是素材库，不是判决书。**

探索器目标：用 vault 的跨领域知识激发新 idea —— 拉广覆盖、抽 transferable moves、做跨领域类比与多卡融合。**探索阶段不做 KILL** —— 真正的 KILL 由 Stage 3 实际数据决定，不由 vault 先例决定。

这条定位反过来约束了几件不该做的事：
- 不用 vault `operative_claims` 触发"先例 kill"。知识库不是无误的，硬 kill 会误杀活机制。
- 不强制 ledger lineage。每条机制不必都有"出处卡"——novel synthesis 是合法产物。
- 不在探索阶段输出 KILL/HOLD 判决。这种判决在没跑数据之前是空头支票。
- 不让多 LLM 互审防外推 —— 让它们**互补不同的生成倾向**，候选机制总数应该变多，不是收敛。

## 三方分工

每个角色只做有信息优势的那段。

| 角色 | 信息优势 | 职责 |
| --- | --- | --- |
| alpha-lab + Claude Code / Codex GUI | factor zoo + 实验卡 + vault 全文 + `transferable_moves` | Stage 1 生成：跨领域融合起草 |
| 网页 GPT | 广博先验 + vault 之外的视角 | Stage 2 拓展：补 vault 没覆盖的视角、跨领域类比；不做 keep/kill 判决 |
| Codex GUI / Claude Code | 仓内 schema、PIT、builder 模板、评估管线 | Stage 3 实现 + 跑标准评估管线，KILL/KEEP 由数据决定 |

vault 是被动第四方：既是 Stage 1 的生成原料（`transferable_moves`），又通过 experiment card 回灌承接闭环（`emergent_moves`）。

## 推理引擎编排（当前选定）

| Stage | 引擎 | 模式 | 备注 |
| --- | --- | --- | --- |
| 1 生成 | **Claude Code（深度组合）+ Codex GUI（广度迁移）** 并行 | 互补 | 不是互审，是互补不同生成倾向 |
| 2 拓展 | **网页 GPT** | 单一 | 不可被 Claude/Codex 替代——它的对抗先验来自"完全没读过 vault" |
| 3 实现+验证 | **单一**（Codex GUI 或 Claude Code 任一） | 单一 | 代码只该有一份，KILL 由评估管线数据 |

### Stage 1 双引擎纪律

并行起草不是"两份 yaml 简单合并"，是受约束的双 voice 生成：

1. **共享 retrieval 上下文**：alpha-lab 出 prompt 时同时给两个引擎，两边看到的卡片集 / `transferable_moves` 必须一字不差相同。差异只能来自模型本身。
2. **倾向分工**：
   - Claude Code 偏**深度组合**：在选定 3-5 张卡内找深层结构相似性、把多个 move 融合进单一假设
   - Codex GUI 偏**广度迁移**：从更远的领域（别的 asset class / 频率 / method）拉跨领域类比
3. **独立产出** `ledger_v1.claude.yaml` / `ledger_v1.codex.yaml`，互不可见。
4. **reconcile 加法不减法**：reconcile 目标是**候选机制总数变多**——两边各保留 + 看能不能再产生 fusion 候选；不要"选哪边对"。
5. 最终 `ledger_v1.yaml` 是 reconcile 后的并集（不是收敛）；两份原始 yaml + reconcile.md 留档。

### 为什么 Stage 2 不能被 Claude+Codex 替代

Claude Code 与 Codex GUI 都需要读 vault 才能起草，他们的视角是**vault 内部相关性**。网页 GPT 的视角是**vault 外部独立性**——它训练数据里见过的因子文献、跨市场结构、统计陷阱，没有被 vault 措辞污染。让 Claude/Codex 替 Stage 2 等同于把"外部拓展"换成"内部互审"。

## 三段流水线

### Stage 1 — 生成

**检索**：alpha-lab 多样化拉卡——不只 top-k 语义最近邻，还按 frequency / asset / method 维度铺开广度，确保跨领域候选进入 prompt。

**抽取**：从每张卡 frontmatter 抽：
- `transferable_moves`（**生成原料**，核心）
- `operative_claims`（**上下文 hint**，弱——给 ledger 提供观察/经验参考，不触发任何 kill）

**起草**：双引擎并行，prompt 重心在**生成性提示**：
- "下面 N 张卡分别来自 [crowding / volatility / liquidity / regime / sentiment] 不同领域，提一个把至少两个领域的 move 融合的机制"
- "卡片 X 的 move 在 daily PV 已被使用——能否搬到 model_factor 的 feature engineering？描述类比 + 差异"
- "vault 里有哪些 transferable_move 你认为被低估了？给一个最有可能跨上下文复用的"

**输出**：`ledger_v1.claude.yaml` + `ledger_v1.codex.yaml` + `reconcile.md` → 收敛后的 `ledger_v1.yaml` + `retrieval_log.md`。

### Stage 2 — 拓展（可选）

输入：`ledger_v1.yaml` + `retrieval_log.md`。
输出：`ledger_v2.yaml`（增量）+ `expansion_notes.md`。

网页 GPT 做的是**拓展视角**：补 vault 没覆盖的角度、提跨领域类比、提示可能被忽略的实证研究。

**不做**：keep/kill/add 判决、写代码、改 ledger lineage、做 alias 审计。

候选机制只增不减。如果 GPT 觉得某条机制弱，标 `concern: ...` 但不删除。

### Stage 3 — 实现 + 数据验证

挑 1-2 个最有趣的 mechanism 进实现：
- builder + tests + 跑实验
- 标准 single-factor / model-factor 评估管线本来就出 alias 相关性 / regime 稳定性 / PIT 合规扫描
- KILL / KEEP 由**实际数据**决定

不需要单独的"Stage 3 lineage 审计"或"vault precedent kill"步骤——这些在简化协议下已删除。

### 闭环回写 vault

experiment card 必填：
- `emergent_moves`：这次实践浮现、可被未来因子借用的新 move（**主回写字段**）
- `operative_claims`：观察到的现象 / 经验 / 边界条件（**弱字段**——是观察记录，不是真理）

下次 Stage 1 检索时，bridge 自动 surface 这些 emergent moves，让素材库越来越厚。

## ledger schema

```yaml
mechanism_1:
  hypothesis: <跨领域 / 融合的假设>
  inspired_by:                       # 可选——溯源便于学习
    - card: <vault path>
      what_i_took: <借鉴的具体动作或观察>
      cross_domain_jump: <如有：从 X 领域搬到 Y 领域>
  fusion_of:                         # 可选——多卡融合时填
    - [<card_1>, <card_2>]
  novel_delta: <这次组合的新颖之处>    # 鼓励但不强制
  signal_sketch: <粗略的信号描述>
  data_needs: [<需要什么数据>]
```

**刻意不要**：
- 强制 lineage（找不到来源 ≠ 缺口，可能是 novel synthesis）
- `inherited_falsification`（KILL 不在这做）
- `transfer_risk` / `adapted_how` / `source_model` 强制项
- `vault_gaps`（无来源不再是缺口）

## vault 卡 frontmatter 双字段

| 字段 | 角色 | 写法 |
| --- | --- | --- |
| `transferable_moves` | **生成原料**（核心） | 列出"这张卡能被偷走的具体动作"，越具体越好 |
| `operative_claims` | **上下文 hint**（弱） | 写"我观察到的现象 / 经验 / 边界"，**不必防御性**——错了无所谓，因为它不喂 kill |

注意：`operative_claims` 在简化协议下角色降级。之前担心"知识库错误会误杀因子"——降级为 hint 后这个担心消失，因为它从不触发 kill，只是给 ledger 提供参考观察。

## 仍生效的硬约束

只有两类东西作为硬约束保留：

1. **`available_data` 硬过滤**（Stage 1 retrieval）：日频项目不应被 HFT 卡片带偏。这是数据物理约束，不算"约束 idea"。优先级：`explicit` > `frequency:<f>` 自动推断 > `none`。
2. **Stage 1 弱命名约束**：避免直接用 `reversal / momentum / value / quality / size / skewness / liquidity / ...` 这类 canonical labels 命名候选——保护假设空间，鼓励描述机制本身而非贴标签。Stage 2/3 不再有此约束。

`mode ∈ {start, free, constrained}` 是 Stage 1 内部对单个 prompt 的严格度微调，不影响协议骨架。

## Model-lab 对齐

三段流水线对称镜像；研究对象从"因子机制"换成"模型改进机制"（loss / regularization / feature interaction / target construction / sample weighting / training window / model selection）。`transferable_moves` / `operative_claims` 字段语义同；ledger schema 同。

## 实现状态

- ⏳ vault 卡 `transferable_moves` + `operative_claims` 双字段批量升级（截至 2026-05-04 约 30%，~150/500 卡）。新协议下 `operative_claims` 写法可放松，**已升级的卡无需返工**。
- ⏳ 现有 Stage 1 prompt builders（`service.py::_build_factor_recipe_*_prompt` / `model_idea.py::_build_model_idea_*_prompt`）仍按旧 lineage-strict 模板写：强制 `borrowed_moves` / `inherited_falsification` / `vault_gaps`。**需要 relax 为 `inspired_by` / `fusion_of` / `cross_domain_jump` 可选模板**，并加入跨领域生成性提示。
- ⏳ `_build_factor_recipe_validation_kill_tests_prompt` / `_build_model_idea_validation_kill_tests_prompt` 在新协议下已无角色，可清理或保留为废弃 helper。
- ⏳ Stage 1 双引擎模式：CLI `alpha-lab idea draft --models claude,codex` 发同一份 prompt + retrieval 上下文给两边，分别落 `ledger_v1.<model>.yaml`，并生成 `reconcile.md` 模板（侧重"两边并集 + 第三轮 fusion"，不侧重"互审差异"）。
- ⏳ Stage 3 实做计算 / 先例 kill / lineage 审计相关代码（如有）需要清理，让评估管线直接接 ledger。
- ⏳ experiment card 回写 `emergent_moves`（主）+ `operative_claims`（弱）闭环。

## 代码入口（当前 scaffold）

| 关注点 | alpha-lab | model-lab |
| --- | --- | --- |
| Stage 1 prompt 派发 | `service.py::_build_factor_recipe_exploration_prompt` | `model_idea.py::_build_model_idea_exploration_prompt` |
| Stage 1 sub-prompt builder | `service.py::_build_factor_recipe_{start,structured,constrained,signal_mapping}_prompt` | `model_idea.py::_build_model_idea_mechanism_*` / `_signal_mapping_prompt` |
| `transferable_moves` / `operative_claims` 提取 | `service.py::_extract_frontmatter_field_items`（line ~2957） | `model_idea.py::_read_card_frontmatter_field_items`（line ~812） |
| `available_data` 自动推断 | `scoring.py::infer_available_data_from_frequency` | 同左 |
| Retrieval 多分量评分 | `scoring.py::score_card` + `service.py::_typed_rank_candidates` | 同左 |
| 诊断输出 | `ExploreIdeaResult.retrieval_diagnostics` | `ModelIdeaResult.retrieval_diagnostics` |

> 注：旧 `_build_*_validation_kill_tests_prompt` 在新协议中已无角色。
