# Research Workflow（想法探索器协议）

本文档锁定 alpha-lab idea explorer 的研究工作流语义：核心定位、三段流水线、ideas/<id>/ 目录约定、ledger schema、推理引擎编排、回灌闭环。Prompt 模板、retrieval 行为、CLI 设计、vault 卡 schema 都以本文档为准；如发现实现与文档不一致，应改实现、不改协议。

## 核心定位

**vault 是素材库，不是判决书。**

探索器目标：用 vault 的跨领域知识激发新 idea —— 拉广覆盖、抽 transferable moves、做跨领域类比与多卡融合。**探索阶段不做 KILL** —— 真正的 KILL 由 Stage 3 实际数据决定，不由 vault 先例决定。

这条定位反过来约束了几件不该做的事：

- 不用 vault `operative_claims` 触发"先例 kill"。知识库不是无误的，硬 kill 会误杀活机制。
- 不强制 ledger lineage。每条机制不必都有"出处卡"——novel synthesis 是合法产物。
- 不在探索阶段输出 KILL/HOLD 判决。这种判决在没跑数据之前是空头支票。
- 探索器输出的不是最终因子定义、不是 case spec、不是 candidate payload —— 这些只有 Stage 2 网页 GPT 才会输出。

## 三方分工

每个角色只做有信息优势的那段。

| 角色 | 信息优势 | 职责 |
| --- | --- | --- |
| 想法探索器（alpha-lab `idea distribute` / `model-idea distribute`） | vault 全文 + 代码库 factor/model/case 现状 + ModelFactorCaseSpec / factor.json schema + validator 硬约束 | 检索 + 分发：产出 retrieval pack + 两份 audience-specific prompts |
| Claude Code（generator） | 跨卡机制深挖 + 多领域 transferable_moves 融合 | Stage 1 generator：输出 `mechanism_deepdive.md`（机制候选 ledger） |
| Codex GUI（reviewer，早期） | 代码库 schema、PIT 合同、validator 规则、字段可用性 | Stage 1 reviewer：输出 `code_feasibility_review.md`（机制可执行性评审） |
| 网页版 GPT | 广博先验 + vault 之外的视角 | Stage 2：合并机制候选 + 可执行性评审，输出 `factor_json_payload` / `model_candidate_payload` 机器合同 |
| Codex GUI（执行者，后期） | 仓内 schema、PIT、builder 模板、评估管线 | Stage 3：写入 research candidate、validate、跑标准 pipeline、artifact 审计、迭代点评 |

vault 是被动第四方：既是 Stage 1 的生成原料（`transferable_moves`），又通过 experiment card 回灌承接闭环（`emergent_moves`）。

## 推理引擎编排（当前选定）

| Stage | 引擎 | 模式 | 备注 |
| --- | --- | --- | --- |
| 0 分发 | **想法探索器（Claude API）** | 单一 | 检索 vault + 代码库，产出 retrieval pack + 2 份 audience-specific prompts，不调用 Stage 1 引擎 |
| 1 生成 + 评审 | **Claude Code（generator）+ Codex GUI（reviewer）** 并行 | 互补 | generator 输出机制候选；reviewer 独立输出代码库可执行性评审，不互相看对方输出 |
| 2 合同化 | **网页版 GPT** | 单一 | 不可被 Claude/Codex 替代——它的视角是"vault 外部 + 代码库外部"，输出 Stage 3 唯一可执行的机器合同 |
| 3 实现+验证 | **单一**（Codex GUI 或 Claude Code 任一） | 单一 | 代码只该有一份，KILL 由评估管线数据 |

### Stage 1 双 audience 纪律（重要更新）

之前协议把 Claude Code 和 Codex GUI 都当 *generator*（深度组合 vs 广度迁移）。新协议下两者扮演**不同的 audience 角色**：

| audience | 引擎 | 输入 prompt | 输出 |
| --- | --- | --- | --- |
| `claude_mechanism` | Claude Code | `prompt_claude_mechanism.md`（vault 卡 + transferable_moves） | `mechanism_deepdive.md`（机制候选 ledger） |
| `codex_review` | Codex GUI | `prompt_codex_review.md`（同 vault 上下文 + 代码库 factor/model 索引 + ModelFactorCaseSpec / factor.json schema + validator 硬约束清单） | `code_feasibility_review.md`（每条候选机制是否在 v1 spec/factor.json 内可执行；不可执行的注明缺什么） |

**关键纪律：**

1. **共享 retrieval 上下文**：两份 prompt 看到的 vault 卡 / `transferable_moves` 必须一字不差相同。差异只来自代码库索引（reviewer 多看到）和 audience-specific 任务说明。
2. **角色不互通**：generator 不写 review，reviewer 不写新机制。两边互不可见。
3. **reviewer 不否决**：不可执行机制不删除，只在 review 中标 `implementation_status: needs_extension`，让 Stage 2 网页 GPT 决定是否进入 v1。
4. **加法不减法**：候选机制总数只增不减；Stage 2 才是合同化收口。

### 为什么 Stage 2 不能被 Claude+Codex 替代

Claude Code 与 Codex GUI 都需要读 vault + 代码库，他们的视角是**仓内相关性**。网页版 GPT 的视角是**仓外独立性**——它训练数据里见过的因子文献、跨市场结构、统计陷阱，没有被本仓措辞污染。让 Claude/Codex 替 Stage 2 等同于把"外部合同化"换成"内部互审"。

## 三段流水线

### Stage 0 — 分发（想法探索器）

入口：

```bash
alpha-lab idea distribute --idea "<模糊 idea>" --output-dir ideas/<idea_id>/
alpha-lab model-idea distribute --idea "<模糊 idea>" --output-dir ideas/<idea_id>/
```

行为：

1. **检索 vault**（多分量评分）：`transferable_moves`（生成原料，核心）、`operative_claims`（弱上下文 hint）、跨领域多样化拉卡（不只 top-k 语义近邻）。
2. **检索代码库**：列出 `custom_factors/{promoted,research}/`、`model_candidates/{promoted,research}/`、`configs/real_cases/{single_factor,model_factor}/` 现有候选；快照 `ModelFactorCaseSpec` / `factor.json` schema；摘录 validator 硬约束清单。
3. **生成两份 audience-specific prompts**：`prompt_claude_mechanism.md`（generator）+ `prompt_codex_review.md`（reviewer）。
4. **写入** `ideas/<idea_id>/`（见目录约定）。

`alpha-lab idea distribute` 不调用任何 Stage 1 引擎；只产出文件。用户自己把对应 prompt 粘贴到 Claude Code / Codex GUI。

### Stage 1 — 生成 + 评审（Claude Code + Codex GUI 并行）

| audience | engine | output |
| --- | --- | --- |
| `claude_mechanism` | Claude Code | `ideas/<id>/mechanism_deepdive.md` |
| `codex_review` | Codex GUI | `ideas/<id>/code_feasibility_review.md` |

最终用户把两份输出 + retrieval_log + reconcile 模板交给网页版 GPT。

### Stage 2 — 合同化（网页版 GPT）

输入：`mechanism_deepdive.md` + `code_feasibility_review.md` + `retrieval_log.md` + reconcile 槽位 + （单因子）`single_factor_stage1_reconcile_contract.md` 或（模型）`model_lab_stage1_reconcile_contract.md` 模板。

输出：

- 单因子：`factor_json_payload`（Stage 3 写入 `custom_factors/research/<f>/factor.json`）
- 模型：`model_candidate_payload` 含完整 `case_spec_payload`（Stage 3 写入 `model_candidates/research/<c>/model_candidate.json`）

合同要求：

- `provenance.idea_id` 必须来自 Stage 0 输出的 `manifest.json::idea_id`。
- `provenance.stage2_payload_sha256` 由 Stage 2 输出生成，让 Stage 3 artifact 可反查到本轮 GPT payload 版本。
- `provenance.audience_chain` 列出 Stage 1 走过的 audience（如 `["claude_mechanism","codex_review"]`），保证可追溯。

### Stage 3 — 实现 + 数据验证（Codex GUI 或 Claude Code）

挑 1-2 个最有趣的 mechanism 进实现：

- `validate-draft-factor` / `validate-draft-model` 必须先通过。
- 标准 `alpha-lab real-case single-factor run` 或 `model-factor run` 作为唯一执行入口。
- artifact 必须包含 `custom_factor_source.provenance` / `draft_model_source.provenance` 审计字段（含 `idea_id` + `stage2_payload_sha256`）。
- KILL / KEEP 由**实际数据**决定。

Stage 3 引擎不得：

- 自行补全机制（Stage 2 payload 缺字段时停下来回写到 review notes，不要改写）。
- 改写 Stage 2 合同（与 payload 冲突时以 payload 为准 + 写入 research_log）。
- 跳过 validator（validator 失败即 Stage 3 失败）。
- 写临时脚本、notebook、散落 .py 文件、修改 core 模块、修改 promoted、修改前端注册。

### 闭环回写 vault

experiment card 必填：

- `emergent_moves`：这次实践浮现、可被未来因子借用的新 move（**主回写字段**）
- `operative_claims`：观察到的现象 / 经验 / 边界条件（**弱字段**——是观察记录，不是真理）

下次 Stage 0 检索时，bridge 自动 surface 这些 emergent moves，让素材库越来越厚。

## ideas/<idea_id>/ 目录约定

每一轮 idea 探索的所有材料都收敛到一个目录，方便 Stage 3 artifact 反查、跨轮迭代追溯。

```
ideas/<idea_id>/
  manifest.json                        # idea_id, created_at, audiences, lab, retrieval_diagnostics
  retrieval_pack.md                    # 共享 vault 卡 + 代码库索引快照 + schema 摘录
  retrieval_log.md                     # 检索打分 / 命中 / 多样化日志
  prompt_claude_mechanism.md           # generator prompt（送 Claude Code）
  prompt_codex_review.md               # reviewer prompt（送 Codex GUI；含代码库索引 + schema + validator 硬约束）
  mechanism_deepdive.md                # Claude Code 输出（用户保存）
  code_feasibility_review.md           # Codex GUI 输出（用户保存）
  reconcile.md                         # Stage 2 网页 GPT 入口模板（含两侧输入槽位）
  stage2_payload.json                  # 网页 GPT 回出的 factor_json_payload / model_candidate_payload
  stage3_runs/                         # Stage 3 实验 artifact 软链或摘要
```

`<idea_id>` 由 `<UTC timestamp>__<safe-slug>` 生成（如 `20260511T143000Z__turnover-conditioned-pv`）。Stage 3 写入 `custom_factors/research/<f>/factor.json` 时把 `idea_id` 复制进 `provenance.idea_id`。

## ledger schema（mechanism_deepdive.md 内嵌）

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
  concern: <可选——已知潜在脆弱点>
```

**刻意不要**：

- 强制 lineage（找不到来源 ≠ 缺口，可能是 novel synthesis）
- `inherited_falsification`（KILL 不在这做）
- `transfer_risk` / `adapted_how` / `source_model` 强制项
- `vault_gaps`（无来源不再是缺口）
- `implementation_status`（这是 reviewer 的事，不是 generator 的事）

## code_feasibility_review.md schema（reviewer 输出）

reviewer 不重写机制，只对 generator 的每条机制给一个评审条目：

```yaml
mechanism_1:
  in_v1_contract: true | false
  required_columns_present: [<col_1>, ...]
  required_columns_missing: [<col_n>, ...]
  spec_fields_touched: [feature_columns, feature_preprocess, model.family, training, ...]
  validator_blockers: [<rule_id>, ...]    # 触发哪些硬约束（PIT / future leakage / forbidden imports）
  implementation_status: in_contract_spec_variant | partial_in_contract | needs_extension | future_enhancement
  reviewer_note: <一段中文说明为什么这样判定>
```

reviewer 只负责"能不能在 v1 schema 内执行"；不否决、不淘汰、不重写假设。

## vault 卡 frontmatter 双字段

| 字段 | 角色 | 写法 |
| --- | --- | --- |
| `transferable_moves` | **生成原料**（核心） | 列出"这张卡能被偷走的具体动作"，越具体越好 |
| `operative_claims` | **上下文 hint**（弱） | 写"我观察到的现象 / 经验 / 边界"，**不必防御性**——错了无所谓，因为它不喂 kill |

注意：`operative_claims` 在简化协议下角色降级。之前担心"知识库错误会误杀因子"——降级为 hint 后这个担心消失，因为它从不触发 kill，只是给 ledger 提供参考观察。

## 仍生效的硬约束

只有两类东西作为硬约束保留：

1. **`available_data` 硬过滤**（Stage 0 retrieval）：日频项目不应被 HFT 卡片带偏。这是数据物理约束，不算"约束 idea"。优先级：`explicit` > `frequency:<f>` 自动推断 > `none`。
2. **Stage 1 弱命名约束**：避免直接用 `reversal / momentum / value / quality / size / skewness / liquidity / ...` 这类 canonical labels 命名候选——保护假设空间，鼓励描述机制本身而非贴标签。Stage 2/3 不再有此约束。

`mode ∈ {start, free, constrained}` 是 Stage 1 内部对单个 prompt 的严格度微调，不影响协议骨架。

## Model-lab 对齐

三段流水线对称镜像；研究对象从"因子机制"换成"模型改进机制"（loss / regularization / feature interaction / target construction / sample weighting / training window / model selection）。`transferable_moves` / `operative_claims` 字段语义同；ledger schema 同；audience 划分同（`claude_mechanism` 写机制，`codex_review` 评审 ModelFactorCaseSpec 可执行性）。

## 实现状态

- ✅ Stage 0 `alpha-lab idea distribute` 双 audience 模式：emit `prompt_claude_mechanism.md` + `prompt_codex_review.md` + retrieval_pack（含代码库索引）。
- ✅ Stage 0 `alpha-lab model-idea distribute` 同构。
- ✅ `ideas/<idea_id>/` 目录约定，manifest.json 含 idea_id。
- ✅ Stage 2 reconcile contract 加 `code_feasibility_review` 输入槽位（model + 单因子两侧）。
- ✅ Stage 3 envelope 加 `forbidden_actions` + `escalation_triggers`。
- ✅ `factor.json` / `model_candidate.json` 加可选 `provenance` 块（idea_id + stage2_payload_sha256 + audience_chain），validator 校验形态，artifact 审计回写。
- ⏳ vault 卡 `transferable_moves` + `operative_claims` 双字段批量升级（截至 2026-05-04 约 30%，~150/500 卡）。新协议下 `operative_claims` 写法可放松，**已升级的卡无需返工**。
- ⏳ experiment card 回写 `emergent_moves`（主）+ `operative_claims`（弱）闭环。

## 代码入口（当前 scaffold）

| 关注点 | alpha-lab（单因子） | model-lab（模型） |
| --- | --- | --- |
| Stage 0 audience prompt 派发 | `audience_prompts.py::build_prompt(SINGLE_FACTOR, audience, ctx)` | `audience_prompts.py::build_prompt(MODEL_FACTOR, audience, ctx)` |
| 代码库索引（reviewer 用） | `codebase_index.py::index_factors / index_existing_cases` | `codebase_index.py::index_model_candidates / index_existing_cases` |
| Stage 0 entry | `service.py::distribute_idea` | `model_idea.py::distribute_model_idea` |
| `transferable_moves` / `operative_claims` 提取 | `service.py::_extract_frontmatter_field_items` | `model_idea.py::_read_card_frontmatter_field_items` |
| `available_data` 自动推断 | `scoring.py::infer_available_data_from_frequency` | 同左 |
| Retrieval 多分量评分 | `scoring.py::score_card` + `service.py::_typed_rank_candidates` | 同左 |
| 诊断输出 | `ExploreIdeaResult.retrieval_diagnostics` | `ModelIdeaResult.retrieval_diagnostics` |
| Stage 3 validator | `draft_factor_validation.py::validate_draft_factor_file` | `draft_model_validation.py::validate_draft_model_file` |
| Audit pass-through | `custom_factors.py::CustomFactorSource.to_audit_dict` | `model_candidates.py::DraftModelSource.to_audit_dict` |

> 注：旧 `idea draft` / `_render_idea_model_dispatch` 双 generator 路径已被 `idea distribute` 替代；旧 `_build_*_validation_kill_tests_prompt` 在新协议中已无角色。
