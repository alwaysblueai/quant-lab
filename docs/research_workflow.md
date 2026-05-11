# Research Workflow（协议不变量）

本文档锁定 alpha-lab idea explorer 的**协议不变量** —— 不随阶段细化或工具迭代而改变的硬规则。每个 Stage 的实际操作步骤（入口、文件、命令、迭代）以 `docs/end_to_end_workflow.md` 为权威。如果实现与本协议不一致，应改实现、不改协议。

## 核心定位

**vault 是素材库，不是判决书。**

探索器目标：用 vault 的跨领域知识激发新 idea —— 拉广覆盖、抽 transferable moves、做跨领域类比与多卡融合。**探索阶段不做 KILL** —— 真正的 KILL 由 Stage 3 实际数据决定，不由 vault 先例决定。

反过来约束的几件不该做的事：

- 不用 vault `operative_claims` 触发"先例 kill"。知识库不是无误的，硬 kill 会误杀活机制。
- 不强制 ledger lineage。每条机制不必都有"出处卡"——novel synthesis 是合法产物。
- 不在探索阶段输出 KILL/HOLD 判决。这种判决在没跑数据之前是空头支票。
- 探索器输出的不是最终因子定义、不是 case spec、不是 candidate payload —— 这些只有 Stage 2 网页 GPT 才会输出。

## 推理引擎编排（2026-05-11 修订版）

| Stage | 引擎 | 模式 | 备注 |
| --- | --- | --- | --- |
| 0 分发 | **alpha-lab `(model-)idea distribute`（Claude API）** | 单一 | 检索 vault + 代码库，产出 5 个文件（manifest + retrieval_pack + 2 份对称 prompt + stage2_input），不调用 Stage 1 引擎 |
| 1 生成+评审 | **Claude Code + Codex GUI** 并行 | 任务**相同**（generator + reviewer 合一）；互不可见 | 两份输出由网页 GPT 在 Stage 2 综合 |
| 2 合同化 | **网页版 GPT** | 单一 | 综合两引擎输出取长补短，输出唯一 `factor_json_payload` / `model_candidate_payload`，含 provenance |
| 3 实现+验证 | **单一**（Codex GUI 或 Claude Code 任一） | 单一 | 代码只该有一份，KILL 由评估管线数据 |
| 4 经验归档 | **后端 CLI + 用户手工** | 单一 | `alpha-lab idea experiment-card`，再由用户导出到 vault |
| 5 上线报告 | **前端 web unified** | 单一 | 仅展示 promoted 候选的完整可视化报告 |

### Stage 1 双引擎纪律

**两引擎做同一份工作**（generator + reviewer 合一），不是分工。是**冗余**。

| 引擎 | prompt | 输出 |
| --- | --- | --- |
| Claude Code | `ideas/<id>/prompt_claude.md` | `ideas/<id>/stage1_claude.md` |
| Codex GUI | `ideas/<id>/prompt_codex.md` | `ideas/<id>/stage1_codex.md` |

每份 prompt 字节级相同（只有自我标识"你是 Claude Code"/"你是 Codex GUI"差别）。每份输出含 Part A（机制候选 ledger，3-8 条）+ Part B（每条机制的可执行性评审）。

模型特点不同 → 输出各有优劣 → 网页 GPT 综合取长补短。

### 核心纪律

1. **共享 retrieval 上下文**：两份 prompt 看到的 vault 卡 / `transferable_moves` 必须一字不差相同。
2. **引擎互不可见**：Claude 不读 Codex 输出，反之亦然。
3. **候选只增不减**：不可执行的标 `implementation_status: needs_extension`，不删除；Stage 2 才是合同化收口。
4. **reviewer 不否决**：Part B 不删除 Part A 的机制。

### 为什么 Stage 2 不能被 Claude+Codex 替代

Claude Code 与 Codex GUI 都需要读 vault + 代码库，他们的视角是**仓内相关性**。网页版 GPT 的视角是**仓外独立性** —— 它训练数据里见过的因子文献、跨市场结构、统计陷阱，没有被本仓措辞污染。让 Claude/Codex 替 Stage 2 等同于把"外部合同化"换成"内部互审"。

## ledger schema（stage1_<engine>.md Part A 内嵌）

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
- `implementation_status` 写在 Part A（这是 Part B 的事）

## code feasibility review schema（stage1_<engine>.md Part B 内嵌）

每条机制对应一个评审条目：

```yaml
mechanism_1:
  in_v1_contract: true | false
  required_columns_present: [<col_1>, ...]
  required_columns_missing: [<col_n>, ...]
  spec_fields_touched: [feature_columns, ...]    # model lab
  factor_json_keys_touched: [code, required_columns, ...]   # single-factor lab
  validator_blockers: [<rule_id>, ...]
  implementation_status: in_contract_factor_def | in_contract_spec_variant | partial_in_contract | needs_extension | future_enhancement
  reviewer_note: <中文说明>
```

Part B 只负责"在 v1 schema 内能不能执行"；不否决、不淘汰、不重写假设。

## vault 卡 frontmatter 双字段

| 字段 | 角色 | 写法 |
| --- | --- | --- |
| `transferable_moves` | **生成原料**（核心） | "这张卡能被偷走的具体动作"，越具体越好 |
| `operative_claims` | **上下文 hint**（弱） | "我观察到的现象 / 经验 / 边界"，**不必防御性**——错了无所谓，因为它不喂 kill |

## 仍生效的硬约束（不变）

只有两类东西作为硬约束保留：

1. **`available_data` 硬过滤**（Stage 0 retrieval）：日频项目不应被 HFT 卡片带偏。这是数据物理约束，不算"约束 idea"。优先级：`explicit` > `frequency:<f>` 自动推断 > `none`。
2. **Stage 1 弱命名约束**：避免直接用 `reversal / momentum / value / quality / size / skewness / liquidity / ...` 这类 canonical labels 命名候选 —— 保护假设空间，鼓励描述机制本身。Stage 2/3 不再有此约束。

`mode ∈ {start, free, constrained}` 是 Stage 1 内部对单个 prompt 的严格度微调，不影响协议骨架。

## Model-lab 对齐

三段流水线对称镜像；研究对象从"因子机制"换成"模型改进机制"（loss / regularization / feature interaction / target construction / sample weighting / training window / model selection）。`transferable_moves` / `operative_claims` 字段语义同；ledger schema 同；两引擎对称协议同（Part B 评审 ModelFactorCaseSpec 可执行性而不是 factor.json）。

## 代码入口

| 关注点 | 模块 |
| --- | --- |
| Stage 0 入口 | `service.py::distribute_idea` / `model_idea.py::distribute_model_idea` |
| Symmetric prompt builder | `engine_prompts.py::build_prompt(engine, ctx)` |
| 代码库索引 | `codebase_index.py::build_codebase_snapshot` |
| Stage 4 card scaffold + cleanup | `experiment_card.py::scaffold_experiment_card` |
| `transferable_moves` / `operative_claims` 提取 | `service.py::_extract_frontmatter_field_items` / `model_idea.py::_read_card_frontmatter_field_items` |
| `available_data` 自动推断 | `scoring.py::infer_available_data_from_frequency` |
| Retrieval 多分量评分 | `scoring.py::score_card` + `service.py::_typed_rank_candidates` |
| Stage 3 validator | `draft_factor_validation.py::validate_draft_factor_file` / `draft_model_validation.py::validate_draft_model_file` |
| Audit pass-through | `custom_factors.py::CustomFactorSource.to_audit_dict` / `model_candidates.py::DraftModelSource.to_audit_dict` |

## 文档关系

- **`docs/end_to_end_workflow.md`** — 端到端使用流程（idea → 上线），**操作权威**。
- **本文（`docs/research_workflow.md`）** — 协议不变量，回答"为什么这样设计"。
- `docs/backend_draft_factor_workflow.md` / `docs/backend_draft_model_workflow.md` — Stage 3 后端细节。
- `docs/factor_promotion_checklist.md` — Stage 5 晋升清单。
- `docs/templates/<lab>_stage{1_reconcile,2_candidate}_contract.md` — 网页 GPT 项目"sources"模板。
