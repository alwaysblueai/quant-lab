# 端到端工作流（idea → 上线候选）

本文是 alpha-lab / model-lab 从"模糊想法"到"可上线因子/模型"的权威使用流程，2026-05-11 修订版。Stage 边界、文件总量约束、前后端分工、经验归档规则都以本文档为准；其它 docs（`research_workflow.md` / `backend_draft_*workflow.md`）是各 Stage 的细化说明，必须与本文一致。

## 总览

```
0. Stage 0 — Claude API 检索 + 分发                                 → 后端
1. Stage 1 — Claude Code + Codex GUI 并行（任务相同，互不可见）     → 桌面 agent
2. Stage 2 — 网页 GPT 综合两份输出 → 合同 + 接收 Stage 3 回执做迭代  → 网页 GPT
3. Stage 3 — Codex GUI 后端快速迭代（validator + pipeline）         → 后端
4. Stage 4 — 经验归档（不管成败都总结，由用户手工写入 vault）       → 后端 → vault
5. Stage 5 — 晋升 + 前端完整报告（仅通过审核的候选）                → 前端
```

**前端只做两件事**：

- 输入模糊想法 + 复制 / 预览 Stage 0 产出文件
- 展示已通过审核的因子/模型的完整可视化报告

其他全部在后端，每个 Stage 只产出**确定数量**的文件，不允许散落写。

## Stage 0 — Claude API 检索 + 分发

**调用 LLM**：是。`alpha-lab idea distribute` 内部经过 `research_bridge.service.explore_idea`，调用 Claude API 做 retrieval rerank / query expansion / mechanism index lookup。需要 `ANTHROPIC_API_KEY` 与已 build 的 `mechanism_index` sidecar（见 CLAUDE.md）。LLM 在这一步**决定**哪些 vault 卡对当前 idea 有帮助，不是简单的语义近邻。

### 入口

```bash
# 单因子
alpha-lab idea distribute --idea "..." --lab single_factor

# 模型
alpha-lab model-idea distribute --idea "..."
```

也支持从前端发起：unified 首页输入框 + "调用 Stage 0 分发"按钮 → 后端跑同一 CLI → 前端预览 `ideas/<idea_id>/` 文件 + 一键复制。

### 产出（精简到 5 个，多了影响下游噪音）

| 文件 | 用途 | 谁消费 |
|---|---|---|
| `manifest.json` | idea_id、retrieval 诊断、codebase 快照、引擎列表 | 审计 + Stage 3 反查 |
| `retrieval_pack.md` | LLM 检索后的卡片摘要 + `transferable_moves` + cross-card synthesis + codebase 索引 | 网页 GPT（共享上下文复印件） |
| `prompt_claude.md` | 给 Claude Code 的完整任务 prompt（generator + reviewer 合一） | Claude Code |
| `prompt_codex.md` | 给 Codex GUI 的**同一任务** prompt | Codex GUI |
| `stage2_input.md` | 网页 GPT 入口模板（reconcile 槽位 + Stage 2 合同提示） | 网页 GPT |

> 删除原协议里的 `retrieval_log.md`（信息并入 `manifest.json::retrieval_diagnostics`）和独立的 `reconcile.md`（合并入 `stage2_input.md`），让网页 GPT 只对单一文件粘贴。

### 两份 prompt 的关系

**字节级相同**，唯一差别是开头"你是 Claude Code"/"你是 Codex GUI"自我标识。任务、上下文、schema、validator 规则、输出格式都一致。每份 prompt 体内都要求：

1. 先 generator：提 3-8 条互补候选机制（ledger schema）
2. 再 reviewer：对自己提的每条机制做 v1 contract 可执行性评审

## Stage 1 — Claude Code + Codex GUI 并行（任务相同）

**两个引擎做同一份工作**（generator + reviewer 合一）。不是分工，是**冗余**。模型特点不同 → 输出各有优劣 → 网页 GPT 综合取长补短。

| 引擎 | prompt | 输出 |
|---|---|---|
| Claude Code | `prompt_claude.md` | `ideas/<idea_id>/stage1_claude.md` |
| Codex GUI | `prompt_codex.md` | `ideas/<idea_id>/stage1_codex.md` |

每份输出含两段：

```markdown
## Part A — Mechanism candidates（generator）
mechanism_1:
  hypothesis: ...
  signal_sketch: ...
  data_needs: ...
  concern: ...
... (3-8 个)

## Part B — Code feasibility review（reviewer）
mechanism_1:
  in_v1_contract: true | false
  required_columns_missing: []
  validator_blockers: []
  implementation_status: in_contract / partial / needs_extension / future_enhancement
  reviewer_note: ...
... (与 Part A 一一对应)
```

### Stage 1 纪律

- 两引擎互不可见（信息隔离）
- 不否决候选；不可执行的只标 `needs_extension`
- 整个 Stage 1 只产 2 个文件，不写中间稿、不分多回合保存

## Stage 2 — 网页 GPT 合同化 + 迭代反馈

### 网页 GPT 项目固定"来源"（**只放 3 个**）

| 文件 | 阶段角色 |
|---|---|
| `docs/templates/<lab>_source_pack.md` | 总览（单因子/模型选一） |
| `docs/templates/<lab>_stage1_reconcile_contract.md` | Stage 2.1 输入合同 |
| `docs/templates/<lab>_stage2_candidate_contract.md` | Stage 2.2 输出合同 |

> Stage 3 envelope 是给 Codex GUI 的开场提示，**不**放网页 GPT 项目（放进去 GPT 会越界写代码或自我执行）。

### 两步输出

| 步骤 | 输入 | 输出 | 写入 |
|---|---|---|---|
| 2.1 reconcile | `stage1_claude.md` + `stage1_codex.md` + `retrieval_pack.md` | `stage1_reconcile_payload`（YAML） | `ideas/<idea_id>/stage1_reconcile.yaml` |
| 2.2 candidate | reconcile payload + 用户取舍 | `factor_json_payload` 或 `model_candidate_payload`（含 provenance） | `ideas/<idea_id>/stage2_payload_v<n>.json` |

### Stage 2.1 综合规则

- 取两引擎机制候选的**并集**
- 冲突的 `implementation_status` 取**更保守**那一方（reviewer 严格优先）
- `source_engines` 字段保留每条机制的来源（claude / codex / both）

### Stage 2.2 输出契约

- 必须含 `provenance.{idea_id, stage2_payload_sha256, audience_chain}`
- 单因子：完整 `factor_json_payload`
- 模型：完整 `case_spec_payload`（不接受 patch）

### 迭代回灌

Stage 3 跑完之后，把后端实验摘要（见 Stage 3）**直接粘回**同一个 GPT 项目 → GPT 出 `stage2_payload_v<n+1>.json`。`idea_id` 不变，`stage2_payload_sha256` 更新。这是网页 GPT 的"正向迭代"路径，**不需要额外文件做中转**。

## Stage 3 — 后端快速迭代

### Codex GUI 入口

`docs/templates/codex_gui_{,model_}stage3_execution_envelope.md`（envelope 已含 `forbidden_actions` + `escalation_triggers`，违反任一即视为本轮失败）。

### 只写 3 类文件（最小集，每轮**覆盖**同一组路径）

| 路径 | 内容 |
|---|---|
| `custom_factors/research/<f>/factor.json` 或 `model_candidates/research/<c>/model_candidate.json` | Stage 2 payload 完整复制 |
| `custom_factors/research/<f>/research_log.md` 或 `model_candidates/research/<c>/research_log.md` | append-only：每轮一行 timestamp + 摘要 |
| `configs/real_cases/{single_factor,model_factor}/<name>_v<n>.yaml` | case spec（模型侧从 `case_spec_payload` materialize） |

### 跑（exploratory_screening → default_research 渐进）

```bash
# 单因子
alpha-lab validate-draft-factor custom_factors/research/<f>/factor.json
alpha-lab real-case single-factor run configs/.../<f>_v1.yaml \
  --evaluation-profile exploratory_screening \
  --render-report --vault-export-mode skip

# 模型
alpha-lab validate-draft-model model_candidates/research/<c>/model_candidate.json
alpha-lab real-case model-factor run configs/.../<c>_v1.yaml \
  --evaluation-profile exploratory_screening \
  --screening-retrain-every-n-dates 40 \
  --render-report --vault-export-mode skip \
  --draft-model-candidate model_candidates/research/<c>/model_candidate.json
```

### 迭代摘要（粘回网页 GPT）

每轮 pipeline 跑完后，Codex GUI 写**1 行**到 `research_log.md`，并输出一段 markdown 直接粘回网页 GPT：

```markdown
### Round v<n> (idea_id=<id>, payload_sha256=<8-char>)
- profile: exploratory_screening / default_research
- coverage / IC / rank IC / decay / turnover / cost-aware：<数据>
- 硬失败：PIT / alias / regime fragility / cost-aware（任一，没有就略）
- 下轮调整建议：仅在 <允许字段> 内
```

**不另生成"实验摘要文件"。** `metrics.json` / `summary.md` / `integrity_report.md` 是 pipeline 自动产物，由 Stage 4 阶段总结时引用，不属于 Stage 3 手工写入。

### Stage 3 退出条件

通过 `default_research`（完整指标 + 完整 pipeline 不报错 + provenance audit OK）→ 进 Stage 4 总结。

## Stage 4 — 经验归档（无论成败都做，新增）

**关键定位**：多轮迭代结束后**一次性**产出经验总结（不是每轮都写）。无论 Stage 3 最终成功还是失败，都要写一份，把多轮散落的研究信号收敛成可被未来 idea 借用的经验。

### 唯一产物

```
ideas/<idea_id>/experiment_card.md
```

### 字段

```markdown
---
idea_id: 20260511T143000Z__signed-jump-reversal
lab: single_factor
outcome: promoted | killed | parked
rounds: <n>
final_artifact_path: custom_factors/research/<f>/factor.json 或 promoted/...
final_metrics_sha256: <run_manifest 中的关键 hash 摘要>
---

# 总结

## 关键改动轨迹
- v1: <初始机制 + 关键参数>
- v2: <做了什么改动 + 为什么>
- ...
- v<n>: 最终形态

## 成功路径分析（如果 outcome=promoted）
- 哪条机制被数据验证
- 哪些假设被支持，哪些被部分修正

## 失败路径分析（如果 outcome=killed/parked）
- 哪条 assumption 在数据上被否定
- 失败类型：PIT / alias / regime fragility / cost-aware / 训练样本不足 / 其他
- 是死路（killed）还是 future_enhancement（parked，等下次条件成熟）

## emergent_moves（**主回写字段**）
- 本次实验浮现的、可被未来 idea 借用的新动作（每条要具体可执行）

## operative_claims（**弱观察**）
- 观察到的现象 / 经验 / 边界条件（错了也无所谓，给未来探索做素材）
```

### 用户手工把这份 card 落到 vault

```python
from alpha_lab.reporting import export_experiment_card
export_experiment_card(
    card_path="ideas/<idea_id>/experiment_card.md",
    name="signed-jump-reversal-202605",
)
# → /mnt/c/quant/vault/quant-knowledge/50_experiments/Exp - 202605 - signed-jump-reversal.md
```

### Stage 4 写完即清理（**避免知识库膨胀的关键**）

- 删除 `ideas/<idea_id>/stage1_*.md` / `stage2_payload_v*.json` / `prompt_*.md` / `retrieval_pack.md` / `stage2_input.md`（信息已沉淀在 `experiment_card.md` + vault）
- 保留 `ideas/<idea_id>/manifest.json` + `experiment_card.md`（idea_id 索引最小集）
- 删除 `custom_factors/research/<f>/` 整个目录（若 outcome=killed/parked），或拷到 promoted 之后删（若 outcome=promoted）

vault 里只留浓缩经验，repo 里只留上线候选。

## Stage 5 — 晋升 + 前端完整报告

仅当 Stage 4 outcome = `promoted`：

```bash
cp -r custom_factors/research/<f>/ custom_factors/promoted/<f>/
# 写 promotion_card.md（机制 + alias 矩阵 + cost 假设 + provenance）
# 更新 docs/promotion_log.md
```

`provenance.idea_id` 在 `factor.json` / `model_candidate.json` 里保留 —— 这是上线候选追溯到原 idea 的唯一证据链，**不要删**。

### 前端职责

```bash
alpha-lab web unified --vault-root /mnt/c/quant/vault/quant-knowledge
```

- 首页：模糊想法输入框（→ Stage 0）+ Stage 0 文件预览/复制
- `/model-lab` Draft Candidates 面板：仅对 `promoted/` 候选生成完整 `default_research` 报告 + artifact 可视化
- 不在前端做 Stage 1-4 任何工作

## 文件总量约束（一份完整流程跑完）

| 阶段 | 产生新文件 | 累计 |
|---|---|---|
| 0 | 5 | 5 |
| 1 | 2（两引擎各一份） | 7 |
| 2 | 1 reconcile + 每轮 1 payload | 8+rounds |
| 3 | 3（覆盖式写） | 11+rounds |
| 4 | 1（experiment_card）+ **删除** Stage 0/1/2/3 中除 manifest 外的临时文件 | manifest + card + promoted（视情况） |
| 5 | promoted/ 目录拷贝 + promotion_card.md（仅 outcome=promoted） | 上线 |

**Stage 4 清理后**：每个 idea 在 repo 里只留 2 个文件 + 可选 promoted/，vault 经验只增不减。

## 何时回到 Stage 0 重新分发

不是每次失败都要回 Stage 0：

- Stage 3 数据失败但机制仍可救（参数 / 字段 / 中性化层）→ **回 Stage 2 出 v2**
- Stage 4 outcome=parked、想换一个相邻 idea 提法 → **新建 idea_id**，**不**复用旧 `ideas/<id>/`（idea_id 是审计单元，不要混层）
- reviewer 评审两个引擎都判断"没有任何机制在 v1 contract 内可执行" → 要么回 Stage 0 换 idea 提法、要么先做 contract 扩展再回来

## 权威文档索引

| 主题 | 文档 |
|---|---|
| 本文（端到端入口） | `docs/end_to_end_workflow.md` |
| 协议骨架 | `docs/research_workflow.md` |
| 单因子 Stage 3 流程 | `docs/backend_draft_factor_workflow.md` |
| 模型 Stage 3 流程 | `docs/backend_draft_model_workflow.md` |
| 晋升清单 | `docs/factor_promotion_checklist.md` |
| Stage 3 执行硬约束 | `docs/templates/codex_gui_{,model_}stage3_execution_envelope.md` |
| Stage 2 合同 | `docs/templates/<lab>_stage{1_reconcile,2_candidate}_contract.md` |

## 三种"必须停下"的硬失败（任何阶段通用）

1. **provenance 链断裂**：Stage 2 payload 缺 `idea_id`，或 Stage 3 写入时被改 → 整个 idea 链路审计失效，必须回 Stage 2 重出
2. **validator 报错**：不要"自行修复"——把错误回灌到 `research_log.md` 的 deferred 段，下一轮让 Stage 2 GPT 解决
3. **PIT / cross-section / future leakage 实验失败**：硬失败，本机制在数据上被否决，但**不在 Stage 1/2 KILL**——只在 Stage 3 数据上 KILL，记录 `emergent_moves` 回写 vault 才有价值
