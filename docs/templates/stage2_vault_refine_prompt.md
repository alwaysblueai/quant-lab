# Stage2 Vault-Refine Prompt（single-factor）

把网页版 GPT 的 **两份 Stage2 产物**（`single_factor_stage1_reconcile_payload` +
`single_factor_stage2_candidate_output`）交给 Claude Code 桌面版，在 **vault 知识约束下**
产出 **两份加强版、且仍然规范的** YAML。

这是一道 **Stage2.5 知识一致性加固**：网页 GPT 已经在「无知识库约束」下做完发散与质疑，
本步在它之上叠一层 vault grounding —— **增量加固，不推翻**。这一步**不跑实验、不写
`factor.json` / case YAML**（那是 Stage3）。

---

## 角色与不变量

- 输入是网页 GPT 的两份 YAML，本步**只输出两份改写后的 YAML**（外加一个变更日志块）。
- `stage2_payload` 是唯一机器事实，驱动 Stage3 实现；`reconcile` 只作机制背景与
  research_log 记录，**不作实现入口**——所以真正的加固价值在 payload。
- 必须遵守：
  - `CLAUDE.md`（知识参考层 + 数据契约 + research rules）
  - `docs/templates/single_factor_stage1_reconcile_contract.md`
  - `docs/templates/single_factor_stage2_candidate_contract.md`
  - `docs/research_workflow.md`（协议不变量）

---

## 本轮输入（调用时填）

```
idea_id:            <来自 manifest.json::idea_id>
reconcile_yaml:     <single_factor_stage1_reconcile_payload 的路径或正文>
stage2_payload_yaml:<single_factor_stage2_candidate_output 的路径或正文>
vault_root:         /mnt/c/quant/vault/quant-knowledge
available_columns:  <可用 prices + 已注册 intraday/daily 列名（必填，缺则上报）>
```

---

## 必做：先检索 vault，再改

针对 payload 选定的机制，**至少检索**以下层并记录命中卡的路径：

- `30_factors/Factor - *.md` —— 因子定义 / 方向先验 / 已知失效条件
- `20_methods/Method - *.md` —— 估计方法 / 公式的标准写法
- `60_playbooks/Playbook - *.md` —— pre-flight 检查、已知陷阱
- `10_concepts/Concept - *.md` —— 概念定义边界

vault 对本仓只读（`CLAUDE.md`）。本步不写任何 vault 文件。

---

## 改进边界（四条硬护栏）

1. **保持规范**：两份 YAML 顶层键封闭，**不得新增顶层字段**。vault 改动的理由走已有
   自由文本槽：payload 写进 `stage3_execution_notes[]`，reconcile 写进
   `unresolved_questions[]`，每条前缀 `vault_refine:` 并附引用卡路径。
   `factor_json_payload.code` 改完仍须能过 `validate-draft-factor`：name snake_case、
   `required_columns` 是 `factor_recipe.py` 真列、定义 `build_factor(frame)`、
   无 `shift(-n)` / 负向 `pct_change` / future label、无全样本均值/标准差标准化
   （横截面 demean / rank 例外）、无 `open/网络/subprocess/eval/exec/os/pathlib/...`、
   rolling/expanding 按 `asset` 分组。

2. **provenance 不断链**：
   - `idea_id` 在两份里原样保留。
   - 一旦改动 `factor_json_payload`，把 `provenance.stage2_payload_sha256` **重置为
     占位 `""`**，交由 Stage3 canonical-materialize 重算（不要手算 sha256）。
   - `audience_chain` 末尾追加 `"claude_code_stage2_vault_refine"`（两份都加）。
   - reconcile 的 `provenance.retrieval_pack_sha256` 原样保留。

3. **锁机制身份**：只许按 vault 加固**实现**（公式对齐 Method 卡、PIT 对齐、中性化、
   列选择、规避 Playbook 记录的陷阱）。**不许**换机制、不许把假设漂移成 Stage1 ledger
   从未提过的东西。若 vault 检索表明该机制根本错误/被支配 → 写进 `unresolved_questions`
   并**停下上报**，不得静默重写。

4. **每条实质改动必须引用驱动它的 vault 卡路径**。无引用的"优化"不合法。

禁止：跑实验、声称因子有效、portfolio / execution / replay / fill simulation、
写 `custom_factors/**` 或 case YAML、改 core / promoted / 前端注册、写脚本/notebook。

---

## escalation_triggers（出现即停、用中文报告、不自动修复）

- payload 选定机制与 vault Factor/Method 卡机器不可调和冲突（方向相反、公式本质不同）。
- `required_columns` 需要 `factor_recipe.py` 未注册的列。
- 加固只能靠引入 future label / 全样本统计 / 被禁 import 才成立。
- `available_columns` 缺失，无法核对 `required_columns` 真实性。

---

## 输出格式

1. 先输出 **变更日志块**（普通正文，不在 YAML 内）：逐条列「改了什么 → 引用哪张 vault
   卡 → 为什么」，并显式声明机制身份未变 / sha256 已重置 / audience_chain 已追加。
2. 再输出 **两个独立 YAML code block**，各自 schema-pure（顶层键封闭、`contract_version`
   正确、`quality_gate` 如实）：
   - block 1：`single_factor_stage1_reconcile_v1`
   - block 2：`single_factor_stage2_candidate_output_v1`
3. YAML code block 外不要夹带因子有效性判断、买卖建议、Level3 / replay 建议。
