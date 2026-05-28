# Stage2 Vault-Refine Prompt（model-lab）

把网页版 GPT 的 **两份 Stage2 产物**（`model_stage1_reconcile_payload` +
`model_stage2_candidate_output`）交给 Claude Code 桌面版，在 **vault 知识约束下**
产出 **两份加强版、且仍然规范的** YAML。

这是一道 **Stage2.5 知识一致性加固**：网页 GPT 已经在「无知识库约束」下做完发散与质疑，
本步在它之上叠一层 vault grounding —— **增量加固，不推翻**。这一步**不跑实验、不写
`model_candidate.json` / case YAML**（那是 Stage3）。

与单因子 vault-refine 的区别：单因子加固的是 `factor_json_payload.code`（一个因子公式）；
model-lab 加固的是 `case_spec_payload`（一个 **spec 变体**：feature 选择、target 构造、
训练窗 / retrain、正则强度、model.family、feature_availability PIT）。机制身份是
**模型改进机制**（loss / regularization / feature interaction / target construction /
sample weighting / training window / model selection），不是因子方向。

---

## 角色与不变量

- 输入是网页 GPT 的两份 YAML，本步**只输出两份改写后的 YAML**（外加一个变更日志块）。
- `model_candidate_payload`（含完整 `case_spec_payload`）是唯一机器事实，驱动 Stage3
  实现；`reconcile` 只作机制背景与 research_log 记录，**不作实现入口**——所以真正的
  加固价值在 payload。
- 必须遵守：
  - `CLAUDE.md`（知识参考层 + 数据契约 + research rules）
  - `docs/templates/model_lab_stage1_reconcile_contract.md`
  - `docs/templates/model_lab_stage2_candidate_contract.md`
  - `docs/research_workflow.md`（协议不变量，含「Model-lab 对齐」一节）

---

## 本轮输入（调用时填）

```
idea_id:            <来自 manifest.json::idea_id>
reconcile_yaml:     <model_stage1_reconcile_payload 的路径或正文>
stage2_payload_yaml:<model_stage2_candidate_output 的路径或正文>
vault_root:         /mnt/c/quant/vault/quant-knowledge
features_path:      <case_spec_payload.features_path（必填，缺则上报）>
available_columns:  <features 文件真实表头 + 是否存在 known_at/available_at（必填，缺则上报）>
```

---

## 必做：先检索 vault，再改

针对 payload 选定的模型改进机制，**至少检索**以下层并记录命中卡的路径：

- `20_methods/Method - *.md` —— 估计方法 / 正则 / 交叉验证 / target 构造 / 样本加权的标准写法
- `30_factors/Factor - *.md` —— 作为输入的 feature 列的方向先验 / 已知失效条件 / 别名风险
- `60_playbooks/Playbook - *.md` —— pre-flight 检查、训练样本 / PIT / 过拟合的已知陷阱
- `10_concepts/Concept - *.md` —— 概念定义边界（如 forward return / horizon / 中性化）

vault 对本仓只读（`CLAUDE.md`）。本步不写任何 vault 文件。

---

## 改进边界（四条硬护栏）

1. **保持规范**：两份 YAML 顶层键封闭，**不得新增顶层字段**。vault 改动的理由走已有
   自由文本槽：payload 写进 `stage3_execution_notes[]`，reconcile 写进
   `unresolved_questions[]`，每条前缀 `vault_refine:` 并附引用卡路径。
   `case_spec_payload` 改完仍须能过 `validate-draft-model`：
   - `model_factor_case_spec_from_mapping` 解析无错（`model.family` 在
     linear/ridge/lasso/elastic_net/gbdt/xgboost/lightgbm/mlp 内；`training` 与
     `model_selection` 字段一致）。
   - `feature_columns` 全部是 `features_path` 真实表头列、不重名、不覆盖保留列
     （date/asset/known_at 等）。
   - `feature_availability` 守 PIT 合同：有 `known_at`/`available_at` 优先
     `mode=required_timestamp` 填 `column`；纯日频收盘后技术特征可用
     `mode=safety_lag` 且 `safety_lag_days>=1`；含 `pe_ttm/pb/ps_ttm/dv_ttm` 等
     基本面字段**不许** `column=null` 的 required_timestamp。
   - **spec_variant only**：不引入表达式列（如 `feat_a * turnover`）、自定义
     sample_weight、自定义 target、双窗口 selection/refit（除非 schema 已正式支持）、
     feature builder / estimator wrapper code。
   - 不出现 Level3 / execution_replay / fill_simulation / portfolio_construction /
     live_trading 关键词。

2. **provenance 不断链**：
   - `idea_id` 在两份里原样保留（外层 + `model_candidate_payload` 内层都在）。
   - 一旦改动 `model_candidate_payload`（含 `case_spec_payload`），把**两处**
     `provenance.stage2_payload_sha256`（外层与内层）**重置为占位 `""`**，交由
     Stage3 canonical-materialize 重算（不要手算 sha256）。
   - `audience_chain` 末尾追加 `"claude_code_stage2_vault_refine"`（外层 + 内层 +
     reconcile 三处都加）。
   - reconcile 的 `provenance.retrieval_pack_sha256` 原样保留。

3. **锁机制身份**：只许按 vault 加固**实现**（target horizon 对齐 Method/Concept 卡、
   特征选择规避 Playbook 记录的 alias/leak 陷阱、正则强度 / 训练窗对齐方法卡、PIT
   对齐、中性化、`feature_preprocess` winsorize/cross-section standardize）。**不许**
   换模型改进机制、不许把假设漂移成 Stage1 ledger 从未提过的东西。若 vault 检索表明
   该机制根本错误/被支配 → 写进 `unresolved_questions` 并**停下上报**，不得静默重写。

4. **每条实质改动必须引用驱动它的 vault 卡路径**。无引用的"优化"不合法。

禁止：跑实验、声称模型有效、portfolio / execution / replay / fill simulation、
写 `custom_models/**` 或 case YAML、改 core / promoted / 前端注册、写脚本/notebook、
引入自定义 feature/estimator code。

---

## escalation_triggers（出现即停、用中文报告、不自动修复）

- payload 选定机制与 vault Factor/Method 卡机器不可调和冲突（target 方向相反、
  方法本质不同、已知被支配）。
- `feature_columns` 需要 `features_path` 表头里没有的列。
- 加固只能靠引入自定义 code / future-return target / 全样本统计 / 被禁 target 才成立。
- `features_path` 或 `available_columns` 缺失，无法核对 `feature_columns` 真实性与
  `feature_availability` PIT 合理性。

---

## 输出格式

1. 先输出 **变更日志块**（普通正文，不在 YAML 内）：逐条列「改了什么 → 引用哪张 vault
   卡 → 为什么」，并显式声明机制身份未变 / 两处 sha256 已重置 / 三处 audience_chain
   已追加。
2. 再输出 **两个独立 YAML code block**，各自 schema-pure（顶层键封闭、`contract_version`
   正确、`quality_gate` 如实）：
   - block 1：`model_stage1_reconcile_v1`
   - block 2：`model_stage2_candidate_output_v1`（含完整 `case_spec_payload`，不是 patch）
3. YAML code block 外不要夹带模型有效性判断、买卖建议、Level3 / replay 建议。
