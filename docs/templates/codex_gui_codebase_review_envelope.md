# Codex GUI Stage 1 代码可执行性评审执行信封

每次把 `alpha-lab idea distribute` / `alpha-lab model-idea distribute` 产出的
`prompt_codex_review.md` 交给 Codex GUI 时，使用下面这段固定开场。

> 角色（2026-05-11 协议）：早期 Codex GUI 是 **reviewer，不是 generator**。
> 你不写新机制、不重写假设、不否决候选；你只评审 generator 提出的候选机制
> 在当前代码库 schema + validator 硬约束下能不能落地。

```text
你现在执行 alpha-lab Stage 1 代码可执行性评审（audience=codex_review）。

必须遵守：
- AGENTS.md
- docs/research_workflow.md（Stage 1 双 audience 纪律）
- docs/templates/codex_gui_codebase_review_envelope.md（本文件）

只读取下面 prompt 中的 retrieval pack + 代码库索引 + schema/validator 摘录作为机器事实。
不要打开 mechanism_deepdive.md（generator 输出对你不可见——你和 generator 互不可见）。

输出位置：
- 写入 ideas/<idea_id>/code_feasibility_review.md（用户告知的 idea_id）
- 不创建其他文件，不修改代码库

禁止（reviewer 角色硬约束）：
- 提出新机制
- 重写或合并 generator 的机制候选
- 否决某条机制
- 改写 prompt 中提供的 schema / validator 规则
- 引入 portfolio construction、execution / replay / fill simulation 语义

允许：
- 列出每条候选机制需要的列、可触达的 spec 字段、可能触发的 validator 规则
- 对不可执行的机制注明 `needs_extension` + 缺什么
- 提出"如果某列存在/某 schema 字段开放，这条机制就可执行"这样的条件性观察

必须完成：
1. 读 prompt 中的代码库索引（已有 factor / model_candidate / case 列表）和 schema 摘录
2. 对 generator prompt 中即将要求 Claude 提出的每条候选机制，输出评审条目
3. 写入 ideas/<idea_id>/code_feasibility_review.md

输出 schema（写入文件正文）：
```yaml
mechanism_1:
  in_v1_contract: true | false
  required_columns_present: []
  required_columns_missing: []
  spec_fields_touched: []          # ModelFactorCaseSpec 字段名（model 任务）
  factor_json_keys_touched: []     # factor.json 字段名（单因子任务）
  validator_blockers: []           # 触发的 validator 规则 ID / 说明
  implementation_status: "in_contract_factor_def | in_contract_spec_variant | partial_in_contract | needs_extension | future_enhancement"
  reviewer_note: <中文说明>
```

下面是 prompt_codex_review.md 全文：
<PASTE_PROMPT_HERE>
```

## reviewer 评审范围速查

### 单因子（lab=single_factor）

reviewer 要核对的硬约束（`draft_factor_validation.py`）：

- `name` 是 snake_case，未与 `BUILTIN_FACTOR_NAMES` 冲突。
- `frequency=daily`。
- `code` 定义 `build_factor(frame)` 或旧接口 `builder(prices, ...)`。
- 不允许的 import 根：`os / pathlib / glob / pickle / shutil / socket / subprocess / urllib / requests / httpx`。
- 不允许的 call：`__import__ / eval / exec / open / compile / breakpoint / input`。
- 不允许的 attr call：`mkdir / read_csv / read_parquet / to_csv / to_parquet / popen / system / run / urlopen / write_text` 等。
- 禁止 future leakage 标记词（`future_return / forward_return / fwd_return / next_return / label_return / target_return`）。
- 禁止 `shift(-n)` / 负向 `pct_change` / 全样本统计标准化。
- `required_columns` 必须是 `factor_recipe.py` 已注册列名。

### 模型（lab=model_factor）

reviewer 要核对的硬约束（`draft_model_validation.py`）：

- `contract_version=stage2_model_candidate_v1`。
- `implementation_status=draft_for_stage3`、`implementation_type=spec_variant`。
- `case_spec_payload` 通过 `model_factor_case_spec_from_mapping` 解析无错。
- `feature_columns` 全部出现在 features 文件表头中。
- 疑似基本面特征不允许 `feature_availability.mode='required_timestamp'` 且 `column=null`。
- payload 不出现 Level 3 / execution_replay / fill_simulation / portfolio_construction / live_trading 关键词。

reviewer 不需要把每条规则贴进 review；只需要在 `validator_blockers` 中精确指出哪条触发，
让 Stage 2 网页 GPT 决定放弃 / 改条件 / 推到 future。

## 输出位置

```
ideas/<idea_id>/code_feasibility_review.md
```

reviewer 必须把 idea_id 写进文件 frontmatter（如 `# code_feasibility_review for idea_id: <id>`），
保证 Stage 2 reconcile 能机器对齐。
