# Web GPT Factor Iteration Summary Prompt

把下面 prompt 粘贴到网页版 GPT 的因子项目对话中。它的任务是总结机制
和思路迭代，不要求也不允许保存原始聊天全文。

```text
你现在要为一次 alpha-lab 单因子优化做 Stage 4 机制侧总结。

目标：
- 不要归档原始对话。
- 从本项目完整对话中提炼出：初始假设、候选演化、关键设计取舍、
  被放弃的方向、最终保留的机制解释、未来可复用经验。
- 输出要适合后续大模型读取，用来启发新的因子发明和优化。
- 只总结你在对话中能看到的机制推理和设计演化；不要编造实验指标、
  artifact 路径、hash、PIT scan、RankIC、IR、coverage 等机器事实。
- 如果你见过 Codex GUI 贴回来的 Stage3 feedback，可以引用其中的结论，
  但必须标注它来自 feedback，而不是你自己重新验证。
- 中文优先；代码符号、字段名、指标名、专业缩写保留英文。

请严格按下面 schema 输出 Markdown，不要添加 schema 外的大段解释：

---
type: web_gpt_mechanism_summary
schema_version: web_gpt_mechanism_summary_v1
idea_id:
factor_name:
source: web_gpt_project_conversation
machine_fact_policy: "web_gpt_does_not_assert_artifact_facts"
---

# 网页版 GPT 机制迭代总结 - <factor_name>

## 1. 初始问题和目标

- `initial_research_question`:
- `initial_mechanism_thesis`:
- `target_market_frequency`:
- `intended_signal_direction`:

## 2. 候选演化轨迹

| Step | 候选机制/实现想法 | 为什么提出 | 为什么保留/修改/放弃 |
| --- | --- | --- | --- |
| 1 |  |  |  |
| 2 |  |  |  |
| final |  |  |  |

## 3. 最终机制压缩

- `final_mechanism_thesis`:
- `observable_proxy_plan`:
- `economic_intuition`:
- `why_not_a_trivial_alias`:
- `expected_failure_modes`:

## 4. 设计取舍

- `kept_choices`:
  - 
- `discarded_choices`:
  - 
- `parameter_or_proxy_lessons`:
  - 

## 5. 可迁移动作 emergent_moves

每条都要写成未来可以复用的动作，而不是泛泛评价。

- `move`:
  `when_to_try`:
  `implementation_hint`:
  `risk`:

## 6. 弱观察 operative_claims

这些不是硬事实，不作为 kill 规则，只作为未来探索 hint。

- 

## 7. 负面约束 negative_constraints

以后不要重复踩的坑、不要再走的过宽方向、容易变成旧因子别名的表达：

- 

## 8. 给 Codex GUI / Stage 4 合并者的交接

- `must_verify_from_artifacts`:
  - RankIC / IR / coverage / regime / cost / PIT / hashes
- `possible_final_writeback_title`:
- `suggested_tags`:
- `open_questions_for_next_research`:
```

