# Codex GUI Stage3 执行信封

每次把网页版 GPT 的 Stage2 输出交给 Codex GUI 时，使用下面这段固定开场。
它的作用是把 Codex GUI 的任务入口收窄到标准后端草稿因子流程。

```text
你现在执行 alpha-lab Stage3 后端草稿因子流程。

必须遵守：
- AGENTS.md
- docs/templates/stage3_backend_draft_factor_prompt.md
- docs/backend_draft_factor_workflow.md

只读取下面 Stage2 输出中的 factor_json_payload 作为机器事实。
如果 human_summary、machine_contract、stage3_notes 或正文与 factor_json_payload 冲突，以 factor_json_payload 为准，并写入 custom_factors/research/<factor_name>/research_log.md。

禁止：
- 创建临时脚本
- 创建 notebook
- 创建散落的 .py 文件
- 绕过 alpha-lab 标准 pipeline
- 修改 custom_factors/promoted
- 修改前端正式注册
- 引入 portfolio construction 结论
- 引入 Level 3 execution / fill simulation / replay 语义

允许写入：
- custom_factors/research/<factor_name>/factor.json
- custom_factors/research/<factor_name>/research_log.md
- configs/real_cases/single_factor/<factor_name>_vN.yaml

必须完成：
1. preflight 检查
2. 写入 factor.json（保留 Stage2 输出中的 provenance 块；不要删 idea_id / stage2_payload_sha256 / audience_chain）
3. 写入或更新 case YAML 时，根字段写入 `project_slug` / `archive_identity` / `evaluation_profile`；`project_slug` 只能来自本信封或用户明确给出的项目上下文，不能从名称猜测
4. 覆盖已有 case YAML 前读取旧 YAML，并保留已有 `project_slug` / `archive_identity` / `evaluation_profile`，除非用户明确要求修改
5. 运行 validate-draft-factor
6. 运行标准 single-factor backend experiment
7. 检查 artifact custom_factor_source 审计字段（含 provenance.idea_id）
8. 追加一行 trail 到 research_log.md，并按 stage3_backend_draft_factor_prompt.md「输出要求」固定 schema 输出 single_factor_stage3_feedback_v1 反馈包（供网页版 GPT 迭代下一版 Stage2 payload）

forbidden_actions（Stage 3 执行者硬约束，违反任一即视为本轮失败）：
- 自行补全 Stage2 payload 中缺失的字段（缺什么停下来回写到 research_log 的 deferred 段，不要凭机制名补）
- 改写 factor_json_payload 中的 code / required_columns / pit_assumption（与机制冲突时以 payload 为准 + 写 research_log）
- 跳过 validate-draft-factor 或 silently 接受 validator warnings
- 删除或重写 provenance 块（provenance 是合同审计字段，不是注释）
- 写临时脚本 / notebook / 散落 .py
- 修改 src/alpha_lab/factors / custom_factors/promoted / 前端正式注册

escalation_triggers（出现以下任一情况，停下并以中文报告，不继续自动修复）：
- validate-draft-factor 报错
- artifact 缺 code_sha256 / factor_json_sha256 / source path / provenance.idea_id
- 实验运行抛 PIT / leakage / cross_section_full_sample 硬错
- factor_json_payload 中 required_columns 包含 factor_recipe.py 未注册列名
- Stage2 payload 与 stage3_notes / human_summary 出现机器不可调和冲突

下面是 Stage2 输出全文：
<PASTE_STAGE2_OUTPUT_HERE>
```

## 最小执行命令

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab validate-draft-factor custom_factors/research/<factor_name>/factor.json
```

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab real-case single-factor run configs/real_cases/single_factor/<factor_name>_vN.yaml --evaluation-profile exploratory_screening --render-report --vault-export-mode skip
```
