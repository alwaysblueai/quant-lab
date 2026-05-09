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
2. 写入 factor.json
3. 运行 validate-draft-factor
4. 运行标准 single-factor backend experiment
5. 检查 artifact custom_factor_source 审计字段
6. 输出结果摘要和下一步建议

如果 validator、字段可用性、leakage 检查或 artifact hash 审计失败，停止并报告失败，不要自行改写为另一个流程。

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
