# Codex GUI Model-Lab Stage3 执行信封

每次把网页版 GPT 的 model-lab Stage2 输出交给 Codex GUI 时，使用下面这段固定开场。
它的作用是把 Codex GUI 的任务入口收窄到标准后端草稿模型流程（v1：spec 变体型）。

```text
你现在执行 alpha-lab Model-Lab Stage3 后端草稿模型流程。

必须遵守：
- AGENTS.md
- docs/templates/model_lab_stage3_backend_draft_prompt.md
- docs/backend_draft_model_workflow.md

只读取下面 Stage2 输出中的 model_candidate_payload 作为机器事实。
如果 human_summary、risk_controls、stage3_notes 或正文与 case_spec_payload 冲突，
以 case_spec_payload 为准，并写入 model_candidates/research/<candidate_name>/research_log.md。

禁止：
- 创建临时脚本
- 创建 notebook
- 创建散落的 .py 文件
- 修改 src/alpha_lab/model_factor、src/alpha_lab/factors 等 core 模块
- 引入自定义 feature builder code 或自定义 estimator code
- 绕过 alpha-lab 标准 model-factor pipeline
- 修改 model_candidates/promoted
- 修改前端正式注册
- 引入 portfolio construction 结论
- 引入 Level 3 execution / fill simulation / replay 语义

允许写入：
- model_candidates/research/<candidate_name>/model_candidate.json
- model_candidates/research/<candidate_name>/research_log.md
- configs/real_cases/model_factor/<candidate_name>_vN.yaml

必须完成：
1. preflight 检查
2. 写入 model_candidate.json（完整 case_spec_payload）
3. 写入 case YAML（与 case_spec_payload 字段一致）
4. 运行 validate-draft-model
5. 运行标准 model-factor backend experiment
6. 检查 artifact draft_model_source 审计字段（candidate_json_sha256、case_spec_sha256、feature_contract_sha256、source path）
7. 输出结果摘要和下一轮 case_spec_payload 字段修改建议

如果本地 Web Model Lab 已启动，也可以优先使用 `/model-lab` 的 `Draft Candidates`
面板执行同一流程：粘贴 payload -> 保存 Candidate -> Validate -> 生成 Case YAML ->
Validate + Run Screening。无论走 CLI 还是 Web UI，最终判断都以 validator 和
artifact 中的 `draft_model_source` hash 审计字段为准。

如果 validator、case_spec_payload schema、feature 字段可用性、PIT 检查或
artifact hash 审计失败，停止并报告失败，不要自行改写为另一个流程。

下面是 Stage2 输出全文：
<PASTE_STAGE2_OUTPUT_HERE>
```

## 最小执行命令

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab validate-draft-model model_candidates/research/<candidate_name>/model_candidate.json
```

```bash
UV_CACHE_DIR=/tmp/uv-cache uv run --no-sync --frozen alpha-lab real-case model-factor run \
  configs/real_cases/model_factor/<candidate_name>_v1.yaml \
  --evaluation-profile exploratory_screening \
  --screening-retrain-every-n-dates 40 \
  --render-report \
  --vault-export-mode skip \
  --draft-model-candidate model_candidates/research/<candidate_name>/model_candidate.json
```
