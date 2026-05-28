# research_log — pv_turnover_liquidity_light_v1

## 来源链路
- idea_id: `20260525T115846Z__idea-f3ca571d`
- Stage0: `alpha-lab model-idea distribute`（mode=start, stage=mechanism_discovery, available-data=daily_price_volume）
- Stage1: `ideas/<idea_id>/stage1_claude.md` + `stage1_codex.md`（generator + reviewer 合一）
- Stage2: `ideas/<idea_id>/stage1_reconcile.yaml`（model_stage1_reconcile_v1）+ `stage2_payload_v1.yaml`（model_stage2_candidate_output_v1）
- 本候选 = stage2 `model_candidate_payload`（contract_version=stage2_model_candidate_v1）的 materialize 结果。
- audience_chain: claude → codex → web_gpt_stage2（本轮三者均由 Claude Code 临时模拟，输出严格符合各自契约）。

## 实现的机制（v1）
- M1（in_contract_spec_variant，两引擎一致）：12 个价格/换手/流动性安全技术特征喂低自由度 ridge，预测 5 日 forward return 横截面排序。
- M2（in_contract_spec_variant，两引擎一致）：横截面 winsorize+zscore 预处理（missing→winsor→zscore，scope=date）。
- implementation_type = spec_variant：只动 ModelFactorCaseSpec 现有字段，无自定义 feature builder / estimator / sample_weight / target 代码。

## 显式不做（保留为上下文，未写入 v1 case_spec_payload）
- M3 换手惩罚 model_selection：两引擎对 `rank_ic_minus_turnover_penalty` 完整支持度判断冲突（claude=in_contract / codex=partial），按 reconcile 规则取更保守（partial）。本轮 `model_selection.enabled=false`，留作下一轮以 base spec 证据确认后启用。
- M5 vol×turnover 显式交互列：needs_extension（features 文件无该列，v1 禁 interaction expression / feature builder）。
- M6 换手样本加权：needs_extension（v1 无 sample_weight 训练入口）。
- M7 波动率状态组合降权：future_enhancement（portfolio construction / Level 2-3 边界，永久不进入 model-factor 候选）。

## 特征集（无 fundamental-like 列）
turnover_rate, turnover_rate_f, volume_ratio, circ_mv, free_share, atr_bfq, bias1_bfq, rsi_bfq_6, mtm_bfq, wr_bfq, vr_bfq, mfi_bfq
- `infer_fundamental_feature_columns` 仅判 pe_ttm/pb/ps_ttm 为基本面，已全部排除；circ_mv/free_share 为市场数据派生，不触发 PIT 基本面错误。
- feature_availability: mode=required_timestamp, column=known_at（features_safe_bfq_35 含 known_at 列）。

## prose vs case_spec_payload 冲突
- 无冲突。本 research_log 的描述与 `case_spec_payload` 一致；如有冲突以 `case_spec_payload` 为准。

## provenance sha 物化
- `stage2_payload_v1.yaml` 中 `stage2_payload_sha256` 为 placeholder `PENDING_STAGE3_MATERIALIZE`。
- 物化约定：对 `model_candidate_payload` 把 `provenance.stage2_payload_sha256` 置空，`json.dumps(sort_keys=True, ensure_ascii=False)` 后取 sha256。
- 物化结果：`3ef2e85af9f3debc3da65818cb9cdfcc6d433e08c7c26373f04a8b15c8110d35`（64 位 hex），写入 `model_candidate.json` 的 provenance。

## 运行
- 评价 profile: exploratory_screening；vault_export_mode: skip；内存预算 ALPHA_LAB_MAX_RSS_MB=14000。
- 目的：非正式端到端后端契约验证，**不晋升**、不注册前端正式版本。
