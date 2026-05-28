idea_id: `20260525T115846Z__idea-f3ca571d`
engine: codex
lab: model_factor
generated: 2026-05-25

## Part A — Mechanism candidates (generator)

mechanism_1:
  hypothesis: "把换手、量比与流动性/规模列（circ_mv, free_share）连同少量价格技术列一起喂给低自由度线性模型，对未来一周收益做横截面排序；流动性维度提供'可信度'背景，活跃且体量适中的股票排序信号噪声更低。"
  inspired_by: [K7:low_freedom_linear_score_blend, K4:universe_fixed_before_sort]
  fusion_of: ["Z-Score Normalization and Aggregation", "Factor Research Operating Manual"]
  novel_delta: "显式把流动性/规模描述符纳入收益预测特征组，而不是仅做 universe 过滤；让模型在排序时把流动性当连续条件信息。"
  signal_sketch: "feature_columns 含 turnover_rate, turnover_rate_f, volume_ratio, circ_mv, free_share + atr_bfq, bias1_bfq, rsi_bfq_6, mtm_bfq, wr_bfq, vr_bfq, mfi_bfq；ridge 预测 forward_return(5d)。"
  data_needs: [turnover_rate, turnover_rate_f, volume_ratio, circ_mv, free_share, atr_bfq, bias1_bfq, rsi_bfq_6, mtm_bfq, wr_bfq, vr_bfq, mfi_bfq]
  concern: "circ_mv / free_share 量纲大、分布偏，必须横截面标准化；它们是市场数据派生（非 as-reported 基本面），known_at PIT 可控。"

mechanism_2:
  hypothesis: "排序任务的稳健性受底层并列值（零成交、涨跌停同价）影响很大；在收益排序场景，winsorize+zscore 的 tie 行为与 group_scope=date 的选择会显著改变截面分布与排序质量。"
  inspired_by: [K5:differentiable_rank_weight_factor_construction, K5:operator_semantics_catalog_with_domain_signature]
  fusion_of: ["Tie-Handling in Cross-Sectional Ranking", "Z-Score Normalization and Aggregation"]
  novel_delta: "把 tie/标准化语义作为显式可审计选择登记，而非默认；在当前 spec 内通过 feature_preprocess 表达。"
  signal_sketch: "cross_sectional_transform=winsorize_zscore, cross_sectional_group_scope=date, missing_policy=median_impute。"
  data_needs: []
  concern: "当前 spec 的 cross_sectional_transform 走 zscore 而非 rank；若要纯 rank tie 语义需确认 spec 是否支持 'rank' 选项。"

mechanism_3:
  hypothesis: "训练窗口选择（rolling 120 日、每 40 日 retrain）对短 horizon 排序的稳定性比模型族更重要；固定一个保守 rolling 窗口比频繁 refit 更不易过拟合。"
  inspired_by: [K4:broker_factor_research_stage_gate]
  fusion_of: []
  novel_delta: "把 training.window_type / train_window_n_dates / retrain_every_n_dates 作为主要调研轴，模型族保持 ridge 固定。"
  signal_sketch: "training.window_type=rolling, train_window_n_dates=120, retrain_every_n_dates=40, min_train_rows=50000, min_score_assets=2000。"
  data_needs: []
  concern: "短窗口在 regime 切换处样本不足；min_train_dates/min_train_rows 需兜底，否则训练通过率不稳。"

mechanism_4:
  hypothesis: "用换手惩罚的 selection metric 选超参，使最终信号偏向低换手、可持有。"
  inspired_by: []
  fusion_of: []
  novel_delta: "selection 阶段引入成本/换手意识，而非纯 rank IC。"
  signal_sketch: "model_selection.metric=rank_ic_minus_turnover_penalty。"
  data_needs: []
  concern: "需确认该 metric 在当前 spec parser 与 model_selection 实现里都已正式支持；若仅部分支持应保守降级。"

mechanism_5:
  hypothesis: "把市场波动率状态写成因子权重的仿射函数（高波动期系统性降权），在高波 regime 自动收缩暴露。"
  inspired_by: [K3:volatility_state_factor_weight_rule, K6:volatility_bucket_deviation_budget_schedule]
  fusion_of: ["Conditional Mean-Variance Multifactor Portfolio with Volatility Management"]
  novel_delta: "用波动率状态做条件降权，理论上能改善高波期净收益。"
  signal_sketch: "组合层按 sigma_t 调整暴露 / 单名偏离预算。"
  data_needs: []
  concern: "这是 portfolio construction / 组合优化语义（Level 2/3 边界），不属于 model-factor 预测信号 spec；仅作上下文。"

## Part B — Code feasibility review (reviewer)

mechanism_1:
  in_v1_contract: true
  required_columns_present: [turnover_rate, turnover_rate_f, volume_ratio, circ_mv, free_share, atr_bfq, bias1_bfq, rsi_bfq_6, mtm_bfq, wr_bfq, vr_bfq, mfi_bfq]
  required_columns_missing: []
  spec_fields_touched: [feature_columns, model.family, target.horizon, feature_preprocess, feature_availability]
  validator_blockers: []
  implementation_status: "in_contract_spec_variant"
  reviewer_note: "全部列存在于表头；circ_mv/free_share 不在 fundamental-like 名单（仅 pe_ttm/pb/ps_ttm 会被判基本面），故 required_timestamp+known_at 不触发 PIT 错误。可直接成为 v1 主候选。"

mechanism_2:
  in_v1_contract: true
  required_columns_present: []
  required_columns_missing: []
  spec_fields_touched: [feature_preprocess.cross_sectional_transform, feature_preprocess.cross_sectional_group_scope, feature_preprocess.missing_policy]
  validator_blockers: []
  implementation_status: "in_contract_spec_variant"
  reviewer_note: "winsorize_zscore + scope=date 受支持。若坚持纯 rank tie 语义则取决于 spec 是否暴露 'rank' transform；不确定项不应阻断 v1，winsorize_zscore 已足够。"

mechanism_3:
  in_v1_contract: true
  required_columns_present: []
  required_columns_missing: []
  spec_fields_touched: [training.window_type, training.train_window_n_dates, training.retrain_every_n_dates, training.min_train_rows, training.min_score_assets]
  validator_blockers: []
  implementation_status: "in_contract_spec_variant"
  reviewer_note: "全部是现有 training 字段；exploratory_screening profile 可能覆盖 retrain cadence，需在 run_controls 记录。"

mechanism_4:
  in_v1_contract: true
  required_columns_present: []
  required_columns_missing: []
  spec_fields_touched: [model_selection.metric, model_selection.enabled, model_selection.candidates]
  validator_blockers: []
  implementation_status: "partial_in_contract"
  reviewer_note: "rank_ic_minus_turnover_penalty 看似受支持，但我未独立确认其在当前 model_selection 实现里的完整行为；保守标 partial_in_contract，留待 Stage2/Stage3 用 base spec 证据确认（reviewer 严格优先）。"

mechanism_5:
  in_v1_contract: false
  required_columns_present: []
  required_columns_missing: []
  spec_fields_touched: ["（组合层，非 model-factor case spec 字段）"]
  validator_blockers: ["payload 不得含 portfolio_construction / 组合优化语义（Level 2/3 边界）", "model-factor case spec 不表达组合权重/偏离预算"]
  implementation_status: "future_enhancement"
  reviewer_note: "波动率状态降权属组合构造，超出预测信号 spec；仅保留为上下文，不进入 v1 case_spec_payload。"
