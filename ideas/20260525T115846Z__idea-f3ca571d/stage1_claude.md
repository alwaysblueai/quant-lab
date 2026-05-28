idea_id: `20260525T115846Z__idea-f3ca571d`
engine: claude
lab: model_factor
generated: 2026-05-25

## Part A — Mechanism candidates (generator)

mechanism_1:
  hypothesis: "在成交活跃但波动未过热的横截面子样本里，价格-换手-流动性技术特征对未来一周收益排序的可预测性更高、更稳定；用低自由度线性族把这些特征压成一个排序分数即可捕捉大部分增量。"
  inspired_by: [K7:low_freedom_linear_score_blend, K4:preprocess_order_universe_data_quality_gate]
  fusion_of: ["Z-Score Normalization and Aggregation", "Factor Research Operating Manual"]
  novel_delta: "不是再造一个动量/反转单因子，而是把换手+流动性+价格技术特征作为一组同 owner 描述符，用透明线性模型做横截面收益排序基线；强调'低自由度'作为抗过拟合先验。"
  signal_sketch: "对 turnover_rate / turnover_rate_f / volume_ratio / circ_mv / free_share / atr_bfq / bias1_bfq / rsi_bfq_6 / mtm_bfq / wr_bfq / vr_bfq / mfi_bfq 做横截面 winsorize+zscore，ridge 回归预测 forward_return(5d)，截面排序取分位。"
  data_needs: [turnover_rate, turnover_rate_f, volume_ratio, circ_mv, free_share, atr_bfq, bias1_bfq, rsi_bfq_6, mtm_bfq, wr_bfq, vr_bfq, mfi_bfq]
  concern: "线性模型可能被高量纲的流动性/市值列主导；需横截面标准化与 winsor 控制。"

mechanism_2:
  hypothesis: "横截面 winsorize + z-score + 方向对齐的预处理顺序本身就是稳定性来源；在收益排序任务里，先固定 universe 与数据可得性、再按异常值→缺失→标准化顺序处理，比换更复杂模型更能提升 rank IC 稳定性。"
  inspired_by: [K7:cross_section_winsor_zscore_direction_align, K4:preprocess_order_universe_data_quality_gate, K5:style_industry_neutralization_gate]
  fusion_of: ["Z-Score Normalization and Aggregation", "Tie-Handling in Cross-Sectional Ranking"]
  novel_delta: "把'预处理顺序'当成可调研究对象而非固定样板：median_impute + winsorize_zscore + scope=date 是一个可审计的低复杂度组合。"
  signal_sketch: "feature_preprocess.missing_policy=median_impute, cross_sectional_transform=winsorize_zscore, cross_sectional_group_scope=date；模型不变。"
  data_needs: []
  concern: "若底层有大量并列值（零成交、涨跌停），rank/zscore 的 tie 处理会改变截面分布；当前 spec 用 zscore 而非 rank，需注意。"

mechanism_3:
  hypothesis: "用'换手惩罚版'的模型选择目标（rank IC 减去换手惩罚）来挑线性超参，可以在不牺牲太多预测力的前提下压低组合换手，这与'成交活跃但不追逐过热'的直觉一致。"
  inspired_by: [K4:broker_factor_research_stage_gate]
  fusion_of: []
  novel_delta: "把'偏好可持有、低换手信号'写进 model_selection.metric，而不是事后筛；属于现有 spec 已支持的 selection metric。"
  signal_sketch: "model_selection.enabled=true, metric=rank_ic_minus_turnover_penalty, 候选为若干 ridge/lasso alpha，n_splits=3，turnover_penalty_lambda=0.1。"
  data_needs: []
  concern: "turnover penalty 的 lambda 是自由参数；若样本内反复调会退化成过拟合，应固定一个保守值。"

mechanism_4:
  hypothesis: "显式构造'波动率×换手'交互列（如 atr_bfq * turnover_rate）能更直接表达'活跃但不过热'的条件可信度，而不是依赖线性模型隐式学到交互。"
  inspired_by: [K3:volatility_state_factor_weight_rule]
  fusion_of: []
  novel_delta: "把条件可信度做成显式交互特征，理论上比纯线性叠加更贴近假设。"
  signal_sketch: "新增列 vol_turnover_interaction = atr_bfq * turnover_rate（features 文件当前不存在该列）。"
  data_needs: ["atr_bfq*turnover_rate（派生交互列，未注册）"]
  concern: "该列在 features_safe_bfq_35.parquet 表头中不存在；v1 不允许写 interaction expression 或 feature builder code，标 needs_extension。"

mechanism_5:
  hypothesis: "把每只股票的线性排序分数按'换手活跃度'做样本权重（活跃股权重更高）训练，可让模型更关注可交易、信息更充分的样本。"
  inspired_by: [K3:volatility_state_factor_weight_rule]
  fusion_of: []
  novel_delta: "条件可信度落在训练样本权重上，而非特征层。"
  signal_sketch: "训练时按 turnover 分桶赋 sample_weight。"
  data_needs: []
  concern: "v1 spec 无 sample_weight 入口；引入需自定义训练 hook，标 needs_extension，仅作上下文。"

## Part B — Code feasibility review (reviewer)

mechanism_1:
  in_v1_contract: true
  required_columns_present: [turnover_rate, turnover_rate_f, volume_ratio, circ_mv, free_share, atr_bfq, bias1_bfq, rsi_bfq_6, mtm_bfq, wr_bfq, vr_bfq, mfi_bfq]
  required_columns_missing: []
  spec_fields_touched: [feature_columns, model.family, model.params, target.horizon, feature_preprocess]
  validator_blockers: []
  implementation_status: "in_contract_spec_variant"
  reviewer_note: "全部列在 features_safe_bfq_35.parquet 表头中；无 fundamental-like 列（pe_ttm/pb/ps_ttm 已排除），feature_availability=required_timestamp+known_at 可过 PIT 检查。ridge family 受支持。这是最干净的 v1 主候选。"

mechanism_2:
  in_v1_contract: true
  required_columns_present: []
  required_columns_missing: []
  spec_fields_touched: [feature_preprocess.missing_policy, feature_preprocess.cross_sectional_transform, feature_preprocess.cross_sectional_group_scope]
  validator_blockers: []
  implementation_status: "in_contract_spec_variant"
  reviewer_note: "纯 feature_preprocess 字段变体，spec parser 已支持 median_impute / winsorize_zscore / scope=date。与 M1 叠加即可，不单独成候选。"

mechanism_3:
  in_v1_contract: true
  required_columns_present: []
  required_columns_missing: []
  spec_fields_touched: [model_selection.enabled, model_selection.metric, model_selection.candidates, model_selection.n_splits, model_selection.turnover_penalty_lambda]
  validator_blockers: []
  implementation_status: "in_contract_spec_variant"
  reviewer_note: "rank_ic_minus_turnover_penalty 是现有 spec 已支持的 selection metric（见 turnover_conditioned_pv_synthesis_v1 base spec）；属合同内变体。"

mechanism_4:
  in_v1_contract: false
  required_columns_present: [atr_bfq, turnover_rate]
  required_columns_missing: ["vol_turnover_interaction（派生交互列，features 文件无此列）"]
  spec_fields_touched: [feature_columns]
  validator_blockers: ["v1 禁止 interaction expression / 自定义 feature builder", "feature_columns 必须是 features 文件真实表头列"]
  implementation_status: "needs_extension"
  reviewer_note: "交互列需新增 feature builder 或预先物化到 features 文件；v1 spec_variant 不允许。保留为上下文，不进入 v1 case_spec_payload。"

mechanism_5:
  in_v1_contract: false
  required_columns_present: []
  required_columns_missing: []
  spec_fields_touched: ["training.sample_weight（无此字段）"]
  validator_blockers: ["v1 无 sample_weight 训练入口", "implementation_type 必须为 spec_variant，禁止自定义训练 hook"]
  implementation_status: "needs_extension"
  reviewer_note: "样本权重需自定义训练逻辑，超出 spec_variant。仅作上下文，M1 的特征层已部分代理'活跃股更可信'的直觉。"
