"""Fixture-based lint tests (P3).

Each (stage, mode) gets a clean fixture (zero error-level violations)
and a broken fixture that triggers each violation type. The fixtures
double as documentation: they show what a "valid" response looks like
for each stage.
"""

from __future__ import annotations

from alpha_lab.research_bridge.output_lint import (
    LintReport,
    describe_lint_contract,
    describe_model_lint_contract,
    extract_model_stage_sections,
    extract_stage_sections,
    lint_explore_response,
    lint_model_idea_response,
)

# ---------------------------------------------------------------------------
# mechanism_discovery
# ---------------------------------------------------------------------------


_MECH_DISCOVERY_CLEAN = """\
## 阶段声明
保留不确定性，提出多个互斥候选。

## 输出

[初步机制假设（Mechanism Hypotheses）]
### 机制 1: 风险厌恶在下行波动放大下的非对称定价
- agent behavior: 投资者对下行风险更敏感
- structure constraint: 卖出意愿在亏损区强化
- dynamic process: 下行波动溢价随波动期延长

### 机制 2: 负面信息释放速度差异
- agent behavior: 上行/下行信息扩散节奏不同
- structure constraint: 信息披露频率
- dynamic process: 短窗口下下行波动更早反映坏消息

[初步信号思路（Signal Sketch）]
- 可能用到的数据: 收盘价、成交量
- 可能的变换方式: 上下行波动率分解
- 直觉上的预测逻辑: 二者差异蕴含可定价的非对称信息

[与已有因子的关系]
- 最接近的已有标签: 与 vol-of-vol、downside risk 类相关
- 差异点: 不仅是量级差异，方向性结构不同

[不确定性与风险点]
- 最易出错: 上下行划分阈值的选取
- 数据风险: 极端日处理
"""


def _has_code(report: LintReport, code: str) -> bool:
    return code in report.violation_codes


def test_extract_stage_sections_returns_canonical_bodies() -> None:
    sections = extract_stage_sections(
        _MECH_DISCOVERY_CLEAN, stage="mechanism_discovery"
    )
    assert "机制候选" in sections
    assert "机制 1" in sections["机制候选"]
    assert "信号思路" in sections
    assert "上下行波动率分解" in sections["信号思路"]


def test_describe_lint_contract_is_stage_specific() -> None:
    mechanism_rules = describe_lint_contract(
        "mechanism_discovery", mode="free"
    )
    signal_rules = describe_lint_contract("signal_mapping", mode="constrained")
    validation_rules = describe_lint_contract(
        "validation_kill_tests", mode="constrained"
    )

    assert any("[初步机制假设]" in rule for rule in mechanism_rules)
    assert any("Confound 控制" in rule for rule in signal_rules)
    assert any("2-3 个可测试信号版本" in rule for rule in signal_rules)
    assert any("KILL 或 HOLD-FOR-AUDIT" in rule for rule in validation_rules)


def test_mechanism_discovery_clean_fixture_has_no_errors() -> None:
    report = lint_explore_response(
        _MECH_DISCOVERY_CLEAN, stage="mechanism_discovery", mode="free"
    )
    error_codes = [v.code for v in report.violations if v.severity == "error"]
    assert error_codes == [], f"unexpected errors: {error_codes}"
    assert report.stage == "mechanism_discovery"
    assert report.mode == "free"
    # All four required sections were observed.
    assert {"机制候选", "信号思路", "与已有因子的关系", "不确定性"} <= set(
        report.sections_seen
    )


def test_mechanism_discovery_missing_section_is_error() -> None:
    # Strip the "[与已有因子的关系]" section.
    text = _MECH_DISCOVERY_CLEAN.replace("[与已有因子的关系]", "[OTHER]")
    report = lint_explore_response(text, stage="mechanism_discovery", mode="free")
    assert _has_code(report, "missing_section")
    assert any(
        v.section == "与已有因子的关系"
        for v in report.violations
        if v.code == "missing_section"
    )


def test_mechanism_discovery_forbidden_direction_is_error() -> None:
    text = _MECH_DISCOVERY_CLEAN + "\n做多波动率较低的股票，做空波动率较高的股票。"
    report = lint_explore_response(text, stage="mechanism_discovery", mode="free")
    assert _has_code(report, "forbidden_direction")


def test_mechanism_discovery_forbidden_label_in_heading_is_error() -> None:
    text = _MECH_DISCOVERY_CLEAN.replace(
        "### 机制 1: 风险厌恶在下行波动放大下的非对称定价",
        "### 机制 1: Reversal-style asymmetry",
    )
    report = lint_explore_response(text, stage="mechanism_discovery", mode="free")
    assert _has_code(report, "forbidden_label_in_name")


def test_mechanism_discovery_single_mechanism_is_error_in_free_mode() -> None:
    # Drop the second mechanism block.
    text = _MECH_DISCOVERY_CLEAN.split("### 机制 2")[0]
    text += "[初步信号思路（Signal Sketch）]\n[与已有因子的关系]\n[不确定性与风险点]\n"
    report = lint_explore_response(text, stage="mechanism_discovery", mode="free")
    assert _has_code(report, "single_mechanism")
    single_mech = next(v for v in report.violations if v.code == "single_mechanism")
    assert single_mech.severity == "error"


def test_mechanism_discovery_single_mechanism_is_warning_in_start_mode() -> None:
    text = _MECH_DISCOVERY_CLEAN.split("### 机制 2")[0]
    text += "[初步信号思路（Signal Sketch）]\n[与已有因子的关系]\n[不确定性与风险点]\n"
    report = lint_explore_response(text, stage="mechanism_discovery", mode="start")
    single_mech = next(
        (v for v in report.violations if v.code == "single_mechanism"), None
    )
    assert single_mech is not None
    assert single_mech.severity == "warning"


# ---------------------------------------------------------------------------
# signal_mapping
# ---------------------------------------------------------------------------


_SIGNAL_MAPPING_CLEAN = """\
## 输出

[Mechanism Mapping]
### 机制 1
- 声明: 风险厌恶非对称定价
- implications:
  - 下行波动溢价 t+1 IC > 0
  - 极端日效应放大
- required_data:
  - close | 频率：daily sufficient | 角色：necessary
  - volume | 频率：daily sufficient | 角色：confound control
- 变量论证: 删除 close，机制无法测量

### 机制 2
- 声明: 负面信息释放速度差异
- implications:
  - 上下行波动短窗 IC 不对称
- required_data:
  - close | 频率：daily sufficient | 角色：necessary

[当前实现解释]
- 捕捉的机制: 机制 1 大部分
- 遗漏的机制: 机制 2 在 daily 下不可分辨
- daily / intraday 区分: 当前 daily 实现无法区分两条路径

[Confound 控制]
- reversal: 残差化 | 论据: 控制 1-5 日反转后仍显著才有意义
- total volatility: 显式控制 | 论据: 不分上下行的总波动作为基线
- skewness / downside risk: 包含 | 论据: 该结构本身蕴含 downside 信息
- liquidity / turnover: 显式控制 | 论据: 控制 Amihud 后再看残差
- size / industry / price level: 残差化 | 论据: 行业市值中性化处理

[可测试信号版本]
- v1: log(downside_vol) - log(upside_vol)，控制 reversal/total vol
- v2: 同 v1 但加 industry-neutral
- v3: 改用 rolling 60 日窗口的版本
"""


def test_signal_mapping_clean_fixture_has_no_errors() -> None:
    report = lint_explore_response(
        _SIGNAL_MAPPING_CLEAN, stage="signal_mapping", mode="free"
    )
    error_codes = [v.code for v in report.violations if v.severity == "error"]
    assert error_codes == [], f"unexpected errors: {error_codes}"
    assert {"Mechanism Mapping", "当前实现解释", "Confound 控制", "可测试信号版本"} <= set(
        report.sections_seen
    )


def test_signal_mapping_missing_confound_is_error() -> None:
    # Drop the size/industry/price line entirely.
    text = _SIGNAL_MAPPING_CLEAN.replace(
        "- size / industry / price level: 残差化 | 论据: 行业市值中性化处理\n",
        "",
    )
    report = lint_explore_response(text, stage="signal_mapping", mode="free")
    assert _has_code(report, "confound_missing")


def test_signal_mapping_confound_without_verdict_is_error() -> None:
    text = _SIGNAL_MAPPING_CLEAN.replace(
        "- reversal: 残差化 | 论据: 控制 1-5 日反转后仍显著才有意义",
        "- reversal: 看了，不太确定 | 论据: 暂存",
    )
    report = lint_explore_response(text, stage="signal_mapping", mode="free")
    assert _has_code(report, "confound_verdict_missing")


def test_signal_mapping_final_pick_language_is_error() -> None:
    text = _SIGNAL_MAPPING_CLEAN + "\n综合来看，我推荐版本 1 作为首选。"
    report = lint_explore_response(text, stage="signal_mapping", mode="free")
    assert _has_code(report, "final_pick")


def test_signal_mapping_constrained_strict_version_count() -> None:
    # Only one v1 line — should fail in strict mode.
    text = _SIGNAL_MAPPING_CLEAN.replace(
        "- v2: 同 v1 但加 industry-neutral\n- v3: 改用 rolling 60 日窗口的版本",
        "",
    )
    report = lint_explore_response(text, stage="signal_mapping", mode="constrained")
    assert _has_code(report, "version_count_out_of_range")


def test_signal_mapping_free_low_version_count_is_warning_only() -> None:
    text = _SIGNAL_MAPPING_CLEAN.replace(
        "- v2: 同 v1 但加 industry-neutral\n- v3: 改用 rolling 60 日窗口的版本",
        "",
    )
    report = lint_explore_response(text, stage="signal_mapping", mode="free")
    assert _has_code(report, "version_count_low")
    warning = next(v for v in report.violations if v.code == "version_count_low")
    assert warning.severity == "warning"
    # Free mode should not flip has_errors purely due to version count.
    other_errors = [
        v for v in report.violations if v.severity == "error"
    ]
    assert other_errors == []


# ---------------------------------------------------------------------------
# validation_kill_tests
# ---------------------------------------------------------------------------


_VALIDATION_CLEAN_FREE = """\
## 输出

[Alias / 换壳审计]
- reversal: 部分重叠 | 论据: 短窗有部分重叠，但下行非对称结构独立
- volatility: 部分重叠 | 论据: 与总波动相关，但分解后仍有信号
- skewness / downside risk: 显著重叠 | 论据: 与 downside risk 高度共线
- liquidity / turnover: 不重叠 | 论据: 与 Amihud 残差化后仍显著
- size / industry / price level: 不重叠 | 论据: 中性化后保留

[暴露分解]
- 行业中性化后：残差仍显著
- 市值中性化后：残差仍显著
- 流动性中性化后：仅在部分中性化下保留
- 波动率中性化后：中性化后失效
- 联合中性化后：残差 IC ~ 0.4×原始
- 残差 IC 上限估计: ~0.025

[数据健全性]
- 极端日期: 排序在大跌日塌缩
- 涨跌停: PIT 排除
- 停牌 / 复牌: 前向填充
- ST: 剔除
- 复权: 已校正
- IPO / 退市窗: 已剔除

[实现稳健性]
- skip_recent 扫描: 一致
- 窗口长度 ±50%: 信号一致
- horizon 扫描: 1d-5d 最强
- 横截面预处理: rank vs zscore 差异不大

[子样本稳定性]
- 分年份: 牛市偏弱
- regime: 震荡市最强
- 行业桶: top3 行业贡献 60%
- 市值桶: 中小盘更强

[最终判定]
- 触发的死亡条件: 波动率中性化后失效（接近触线，但未越线）
- 判定: ITERATE
- 下一步实证步骤: 1) 改用波动率残差版本 2) 复测 OOS
"""


_VALIDATION_CLEAN_CONSTRAINED = _VALIDATION_CLEAN_FREE.replace(
    "- reversal: 部分重叠 | 论据: 短窗有部分重叠，但下行非对称结构独立",
    "- reversal: 部分重叠 | 论据: Jegadeesh-Titman baseline 下短窗重叠，但 [K1] 下行非对称结构独立",
).replace(
    "- volatility: 部分重叠 | 论据: 与总波动相关，但分解后仍有信号",
    "- volatility: 部分重叠 | 论据: Ang et al 下与总波动相关，分解后 [K2] 仍有信号",
).replace(
    "- skewness / downside risk: 显著重叠 | 论据: 与 downside risk 高度共线",
    "- skewness / downside risk: 显著重叠 | 论据: Ang et al 显示与 downside risk 共线 [K3]",
).replace(
    "- liquidity / turnover: 不重叠 | 论据: 与 Amihud 残差化后仍显著",
    "- liquidity / turnover: 不重叠 | 论据: Amihud 残差化后仍显著 [K4]",
).replace(
    "- size / industry / price level: 不重叠 | 论据: 中性化后保留",
    "- size / industry / price level: 不重叠 | 论据: Fama-French 中性化后保留 [K5]",
).replace(
    "- 判定: ITERATE",
    "- 判定: HOLD-FOR-AUDIT",
)


def test_validation_clean_free_fixture_has_no_errors() -> None:
    report = lint_explore_response(
        _VALIDATION_CLEAN_FREE, stage="validation_kill_tests", mode="free"
    )
    error_codes = [v.code for v in report.violations if v.severity == "error"]
    assert error_codes == [], f"unexpected errors: {error_codes}"


def test_validation_clean_constrained_fixture_has_no_errors() -> None:
    report = lint_explore_response(
        _VALIDATION_CLEAN_CONSTRAINED,
        stage="validation_kill_tests",
        mode="constrained",
    )
    error_codes = [v.code for v in report.violations if v.severity == "error"]
    assert error_codes == [], f"unexpected errors: {error_codes}"


def test_validation_missing_alias_target_is_error() -> None:
    text = _VALIDATION_CLEAN_FREE.replace(
        "- liquidity / turnover: 不重叠 | 论据: 与 Amihud 残差化后仍显著\n",
        "",
    )
    report = lint_explore_response(text, stage="validation_kill_tests", mode="free")
    assert _has_code(report, "alias_target_missing")


def test_validation_alias_without_verdict_is_error() -> None:
    text = _VALIDATION_CLEAN_FREE.replace(
        "- reversal: 部分重叠 | 论据: 短窗有部分重叠，但下行非对称结构独立",
        "- reversal: 看着像 | 论据: 暂时存疑",
    )
    report = lint_explore_response(text, stage="validation_kill_tests", mode="free")
    assert _has_code(report, "alias_verdict_missing")


def test_validation_hedging_in_final_section_is_error() -> None:
    text = _VALIDATION_CLEAN_FREE.replace(
        "- 判定: ITERATE", "- 判定: 看情况，需要更多数据"
    )
    report = lint_explore_response(text, stage="validation_kill_tests", mode="free")
    assert _has_code(report, "hedging_verdict")


def test_validation_constrained_requires_binary_verdict() -> None:
    # Free-mode-style ITERATE in strict mode -> binary missing.
    report = lint_explore_response(
        _VALIDATION_CLEAN_FREE,
        stage="validation_kill_tests",
        mode="constrained",
    )
    # Free fixture uses ITERATE which is not KILL or HOLD-FOR-AUDIT,
    # so strict mode flags missing_binary_verdict.
    assert _has_code(report, "missing_binary_verdict")


def test_validation_constrained_requires_anchor_for_each_alias() -> None:
    # Take constrained-clean fixture and strip one citation.
    text = _VALIDATION_CLEAN_CONSTRAINED.replace(
        "- liquidity / turnover: 不重叠 | 论据: Amihud 残差化后仍显著 [K4]",
        "- liquidity / turnover: 不重叠 | 论据: 我直觉觉得不重叠",
    )
    report = lint_explore_response(
        text, stage="validation_kill_tests", mode="constrained"
    )
    assert _has_code(report, "alias_unanchored")


def test_validation_free_no_explicit_verdict_is_error() -> None:
    text = _VALIDATION_CLEAN_FREE.replace("- 判定: ITERATE", "- 判定: 待定")
    report = lint_explore_response(text, stage="validation_kill_tests", mode="free")
    assert _has_code(report, "missing_verdict")


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------


def test_lint_report_to_dict_round_trips_fields() -> None:
    report = lint_explore_response(
        "totally empty", stage="mechanism_discovery", mode="free"
    )
    payload = report.to_dict()
    assert payload["stage"] == "mechanism_discovery"
    assert payload["mode"] == "free"
    assert isinstance(payload["violations"], list)
    assert payload["has_errors"] is True


def test_unknown_stage_normalizes_to_mechanism_discovery() -> None:
    report = lint_explore_response("nothing", stage="garbage", mode="free")
    assert report.stage == "mechanism_discovery"


# ---------------------------------------------------------------------------
# model-lab lint
# ---------------------------------------------------------------------------


_MODEL_MECHANISM_CLEAN = """\
[模型机制候选]
### 机制 1
- mechanism family: loss/regularization
- touched contract surfaces: model, training
- early falsifier: OOS rank_ic_ir 不改善

### 机制 2
- mechanism family: feature interaction
- touched contract surfaces: model, feature_preprocess
- early falsifier: top feature importance 不稳定

[实现假设草图]
- 机制 1: in-contract, 只讨论正则结构，不写 patch
- 机制 2: in-contract, 讨论非线性交互

[与当前 spec / baseline 的关系]
- current baseline captured: ridge linear baseline
- structural difference: loss/regularization 与 feature interaction 是两条不同机制

[不确定性与失败路径]
- PIT / label leakage risk: known_at 对齐
- overfit / split fragility risk: walk-forward 验证
"""


_MODEL_SIGNAL_CLEAN = """\
[Model Mechanism Mapping]
### 机制 1
- mechanism: feature interaction
- implication: 非线性模型应在行业/市值桶内改善残差 rank_ic
- required_data_or_spec_fields:
  - field: feature_columns | role: necessary | remove-and-test reason: 删除后无法测试交互

[当前实现解释]
- current implementation captures: ridge baseline 的线性部分
- current implementation misses: 非线性交互
- cannot disambiguate at current data/spec tier: high-frequency 特征不可用

[模型风险控制]
- `feature availability / PIT`: 规避 - 使用 known_at / safety_lag
- `label / target leakage`: 显式控制 - forward label 与 feature date 分离
- `overfit / complexity`: 压力测试 - 限制候选数量和 walk-forward
- `turnover / cost`: 显式控制 - 使用 cost-aware IR
- `feature instability`: 压力测试 - top feature stability
- `split / regime fragility`: 压力测试 - year/regime split

[可测试模型版本]
- v1: ridge + stronger regularization | controls | residual assumptions
- v2: gbdt shallow trees | controls | residual assumptions
"""


_MODEL_VALIDATION_CLEAN = """\
[Alias / 问题归因审计]
- `baseline linear/ridge`: 部分风险 - anchor: current spec ridge baseline
- `regularization-only`: 不构成风险 - anchor: current spec model params
- `feature-count / complexity`: 部分风险 - anchor: feature_count
- `leakage / PIT`: 不构成风险 - anchor: feature_availability known_at
- `split luck / regime overfit`: 部分风险 - anchor: walk-forward runs
- `turnover / cost artifact`: 不构成风险 - anchor: cost-aware IR

[数据与时间完整性]
- PIT / known_at: pass
- target leakage / overlapping label: pass

[训练与验证稳健性]
- split design: walk-forward + year split
- hyperparameter freedom: bounded

[特征与解释稳定性]
- feature count dependence: tested
- top feature stability: tested

[成本与组合影响]
- turnover and transaction cost: net IR checked
- portfolio construction sensitivity: checked

[最终判定]
- verdict: HOLD-FOR-AUDIT
- hard kill trigger, if any: none
"""


def test_model_lint_contract_is_stage_specific() -> None:
    mechanism_rules = describe_model_lint_contract("mechanism_discovery")
    signal_rules = describe_model_lint_contract("signal_mapping", mode="constrained")
    validation_rules = describe_model_lint_contract(
        "validation_kill_tests", mode="constrained"
    )

    assert any("[模型机制候选]" in rule for rule in mechanism_rules)
    assert any("feature availability / PIT" in rule for rule in signal_rules)
    assert any("KILL 或 HOLD-FOR-AUDIT" in rule for rule in validation_rules)


def test_extract_model_stage_sections_returns_canonical_bodies() -> None:
    sections = extract_model_stage_sections(
        _MODEL_MECHANISM_CLEAN, stage="mechanism_discovery"
    )
    assert "模型机制候选" in sections
    assert "loss/regularization" in sections["模型机制候选"]


def test_model_mechanism_discovery_clean_fixture_has_no_errors() -> None:
    report = lint_model_idea_response(
        _MODEL_MECHANISM_CLEAN, stage="mechanism_discovery", mode="explore"
    )
    assert [v.code for v in report.violations if v.severity == "error"] == []


def test_model_mechanism_discovery_rejects_premature_patch() -> None:
    text = _MODEL_MECHANISM_CLEAN + "\n```json\n{\"model\": {\"family\": \"gbdt\"}}\n```"
    report = lint_model_idea_response(
        text, stage="mechanism_discovery", mode="explore"
    )
    assert _has_code(report, "premature_model_convergence")


def test_model_signal_mapping_clean_fixture_has_no_errors() -> None:
    report = lint_model_idea_response(
        _MODEL_SIGNAL_CLEAN, stage="signal_mapping", mode="constrained"
    )
    assert [v.code for v in report.violations if v.severity == "error"] == []


def test_model_signal_mapping_missing_risk_is_error() -> None:
    text = _MODEL_SIGNAL_CLEAN.replace(
        "- `turnover / cost`: 显式控制 - 使用 cost-aware IR\n",
        "",
    )
    report = lint_model_idea_response(text, stage="signal_mapping", mode="explore")
    assert _has_code(report, "model_risk_missing")


def test_model_validation_clean_fixture_has_no_errors() -> None:
    report = lint_model_idea_response(
        _MODEL_VALIDATION_CLEAN,
        stage="validation_kill_tests",
        mode="constrained",
    )
    assert [v.code for v in report.violations if v.severity == "error"] == []


def test_model_validation_missing_binary_verdict_is_error() -> None:
    text = _MODEL_VALIDATION_CLEAN.replace(
        "- verdict: HOLD-FOR-AUDIT", "- verdict: ITERATE"
    )
    report = lint_model_idea_response(
        text, stage="validation_kill_tests", mode="constrained"
    )
    assert _has_code(report, "missing_binary_verdict")
