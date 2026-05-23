"""Stage-aware lint over LLM responses to explore prompts (P3).

Each workflow stage's prompt enforces a specific output schema in its
``## 输出格式（严格遵守）`` block. This module checks that the LLM's
response actually follows the schema — and surfaces violations as
machine-readable codes that downstream code can persist (session
memory) and reinject (next-call drift header).

Rules are stage-specific by design: the same word means different
things at different stages (forbidden label / confound control / audit
target). See ``docs/research_workflow.md``.

Pure module — no I/O. Caller decides what to do with the report.
"""
from __future__ import annotations

# ruff: noqa: E501
import re
from collections.abc import Iterable
from dataclasses import dataclass

from alpha_lab.research_bridge.scoring import (
    FORBIDDEN_FACTOR_LABELS,
    MECHANISM_DISCOVERY,
    SIGNAL_MAPPING,
    SIGNAL_MAPPING_CONFOUND_CONTROLS,
    VALIDATION_ALIAS_TARGETS,
    VALIDATION_KILL_TESTS,
    normalize_workflow_stage,
)


@dataclass(frozen=True, slots=True)
class LintViolation:
    code: str
    severity: str  # "error" | "warning"
    section: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "section": self.section,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class LintReport:
    stage: str
    mode: str
    violations: tuple[LintViolation, ...]
    sections_seen: tuple[str, ...]

    @property
    def has_errors(self) -> bool:
        return any(v.severity == "error" for v in self.violations)

    @property
    def violation_codes(self) -> tuple[str, ...]:
        return tuple(v.code for v in self.violations)

    def to_dict(self) -> dict[str, object]:
        return {
            "stage": self.stage,
            "mode": self.mode,
            "has_errors": self.has_errors,
            "violations": [v.to_dict() for v in self.violations],
            "sections_seen": list(self.sections_seen),
        }


# Direction-related phrases that mechanism_discovery forbids in the
# response body — the stage explicitly bans pre-supposing return sign.
_DIRECTION_PHRASE_PATTERNS: tuple[str, ...] = (
    r"做多",
    r"做空",
    r"\blong\s+(?:the|this|that|stock|name)\b",
    r"\bshort\s+(?:the|this|that|stock|name)\b",
    r"\bbuy\s+(?:the|this|that|stock|name)\b",
    r"\bsell\s+(?:the|this|that|stock|name)\b",
)

# Hedging / escape verdicts retained for archived validation_kill_tests
# sessions. The public idea explorer no longer generates that stage.
_HEDGING_PHRASE_PATTERNS: tuple[str, ...] = (
    r"看情况",
    r"需要更多数据",
    r"进一步研究",
    r"进一步观察",
    r"暂时无法判断",
    r"尚不能定论",
    r"\bdepends?\s+on\s+the\s+data\b",
    r"\bneeds?\s+more\s+data\b",
    r"\bfurther\s+(?:investigation|research)\b",
)

# "Final pick" language that signal_mapping forbids — selection belongs
# to Stage 3 data validation, not the Stage 1 idea explorer.
_FINAL_PICK_PHRASE_PATTERNS: tuple[str, ...] = (
    r"推荐版本",
    r"最优版本",
    r"选择版本",
    r"\bfinal\s+pick\b",
    r"\bI\s+recommend\b",
    r"\bbest\s+version\b",
)

_MECHANISM_DISCOVERY_REQUIRED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("机制候选", ("[初步机制假设", "[Mechanism Hypotheses", "[候选机制")),
    ("信号思路", ("[初步信号思路", "[Signal Sketch", "[候选表达")),
    ("与已有因子的关系", ("[与已有因子",)),
    ("不确定性", ("[不确定性与风险点", "[风险识别")),
)

_SIGNAL_MAPPING_REQUIRED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Mechanism Mapping", ("[Mechanism Mapping]", "[mechanism mapping]")),
    ("当前实现解释", ("[当前实现解释]",)),
    ("Confound 控制", ("[Confound 控制]", "[confound 控制]")),
    ("可测试信号版本", ("[可测试信号版本]",)),
)

_VALIDATION_REQUIRED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Alias 审计", ("[Alias / 换壳审计]", "[alias / 换壳审计]")),
    ("暴露分解", ("[暴露分解]",)),
    ("数据健全性", ("[数据健全性]",)),
    ("实现稳健性", ("[实现稳健性]",)),
    ("子样本稳定性", ("[子样本稳定性]",)),
    ("最终判定", ("[最终判定]",)),
)

_MODEL_MECHANISM_REQUIRED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("模型机制候选", ("[模型机制候选]",)),
    ("实现假设草图", ("[实现假设草图]",)),
    ("与当前 spec / baseline 的关系", ("[与当前 spec / baseline 的关系]",)),
    ("不确定性与失败路径", ("[不确定性与失败路径]",)),
)

_MODEL_SIGNAL_REQUIRED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Model Mechanism Mapping", ("[Model Mechanism Mapping]",)),
    ("当前实现解释", ("[当前实现解释]",)),
    ("模型风险控制", ("[模型风险控制]",)),
    ("可测试模型版本", ("[可测试模型版本]",)),
)

_MODEL_VALIDATION_REQUIRED: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Alias / 问题归因审计", ("[Alias / 问题归因审计]",)),
    ("数据与时间完整性", ("[数据与时间完整性]",)),
    ("训练与验证稳健性", ("[训练与验证稳健性]",)),
    ("特征与解释稳定性", ("[特征与解释稳定性]",)),
    ("成本与组合影响", ("[成本与组合影响]",)),
    ("最终判定", ("[最终判定]",)),
)

MODEL_SIGNAL_RISK_CONTROLS: tuple[tuple[str, str], ...] = (
    ("feature availability / PIT", "known_at / safety_lag / as-of 对齐"),
    ("label / target leakage", "forward label、重叠窗口、目标构造泄漏"),
    ("overfit / complexity", "模型复杂度、样本量、超参自由度"),
    ("turnover / cost", "换手惩罚、成本后 IR、可交易性"),
    ("feature instability", "特征重要性、top feature 稳定性、冗余特征"),
    ("split / regime fragility", "walk-forward、purged split、年份/市场状态稳定性"),
)

MODEL_VALIDATION_ALIAS_TARGETS: tuple[tuple[str, str], ...] = (
    ("baseline linear/ridge", "是否只是线性基线或 ridge 的轻微调参"),
    ("regularization-only", "是否只是正则强弱变化而非新模型机制"),
    ("feature-count / complexity", "是否由更多特征或更高复杂度驱动"),
    ("leakage / PIT", "是否由未来信息、known_at 或目标泄漏驱动"),
    ("split luck / regime overfit", "是否只在某个切分或市场状态有效"),
    ("turnover / cost artifact", "是否被换手、成本或组合再平衡假象驱动"),
)


def lint_explore_response(
    text: str, *, stage: str, mode: str = "free"
) -> LintReport:
    """Stage-aware lint dispatcher.

    ``mode`` modulates strictness within a stage but does not change the
    structural skeleton: the same required sections apply across modes,
    and constrained mode adds extra rules (anchor citation, binary
    verdict, etc.).
    """
    normalized_stage = normalize_workflow_stage(stage, allow_stage3_helper=True)
    normalized_mode = (mode or "").strip().lower() or "free"
    body = text or ""
    if normalized_stage == MECHANISM_DISCOVERY:
        return _lint_mechanism_discovery(body, normalized_mode)
    if normalized_stage == SIGNAL_MAPPING:
        return _lint_signal_mapping(body, normalized_mode)
    return _lint_validation_kill_tests(body, normalized_mode)


def lint_model_idea_response(
    text: str, *, stage: str, mode: str = "explore"
) -> LintReport:
    """Stage-aware lint for model-lab idea responses."""
    normalized_stage = normalize_workflow_stage(stage, allow_stage3_helper=True)
    normalized_mode = (mode or "").strip().lower() or "explore"
    body = text or ""
    if normalized_stage == MECHANISM_DISCOVERY:
        return _lint_model_mechanism_discovery(body, normalized_mode)
    if normalized_stage == SIGNAL_MAPPING:
        return _lint_model_signal_mapping(body, normalized_mode)
    return _lint_model_validation_kill_tests(body, normalized_mode)


def extract_stage_sections(text: str, *, stage: str) -> dict[str, str]:
    """Extract canonical stage sections from a response.

    This is intentionally tolerant: each canonical section may have
    multiple prompt-era aliases, and missing sections are simply omitted.
    Session chaining uses this to pass the useful slice of one stage into
    the next prompt without needing to understand the full response.
    """
    normalized_stage = normalize_workflow_stage(stage, allow_stage3_helper=True)
    required = _required_sections_for_stage(normalized_stage)
    sections: dict[str, str] = {}
    for canonical, alternates in required:
        body = _extract_section_body(text or "", *alternates).strip()
        if body:
            sections[canonical] = body
    return sections


def extract_model_stage_sections(text: str, *, stage: str) -> dict[str, str]:
    """Extract canonical model-lab response sections for workflow chaining."""
    normalized_stage = normalize_workflow_stage(stage, allow_stage3_helper=True)
    required = _model_required_sections_for_stage(normalized_stage)
    sections: dict[str, str] = {}
    for canonical, alternates in required:
        body = _extract_section_body(text or "", *alternates).strip()
        if body:
            sections[canonical] = body
    return sections


def describe_lint_contract(stage: str, *, mode: str = "free") -> tuple[str, ...]:
    """Return human-facing lint rules for prompt self-check sections."""
    normalized_stage = normalize_workflow_stage(stage, allow_stage3_helper=True)
    normalized_mode = (mode or "").strip().lower() or "free"
    if normalized_stage == MECHANISM_DISCOVERY:
        return (
            "必须包含 [初步机制假设] / [初步信号思路] / [与已有因子的关系] / "
            "[不确定性与风险点] 四类结构段。",
            "机制候选至少 2 个；start 模式可作为 warning，"
            "free/constrained 模式下少于 2 个会被视为违规。",
            "机制名称不得直接使用 reversal / momentum / value / quality / "
            "size / skewness / liquidity 等既有标签。",
            "不得出现做多 / 做空 / long the / short the / buy the / sell the 等预设收益方向语言。",
            "每个机制候选应能落入 Stage 1 mechanism schema：包含 hypothesis / signal_sketch / data_needs，"
            "inspired_by / fusion_of / cross_domain_jump 只在有助于学习时填写；无来源不算缺口。",
        )
    if normalized_stage == SIGNAL_MAPPING:
        rules = [
            "必须包含 [Mechanism Mapping] / [当前实现解释] / [Confound 控制] / "
            "[可测试信号版本] 四段。",
            "Confound 控制必须逐项覆盖 reversal / total volatility / "
            "skewness-downside / liquidity-turnover / size-industry-price。",
            "每个 confound 必须给出 {包含 / 残差化 / 显式控制 / 不控制} 之一的处理判定。",
            "不得出现“推荐版本”“最优版本”“final pick”“I recommend”等最终选择语言；"
            "最终选择属于 Stage 3 数据验证。",
        ]
        if normalized_mode == "constrained":
            rules.append(
                "constrained 模式必须输出 2-3 个可测试信号版本，"
                "少于 2 个或多于 3 个都会被 lint 拒绝。"
            )
        else:
            rules.append("建议输出 2-3 个可测试信号版本；少于 2 个会被标记为 warning。")
        return tuple(rules)
    rules = [
        "必须包含 [Alias / 换壳审计] / [暴露分解] / [数据健全性] / "
        "[实现稳健性] / [子样本稳定性] / [最终判定] 六段。",
        "Alias 审计必须逐项覆盖 reversal / volatility / skewness-downside / "
        "liquidity-turnover / size-industry-price。",
        "每个 alias 必须给出 {显著重叠 / 部分重叠 / 不重叠} 之一的判定。",
        "最终判定不得使用“看情况”“需要更多数据”“进一步研究”等回避语。",
    ]
    if normalized_mode == "constrained":
        rules.extend(
            [
                "constrained 模式下每个 alias 判定必须带 [Kx] 知识库引用或标准 baseline 引用。",
                "constrained 模式最终判定只能是 KILL 或 HOLD-FOR-AUDIT。",
            ]
        )
    else:
        rules.append("最终判定必须明确落在 KILL / HOLD / ITERATE / HOLD-FOR-AUDIT 之一。")
    return tuple(rules)


def describe_model_lint_contract(
    stage: str, *, mode: str = "explore"
) -> tuple[str, ...]:
    """Return model-lab lint rules for prompt self-check sections."""
    normalized_stage = normalize_workflow_stage(stage, allow_stage3_helper=True)
    normalized_mode = (mode or "").strip().lower() or "explore"
    if normalized_stage == MECHANISM_DISCOVERY:
        return (
            "必须包含 [模型机制候选] / [实现假设草图] / [与当前 spec / baseline 的关系] / [不确定性与失败路径] 四段。",
            "模型机制候选至少 2 个；每个候选必须写清 touched contract surfaces 与 concern。",
            "机制发现阶段不得输出最终 spec patch、不得推荐 single best model、不得把方向写成单纯调参。",
            "必须至少覆盖两类不同模型机制：loss/regularization、feature interaction、target construction、sample weighting、training window、model selection 中的两类。",
            "每个机制候选应能落入 Stage 1 mechanism schema：包含 hypothesis / signal_sketch / data_needs，"
            "inspired_by / fusion_of / cross_domain_jump 可选；无来源不算缺口。",
        )
    if normalized_stage == SIGNAL_MAPPING:
        rules = [
            "必须包含 [Model Mechanism Mapping] / [当前实现解释] / [模型风险控制] / [可测试模型版本] 四段。",
            "模型风险控制必须逐项覆盖 feature availability / PIT、label / target leakage、overfit / complexity、turnover / cost、feature instability、split / regime fragility。",
            "每个风险项必须给出 {规避 / 显式控制 / 压力测试 / 暂不控制} 之一的处理判定。",
            "不得出现“推荐版本”“最优版本”“final pick”“I recommend”等最终选择语言；"
            "最终选择属于 Stage 3 数据验证。",
        ]
        if normalized_mode == "constrained":
            rules.append("constrained 模式必须输出 2-3 个可测试模型版本，并对每个 necessary 字段写 remove-and-test 理由。")
        else:
            rules.append("建议输出 2-3 个可测试模型版本；少于 2 个会被标记为 warning。")
        return tuple(rules)
    rules = [
        "必须包含 [Alias / 问题归因审计] / [数据与时间完整性] / [训练与验证稳健性] / [特征与解释稳定性] / [成本与组合影响] / [最终判定] 六段。",
        "Alias / 问题归因审计必须逐项覆盖 baseline linear/ridge、regularization-only、feature-count / complexity、leakage / PIT、split luck / regime overfit、turnover / cost artifact。",
        "每个 alias 必须给出 {显著风险 / 部分风险 / 不构成风险} 之一的判定。",
        "最终判定不得使用“看情况”“需要更多数据”“进一步研究”等回避语。",
    ]
    if normalized_mode == "constrained":
        rules.append("constrained 模式最终判定只能是 KILL 或 HOLD-FOR-AUDIT。")
    else:
        rules.append("最终判定必须明确落在 KILL / HOLD / ITERATE / HOLD-FOR-AUDIT 之一。")
    return tuple(rules)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _required_sections_for_stage(
    normalized_stage: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if normalized_stage == MECHANISM_DISCOVERY:
        return _MECHANISM_DISCOVERY_REQUIRED
    if normalized_stage == SIGNAL_MAPPING:
        return _SIGNAL_MAPPING_REQUIRED
    return _VALIDATION_REQUIRED


def _model_required_sections_for_stage(
    normalized_stage: str,
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    if normalized_stage == MECHANISM_DISCOVERY:
        return _MODEL_MECHANISM_REQUIRED
    if normalized_stage == SIGNAL_MAPPING:
        return _MODEL_SIGNAL_REQUIRED
    return _MODEL_VALIDATION_REQUIRED


def _check_required_sections(
    text: str,
    required: Iterable[tuple[str, tuple[str, ...]]],
    violations: list[LintViolation],
    sections_seen: list[str],
) -> None:
    """For each (canonical_label, alternates) pair, check at least one alt
    appears in text. Records the canonical label in sections_seen on hit;
    emits a missing_section violation otherwise.
    """
    lowered = text.lower()
    for canonical, alternates in required:
        if any(alt.lower() in lowered for alt in alternates):
            sections_seen.append(canonical)
        else:
            violations.append(
                LintViolation(
                    code="missing_section",
                    severity="error",
                    section=canonical,
                    detail=(
                        f"required section not found: {canonical}; "
                        f"any of {list(alternates)} would have matched"
                    ),
                )
            )


def _check_phrase_ban(
    text: str,
    *,
    patterns: tuple[str, ...],
    code: str,
    severity: str,
    detail_template: str,
    violations: list[LintViolation],
    section: str = "",
    flags: int = re.IGNORECASE,
) -> None:
    """Emit one violation per pattern that fires (at most once per pattern)."""
    for pattern in patterns:
        match = re.search(pattern, text, flags=flags)
        if match:
            violations.append(
                LintViolation(
                    code=code,
                    severity=severity,
                    section=section,
                    detail=detail_template.format(match=match.group(0)),
                )
            )


def _extract_section_body(text: str, *header_alternates: str) -> str:
    """Return the slice between a matching section header and the next ## /
    [bracket-section] / end-of-text. Used to scope per-section checks.
    """
    lowered = text.lower()
    start: int | None = None
    for header in header_alternates:
        idx = lowered.find(header.lower())
        if idx != -1:
            start = idx + len(header)
            break
    if start is None:
        return ""
    # Find next section break: a line beginning with `## `, `[`, `### 机制`, etc.
    body = text[start:]
    end_match = re.search(r"\n##\s|\n\[", body)
    return body[: end_match.start()] if end_match else body


# ---------------------------------------------------------------------------
# mechanism_discovery
# ---------------------------------------------------------------------------


_MECHANISM_HEADING_PATTERN = re.compile(
    r"^(?:###\s*(?:机制|Mechanism)\s*\d+"
    r"|\*?\*?机制\s*\d+\s*[：:]"
    r"|-\s*\*?\*?(?:机制|Mechanism)\s*\d+\s*[：:])",
    re.MULTILINE,
)
_MECHANISM_TABLE_ROW_PATTERN = re.compile(
    r"^\|\s*(?:\*\*)?\s*\d+\s*[\.\、]\s*[^|]+?\s*\|",
    re.MULTILINE,
)


def _count_mechanism_candidates(text: str) -> int:
    heading_count = len(_MECHANISM_HEADING_PATTERN.findall(text))
    table_count = len(_MECHANISM_TABLE_ROW_PATTERN.findall(text))
    return max(heading_count, table_count)


def _looks_like_mechanism_table(text: str) -> bool:
    lowered = text.lower()
    return (
        _count_mechanism_candidates(text) >= 2
        and "机制候选" in text
        and ("经济解释" in text or "economic" in lowered)
        and ("失败路径" in text or "反例" in text or "failure" in lowered)
    )


def _relax_start_mechanism_table_sections(
    violations: list[LintViolation], text: str, sections_seen: list[str]
) -> list[LintViolation]:
    if not _looks_like_mechanism_table(text):
        return violations
    if "机制候选" not in sections_seen:
        sections_seen.append("机制候选")
    relaxed: list[LintViolation] = []
    for violation in violations:
        if violation.code == "missing_section":
            relaxed.append(
                LintViolation(
                    code=violation.code,
                    severity="warning",
                    section=violation.section,
                    detail=(
                        f"{violation.detail}; mechanism-only kickoff table accepted"
                    ),
                )
            )
        else:
            relaxed.append(violation)
    return relaxed


def _lint_mechanism_discovery(text: str, mode: str) -> LintReport:
    violations: list[LintViolation] = []
    sections_seen: list[str] = []

    _check_required_sections(
        text, _MECHANISM_DISCOVERY_REQUIRED, violations, sections_seen
    )

    _check_phrase_ban(
        text,
        patterns=_DIRECTION_PHRASE_PATTERNS,
        code="forbidden_direction",
        severity="error",
        detail_template=(
            "predetermined return-direction phrase used: {match!r}; "
            "mechanism_discovery forbids long/short presupposition"
        ),
        violations=violations,
    )

    # Forbidden labels appearing as a mechanism heading (premature naming).
    for match in _MECHANISM_HEADING_PATTERN.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        if line_end == -1:
            line_end = len(text)
        line = text[line_start:line_end]
        line_lower = line.lower()
        for label in FORBIDDEN_FACTOR_LABELS:
            label_lower = label.lower()
            # Word-boundary check for ASCII labels; substring for CJK.
            if label_lower.isascii():
                if re.search(rf"\b{re.escape(label_lower)}\b", line_lower):
                    violations.append(
                        LintViolation(
                            code="forbidden_label_in_name",
                            severity="error",
                            section=line.strip()[:60],
                            detail=(
                                f"mechanism heading uses forbidden label "
                                f"{label!r}; rename to mechanism vocabulary"
                            ),
                        )
                    )
                    break
            else:
                if label_lower in line_lower:
                    violations.append(
                        LintViolation(
                            code="forbidden_label_in_name",
                            severity="error",
                            section=line.strip()[:60],
                            detail=(
                                f"mechanism heading uses forbidden label "
                                f"{label!r}; rename to mechanism vocabulary"
                            ),
                        )
                    )
                    break

    if mode == "start":
        violations = _relax_start_mechanism_table_sections(
            violations, text, sections_seen
        )

    mechanism_count = _count_mechanism_candidates(text)
    if mechanism_count < 2:
        violations.append(
            LintViolation(
                code="single_mechanism",
                severity="warning" if mode == "start" else "error",
                section="机制候选",
                detail=(
                    f"expected ≥2 candidate mechanisms; "
                    f"found {mechanism_count}"
                ),
            )
        )

    return LintReport(
        stage=MECHANISM_DISCOVERY,
        mode=mode,
        violations=tuple(violations),
        sections_seen=tuple(sections_seen),
    )


# ---------------------------------------------------------------------------
# signal_mapping
# ---------------------------------------------------------------------------


_CONFOUND_VERDICT_PATTERN = re.compile(r"包含|残差化|显式控制|不控制")
_VERSION_LINE_PATTERN = re.compile(
    r"^\s*[-*]?\s*v\s*[123]\s*[：:]", re.MULTILINE | re.IGNORECASE
)


def _lint_signal_mapping(text: str, mode: str) -> LintReport:
    violations: list[LintViolation] = []
    sections_seen: list[str] = []

    _check_required_sections(text, _SIGNAL_MAPPING_REQUIRED, violations, sections_seen)

    # Confound completeness — each of the 5 canonical labels must have a
    # verdict from {包含 / 残差化 / 显式控制 / 不控制}.
    confound_section = _extract_section_body(text, "[Confound 控制]", "[confound 控制]")
    if confound_section:
        for label, _detail in SIGNAL_MAPPING_CONFOUND_CONTROLS:
            label_lower = label.lower()
            confound_lower = confound_section.lower()
            if label_lower not in confound_lower:
                violations.append(
                    LintViolation(
                        code="confound_missing",
                        severity="error",
                        section="Confound 控制",
                        detail=f"confound family {label!r} not addressed in section",
                    )
                )
                continue
            # Find the line that mentions this label and check it has a verdict.
            for line in confound_section.splitlines():
                if label_lower in line.lower():
                    if not _CONFOUND_VERDICT_PATTERN.search(line):
                        violations.append(
                            LintViolation(
                                code="confound_verdict_missing",
                                severity="error",
                                section="Confound 控制",
                                detail=(
                                    f"confound {label!r}: no verdict from "
                                    f"{{包含 / 残差化 / 显式控制 / 不控制}}"
                                ),
                            )
                        )
                    break

    _check_phrase_ban(
        text,
        patterns=_FINAL_PICK_PHRASE_PATTERNS,
        code="final_pick",
        severity="error",
        detail_template=(
            "final-pick language used: {match!r}; signal_mapping must NOT "
            "select a winner — defer to Stage 3 data validation"
        ),
        violations=violations,
    )

    versions_section = _extract_section_body(text, "[可测试信号版本]")
    version_count = len(_VERSION_LINE_PATTERN.findall(versions_section or text))
    if mode == "constrained":
        if version_count < 2 or version_count > 3:
            violations.append(
                LintViolation(
                    code="version_count_out_of_range",
                    severity="error",
                    section="可测试信号版本",
                    detail=(
                        f"strict mode requires 2-3 testable signal versions; "
                        f"found {version_count}"
                    ),
                )
            )
    elif version_count < 2:
        violations.append(
            LintViolation(
                code="version_count_low",
                severity="warning",
                section="可测试信号版本",
                detail=(
                    f"only {version_count} testable signal version(s); "
                    f"prefer 2-3 to keep the hypothesis space open"
                ),
            )
        )

    return LintReport(
        stage=SIGNAL_MAPPING,
        mode=mode,
        violations=tuple(violations),
        sections_seen=tuple(sections_seen),
    )


# ---------------------------------------------------------------------------
# validation_kill_tests
# ---------------------------------------------------------------------------


_ALIAS_VERDICT_PATTERN = re.compile(r"显著重叠|部分重叠|不重叠")
_FINAL_VERDICT_PATTERN = re.compile(
    r"\bKILL\b|\bHOLD(?:-FOR-AUDIT)?\b|\bITERATE\b", re.IGNORECASE
)
_BINARY_VERDICT_PATTERN = re.compile(
    r"\bKILL\b|\bHOLD-FOR-AUDIT\b|HOLD\s*-?\s*FOR\s*-?\s*AUDIT", re.IGNORECASE
)
_KNOWLEDGE_ANCHOR_PATTERN = re.compile(
    r"\[K\d+\]"
    r"|Jegadeesh|Titman|Amihud|Ang\s*et\s*al"
    r"|Fama|French|Carhart|Asness|Frazzini|Pedersen"
    r"|Pastor|Stambaugh",
    re.IGNORECASE,
)


def _lint_validation_kill_tests(text: str, mode: str) -> LintReport:
    violations: list[LintViolation] = []
    sections_seen: list[str] = []

    _check_required_sections(text, _VALIDATION_REQUIRED, violations, sections_seen)

    alias_section = _extract_section_body(
        text, "[Alias / 换壳审计]", "[alias / 换壳审计]"
    )
    if alias_section:
        for label, _detail in VALIDATION_ALIAS_TARGETS:
            label_lower = label.lower()
            if label_lower not in alias_section.lower():
                violations.append(
                    LintViolation(
                        code="alias_target_missing",
                        severity="error",
                        section="Alias 审计",
                        detail=f"alias target {label!r} not addressed",
                    )
                )
                continue
            # Find the line containing the label and check verdict.
            for line in alias_section.splitlines():
                if label_lower in line.lower():
                    if not _ALIAS_VERDICT_PATTERN.search(line):
                        violations.append(
                            LintViolation(
                                code="alias_verdict_missing",
                                severity="error",
                                section="Alias 审计",
                                detail=(
                                    f"alias {label!r}: no verdict from "
                                    f"{{显著重叠 / 部分重叠 / 不重叠}}"
                                ),
                            )
                        )
                    if mode == "constrained" and not _KNOWLEDGE_ANCHOR_PATTERN.search(
                        line
                    ):
                        violations.append(
                            LintViolation(
                                code="alias_unanchored",
                                severity="error",
                                section="Alias 审计",
                                detail=(
                                    f"strict mode: alias {label!r} verdict "
                                    f"missing [Kx] citation or standard "
                                    f"baseline (Jegadeesh-Titman / Amihud / "
                                    f"Ang et al / Fama-French / etc.)"
                                ),
                            )
                        )
                    break

    final_section = _extract_section_body(text, "[最终判定]")
    if final_section:
        _check_phrase_ban(
            final_section,
            patterns=_HEDGING_PHRASE_PATTERNS,
            code="hedging_verdict",
            severity="error",
            detail_template=(
                "hedging language in final verdict: {match!r}; "
                "validation_kill_tests requires a concrete verdict"
            ),
            violations=violations,
            section="最终判定",
        )
        if mode == "constrained":
            verdict_text = _extract_verdict_text(final_section)
            if not _BINARY_VERDICT_PATTERN.search(verdict_text):
                violations.append(
                    LintViolation(
                        code="missing_binary_verdict",
                        severity="error",
                        section="最终判定",
                        detail=(
                            "strict mode requires explicit "
                            "{KILL / HOLD-FOR-AUDIT} verdict"
                        ),
                    )
                )
        elif not _FINAL_VERDICT_PATTERN.search(_extract_verdict_text(final_section)):
            violations.append(
                LintViolation(
                    code="missing_verdict",
                    severity="error",
                    section="最终判定",
                    detail=(
                        "no explicit verdict found; expected one of "
                        "{KILL / HOLD / ITERATE / HOLD-FOR-AUDIT}"
                    ),
                )
            )

    return LintReport(
        stage=VALIDATION_KILL_TESTS,
        mode=mode,
        violations=tuple(violations),
        sections_seen=tuple(sections_seen),
    )


def _extract_verdict_text(section: str) -> str:
    """Return likely verdict lines, excluding kill-test labels."""
    lines: list[str] = []
    for line in section.splitlines():
        lowered = line.lower()
        if "hard kill" in lowered or "kill trigger" in lowered:
            continue
        if "verdict" in lowered or "判定" in line:
            lines.append(line)
    return "\n".join(lines) if lines else section


# ---------------------------------------------------------------------------
# model-lab: mechanism_discovery
# ---------------------------------------------------------------------------


_MODEL_DIRECTION_HEADING_PATTERN = re.compile(
    r"^(?:###\s*(?:方向|机制|Direction|Mechanism)\s*\d+"
    r"|\*?\*?(?:方向|机制)\s*\d+\s*[：:]"
    r"|-\s*\*?\*?(?:方向|机制|Direction|Mechanism)\s*\d+\s*[：:])",
    re.MULTILINE | re.IGNORECASE,
)

_MODEL_FINAL_PATCH_PATTERN = re.compile(
    r"```json|\bpatch_fields\b|\bspec\s+patch\b|推荐版本|最优版本|single\s+best|final\s+pick",
    re.IGNORECASE,
)

_MODEL_MECHANISM_FAMILY_TOKENS: tuple[str, ...] = (
    "loss",
    "regularization",
    "feature interaction",
    "target construction",
    "sample weighting",
    "training window",
    "model selection",
    "损失",
    "正则",
    "特征交互",
    "目标构造",
    "样本权重",
    "训练窗口",
    "模型选择",
)


def _lint_model_mechanism_discovery(text: str, mode: str) -> LintReport:
    violations: list[LintViolation] = []
    sections_seen: list[str] = []
    _check_required_sections(
        text, _MODEL_MECHANISM_REQUIRED, violations, sections_seen
    )

    direction_count = len(_MODEL_DIRECTION_HEADING_PATTERN.findall(text))
    if direction_count < 2:
        violations.append(
            LintViolation(
                code="single_model_direction",
                severity="warning" if mode == "start" else "error",
                section="模型机制候选",
                detail=f"expected >=2 candidate model directions; found {direction_count}",
            )
        )

    if _MODEL_FINAL_PATCH_PATTERN.search(text):
        violations.append(
            LintViolation(
                code="premature_model_convergence",
                severity="error",
                section="模型机制候选",
                detail="mechanism_discovery must not output final spec patches, final picks, or single-best recommendations",
            )
        )

    family_hits = {
        token
        for token in _MODEL_MECHANISM_FAMILY_TOKENS
        if token.lower() in text.lower()
    }
    if len(family_hits) < 2:
        violations.append(
            LintViolation(
                code="mechanism_family_diversity_low",
                severity="warning",
                section="模型机制候选",
                detail="expected at least two distinct model-mechanism families such as loss/regularization/target/sample-weighting/window/selection",
            )
        )

    return LintReport(
        stage=MECHANISM_DISCOVERY,
        mode=mode,
        violations=tuple(violations),
        sections_seen=tuple(sections_seen),
    )


# ---------------------------------------------------------------------------
# model-lab: signal_mapping
# ---------------------------------------------------------------------------


_MODEL_RISK_VERDICT_PATTERN = re.compile(r"规避|显式控制|压力测试|暂不控制")
_MODEL_VERSION_LINE_PATTERN = re.compile(
    r"^\s*[-*]?\s*(?:v|版本)\s*[123]\s*[：:]",
    re.MULTILINE | re.IGNORECASE,
)


def _lint_model_signal_mapping(text: str, mode: str) -> LintReport:
    violations: list[LintViolation] = []
    sections_seen: list[str] = []
    _check_required_sections(text, _MODEL_SIGNAL_REQUIRED, violations, sections_seen)

    risk_section = _extract_section_body(text, "[模型风险控制]")
    if risk_section:
        risk_lower = risk_section.lower()
        for label, _detail in MODEL_SIGNAL_RISK_CONTROLS:
            label_lower = label.lower()
            if label_lower not in risk_lower:
                violations.append(
                    LintViolation(
                        code="model_risk_missing",
                        severity="error",
                        section="模型风险控制",
                        detail=f"model risk family {label!r} not addressed",
                    )
                )
                continue
            for line in risk_section.splitlines():
                if label_lower in line.lower():
                    if not _MODEL_RISK_VERDICT_PATTERN.search(line):
                        violations.append(
                            LintViolation(
                                code="model_risk_verdict_missing",
                                severity="error",
                                section="模型风险控制",
                                detail=(
                                    f"model risk {label!r}: no verdict from "
                                    "{规避 / 显式控制 / 压力测试 / 暂不控制}"
                                ),
                            )
                        )
                    break

    _check_phrase_ban(
        text,
        patterns=_FINAL_PICK_PHRASE_PATTERNS,
        code="final_pick",
        severity="error",
        detail_template=(
            "final-pick language used: {match!r}; signal_mapping must not select a winner"
        ),
        violations=violations,
    )

    version_section = _extract_section_body(text, "[可测试模型版本]")
    version_count = len(_MODEL_VERSION_LINE_PATTERN.findall(version_section or text))
    if mode == "constrained":
        if version_count < 2 or version_count > 3:
            violations.append(
                LintViolation(
                    code="model_version_count_out_of_range",
                    severity="error",
                    section="可测试模型版本",
                    detail=f"strict mode requires 2-3 testable model versions; found {version_count}",
                )
            )
    elif version_count < 2:
        violations.append(
            LintViolation(
                code="model_version_count_low",
                severity="warning",
                section="可测试模型版本",
                detail=f"only {version_count} model version(s); prefer 2-3 alternatives",
            )
        )

    return LintReport(
        stage=SIGNAL_MAPPING,
        mode=mode,
        violations=tuple(violations),
        sections_seen=tuple(sections_seen),
    )


# ---------------------------------------------------------------------------
# model-lab: validation_kill_tests
# ---------------------------------------------------------------------------


_MODEL_ALIAS_VERDICT_PATTERN = re.compile(r"显著风险|部分风险|不构成风险")


def _lint_model_validation_kill_tests(text: str, mode: str) -> LintReport:
    violations: list[LintViolation] = []
    sections_seen: list[str] = []
    _check_required_sections(text, _MODEL_VALIDATION_REQUIRED, violations, sections_seen)

    alias_section = _extract_section_body(text, "[Alias / 问题归因审计]")
    if alias_section:
        alias_lower = alias_section.lower()
        for label, _detail in MODEL_VALIDATION_ALIAS_TARGETS:
            label_lower = label.lower()
            if label_lower not in alias_lower:
                violations.append(
                    LintViolation(
                        code="model_alias_target_missing",
                        severity="error",
                        section="Alias / 问题归因审计",
                        detail=f"model alias target {label!r} not addressed",
                    )
                )
                continue
            for line in alias_section.splitlines():
                if label_lower in line.lower():
                    if not _MODEL_ALIAS_VERDICT_PATTERN.search(line):
                        violations.append(
                            LintViolation(
                                code="model_alias_verdict_missing",
                                severity="error",
                                section="Alias / 问题归因审计",
                                detail=(
                                    f"alias {label!r}: no verdict from "
                                    "{显著风险 / 部分风险 / 不构成风险}"
                                ),
                            )
                        )
                    break

    final_section = _extract_section_body(text, "[最终判定]")
    if final_section:
        _check_phrase_ban(
            final_section,
            patterns=_HEDGING_PHRASE_PATTERNS,
            code="hedging_verdict",
            severity="error",
            detail_template="hedging language in final verdict: {match!r}",
            violations=violations,
            section="最终判定",
        )
        if mode == "constrained":
            verdict_text = _extract_verdict_text(final_section)
            if not _BINARY_VERDICT_PATTERN.search(verdict_text):
                violations.append(
                    LintViolation(
                        code="missing_binary_verdict",
                        severity="error",
                        section="最终判定",
                        detail="strict mode requires explicit {KILL / HOLD-FOR-AUDIT} verdict",
                    )
                )
        elif not _FINAL_VERDICT_PATTERN.search(_extract_verdict_text(final_section)):
            violations.append(
                LintViolation(
                    code="missing_verdict",
                    severity="error",
                    section="最终判定",
                    detail="expected one of {KILL / HOLD / ITERATE / HOLD-FOR-AUDIT}",
                )
            )

    return LintReport(
        stage=VALIDATION_KILL_TESTS,
        mode=mode,
        violations=tuple(violations),
        sections_seen=tuple(sections_seen),
    )
