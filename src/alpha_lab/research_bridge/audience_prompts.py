"""Audience-based Stage 0 prompt distribution.

Stage 1 of the idea explorer is split into two audiences (see
``docs/research_workflow.md``):

- ``CLAUDE_MECHANISM`` — generator. Produces ``mechanism_deepdive.md``: a
  ledger of candidate mechanisms drawn from vault ``transferable_moves`` and
  cross-card synthesis. Does not look at the codebase.
- ``CODEX_REVIEW`` — reviewer. Produces ``code_feasibility_review.md``: for
  each candidate mechanism the generator will propose, judge whether it is
  executable in the v1 ``factor.json`` / ``ModelFactorCaseSpec`` schema, what
  columns are missing, and what validator rules might block it. Does not
  propose new mechanisms or veto.

The two audiences see the **same** vault retrieval pack so they argue from
identical evidence. The reviewer additionally sees a codebase snapshot
(existing factors / candidates / cases / schema / validator rules); the
generator does not.

This module is the single source of truth for the prompt body. Both
``service.distribute_idea`` (single-factor lab) and
``model_idea.distribute_model_idea`` (model lab) call ``build_prompt`` here.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

from alpha_lab.research_bridge.codebase_index import CodebaseSnapshot


class Lab(str, Enum):
    """Which research lab the idea belongs to."""

    SINGLE_FACTOR = "single_factor"
    MODEL_FACTOR = "model_factor"


class Audience(str, Enum):
    """Stage 1 audience role."""

    CLAUDE_MECHANISM = "claude_mechanism"
    CODEX_REVIEW = "codex_review"


DEFAULT_AUDIENCES: tuple[Audience, ...] = (
    Audience.CLAUDE_MECHANISM,
    Audience.CODEX_REVIEW,
)


@dataclass(frozen=True)
class CardForPrompt:
    """Subset of an :class:`ExploreIdeaCard` that the prompt actually needs."""

    name: str
    path: str
    summary: str
    transferable_moves: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromptContext:
    """All inputs required by both audience prompts.

    Lives next to the produced files so the prompt body can render absolute
    paths back into ``ideas/<idea_id>/`` for the agents.
    """

    idea: str
    idea_id: str
    lab: Lab
    draft_dir: Path
    vault_root: Path
    related_cards: Sequence[CardForPrompt]
    insight_brief: tuple[str, ...]
    codebase: CodebaseSnapshot
    mode: str = "start"


def normalize_audiences(values: Sequence[str] | str | None) -> tuple[Audience, ...]:
    """Parse the CLI-style ``--audiences`` arg into a unique ordered tuple."""

    if values is None:
        return DEFAULT_AUDIENCES
    if isinstance(values, str):
        raw = [values]
    else:
        raw = list(values)
    items: list[str] = []
    for item in raw:
        items.extend(str(item).split(","))
    out: list[Audience] = []
    for item in items:
        token = item.strip().lower()
        if not token:
            continue
        try:
            audience = Audience(token)
        except ValueError as exc:
            raise ValueError(
                f"unknown audience {token!r}; expected one of "
                f"{[a.value for a in Audience]}"
            ) from exc
        if audience not in out:
            out.append(audience)
    if not out:
        raise ValueError("at least one audience must be provided")
    return tuple(out)


def build_prompt(*, audience: Audience, ctx: PromptContext) -> str:
    """Render the Markdown prompt for one audience."""

    if audience is Audience.CLAUDE_MECHANISM:
        return _build_claude_mechanism_prompt(ctx)
    if audience is Audience.CODEX_REVIEW:
        return _build_codex_review_prompt(ctx)
    raise ValueError(f"unhandled audience: {audience}")


# ---------------------------------------------------------------------------
# Generator prompt (Claude Code, audience=claude_mechanism)
# ---------------------------------------------------------------------------


def _build_claude_mechanism_prompt(ctx: PromptContext) -> str:
    lines: list[str] = [
        f"# Stage 1 Generator Prompt — claude_mechanism (lab={ctx.lab.value})",
        "",
        f"idea_id: `{ctx.idea_id}`",
        "",
        "## 1. Idea",
        ctx.idea,
        "",
        "## 2. 你的角色（generator）",
        "- 你是 Stage 1 generator。你的任务是基于下面的 vault 卡 + transferable_moves "
        "提出 3-8 条互补的候选机制。",
        "- 你**不读** Codex GUI 的输出（reviewer 输出对你不可见，反之亦然）。",
        "- 你**不评审**自己机制的可执行性——可执行性由 reviewer 在另一份 prompt 里独立判断。",
        "- 你**不输出** factor 定义、case spec、ranking、单一 best idea。",
        "",
        "## 3. 工作目录",
        f"- draft_dir：`{ctx.draft_dir}`",
        f"- vault_root：`{ctx.vault_root}`",
        f"- 写入：`{ctx.draft_dir / 'mechanism_deepdive.md'}`",
        "- 不创建其他文件，不修改代码库。",
        "",
        "## 4. Stage 1 纪律",
        "- 候选机制只增不减，不做 keep/kill。",
        "- vault 是素材库，不是判决书；`transferable_moves` 是主要生成原料。",
        "- `operative_claims` 只是弱上下文 hint，不能作为淘汰条件。",
        "- 机制命名不要直接使用 reversal / momentum / value / quality / size / "
        "liquidity 等既有标签——保护假设空间，描述机制本身。",
        "- novel synthesis 合法；找不到来源卡片不是缺口。",
        "",
        "## 5. 相关卡片（共享 retrieval 上下文，与 reviewer prompt 一字不差相同）",
        "卡片路径相对于 vault root；请按 path 打开 quant-knowledge 原文后再写 Markdown。",
    ]
    lines.extend(_format_cards(ctx.related_cards))
    lines.extend(_format_insight_brief(ctx.insight_brief))
    lines.extend(
        [
            "",
            "## 6. Markdown 输出格式",
            f"写入 `{ctx.draft_dir / 'mechanism_deepdive.md'}`，包含：",
            "- `阅读摘要`：实际打开了哪些卡片，分别拿走了什么 move。",
            "- `候选机制 ledger`：3-8 条机制，每条按 ledger schema：",
            "  ```yaml",
            "  mechanism_<id>:",
            "    hypothesis: \"\"",
            "    inspired_by: []          # optional",
            "    fusion_of: []             # optional [card_a, card_b]",
            "    novel_delta: \"\"",
            "    signal_sketch: \"\"",
            "    data_needs: []",
            "    concern: \"\"",
            "  ```",
            "- `进入 Stage 2 的讨论问题`：给网页版 GPT 和用户继续融合时使用。",
            "",
            "**禁止**输出 YAML 文件、final factor code、case spec、portfolio 建议。",
            "**禁止**给机制贴 reversal / momentum 等 canonical 标签。",
        ]
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Reviewer prompt (Codex GUI, audience=codex_review)
# ---------------------------------------------------------------------------


def _build_codex_review_prompt(ctx: PromptContext) -> str:
    lab_payload_label = (
        "factor.json" if ctx.lab is Lab.SINGLE_FACTOR else "ModelFactorCaseSpec"
    )
    review_status_choices = (
        "in_contract_factor_def | partial_in_contract | needs_extension | future_enhancement"
        if ctx.lab is Lab.SINGLE_FACTOR
        else "in_contract_spec_variant | partial_in_contract | needs_extension | future_enhancement"
    )
    lines: list[str] = [
        f"# Stage 1 Reviewer Prompt — codex_review (lab={ctx.lab.value})",
        "",
        f"idea_id: `{ctx.idea_id}`",
        "",
        "## 1. Idea",
        ctx.idea,
        "",
        "## 2. 你的角色（reviewer，不是 generator）",
        "- 你是 Stage 1 **reviewer**。你的任务是评审 generator（Claude Code）将基于"
        " 下方相同的 vault 卡提出的每条候选机制，在当前 v1 schema + validator 硬约束下"
        " 能不能落地。",
        "- 你**不读** generator 的输出（mechanism_deepdive.md 对你不可见）。当 generator"
        " 还没产出输出时，你也可以先基于卡片预判：哪些可能机制可执行、哪些缺什么。",
        "- 你**不写**新机制、不重写假设、不否决任何机制；不可执行的标 `needs_extension`"
        " 加注缺什么即可。",
        "- 你**不修改**代码库、不调用 alpha-lab，仅产出一份评审 markdown。",
        "",
        "## 3. 工作目录",
        f"- draft_dir：`{ctx.draft_dir}`",
        f"- vault_root：`{ctx.vault_root}`",
        f"- 写入：`{ctx.draft_dir / 'code_feasibility_review.md'}`",
        "- 不创建其他文件，不修改代码库。",
        "",
        "## 4. 相关卡片（与 generator prompt 一字不差相同）",
    ]
    lines.extend(_format_cards(ctx.related_cards))
    lines.extend(_format_insight_brief(ctx.insight_brief))
    lines.extend(
        [
            "",
            "## 5. 代码库索引（reviewer 专用上下文）",
            "用于判断重做、命名冲突、可触达 schema 字段。",
            "",
            "**已有 single-factor**：",
            f"- promoted: {_format_inline_list(ctx.codebase.factors_promoted)}",
            f"- research: {_format_inline_list(ctx.codebase.factors_research)}",
            "",
            "**已有 model-factor candidate**：",
            f"- promoted: {_format_inline_list(ctx.codebase.model_candidates_promoted)}",
            f"- research: {_format_inline_list(ctx.codebase.model_candidates_research)}",
            "",
            "**已注册 case yaml**：",
            f"- single_factor: {_format_inline_list(ctx.codebase.single_factor_cases)}",
            f"- model_factor: {_format_inline_list(ctx.codebase.model_factor_cases)}",
            "",
            "## 6. Schema + Validator 硬约束",
        ]
    )
    if ctx.lab is Lab.SINGLE_FACTOR:
        lines.extend(
            [
                "",
                "**factor.json required keys**：",
                f"- {', '.join(ctx.codebase.factor_json_required_keys)}",
                "",
                "**factor validator 硬规则（paraphrased）**：",
            ]
        )
        for rule in ctx.codebase.factor_validator_rules:
            lines.append(f"- {rule}")
    else:
        lines.extend(
            [
                "",
                "**ModelFactorCaseSpec top-level fields**：",
                f"- {', '.join(ctx.codebase.model_case_spec_top_keys)}",
                "",
                "**model validator 硬规则（paraphrased）**：",
            ]
        )
        for rule in ctx.codebase.model_validator_rules:
            lines.append(f"- {rule}")

    lines.extend(
        [
            "",
            "## 7. Markdown 输出格式",
            f"写入 `{ctx.draft_dir / 'code_feasibility_review.md'}`，对你预判"
            " generator 可能提出的每条候选机制（或 generator 已产出后逐条），输出 YAML 评审条目：",
            "",
            "```yaml",
            "mechanism_<id>:",
            "  in_v1_contract: true | false",
            "  required_columns_present: []",
            "  required_columns_missing: []",
            f"  {'factor_json_keys_touched' if ctx.lab is Lab.SINGLE_FACTOR else 'spec_fields_touched'}: []",
            "  validator_blockers: []",
            f"  implementation_status: \"{review_status_choices}\"",
            "  reviewer_note: \"\"",
            "```",
            "",
            f"判定 `implementation_status` 时严格对照第 6 节列出的 {lab_payload_label}"
            " required/支持字段 + validator 硬规则；不要凭机制名主观判定。",
            "",
            "**禁止**：提新机制、重写 generator 假设、否决候选、改写 schema/validator 规则。",
        ]
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _format_cards(cards: Sequence[CardForPrompt]) -> list[str]:
    if not cards:
        return [
            "",
            "- 未命中相关卡片；可以做 novel synthesis，但需要把假设边界写清楚。",
        ]
    lines: list[str] = []
    for idx, card in enumerate(cards, start=1):
        lines.extend(
            [
                "",
                f"### K{idx}: {card.name}",
                f"- path: `{card.path}`",
                f"- summary: {card.summary}",
            ]
        )
        if card.transferable_moves:
            lines.append("- transferable_moves:")
            for move in card.transferable_moves[:4]:
                lines.append(f"  - {move}")
        else:
            lines.append("- transferable_moves: []")
    return lines


def _format_insight_brief(briefs: Sequence[str]) -> list[str]:
    lines: list[str] = ["", "### Cross-card synthesis"]
    if not briefs:
        lines.append(
            "- 本轮未生成可用的跨卡合成摘要；请以相关卡片原文为准自行寻找可迁移动作。"
        )
        return lines
    lines.append(
        "以下只作为候选生成素材；不是约束或 keep/kill 规则。"
    )
    for brief in briefs:
        text = brief.strip()
        if not text:
            continue
        lines.append(f"- {text}")
    return lines


def _format_inline_list(items: Sequence[str]) -> str:
    if not items:
        return "（空）"
    return ", ".join(items)
