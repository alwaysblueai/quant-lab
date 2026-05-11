"""Engine-based Stage 0 prompt distribution (symmetric task).

Stage 1 of the idea explorer (see ``docs/end_to_end_workflow.md``) gives the
**same** task to two desktop agents (Claude Code and Codex GUI). Both engines
do generator + reviewer合一:

- generator: propose 3-8 candidate mechanisms (ledger schema)
- reviewer: for each proposed mechanism, judge whether it is executable in
  the v1 ``factor.json`` / ``ModelFactorCaseSpec`` schema; tag with
  ``implementation_status``

The two engines run in parallel and **互不可见**. The web GPT综合两份输出
取长补短 — different models have different strengths, the combination
beats either alone.

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


class Engine(str, Enum):
    """Stage 1 desktop agent identity."""

    CLAUDE = "claude"
    CODEX = "codex"


DEFAULT_ENGINES: tuple[Engine, ...] = (Engine.CLAUDE, Engine.CODEX)


@dataclass(frozen=True)
class CardForPrompt:
    """Subset of an :class:`ExploreIdeaCard` that the prompt actually needs."""

    name: str
    path: str
    summary: str
    transferable_moves: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromptContext:
    """All inputs required by the Stage 1 prompt.

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


def normalize_engines(values: Sequence[str] | str | None) -> tuple[Engine, ...]:
    """Parse the CLI-style ``--engines`` arg into a unique ordered tuple."""

    if values is None:
        return DEFAULT_ENGINES
    if isinstance(values, str):
        raw = [values]
    else:
        raw = list(values)
    items: list[str] = []
    for item in raw:
        items.extend(str(item).split(","))
    out: list[Engine] = []
    for item in items:
        token = item.strip().lower()
        if not token:
            continue
        try:
            engine = Engine(token)
        except ValueError as exc:
            raise ValueError(
                f"unknown engine {token!r}; expected one of "
                f"{[e.value for e in Engine]}"
            ) from exc
        if engine not in out:
            out.append(engine)
    if not out:
        raise ValueError("at least one engine must be provided")
    return tuple(out)


def output_filename_for(engine: Engine) -> str:
    """Where the engine writes its Stage 1 markdown output."""

    return f"stage1_{engine.value}.md"


def prompt_filename_for(engine: Engine) -> str:
    """Where Stage 0 writes the prompt the user pastes into the engine."""

    return f"prompt_{engine.value}.md"


def build_prompt(*, engine: Engine, ctx: PromptContext) -> str:
    """Render the Stage 1 prompt body for one engine.

    The body is identical between engines except for the opening identity
    line — both produce mechanism candidates AND code-feasibility review in
    the same markdown file. This is intentional: parallel冗余 with two
    different models, combined by the web GPT in Stage 2.
    """

    engine_label = {
        Engine.CLAUDE: "Claude Code",
        Engine.CODEX: "Codex GUI",
    }[engine]

    if ctx.lab is Lab.SINGLE_FACTOR:
        review_status_choices = (
            "in_contract_factor_def | partial_in_contract | "
            "needs_extension | future_enhancement"
        )
        contract_label = "factor.json"
        spec_field_label = "factor_json_keys_touched"
    else:
        review_status_choices = (
            "in_contract_spec_variant | partial_in_contract | "
            "needs_extension | future_enhancement"
        )
        contract_label = "ModelFactorCaseSpec"
        spec_field_label = "spec_fields_touched"

    lines: list[str] = [
        f"# Stage 1 Prompt — {engine_label} (lab={ctx.lab.value})",
        "",
        f"你是 **{engine_label}**。idea_id: `{ctx.idea_id}`。",
        "",
    ]
    lines.extend(_governance_preamble())
    lines.extend([
        "## 1. Idea",
        ctx.idea,
        "",
        "## 2. 你的任务（generator + reviewer 合一）",
        "本 Stage 1 prompt 在 Claude Code 和 Codex GUI 上**并行**执行同一份任务。",
        "两个引擎互不可见；网页版 GPT 在 Stage 2 综合两份输出取长补短。",
        "",
        "你需要先做 generator，再做 reviewer：",
        "- **Part A (generator)**: 基于下方 vault 卡 + transferable_moves，提出 "
        "3-8 条互补的候选机制。",
        "- **Part B (reviewer)**: 对 Part A 的每条机制，逐条评审在当前 "
        f"v1 {contract_label} schema + validator 硬规则下是否可执行。",
        "",
        "## 3. 工作目录",
        f"- draft_dir：`{ctx.draft_dir}`",
        f"- vault_root：`{ctx.vault_root}`",
        f"- 写入唯一文件：`{ctx.draft_dir / output_filename_for(engine)}`",
        "- 不创建其他文件，不修改代码库，不调用 alpha-lab。",
        "",
        "## 4. Stage 1 纪律",
        "- Part A 和 Part B 都只写在唯一输出文件里，不分多份。",
        "- 候选机制只增不减；不可执行的标 `needs_extension`，不要删除。",
        "- vault 是素材库，不是判决书；`transferable_moves` 是主要生成原料。",
        "- `operative_claims` 只是弱上下文 hint，不作为淘汰条件。",
        "- 机制命名不要直接用 reversal / momentum / value / quality / size / "
        "liquidity 等 canonical 标签——保护假设空间，描述机制本身。",
        "- novel synthesis 合法；找不到来源卡片不是缺口。",
        "- Part B 不要否决任何机制；可执行性边界用 `implementation_status` 表达。",
        "",
        "## 5. 共享 retrieval（两个引擎一字不差相同）",
        "卡片路径相对于 vault_root；按 path 打开 quant-knowledge 原文。",
    ])
    lines.extend(_format_cards(ctx.related_cards))
    lines.extend(_format_insight_brief(ctx.insight_brief))
    lines.extend(
        [
            "",
            "## 6. 代码库索引（两个引擎都看，用于 reviewer 部分）",
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
            "## 7. Schema + Validator 硬约束",
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
            "## 8. 输出格式",
            f"写入 `{ctx.draft_dir / output_filename_for(engine)}`，单一 markdown 含两段：",
            "",
            "```markdown",
            "## Part A — Mechanism candidates (generator)",
            "",
            "mechanism_1:",
            "  hypothesis: \"\"",
            "  inspired_by: []          # optional",
            "  fusion_of: []             # optional [card_a, card_b]",
            "  novel_delta: \"\"",
            "  signal_sketch: \"\"",
            "  data_needs: []",
            "  concern: \"\"",
            "... (3-8 条)",
            "",
            "## Part B — Code feasibility review (reviewer)",
            "",
            "mechanism_1:",
            "  in_v1_contract: true | false",
            "  required_columns_present: []",
            "  required_columns_missing: []",
            f"  {spec_field_label}: []",
            "  validator_blockers: []",
            f"  implementation_status: \"{review_status_choices}\"",
            "  reviewer_note: \"\"",
            "... (与 Part A 一一对应)",
            "```",
            "",
            "## 9. 禁止",
            "- 输出 YAML 文件、final factor code、case spec、ranking、single best idea",
            "- 给机制贴 reversal / momentum 等 canonical 标签",
            "- 否决 / 删除任何机制（不可执行的标 `needs_extension`）",
            "- 改写第 6/7 节列出的 schema 或 validator 规则",
            "- 引入 portfolio construction、execution / replay / fill simulation 语义",
            "",
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
    lines.append("以下只作为候选生成素材；不是约束或 keep/kill 规则。")
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


# ---------------------------------------------------------------------------
# Governance preamble (in-prompt persistent constraints)
# ---------------------------------------------------------------------------
#
# Codex GUI 桌面版的 Project 没有持久化 Instructions 槽位 —— 每次新会话只能
# 靠粘贴的内容承载约束。把 governance 内联进 prompt 体内，使 Stage 1 不依赖
# 任何"外部 system prompt 配置"，提升跨引擎 / 跨会话的稳定性。
#
# 内容浓缩自：AGENTS.md / CLAUDE.md / docs/research_workflow.md /
# docs/end_to_end_workflow.md。冲突时以原始文档为准。


def _governance_preamble() -> list[str]:
    """Return the in-prompt governance section (lab-agnostic, engine-agnostic)."""

    return [
        "## 0. Governance（Stage 1 协议硬约束浓缩）",
        "",
        "完整规则见仓库 `AGENTS.md` / `CLAUDE.md` / `docs/research_workflow.md` /"
        " `docs/end_to_end_workflow.md`。冲突时以原始文档为准。",
        "",
        "### 0.1 你的位置",
        "- 当前是 Stage 1（generator + reviewer 合一）；不是 Stage 0/2/3/4/5。",
        "- 唯一允许写入的文件：`<draft_dir>/stage1_<engine>.md`（路径见 §3）。",
        "",
        "### 0.2 禁止越界",
        "- 不输出 final factor code / `factor.json` / `case_spec_payload` /"
        " `model_candidate_payload` / ranking / single best idea —— Stage 2 网页"
        " GPT 的事。",
        "- 不调用 `alpha-lab` 任何子命令；不写 validator / pipeline 指令 ——"
        " Stage 3 后端的事。",
        "- 不做 promotion 判断 / 不引用 `factor_promotion_checklist.md` ——"
        " Stage 4/5 的事。",
        "- 不读对方引擎的输出；不读 `docs/templates/<lab>_stage{2_*,3_*}_*` /"
        " `backend_draft_*workflow.md` / `factor_promotion_checklist.md`。",
        "- 不引入 portfolio construction / execution / replay / fill simulation /"
        " live trading 语义（Level 3 永久禁止）。",
        "- 不修改仓库代码；不创建散落 .py / .ipynb / .yaml；不写 `<draft_dir>`"
        " 外路径。",
        "",
        "### 0.3 生成纪律（vault = 素材库不是判决书）",
        "- 候选机制只增不减；不可执行的标 `needs_extension`，**不删除**。",
        "- 探索阶段**不做 KILL / HOLD** —— 真正的 KILL 由 Stage 3 数据决定，"
        "不在这里下判决。",
        '- `transferable_moves` 是主要生成原料；`operative_claims` 是弱'
        ' hint，**不触发"先例 kill"**（知识库可能有错，硬 kill 会误杀活机制）。',
        "- 找不到来源卡 ≠ 缺口；novel synthesis 合法，不强制 lineage。",
        "- 机制命名禁用 canonical labels（reversal / momentum / value / quality"
        " / size / skewness / liquidity / volatility / amplitude / crowding"
        " 等）—— 保护假设空间，描述机制本身。",
        "",
        "### 0.4 PIT / future-leakage 硬约束（Part B 评审必须对照）",
        "- factor / feature value at `t` 只用 `≤ t` 的信息。",
        "- 禁 `shift(-n)`、负向 `pct_change`、`future_return / forward_return /"
        " next_return / label_return / target_return` 等 future-leakage tokens。",
        "- 禁全样本均值 / 标准差作为标准化（横截面 demean / rank 例外）。",
        "- rolling / expanding 必须按 `asset` 分组。",
        "- `required_columns` / `feature_columns` 必须是真实已注册列名；编造列名"
        " = `needs_extension`。",
        "",
        "### 0.5 输出（与 §3 / §8 一致）",
        "- 唯一输出：`<draft_dir>/stage1_<engine>.md`，单一 Markdown 含 Part A"
        " + Part B；不另起 YAML / 代码文件。",
        "- 内联 YAML code block 是 Markdown 内嵌格式的一部分，**不是**独立文件。",
        "",
        "---",
        "",
    ]
