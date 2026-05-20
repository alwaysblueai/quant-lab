"""Stage 4 experiment_card scaffolding + cleanup (per docs/end_to_end_workflow.md).

After multi-round iteration on an idea ends (regardless of outcome), the user
writes a single ``experiment_card.md`` summarising the run, then exports it to
the vault. This module:

1. Scaffolds the card with a frontmatter skeleton + section headings populated
   from ``manifest.json`` + ``research_log.md``. The user fills the
   summary/analysis prose by hand.
2. Optionally cleans up Stage 0/1/2/3 temp files under ``ideas/<idea_id>/``,
   keeping only ``manifest.json`` + ``experiment_card.md``.

Designed to be call-once per idea_id. ``scaffold_experiment_card`` is
idempotent: re-running over an existing card refuses to overwrite (raises
``FileExistsError``).
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ExperimentCardOutcome(StrEnum):
    """Stage 4 outcome label."""

    PROMOTED = "promoted"
    KILLED = "killed"
    PARKED = "parked"


_TEMP_FILES_GLOB: tuple[str, ...] = (
    "prompt_*.md",
    "stage1_*.md",
    "stage2_payload_v*.json",
    "stage2_input.md",
    "stage1_reconcile.yaml",
    "retrieval_pack.md",
    # pre-2026-05-11 temporary layout cleanup compatibility:
    "dispatch.*.md",
    "ledger_v1.*.yaml",
    "ledger_v1.yaml",
    "reconcile.md",
    "retrieval_log.md",
    "prompt.shared.md",
)

_KEEP_FILES: frozenset[str] = frozenset({"manifest.json", "experiment_card.md"})


@dataclass(frozen=True)
class ScaffoldedCard:
    """Result of ``scaffold_experiment_card`` (returned for tests; CLI prints path)."""

    card_path: Path
    cleaned: tuple[Path, ...]


def scaffold_experiment_card(
    *,
    idea_id: str,
    outcome: ExperimentCardOutcome,
    workspace_root: str | Path = ".",
    cleanup: bool = False,
) -> Path:
    """Write ``ideas/<idea_id>/experiment_card.md`` and optionally clean up.

    Raises:
        FileNotFoundError: ``ideas/<idea_id>/`` does not exist.
        FileExistsError: ``experiment_card.md`` already exists for this idea.
    """

    root = Path(workspace_root).expanduser().resolve()
    idea_dir = root / "ideas" / idea_id
    if not idea_dir.exists() or not idea_dir.is_dir():
        raise FileNotFoundError(f"ideas/<idea_id>/ not found: {idea_dir}")

    card_path = idea_dir / "experiment_card.md"
    if card_path.exists():
        raise FileExistsError(f"experiment_card.md already exists: {card_path}")

    manifest_path = idea_dir / "manifest.json"
    manifest: dict[str, object] = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            manifest = {}

    research_log_excerpt = _collect_research_log_excerpt(root, idea_id, manifest)

    card_text = _render_card(
        idea_id=idea_id,
        outcome=outcome,
        manifest=manifest,
        research_log_excerpt=research_log_excerpt,
    )
    card_path.write_text(card_text, encoding="utf-8")

    if cleanup:
        _cleanup_temp_files(idea_dir)

    return card_path


def _cleanup_temp_files(idea_dir: Path) -> list[Path]:
    """Delete Stage 0/1/2/3 temp files; keep manifest + experiment_card."""

    removed: list[Path] = []
    for pattern in _TEMP_FILES_GLOB:
        for path in sorted(idea_dir.glob(pattern)):
            if path.name in _KEEP_FILES:
                continue
            if path.is_file():
                path.unlink()
                removed.append(path)
    return removed


def _collect_research_log_excerpt(
    workspace_root: Path, idea_id: str, manifest: dict[str, object]
) -> str:
    """Try to locate the research_log.md tied to this idea and grab the last rounds.

    Lookup priority:
    1. ``custom_factors/research/<name>/research_log.md`` where <name> matches
       any factor mentioned in the manifest's codebase_snapshot.factors.research
       (best-effort heuristic).
    2. ``custom_models/research/<name>/research_log.md`` likewise.

    Returns an empty string if no log can be located; the user will fill it in.
    """

    factor_root = workspace_root / "custom_factors" / "research"
    model_root = workspace_root / "custom_models" / "research"

    candidates: list[Path] = []
    for root_dir in (factor_root, model_root):
        if not root_dir.exists():
            continue
        for child in root_dir.iterdir():
            log = child / "research_log.md"
            if log.exists() and idea_id in log.read_text(encoding="utf-8", errors="ignore"):
                candidates.append(log)

    if not candidates:
        return ""

    # Use the most recently modified log file matching the idea_id.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0].read_text(encoding="utf-8")


def _render_card(
    *,
    idea_id: str,
    outcome: ExperimentCardOutcome,
    manifest: dict[str, object],
    research_log_excerpt: str,
) -> str:
    """Render the experiment_card.md body.

    Frontmatter is populated from manifest + outcome; section headings are
    boilerplate. The user fills in 关键改动轨迹 / 成败路径 / emergent_moves /
    operative_claims by hand.
    """

    lab = str(manifest.get("lab") or "unknown")
    idea_text = str(manifest.get("idea") or "")
    created = str(manifest.get("created_at_utc") or "")
    engines = manifest.get("engines") or manifest.get("models") or []
    if isinstance(engines, list):
        engines_text = ", ".join(str(e) for e in engines)
    else:
        engines_text = str(engines)
    now = dt.datetime.now(dt.UTC).isoformat()

    log_block = (
        research_log_excerpt.strip()
        or "<!-- research_log.md not located; paste rounds manually -->"
    )

    return "\n".join(
        [
            "---",
            f"idea_id: {idea_id}",
            f"lab: {lab}",
            f"outcome: {outcome.value}",
            f"idea_distributed_at: {created}",
            f"experiment_card_written_at: {now}",
            f"engines: [{engines_text}]",
            "final_artifact_path: <fill in: custom_factors/... or custom_models/... or "
            "promoted/...>",
            "final_metrics_sha256: <fill in: run_manifest 中的关键 hash 摘要>",
            "---",
            "",
            f"# Experiment Card — {idea_text or idea_id}",
            "",
            "## 关键改动轨迹",
            "<!-- 列出 v1 → vN 每轮做了什么 + 为什么 -->",
            "",
            "- v1: ",
            "- v2: ",
            "- ...",
            "",
            "## 成功路径分析（如果 outcome=promoted）",
            "<!-- 哪条机制被数据验证；哪些假设被支持/部分修正 -->",
            "",
            "## 失败路径分析（如果 outcome=killed / parked）",
            "<!--",
            "  哪条 assumption 在数据上被否定",
            "  失败类型：PIT / alias / regime fragility / cost-aware / 训练样本不足 / 其他",
            "  是死路（killed）还是 future_enhancement（parked，等下次条件成熟）",
            "-->",
            "",
            "## emergent_moves（主回写字段）",
            "<!-- 本次实验浮现的、可被未来 idea 借用的新动作（具体可执行） -->",
            "",
            "- ",
            "",
            "## operative_claims（弱观察）",
            "<!-- 观察到的现象 / 经验 / 边界条件，错了无所谓 -->",
            "",
            "- ",
            "",
            "## research_log 摘要（自动注入，供撰写时参考）",
            "",
            "```",
            log_block,
            "```",
            "",
        ]
    )
