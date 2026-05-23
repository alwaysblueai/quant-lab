from __future__ import annotations

import datetime as dt
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from alpha_lab.vault_export import ExportResult

# ``alpha_lab.reporting.factor_correlation`` / ``factor_decomposition`` pull
# in ``alpha_lab.reporting/__init__.py`` → ``experiment`` → ``evaluation`` →
# scipy.stats. To keep CLI cold-start fast (--help in 2s instead of 4s), we
# defer those imports to the call sites that actually need them, below.

if TYPE_CHECKING:
    from alpha_lab.research_bridge.graph_view import VaultGraph
    from alpha_lab.research_bridge.models import ProjectConfig


@dataclass(frozen=True, slots=True)
class GraphFeedbackResult:
    tested_in_updates: list[str]
    exploration_updated: bool
    failure_entry_id: str
    suggested_similar_to: list[str]
    correlation_summary: str

    def to_payload(self) -> dict[str, object]:
        return {
            "tested_in_updates": list(self.tested_in_updates),
            "exploration_updated": self.exploration_updated,
            "failure_entry_id": self.failure_entry_id,
            "suggested_similar_to": list(self.suggested_similar_to),
            "correlation_summary": self.correlation_summary,
        }


@dataclass(frozen=True, slots=True)
class GraphFeedbackSummary:
    """Normalized graph feedback payload for summarize/apply paths."""

    suggested_similar_to: list[str]
    correlation_summary: str


def collect_graph_feedback_summary(
    *,
    vault_root: Path,
    run_root: Path,
    project: ProjectConfig | None = None,
    include_embeddings: bool = False,
    max_items: int = 5,
) -> GraphFeedbackSummary:
    """Collect and render graph feedback candidates.

    This is shared by both summarize and apply-writeback paths so their
    suggestion behavior stays consistent.
    """
    from alpha_lab.reporting.factor_decomposition import inspect_run_decomposition

    decomposition = inspect_run_decomposition(run_root)
    raw_correlation_summary = collect_correlation_summary(run_root)
    suggested_sources: dict[str, str] = {}
    suggested_similar_to: list[str] = []

    if decomposition is not None and decomposition.redundant and decomposition.top_match:
        suggested_similar_to.append(decomposition.top_match)
        suggested_sources[decomposition.top_match] = "decomposition"

    if include_embeddings and project is not None:
        embeddings = _load_embeddings_optional(vault_root)
        if embeddings is not None:
            for raw_card in project.origin_cards:
                card_name = Path(raw_card).stem.split(" - ", 1)[-1].strip()
                if not card_name:
                    continue
                try:
                    candidates = embeddings.suggest_similar(card_name, threshold=0.82)
                except Exception:
                    continue
                for candidate in candidates:
                    normalized = str(candidate).strip()
                    if not normalized or normalized in suggested_similar_to:
                        continue
                    suggested_similar_to.append(normalized)
                    suggested_sources[normalized] = "embedding"

    correlation_summary = _render_correlation_summary(
        suggested_similar_to=suggested_similar_to,
        correlation_items=raw_correlation_summary,
        suggestion_sources=suggested_sources,
        max_items=max_items,
    )

    return GraphFeedbackSummary(
        suggested_similar_to=suggested_similar_to,
        correlation_summary=correlation_summary,
    )


def apply_graph_feedback(
    *,
    vault_root: Path,
    project: ProjectConfig,
    draft_frontmatter: dict[str, Any],
    export_result: ExportResult,
) -> GraphFeedbackResult:
    from alpha_lab.research_bridge.exploration import ExplorationMap
    from alpha_lab.research_bridge.graph_view import VaultGraph

    case_name = str(draft_frontmatter.get("case_name") or "").strip()
    verdict_status = str(draft_frontmatter.get("verdict_status") or "").strip().lower()
    one_sentence = str(draft_frontmatter.get("one_sentence_verdict") or "").strip()
    run_root = Path(str(draft_frontmatter.get("run_root") or "")).expanduser()
    if not run_root.is_absolute():
        run_root = run_root.resolve()

    # tested_in update scope (finalised policy):
    #   - origin_cards: ALWAYS updated — these are the primary factor/method
    #     cards that the experiment directly tests.
    #   - supporting_cards: NOT updated — these provide background context
    #     (concepts, papers) and are not themselves "tested" by the experiment.
    #   - related_experiment_cards: NOT updated — these are pre-existing
    #     experiment cards referenced for comparison, not tested again.
    #   - failure_cards: NOT updated — same reasoning as supporting_cards.
    tested_in_updates: list[str] = []
    for raw_card in project.origin_cards:
        card_path = (vault_root / raw_card).resolve()
        if not card_path.exists():
            continue
        if _upsert_tested_in(card_path=card_path, experiment_name=case_name):
            tested_in_updates.append(raw_card)

    graph = VaultGraph.from_vault_root(vault_root)
    try:
        graph.build(vault_root=vault_root)
    except Exception:
        graph = None  # type: ignore[assignment]

    exploration = ExplorationMap.from_vault_root(vault_root)
    try:
        exploration.build(vault_root=vault_root)
    except Exception:
        exploration = None  # type: ignore[assignment]

    exploration_updated = False
    if exploration is not None:
        mechanism, factor_family, data_types = _infer_region(
            project=project,
            graph=graph,
        )
        exploration.record_exploration(
            hypothesis=str(draft_frontmatter.get("current_hypothesis") or case_name),
            mechanism=mechanism,
            factor_family=factor_family,
            data_types=data_types,
            project_slug=project.slug,
            experiment_name=case_name,
            best_verdict=one_sentence or verdict_status,
            exhaustion_level=_infer_exhaustion_level(verdict_status, one_sentence),
            last_explored=dt.date.today().isoformat(),
        )
        exploration_updated = True

    failure_entry_id = ""
    if _should_capture_failure(verdict_status, one_sentence):
        failure_entry_id = _append_failure_knowledge(
            registry_path=vault_root / "90_moc" / "Registry - Failure Knowledge.md",
            project=project,
            case_name=case_name,
            verdict_status=verdict_status,
            one_sentence_verdict=one_sentence,
            exported_targets=tuple(export_result.target_paths),
        )

    feedback = collect_graph_feedback_summary(
        vault_root=vault_root,
        run_root=run_root,
        project=project,
        include_embeddings=True,
        max_items=5,
    )

    _run_rebuild_script(vault_root, "rebuild-graph.py")
    _run_rebuild_script(vault_root, "rebuild-exploration-map.py")
    return GraphFeedbackResult(
        tested_in_updates=tested_in_updates,
        exploration_updated=exploration_updated,
        failure_entry_id=failure_entry_id,
        suggested_similar_to=feedback.suggested_similar_to,
        correlation_summary=feedback.correlation_summary,
    )


def collect_correlation_summary(
    run_root: str | Path,
    *,
    limit: int = 3,
) -> list[dict[str, object]]:
    from alpha_lab.reporting.factor_correlation import collect_run_factor_correlation_summary

    matches = collect_run_factor_correlation_summary(run_root, limit=limit)
    return [item.to_payload() for item in matches]


def _load_embeddings_optional(vault_root: Path) -> Any | None:
    try:
        from alpha_lab.research_bridge.embeddings import VaultEmbeddings

        embeddings = VaultEmbeddings.from_vault_root(vault_root)
        embeddings.build(vault_root=vault_root)
        return embeddings
    except Exception:
        return None


def _render_correlation_summary(
    *,
    suggested_similar_to: list[str],
    correlation_items: list[dict[str, object]],
    suggestion_sources: dict[str, str],
    max_items: int = 5,
) -> str:
    parts: list[str] = []
    for name in suggested_similar_to[:max_items]:
        source = suggestion_sources.get(name, "embedding").strip() or "embedding"
        parts.append(f"- {name} (via {source})")
    if not parts:
        for row in correlation_items[:max_items]:
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            source = str(row.get("source") or "factor_correlation").strip() or "factor_correlation"
            parts.append(f"- {name} (via {source})")
    if not parts:
        return ""
    return "Similar factor suggestions:\n" + "\n".join(parts)


def _upsert_tested_in(*, card_path: Path, experiment_name: str) -> bool:
    text = card_path.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return False
    end = text.find("\n---", 4)
    if end == -1:
        return False
    frontmatter = text[4:end]
    body = text[end + 5 :]
    lines = frontmatter.splitlines()
    new_lines: list[str] = []
    changed = False
    found = False
    collecting = False
    current_items: list[str] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if collecting and line.startswith("  - "):
            current_items.append(line[4:].strip())
            continue
        if collecting:
            merged = sorted(set(current_items + [experiment_name]))
            new_lines.append("tested_in:")
            new_lines.extend([f"  - {item}" for item in merged])
            collecting = False
            changed = merged != current_items
        if stripped.startswith("tested_in:"):
            found = True
            suffix = stripped.split(":", 1)[1].strip()
            if suffix.startswith("[") and suffix.endswith("]"):
                raw_items = [
                    part.strip().strip("'\"") for part in suffix[1:-1].split(",") if part.strip()
                ]
                merged = sorted(set(raw_items + [experiment_name]))
                new_lines.append("tested_in:")
                new_lines.extend([f"  - {item}" for item in merged])
                changed = merged != raw_items
                continue
            current_items = []
            collecting = True
            continue
        new_lines.append(line)
        if idx == len(lines) - 1 and collecting:
            merged = sorted(set(current_items + [experiment_name]))
            new_lines.append("tested_in:")
            new_lines.extend([f"  - {item}" for item in merged])
            collecting = False
            changed = merged != current_items
    if collecting:
        merged = sorted(set(current_items + [experiment_name]))
        new_lines.append("tested_in:")
        new_lines.extend([f"  - {item}" for item in merged])
        changed = merged != current_items
    if not found:
        inserted = False
        final_lines: list[str] = []
        for line in new_lines:
            final_lines.append(line)
            if line.startswith("lifecycle:") and not inserted:
                final_lines.append("tested_in:")
                final_lines.append(f"  - {experiment_name}")
                inserted = True
                changed = True
        if not inserted:
            final_lines.extend(["tested_in:", f"  - {experiment_name}"])
            changed = True
        new_lines = final_lines
    if not changed:
        return False
    rendered = "---\n" + "\n".join(new_lines).rstrip() + "\n---" + body
    card_path.write_text(rendered, encoding="utf-8")
    return True


def _infer_region(
    *,
    project: ProjectConfig,
    graph: VaultGraph | None,
) -> tuple[str, str, list[str]]:
    mechanism = ""
    factor_family = ""
    data_types: list[str] = []
    if graph is None:
        return mechanism, factor_family, data_types
    for raw_card in project.origin_cards:
        guess = Path(raw_card).stem.split(" - ", 1)[-1].strip()
        node = graph.get_node(guess)
        if node is None:
            continue
        mechanism = mechanism or node.mechanism
        factor_family = factor_family or node.factor_family
        for item in graph.get_neighbors(node.name, "uses_data"):
            if item not in data_types:
                data_types.append(item)
    return mechanism, factor_family, data_types


def _infer_exhaustion_level(verdict_status: str, one_sentence_verdict: str) -> str:
    """Map verdict text to an exploration exhaustion level.

    Policy (finalised):
      exhausted — the hypothesis direction is decisively negative and should
                  not be revisited without a structurally different angle.
                  Trigger: explicit drop/fail/reject language.
      moderate  — the direction showed mixed or fragile results; further work
                  is possible but must bring stronger evidence or a design
                  change (e.g. different neutralisation, different universe).
                  Trigger: revise/fragile/mixed/unstable/weak/inconclusive.
      light     — the direction produced encouraging or neutral-positive
                  results; the hypothesis space is still open.
                  Trigger: promising/keep/promote, or no negative signal.
      none      — the verdict is too vague to classify (empty or
                  unrecognised tokens).  Downstream should treat this the
                  same as ``light`` but flag for human review.
    """
    merged = f"{verdict_status} {one_sentence_verdict}".lower()
    if not merged.strip():
        return "none"
    if any(tok in merged for tok in ("drop", "fail", "reject", "rejected", "abandon")):
        return "exhausted"
    if any(
        tok in merged
        for tok in (
            "revise",
            "fragile",
            "mixed",
            "unstable",
            "weak",
            "inconclusive",
        )
    ):
        return "moderate"
    if any(tok in merged for tok in ("promising", "keep", "promote", "pass", "validated")):
        return "light"
    return "none"


def _should_capture_failure(verdict_status: str, one_sentence_verdict: str) -> bool:
    """Decide whether a writeback should auto-create a Failure Knowledge entry.

    Policy (finalised):
      Create an FK entry when the verdict unambiguously indicates the
      hypothesis direction failed — not merely underperformed or was
      inconclusive.

      Positive triggers  : drop, fail, reject, rejected, abandon
      Negative triggers  : revise, mixed, fragile (these are "moderate",
                           not hard failures — worth logging in
                           exploration_map but not in FK-registry)

      Additional guard: the draft must contain a non-empty
      ``one_sentence_verdict`` so the FK entry has a meaningful
      ``failure_statement``.  Verdicts without a sentence are too vague
      to produce a reusable lesson.
    """
    if not one_sentence_verdict.strip():
        return False
    merged = f"{verdict_status} {one_sentence_verdict}".lower()
    return any(tok in merged for tok in ("drop", "fail", "reject", "rejected", "abandon"))


def _append_failure_knowledge(
    *,
    registry_path: Path,
    project: ProjectConfig,
    case_name: str,
    verdict_status: str,
    one_sentence_verdict: str,
    exported_targets: tuple[str, ...],
) -> str:
    if not registry_path.exists():
        return ""
    text = registry_path.read_text(encoding="utf-8")
    title = f"{project.slug} / {case_name} failure pattern"
    if title in text:
        return ""
    existing_ids = [int(match.group(1)) for match in re.finditer(r"\[FK-(\d+)\]", text)]
    next_id = max(existing_ids, default=0) + 1
    failure_id = f"FK-{next_id:03d}"
    failure_class = _infer_failure_class(one_sentence_verdict, verdict_status)
    block = "\n".join(
        [
            f"### [{failure_id}] {title}",
            "",
            "- `status`: `watch`",
            f"- `failure_class`: `{failure_class}`",
            "- `origin`:",
            *[f"  - `{path}`" for path in exported_targets[:3]],
            "- `failure_statement`:",
            f"  - {one_sentence_verdict or verdict_status or '实验结果未通过基本稳健性要求。'}",
            "- `boundary`:",
            f"  - {project.market} / {project.frequency} / {project.category} project context",
            "- `reusable_lesson`:",
            "  - 同机制同家族的新假设在进入下一轮前，需要更强的增量证据或更明确的结构差异。",
            "- `prevention_rule`:",
            "  - 后续相近方向进入 scaffold-case 前，必须先复核 novelty、"
            "dependency、PIT 与最近失败记录。",
            "",
        ]
    )
    marker = "## 三、Retired 条目"
    if marker in text:
        text = text.replace(marker, block + marker)
    else:
        text = text.rstrip() + "\n\n" + block
    registry_path.write_text(text, encoding="utf-8")
    return failure_id


def _infer_failure_class(one_sentence_verdict: str, verdict_status: str) -> str:
    merged = f"{one_sentence_verdict} {verdict_status}".lower()
    if any(token in merged for token in ("pit", "future", "restated", "revision", "data")):
        return "data-contract-failure"
    if any(token in merged for token in ("execution", "tradability", "slippage", "auction")):
        return "execution-nonrealism"
    if any(token in merged for token in ("capacity", "liquidity", "turnover", "impact")):
        return "capacity-liquidity-failure"
    if any(token in merged for token in ("regime", "unstable", "fragile")):
        return "regime-instability"
    if any(token in merged for token in ("validation", "benchmark", "multiple testing")):
        return "validation-design-failure"
    return "signal-invalidity"


def _run_rebuild_script(vault_root: Path, script_name: str) -> None:
    script_path = vault_root / "00_protocols" / script_name
    if not script_path.exists():
        return
    subprocess.run(
        [sys.executable, str(script_path), str(vault_root)],
        capture_output=True,
        text=True,
        check=False,
    )
