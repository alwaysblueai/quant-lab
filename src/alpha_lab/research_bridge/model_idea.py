from __future__ import annotations

# ruff: noqa: E501
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

from alpha_lab.baseline_factor_suite import baseline_factor_suite_payload
from alpha_lab.config import PROJECT_ROOT
from alpha_lab.custom_factors import iter_custom_factor_meta_paths, read_custom_factor_source
from alpha_lab.factor_recipe import factor_registry
from alpha_lab.model_factor import list_model_contracts
from alpha_lab.real_cases.model_factor.spec import load_model_factor_case_spec
from alpha_lab.real_cases.single_factor.spec import load_single_factor_case_spec
from alpha_lab.research_bridge.embeddings import SearchResult, VaultEmbeddings
from alpha_lab.research_bridge.scoring import (
    MECHANISM_DISCOVERY,
    MODEL_MODE_WEIGHTS,
    SIGNAL_MAPPING,
    VALIDATION_KILL_TESTS,
    CardMetadata,
    QueryAnchor,
    ScoreComponents,
    ScoreWeights,
    aggregate_score,
    derive_failure_keywords,
    infer_available_data_from_frequency,
    is_stage3_llm_helper_stage,
    normalize_data_set,
    normalize_workflow_stage,
    recommend_next_stage,
    score_card,
)
from alpha_lab.vault_export import resolve_vault_root

from . import loaders as bridge_loaders
from . import mechanism_index
from .llm_rerank import (
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MODEL,
    CategorizeOutcome,
    RerankCandidate,
    RerankOutcome,
    categorize_and_compress,
    rerank_candidates,
)
from .output_lint import (
    MODEL_SIGNAL_RISK_CONTROLS,
    MODEL_VALIDATION_ALIAS_TARGETS,
    LintReport,
    describe_model_lint_contract,
    extract_model_stage_sections,
    lint_model_idea_response,
)
from .query_expansion import ExpansionOutcome, expand_query

_SUPPORTED_MODES: frozenset[str] = frozenset({"start", "explore", "constrained"})
_MODEL_SPEC_CONFIG_DIR = PROJECT_ROOT / "configs" / "real_cases" / "model_factor"
_MODEL_IDEA_SESSIONS_REL_PATH = Path("artifacts") / "model_lab_explorer" / "sessions"
_SOURCE_ANCHORS: tuple[dict[str, str], ...] = (
    {
        "path": "src/alpha_lab/model_factor/core.py",
        "focus": "ModelFamily/TrainingSpec/ModelSelectionSpec and estimator behavior",
    },
    {
        "path": "src/alpha_lab/real_cases/model_factor/spec.py",
        "focus": "spec contracts, field constraints, feature availability rules",
    },
    {
        "path": "src/alpha_lab/real_cases/model_factor/pipeline.py",
        "focus": "end-to-end run stages, integrity checks, artifact generation",
    },
    {
        "path": "src/alpha_lab/real_cases/model_factor/artifacts.py",
        "focus": "metrics/manifest export schema and audit payload layout",
    },
)
_UNSUPPORTED_MODEL_HINTS: tuple[str, ...] = (
    "tabnet",
    "transformer",
    "lstm",
    "gru",
    "catboost",
)
_MODEL_ALIAS_MAP: tuple[tuple[str, str], ...] = (
    ("xgboost", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("lgbm", "lightgbm"),
    ("gbdt", "gbdt"),
    ("tree", "gbdt"),
    ("elastic_net", "elastic_net"),
    ("elastic", "elastic_net"),
    ("lasso", "lasso"),
    ("ridge", "ridge"),
    ("mlp", "mlp"),
    ("linear", "linear"),
)
_RECOMMENDATION_PRIORITY_ORDER: tuple[str, ...] = (
    "contracts",
    "failures",
    "validated",
    "knowledge",
    "sources",
)
_MODEL_PREVIOUS_STAGE: dict[str, str] = {
    SIGNAL_MAPPING: MECHANISM_DISCOVERY,
}
_MODEL_UPSTREAM_SECTION_PRIORITY: dict[str, tuple[str, ...]] = {
    SIGNAL_MAPPING: (
        "模型机制候选",
        "实现假设草图",
        "与当前 spec / baseline 的关系",
        "不确定性与失败路径",
    ),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="alpha-lab model-idea",
        description=(
            "Model-lab idea explorer for prompt generation. "
            "Batch 3 adds session/memory context for iterative exploration."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    commands = parser.add_subparsers(dest="action", required=True)

    explore = commands.add_parser(
        "explore",
        help="Generate a model-research prompt from local context.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    explore.add_argument(
        "--idea",
        required=True,
        help="Natural-language model idea to explore.",
    )
    explore.add_argument(
        "--mode",
        default="explore",
        choices=sorted(_SUPPORTED_MODES),
        help=(
            "Prompt mode. 'start' kicks off cross-domain brainstorming "
            "(contracts shown as discussion-only); 'explore' is the default "
            "structured expansion; 'constrained' enforces system-contract "
            "constraints and emits a candidate spec patch hint."
        ),
    )
    explore.add_argument(
        "--spec",
        default=None,
        help=(
            "Optional model-factor spec path. Relative names are resolved against "
            "configs/real_cases/model_factor/."
        ),
    )
    explore.add_argument(
        "--workspace-root",
        default=".",
        help="Workspace root used for scanning model-lab run artifacts under dist/.",
    )
    explore.add_argument(
        "--vault-root",
        default=None,
        help="Optional quant-knowledge vault root. Defaults to OBSIDIAN_VAULT_PATH.",
    )
    explore.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Top-K context rows to include for knowledge and run matching.",
    )
    explore.add_argument(
        "--memory-limit",
        type=int,
        default=3,
        help="Max prior idea sessions injected as memory context.",
    )
    explore.add_argument(
        "--stage",
        default=None,
        choices=["mechanism_discovery", "signal_mapping"],
        help=(
            "Workflow stage. mechanism_discovery finds model mechanisms; "
            "signal_mapping maps them to spec/run variants. Stage 3 data validation "
            "is handled by the structured kill workflow, not this explorer."
        ),
    )
    explore.add_argument(
        "--inject-recent-drift",
        action="store_true",
        help="Inject recent lint violations and upstream stage artifacts into the prompt.",
    )
    explore.add_argument(
        "--parent-session-id",
        default=None,
        help="Optional upstream model-idea session id used for cross-stage chaining.",
    )
    explore.add_argument(
        "--save-session",
        action="store_true",
        help="Persist this exploration under artifacts/model_lab_explorer/sessions/.",
    )
    explore.add_argument(
        "--session-id",
        default=None,
        help="Optional explicit session id when --save-session is enabled.",
    )

    record = commands.add_parser(
        "record-response",
        help="Lint and persist an externally produced model-idea response.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    record.add_argument("--session-id", required=True, help="Session id to update.")
    response_group = record.add_mutually_exclusive_group(required=True)
    response_group.add_argument(
        "--response-text",
        default=None,
        help="Raw LLM response text to lint and persist.",
    )
    response_group.add_argument(
        "--response-file",
        default=None,
        help="UTF-8 file containing the LLM response text.",
    )
    record.add_argument(
        "--workspace-root",
        default=".",
        help="Workspace root containing artifacts/model_lab_explorer/sessions/.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "record-response":
        response_text = str(args.response_text or "")
        if args.response_file:
            try:
                response_text = Path(str(args.response_file)).read_text(encoding="utf-8")
            except OSError as exc:
                parser.error(f"unable to read response file: {exc}")
        try:
            report = record_model_idea_response(
                session_id=str(args.session_id),
                response_text=response_text,
                workspace_root=str(args.workspace_root),
            )
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))
        print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0

    if args.action != "explore":
        parser.error(f"unsupported action: {args.action!r}")

    try:
        payload = explore_model_idea(
            idea=str(args.idea),
            mode=str(args.mode),
            spec=str(args.spec) if args.spec else None,
            workspace_root=str(args.workspace_root),
            vault_root=str(args.vault_root) if args.vault_root else None,
            top_k=int(args.top_k),
            memory_limit=int(args.memory_limit),
            stage=str(args.stage) if args.stage else None,
            inject_recent_drift=bool(args.inject_recent_drift),
            parent_session_id=(
                str(args.parent_session_id) if args.parent_session_id else None
            ),
        )
        if bool(args.save_session):
            payload["session"] = save_model_idea_session(
                payload=payload,
                workspace_root=str(args.workspace_root),
                session_id=str(args.session_id) if args.session_id else None,
                parent_session_id=(
                    str(args.parent_session_id) if args.parent_session_id else None
                ),
            )
    except ValueError as exc:
        parser.error(str(exc))

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def explore_model_idea(
    *,
    idea: str,
    mode: str = "explore",
    spec: str | None = None,
    workspace_root: str | Path = PROJECT_ROOT,
    vault_root: str | Path | None = None,
    top_k: int = 6,
    memory_limit: int = 3,
    available_data: frozenset[str] | list[str] | None = None,
    stage: str | None = None,
    inject_recent_drift: bool = False,
    parent_session_id: str | None = None,
) -> dict[str, object]:
    normalized_idea = str(idea or "").strip()
    if not normalized_idea:
        raise ValueError("idea must be non-empty")
    normalized_mode = str(mode or "").strip().lower()
    if normalized_mode not in _SUPPORTED_MODES:
        raise ValueError(
            f"unsupported mode: {mode!r}; expected one of {sorted(_SUPPORTED_MODES)}"
        )
    if is_stage3_llm_helper_stage(stage):
        raise ValueError(
            "validation_kill_tests has moved out of model-lab idea exploration. "
            "Use mechanism_discovery/signal_mapping here, then pass structured "
            "artifacts into the Stage 3 data-kill workflow."
        )
    # Stage axis is symmetric with alpha-lab's: a named dispatcher picks
    # the workflow stage first, then mode adjusts strictness inside that stage.
    normalized_stage = normalize_workflow_stage(stage)
    normalized_top_k = max(int(top_k), 1)
    normalized_memory_limit = max(int(memory_limit), 0)
    normalized_parent_session_id = (
        _normalize_session_id(parent_session_id) if parent_session_id else None
    )

    resolved_workspace = Path(workspace_root).expanduser().resolve()
    contracts = list_model_contracts()
    current_spec, spec_warnings = _load_current_spec_summary(spec)
    available_data_set: frozenset[str] | None
    available_data_source = "none"
    if available_data is not None:
        available_data_set = normalize_data_set(available_data)
        available_data_source = "explicit"
    else:
        inferred_frequency = str(current_spec.get("rebalance_frequency") or "").strip()
        available_data_set = infer_available_data_from_frequency(inferred_frequency)
        if available_data_set is not None:
            available_data_source = f"frequency:{inferred_frequency}"
    knowledge_context = _collect_knowledge_context(
        idea=normalized_idea,
        vault_root=vault_root,
        workspace_root=resolved_workspace,
        top_k=normalized_top_k,
        mode=normalized_mode,
        available_data=available_data_set,
    )
    experiment_context = _collect_experiment_context(
        idea=normalized_idea,
        workspace_root=resolved_workspace,
        current_spec=current_spec,
        top_k=normalized_top_k,
    )
    session_memory_context = _collect_session_memory_context(
        workspace_root=resolved_workspace,
        current_idea=normalized_idea,
        limit=normalized_memory_limit,
    )
    warnings = [
        *spec_warnings,
        *_as_str_list(knowledge_context.get("warnings")),
        *_as_str_list(experiment_context.get("warnings")),
        *_as_str_list(session_memory_context.get("warnings")),
    ]
    injected_violations: list[dict[str, object]] = []
    upstream_session: dict[str, object] | None = None
    upstream_header = ""
    drift_header = ""
    if inject_recent_drift:
        injected_violations = list_recent_model_idea_violations(
            workspace_root=resolved_workspace,
            stage=normalized_stage,
            limit=5,
        )
        drift_header = render_model_idea_drift_header(injected_violations)
        upstream_session = find_upstream_model_idea_session(
            workspace_root=resolved_workspace,
            stage=normalized_stage,
            parent_session_id=normalized_parent_session_id,
        )
        if upstream_session is not None:
            upstream_header = render_model_upstream_artifact_header(
                upstream_session,
                current_stage=normalized_stage,
            )
    recommendations = _build_recommendations(
        mode=normalized_mode,
        stage=normalized_stage,
        warnings=warnings,
        knowledge_context=knowledge_context,
        experiment_context=experiment_context,
        session_memory_context=session_memory_context,
    )
    constraint_report: dict[str, object] = {
        "system_contracts": contracts,
        "current_spec": current_spec,
        "validated_baselines": experiment_context["validated_baselines"],
        "recent_failures": experiment_context["recent_failures"],
        "recommendations": recommendations,
    }
    spec_patch_hint = build_spec_patch_hint(
        idea=normalized_idea,
        mode=normalized_mode,
        constraint_report=constraint_report,
    )
    gpt_prompt = _build_model_idea_exploration_prompt(
        idea=normalized_idea,
        mode=normalized_mode,
        report=constraint_report,
        spec_patch_hint=spec_patch_hint,
        stage=normalized_stage,
        drift_header=drift_header,
        upstream_header=upstream_header,
    )
    raw_diagnostics = knowledge_context.get("retrieval_diagnostics")
    diagnostics = dict(raw_diagnostics) if isinstance(raw_diagnostics, dict) else {}
    diagnostics["stage"] = normalized_stage
    diagnostics["recommended_next_stage"] = recommend_next_stage(normalized_stage)
    diagnostics["available_data_provided"] = available_data_set is not None
    diagnostics["available_data_source"] = available_data_source
    diagnostics["inject_recent_drift"] = bool(inject_recent_drift)
    diagnostics["recent_lint_violations_injected"] = len(injected_violations)
    diagnostics["parent_session_id"] = normalized_parent_session_id
    diagnostics["upstream_session_id"] = (
        str(upstream_session.get("session_id") or "") if upstream_session else None
    )
    diagnostics["upstream_sections_injected"] = (
        count_model_upstream_sections(upstream_session, current_stage=normalized_stage)
        if upstream_session
        else 0
    )
    return {
        "idea": normalized_idea,
        "mode": normalized_mode,
        "stage": normalized_stage,
        "constraint_report": constraint_report,
        "spec_patch_hint": spec_patch_hint,
        "gpt_prompt": gpt_prompt,
        "retrieval_diagnostics": diagnostics,
    }


def build_spec_patch_hint(
    *,
    idea: str,
    mode: str,
    constraint_report: dict[str, object],
) -> dict[str, object] | None:
    # Kickoff mode is intentionally pre-formal: do not collapse the
    # idea into a deterministic spec patch — let the LLM brainstorm.
    if mode == "start":
        return None
    current_spec = _as_mapping(constraint_report.get("current_spec"))
    if current_spec.get("status") != "loaded":
        return None

    contracts = _as_mapping(constraint_report.get("system_contracts"))
    supported_families = _as_str_list(contracts.get("supported_model_families"))
    idea_l = idea.lower()

    blocked_by: list[str] = []
    for hint in _UNSUPPORTED_MODEL_HINTS:
        if hint in idea_l:
            blocked_by.append(f"unsupported model family in current contracts: {hint}")

    patch_fields: dict[str, object] = {}
    summary_bits: list[str] = []
    target_family = _infer_target_family(idea_l, supported_families)
    current_family = str(current_spec.get("model_family") or "").strip()
    if target_family and target_family != current_family:
        patch_fields["model"] = {"family": target_family}
        summary_bits.append(f"switch model.family from {current_family} to {target_family}")

    if ("turnover" in idea_l or "换手" in idea_l) and mode == "constrained":
        patch_fields["model_selection"] = {
            "enabled": True,
            "metric": "rank_ic_minus_turnover_penalty",
            "turnover_penalty_lambda": 0.1,
        }
        summary_bits.append("enable turnover-aware model selection")

    if "industry" in idea_l or "行业" in idea_l:
        patch_fields["feature_preprocess"] = {
            "cross_sectional_group_scope": "date_and_industry",
            "industry_group_column": "industry",
        }
        summary_bits.append("use date_and_industry cross-sectional grouping")
    elif "rank" in idea_l:
        patch_fields["feature_preprocess"] = {"cross_sectional_transform": "rank"}
        summary_bits.append("use rank cross-sectional transform")
    elif "winsor" in idea_l:
        patch_fields["feature_preprocess"] = {"cross_sectional_transform": "winsorize_zscore"}
        summary_bits.append("use winsorize_zscore cross-sectional transform")

    if "expanding" in idea_l or "扩展窗口" in idea_l:
        patch_fields["training"] = {"window_type": "expanding"}
        summary_bits.append("switch training.window_type to expanding")
    elif "rolling" in idea_l or "滚动" in idea_l:
        patch_fields["training"] = {"window_type": "rolling"}
        summary_bits.append("keep training.window_type as rolling")

    if not summary_bits:
        summary = "No deterministic spec patch extracted; keep exploratory discussion."
    else:
        summary = "; ".join(summary_bits)

    return {
        "summary": summary,
        "requires_code_change": bool(blocked_by),
        "blocked_by": blocked_by,
        "patch_fields": patch_fields,
    }


def apply_spec_patch_hint(
    spec_payload: dict[str, object],
    patch_hint: dict[str, object],
) -> dict[str, object]:
    patch_fields_raw = patch_hint.get("patch_fields")
    if not isinstance(patch_fields_raw, dict):
        raise ValueError("patch_hint.patch_fields must be an object")
    merged = copy.deepcopy(spec_payload)
    _deep_merge_dict(merged, patch_fields_raw)
    return merged


def _deep_merge_dict(target: dict[str, object], updates: dict[str, object]) -> None:
    for key, value in updates.items():
        key_str = str(key)
        if isinstance(value, dict):
            existing = target.get(key_str)
            if not isinstance(existing, dict):
                target[key_str] = {}
                existing = target[key_str]
            _deep_merge_dict(existing, value)  # type: ignore[arg-type]
        else:
            target[key_str] = value


def _resolve_spec_path(spec: str) -> Path:
    raw = Path(spec).expanduser()
    if raw.exists():
        return raw.resolve()
    if raw.is_absolute():
        raise FileNotFoundError(f"spec file not found: {raw}")
    config_candidate = (_MODEL_SPEC_CONFIG_DIR / raw).resolve()
    if config_candidate.exists():
        return config_candidate
    raise FileNotFoundError(
        f"spec file not found: {raw} (also checked {_MODEL_SPEC_CONFIG_DIR / raw})"
    )


def _load_current_spec_summary(spec: str | None) -> tuple[dict[str, object], list[str]]:
    if spec is None or not str(spec).strip():
        return {
            "status": "not_provided",
            "requested_spec": None,
        }, []

    requested = str(spec).strip()
    try:
        resolved = _resolve_spec_path(requested)
        parsed = load_model_factor_case_spec(resolved)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return {
            "status": "unavailable",
            "requested_spec": requested,
            "resolved_spec_path": None,
        }, [f"spec context unavailable: {exc}"]

    feature_columns = list(parsed.feature_columns)
    model_candidates = list(getattr(parsed.model_selection, "candidates", ()) or ())
    candidate_families = [
        str(getattr(candidate, "family", "") or "").strip()
        for candidate in model_candidates
        if str(getattr(candidate, "family", "") or "").strip()
    ]
    return {
        "status": "loaded",
        "requested_spec": requested,
        "resolved_spec_path": str(resolved),
        "name": parsed.name,
        "factor_name": parsed.factor_name,
        "model_family": parsed.model.family,
        "model_params": dict(getattr(parsed.model, "params", {}) or {}),
        "feature_count": len(parsed.feature_columns),
        "feature_columns_sample": feature_columns[:12],
        "feature_columns_truncated": len(feature_columns) > 12,
        "target_horizon": int(parsed.target.horizon),
        "target_column": str(getattr(parsed.target, "column", "") or ""),
        "direction": str(parsed.direction),
        "rebalance_frequency": str(parsed.rebalance_frequency),
        "n_quantiles": int(parsed.n_quantiles),
        "feature_availability_mode": parsed.feature_availability.mode,
        "feature_availability_column": parsed.feature_availability.column,
        "feature_availability_safety_lag_days": (
            parsed.feature_availability.safety_lag_days
        ),
        "feature_preprocess": {
            "missing_policy": parsed.feature_preprocess.missing_policy,
            "scale_features": parsed.feature_preprocess.scale_features,
            "cross_sectional_transform": parsed.feature_preprocess.cross_sectional_transform,
            "cross_sectional_group_scope": parsed.feature_preprocess.cross_sectional_group_scope,
            "industry_group_column": parsed.feature_preprocess.industry_group_column,
        },
        "training_window_type": parsed.training.window_type,
        "training": {
            "window_type": parsed.training.window_type,
            "train_window_n_dates": parsed.training.train_window_n_dates,
            "min_train_dates": parsed.training.min_train_dates,
            "min_train_rows": parsed.training.min_train_rows,
            "retrain_every_n_dates": parsed.training.retrain_every_n_dates,
            "min_score_assets": parsed.training.min_score_assets,
        },
        "model_selection_enabled": bool(parsed.model_selection.enabled),
        "model_selection_metric": parsed.model_selection.metric,
        "model_selection": {
            "enabled": bool(parsed.model_selection.enabled),
            "metric": parsed.model_selection.metric,
            "candidate_count": len(model_candidates),
            "candidate_families": candidate_families[:8],
            "n_splits": parsed.model_selection.n_splits,
        },
        "feature_importance": {
            "mode": parsed.feature_importance.mode,
            "permutation_max_rows": parsed.feature_importance.permutation_max_rows,
        },
        "neutralization": {
            "enabled": bool(parsed.neutralization.enabled),
            "exposures_path": (
                str(parsed.neutralization.exposures_path)
                if parsed.neutralization.exposures_path is not None
                else None
            ),
        },
        "transaction_cost": {
            "one_way_rate": parsed.transaction_cost.one_way_rate,
        },
    }, []


def _collect_knowledge_context(
    *,
    idea: str,
    vault_root: str | Path | None,
    top_k: int,
    workspace_root: Path | None = None,
    mode: str = "explore",
    available_data: frozenset[str] | None = None,
) -> dict[str, object]:
    resolved_vault, warnings = _resolve_optional_vault_root(vault_root)
    if resolved_vault is None:
        return {
            "status": "not_configured",
            "knowledge_matches": [],
            "knowledge_handling_patterns": [],
            "insight_brief": [],
            "frontier_matches": [],
            "failure_refs": [],
            "warnings": warnings,
            "retrieval_diagnostics": {
                "mode": mode,
                "score_components_by_name": {},
                "dropped_cards": [],
                "score_weights": _score_weights_to_dict(
                    MODEL_MODE_WEIGHTS.get(mode, MODEL_MODE_WEIGHTS["explore"])
                ),
                "llm_rerank": _llm_rerank_diagnostics(
                    RerankOutcome(
                        enabled=False,
                        model=DEFAULT_MODEL,
                        scores={},
                        reasons={},
                        tokens_input=0,
                        tokens_output=0,
                        cache_hit_input=0,
                        dropped_invalid_names=[],
                        fallback_reason="no_candidates",
                    )
                ),
                "query_anchor": {
                    "domain": "",
                    "market": "",
                    "mechanism": "",
                    "factor_family": "",
                },
                "available_data_provided": available_data is not None,
            },
        }

    graph = bridge_loaders.load_explore_graph(resolved_vault)
    embeddings = bridge_loaders.load_explore_embeddings(resolved_vault)
    exploration = bridge_loaders.load_explore_exploration_map(resolved_vault)
    index_rows = bridge_loaders.load_card_index_rows(resolved_vault)
    raw_matches = bridge_loaders.search_explore_matches(
        idea=idea,
        embeddings=embeddings,
        index_rows=index_rows,
        top_k=max(top_k * 3, 18),
    )
    index_by_name = {
        row.get("name", ""): row for row in index_rows if isinstance(row.get("name"), str)
    }

    weights = MODEL_MODE_WEIGHTS.get(mode, MODEL_MODE_WEIGHTS["explore"])
    anchor = _derive_model_query_anchor(graph=graph, semantic_matches=raw_matches)
    failure_keywords: frozenset[str] = frozenset()
    if exploration is not None:
        failure_keywords = derive_failure_keywords(
            {
                "title": item.title,
                "failure_class": item.failure_class,
                "failure_statement": item.failure_statement,
                "prevention_rule": item.prevention_rule,
            }
            for item in exploration.related_failures(text_query=idea, max_items=8)
        )

    v2_payload = _try_model_v2_knowledge_ranking(
        idea=idea,
        mode=mode,
        vault_root=resolved_vault,
        workspace_root=workspace_root,
        graph=graph,
        embeddings=embeddings,
        index_rows=index_rows,
        anchor=anchor,
        weights=weights,
        available_data=available_data,
        failure_keywords=failure_keywords,
    )
    insight_brief: list[str] = []
    extra_retrieval_diagnostics: dict[str, object] = {}
    if v2_payload is not None:
        ranked = v2_payload.ranked
        score_components_by_name = v2_payload.score_components_by_name
        dropped_cards = v2_payload.dropped_cards
        rerank_outcome = _disabled_rerank_outcome("superseded_by_v2_categorize")
        insight_brief = v2_payload.insight_brief
        extra_retrieval_diagnostics = v2_payload.diagnostics
    else:
        score_components_by_name = {}
        dropped_cards = []
        scored: list[tuple[SearchResult, CardMetadata, ScoreComponents]] = []
        for match in raw_matches:
            if not match.name:
                continue
            card = _build_model_card_metadata(
                graph=graph, name=match.name, fallback_type=match.type
            )
            components, kept, drop_reason = score_card(
                anchor=anchor,
                card=card,
                semantic_score=float(match.score),
                available_data=available_data,
                failure_keywords=failure_keywords,
            )
            if not kept:
                dropped_cards.append({"name": match.name, "reason": drop_reason})
                continue
            scored.append((match, card, components))

        rerank_outcome = rerank_candidates(
            idea=idea,
            mode=mode,
            candidates=[
                RerankCandidate(
                    name=match.name,
                    summary=match.summary or "",
                    domain=card.domain,
                    mechanism=card.mechanism,
                    factor_family=card.factor_family,
                    path=match.path,
                )
                for match, card, _components in scored[:DEFAULT_MAX_CANDIDATES]
            ],
        )

        ranked = []
        for match, _card, components in scored:
            components = replace(
                components,
                llm_relevance=rerank_outcome.scores.get(match.name, 0.0),
            )
            aggregate = aggregate_score(components, weights)
            score_components_by_name[match.name] = {
                **components.to_dict(),
                "aggregate": aggregate,
            }
            ranked.append((aggregate, match))
        ranked.sort(key=lambda pair: (-pair[0], getattr(pair[1], "name", "")))

    knowledge_matches: list[dict[str, object]] = []
    for _, match in ranked[:top_k]:
        row = index_by_name.get(match.name, {})
        rel_path = str(match.path or row.get("path") or "").strip()
        components_payload = score_components_by_name.get(match.name, {})
        knowledge_matches.append(
            {
                "name": match.name,
                "path": rel_path,
                "type": str(match.type or row.get("type") or "").strip(),
                "score": round(float(match.score), 6),
                "summary": str(match.summary or row.get("summary") or "").strip(),
                "snippet": _read_card_snippet(resolved_vault, rel_path),
                "transferable_moves": _read_card_frontmatter_field_items(
                    resolved_vault, rel_path, "transferable_moves"
                ),
                "operative_claims": _read_card_frontmatter_field_items(
                    resolved_vault, rel_path, "operative_claims"
                ),
                "score_components": components_payload,
            }
        )

    handling_patterns = _derive_knowledge_handling_patterns(
        idea=idea,
        knowledge_matches=knowledge_matches,
    )
    frontier_matches: list[dict[str, object]] = []
    failure_refs: list[dict[str, object]] = []
    if exploration is not None:
        frontier_rows = exploration.frontier(priority="high")[:3] or exploration.frontier()[:3]
        frontier_matches = [
            {
                "direction": item.direction,
                "factor_family": item.factor_family,
                "mechanism": item.mechanism,
                "reason": item.reason,
                "priority": item.priority,
            }
            for item in frontier_rows
        ]
        failure_refs = [
            {
                "failure_id": item.failure_id,
                "title": item.title,
                "status": item.status,
                "failure_class": item.failure_class,
                "failure_statement": item.failure_statement,
            }
            for item in exploration.related_failures(text_query=idea, max_items=3)
        ]

    status = "loaded" if knowledge_matches else "loaded_no_match"
    if not knowledge_matches:
        warnings.append("no knowledge match found; treat this as a greenfield idea")
    if graph is None:
        warnings.append("graph context unavailable; relation expansion is disabled")

    retrieval_diagnostics: dict[str, object] = {
        "mode": mode,
        "score_components_by_name": score_components_by_name,
        "dropped_cards": dropped_cards,
        "score_weights": _score_weights_to_dict(weights),
        "llm_rerank": _llm_rerank_diagnostics(rerank_outcome),
        "query_anchor": {
            "domain": anchor.domain,
            "market": anchor.market,
            "mechanism": anchor.mechanism,
            "factor_family": anchor.factor_family,
        },
        "available_data_provided": available_data is not None,
    }
    retrieval_diagnostics.update(extra_retrieval_diagnostics)

    return {
        "status": status,
        "knowledge_matches": knowledge_matches,
        "knowledge_handling_patterns": handling_patterns,
        "insight_brief": insight_brief,
        "frontier_matches": frontier_matches,
        "failure_refs": failure_refs,
        "warnings": warnings,
        "retrieval_diagnostics": retrieval_diagnostics,
    }


@dataclass(slots=True)
class _ModelV2PooledCandidate:
    result: SearchResult
    hit_by: list[str]
    literal_top: bool = False


@dataclass(frozen=True, slots=True)
class _ModelV2RankingPayload:
    ranked: list[tuple[float, SearchResult]]
    score_components_by_name: dict[str, dict[str, float]]
    dropped_cards: list[dict[str, str]]
    insight_brief: list[str]
    diagnostics: dict[str, object]


def _score_weights_to_dict(weights: ScoreWeights) -> dict[str, float]:
    return {
        "semantic": weights.semantic,
        "metadata": weights.metadata,
        "mechanism": weights.mechanism,
        "dependency": weights.dependency,
        "failure": weights.failure,
        "llm_relevance": weights.llm_relevance,
    }


def _llm_rerank_diagnostics(outcome: RerankOutcome) -> dict[str, object]:
    return {
        "enabled": outcome.enabled,
        "model": outcome.model,
        "tokens_input": outcome.tokens_input,
        "tokens_output": outcome.tokens_output,
        "cache_hit_input": outcome.cache_hit_input,
        "dropped_invalid_names": outcome.dropped_invalid_names,
        "fallback_reason": outcome.fallback_reason,
    }


def _disabled_rerank_outcome(reason: str) -> RerankOutcome:
    return RerankOutcome(
        enabled=False,
        model=DEFAULT_MODEL,
        scores={},
        reasons={},
        tokens_input=0,
        tokens_output=0,
        cache_hit_input=0,
        dropped_invalid_names=[],
        fallback_reason=reason,
    )


def _research_bridge_v2_enabled() -> bool:
    return os.environ.get("ALPHA_LAB_RESEARCH_BRIDGE_V2") == "1" and bool(
        os.environ.get("ANTHROPIC_API_KEY")
    )


def _try_model_v2_knowledge_ranking(
    *,
    idea: str,
    mode: str,
    vault_root: Path,
    workspace_root: Path | None,
    graph: object | None,
    embeddings: VaultEmbeddings | None,
    index_rows: list[dict[str, str]],
    anchor: QueryAnchor,
    weights: ScoreWeights,
    available_data: frozenset[str] | None,
    failure_keywords: frozenset[str],
) -> _ModelV2RankingPayload | None:
    if not _research_bridge_v2_enabled():
        return None

    expansion = expand_query(idea=idea, mode=mode)
    if not expansion.enabled:
        return None

    resolved_workspace = Path(workspace_root or PROJECT_ROOT).expanduser().resolve()
    mechanism_embeddings = mechanism_index.load_mechanism_embeddings(
        workspace_root=resolved_workspace,
        vault_root=vault_root,
    )
    candidate_pool, literal_top_paths = _collect_model_v2_candidate_pool(
        idea=idea,
        expansion=expansion,
        embeddings=embeddings,
        mechanism_embeddings=mechanism_embeddings,
        index_rows=index_rows,
    )
    if not candidate_pool:
        return None
    capped_pool, pool_cap_dropped = _apply_model_v2_pool_cap(
        pool=candidate_pool,
        literal_top_paths=literal_top_paths,
        cap=40,
    )

    dropped_cards: list[dict[str, str]] = []
    scored: list[tuple[SearchResult, CardMetadata, ScoreComponents]] = []
    for pooled in capped_pool.values():
        match = pooled.result
        if not match.name:
            continue
        card = _build_model_card_metadata(
            graph=graph,
            name=match.name,
            fallback_type=match.type,
        )
        components, kept, drop_reason = score_card(
            anchor=anchor,
            card=card,
            semantic_score=float(match.score),
            available_data=available_data,
            failure_keywords=failure_keywords,
        )
        if not kept:
            dropped_cards.append({"name": match.name, "reason": drop_reason})
            continue
        scored.append((match, card, components))
    if not scored:
        return None

    fingerprints = {
        match.path: fingerprint
        for match, _card, _components in scored
        if match.path
        for fingerprint in [
            mechanism_index.load_card_fingerprint(
                workspace_root=resolved_workspace,
                vault_root=vault_root,
                path=match.path,
            )
        ]
        if fingerprint is not None
    }
    provenance = {
        path: list(candidate.hit_by) for path, candidate in capped_pool.items()
    }
    categorize_outcome = categorize_and_compress(
        idea=idea,
        mode=mode,
        candidates=[
            RerankCandidate(
                name=match.name,
                summary=match.summary or "",
                domain=card.domain,
                mechanism=card.mechanism,
                factor_family=card.factor_family,
                path=match.path,
            )
            for match, card, _components in scored[:DEFAULT_MAX_CANDIDATES]
        ],
        fingerprints=fingerprints,
        provenance=provenance,
    )
    if not categorize_outcome.enabled:
        return None

    categorized_by_path = categorize_outcome.by_path
    score_components_by_name: dict[str, dict[str, float]] = {}
    ranked: list[tuple[float, SearchResult]] = []
    for match, _card, components in scored:
        category = categorized_by_path.get(match.path)
        if category is not None and category.category == "unrelated":
            if match.path in literal_top_paths:
                llm_relevance = 0.0
            else:
                continue
        else:
            llm_relevance = category.relevance if category is not None else 0.0
        components = replace(components, llm_relevance=llm_relevance)
        aggregate = aggregate_score(components, weights)
        score_components_by_name[match.name] = {
            **components.to_dict(),
            "aggregate": aggregate,
        }
        ranked.append((aggregate, match))
    if not ranked:
        return None
    ranked.sort(key=lambda pair: (-pair[0], getattr(pair[1], "name", "")))

    diagnostics = {
        "v2_enabled": True,
        "query_expansion": _model_expansion_diagnostics(expansion),
        "mechanism_tier": {
            "enabled": mechanism_embeddings is not None,
            "hit_paths": sorted(
                path
                for path, candidate in capped_pool.items()
                if any(tag.endswith("@mechanism") for tag in candidate.hit_by)
            ),
            "fallback_reason": None if mechanism_embeddings is not None else "missing_index",
        },
        "categorize": _model_categorize_diagnostics(categorize_outcome),
        "pool_cap_dropped": pool_cap_dropped,
    }
    return _ModelV2RankingPayload(
        ranked=ranked,
        score_components_by_name=score_components_by_name,
        dropped_cards=dropped_cards,
        insight_brief=list(categorize_outcome.insight_brief),
        diagnostics=diagnostics,
    )


def _collect_model_v2_candidate_pool(
    *,
    idea: str,
    expansion: ExpansionOutcome,
    embeddings: VaultEmbeddings | None,
    mechanism_embeddings: VaultEmbeddings | None,
    index_rows: list[dict[str, str]],
) -> tuple[dict[str, _ModelV2PooledCandidate], set[str]]:
    pool: dict[str, _ModelV2PooledCandidate] = {}
    literal_top_paths: set[str] = set()
    for probe in _dedupe_model_v2([idea, *expansion.probes.direct]):
        for match in bridge_loaders.search_explore_matches(
            idea=probe,
            embeddings=embeddings,
            index_rows=index_rows,
            top_k=10,
        ):
            is_literal_top = probe == idea
            _model_pool_add(
                pool,
                match=match,
                tag=f"direct:{probe}@literal",
                literal_top=is_literal_top,
            )
            if is_literal_top and match.path:
                literal_top_paths.add(match.path)

    if mechanism_embeddings is not None:
        for probe_class, probes in _model_query_probe_items(expansion):
            for probe in probes:
                for match in mechanism_embeddings.search(probe, top_k=5):
                    _model_pool_add(
                        pool,
                        match=match,
                        tag=f"{probe_class}:{probe}@mechanism",
                        literal_top=False,
                    )
    return pool, literal_top_paths


def _model_pool_add(
    pool: dict[str, _ModelV2PooledCandidate],
    *,
    match: SearchResult,
    tag: str,
    literal_top: bool,
) -> None:
    path = str(match.path or "").strip()
    if not path:
        return
    existing = pool.get(path)
    if existing is None:
        pool[path] = _ModelV2PooledCandidate(
            result=match,
            hit_by=[tag],
            literal_top=literal_top,
        )
        return
    if tag not in existing.hit_by:
        existing.hit_by.append(tag)
    if literal_top:
        existing.literal_top = True
    if match.score > existing.result.score:
        existing.result = match


def _apply_model_v2_pool_cap(
    *,
    pool: dict[str, _ModelV2PooledCandidate],
    literal_top_paths: set[str],
    cap: int,
) -> tuple[dict[str, _ModelV2PooledCandidate], list[str]]:
    selected: dict[str, _ModelV2PooledCandidate] = {
        path: pool[path] for path in pool if path in literal_top_paths
    }
    remaining = [
        (path, candidate)
        for path, candidate in pool.items()
        if path not in selected
    ]
    remaining.sort(key=lambda item: _model_v2_pool_priority(item[1]), reverse=True)
    class_limits = {
        "direct": 15,
        "mechanism": 10,
        "analogy": 8,
        "failure": 5,
        "construction": 5,
    }
    class_counts: Counter[str] = Counter()
    for path, candidate in remaining:
        if len(selected) >= cap:
            break
        classes = _model_candidate_probe_classes(candidate)
        chosen_class = next(
            (
                probe_class
                for probe_class in classes
                if class_counts[probe_class] < class_limits.get(probe_class, 5)
            ),
            "",
        )
        if not chosen_class:
            continue
        class_counts[chosen_class] += 1
        selected[path] = candidate
    dropped = sorted(path for path in pool if path not in selected)
    return selected, dropped


def _model_v2_pool_priority(
    candidate: _ModelV2PooledCandidate,
) -> tuple[float, float, float]:
    distinct_provenance = float(len(set(candidate.hit_by)))
    mechanism_hit = (
        1.0 if any(tag.endswith("@mechanism") for tag in candidate.hit_by) else 0.0
    )
    return distinct_provenance, mechanism_hit, float(candidate.result.score)


def _model_candidate_probe_classes(candidate: _ModelV2PooledCandidate) -> list[str]:
    classes: list[str] = []
    for tag in candidate.hit_by:
        probe_class = tag.split(":", 1)[0].strip()
        if probe_class and probe_class not in classes:
            classes.append(probe_class)
    return classes or ["direct"]


def _model_query_probe_items(expansion: ExpansionOutcome) -> list[tuple[str, list[str]]]:
    probes = expansion.probes
    return [
        ("direct", probes.direct),
        ("mechanism", probes.mechanism),
        ("analogy", probes.analogy),
        ("failure", probes.failure),
        ("construction", probes.construction),
    ]


def _dedupe_model_v2(values: list[str]) -> list[str]:
    rows: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(text)
    return rows


def _model_expansion_diagnostics(outcome: ExpansionOutcome) -> dict[str, object]:
    return {
        "enabled": outcome.enabled,
        "model": outcome.model,
        "tokens_input": outcome.tokens_input,
        "tokens_output": outcome.tokens_output,
        "cache_hit_input": outcome.cache_hit_input,
        "fallback_reason": outcome.fallback_reason,
        "probes": {
            "direct": list(outcome.probes.direct),
            "mechanism": list(outcome.probes.mechanism),
            "analogy": list(outcome.probes.analogy),
            "failure": list(outcome.probes.failure),
            "construction": list(outcome.probes.construction),
        },
    }


def _model_categorize_diagnostics(outcome: CategorizeOutcome) -> dict[str, object]:
    return {
        "enabled": outcome.enabled,
        "model": outcome.model,
        "tokens_input": outcome.tokens_input,
        "tokens_output": outcome.tokens_output,
        "cache_hit_input": outcome.cache_hit_input,
        "dropped_invalid_paths": list(outcome.dropped_invalid_paths),
        "fallback_reason": outcome.fallback_reason,
        "categories_by_path": {
            item.path: item.category for item in outcome.categorized
        },
    }


def _build_model_card_metadata(
    *,
    graph: object | None,
    name: str,
    fallback_type: str = "",
) -> CardMetadata:
    """Pull structured metadata for a card via the vault graph.

    Imports the dataclass lazily to avoid a top-level circular import
    risk between scoring/model_idea — both modules are loaded eagerly at
    process start so the late binding is purely defensive.
    """
    if graph is None:
        return CardMetadata(name=name, type=fallback_type)
    node = graph.get_node(name) if hasattr(graph, "get_node") else None
    if node is None:
        return CardMetadata(name=name, type=fallback_type)
    uses_data = (
        graph.get_uses_data(name) if hasattr(graph, "get_uses_data") else []
    )
    depends_on = (
        graph.get_dependency_cards(name)
        if hasattr(graph, "get_dependency_cards")
        else []
    )
    return CardMetadata(
        name=name,
        type=node.type or fallback_type,
        domain=node.domain,
        lifecycle=node.lifecycle,
        market=node.market,
        mechanism=node.mechanism,
        factor_family=node.factor_family,
        uses_data=normalize_data_set(uses_data),
        depends_on=frozenset(depends_on),
    )


def _derive_model_query_anchor(
    *,
    graph: object | None,
    semantic_matches: list,
) -> QueryAnchor:
    if graph is None:
        return QueryAnchor()
    domain = ""
    mechanism = ""
    factor_family = ""
    market = ""
    for match in semantic_matches[:5]:
        node = graph.get_node(match.name) if hasattr(graph, "get_node") else None
        if node is None:
            continue
        if not domain and node.domain:
            domain = node.domain
        if not market and node.market:
            market = node.market
        if not mechanism and node.mechanism:
            mechanism = node.mechanism
        if not factor_family and node.factor_family:
            factor_family = node.factor_family
        if domain and market and mechanism and factor_family:
            break
    return QueryAnchor(
        domain=domain,
        market=market,
        mechanism=mechanism,
        factor_family=factor_family,
    )


def _collect_experiment_context(
    *,
    idea: str,
    workspace_root: Path,
    current_spec: dict[str, object],
    top_k: int,
) -> dict[str, object]:
    model_specs = _collect_model_spec_inventory(
        workspace_root=workspace_root,
        idea=idea,
        top_k=top_k,
    )
    factor_inventory = _collect_factor_inventory(
        workspace_root=workspace_root,
        idea=idea,
        top_k=top_k,
    )
    dist_root = workspace_root / "dist"
    if not dist_root.exists():
        return {
            "status": "no_dist",
            "run_matches": [],
            "validated_baselines": [],
            "recent_failures": [],
            "single_factor_runs": [],
            "model_specs": model_specs,
            "factor_inventory": factor_inventory,
            "warnings": [],
        }

    manifest_paths = sorted(
        dist_root.rglob("run_manifest.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )[:200]
    run_rows: list[dict[str, object]] = []
    single_factor_run_rows: list[dict[str, object]] = []
    for manifest_path in manifest_paths:
        parsed = _parse_model_run_record(manifest_path)
        if parsed is not None:
            run_rows.append(parsed)
        single_factor_parsed = _parse_single_factor_run_record(manifest_path)
        if single_factor_parsed is not None:
            single_factor_run_rows.append(single_factor_parsed)
    single_factor_runs = _summarize_single_factor_runs(
        rows=single_factor_run_rows,
        idea=idea,
        top_k=top_k,
    )
    if not run_rows:
        return {
            "status": "no_model_runs",
            "run_matches": [],
            "validated_baselines": [],
            "recent_failures": [],
            "single_factor_runs": single_factor_runs,
            "model_specs": model_specs,
            "factor_inventory": factor_inventory,
            "warnings": [],
        }

    ranked = _rank_run_rows(
        run_rows=run_rows,
        idea=idea,
        current_spec=current_spec,
    )
    run_matches = [
        {
            "run_id": row["run_id"],
            "case_name": row["case_name"],
            "factor_name": row["factor_name"],
            "model_family": row["model_family"],
            "status": row["status"],
            "score": row["score"],
            "mean_rank_ic": row["mean_rank_ic"],
            "mean_ic": row["mean_ic"],
            "rank_ic_ir": row.get("rank_ic_ir"),
            "long_short_ir": row.get("long_short_ir"),
            "cost_aware_long_short_ir": row.get("cost_aware_long_short_ir"),
            "mean_long_short_turnover": row.get("mean_long_short_turnover"),
            "coverage_mean": row.get("coverage_mean"),
            "output_keys": row.get("output_keys", []),
        }
        for row in ranked[:top_k]
    ]

    validated_baselines: list[dict[str, object]] = []
    for row in ranked:
        if _is_failure_row(row):
            continue
        validated_baselines.append(
            {
                "run_id": row["run_id"],
                "case_name": row["case_name"],
                "factor_name": row["factor_name"],
                "model_family": row["model_family"],
                "status": row["status"],
                "mean_rank_ic": row["mean_rank_ic"],
                "mean_ic": row["mean_ic"],
                "rank_ic_ir": row.get("rank_ic_ir"),
                "long_short_ir": row.get("long_short_ir"),
                "cost_aware_long_short_ir": row.get("cost_aware_long_short_ir"),
                "mean_long_short_turnover": row.get("mean_long_short_turnover"),
                "coverage_mean": row.get("coverage_mean"),
                "output_keys": row.get("output_keys", []),
            }
        )
        if len(validated_baselines) >= min(top_k, 3):
            break

    recent_failures: list[dict[str, object]] = []
    failed_sorted = sorted(
        [row for row in run_rows if _is_failure_row(row)],
        key=lambda item: _sort_float(item.get("mtime")),
        reverse=True,
    )
    for row in failed_sorted[: min(top_k, 3)]:
        recent_failures.append(
            {
                "run_id": row["run_id"],
                "case_name": row["case_name"],
                "factor_name": row["factor_name"],
                "model_family": row["model_family"],
                "status": row["status"],
                "failure_reason": row["failure_reason"],
                "output_keys": row.get("output_keys", []),
            }
        )

    return {
        "status": "loaded",
        "run_matches": run_matches,
        "validated_baselines": validated_baselines,
        "recent_failures": recent_failures,
        "single_factor_runs": single_factor_runs,
        "model_specs": model_specs,
        "factor_inventory": factor_inventory,
        "warnings": [],
    }


def _collect_model_spec_inventory(
    *,
    workspace_root: Path,
    idea: str,
    top_k: int,
) -> list[dict[str, object]]:
    spec_root = workspace_root / "configs" / "real_cases" / "model_factor"
    if not spec_root.exists():
        return []
    rows: list[dict[str, object]] = []
    for path in sorted(spec_root.glob("*.yaml"))[:200]:
        try:
            parsed = load_model_factor_case_spec(path)
        except Exception:
            continue
        feature_columns = [str(item) for item in parsed.feature_columns]
        rows.append(
            {
                "name": parsed.name,
                "path": _relative_path_text(path, workspace_root),
                "factor_name": parsed.factor_name,
                "model_family": parsed.model.family,
                "feature_count": len(feature_columns),
                "feature_columns_sample": feature_columns[:16],
                "features_path": str(parsed.features_path),
                "feature_availability_mode": parsed.feature_availability.mode,
                "feature_availability_column": parsed.feature_availability.column,
                "feature_availability_safety_lag_days": (
                    parsed.feature_availability.safety_lag_days
                ),
                "training_window_type": parsed.training.window_type,
                "model_selection_enabled": bool(parsed.model_selection.enabled),
                "model_selection_metric": parsed.model_selection.metric,
                "mtime": float(path.stat().st_mtime),
            }
        )
    return _rank_inventory_rows(
        rows=rows,
        idea=idea,
        fields=(
            "name",
            "path",
            "factor_name",
            "model_family",
            "features_path",
            "feature_columns_sample",
        ),
        top_k=max(top_k, 8),
    )


def _collect_factor_inventory(
    *,
    workspace_root: Path,
    idea: str,
    top_k: int,
) -> dict[str, object]:
    builtin_methods = sorted(set(factor_registry.supported_methods()) | {"vcimom"})
    baseline_factors = _rank_inventory_rows(
        rows=[
            {
                "name": item.get("name"),
                "family": item.get("family"),
                "label": item.get("label"),
                "description": item.get("description"),
                "base_method": item.get("base_method"),
                "required_columns": item.get("required_columns", []),
                "tags": item.get("tags", []),
            }
            for item in baseline_factor_suite_payload(include_non_default=True)
        ],
        idea=idea,
        fields=("name", "family", "label", "description", "base_method", "tags"),
        top_k=max(top_k, 8),
    )
    custom_factors: list[dict[str, object]] = []
    for path in iter_custom_factor_meta_paths(workspace_root):
        try:
            source = read_custom_factor_source(path)
        except Exception:
            continue
        custom_factors.append(
            {
                "name": source.name,
                "scope": source.scope,
                "path": _relative_path_text(source.path, workspace_root),
                "description": source.description,
                "required_columns": list(source.required_columns),
                "optional_columns": list(source.optional_columns),
                "frequency": source.frequency,
                "pit_assumption": source.pit_assumption,
                "mtime": float(source.path.stat().st_mtime),
            }
        )
    custom_factors = _rank_inventory_rows(
        rows=custom_factors,
        idea=idea,
        fields=("name", "scope", "description", "required_columns", "optional_columns"),
        top_k=max(top_k, 8),
    )
    single_factor_cases = _collect_single_factor_case_inventory(
        workspace_root=workspace_root,
        idea=idea,
        top_k=max(top_k, 8),
    )
    return {
        "builtin_methods": builtin_methods,
        "baseline_factors": baseline_factors,
        "custom_factors": custom_factors,
        "single_factor_cases": single_factor_cases,
    }


def _collect_single_factor_case_inventory(
    *,
    workspace_root: Path,
    idea: str,
    top_k: int,
) -> list[dict[str, object]]:
    spec_root = workspace_root / "configs" / "real_cases" / "single_factor"
    if not spec_root.exists():
        return []
    rows: list[dict[str, object]] = []
    for path in sorted(spec_root.glob("*.yaml"))[:200]:
        try:
            parsed = load_single_factor_case_spec(path)
        except Exception:
            continue
        recipe = parsed.factor_input.recipe if parsed.factor_input is not None else None
        base_method = ""
        if isinstance(recipe, dict):
            base = recipe.get("base")
            if isinstance(base, dict):
                base_method = str(base.get("method") or "")
        rows.append(
            {
                "name": parsed.name,
                "path": _relative_path_text(path, workspace_root),
                "factor_name": parsed.factor_name,
                "factor_path": parsed.factor_path,
                "factor_input_mode": (
                    parsed.factor_input.mode if parsed.factor_input is not None else "file"
                ),
                "base_method": base_method,
                "target_horizon": parsed.target.horizon,
                "rebalance_frequency": parsed.rebalance_frequency,
                "preprocess": {
                    "winsorize": parsed.preprocess.winsorize,
                    "standardization": parsed.preprocess.standardization,
                },
                "mtime": float(path.stat().st_mtime),
            }
        )
    return _rank_inventory_rows(
        rows=rows,
        idea=idea,
        fields=("name", "path", "factor_name", "factor_path", "base_method"),
        top_k=top_k,
    )


def _rank_inventory_rows(
    *,
    rows: list[dict[str, object]],
    idea: str,
    fields: tuple[str, ...],
    top_k: int,
) -> list[dict[str, object]]:
    keywords = _idea_keywords(idea)
    ranked: list[dict[str, object]] = []
    for row in rows:
        haystack_parts: list[str] = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, list):
                haystack_parts.extend(str(item) for item in value)
            else:
                haystack_parts.append(str(value or ""))
        haystack = " ".join(haystack_parts).lower()
        score = float(sum(1 for keyword in keywords if keyword in haystack))
        ranked.append({**row, "score": round(score, 4)})
    ranked.sort(
        key=lambda item: (
            -_sort_float(item.get("score")),
            -_sort_float(item.get("mtime")),
            str(item.get("name") or ""),
        )
    )
    cleaned: list[dict[str, object]] = []
    for item in ranked[:top_k]:
        row = dict(item)
        row.pop("mtime", None)
        cleaned.append(row)
    return cleaned


def _relative_path_text(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def model_idea_sessions_root(*, workspace_root: str | Path = PROJECT_ROOT) -> Path:
    return (Path(workspace_root).expanduser().resolve() / _MODEL_IDEA_SESSIONS_REL_PATH).resolve()


def save_model_idea_session(
    *,
    payload: dict[str, object],
    workspace_root: str | Path = PROJECT_ROOT,
    session_id: str | None = None,
    parent_session_id: str | None = None,
) -> dict[str, object]:
    session_root = model_idea_sessions_root(workspace_root=workspace_root)
    session_root.mkdir(parents=True, exist_ok=True)
    created_at = dt.datetime.now(dt.UTC)
    resolved_session_id = _allocate_session_id(
        root=session_root,
        idea=str(payload.get("idea") or ""),
        created_at=created_at,
        preferred=session_id,
    )
    session_payload = _build_session_payload(
        session_id=resolved_session_id,
        created_at=created_at,
        payload=payload,
        parent_session_id=parent_session_id,
    )
    session_path = (session_root / f"{resolved_session_id}.json").resolve()
    session_path.write_text(
        json.dumps(session_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return _summarize_session_payload(session_payload, path=session_path)


def list_model_idea_sessions(
    *,
    workspace_root: str | Path = PROJECT_ROOT,
    limit: int = 20,
    include_archived: bool = False,
) -> list[dict[str, object]]:
    session_root = model_idea_sessions_root(workspace_root=workspace_root)
    if not session_root.exists():
        return []
    normalized_limit = max(int(limit), 1)
    rows: list[dict[str, object]] = []
    for path in _iter_session_paths(session_root):
        payload = _read_model_idea_session_file(path)
        if payload is None:
            continue
        if bool(payload.get("archived")) and not include_archived:
            continue
        rows.append(_summarize_session_payload(payload, path=path))
        if len(rows) >= normalized_limit:
            break
    return rows


def read_model_idea_session(
    *,
    session_id: str,
    workspace_root: str | Path = PROJECT_ROOT,
) -> dict[str, object]:
    normalized_session_id = _normalize_session_id(session_id)
    session_root = model_idea_sessions_root(workspace_root=workspace_root)
    session_path = (session_root / f"{normalized_session_id}.json").resolve()
    if not session_path.exists() or not session_path.is_file():
        raise FileNotFoundError(f"model-idea session not found: {normalized_session_id}")
    payload = _read_model_idea_session_file(session_path)
    if payload is None:
        raise ValueError(f"invalid model-idea session payload: {normalized_session_id}")
    return payload


def delete_model_idea_session(
    *,
    session_id: str,
    workspace_root: str | Path = PROJECT_ROOT,
) -> dict[str, object]:
    normalized_session_id = _normalize_session_id(session_id)
    session_root = model_idea_sessions_root(workspace_root=workspace_root)
    session_path = (session_root / f"{normalized_session_id}.json").resolve()
    if not session_path.exists() or not session_path.is_file():
        raise FileNotFoundError(f"model-idea session not found: {normalized_session_id}")
    payload = _read_model_idea_session_file(session_path)
    if payload is None:
        raise ValueError(f"invalid model-idea session payload: {normalized_session_id}")
    payload["archived"] = True
    payload["archived_at_utc"] = dt.datetime.now(dt.UTC).isoformat()
    session_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return {"ok": True, "session_id": normalized_session_id, "archived": True}


def record_model_idea_response(
    *,
    session_id: str,
    response_text: str,
    workspace_root: str | Path = PROJECT_ROOT,
) -> LintReport:
    """Lint and persist an external LLM response for a model-lab session."""
    normalized_session_id = _normalize_session_id(session_id)
    text = str(response_text or "")
    if not text.strip():
        raise ValueError("response_text must be non-empty")
    session_root = model_idea_sessions_root(workspace_root=workspace_root)
    session_path = (session_root / f"{normalized_session_id}.json").resolve()
    if not session_path.exists() or not session_path.is_file():
        raise FileNotFoundError(f"model-idea session not found: {normalized_session_id}")
    payload = _read_model_idea_session_file(session_path)
    if payload is None:
        raise ValueError(f"invalid model-idea session payload: {normalized_session_id}")

    stage = normalize_workflow_stage(
        str(payload.get("stage") or MECHANISM_DISCOVERY),
        allow_stage3_helper=True,
    )
    mode = str(payload.get("mode") or "explore").strip().lower() or "explore"
    report = lint_model_idea_response(text, stage=stage, mode=mode)
    response_sections = extract_model_stage_sections(text, stage=stage)
    updated = {
        **payload,
        "response": text,
        "response_sections": response_sections,
        "responded_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
        "lint_report": report.to_dict(),
    }
    session_path.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def list_recent_model_idea_violations(
    *,
    workspace_root: str | Path = PROJECT_ROOT,
    stage: str,
    limit: int = 5,
) -> list[dict[str, object]]:
    normalized_stage = normalize_workflow_stage(stage)
    rows: list[dict[str, object]] = []
    try:
        sessions = list_model_idea_sessions(
            workspace_root=workspace_root,
            limit=max(int(limit) * 10, 20),
        )
    except OSError:
        return []
    for summary in sessions:
        if str(summary.get("stage") or "") != normalized_stage:
            continue
        session_id = str(summary.get("session_id") or "")
        if not session_id:
            continue
        try:
            payload = read_model_idea_session(
                session_id=session_id,
                workspace_root=workspace_root,
            )
        except (FileNotFoundError, ValueError):
            continue
        lint_report = _as_mapping(payload.get("lint_report"))
        for violation in _as_dict_list(lint_report.get("violations")):
            rows.append(
                {
                    "session_id": session_id,
                    "stage": normalized_stage,
                    "code": violation.get("code"),
                    "section": violation.get("section"),
                    "detail": violation.get("detail"),
                }
            )
            if len(rows) >= limit:
                return rows
    return rows


def render_model_idea_drift_header(violations: list[dict[str, object]]) -> str:
    if not violations:
        return ""
    lines = [
        "## 已知漂移模式（来自最近 model-lab lint）",
        "上一轮 model-lab 响应出现过以下结构漂移；本轮必须主动避免：",
    ]
    for item in violations[:5]:
        lines.append(
            "- session={sid} code={code} section={section}: {detail}".format(
                sid=item.get("session_id") or "-",
                code=item.get("code") or "-",
                section=item.get("section") or "-",
                detail=item.get("detail") or "-",
            )
        )
    return "\n".join(lines)


def find_upstream_model_idea_session(
    *,
    workspace_root: str | Path = PROJECT_ROOT,
    stage: str,
    parent_session_id: str | None = None,
) -> dict[str, object] | None:
    normalized_stage = normalize_workflow_stage(stage)
    previous_stage = _MODEL_PREVIOUS_STAGE.get(normalized_stage)
    if previous_stage is None:
        return None
    if parent_session_id:
        try:
            candidate = read_model_idea_session(
                session_id=parent_session_id,
                workspace_root=workspace_root,
            )
        except (FileNotFoundError, ValueError):
            return None
        if str(candidate.get("stage") or "") == previous_stage and _as_mapping(
            candidate.get("response_sections")
        ):
            return candidate
        return None
    try:
        sessions = list_model_idea_sessions(workspace_root=workspace_root, limit=50)
    except OSError:
        return None
    for summary in sessions:
        if str(summary.get("stage") or "") != previous_stage:
            continue
        session_id = str(summary.get("session_id") or "")
        if not session_id:
            continue
        try:
            candidate = read_model_idea_session(
                session_id=session_id,
                workspace_root=workspace_root,
            )
        except (FileNotFoundError, ValueError):
            continue
        if _as_mapping(candidate.get("response_sections")):
            return candidate
    return None


def count_model_upstream_sections(
    upstream_session: dict[str, object] | None,
    *,
    current_stage: str,
) -> int:
    if upstream_session is None:
        return 0
    sections = _as_mapping(upstream_session.get("response_sections"))
    priority = _MODEL_UPSTREAM_SECTION_PRIORITY.get(
        normalize_workflow_stage(current_stage), ()
    )
    return sum(1 for key in priority if str(sections.get(key) or "").strip())


def render_model_upstream_artifact_header(
    upstream_session: dict[str, object],
    *,
    current_stage: str,
) -> str:
    normalized_stage = normalize_workflow_stage(current_stage)
    previous_stage = _MODEL_PREVIOUS_STAGE.get(normalized_stage, "")
    sections = _as_mapping(upstream_session.get("response_sections"))
    priority = _MODEL_UPSTREAM_SECTION_PRIORITY.get(normalized_stage, ())
    if not priority or not sections:
        return ""
    lines = [
        "## 上游模型研究产物",
        "- source_session_id: {sid}".format(
            sid=upstream_session.get("session_id") or "-"
        ),
        f"- stage_chain: {previous_stage} -> {normalized_stage}",
        "- 使用规则：以下内容是上一阶段产物；本阶段应继续审计/映射，不要从零重写。",
    ]
    for label in priority:
        body = str(sections.get(label) or "").strip()
        if not body:
            continue
        lines.extend(["", f"### {label}", body[:4000]])
    return "\n".join(lines)


def _collect_session_memory_context(
    *,
    workspace_root: Path,
    current_idea: str,
    limit: int,
) -> dict[str, object]:
    if limit <= 0:
        return {
            "status": "disabled",
            "session_memory": [],
            "warnings": [],
        }
    memory_rows: list[dict[str, object]] = []
    try:
        sessions = list_model_idea_sessions(
            workspace_root=workspace_root,
            limit=max(limit * 5, 20),
        )
    except OSError as exc:
        return {
            "status": "unavailable",
            "session_memory": [],
            "warnings": [f"session memory unavailable: {exc}"],
        }
    current_idea_l = current_idea.strip().lower()
    for item in sessions:
        idea = str(item.get("idea") or "").strip()
        if current_idea_l and idea.lower() == current_idea_l:
            continue
        memory_rows.append(
            {
                "session_id": item.get("session_id"),
                "created_at_utc": item.get("created_at_utc"),
                "idea": idea,
                "mode": item.get("mode"),
                "stage": item.get("stage"),
                "spec_name": item.get("spec_name"),
                "patch_summary": item.get("patch_summary"),
                "has_response": bool(item.get("has_response")),
                "has_lint_errors": bool(item.get("has_lint_errors")),
                "violation_codes": item.get("violation_codes", []),
            }
        )
        if len(memory_rows) >= limit:
            break
    return {
        "status": "loaded" if memory_rows else "no_sessions",
        "session_memory": memory_rows,
        "warnings": [],
    }


def _parse_model_run_record(manifest_path: Path) -> dict[str, object] | None:
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest_payload, dict):
        return None

    workflow = str(manifest_payload.get("workflow") or "").strip().lower()
    outputs = manifest_payload.get("outputs")
    output_keys = sorted(str(key) for key in outputs.keys()) if isinstance(outputs, dict) else []
    metrics_path = _resolve_metrics_path(manifest_payload, manifest_path.parent)
    metrics_summary = _load_metrics_summary(metrics_path)
    if "model_factor" not in workflow and not metrics_summary.get("model_family"):
        return None

    run_id = str(manifest_payload.get("run_id") or manifest_path.parent.name).strip()
    status = str(
        manifest_payload.get("status")
        or manifest_payload.get("run_status")
        or metrics_summary.get("run_status")
        or "unknown"
    ).strip()
    factor_verdict = str(metrics_summary.get("factor_verdict") or "").strip()
    severity_hint = str(
        metrics_summary.get("highest_severity")
        or metrics_summary.get("integrity_highest_severity")
        or ""
    ).strip()
    failure_reason = _first_non_empty(
        [
            str(manifest_payload.get("error") or "").strip(),
            str(metrics_summary.get("run_error") or "").strip(),
            str(metrics_summary.get("integrity_failure_reason") or "").strip(),
            severity_hint,
            factor_verdict,
        ]
    )
    return {
        "run_id": run_id,
        "case_name": str(manifest_payload.get("case_name") or manifest_path.parent.name).strip(),
        "factor_name": str(metrics_summary.get("factor_name") or "").strip(),
        "model_family": str(metrics_summary.get("model_family") or "").strip(),
        "status": status.lower(),
        "factor_verdict": factor_verdict.lower(),
        "severity_hint": severity_hint.lower(),
        "mean_rank_ic": _to_float_or_none(metrics_summary.get("mean_rank_ic")),
        "mean_ic": _to_float_or_none(metrics_summary.get("mean_ic")),
        "rank_ic_ir": _first_metric_float(
            metrics_summary, "rank_ic_ir", "rank_ic_information_ratio"
        ),
        "ic_ir": _first_metric_float(metrics_summary, "ic_ir", "information_ratio"),
        "long_short_ir": _first_metric_float(
            metrics_summary, "long_short_ir", "long_short_information_ratio"
        ),
        "cost_aware_long_short_ir": _first_metric_float(
            metrics_summary,
            "cost_aware_long_short_ir",
            "net_long_short_ir",
            "cost_adjusted_long_short_ir",
        ),
        "mean_long_short_turnover": _first_metric_float(
            metrics_summary,
            "mean_long_short_turnover",
            "long_short_turnover",
            "turnover_mean",
        ),
        "max_drawdown": _first_metric_float(metrics_summary, "max_drawdown", "mdd"),
        "coverage_mean": _first_metric_float(
            metrics_summary, "coverage_mean", "mean_coverage", "score_coverage_mean"
        ),
        "n_dates_used": _first_metric_float(
            metrics_summary, "n_dates_used", "n_dates", "n_eval_dates"
        ),
        "output_keys": output_keys,
        "path": str(manifest_path),
        "mtime": float(manifest_path.stat().st_mtime),
        "failure_reason": failure_reason,
    }


def _parse_single_factor_run_record(manifest_path: Path) -> dict[str, object] | None:
    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(manifest_payload, dict):
        return None

    workflow = str(manifest_payload.get("workflow") or "").strip().lower()
    if workflow and "single_factor" not in workflow:
        return None
    metrics_path = _resolve_metrics_path(manifest_payload, manifest_path.parent)
    metrics_summary = _load_metrics_summary(metrics_path)
    if metrics_summary.get("model_family"):
        return None
    factor_name = str(
        metrics_summary.get("factor_name")
        or manifest_payload.get("factor_name")
        or ""
    ).strip()
    if "single_factor" not in workflow and not factor_name:
        return None
    outputs = manifest_payload.get("outputs")
    output_keys = sorted(str(key) for key in outputs.keys()) if isinstance(outputs, dict) else []
    status = str(
        manifest_payload.get("status")
        or manifest_payload.get("run_status")
        or metrics_summary.get("run_status")
        or "unknown"
    ).strip()
    return {
        "run_id": str(manifest_payload.get("run_id") or manifest_path.parent.name).strip(),
        "case_name": str(manifest_payload.get("case_name") or manifest_path.parent.name).strip(),
        "factor_name": factor_name,
        "status": status.lower(),
        "mean_rank_ic": _to_float_or_none(metrics_summary.get("mean_rank_ic")),
        "mean_ic": _to_float_or_none(metrics_summary.get("mean_ic")),
        "rank_ic_ir": _first_metric_float(
            metrics_summary, "rank_ic_ir", "rank_ic_information_ratio"
        ),
        "long_short_ir": _first_metric_float(
            metrics_summary, "long_short_ir", "long_short_information_ratio"
        ),
        "cost_aware_long_short_ir": _first_metric_float(
            metrics_summary,
            "cost_aware_long_short_ir",
            "net_long_short_ir",
            "cost_adjusted_long_short_ir",
        ),
        "factor_verdict": str(metrics_summary.get("factor_verdict") or "").strip(),
        "output_keys": output_keys,
        "path": str(manifest_path),
        "mtime": float(manifest_path.stat().st_mtime),
    }


def _summarize_single_factor_runs(
    *,
    rows: list[dict[str, object]],
    idea: str,
    top_k: int,
) -> list[dict[str, object]]:
    ranked = _rank_inventory_rows(
        rows=rows,
        idea=idea,
        fields=("run_id", "case_name", "factor_name", "status", "factor_verdict"),
        top_k=max(top_k, 5),
    )
    return [
        {
            "run_id": row.get("run_id"),
            "case_name": row.get("case_name"),
            "factor_name": row.get("factor_name"),
            "status": row.get("status"),
            "mean_rank_ic": row.get("mean_rank_ic"),
            "mean_ic": row.get("mean_ic"),
            "rank_ic_ir": row.get("rank_ic_ir"),
            "long_short_ir": row.get("long_short_ir"),
            "cost_aware_long_short_ir": row.get("cost_aware_long_short_ir"),
            "factor_verdict": row.get("factor_verdict"),
            "output_keys": row.get("output_keys", []),
        }
        for row in ranked
    ]


def _resolve_metrics_path(manifest_payload: dict[str, object], run_dir: Path) -> Path | None:
    outputs = manifest_payload.get("outputs")
    if isinstance(outputs, dict):
        metrics = outputs.get("metrics")
        if isinstance(metrics, str) and metrics.strip():
            candidate = Path(metrics.strip()).expanduser().resolve()
            if candidate.exists():
                return candidate
    fallback = (run_dir / "metrics.json").resolve()
    if fallback.exists():
        return fallback
    return None


def _load_metrics_summary(metrics_path: Path | None) -> dict[str, object]:
    if metrics_path is None or not metrics_path.exists():
        return {}
    try:
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, dict):
        return {}
    raw_metrics = payload.get("metrics")
    if isinstance(raw_metrics, dict):
        return {str(key): value for key, value in raw_metrics.items()}
    return {str(key): value for key, value in payload.items()}


def _rank_run_rows(
    *,
    run_rows: list[dict[str, object]],
    idea: str,
    current_spec: dict[str, object],
) -> list[dict[str, object]]:
    """Rank experiment rows with explicit priority: contracts > failures > validated > others."""

    keywords = _idea_keywords(idea)
    current_model_family = str(current_spec.get("model_family") or "").strip().lower()
    current_factor_name = str(current_spec.get("factor_name") or "").strip().lower()
    ranked: list[dict[str, object]] = []

    for row in run_rows:
        haystack = " ".join(
            [
                str(row.get("case_name") or ""),
                str(row.get("factor_name") or ""),
                str(row.get("model_family") or ""),
                str(row.get("status") or ""),
                str(row.get("path") or ""),
            ]
        ).lower()
        score = float(sum(1 for keyword in keywords if keyword in haystack))
        is_failure = _is_failure_row(row)
        if current_model_family and current_model_family == str(
            row.get("model_family") or ""
        ).lower():
            score += 3.0
        if current_factor_name and current_factor_name == str(
            row.get("factor_name") or ""
        ).lower():
            score += 1.0
        if is_failure:
            score -= 100.0
        elif str(row.get("status") or "").lower() in {"succeeded", "success", "completed"}:
            score += 1.0
        ranked.append({**row, "score": round(score, 4), "is_failure": is_failure})
    ranked.sort(
        key=lambda item: (
            -_sort_float(item.get("score")),
            -_sort_float(item.get("mtime")),
        )
    )
    return ranked


def _is_failure_row(row: dict[str, object]) -> bool:
    status = str(row.get("status") or "").lower()
    if status in {"failed", "failure", "error", "cancelled"}:
        return True
    severity = str(row.get("severity_hint") or "").lower()
    if severity in {"fail", "failed", "error"}:
        return True
    verdict = str(row.get("factor_verdict") or "").lower()
    if any(token in verdict for token in ("fail", "reject", "not_pass", "bad")):
        return True
    return False


def _build_recommendations(
    *,
    mode: str,
    stage: str,
    warnings: list[str],
    knowledge_context: dict[str, object],
    experiment_context: dict[str, object],
    session_memory_context: dict[str, object],
) -> dict[str, object]:
    """Build stable recommendation payload with explicit evidence priority ordering."""

    if stage == MECHANISM_DISCOVERY:
        next_actions = [
            "Surface 2-4 structurally different model mechanisms; do NOT converge yet.",
            "Separate loss/regularization, feature interaction, target construction, sample weighting, training window, and model selection stories.",
            "For each mechanism, mark optional inspired_by/fusion_of, touched contract surfaces, and data-validation concerns.",
            "Defer final spec patches and winner selection to Stage 3 data validation after additive ledger expansion.",
        ]
    elif stage == SIGNAL_MAPPING:
        next_actions = [
            "Map upstream mechanisms into 2-3 concrete model/spec variants without picking a winner.",
            "Tag every required field as necessary, decorative, or risk control.",
            "Address model confounds: PIT, target leakage, complexity, turnover/cost, feature stability, and split fragility.",
            "Explain which mechanism the current implementation captures and which cannot be disambiguated at this data tier.",
            "Emit ledger_v1/retrieval_log-ready synthesis notes; do not make kill verdicts in Stage 1.",
        ]
    else:
        next_actions = [
            "Try to kill the model idea before improving it.",
            "Audit whether performance is baseline/ridge, regularization-only, feature-count, leakage, split luck, or turnover/cost artifact.",
            "Require data-time, split, feature-importance, and cost/portfolio stability checks before HOLD-FOR-AUDIT.",
        ]
    if mode == "constrained":
        next_actions.insert(
            0,
            "Reject any idea requiring unsupported model/preprocess/selection contracts unless explicitly marked requires_code_change.",
        )
    elif mode == "start" and stage == MECHANISM_DISCOVERY:
        next_actions.append(
            "Use cross-domain analogies, but keep contracts discussion-only and hypotheses modifiable."
        )

    if mode == "start":
        mode_guidance = (
            "Cross-domain kickoff: leverage analogies first, treat contracts "
            "as discussion-only, and keep the hypothesis space open."
        )
    elif mode == "explore":
        mode_guidance = "Use broad hypothesis expansion."
    else:
        mode_guidance = "Keep only executable ideas."

    return {
        "mode_guidance": mode_guidance,
        "source_anchors": [
            {
                "path": item["path"],
                "focus": item["focus"],
            }
            for item in _SOURCE_ANCHORS
        ],
        "next_actions": next_actions,
        "extras": {
            "priority_order": list(_RECOMMENDATION_PRIORITY_ORDER),
            "warnings": list(warnings),
            "knowledge_context_status": knowledge_context["status"],
            "knowledge_matches": knowledge_context["knowledge_matches"],
            "knowledge_handling_patterns": knowledge_context["knowledge_handling_patterns"],
            "insight_brief": knowledge_context.get("insight_brief", []),
            "frontier_matches": knowledge_context["frontier_matches"],
            "failure_refs": knowledge_context["failure_refs"],
            "run_context_status": experiment_context["status"],
            "run_matches": experiment_context["run_matches"],
            "existing_model_specs": experiment_context["model_specs"],
            "factor_inventory": experiment_context["factor_inventory"],
            "single_factor_runs": experiment_context["single_factor_runs"],
            "session_memory_status": session_memory_context["status"],
            "session_memory": session_memory_context["session_memory"],
        },
    }


def _build_model_idea_exploration_prompt(
    *,
    idea: str,
    mode: str,
    report: dict[str, object],
    spec_patch_hint: dict[str, object] | None,
    stage: str = "mechanism_discovery",
    drift_header: str = "",
    upstream_header: str = "",
) -> str:
    if stage == SIGNAL_MAPPING:
        return _build_model_idea_signal_mapping_prompt(
            idea=idea,
            mode=mode,
            report=report,
            spec_patch_hint=spec_patch_hint,
            stage=stage,
            strict=mode == "constrained",
            drift_header=drift_header,
            upstream_header=upstream_header,
        )
    if stage == VALIDATION_KILL_TESTS:
        return _build_model_idea_validation_kill_tests_prompt(
            idea=idea,
            mode=mode,
            report=report,
            spec_patch_hint=spec_patch_hint,
            stage=stage,
            strict=mode == "constrained",
            drift_header=drift_header,
            upstream_header=upstream_header,
        )
    if mode == "start":
        return _build_model_idea_mechanism_start_prompt(
            idea=idea,
            mode=mode,
            report=report,
            spec_patch_hint=spec_patch_hint,
            stage=MECHANISM_DISCOVERY,
            drift_header=drift_header,
            upstream_header=upstream_header,
        )
    if mode == "constrained":
        return _build_model_idea_mechanism_constrained_prompt(
            idea=idea,
            mode=mode,
            report=report,
            spec_patch_hint=spec_patch_hint,
            stage=MECHANISM_DISCOVERY,
            drift_header=drift_header,
            upstream_header=upstream_header,
        )
    return _build_model_idea_mechanism_structured_prompt(
        idea=idea,
        mode=mode,
        report=report,
        spec_patch_hint=spec_patch_hint,
        stage=MECHANISM_DISCOVERY,
        drift_header=drift_header,
        upstream_header=upstream_header,
    )


def _build_model_idea_mechanism_start_prompt(
    *,
    idea: str,
    mode: str,
    report: dict[str, object],
    spec_patch_hint: dict[str, object] | None,
    stage: str = MECHANISM_DISCOVERY,
    drift_header: str = "",
    upstream_header: str = "",
) -> str:
    lines = _build_model_idea_common_prompt_lines(
        idea=idea,
        mode=mode,
        stage=stage,
        report=report,
        spec_patch_hint=spec_patch_hint,
        drift_header=drift_header,
        upstream_header=upstream_header,
    )
    lines.extend(["", "## Task"])
    _append_model_mechanism_discovery_task(lines, mode="start")
    _append_model_lint_self_check(lines, stage=stage, mode=mode)
    return "\n".join(lines)


def _build_model_idea_mechanism_structured_prompt(
    *,
    idea: str,
    mode: str,
    report: dict[str, object],
    spec_patch_hint: dict[str, object] | None,
    stage: str = MECHANISM_DISCOVERY,
    drift_header: str = "",
    upstream_header: str = "",
) -> str:
    lines = _build_model_idea_common_prompt_lines(
        idea=idea,
        mode=mode,
        stage=stage,
        report=report,
        spec_patch_hint=spec_patch_hint,
        drift_header=drift_header,
        upstream_header=upstream_header,
    )
    lines.extend(["", "## Task"])
    _append_model_mechanism_discovery_task(lines, mode="explore")
    _append_model_lint_self_check(lines, stage=stage, mode=mode)
    return "\n".join(lines)


def _build_model_idea_mechanism_constrained_prompt(
    *,
    idea: str,
    mode: str,
    report: dict[str, object],
    spec_patch_hint: dict[str, object] | None,
    stage: str = MECHANISM_DISCOVERY,
    drift_header: str = "",
    upstream_header: str = "",
) -> str:
    lines = _build_model_idea_common_prompt_lines(
        idea=idea,
        mode=mode,
        stage=stage,
        report=report,
        spec_patch_hint=spec_patch_hint,
        drift_header=drift_header,
        upstream_header=upstream_header,
    )
    lines.extend(["", "## Task"])
    _append_model_mechanism_discovery_task(lines, mode="constrained")
    _append_model_lint_self_check(lines, stage=stage, mode=mode)
    return "\n".join(lines)


def _build_model_idea_signal_mapping_prompt(
    *,
    idea: str,
    mode: str,
    report: dict[str, object],
    spec_patch_hint: dict[str, object] | None,
    stage: str = SIGNAL_MAPPING,
    strict: bool = False,
    drift_header: str = "",
    upstream_header: str = "",
) -> str:
    lines = _build_model_idea_common_prompt_lines(
        idea=idea,
        mode=mode,
        stage=stage,
        report=report,
        spec_patch_hint=spec_patch_hint,
        drift_header=drift_header,
        upstream_header=upstream_header,
    )
    lines.extend(["", "## Task"])
    _append_model_signal_mapping_task(
        lines, mode="constrained" if strict else mode
    )
    _append_model_lint_self_check(lines, stage=stage, mode=mode)
    return "\n".join(lines)


def _build_model_idea_validation_kill_tests_prompt(
    *,
    idea: str,
    mode: str,
    report: dict[str, object],
    spec_patch_hint: dict[str, object] | None,
    stage: str = VALIDATION_KILL_TESTS,
    strict: bool = False,
    drift_header: str = "",
    upstream_header: str = "",
) -> str:
    raise ValueError(
        "validation_kill_tests has moved out of model-lab idea exploration; "
        "use mechanism_discovery/signal_mapping here and run Stage 3 data "
        "validation separately."
    )
    lines = _build_model_idea_common_prompt_lines(
        idea=idea,
        mode=mode,
        stage=stage,
        report=report,
        spec_patch_hint=spec_patch_hint,
        drift_header=drift_header,
        upstream_header=upstream_header,
    )
    lines.extend(["", "## Task"])
    _append_model_validation_kill_tests_task(
        lines, mode="constrained" if strict else mode
    )
    _append_model_lint_self_check(lines, stage=stage, mode=mode)
    return "\n".join(lines)


def _build_model_idea_common_prompt_lines(
    *,
    idea: str,
    mode: str,
    stage: str,
    report: dict[str, object],
    spec_patch_hint: dict[str, object] | None,
    drift_header: str = "",
    upstream_header: str = "",
) -> list[str]:
    system_contracts = _as_mapping(report.get("system_contracts"))
    current_spec = _as_mapping(report.get("current_spec"))
    validated_baselines = _as_dict_list(report.get("validated_baselines"))
    recent_failures = _as_dict_list(report.get("recent_failures"))
    recommendations = _as_mapping(report.get("recommendations"))
    model_families = _as_str_list(system_contracts.get("supported_model_families"))
    supported_preprocess = _as_mapping(
        system_contracts.get("supported_feature_preprocess")
    )
    supported_training = _as_mapping(system_contracts.get("supported_training"))
    supported_selection_metrics = _as_str_list(
        system_contracts.get("supported_selection_metrics")
    )
    supported_feature_importance = _as_mapping(
        system_contracts.get("supported_feature_importance")
    )
    extras = _as_mapping(recommendations.get("extras"))
    knowledge_matches = _as_dict_list(extras.get("knowledge_matches"))
    handling_patterns = _as_str_list(extras.get("knowledge_handling_patterns"))
    insight_brief = _as_str_list(extras.get("insight_brief"))
    existing_model_specs = _as_dict_list(extras.get("existing_model_specs"))
    factor_inventory = _as_mapping(extras.get("factor_inventory"))
    single_factor_runs = _as_dict_list(extras.get("single_factor_runs"))
    session_memory_rows = _as_dict_list(extras.get("session_memory"))
    warnings = _as_str_list(extras.get("warnings"))
    source_anchors = _as_dict_list(recommendations.get("source_anchors"))
    next_stage = recommend_next_stage(stage)

    lines: list[str] = []
    _append_model_research_idea_section(
        lines, idea=idea, mode=mode, stage=stage, next_stage=next_stage
    )
    if upstream_header.strip():
        lines = [upstream_header.strip(), "", *lines]
    if drift_header.strip():
        lines = [drift_header.strip(), "", *lines]
    _append_model_system_contracts_section(
        lines,
        model_families=model_families,
        supported_preprocess=supported_preprocess,
        supported_training=supported_training,
        supported_selection_metrics=supported_selection_metrics,
        supported_feature_importance=supported_feature_importance,
        mode=mode,
    )
    _append_model_current_spec_section(lines, current_spec=current_spec)
    _append_model_knowledge_cards_section(
        lines,
        knowledge_matches=knowledge_matches,
        handling_patterns=handling_patterns,
        insight_brief=insight_brief,
    )
    _append_model_experiment_session_memory_section(
        lines,
        recent_failures=recent_failures,
        validated_baselines=validated_baselines,
        existing_model_specs=existing_model_specs,
        factor_inventory=factor_inventory,
        single_factor_runs=single_factor_runs,
        session_memory_rows=session_memory_rows,
        source_anchors=source_anchors,
        spec_patch_hint=spec_patch_hint,
    )
    _append_model_warnings_section(lines, warnings=warnings)
    return lines


def _append_model_research_idea_section(
    lines: list[str],
    *,
    idea: str,
    mode: str,
    stage: str,
    next_stage: str | None,
) -> None:
    lines.extend(
        [
            "# Model Lab Idea Explorer Prompt",
            f"> Mode: {mode}",
            f"> Stage: {stage}"
            + (f" (next recommended: {next_stage})" if next_stage else ""),
            "",
            "## Research Idea",
            idea,
            "",
        ]
    )


def _append_model_system_contracts_section(
    lines: list[str],
    *,
    model_families: list[str],
    supported_preprocess: dict[str, object],
    supported_training: dict[str, object],
    supported_selection_metrics: list[str],
    supported_feature_importance: dict[str, object],
    mode: str,
) -> None:
    lines.extend(
        [
            "## System Contracts",
            f"- Supported model families: "
            f"{', '.join(model_families) if model_families else '(none)'}",
            "- Supported preprocess: missing_policy={missing}; scale={scale}; transform={transform}; group_scope={group}".format(
                missing=", ".join(_as_str_list(supported_preprocess.get("missing_policy"))) or "-",
                scale=", ".join(_as_str_list(supported_preprocess.get("scale_features"))) or "-",
                transform=", ".join(_as_str_list(supported_preprocess.get("cross_sectional_transform"))) or "-",
                group=", ".join(_as_str_list(supported_preprocess.get("cross_sectional_group_scope"))) or "-",
            ),
            "- Supported training windows: {windows}".format(
                windows=", ".join(_as_str_list(supported_training.get("window_type"))) or "-"
            ),
            "- Supported selection metrics: {metrics}".format(
                metrics=", ".join(supported_selection_metrics) or "-"
            ),
            "- Supported feature importance modes: {modes}".format(
                modes=", ".join(_as_str_list(supported_feature_importance.get("mode"))) or "-"
            ),
            (
                "- Contracts are discussion-only in kickoff; flag any extension you "
                "would need but do not require it to land in this stage."
                if mode == "start"
                else (
                    "- Stay inside current model-factor contracts unless explicitly "
                    "marked as code-change needed."
                )
            ),
            "",
            "## Current Spec Context",
        ]
    )


def _append_model_current_spec_section(
    lines: list[str],
    *,
    current_spec: dict[str, object],
) -> None:
    if current_spec.get("status") == "loaded":
        lines.extend(
            [
                f"- Spec name: {current_spec.get('name')}",
                f"- Factor name: {current_spec.get('factor_name')}",
                f"- Model family: {current_spec.get('model_family')}",
                f"- Feature count: {current_spec.get('feature_count')}",
                "- Feature sample: {features}{suffix}".format(
                    features=", ".join(
                        _as_str_list(current_spec.get("feature_columns_sample"))
                    )
                    or "-",
                    suffix=" ..." if current_spec.get("feature_columns_truncated") else "",
                ),
                f"- Target horizon: {current_spec.get('target_horizon')}",
                f"- Direction / rebalance / quantiles: {current_spec.get('direction')} / {current_spec.get('rebalance_frequency')} / {current_spec.get('n_quantiles')}",
                f"- Feature availability mode: {current_spec.get('feature_availability_mode')}",
                f"- Feature availability column: {current_spec.get('feature_availability_column')}",
                f"- Feature availability safety lag days: {current_spec.get('feature_availability_safety_lag_days')}",
                f"- Training window type: {current_spec.get('training_window_type')}",
                f"- Training details: {_compact_json(current_spec.get('training'))}",
                f"- Feature preprocess: {_compact_json(current_spec.get('feature_preprocess'))}",
                f"- Model selection enabled: {current_spec.get('model_selection_enabled')}",
                f"- Model selection metric: {current_spec.get('model_selection_metric')}",
                f"- Model selection details: {_compact_json(current_spec.get('model_selection'))}",
                f"- Feature importance: {_compact_json(current_spec.get('feature_importance'))}",
                f"- Neutralization: {_compact_json(current_spec.get('neutralization'))}",
                f"- Transaction cost: {_compact_json(current_spec.get('transaction_cost'))}",
            ]
        )
    else:
        lines.append("- Spec context unavailable; keep assumptions explicit.")


def _append_model_knowledge_cards_section(
    lines: list[str],
    *,
    knowledge_matches: list[dict[str, object]],
    handling_patterns: list[str],
    insight_brief: list[str],
) -> None:
    lines.extend(["", "## Knowledge Context"])
    if knowledge_matches:
        for idx, item in enumerate(knowledge_matches[:5], start=1):
            lines.append(
                f"- [K{idx}] {item.get('name')} ({item.get('type')}): {item.get('summary')}"
            )
            transferable_moves = _as_str_list(item.get("transferable_moves"))
            operative_claims = _as_str_list(item.get("operative_claims"))
            if transferable_moves:
                lines.append(
                    "  - transferable_moves: "
                    + "; ".join(transferable_moves[:3])
                )
            if operative_claims:
                lines.append(
                    "  - operative_claims (weak_hint_only): "
                    + "; ".join(operative_claims[:3])
                )
    else:
        lines.append("- No direct knowledge matches; treat this as greenfield.")
    for pattern in handling_patterns[:4]:
        lines.append(f"- handling: {pattern}")
    if insight_brief:
        lines.extend(["", "## Cross-card synthesis"])
        for brief in insight_brief[:6]:
            lines.append(f"- {brief}")


def _append_model_experiment_session_memory_section(
    lines: list[str],
    *,
    recent_failures: list[dict[str, object]],
    validated_baselines: list[dict[str, object]],
    existing_model_specs: list[dict[str, object]],
    factor_inventory: dict[str, object],
    single_factor_runs: list[dict[str, object]],
    session_memory_rows: list[dict[str, object]],
    source_anchors: list[dict[str, object]],
    spec_patch_hint: dict[str, object] | None,
) -> None:
    lines.extend(["", "## Experiment Context"])
    if recent_failures:
        for idx, item in enumerate(recent_failures[:3], start=1):
            lines.append(
                "- [F{idx}] run={run_id} status={status} outputs={outputs} reason={reason}".format(
                    idx=idx,
                    run_id=item.get("run_id"),
                    status=item.get("status"),
                    outputs=", ".join(_as_str_list(item.get("output_keys"))) or "-",
                    reason=item.get("failure_reason"),
                )
            )
    else:
        lines.append("- No recent failed runs found in local dist/ artifacts.")
    if validated_baselines:
        for idx, item in enumerate(validated_baselines[:3], start=1):
            lines.append(
                "- [E{idx}] run={run_id} model={model_family} mean_rank_ic={rank_ic} rank_ic_ir={rank_ir} ls_ir={ls_ir} net_ls_ir={net_ir} turnover={turnover} coverage={coverage} outputs={outputs}".format(
                    idx=idx,
                    run_id=item.get("run_id"),
                    model_family=item.get("model_family"),
                    rank_ic=item.get("mean_rank_ic"),
                    rank_ir=item.get("rank_ic_ir"),
                    ls_ir=item.get("long_short_ir"),
                    net_ir=item.get("cost_aware_long_short_ir"),
                    turnover=item.get("mean_long_short_turnover"),
                    coverage=item.get("coverage_mean"),
                    outputs=", ".join(_as_str_list(item.get("output_keys"))) or "-",
                )
            )
    else:
        lines.append("- No validated baseline runs found in local dist/ artifacts.")

    lines.extend(["", "## Existing Model And Factor Inventory"])
    if existing_model_specs:
        lines.append("### Model-factor specs")
        for idx, item in enumerate(existing_model_specs[:6], start=1):
            features = ", ".join(_as_str_list(item.get("feature_columns_sample"))[:8])
            lines.append(
                "- [S{idx}] {name} model={model} factor={factor} features={n} sample={sample} path={path}".format(
                    idx=idx,
                    name=item.get("name") or "-",
                    model=item.get("model_family") or "-",
                    factor=item.get("factor_name") or "-",
                    n=item.get("feature_count") or "-",
                    sample=features or "-",
                    path=item.get("path") or "-",
                )
            )
    else:
        lines.append("- No model-factor specs found under configs/real_cases/model_factor.")

    baseline_factors = _as_dict_list(factor_inventory.get("baseline_factors"))
    custom_factors = _as_dict_list(factor_inventory.get("custom_factors"))
    single_factor_cases = _as_dict_list(factor_inventory.get("single_factor_cases"))
    builtin_methods = _as_str_list(factor_inventory.get("builtin_methods"))
    lines.append("### Factor registry / cases")
    lines.append("- Built-in factor methods: " + (", ".join(builtin_methods) or "-"))
    if baseline_factors:
        for idx, item in enumerate(baseline_factors[:6], start=1):
            lines.append(
                "- [B{idx}] {name} family={family} method={method} required={required}".format(
                    idx=idx,
                    name=item.get("name") or "-",
                    family=item.get("family") or "-",
                    method=item.get("base_method") or "-",
                    required=", ".join(_as_str_list(item.get("required_columns"))) or "-",
                )
            )
    if custom_factors:
        for idx, item in enumerate(custom_factors[:6], start=1):
            lines.append(
                "- [C{idx}] {name} scope={scope} required={required} path={path}".format(
                    idx=idx,
                    name=item.get("name") or "-",
                    scope=item.get("scope") or "-",
                    required=", ".join(_as_str_list(item.get("required_columns"))) or "-",
                    path=item.get("path") or "-",
                )
            )
    if single_factor_cases:
        for idx, item in enumerate(single_factor_cases[:6], start=1):
            lines.append(
                "- [FCase{idx}] {name} factor={factor} method={method} horizon={horizon} path={path}".format(
                    idx=idx,
                    name=item.get("name") or "-",
                    factor=item.get("factor_name") or "-",
                    method=item.get("base_method") or "-",
                    horizon=item.get("target_horizon") or "-",
                    path=item.get("path") or "-",
                )
            )
    if single_factor_runs:
        lines.append("### Single-factor run references")
        for idx, item in enumerate(single_factor_runs[:5], start=1):
            lines.append(
                "- [FR{idx}] run={run_id} factor={factor} rank_ic={rank_ic} ls_ir={ls_ir} verdict={verdict}".format(
                    idx=idx,
                    run_id=item.get("run_id") or "-",
                    factor=item.get("factor_name") or "-",
                    rank_ic=item.get("mean_rank_ic"),
                    ls_ir=item.get("long_short_ir"),
                    verdict=item.get("factor_verdict") or "-",
                )
            )

    lines.extend(["", "## Session Memory"])
    if session_memory_rows:
        for idx, row in enumerate(session_memory_rows[:3], start=1):
            lines.append(
                "- [M{idx}] {created} stage={stage} mode={mode} responded={responded} lint_errors={lint_errors} idea={idea} patch={patch}".format(
                    idx=idx,
                    created=row.get("created_at_utc") or "-",
                    stage=row.get("stage") or "-",
                    mode=row.get("mode") or "-",
                    responded=row.get("has_response"),
                    lint_errors=row.get("has_lint_errors"),
                    idea=row.get("idea") or "-",
                    patch=row.get("patch_summary") or "-",
                )
            )
    else:
        lines.append("- No prior model-idea sessions available.")

    lines.extend(["", "## Code Anchors"])
    for item in source_anchors:
        path = str(item.get("path") or "").strip()
        focus = str(item.get("focus") or "").strip()
        if path:
            lines.append(f"- {path}: {focus}")

    if spec_patch_hint is not None:
        lines.extend(
            [
                "",
                "## Candidate Spec Patch Hint",
                f"- summary: {spec_patch_hint.get('summary')}",
                f"- requires_code_change: {spec_patch_hint.get('requires_code_change')}",
                "- patch_fields:",
                "```json",
                json.dumps(spec_patch_hint.get("patch_fields", {}), ensure_ascii=False, indent=2),
                "```",
            ]
        )


def _append_model_warnings_section(
    lines: list[str],
    *,
    warnings: list[str],
) -> None:
    if warnings:
        lines.extend(["", "## Warnings"])
        for warning in warnings:
            lines.append(f"- {warning}")


def _append_model_stage1_ledger_protocol(lines: list[str]) -> None:
    lines.extend(
        [
            "",
            "## Stage 1 ledger 协议",
            "本 prompt 只负责 model-lab 的 Stage 1：用 vault 素材起草模型改进机制。"
            "输出必须能转写为 `ledger_v1.yaml` + `retrieval_log.md`；不要给 kill 结论。",
            "- vault 是素材库，不是判决书；`transferable_moves` 是主要生成原料。",
            "- `operative_claims` 只能作为弱上下文 hint，不能触发 precedent kill。",
            "- 每条 model mechanism 至少写 `hypothesis` / `signal_sketch` / `data_needs`，"
            "并说明 touched contract surfaces。",
            "- `inspired_by` / `fusion_of` / `cross_domain_jump` 都是可选溯源字段；"
            "有来源就写，novel synthesis 无来源也合法。",
            "- 不要强制 lineage 或从来源卡继承 kill 条件。"
            "需要担心的点写成 `concern`，留给 Stage 3 数据验证。",
        ]
    )


def _append_model_mechanism_discovery_task(lines: list[str], *, mode: str) -> None:
    lines.extend(
        [
            "## 阶段声明",
            "你处于 model-lab 的 mechanism_discovery 阶段。",
            "目标是发现可能提升模型的结构性机制，而不是给出最终模型、最终 spec patch 或单一最佳方案。",
            "",
            "## 机制发现规则",
            "1. 只讨论模型改进机制：loss/regularization、feature interaction、target construction、sample weighting、training window、model selection。",
            "2. 每个机制必须写清 touched contract surfaces：model、feature_preprocess、feature_availability、training、model_selection、feature_importance。",
            "3. 必须说明该机制与当前 spec / baseline 的差异：是新机制、现有机制强化，还是只是参数调节。",
            "4. 必须保留不确定性，并写 concern：第一轮数据验证要重点观察什么。",
            "5. 禁止输出最终 spec patch、JSON patch、single best model、推荐版本或完整训练方案。",
        ]
    )
    _append_model_stage1_ledger_protocol(lines)
    lines.extend(
        [
            "6. 优先尝试跨领域迁移：把 daily PV、组合构建、稳健统计或其他 asset class 的 move 搬到模型机制。",
            "7. 尝试至少一个多卡 fusion 候选；弱点只标 concern，不删除。",
        ]
    )
    if mode == "start":
        lines.extend(
            [
                "8. start 模式允许提出 contract extension 作为讨论对象，但必须标记 needs-extension。",
                "9. 使用跨领域类比时，必须说明 transfer cost：金融 panel 数据的 PIT、稀疏样本、噪声标签或交易成本会如何破坏类比。",
            ]
        )
    elif mode == "constrained":
        lines.extend(
            [
                "8. constrained 模式输出 2-4 个机制候选，且每个候选必须说明当前 contracts 内可落地还是 requires_code_change。",
                "9. 如果一个候选只是在调 alpha/lambda/window/depth 等参数，必须改写成真正机制或标 `concern: parameter-only`。",
            ]
        )
    lines.extend(
        [
            "",
            "## 输出格式（严格遵守）",
            "[模型机制候选]",
            "### 机制 1",
            "- mechanism family: loss/regularization | feature interaction | target construction | sample weighting | training window | model selection",
            "- agent / data-generating story:",
            "- touched contract surfaces:",
            "- why it is not just parameter tuning:",
            "- evidence anchor: [Kx] / [Ex] / [Fx] / external analogy + transfer cost",
            "- inspired_by: [{card, what_i_took, cross_domain_jump}]（可选）",
            "- fusion_of: [[card_1, card_2]]（可选）",
            "- novel_delta:",
            "- concern:",
            "",
            "[实现假设草图]",
            "- 每个候选只写可讨论的实现轮廓，不写最终 patch。",
            "- 标记 in-contract / needs-extension / requires_code_change。",
            "",
            "[与当前 spec / baseline 的关系]",
            "- current baseline captured:",
            "- structural difference:",
            "- likely unchanged pieces:",
            "",
            "[不确定性与失败路径]",
            "- PIT / label leakage risk:",
            "- overfit / split fragility risk:",
            "- turnover / cost risk:",
            "- feature instability risk:",
            "",
            "[ledger_v1.yaml 草案]",
            "- mechanisms: hypothesis / inspired_by（可选）/ fusion_of（可选）/ novel_delta / signal_sketch / data_needs / concern",
            "- retrieval_log: surfaced_cards + transferable_moves + operative_claims weak hints",
        ]
    )


def _append_model_signal_mapping_task(lines: list[str], *, mode: str) -> None:
    lines.extend(
        [
            "## 阶段声明",
            "你处于 model-lab 的 signal_mapping 阶段。",
            "目标是把上游模型机制翻译成可测试 spec/run 版本；不是生成新机制，也不是选择赢家。",
            "",
            "## 映射规则",
            "1. 使用 Mechanism -> implication -> spec/data 三段映射；每个字段必须解释为什么必要。",
            "2. 对每个 required field 标注 role: necessary / decorative / risk control。",
            "3. necessary 字段必须写 remove-and-test 理由：删掉它会破坏哪条机制链。",
            "4. 每个版本都要说明控制哪些模型风险，哪些风险暂不控制以及原因。",
            "5. 输出 2-3 个可测试模型版本；禁止推荐 final pick。"
            "Stage 3 数据验证 / spec 审计才做最终 KILL / HOLD-FOR-AUDIT。",
        ]
    )
    _append_model_stage1_ledger_protocol(lines)
    if mode == "constrained":
        lines.extend(
            [
                "6. constrained 模式下，每个版本必须是当前 contracts 内可运行的最小变更，或明确 requires_code_change。",
                "7. 当前实现必须给出 binary alias-tag：是否只是 baseline linear/ridge 或 regularization-only 的换壳。",
            ]
        )
    lines.extend(
        [
            "",
            "## 模型风险控制清单",
        ]
    )
    for label, detail in MODEL_SIGNAL_RISK_CONTROLS:
        lines.append(f"- `{label}`: {detail}")
    lines.extend(
        [
            "",
            "## 输出格式（严格遵守）",
            "[Model Mechanism Mapping]",
            "### 机制 1",
            "- mechanism:",
            "- implication:",
            "- required_data_or_spec_fields:",
            "  - field: name | role: necessary/decorative/risk control | remove-and-test reason:",
            "- current-contract fit:",
            "",
            "[当前实现解释]",
            "- current implementation captures:",
            "- current implementation misses:",
            "- cannot disambiguate at current data/spec tier:",
        ]
    )
    lines.extend(["", "[模型风险控制]"])
    for label, detail in MODEL_SIGNAL_RISK_CONTROLS:
        lines.append(f"- `{label}`: 规避 / 显式控制 / 压力测试 / 暂不控制 - {detail} - one-line reason")
    lines.extend(
        [
            "",
            "[可测试模型版本]",
            "- v1: mechanism | minimal spec/run delta | controls | residual assumptions",
            "- v2: mechanism | minimal spec/run delta | controls | residual assumptions",
            "- v3: optional; only if structurally different",
            "",
            "[ledger_v1.yaml 草案]",
            "- mechanisms: 每条机制保留 hypothesis / inspired_by（可选）/ fusion_of（可选）/ "
            "novel_delta / signal_sketch / data_needs / concern",
            "- retrieval_log: surfaced_cards + transferable_moves + operative_claims weak hints",
            "",
            "[retrieval_log.md 草案]",
            "- surfaced_cards: 本阶段实际使用的 [Kx] / [Ex] / [Fx] 与用途",
            "- spec_dependency_notes: 当前 contracts、required fields 与 data tier 边界",
        ]
    )


def _append_model_validation_kill_tests_task(lines: list[str], *, mode: str) -> None:
    lines.extend(
        [
            "## 阶段声明",
            "你处于 model-lab 的 validation_kill_tests 阶段。",
            "目标不是证明模型有效，而是尽快判断它是否只是伪改进、泄漏、过拟合或成本假象。",
            "请用强审计口吻；回避性结论无效。",
            "",
            "## Alias / 问题归因靶子",
        ]
    )
    for label, detail in MODEL_VALIDATION_ALIAS_TARGETS:
        lines.append(f"- `{label}`: {detail}")
    lines.extend(
        [
            "",
            "## Kill Test 要求",
            "1. 数据与时间完整性：known_at / safety_lag / target horizon / 重叠窗口 / as-of 对齐。",
            "2. 训练与验证稳健性：walk-forward、purged split、年份/市场状态切分、窗口和 retrain 频率扰动。",
            "3. 特征与解释稳定性：top feature 稳定性、特征数量依赖、feature importance 漂移、冗余特征删除。",
            "4. 成本与组合影响：turnover、交易成本后 IR、行业/市值/流动性桶、组合约束敏感性。",
            "5. 如果 strict/constrained 模式下无法排除 hard kill 条件，只能输出 KILL。",
        ]
    )
    if mode == "constrained":
        lines.extend(
            [
                "6. constrained 模式必须给出 KILL 或 HOLD-FOR-AUDIT 二值判定。",
                "7. 每个 alias verdict 必须引用 [Kx] / [Ex] / [Fx] 或当前 spec/run 字段作为锚点。",
            ]
        )
    else:
        lines.append("6. 非 constrained 模式最终判定可为 KILL / HOLD / ITERATE / HOLD-FOR-AUDIT。")
    lines.extend(
        [
            "",
            "## 输出格式（严格遵守）",
            "[Alias / 问题归因审计]",
        ]
    )
    for label, detail in MODEL_VALIDATION_ALIAS_TARGETS:
        lines.append(f"- `{label}`: 显著风险 / 部分风险 / 不构成风险 - {detail} - anchor:")
    lines.extend(
        [
            "",
            "[数据与时间完整性]",
            "- PIT / known_at:",
            "- target leakage / overlapping label:",
            "- safety lag / as-of:",
            "- missing / survivorship / sample filter:",
            "",
            "[训练与验证稳健性]",
            "- split design:",
            "- window / retrain perturbation:",
            "- hyperparameter freedom:",
            "- regime and year stability:",
            "",
            "[特征与解释稳定性]",
            "- feature count dependence:",
            "- top feature stability:",
            "- redundancy / remove-and-test:",
            "- feature importance drift:",
            "",
            "[成本与组合影响]",
            "- turnover and transaction cost:",
            "- industry/size/liquidity bucket:",
            "- portfolio construction sensitivity:",
            "- capacity or implementation caveat:",
            "",
            "[最终判定]",
            "- verdict: KILL / HOLD / ITERATE / HOLD-FOR-AUDIT",
            "- hard kill trigger, if any:",
            "- next experiments if not killed:",
        ]
    )


def _append_model_lint_self_check(
    lines: list[str], *, stage: str, mode: str
) -> None:
    lines.extend(["", "## 输出自检（系统会用 lint 校验你的输出）"])
    for rule in describe_model_lint_contract(stage, mode=mode):
        lines.append(f"- {rule}")


def _resolve_optional_vault_root(
    vault_root: str | Path | None,
) -> tuple[Path | None, list[str]]:
    warnings: list[str] = []
    resolved = resolve_vault_root(vault_root)
    if resolved is None:
        return None, warnings
    if not resolved.exists():
        warnings.append(f"vault root does not exist: {resolved}")
        return None, warnings
    if not resolved.is_dir():
        warnings.append(f"vault root is not a directory: {resolved}")
        return None, warnings
    return resolved, warnings


def _derive_knowledge_handling_patterns(
    *,
    idea: str,
    knowledge_matches: list[dict[str, object]],
) -> list[str]:
    text_parts = [idea]
    for item in knowledge_matches[:6]:
        text_parts.append(str(item.get("summary") or ""))
        text_parts.append(str(item.get("snippet") or ""))
    haystack = " ".join(text_parts).lower()
    patterns: list[str] = []

    def _append_if_missing(text: str) -> None:
        if text not in patterns:
            patterns.append(text)

    if any(token in haystack for token in ("known_at", "as-of", "pit", "公告", "披露")):
        _append_if_missing("enforce known_at/as-of or safety_lag for PIT consistency")
    if any(token in haystack for token in ("winsor", "zscore", "rank", "截面")):
        _append_if_missing("define cross-sectional transform policy explicitly")
    if any(token in haystack for token in ("industry", "行业")):
        _append_if_missing("validate industry history before date_and_industry grouping")
    if any(token in haystack for token in ("turnover", "换手", "cost", "交易成本")):
        _append_if_missing("evaluate turnover-aware selection and cost-aware IR together")
    if any(token in haystack for token in ("overfit", "过拟合", "regularization", "正则")):
        _append_if_missing("prefer stronger regularization and stricter out-of-sample checks")
    if any(token in haystack for token in ("lightgbm", "xgboost", "gbdt", "tree")):
        _append_if_missing("tree models require robust outlier/missing handling discipline")
    return patterns


def _read_card_snippet(vault_root: Path, rel_path: str, max_chars: int = 220) -> str:
    normalized = str(rel_path or "").strip()
    if not normalized:
        return ""
    card_path = (vault_root / normalized).resolve()
    if not str(card_path).startswith(str(vault_root.resolve())):
        return ""
    if not card_path.exists() or not card_path.is_file():
        return ""
    try:
        text = card_path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""
    return text[:max_chars].strip()


def _read_card_frontmatter_field_items(
    vault_root: Path,
    rel_path: str,
    field_name: str,
    *,
    max_items: int = 4,
    max_chars: int = 180,
) -> list[str]:
    normalized = str(rel_path or "").strip()
    if not normalized:
        return []
    card_path = (vault_root / normalized).resolve()
    if not str(card_path).startswith(str(vault_root.resolve())):
        return []
    if not card_path.exists() or not card_path.is_file():
        return []
    try:
        text = card_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return _extract_frontmatter_field_items(
        text,
        field_name,
        max_items=max_items,
        max_chars=max_chars,
    )


def _extract_frontmatter_field_items(
    text: str,
    field_name: str,
    *,
    max_items: int,
    max_chars: int,
) -> list[str]:
    if not text.startswith("---\n"):
        return []
    try:
        _, frontmatter, _ = text.split("---\n", 2)
    except ValueError:
        return []
    target = field_name.strip()
    if not target:
        return []
    captured: list[str] = []
    in_field = False
    for line in frontmatter.splitlines():
        stripped = line.strip()
        if not in_field:
            if stripped.startswith(f"{target}:"):
                value = stripped.split(":", 1)[1].strip()
                if value:
                    captured.append(value)
                    break
                in_field = True
            continue
        if stripped and not line[:1].isspace() and not stripped.startswith("-"):
            break
        captured.append(stripped)
    if not captured:
        return []

    items: list[str] = []
    current: list[str] = []
    for line in captured:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("- "):
            if current:
                items.append(" ".join(current))
            current = [stripped[2:].strip()]
        else:
            current.append(stripped)
    if current:
        items.append(" ".join(current))
    if not items:
        items = [" ".join(captured)]
    return [
        _compact_prompt_text(item, max_chars=max_chars)
        for item in items[:max_items]
    ]


def _compact_prompt_text(text: str, *, max_chars: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[: max(max_chars - 3, 0)].rstrip() + "..."


def _infer_target_family(idea_l: str, supported_families: list[str]) -> str:
    supported = {item.strip().lower() for item in supported_families if item.strip()}
    for token, family in _MODEL_ALIAS_MAP:
        if token in idea_l and family in supported:
            return family
    return ""


def _idea_keywords(text: str) -> set[str]:
    keywords: set[str] = set()
    for token in re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", text):
        normalized = token.lower().strip()
        if len(normalized) <= 1:
            continue
        keywords.add(normalized)
        if re.fullmatch(r"[\u4e00-\u9fff]+", normalized) and len(normalized) >= 4:
            for idx in range(len(normalized) - 1):
                keywords.add(normalized[idx : idx + 2])
    return keywords


def _first_non_empty(values: list[str]) -> str:
    for value in values:
        text = str(value or "").strip()
        if text:
            return text
    return ""


def _to_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not (parsed == parsed):  # NaN
        return None
    return parsed


def _sort_float(value: object) -> float:
    parsed = _to_float_or_none(value)
    return parsed if parsed is not None else 0.0


def _first_metric_float(
    metrics: dict[str, object], *keys: str
) -> float | None:
    for key in keys:
        parsed = _to_float_or_none(metrics.get(key))
        if parsed is not None:
            return parsed
    return None


def _as_mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, dict) else {}


def _compact_json(value: object) -> str:
    if value is None:
        return "{}"
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _as_dict_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    rows: list[dict[str, object]] = []
    for item in value:
        if isinstance(item, dict):
            rows.append({str(k): v for k, v in item.items()})
    return rows


def _as_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    rows: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            rows.append(text)
    return rows


def _iter_session_paths(session_root: Path) -> list[Path]:
    paths = [path for path in session_root.glob("*.json") if path.is_file()]
    paths.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return paths


def _build_session_payload(
    *,
    session_id: str,
    created_at: dt.datetime,
    payload: dict[str, object],
    parent_session_id: str | None = None,
) -> dict[str, object]:
    constraint_report = _as_mapping(payload.get("constraint_report"))
    current_spec = _as_mapping(constraint_report.get("current_spec"))
    recommendations = _as_mapping(constraint_report.get("recommendations"))
    extras = _as_mapping(recommendations.get("extras"))
    spec_reference = str(
        current_spec.get("resolved_spec_path") or current_spec.get("requested_spec") or ""
    ).strip()
    spec_name = Path(spec_reference).name if spec_reference else ""
    return {
        "session_id": session_id,
        "created_at_utc": created_at.replace(microsecond=0).isoformat(),
        "parent_session_id": (
            _normalize_session_id(parent_session_id) if parent_session_id else None
        ),
        "idea": str(payload.get("idea") or "").strip(),
        "mode": str(payload.get("mode") or "").strip(),
        "stage": normalize_workflow_stage(
            str(payload.get("stage") or MECHANISM_DISCOVERY)
        ),
        "spec_name": spec_name,
        "spec_reference": spec_reference,
        "constraint_report": copy.deepcopy(constraint_report),
        "spec_patch_hint": copy.deepcopy(payload.get("spec_patch_hint")),
        "gpt_prompt": str(payload.get("gpt_prompt") or ""),
        "retrieval_diagnostics": copy.deepcopy(
            payload.get("retrieval_diagnostics")
        ),
        "memory_snapshot": copy.deepcopy(extras.get("session_memory")),
        "response": None,
        "response_sections": {},
        "responded_at_utc": None,
        "lint_report": None,
        "archived": False,
        "archived_at_utc": None,
    }


def _summarize_session_payload(
    payload: dict[str, object],
    *,
    path: Path | None = None,
) -> dict[str, object]:
    patch_hint = _as_mapping(payload.get("spec_patch_hint"))
    spec_name = str(payload.get("spec_name") or "").strip()
    if not spec_name:
        spec_reference = str(payload.get("spec_reference") or "").strip()
        if spec_reference:
            spec_name = Path(spec_reference).name
    lint_report = _as_mapping(payload.get("lint_report"))
    violations = _as_dict_list(lint_report.get("violations"))
    return {
        "session_id": str(payload.get("session_id") or "").strip(),
        "created_at_utc": str(payload.get("created_at_utc") or "").strip(),
        "parent_session_id": payload.get("parent_session_id"),
        "idea": str(payload.get("idea") or "").strip(),
        "mode": str(payload.get("mode") or "").strip(),
        "stage": str(payload.get("stage") or MECHANISM_DISCOVERY).strip(),
        "spec_name": spec_name,
        "patch_summary": str(patch_hint.get("summary") or "").strip(),
        "requires_code_change": bool(patch_hint.get("requires_code_change", False)),
        "archived": bool(payload.get("archived")),
        "archived_at_utc": payload.get("archived_at_utc"),
        "prompt_chars": len(str(payload.get("gpt_prompt") or "")),
        "has_response": bool(str(payload.get("response") or "").strip()),
        "has_lint_errors": bool(lint_report.get("has_errors", False)),
        "violation_codes": [
            str(item.get("code") or "").strip()
            for item in violations
            if str(item.get("code") or "").strip()
        ],
        "path": str(path) if path is not None else "",
    }


def _read_model_idea_session_file(path: Path) -> dict[str, object] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    return {str(key): value for key, value in raw.items()}


def _allocate_session_id(
    *,
    root: Path,
    idea: str,
    created_at: dt.datetime,
    preferred: str | None,
) -> str:
    if preferred:
        base = _normalize_session_id(preferred)
    else:
        base = _build_default_session_id(idea=idea, created_at=created_at)
    candidate = base
    suffix = 2
    while (root / f"{candidate}.json").exists():
        candidate = f"{base}-{suffix}"
        suffix += 1
    return candidate


def _normalize_session_id(raw: str) -> str:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("session_id must be non-empty")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,80}", text):
        raise ValueError(f"invalid session_id: {raw!r}")
    return text


def _build_default_session_id(*, idea: str, created_at: dt.datetime) -> str:
    signature = hashlib.sha1(idea.encode("utf-8")).hexdigest()[:8]
    return f"{created_at.strftime('%Y%m%d-%H%M%S')}-{signature}"
