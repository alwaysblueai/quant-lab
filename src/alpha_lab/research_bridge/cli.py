from __future__ import annotations

import argparse

from alpha_lab.research_bridge.exploration import ExplorationMap
from alpha_lab.research_bridge.graph_view import VaultGraph
from alpha_lab.research_bridge.models import AlphaLabDefaults, ProjectStatus, WritebackPolicy
from alpha_lab.research_bridge.service import (
    init_project,
    refresh_project_pack,
    scaffold_case,
    start_round,
    structure_candidates,
    summarize_run,
)


def build_bridge_parser(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        "Manage research-bridge project packs between quant-knowledge and alpha-lab."
    )
    commands = parser.add_subparsers(dest="bridge_action", required=True)

    init_cmd = commands.add_parser(
        "init-project",
        help="Create one 55_projects/<slug>/ research-bridge project skeleton.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    init_cmd.add_argument("--slug", required=True, help="Stable project slug.")
    init_cmd.add_argument("--title-zh", required=True, help="Chinese-first project title.")
    init_cmd.add_argument("--category", required=True, help="Project category, e.g. factor_family.")
    init_cmd.add_argument("--owner", required=True, help="Project owner.")
    init_cmd.add_argument("--market", required=True, help="Research market, e.g. ashare.")
    init_cmd.add_argument(
        "--frequency",
        required=True,
        help="Research frequency, e.g. daily.",
    )
    init_cmd.add_argument(
        "--chatgpt-project-name",
        required=True,
        help="Matching ChatGPT Project name.",
    )
    init_cmd.add_argument(
        "--origin-card",
        action="append",
        default=[],
        help="Origin card relative path inside quant-knowledge.",
    )
    init_cmd.add_argument(
        "--supporting-card",
        action="append",
        default=[],
        help="Supporting card relative path inside quant-knowledge.",
    )
    init_cmd.add_argument(
        "--failure-card",
        action="append",
        default=[],
        help="Failure-pattern card relative path inside quant-knowledge.",
    )
    init_cmd.add_argument(
        "--related-experiment-card",
        action="append",
        default=[],
        help="Related experiment card relative path inside quant-knowledge.",
    )
    init_cmd.add_argument(
        "--preferred-web-source",
        action="append",
        default=[],
        help="Preferred website/domain hints for supplemental web research.",
    )
    init_cmd.add_argument(
        "--data-source",
        default="tushare",
        help="Default alpha-lab data source.",
    )
    init_cmd.add_argument(
        "--slice-preset",
        default="standard",
        help="Default alpha-lab slice preset.",
    )
    init_cmd.add_argument("--universe", default="listed_90d", help="Default alpha-lab universe.")
    init_cmd.add_argument("--adjustment", default="qfq", help="Default alpha-lab price adjustment.")
    init_cmd.add_argument(
        "--evaluation-profile",
        default="exploratory_screening",
        help="Default Level 1/2 evaluation profile.",
    )
    init_cmd.add_argument(
        "--current-hypothesis",
        default="待定义",
        help="Initial project hypothesis.",
    )
    init_cmd.add_argument(
        "--current-focus",
        default="待开始第一轮讨论",
        help="Initial project focus.",
    )
    init_cmd.add_argument(
        "--next-action",
        default="刷新项目包并启动第一轮讨论",
        help="Initial next action.",
    )
    init_cmd.add_argument(
        "--max-research-level",
        type=int,
        default=2,
        help="Maximum research level for this project (1, 2, or 3).",
    )
    init_cmd.add_argument("--lifecycle", default="active", help="Initial lifecycle label.")
    init_cmd.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing project.yaml and stable pack in place.",
    )
    init_cmd.add_argument(
        "--vault-root",
        default=None,
        help="Quant-knowledge vault root. Defaults to OBSIDIAN_VAULT_PATH.",
    )

    refresh_cmd = commands.add_parser(
        "refresh-project-pack",
        help="Regenerate the stable project pack from project.yaml.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    refresh_cmd.add_argument("--project", required=True, help="Project slug.")
    refresh_cmd.add_argument("--vault-root", default=None, help="Quant-knowledge vault root.")

    round_cmd = commands.add_parser(
        "start-round",
        help="Create one legacy round context pack.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    round_cmd.add_argument("--project", required=True, help="Project slug.")
    round_cmd.add_argument("--topic", required=True, help="Round topic.")
    round_cmd.add_argument("--round-id", default=None, help="Optional round id override.")
    round_cmd.add_argument("--mode", default="standard", help="Round mode.")
    round_cmd.add_argument("--vault-root", default=None, help="Quant-knowledge vault root.")

    structure_cmd = commands.add_parser(
        "structure-candidates",
        help="Structure candidate ideas from one legacy round.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    structure_cmd.add_argument("--project", required=True, help="Project slug.")
    structure_cmd.add_argument("--round", required=True, help="Round id.")
    structure_cmd.add_argument(
        "--candidate", action="append", default=[], help="Optional candidate idea override."
    )
    structure_cmd.add_argument("--top-k", type=int, default=8, help="Semantic retrieval depth.")
    structure_cmd.add_argument("--limit", type=int, default=5, help="Max candidates to render.")
    structure_cmd.add_argument("--vault-root", default=None, help="Quant-knowledge vault root.")

    scaffold_case_cmd = commands.add_parser(
        "scaffold-case",
        help="Generate a case spec for alpha-lab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    scaffold_case_cmd.add_argument("--project", required=True, help="Project slug.")
    scaffold_case_cmd.add_argument("--case-name", required=True, help="Case/spec name.")
    scaffold_case_cmd.add_argument(
        "--case-type",
        default="factor_recipe",
        help="Case archetype. Currently supported: factor_recipe.",
    )
    scaffold_case_cmd.add_argument(
        "--factor-name",
        default=None,
        help="Optional factor label override.",
    )
    scaffold_case_cmd.add_argument(
        "--base-method",
        default="momentum",
        help="Base factor recipe method.",
    )
    scaffold_case_cmd.add_argument("--lookback", type=int, default=20, help="Recipe lookback.")
    scaffold_case_cmd.add_argument(
        "--skip-recent",
        type=int,
        default=5,
        help="Recipe skip_recent parameter.",
    )
    scaffold_case_cmd.add_argument(
        "--target-horizon",
        type=int,
        default=5,
        help="Forward-return horizon.",
    )
    scaffold_case_cmd.add_argument(
        "--rebalance-frequency",
        default="W",
        help="Case rebalance frequency.",
    )
    scaffold_case_cmd.add_argument("--direction", default="long", help="Case direction.")
    scaffold_case_cmd.add_argument(
        "--prices-path",
        default="./placeholder_prices.csv",
        help="Placeholder prices path.",
    )
    scaffold_case_cmd.add_argument(
        "--universe-path",
        default="./placeholder_universe.csv",
        help="Placeholder universe path.",
    )
    scaffold_case_cmd.add_argument(
        "--factor-path",
        default="./placeholder_factor.csv",
        help="Placeholder factor path.",
    )
    scaffold_case_cmd.add_argument("--vault-root", default=None, help="Quant-knowledge vault root.")

    summarize_cmd = commands.add_parser(
        "summarize-run",
        help="Turn one alpha-lab run directory into a slim run summary.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    summarize_cmd.add_argument("--project", required=True, help="Project slug.")
    summarize_cmd.add_argument("--run-root", required=True, help="alpha-lab run output directory.")
    summarize_cmd.add_argument("--vault-root", default=None, help="Quant-knowledge vault root.")

    factor_coverage_cmd = commands.add_parser(
        "factor-coverage",
        help="Show the mechanism × family factor coverage matrix from 90_computed/graph.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    factor_coverage_cmd.add_argument(
        "--vault-root",
        required=True,
        help="Quant-knowledge vault root containing 90_computed/graph.json.",
    )

    frontier_cmd = commands.add_parser(
        "explore-frontier",
        help="Show the current exploration frontier from 90_computed/exploration_map.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    frontier_cmd.add_argument(
        "--vault-root",
        required=True,
        help="Quant-knowledge vault root containing 90_computed/exploration_map.json.",
    )
    frontier_cmd.add_argument(
        "--priority",
        default=None,
        choices=["high", "medium", "low"],
        help="Optional priority filter.",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="alpha-lab bridge",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    build_bridge_parser(parser)
    args = parser.parse_args(argv)

    if args.bridge_action == "init-project":
        init_result = init_project(
            vault_root=args.vault_root,
            slug=args.slug,
            title_zh=args.title_zh,
            category=args.category,
            owner=args.owner,
            market=args.market,
            frequency=args.frequency,
            chatgpt_project_name=args.chatgpt_project_name,
            max_research_level=args.max_research_level,
            origin_cards=list(args.origin_card),
            supporting_cards=list(args.supporting_card),
            failure_cards=list(args.failure_card),
            related_experiment_cards=list(args.related_experiment_card),
            preferred_web_sources=list(args.preferred_web_source),
            alpha_lab_defaults=AlphaLabDefaults(
                data_source=args.data_source,
                slice_preset=args.slice_preset,
                universe=args.universe,
                adjustment=args.adjustment,
                evaluation_profile=args.evaluation_profile,
            ),
            writeback_policy=WritebackPolicy(),
            status=ProjectStatus(
                lifecycle=args.lifecycle,
                current_hypothesis=args.current_hypothesis,
                current_focus=args.current_focus,
                next_action=args.next_action,
            ),
            overwrite=bool(args.overwrite),
        )
        print("")
        print("  Workflow : bridge-init-project")
        print("  Status   : success")
        print(f"  Project  : {init_result.project.slug}")
        print(f"  Path     : {init_result.paths.project_dir}")
        return 0

    if args.bridge_action == "refresh-project-pack":
        refresh_result = refresh_project_pack(
            vault_root=args.vault_root,
            project_slug=args.project,
        )
        print("")
        print("  Workflow : bridge-refresh-project-pack")
        print("  Status   : success")
        print(f"  Project  : {refresh_result.project.slug}")
        print(f"  Path     : {refresh_result.paths.project_dir}")
        return 0

    if args.bridge_action == "start-round":
        round_result = start_round(
            vault_root=args.vault_root,
            project_slug=args.project,
            topic=args.topic,
            round_id=args.round_id,
            mode=args.mode,
        )
        print("")
        print("  Workflow : bridge-start-round")
        print("  Status   : success")
        print(f"  Project  : {round_result.project.slug}")
        print(f"  Round    : {round_result.round_id}")
        print(f"  Path     : {round_result.round_dir}")
        return 0

    if args.bridge_action == "structure-candidates":
        structure_result = structure_candidates(
            vault_root=args.vault_root,
            project_slug=args.project,
            round_id=args.round,
            candidate_ideas=list(args.candidate),
            top_k=args.top_k,
            limit=args.limit,
        )
        print("")
        print("  Workflow : bridge-structure-candidates")
        print("  Status   : success")
        print(f"  Project  : {structure_result.project.slug}")
        print(f"  Round    : {structure_result.round_id}")
        print(f"  Output   : {structure_result.structured_candidates_path}")
        print(f"  Draft    : {structure_result.knowledge_handoff_draft_path}")
        print(f"  Count    : {len(structure_result.candidates)}")
        return 0

    if args.bridge_action == "scaffold-case":
        scaffold_result = scaffold_case(
            vault_root=args.vault_root,
            project_slug=args.project,
            case_name=args.case_name,
            case_type=args.case_type,
            factor_name=args.factor_name,
            base_method=args.base_method,
            lookback=args.lookback,
            skip_recent=args.skip_recent,
            target_horizon=args.target_horizon,
            rebalance_frequency=args.rebalance_frequency,
            direction=args.direction,
            prices_path=args.prices_path,
            universe_path=args.universe_path,
            factor_path=args.factor_path,
        )
        print("")
        print("  Workflow : bridge-scaffold-case")
        print("  Status   : success")
        print(f"  Project  : {scaffold_result.project.slug}")
        print(f"  Current  : {scaffold_result.current_case_path}")
        return 0

    if args.bridge_action == "summarize-run":
        summarize_result = summarize_run(
            vault_root=args.vault_root,
            project_slug=args.project,
            run_root=args.run_root,
        )
        print("")
        print("  Workflow : bridge-summarize-run")
        print("  Status   : success")
        print(f"  Project  : {summarize_result.project.slug}")
        print(f"  Summary  : {summarize_result.summary_path}")
        print(f"  Latest   : {summarize_result.latest_path}")
        print(f"  Log      : {summarize_result.decision_log_path}")
        return 0

    if args.bridge_action == "factor-coverage":
        graph = VaultGraph.from_vault_root(args.vault_root)
        graph.build(vault_root=args.vault_root)
        coverage = graph.coverage_by_type().get(
            "factor", {"annotated": 0, "unannotated": 0, "total": 0}
        )
        matrix = graph.mechanism_family_matrix()
        print("")
        print("  Workflow : bridge-factor-coverage")
        print("  Status   : success")
        print(
            "  Factors  : "
            f"annotated={coverage.get('annotated', 0)} "
            f"unannotated={coverage.get('unannotated', 0)} "
            f"total={coverage.get('total', 0)}"
        )
        print("")
        for mechanism in sorted(matrix):
            families = matrix[mechanism]
            family_parts = [f"{family}:{len(names)}" for family, names in sorted(families.items())]
            rendered = ", ".join(family_parts) if family_parts else "none"
            print(f"  {mechanism:<14} {rendered}")
        return 0

    if args.bridge_action == "explore-frontier":
        exploration = ExplorationMap.from_vault_root(args.vault_root)
        exploration.build(vault_root=args.vault_root)
        frontier = exploration.frontier(priority=args.priority)
        print("")
        print("  Workflow : bridge-explore-frontier")
        print("  Status   : success")
        print(f"  Count    : {len(frontier)}")
        print("")
        for item in frontier:
            print(f"  {item.priority:<6} {item.direction}")
            print(f"           reason: {item.reason}")
        return 0

    parser.error(f"unsupported bridge command: {args.bridge_action!r}")


def resolved_argv_for_bridge(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = [str(args.bridge_action)]
    for key, value in vars(args).items():
        if key in {"top_command", "bridge_action"}:
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                forwarded.append(f"--{key.replace('_', '-')}")
            continue
        if isinstance(value, list):
            for item in value:
                forwarded.extend([f"--{key.replace('_', '-')}", str(item)])
            continue
        forwarded.extend([f"--{key.replace('_', '-')}", str(value)])
    return forwarded
