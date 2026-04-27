"""A/B bench comparator with sigma-based noise gating.

Compares two groups of bench `summary.json` runs (pre vs post), then reports
whether each stage is faster/slower than noise using:

    threshold = sigma_k * sqrt(pre_std^2 + post_std^2)

Verdict per stage:
    - faster_than_noise
    - neutral
    - slower_than_noise

Examples:
    python scripts/ab_compare.py \
      --pre-runs 20260424T195210,20260424T195240,20260424T195310,20260424T195339,20260424T195409 \
      --post-runs 20260424T195456,20260424T195526,20260424T195555,20260424T195623,20260424T195653 \
      --stage train

    python scripts/ab_compare.py \
      --pre outputs/benchmarks/ab_test/pre_step1 \
      --post outputs/benchmarks/ab_test/step1 \
      --stage all \
      --sigma-k 2 \
      --min-runs 5
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEARCH_ROOTS: tuple[Path, ...] = (
    REPO_ROOT / "outputs" / "benchmarks",
    REPO_ROOT / "outputs",
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "benchmarks" / "ab_compare"


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    size: str | None
    total_wall_seconds: float
    stage_seconds: dict[str, float]
    source_summary_path: str


@dataclass(frozen=True)
class GroupStats:
    n_runs: int
    values: list[float]
    mean_seconds: float
    std_seconds: float


@dataclass(frozen=True)
class StageComparison:
    stage: str
    pre: GroupStats
    post: GroupStats
    delta_seconds: float
    delta_ratio: float
    combined_sigma: float
    threshold_seconds: float
    verdict: str


def _split_csv_items(raw_items: list[str] | None) -> list[str]:
    items: list[str] = []
    for raw in raw_items or []:
        for token in raw.split(","):
            value = token.strip()
            if value:
                items.append(value)
    return items


def _resolve_group_specs(
    *,
    single: str | None,
    many: list[str] | None,
    group_name: str,
) -> list[str]:
    csv_items = _split_csv_items(many)
    if single and csv_items:
        raise ValueError(
            f"{group_name}: use either --{group_name} or --{group_name}-runs, not both"
        )
    if single:
        return [single]
    if csv_items:
        return csv_items
    raise ValueError(
        f"{group_name}: missing run input, provide --{group_name} or --{group_name}-runs"
    )


def _candidate_paths_by_run_id(run_id: str, search_roots: list[Path]) -> list[Path]:
    matches: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for candidate in root.glob(f"**/{run_id}/summary.json"):
            if candidate.parent.name == run_id:
                matches.append(candidate.resolve())
    dedup = sorted({str(path): path for path in matches}.values(), key=lambda p: str(p))
    return dedup


def _resolve_spec_to_summary_paths(spec: str, search_roots: list[Path]) -> list[Path]:
    path = Path(spec)
    if path.exists():
        if path.is_file():
            return [path.resolve()]
        summary = path / "summary.json"
        if summary.exists():
            return [summary.resolve()]
        nested = sorted(path.glob("**/summary.json"))
        if nested:
            return [candidate.resolve() for candidate in nested]
        raise ValueError(f"path exists but no summary.json found: {path}")

    matches = _candidate_paths_by_run_id(spec, search_roots)
    if not matches:
        roots = ", ".join(str(root) for root in search_roots)
        raise ValueError(f"run id not found: {spec} (search_roots: {roots})")
    if len(matches) > 1:
        preview = ", ".join(str(path) for path in matches[:5])
        raise ValueError(
            f"run id is ambiguous: {spec}. matched {len(matches)} summary files: {preview}"
        )
    return matches


def _resolve_group_paths(specs: list[str], search_roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for spec in specs:
        paths.extend(_resolve_spec_to_summary_paths(spec, search_roots))
    dedup = sorted({str(path): path for path in paths}.values(), key=lambda p: str(p))
    return dedup


def _require_float(value: Any, *, field_name: str, source: Path) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source}: field {field_name!r} is not numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{source}: field {field_name!r} must be finite, got {parsed!r}")
    return parsed


def _load_summary(path: Path) -> RunSummary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    run_id = str(payload.get("run_id") or path.parent.name)
    size_raw = payload.get("size")
    size = str(size_raw) if size_raw is not None else None
    total = _require_float(
        payload.get("total_wall_seconds"),
        field_name="total_wall_seconds",
        source=path,
    )

    stages = payload.get("stages")
    if not isinstance(stages, list):
        raise ValueError(f"{path}: stages must be a list")

    stage_seconds: dict[str, float] = {}
    for entry in stages:
        if not isinstance(entry, dict):
            continue
        name = str(entry.get("name") or "").strip()
        if not name:
            continue
        stage_seconds[name] = _require_float(
            entry.get("wall_seconds"),
            field_name=f"stages[{name}].wall_seconds",
            source=path,
        )

    if not stage_seconds:
        raise ValueError(f"{path}: no valid stage wall_seconds found")

    return RunSummary(
        run_id=run_id,
        size=size,
        total_wall_seconds=total,
        stage_seconds=stage_seconds,
        source_summary_path=str(path),
    )


def _group_stats(values: list[float]) -> GroupStats:
    return GroupStats(
        n_runs=len(values),
        values=values,
        mean_seconds=mean(values),
        std_seconds=stdev(values) if len(values) > 1 else 0.0,
    )


def _stage_order_from_runs(runs: list[RunSummary]) -> list[str]:
    # Keep ordering stable by using first-run stage order.
    first_order = list(runs[0].stage_seconds.keys())
    shared = set(first_order)
    for run in runs[1:]:
        shared &= set(run.stage_seconds.keys())
    return [stage for stage in first_order if stage in shared]


def _resolve_stages(
    *,
    stage_arg: str,
    pre_runs: list[RunSummary],
    post_runs: list[RunSummary],
) -> list[str]:
    common_stages = set(_stage_order_from_runs(pre_runs)) & set(_stage_order_from_runs(post_runs))
    if not common_stages:
        raise ValueError("no common stages between pre and post groups")

    if stage_arg.strip().lower() == "all":
        order = _stage_order_from_runs(pre_runs)
        return [stage for stage in order if stage in common_stages]

    requested = [token.strip() for token in stage_arg.split(",") if token.strip()]
    if not requested:
        raise ValueError("--stage is empty")
    missing = [stage for stage in requested if stage not in common_stages]
    if missing:
        raise ValueError(f"requested stages not present in both groups: {missing}")
    return requested


def _compare_stage(
    *,
    stage: str,
    pre_runs: list[RunSummary],
    post_runs: list[RunSummary],
    sigma_k: float,
) -> StageComparison:
    pre_values = [run.stage_seconds[stage] for run in pre_runs]
    post_values = [run.stage_seconds[stage] for run in post_runs]
    pre_stats = _group_stats(pre_values)
    post_stats = _group_stats(post_values)
    delta = post_stats.mean_seconds - pre_stats.mean_seconds
    delta_ratio = delta / pre_stats.mean_seconds if pre_stats.mean_seconds > 0 else float("nan")
    combined_sigma = math.sqrt((pre_stats.std_seconds ** 2) + (post_stats.std_seconds ** 2))
    threshold = sigma_k * combined_sigma
    if abs(delta) <= threshold:
        verdict = "neutral"
    elif delta < 0:
        verdict = "faster_than_noise"
    else:
        verdict = "slower_than_noise"
    return StageComparison(
        stage=stage,
        pre=pre_stats,
        post=post_stats,
        delta_seconds=delta,
        delta_ratio=delta_ratio,
        combined_sigma=combined_sigma,
        threshold_seconds=threshold,
        verdict=verdict,
    )


def _build_markdown(
    *,
    sigma_k: float,
    min_runs: int,
    stage_filter: str,
    pre_specs: list[str],
    post_specs: list[str],
    pre_runs: list[RunSummary],
    post_runs: list[RunSummary],
    total_comparison: StageComparison,
    stage_rows: list[StageComparison],
) -> str:
    lines: list[str] = []
    lines.append("# A/B 性能对比（sigma 门限）")
    lines.append("")
    lines.append(f"- sigma_k: `{sigma_k}`")
    lines.append(f"- min_runs: `{min_runs}`")
    lines.append(f"- stage_filter: `{stage_filter}`")
    lines.append(f"- pre specs: `{', '.join(pre_specs)}`")
    lines.append(f"- post specs: `{', '.join(post_specs)}`")
    lines.append("")
    lines.append("| scope | pre mean±σ | post mean±σ | delta_s | delta_% | kσ阈值 | verdict |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")

    def _fmt_group(stats: GroupStats) -> str:
        return f"{stats.mean_seconds:.4f}±{stats.std_seconds:.4f} (n={stats.n_runs})"

    lines.append(
        "| total | "
        f"{_fmt_group(total_comparison.pre)} | "
        f"{_fmt_group(total_comparison.post)} | "
        f"{total_comparison.delta_seconds:+.4f} | "
        f"{total_comparison.delta_ratio * 100:+.2f}% | "
        f"{total_comparison.threshold_seconds:.4f} | "
        f"{total_comparison.verdict} |"
    )
    for row in stage_rows:
        lines.append(
            f"| {row.stage} | "
            f"{_fmt_group(row.pre)} | "
            f"{_fmt_group(row.post)} | "
            f"{row.delta_seconds:+.4f} | "
            f"{row.delta_ratio * 100:+.2f}% | "
            f"{row.threshold_seconds:.4f} | "
            f"{row.verdict} |"
        )
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two bench run groups with sigma-based noise threshold.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pre", default=None, help="Pre group: run dir / summary path / run_id.")
    parser.add_argument(
        "--pre-runs",
        action="append",
        default=None,
        help="Pre group list (comma-separated), each item can be path or run_id.",
    )
    parser.add_argument("--post", default=None, help="Post group: run dir / summary path / run_id.")
    parser.add_argument(
        "--post-runs",
        action="append",
        default=None,
        help="Post group list (comma-separated), each item can be path or run_id.",
    )
    parser.add_argument(
        "--stage",
        default="all",
        help="Stage filter: 'all' or comma-separated stage names, e.g. train or load,train.",
    )
    parser.add_argument(
        "--sigma-k",
        type=float,
        default=2.0,
        help="Sigma multiplier for threshold.",
    )
    parser.add_argument("--min-runs", type=int, default=5, help="Minimum runs required per group.")
    parser.add_argument(
        "--search-root",
        action="append",
        default=None,
        help="Search roots for resolving run_id to summary.json. Can be repeated.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output root for compare artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write compare.json/md directly to this directory (overrides --output-root/--tag).",
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional suffix for output folder name.",
    )
    parser.add_argument(
        "--fail-on-slower",
        action="store_true",
        help="Exit non-zero if any stage verdict is slower_than_noise.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.sigma_k <= 0:
        raise SystemExit("--sigma-k must be > 0")
    if args.min_runs < 1:
        raise SystemExit("--min-runs must be >= 1")

    pre_specs = _resolve_group_specs(single=args.pre, many=args.pre_runs, group_name="pre")
    post_specs = _resolve_group_specs(single=args.post, many=args.post_runs, group_name="post")

    search_roots_raw = args.search_root or [str(path) for path in DEFAULT_SEARCH_ROOTS]
    search_roots = [Path(raw).resolve() for raw in search_roots_raw]

    pre_paths = _resolve_group_paths(pre_specs, search_roots)
    post_paths = _resolve_group_paths(post_specs, search_roots)
    pre_runs = [_load_summary(path) for path in pre_paths]
    post_runs = [_load_summary(path) for path in post_paths]

    if len(pre_runs) < args.min_runs:
        raise SystemExit(
            f"pre group has {len(pre_runs)} runs, below min_runs={args.min_runs}"
        )
    if len(post_runs) < args.min_runs:
        raise SystemExit(
            f"post group has {len(post_runs)} runs, below min_runs={args.min_runs}"
        )

    stage_names = _resolve_stages(stage_arg=args.stage, pre_runs=pre_runs, post_runs=post_runs)

    total_comparison = _compare_stage(
        stage="total",
        pre_runs=[
            RunSummary(
                run_id=run.run_id,
                size=run.size,
                total_wall_seconds=run.total_wall_seconds,
                stage_seconds={"total": run.total_wall_seconds},
                source_summary_path=run.source_summary_path,
            )
            for run in pre_runs
        ],
        post_runs=[
            RunSummary(
                run_id=run.run_id,
                size=run.size,
                total_wall_seconds=run.total_wall_seconds,
                stage_seconds={"total": run.total_wall_seconds},
                source_summary_path=run.source_summary_path,
            )
            for run in post_runs
        ],
        sigma_k=float(args.sigma_k),
    )
    stage_rows = [
        _compare_stage(
            stage=stage,
            pre_runs=pre_runs,
            post_runs=post_runs,
            sigma_k=float(args.sigma_k),
        )
        for stage in stage_names
    ]

    payload = {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_ab_compare",
        "sigma_k": float(args.sigma_k),
        "min_runs": int(args.min_runs),
        "stage_filter": str(args.stage),
        "search_roots": [str(path) for path in search_roots],
        "pre_specs": pre_specs,
        "post_specs": post_specs,
        "pre_runs": [asdict(run) for run in pre_runs],
        "post_runs": [asdict(run) for run in post_runs],
        "total": asdict(total_comparison),
        "stages": [asdict(row) for row in stage_rows],
    }
    markdown = _build_markdown(
        sigma_k=float(args.sigma_k),
        min_runs=int(args.min_runs),
        stage_filter=str(args.stage),
        pre_specs=pre_specs,
        post_specs=post_specs,
        pre_runs=pre_runs,
        post_runs=post_runs,
        total_comparison=total_comparison,
        stage_rows=stage_rows,
    )

    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
    else:
        run_id = time.strftime("%Y%m%dT%H%M%S")
        if args.tag:
            run_id = f"{run_id}_{args.tag}"
        out_dir = Path(args.output_root).resolve() / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "compare.json"
    md_path = out_dir / "compare.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"written: {json_path}")
    print(f"written: {md_path}")

    if args.fail_on_slower and any(row.verdict == "slower_than_noise" for row in stage_rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
