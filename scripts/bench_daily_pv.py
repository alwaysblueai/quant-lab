"""Benchmark harness for the daily price-volume single-factor link path.

Times the end-to-end case run plus per-stage breakdowns. Intended to be run
before and after optimization PRs in the Phase 1/2 backlog so regressions show
up with a single diff.

Stages covered:
  - load_inputs         : load_standard_inputs (IO + feature cache prep)
  - pipeline            : run_single_factor_case (full end-to-end)

Output lands under outputs/benchmarks/bench_daily_pv/<run_id>/ with:
  - summary.json   (machine-readable; used by regression comparisons)
  - summary.txt    (human-readable)

Usage:
  python scripts/bench_daily_pv.py
  python scripts/bench_daily_pv.py --spec path/to/spec.yaml
  python scripts/bench_daily_pv.py --stages load_inputs
  python scripts/bench_daily_pv.py --repeat 3
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPEC = (
    REPO_ROOT
    / "configs"
    / "real_cases"
    / "single_factor"
    / "mom20_raw_reversal_5d_tushare_qfq_listed90.yaml"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "benchmarks" / "bench_daily_pv"
ALL_STAGES = ("load_inputs", "pipeline")


def _peak_rss_mb() -> float:
    """Peak resident-set size in MiB for the current process."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return float(ru.ru_maxrss) / (1024.0 * 1024.0)
    return float(ru.ru_maxrss) / 1024.0


def _git_sha_short() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        return out.decode("utf-8").strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


@dataclass
class StageResult:
    name: str
    ok: bool
    wall_seconds: float
    peak_rss_mb_after: float
    repeat_wall_seconds: list[float] = field(default_factory=list)
    error: str | None = None


@dataclass
class BenchReport:
    run_id: str
    spec_path: str
    python_version: str
    platform: str
    git_sha: str | None
    repeat: int
    stages: list[StageResult]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=False, default=str)


def _time_stage(
    name: str,
    fn: Callable[[], object],
    *,
    repeat: int,
) -> StageResult:
    walls: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 — capture any error to report
            return StageResult(
                name=name,
                ok=False,
                wall_seconds=float("nan"),
                peak_rss_mb_after=_peak_rss_mb(),
                repeat_wall_seconds=walls,
                error=f"{type(exc).__name__}: {exc}",
            )
        walls.append(time.perf_counter() - t0)
    # Report the minimum as wall_seconds (best-of-N is the least-noisy
    # single-number estimate; repeat_wall_seconds exposes the full list).
    return StageResult(
        name=name,
        ok=True,
        wall_seconds=min(walls),
        peak_rss_mb_after=_peak_rss_mb(),
        repeat_wall_seconds=walls,
    )


def _stage_load_inputs(spec_path: Path) -> Callable[[], object]:
    def run() -> object:
        from alpha_lab.real_cases.single_factor.pipeline import load_standard_inputs

        return load_standard_inputs(str(spec_path))

    return run


def _stage_pipeline(
    spec_path: Path,
    output_root: Path,
    evaluation_profile: str,
) -> Callable[[], object]:
    def run() -> object:
        from alpha_lab.real_cases.single_factor.pipeline import run_single_factor_case

        return run_single_factor_case(
            str(spec_path),
            output_root_dir=str(output_root),
            evaluation_profile=evaluation_profile,
        )

    return run


def _format_summary(report: BenchReport) -> str:
    lines: list[str] = []
    lines.append(f"Benchmark run : {report.run_id}")
    lines.append(f"Spec          : {report.spec_path}")
    lines.append(f"Python        : {report.python_version}")
    lines.append(f"Platform      : {report.platform}")
    lines.append(f"Git SHA       : {report.git_sha or '(unknown)'}")
    lines.append(f"Repeat        : {report.repeat}")
    lines.append("")
    header = f"  {'stage':<22} {'ok':<3} {'wall_s':>10} {'peak_rss_mb':>14}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for stage in report.stages:
        ok = "y" if stage.ok else "n"
        wall = "nan" if stage.wall_seconds != stage.wall_seconds else f"{stage.wall_seconds:.3f}"
        rss = f"{stage.peak_rss_mb_after:.1f}"
        lines.append(f"  {stage.name:<22} {ok:<3} {wall:>10} {rss:>14}")
        if stage.error:
            lines.append(f"    error: {stage.error}")
        if len(stage.repeat_wall_seconds) > 1:
            samples = ", ".join(f"{w:.3f}" for w in stage.repeat_wall_seconds)
            lines.append(f"    samples_s: [{samples}]")
    return "\n".join(lines) + "\n"


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Daily price-volume single-factor benchmark harness.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--spec",
        default=str(DEFAULT_SPEC),
        help="Path to the case spec to benchmark.",
    )
    parser.add_argument(
        "--stages",
        default=",".join(ALL_STAGES),
        help=f"Comma-separated subset of: {','.join(ALL_STAGES)}.",
    )
    parser.add_argument(
        "--evaluation-profile",
        default="default_research",
        help="Research evaluation profile for the pipeline stage.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run each stage N times (reported wall_s is the minimum).",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Root dir for summary.json / summary.txt outputs.",
    )
    parser.add_argument(
        "--pipeline-output-root",
        default=None,
        help=(
            "Where the pipeline stage writes its case artifacts. Defaults to a "
            "temp dir (artifacts discarded after the run)."
        ),
    )
    return parser.parse_args(argv)


def _resolve_stages(raw: str) -> list[str]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    unknown = [token for token in tokens if token not in ALL_STAGES]
    if unknown:
        raise SystemExit(f"unknown stages: {unknown}; choose from {list(ALL_STAGES)}")
    return tokens


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    spec_path = Path(args.spec).resolve()
    if not spec_path.exists():
        print(f"spec not found: {spec_path}", file=sys.stderr)
        return 2
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    stages = _resolve_stages(args.stages)

    run_id = time.strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.output_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pipeline_out_cm: tempfile.TemporaryDirectory[str] | None = None
    if args.pipeline_output_root is None:
        pipeline_out_cm = tempfile.TemporaryDirectory(prefix="bench_daily_pv_")
        pipeline_out_root = Path(pipeline_out_cm.name)
    else:
        pipeline_out_root = Path(args.pipeline_output_root).resolve()
        pipeline_out_root.mkdir(parents=True, exist_ok=True)

    try:
        results: list[StageResult] = []
        if "load_inputs" in stages:
            results.append(
                _time_stage(
                    "load_inputs",
                    _stage_load_inputs(spec_path),
                    repeat=args.repeat,
                )
            )
        if "pipeline" in stages:
            results.append(
                _time_stage(
                    "pipeline",
                    _stage_pipeline(spec_path, pipeline_out_root, args.evaluation_profile),
                    repeat=args.repeat,
                )
            )

        report = BenchReport(
            run_id=run_id,
            spec_path=str(spec_path),
            python_version=platform.python_version(),
            platform=platform.platform(),
            git_sha=_git_sha_short(),
            repeat=args.repeat,
            stages=results,
        )
        (out_dir / "summary.json").write_text(report.to_json() + "\n", encoding="utf-8")
        summary_text = _format_summary(report)
        (out_dir / "summary.txt").write_text(summary_text, encoding="utf-8")
        print(summary_text, end="")
        print(f"written: {out_dir / 'summary.json'}")
        return 0 if all(stage.ok for stage in results) else 1
    finally:
        if pipeline_out_cm is not None:
            pipeline_out_cm.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
