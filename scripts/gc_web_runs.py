#!/usr/bin/env python3
"""Garbage-collect old web UI run directories under outputs/real_cases/_web_runs/.

Each web UI submission writes its per-run output bundle (manifests, metrics,
artifacts, intermediate CSVs, tearsheets) under
``outputs/real_cases/_web_runs/<run_id>/<case_name>/``. The big shared
dataset cache lives at ``outputs/real_cases/_model_factor_shared_cache/``
and is reused across runs via the ``--cache-root-dir`` argument propagated
by the web launcher (see ``_resolve_preparation_cache_dir`` for the
contract). Per-run directories themselves are typically tens of MB; this
script trims them when they accumulate.

Historical note: prior to the Phase 2 web_unified hardening, the dataset
cache landed under each per-run directory (``_web_runs/<run_id>/
_model_factor_cache/``) at ~4.5GB per submission. That leak is fixed and
guarded by ``test_resolve_preparation_cache_dir_rejects_web_runs_fallback``
plus ``test_preparation_cache_key_is_invariant_to_output_dir``.

Examples::

    # Preview: keep latest 3 runs, delete older
    uv run python scripts/gc_web_runs.py --keep-last 3

    # Preview: delete runs older than 7 days
    uv run python scripts/gc_web_runs.py --older-than 7

    # Apply (combines both filters; a run survives only if it satisfies BOTH)
    uv run python scripts/gc_web_runs.py --keep-last 3 --older-than 7 --apply
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

WEB_RUNS_RELATIVE = Path("outputs/real_cases/_web_runs")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--keep-last", type=int, default=None, metavar="N",
                        help="Keep the N most recently modified runs.")
    parser.add_argument("--older-than", type=float, default=None, metavar="DAYS",
                        help="Delete runs whose mtime is older than DAYS days.")
    parser.add_argument("--apply", action="store_true",
                        help="Actually delete. Without this flag, dry-run only.")
    parser.add_argument(
        "--root", type=Path, default=None,
        help=f"Override web_runs dir (default: <repo>/{WEB_RUNS_RELATIVE}).",
    )
    return parser.parse_args()


def _resolve_root(override: Path | None) -> Path:
    if override is not None:
        return override.resolve()
    repo_root = Path(__file__).resolve().parent.parent
    return (repo_root / WEB_RUNS_RELATIVE).resolve()


def _dir_size_bytes(path: Path) -> int:
    total = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                pass
    return total


def _format_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024 or unit == "TB":
            return f"{num_bytes:.1f} {unit}"
        num_bytes //= 1024  # type: ignore[assignment]
    return f"{num_bytes:.1f} TB"


def main() -> int:
    args = _parse_args()
    if args.keep_last is None and args.older_than is None:
        print("error: must specify at least one of --keep-last / --older-than", file=sys.stderr)
        return 2

    root = _resolve_root(args.root)
    if not root.is_dir():
        print(f"web_runs dir not found: {root}")
        return 0

    runs = sorted(
        (p for p in root.iterdir() if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        print(f"no runs under {root}")
        return 0

    now = time.time()
    age_cutoff = (args.older_than * 86400) if args.older_than is not None else None

    keep: list[Path] = []
    drop: list[Path] = []
    for idx, run in enumerate(runs):
        keep_by_count = args.keep_last is None or idx < args.keep_last
        keep_by_age = age_cutoff is None or (now - run.stat().st_mtime) <= age_cutoff
        if keep_by_count and keep_by_age:
            keep.append(run)
        else:
            drop.append(run)

    print(f"web_runs root: {root}")
    print(f"total runs: {len(runs)}  keep: {len(keep)}  drop: {len(drop)}")

    drop_total = 0
    for run in drop:
        size = _dir_size_bytes(run)
        drop_total += size
        age_days = (now - run.stat().st_mtime) / 86400
        print(f"  drop {run.name}  {_format_size(size):>10}  age={age_days:.1f}d")

    print(f"reclaimable: {_format_size(drop_total)}")

    if not args.apply:
        print("dry-run only. Re-run with --apply to delete.")
        return 0

    if not drop:
        return 0

    for run in drop:
        shutil.rmtree(run)
    print(f"deleted {len(drop)} run(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
