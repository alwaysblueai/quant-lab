"""Benchmark mean RankIC permutation-null kernel.

Compares:
1) legacy implementation (trial/date nested Python loops + rng.permutation)
2) optimized implementation (date-wise batched permutation kernel)

Usage:
  python scripts/bench_rank_ic_permutation_null.py
  python scripts/bench_rank_ic_permutation_null.py --dates 400 --assets 600 --repeat 3
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.evaluation import (
    _prepare_rank_ic_permutation_kernel,
    _resolve_merged_pairs,
    compute_mean_rank_ic_permutation_null,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "benchmarks" / "bench_rank_ic_permutation_null"


def _build_inputs(
    *,
    n_dates: int,
    n_assets: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-02", periods=n_dates, freq="B")
    assets = np.array([f"S{idx:05d}" for idx in range(n_assets)], dtype=object)

    date_values = np.tile(dates.to_numpy(), n_assets)
    asset_values = np.repeat(assets, n_dates)
    n_rows = n_dates * n_assets

    factor_values = rng.normal(loc=0.0, scale=1.0, size=n_rows)
    noise = rng.normal(loc=0.0, scale=1.0, size=n_rows)
    label_values = 0.08 * factor_values + noise

    factors = pd.DataFrame(
        {
            "date": pd.to_datetime(date_values),
            "asset": asset_values,
            "factor": "factor",
            "value": factor_values,
        }
    )
    labels = pd.DataFrame(
        {
            "date": pd.to_datetime(date_values),
            "asset": asset_values,
            "factor": "label",
            "value": label_values,
        }
    )
    return factors, labels


def _legacy_mean_rank_ic_permutation_null(
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    n_permutations: int,
    seed: int,
    min_assets_per_date: int = 3,
) -> tuple[float, np.ndarray]:
    merged = _resolve_merged_pairs(factors=factors, labels=labels, merged_pairs=None)
    kernel = _prepare_rank_ic_permutation_kernel(
        merged,
        min_assets_per_date=min_assets_per_date,
    )
    if kernel is None:
        return float("nan"), np.asarray([], dtype=float)
    observed_per_date, factor_centered, label_scaled = kernel
    observed_mean = float(np.mean(observed_per_date)) if observed_per_date.size else float("nan")

    n_trials = int(n_permutations)
    rng = np.random.default_rng(seed)
    n_dates = len(factor_centered)
    null = np.empty(n_trials, dtype=float)
    for trial in range(n_trials):
        trial_sum = 0.0
        for idx in range(n_dates):
            trial_sum += float(np.dot(rng.permutation(factor_centered[idx]), label_scaled[idx]))
        null[trial] = trial_sum / float(n_dates)
    return observed_mean, null[np.isfinite(null)]


def _time_once(fn) -> tuple[float, tuple[float, np.ndarray]]:
    t0 = time.perf_counter()
    result = fn()
    return time.perf_counter() - t0, result


@dataclass
class BenchSummary:
    n_dates: int
    n_assets: int
    n_permutations: int
    seed: int
    repeat: int
    batch_size: int
    legacy_seconds: list[float]
    optimized_seconds: list[float]
    legacy_best: float
    optimized_best: float
    speedup_ratio: float
    observed_mean_abs_diff: float
    null_mean_abs_diff: float
    null_std_abs_diff: float


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark RankIC permutation-null legacy vs optimized kernels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dates", type=int, default=250, help="Number of trading dates.")
    parser.add_argument("--assets", type=int, default=400, help="Number of assets per date.")
    parser.add_argument(
        "--permutations",
        type=int,
        default=200,
        help="Permutation draws for null distribution.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--repeat", type=int, default=3, help="Timing repeats (best-of-N).")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for optimized kernel.",
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where summary.json / summary.txt are written.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.dates <= 0 or args.assets <= 2 or args.permutations <= 0:
        raise SystemExit("dates>0, assets>2, permutations>0 are required")
    if args.repeat <= 0 or args.batch_size <= 0:
        raise SystemExit("repeat>0, batch-size>0 are required")

    factors, labels = _build_inputs(
        n_dates=int(args.dates),
        n_assets=int(args.assets),
        seed=int(args.seed),
    )

    legacy_times: list[float] = []
    optimized_times: list[float] = []
    legacy_output: tuple[float, np.ndarray] | None = None
    optimized_output: tuple[float, np.ndarray] | None = None

    for _ in range(int(args.repeat)):
        legacy_t, legacy_out = _time_once(
            lambda: _legacy_mean_rank_ic_permutation_null(
                factors,
                labels,
                n_permutations=int(args.permutations),
                seed=int(args.seed),
                min_assets_per_date=3,
            )
        )
        optimized_t, optimized_out = _time_once(
            lambda: compute_mean_rank_ic_permutation_null(
                factors,
                labels,
                n_permutations=int(args.permutations),
                seed=int(args.seed),
                min_assets_per_date=3,
                batch_size=int(args.batch_size),
            )
        )
        legacy_times.append(legacy_t)
        optimized_times.append(optimized_t)
        legacy_output = legacy_out
        optimized_output = optimized_out

    if legacy_output is None or optimized_output is None:
        raise RuntimeError("benchmark did not produce outputs")

    legacy_observed, legacy_null = legacy_output
    optimized_observed, optimized_null = optimized_output
    legacy_best = min(legacy_times)
    optimized_best = min(optimized_times)
    speedup = legacy_best / optimized_best if optimized_best > 0 else float("inf")

    summary = BenchSummary(
        n_dates=int(args.dates),
        n_assets=int(args.assets),
        n_permutations=int(args.permutations),
        seed=int(args.seed),
        repeat=int(args.repeat),
        batch_size=int(args.batch_size),
        legacy_seconds=legacy_times,
        optimized_seconds=optimized_times,
        legacy_best=legacy_best,
        optimized_best=optimized_best,
        speedup_ratio=float(speedup),
        observed_mean_abs_diff=float(abs(legacy_observed - optimized_observed)),
        null_mean_abs_diff=float(abs(float(np.mean(legacy_null)) - float(np.mean(optimized_null)))),
        null_std_abs_diff=float(
            abs(float(np.std(legacy_null, ddof=1)) - float(np.std(optimized_null, ddof=1)))
        ),
    )

    run_id = time.strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.output_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=False),
        encoding="utf-8",
    )

    text = (
        f"dates={summary.n_dates}, assets={summary.n_assets}, "
        f"permutations={summary.n_permutations}\n"
        f"repeat={summary.repeat}, seed={summary.seed}, batch_size={summary.batch_size}\n"
        f"legacy_best_s={summary.legacy_best:.6f}, "
        f"optimized_best_s={summary.optimized_best:.6f}\n"
        f"speedup={summary.speedup_ratio:.3f}x\n"
        f"observed_mean_abs_diff={summary.observed_mean_abs_diff:.3e}\n"
        f"null_mean_abs_diff={summary.null_mean_abs_diff:.3e}\n"
        f"null_std_abs_diff={summary.null_std_abs_diff:.3e}\n"
        f"legacy_samples={summary.legacy_seconds}\n"
        f"optimized_samples={summary.optimized_seconds}\n"
    )
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"summary saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
