"""Micro-benchmark for `adjust_for_dividends`.

Compares the optimized implementation against the legacy row-iteration logic on
synthetic daily price panels with deterministic random seeds.

Usage:
  python scripts/bench_adjust_for_dividends.py
  python scripts/bench_adjust_for_dividends.py --assets 3000 --days 1200 --repeat 5
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.data_quality.corporate_actions import adjust_for_dividends

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "benchmarks" / "bench_adjust_for_dividends"


def _legacy_adjust_for_dividends(
    prices_df: pd.DataFrame,
    dividend_df: pd.DataFrame,
) -> pd.DataFrame:
    df = prices_df.copy()
    df = df.sort_values(["asset", "date"]).reset_index(drop=True)
    for _, row in dividend_df.iterrows():
        asset = row["asset"]
        ex_date = row["date"]
        div = row["dividend_per_share"]

        mask = (df["asset"] == asset) & (df["date"] < ex_date)
        pre_ex = df[(df["asset"] == asset) & (df["date"] < ex_date)]
        if pre_ex.empty:
            continue
        prev_close = pre_ex.sort_values("date").iloc[-1]["close"]
        if prev_close <= 0:
            continue
        ratio = 1.0 - div / prev_close
        if ratio <= 0:
            continue
        df.loc[mask, "close"] = df.loc[mask, "close"] * ratio
    return df


def _build_synthetic_panel(
    *,
    n_assets: int,
    n_days: int,
    events_per_asset: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")
    assets = np.array([f"S{idx:05d}" for idx in range(n_assets)], dtype=object)

    dates_tile = np.tile(dates.to_numpy(), n_assets)
    assets_rep = np.repeat(assets, n_days)

    daily_ret = rng.normal(loc=0.0002, scale=0.02, size=n_assets * n_days)
    grouped_ret = daily_ret.reshape(n_assets, n_days)
    close = 10.0 * np.exp(np.cumsum(grouped_ret, axis=1))

    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(dates_tile),
            "asset": assets_rep,
            "close": close.reshape(-1),
        }
    )

    event_rows: list[tuple[str, np.datetime64, float]] = []
    for asset in assets:
        ex_positions = rng.integers(1, n_days, size=events_per_asset)
        ex_positions.sort()
        for pos in ex_positions:
            event_rows.append(
                (
                    asset,
                    dates.to_numpy()[int(pos)],
                    float(rng.uniform(0.01, 0.20)),
                )
            )

    dividends = pd.DataFrame(
        event_rows,
        columns=["asset", "date", "dividend_per_share"],
    )
    return prices, dividends


def _time_once(fn: Callable[[], object]) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


@dataclass
class BenchSummary:
    n_assets: int
    n_days: int
    events_per_asset: int
    seed: int
    repeat: int
    legacy_seconds: list[float]
    optimized_seconds: list[float]
    legacy_best: float
    optimized_best: float
    speedup_ratio: float
    max_abs_close_diff: float


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark adjust_for_dividends against legacy implementation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--assets", type=int, default=2000, help="Number of assets.")
    parser.add_argument("--days", type=int, default=750, help="Number of dates per asset.")
    parser.add_argument(
        "--events-per-asset",
        type=int,
        default=2,
        help="Dividend events per asset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--repeat", type=int, default=3, help="Timing repeats (best-of-N).")
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where summary.json / summary.txt are written.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.assets <= 0 or args.days <= 1 or args.events_per_asset <= 0 or args.repeat <= 0:
        raise SystemExit("assets>0, days>1, events-per-asset>0, repeat>0 are required")

    prices, dividends = _build_synthetic_panel(
        n_assets=args.assets,
        n_days=args.days,
        events_per_asset=args.events_per_asset,
        seed=args.seed,
    )

    legacy_times: list[float] = []
    optimized_times: list[float] = []
    for _ in range(args.repeat):
        legacy_times.append(_time_once(lambda: _legacy_adjust_for_dividends(prices, dividends)))
        optimized_times.append(_time_once(lambda: adjust_for_dividends(prices, dividends)))

    legacy_out = _legacy_adjust_for_dividends(prices, dividends)
    optimized_out = adjust_for_dividends(prices, dividends)
    max_abs_diff = float(
        np.max(np.abs(legacy_out["close"].to_numpy() - optimized_out["close"].to_numpy()))
    )

    legacy_best = min(legacy_times)
    optimized_best = min(optimized_times)
    speedup = legacy_best / optimized_best if optimized_best > 0 else float("inf")
    summary = BenchSummary(
        n_assets=args.assets,
        n_days=args.days,
        events_per_asset=args.events_per_asset,
        seed=args.seed,
        repeat=args.repeat,
        legacy_seconds=legacy_times,
        optimized_seconds=optimized_times,
        legacy_best=legacy_best,
        optimized_best=optimized_best,
        speedup_ratio=speedup,
        max_abs_close_diff=max_abs_diff,
    )

    run_id = time.strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.output_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(asdict(summary), indent=2, sort_keys=False),
        encoding="utf-8",
    )
    text = (
        "assets="
        f"{summary.n_assets}, days={summary.n_days}, "
        f"events_per_asset={summary.events_per_asset}\n"
        f"repeat={summary.repeat}, seed={summary.seed}\n"
        f"legacy_best_s={summary.legacy_best:.6f}, "
        f"optimized_best_s={summary.optimized_best:.6f}\n"
        f"speedup={summary.speedup_ratio:.3f}x\n"
        f"max_abs_close_diff={summary.max_abs_close_diff:.3e}\n"
        f"legacy_samples={summary.legacy_seconds}\n"
        f"optimized_samples={summary.optimized_seconds}\n"
    )
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"summary saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
