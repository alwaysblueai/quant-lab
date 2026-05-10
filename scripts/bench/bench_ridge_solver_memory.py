from __future__ import annotations

import argparse
import gc
import json
import resource
import threading
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge


class _Sampler:
    def __init__(self, *, interval_seconds: float) -> None:
        self._interval_seconds = float(interval_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_kb = 0
        self.peak_swap_kb = 0
        self.samples = 0

    def start(self) -> None:
        self._sample()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self._interval_seconds * 2.0, 1.0))
        self._sample()

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_seconds):
            self._sample()

    def _sample(self) -> None:
        rss_kb = 0
        swap_kb = 0
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                rss_kb = _status_kb_value(line)
            elif line.startswith("VmSwap:"):
                swap_kb = _status_kb_value(line)
        self.peak_rss_kb = max(self.peak_rss_kb, rss_kb)
        self.peak_swap_kb = max(self.peak_swap_kb, swap_kb)
        self.samples += 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure pure sklearn Ridge fit peak RSS for model-factor window sizes."
    )
    parser.add_argument("--rows", type=int, default=450_000)
    parser.add_argument("--features", type=int, default=94)
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"])
    parser.add_argument("--copy-x", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--sample-interval-seconds", type=float, default=0.05)
    args = parser.parse_args()

    if args.rows <= 0:
        raise ValueError("--rows must be > 0")
    if args.features <= 0:
        raise ValueError("--features must be > 0")

    rng = np.random.default_rng(int(args.seed))
    dtype = np.float32 if args.dtype == "float32" else np.float64
    alloc_start = time.perf_counter()
    x = rng.standard_normal((int(args.rows), int(args.features))).astype(dtype, copy=False)
    beta = rng.standard_normal(int(args.features)).astype(dtype, copy=False)
    noise = rng.standard_normal(int(args.rows)).astype(dtype, copy=False)
    y = (x @ beta + noise * 0.01).astype(dtype, copy=False)  # type: ignore[operator]
    alloc_seconds = time.perf_counter() - alloc_start
    gc.collect()

    baseline_rss_kb = _current_rss_kb()
    sampler = _Sampler(interval_seconds=float(args.sample_interval_seconds))
    model = Ridge(alpha=float(args.alpha), solver=str(args.solver), copy_X=bool(args.copy_x))
    fit_start = time.perf_counter()
    sampler.start()
    model.fit(x, y)
    sampler.stop()
    fit_seconds = time.perf_counter() - fit_start
    final_rss_kb = _current_rss_kb()
    rusage_maxrss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    payload = {
        "rows": int(args.rows),
        "features": int(args.features),
        "dtype": args.dtype,
        "solver": str(args.solver),
        "copy_x": bool(args.copy_x),
        "alpha": float(args.alpha),
        "alloc_seconds": round(alloc_seconds, 6),
        "fit_seconds": round(fit_seconds, 6),
        "baseline_rss_mb": round(baseline_rss_kb / 1024.0, 3),
        "peak_rss_mb": round(max(sampler.peak_rss_kb, rusage_maxrss_kb) / 1024.0, 3),
        "fit_peak_delta_mb": round(
            (max(sampler.peak_rss_kb, rusage_maxrss_kb) - baseline_rss_kb) / 1024.0,
            3,
        ),
        "final_rss_mb": round(final_rss_kb / 1024.0, 3),
        "peak_swap_mb": round(sampler.peak_swap_kb / 1024.0, 3),
        "samples": int(sampler.samples),
        "coef_norm": float(np.linalg.norm(model.coef_)),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def _status_kb_value(line: str) -> int:
    parts = line.split()
    if len(parts) < 2:
        return 0
    try:
        return int(parts[1])
    except ValueError:
        return 0


def _current_rss_kb() -> int:
    for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return _status_kb_value(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
