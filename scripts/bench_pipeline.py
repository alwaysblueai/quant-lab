"""Phase 0 baseline bench harness for the alpha-lab experiment pipeline.

Segments the model_factor / single_factor experiment path into the stages that
subsequent optimization phases will target, then reports per-stage wall time
plus peak incremental memory. Inputs are fully synthetic so the bench runs
anywhere without Tushare caches or vault access.

Stages (default order):
    load          synthesize + canonicalize prices / factor panels
    label         forward_return for a horizon set (per horizon)
    transforms    cross-sectional winsorize + zscore on the factor
    neutralize    neutralize_signal against size / beta / industry exposures
    evaluate      compute_ic + compute_rank_ic against target horizon labels
    diagnostics   compute_ic_decay + compute_factor_autocorrelation
    integrity     leakage_checks.check_no_future_dates_in_input (+ siblings)
    artifact      json.dumps(indent=2, sort_keys=True) + csv write to tempdir
    train         bounded synthetic model-family fit benchmark

Outputs land under outputs/benchmarks/bench_pipeline/<run_id>/:
    summary.json    machine-readable; feed this into --compare later
    summary.md      human-readable table
    compare.json    machine-readable diff when --compare is provided
    compare.md      human-readable diff when --compare is provided

Usage:
    python scripts/bench_pipeline.py --size small
    python scripts/bench_pipeline.py --size medium --save baseline.json
    python scripts/bench_pipeline.py --size medium --compare baseline.json
    REGRESSION_THRESHOLD=0.05 python scripts/bench_pipeline.py --compare baseline.json
    python scripts/bench_pipeline.py --stages label,neutralize,diagnostics
    python scripts/bench_pipeline.py --size small --repeat 3
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata as importlib_metadata
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from alpha_lab.sorted_panel import mark_sorted

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "benchmarks" / "bench_pipeline"

ALL_STAGES: tuple[str, ...] = (
    "load",
    "label",
    "transforms",
    "neutralize",
    "evaluate",
    "diagnostics",
    "integrity",
    "artifact",
    "train",
)

# Named data sizes. Keep these stable; changing them invalidates prior baselines.
SIZES: dict[str, dict[str, int]] = {
    "small": {"n_dates": 120, "n_assets": 100, "n_industries": 10},
    "medium": {"n_dates": 500, "n_assets": 500, "n_industries": 24},
    "large": {"n_dates": 2000, "n_assets": 3000, "n_industries": 32},
}

TARGET_HORIZON = 5
DECAY_HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
AUTOCORR_LAGS: tuple[int, ...] = (1, 5, 10, 20)

# Report a regression when a stage slows by more than this ratio vs baseline.
# Small synthetic runs are intentionally quick, so timer/import/tracemalloc jitter
# is a larger share of each stage. Larger runs get stricter gates.
REGRESSION_THRESHOLDS_BY_SIZE: dict[str, float] = {
    "small": 0.10,
    "medium": 0.05,
    "large": 0.03,
}
REGRESSION_THRESHOLD = REGRESSION_THRESHOLDS_BY_SIZE["medium"]
TRAIN_MAX_DATES = 120
TRAIN_MAX_ROWS = 400_000


# ---------------------------------------------------------------------------
# Data generation


@dataclass(frozen=True)
class SyntheticPanels:
    prices: pd.DataFrame  # [date, asset, OHLCV-style fields]
    factor: pd.DataFrame  # [date, asset, factor, value]
    exposures: pd.DataFrame  # [date, asset, size, beta, industry, value]


def _trading_calendar(n_dates: int, *, start: str = "2015-01-05") -> pd.DatetimeIndex:
    # Business-day calendar is sufficient for timing; it does not need to match
    # any exchange calendar.
    return pd.bdate_range(start=start, periods=n_dates)


def _generate_panels(
    *, n_dates: int, n_assets: int, n_industries: int, seed: int
) -> SyntheticPanels:
    rng = np.random.default_rng(seed)
    dates = _trading_calendar(n_dates)
    assets = np.array([f"A{idx:05d}" for idx in range(n_assets)])

    # Geometric random walk for closes; open/vwap are small perturbations so
    # every execution_price_mode is exercisable.
    log_returns = rng.normal(loc=0.0, scale=0.015, size=(n_dates, n_assets))
    close = 10.0 * np.exp(np.cumsum(log_returns, axis=0))
    open_ = close * (1.0 + rng.normal(0.0, 0.002, size=close.shape))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.001, size=close.shape)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.001, size=close.shape)))
    vwap = close * (1.0 + rng.normal(0.0, 0.0015, size=close.shape))
    volume = rng.lognormal(mean=12.0, sigma=0.5, size=close.shape)
    amount = volume * vwap

    date_idx = np.repeat(dates.values, n_assets)
    asset_idx = np.tile(assets, n_dates)

    prices = pd.DataFrame(
        {
            "date": date_idx,
            "asset": asset_idx,
            "open": open_.reshape(-1),
            "high": high.reshape(-1),
            "low": low.reshape(-1),
            "close": close.reshape(-1),
            "vwap": vwap.reshape(-1),
            "volume": volume.reshape(-1),
            "amount": amount.reshape(-1),
        }
    )

    factor_values = rng.normal(0.0, 1.0, size=(n_dates, n_assets))
    factor = pd.DataFrame(
        {
            "date": date_idx,
            "asset": asset_idx,
            "factor": "bench_factor",
            "value": factor_values.reshape(-1),
        }
    )

    # Exposures: size (log market cap proxy), beta, industry. Industries are
    # stable per asset; size drifts; beta is per-date random.
    industries = np.array(
        [f"IND{idx:02d}" for idx in rng.integers(0, n_industries, size=n_assets)]
    )
    size_per_date = rng.normal(22.0, 1.0, size=(n_dates, n_assets))
    beta_per_date = rng.normal(1.0, 0.3, size=(n_dates, n_assets))

    exposures = pd.DataFrame(
        {
            "date": date_idx,
            "asset": asset_idx,
            "size": size_per_date.reshape(-1),
            "beta": beta_per_date.reshape(-1),
            "industry": np.tile(industries, n_dates),
            # Attach the same factor value so neutralize_signal has its input
            # column co-located with exposures — avoids a merge inside the bench.
            "value": factor_values.reshape(-1),
        }
    )

    return SyntheticPanels(prices=prices, factor=factor, exposures=exposures)


# ---------------------------------------------------------------------------
# Timing primitives


@dataclass
class StageResult:
    name: str
    ok: bool
    wall_seconds: float
    peak_mem_delta_mb: float
    n_rows_touched: int
    n_repeats: int
    samples_seconds: list[float] = field(default_factory=list)
    error: str | None = None
    note: str | None = None


@dataclass
class BenchReport:
    run_id: str
    size: str
    n_dates: int
    n_assets: int
    python_version: str
    platform: str
    git_sha: str | None
    git_dirty: bool | None
    machine: str
    processor: str
    cpu_count: int | None
    package_versions: dict[str, str]
    repeat: int
    seed: int
    stages_requested: list[str]
    total_wall_seconds: float
    peak_rss_mb: float
    stages: list[StageResult]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=False, default=str)


@contextmanager
def _measure() -> Iterator[dict[str, float]]:
    """Time a block and capture peak incremental memory via tracemalloc.

    Memory delta is reported in MiB. Using tracemalloc avoids the noise of
    process RSS, which can be dominated by pandas allocator reuse.
    """
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    out: dict[str, float] = {}
    try:
        yield out
    finally:
        elapsed = time.perf_counter() - t0
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        out["wall_seconds"] = elapsed
        out["peak_mem_delta_mb"] = peak / (1024.0 * 1024.0)
        del current


def _time_stage(
    name: str,
    fn: Callable[[], tuple[Any, int]],
    *,
    repeat: int,
    note: str | None = None,
) -> StageResult:
    samples: list[float] = []
    peak_max = 0.0
    rows = 0
    last_error: str | None = None
    for _ in range(repeat):
        try:
            with _measure() as m:
                _result, rows = fn()
        except Exception as exc:  # noqa: BLE001 — capture to report, not crash the bench
            last_error = f"{type(exc).__name__}: {exc}"
            return StageResult(
                name=name,
                ok=False,
                wall_seconds=float("nan"),
                peak_mem_delta_mb=float("nan"),
                n_rows_touched=0,
                n_repeats=len(samples),
                samples_seconds=samples,
                error=last_error,
                note=note,
            )
        samples.append(m["wall_seconds"])
        peak_max = max(peak_max, m["peak_mem_delta_mb"])
    return StageResult(
        name=name,
        ok=True,
        wall_seconds=min(samples),  # best-of-N: least noisy single number
        peak_mem_delta_mb=peak_max,
        n_rows_touched=int(rows),
        n_repeats=len(samples),
        samples_seconds=samples,
        note=note,
    )


def _peak_rss_mb() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # Linux reports KiB; macOS reports bytes.
    divisor = 1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0
    return float(ru.ru_maxrss) / divisor


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


def _git_dirty() -> bool | None:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.strip())


def _package_versions() -> dict[str, str]:
    packages = ("numpy", "pandas", "scipy", "scikit-learn", "numba", "lightgbm", "xgboost")
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _machine_metadata() -> dict[str, object]:
    return {
        "git_dirty": _git_dirty(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "package_versions": _package_versions(),
    }


# ---------------------------------------------------------------------------
# Stage runners. Each returns (result_obj, n_rows_touched).
#
# The stages call real alpha_lab functions so the numbers reflect production
# code paths. Imports are deferred to keep module import cheap and to let us
# bench cold-start separately in a follow-up.


def _stage_load(seed: int, shape: dict[str, int]) -> Callable[[], tuple[Any, int]]:
    def run() -> tuple[SyntheticPanels, int]:
        panels = _generate_panels(
            n_dates=shape["n_dates"],
            n_assets=shape["n_assets"],
            n_industries=shape["n_industries"],
            seed=seed,
        )
        # Include canonical sort so downstream stages start from a sorted panel
        # (this matches what common_io.load_prices does today).
        panels.prices.sort_values(["asset", "date"], inplace=True, kind="mergesort")
        panels.prices.reset_index(drop=True, inplace=True)
        mark_sorted(panels.prices, ("asset", "date"))
        panels.factor.sort_values(["date", "asset"], inplace=True, kind="mergesort")
        panels.factor.reset_index(drop=True, inplace=True)
        mark_sorted(panels.factor, ("date", "asset"))
        return panels, len(panels.prices)

    return run


def _stage_label(panels: SyntheticPanels) -> Callable[[], tuple[Any, int]]:
    def run() -> tuple[dict[int, pd.DataFrame], int]:
        from alpha_lab.labels import LabelCache

        label_cache = LabelCache(panels.prices)
        labels_by_horizon: dict[int, pd.DataFrame] = {}
        rows = 0
        for h in DECAY_HORIZONS:
            lab = label_cache.forward_return(h)
            labels_by_horizon[h] = lab
            rows += len(lab)
        return labels_by_horizon, rows

    return run


def _stage_transforms(panels: SyntheticPanels) -> Callable[[], tuple[Any, int]]:
    def run() -> tuple[pd.DataFrame, int]:
        from alpha_lab.signal_transforms import (
            winsorize_cross_section,
            zscore_cross_section,
        )

        wins = winsorize_cross_section(panels.factor)
        out = zscore_cross_section(wins)
        return out, len(out)

    return run


def _stage_neutralize(panels: SyntheticPanels) -> Callable[[], tuple[Any, int]]:
    def run() -> tuple[Any, int]:
        from alpha_lab.neutralization import neutralize_signal

        result = neutralize_signal(
            panels.exposures,
            value_col="value",
            by="date",
            size_col="size",
            beta_col="beta",
            industry_col="industry",
            enforce_integrity=False,  # bench synth data has no known_at_col
        )
        return result, len(result.data)

    return run


def _stage_evaluate(
    panels: SyntheticPanels, labels_by_horizon: dict[int, pd.DataFrame]
) -> Callable[[], tuple[Any, int]]:
    def run() -> tuple[dict[str, pd.DataFrame], int]:
        from alpha_lab.evaluation import compute_ic, compute_rank_ic

        labels = labels_by_horizon[TARGET_HORIZON]
        ic = compute_ic(panels.factor, labels)
        rank_ic = compute_rank_ic(panels.factor, labels)
        return {"ic": ic, "rank_ic": rank_ic}, len(ic) + len(rank_ic)

    return run


def _stage_diagnostics(
    panels: SyntheticPanels, labels_by_horizon: dict[int, pd.DataFrame]
) -> Callable[[], tuple[Any, int]]:
    def run() -> tuple[dict[str, pd.DataFrame], int]:
        from alpha_lab.decay import (
            compute_factor_autocorrelation,
            compute_ic_decay,
        )

        decay = compute_ic_decay(
            panels.factor,
            panels.prices,
            horizons=DECAY_HORIZONS,
            precomputed_labels_by_horizon=labels_by_horizon,
        )
        autocorr = compute_factor_autocorrelation(panels.factor, lags=AUTOCORR_LAGS)
        return {"decay": decay, "autocorr": autocorr}, len(decay) + len(autocorr)

    return run


def _stage_integrity(
    panels: SyntheticPanels,
    labels_by_horizon: dict[int, pd.DataFrame],
) -> Callable[[], tuple[Any, int]]:
    def run() -> tuple[Any, int]:
        from alpha_lab.research_integrity.leakage_checks import (
            check_cross_section_transform_scope,
            check_factor_label_temporal_order,
            check_no_future_dates_in_input,
        )

        cutoff = pd.Timestamp(panels.prices["date"].max())
        labels = labels_by_horizon[TARGET_HORIZON]
        result_prices = check_no_future_dates_in_input(
            panels.prices, max_allowed_date=cutoff, object_name="bench_prices"
        )
        result_factor = check_no_future_dates_in_input(
            panels.factor, max_allowed_date=cutoff, object_name="bench_factor"
        )
        result_labels = check_no_future_dates_in_input(
            labels, max_allowed_date=cutoff, object_name="bench_labels"
        )
        result_order = check_factor_label_temporal_order(
            panels.factor,
            labels,
            object_name="bench_factor_label_alignment",
        )
        result_scope = check_cross_section_transform_scope(
            panels.prices[["date", "asset"]],
            panels.factor[["date", "asset", "value"]],
            object_name="bench_factor_scope",
        )
        rows = (len(panels.prices) * 2) + (len(panels.factor) * 3) + len(labels)
        return (result_prices, result_factor, result_labels, result_order, result_scope), rows

    return run


def _stage_artifact(
    panels: SyntheticPanels, labels_by_horizon: dict[int, pd.DataFrame]
) -> Callable[[], tuple[Any, int]]:
    def run() -> tuple[Path, int]:
        with tempfile.TemporaryDirectory(prefix="bench_pipeline_artifact_") as tmp:
            out = Path(tmp)
            # Emulate the current artifact contract: pretty JSON for metrics
            # + CSV for diagnostics tables. Phase 1.4 will flip some of these
            # to compact JSON; the stage stays the same so deltas are visible.
            metrics_blob = {
                "n_dates": int(panels.prices["date"].nunique()),
                "n_assets": int(panels.prices["asset"].nunique()),
                "horizons": list(DECAY_HORIZONS),
                "target_horizon": TARGET_HORIZON,
                "factor_summary": {
                    "mean": float(panels.factor["value"].mean()),
                    "std": float(panels.factor["value"].std()),
                    "rows": int(len(panels.factor)),
                },
            }
            (out / "metrics.json").write_text(
                json.dumps(metrics_blob, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            labels_by_horizon[TARGET_HORIZON].to_csv(out / "labels.csv", index=False)
            panels.factor.to_csv(out / "factor.csv", index=False)
            rows = len(panels.factor) + len(labels_by_horizon[TARGET_HORIZON])
            return out, rows

    return run


def _stage_train(
    panels: SyntheticPanels,
    labels_by_horizon: dict[int, pd.DataFrame],
) -> Callable[[], tuple[Any, int]]:
    def run() -> tuple[dict[str, object], int]:
        from alpha_lab.model_factor.core import ModelSpec, _fit_model_bundle

        labels = labels_by_horizon[TARGET_HORIZON][["date", "asset", "value"]].rename(
            columns={"value": "label"}
        )
        features = (
            panels.factor[["date", "asset", "value"]]
            .rename(columns={"value": "factor_value"})
            .merge(
                panels.exposures[["date", "asset", "size", "beta", "industry"]],
                on=["date", "asset"],
                how="inner",
                validate="one_to_one",
            )
            .merge(labels, on=["date", "asset"], how="inner", validate="one_to_one")
            .dropna(subset=["label"])
        )
        train_dates = pd.Index(features["date"].drop_duplicates()).sort_values()
        if len(train_dates) > TRAIN_MAX_DATES:
            features = features[features["date"].isin(train_dates[-TRAIN_MAX_DATES:])].copy()
        if len(features) > TRAIN_MAX_ROWS:
            features = features.sample(n=TRAIN_MAX_ROWS, random_state=42).sort_values(
                ["asset", "date"],
                kind="mergesort",
            )
        features["industry_code"] = features["industry"].astype("category").cat.codes.astype(float)
        x_cols = ["factor_value", "size", "beta", "industry_code"]
        x = features.loc[:, x_cols]
        y = pd.to_numeric(features["label"], errors="coerce")
        valid = y.notna()
        x = x.loc[valid]
        y = y.loc[valid]

        families: list[str] = []
        results: dict[str, object] = {
            "n_train_rows": int(len(x)),
            "n_train_dates": int(features.loc[valid, "date"].nunique()),
            "families": families,
        }
        if len(x) == 0:
            return results, 0

        train_slice = features.loc[valid, ["date", *x_cols, "label"]].copy()
        _fit_model_bundle(
            train_slice=train_slice,
            config=None,
            model_version=1,
            model_spec=ModelSpec(
                family="elastic_net",
                params={
                    "alpha": 0.001,
                    "l1_ratio": 0.2,
                    "max_iter": 500,
                    "random_state": 7,
                },
            ),
            feature_columns_override=tuple(x_cols),
        )
        families.append("elastic_net")

        tree_family = "lightgbm"
        try:
            _fit_model_bundle(
                train_slice=train_slice,
                config=None,
                model_version=2,
                model_spec=ModelSpec(
                    family="lightgbm",
                    params={
                        "n_estimators": 80,
                        "min_data_in_leaf": 50,
                        "learning_rate": 0.05,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "reg_lambda": 1.0,
                        "random_state": 7,
                        "n_jobs": 1,
                        "verbosity": -1,
                    },
                ),
                feature_columns_override=tuple(x_cols),
            )
        except RuntimeError:
            tree_family = "gbdt"
            _fit_model_bundle(
                train_slice=train_slice,
                config=None,
                model_version=2,
                model_spec=ModelSpec(
                    family="gbdt",
                    params={
                        "max_iter": 50,
                        "min_samples_leaf": 50,
                        "l2_regularization": 1.0,
                        "random_state": 7,
                    },
                ),
                feature_columns_override=tuple(x_cols),
            )
        families.append(tree_family)
        return results, int(len(x) * 2)

    return run


# ---------------------------------------------------------------------------
# Reporting


def _format_markdown(report: BenchReport) -> str:
    lines: list[str] = []
    lines.append(f"# bench_pipeline — {report.run_id}")
    lines.append("")
    lines.append(f"- size: `{report.size}` "
                 f"({report.n_dates} dates × {report.n_assets} assets)")
    lines.append(f"- repeat: {report.repeat}  |  seed: {report.seed}")
    lines.append(f"- git: `{report.git_sha or 'unknown'}`  "
                 f"| dirty: `{report.git_dirty}`  "
                 f"|  python: {report.python_version}  "
                 f"|  platform: {report.platform}")
    lines.append(f"- machine: `{report.machine}`  "
                 f"|  processor: `{report.processor or 'unknown'}`  "
                 f"|  cpu_count: {report.cpu_count}")
    lines.append(f"- total wall: **{report.total_wall_seconds:.3f}s**  "
                 f"|  peak RSS: {report.peak_rss_mb:.1f} MiB")
    lines.append("")
    lines.append("| stage | ok | wall_s | peak_mem_MiB | rows | note |")
    lines.append("|---|---|---:|---:|---:|---|")
    for s in report.stages:
        ok = "✓" if s.ok else "✗"
        wall = "nan" if s.wall_seconds != s.wall_seconds else f"{s.wall_seconds:.4f}"
        mem = "nan" if s.peak_mem_delta_mb != s.peak_mem_delta_mb else f"{s.peak_mem_delta_mb:.1f}"
        note = s.note or (s.error or "")
        lines.append(f"| {s.name} | {ok} | {wall} | {mem} | {s.n_rows_touched} | {note} |")
    return "\n".join(lines) + "\n"


def _compare_payload(
    current: BenchReport,
    baseline: dict[str, Any],
    *,
    threshold: float,
) -> dict[str, object]:
    base_stages = {s["name"]: s for s in baseline.get("stages", [])}
    rows: list[dict[str, object]] = []
    any_regression = False
    for stage in current.stages:
        base = base_stages.get(stage.name)
        if base is None:
            rows.append(
                {
                    "stage": stage.name,
                    "baseline_seconds": None,
                    "current_seconds": stage.wall_seconds,
                    "delta_seconds": None,
                    "delta_ratio": None,
                    "verdict": "new",
                }
            )
            continue
        baseline_seconds = float(base.get("wall_seconds", float("nan")))
        current_seconds = stage.wall_seconds
        if not (baseline_seconds == baseline_seconds and current_seconds == current_seconds):
            rows.append(
                {
                    "stage": stage.name,
                    "baseline_seconds": baseline_seconds,
                    "current_seconds": current_seconds,
                    "delta_seconds": None,
                    "delta_ratio": None,
                    "verdict": "skipped",
                }
            )
            continue
        delta_seconds = current_seconds - baseline_seconds
        delta_ratio = delta_seconds / baseline_seconds if baseline_seconds > 0 else float("nan")
        if delta_ratio > threshold:
            verdict = "regression"
            any_regression = True
        elif delta_ratio < -threshold:
            verdict = "faster"
        else:
            verdict = "flat"
        rows.append(
            {
                "stage": stage.name,
                "baseline_seconds": baseline_seconds,
                "current_seconds": current_seconds,
                "delta_seconds": delta_seconds,
                "delta_ratio": delta_ratio,
                "verdict": verdict,
            }
        )

    base_total = float(baseline.get("total_wall_seconds", float("nan")))
    current_total = current.total_wall_seconds
    total_delta = (
        current_total - base_total
        if base_total == base_total and current_total == current_total
        else None
    )
    total_delta_ratio = (
        total_delta / base_total
        if total_delta is not None and base_total > 0
        else None
    )
    return {
        "schema_version": "1.0.0",
        "artifact_type": "alpha_lab_bench_compare",
        "threshold": float(threshold),
        "baseline_run_id": baseline.get("run_id"),
        "current_run_id": current.run_id,
        "baseline_size": baseline.get("size"),
        "current_size": current.size,
        "size_mismatch": baseline.get("size") != current.size,
        "any_regression": any_regression,
        "total": {
            "baseline_seconds": base_total,
            "current_seconds": current_total,
            "delta_seconds": total_delta,
            "delta_ratio": total_delta_ratio,
        },
        "stages": rows,
    }


def _format_compare(
    current: BenchReport,
    baseline: dict[str, Any],
    *,
    threshold: float,
) -> str:
    compare = _compare_payload(current, baseline, threshold=threshold)
    lines: list[str] = []
    lines.append("# bench_pipeline compare")
    lines.append("")
    lines.append(f"- baseline run: `{baseline.get('run_id', '?')}` "
                 f"(git `{baseline.get('git_sha', '?')}`, size `{baseline.get('size', '?')}`)")
    lines.append(f"- current run:  `{current.run_id}` "
                 f"(git `{current.git_sha or '?'}`, size `{current.size}`)")
    lines.append(f"- regression threshold: `{threshold * 100:.1f}%`")
    if baseline.get("size") != current.size:
        lines.append("")
        lines.append("> ⚠️  size differs — deltas below are not meaningful.")
    lines.append("")
    lines.append("| stage | baseline_s | current_s | Δ_s | Δ_% | verdict |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in compare["stages"]:
        row_dict = dict(row) if isinstance(row, dict) else {}
        stage_name = str(row_dict.get("stage") or "?")
        b_raw = row_dict.get("baseline_seconds")
        c_raw = row_dict.get("current_seconds")
        delta_raw = row_dict.get("delta_seconds")
        pct_raw = row_dict.get("delta_ratio")
        verdict_key = str(row_dict.get("verdict") or "?")
        if b_raw is None:
            lines.append(f"| {stage_name} | — | {float(c_raw):.4f} | — | — | new |")
            continue
        if delta_raw is None or pct_raw is None:
            lines.append(f"| {stage_name} | {b_raw} | {c_raw} | — | — | skipped |")
            continue
        b = float(b_raw)
        c = float(c_raw)
        delta = float(delta_raw)
        pct = float(pct_raw)
        verdict = {
            "regression": "⚠️ regression",
            "faster": "✅ faster",
            "flat": "≈ flat",
            "new": "new",
            "skipped": "skipped",
        }.get(verdict_key, verdict_key)
        lines.append(
            f"| {stage_name} | {b:.4f} | {c:.4f} | {delta:+.4f} | "
            f"{pct * 100:+.1f}% | {verdict} |"
        )
    total = compare.get("total", {})
    if isinstance(total, dict) and total.get("delta_seconds") is not None:
        base_total = float(total["baseline_seconds"])
        cur_total = float(total["current_seconds"])
        d = float(total["delta_seconds"])
        p = float(total["delta_ratio"]) if total.get("delta_ratio") is not None else float("nan")
        lines.append("")
        lines.append(f"**Total**: {base_total:.3f}s → {cur_total:.3f}s "
                     f"({d:+.3f}s, {p * 100:+.1f}%)")
    if bool(compare.get("any_regression")):
        lines.append("")
        lines.append(f"> ⚠️ one or more stages regressed by > {threshold * 100:.0f}%.")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI


def _default_regression_threshold(size: str) -> float:
    raw = os.getenv("REGRESSION_THRESHOLD")
    if raw is None:
        return REGRESSION_THRESHOLDS_BY_SIZE.get(size, REGRESSION_THRESHOLD)
    try:
        return float(raw)
    except ValueError:
        return REGRESSION_THRESHOLDS_BY_SIZE.get(size, REGRESSION_THRESHOLD)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 0 bench harness for alpha-lab experiment pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--size", default="small", choices=tuple(SIZES.keys()),
        help="Named data size. small=CI, medium=dev, large=nightly.",
    )
    parser.add_argument(
        "--stages", default=",".join(ALL_STAGES),
        help=f"Comma-separated subset of: {','.join(ALL_STAGES)}.",
    )
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Run each stage N times (reported wall is best-of-N).",
    )
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument(
        "--output-root", default=str(DEFAULT_OUTPUT_ROOT),
        help="Root dir for run outputs.",
    )
    parser.add_argument(
        "--save", default=None,
        help="Also copy summary.json to this path (useful for pinning a baseline).",
    )
    parser.add_argument(
        "--compare", default=None,
        help="Prior summary.json to diff current run against.",
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=None,
        help=(
            "Allowed per-stage slowdown ratio before --compare exits non-zero. "
            "Defaults by --size: small=10%, medium=5%, large=3%. "
            "Can also be set via REGRESSION_THRESHOLD."
        ),
    )
    return parser.parse_args(argv)


def _resolve_stages(raw: str) -> list[str]:
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    unknown = [t for t in tokens if t not in ALL_STAGES]
    if unknown:
        raise SystemExit(f"unknown stages: {unknown}; choose from {list(ALL_STAGES)}")
    # Preserve canonical order so reports stay comparable.
    return [s for s in ALL_STAGES if s in tokens]


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    stages = _resolve_stages(args.stages)
    shape = SIZES[args.size]
    regression_threshold = (
        float(args.regression_threshold)
        if args.regression_threshold is not None
        else _default_regression_threshold(args.size)
    )

    run_id = time.strftime("%Y%m%dT%H%M%S")
    out_dir = Path(args.output_root) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 'load' always runs first (everything else needs the panels). Other
    # stages can be opted out via --stages.
    load_result = _time_stage(
        "load",
        _stage_load(args.seed, shape),
        repeat=args.repeat,
    )
    if not load_result.ok:
        print(f"load stage failed: {load_result.error}", file=sys.stderr)
        return 1

    # Rerun load once more (outside the timed block) to hand panels to later
    # stages; the timed fn inside _time_stage discards its result.
    panels = _stage_load(args.seed, shape)()[0]

    # Shared prerequisites for downstream stages.
    labels_by_horizon: dict[int, pd.DataFrame] | None = None

    results: list[StageResult] = [load_result]
    for stage in stages:
        if stage == "load":
            continue  # already timed above
        if stage == "label":
            res = _time_stage("label", _stage_label(panels), repeat=args.repeat)
            results.append(res)
            if res.ok:
                # Materialize once for reuse in evaluate / diagnostics / artifact.
                labels_by_horizon = _stage_label(panels)()[0]
        elif stage == "transforms":
            results.append(_time_stage("transforms", _stage_transforms(panels), repeat=args.repeat))
        elif stage == "neutralize":
            results.append(_time_stage("neutralize", _stage_neutralize(panels), repeat=args.repeat))
        elif stage == "evaluate":
            if labels_by_horizon is None:
                labels_by_horizon = _stage_label(panels)()[0]
            results.append(
                _time_stage(
                    "evaluate",
                    _stage_evaluate(panels, labels_by_horizon),
                    repeat=args.repeat,
                )
            )
        elif stage == "diagnostics":
            if labels_by_horizon is None:
                labels_by_horizon = _stage_label(panels)()[0]
            results.append(
                _time_stage(
                    "diagnostics",
                    _stage_diagnostics(panels, labels_by_horizon),
                    repeat=args.repeat,
                )
            )
        elif stage == "integrity":
            if labels_by_horizon is None:
                labels_by_horizon = _stage_label(panels)()[0]
            results.append(
                _time_stage(
                    "integrity",
                    _stage_integrity(panels, labels_by_horizon),
                    repeat=args.repeat,
                )
            )
        elif stage == "artifact":
            if labels_by_horizon is None:
                labels_by_horizon = _stage_label(panels)()[0]
            results.append(
                _time_stage(
                    "artifact",
                    _stage_artifact(panels, labels_by_horizon),
                    repeat=args.repeat,
                )
            )
        elif stage == "train":
            if labels_by_horizon is None:
                labels_by_horizon = _stage_label(panels)()[0]
            results.append(
                _time_stage(
                    "train",
                    _stage_train(panels, labels_by_horizon),
                    repeat=args.repeat,
                )
            )

    total_wall = sum(s.wall_seconds for s in results if s.ok)
    report = BenchReport(
        run_id=run_id,
        size=args.size,
        n_dates=shape["n_dates"],
        n_assets=shape["n_assets"],
        python_version=platform.python_version(),
        platform=platform.platform(),
        git_sha=_git_sha_short(),
        git_dirty=_git_dirty(),
        machine=platform.machine(),
        processor=platform.processor(),
        cpu_count=os.cpu_count(),
        package_versions=_package_versions(),
        repeat=args.repeat,
        seed=args.seed,
        stages_requested=stages,
        total_wall_seconds=total_wall,
        peak_rss_mb=_peak_rss_mb(),
        stages=results,
    )

    summary_json = out_dir / "summary.json"
    summary_json.write_text(report.to_json() + "\n", encoding="utf-8")
    summary_md = out_dir / "summary.md"
    summary_md.write_text(_format_markdown(report), encoding="utf-8")

    print(_format_markdown(report), end="")
    print(f"\nwritten: {summary_json}")

    if args.save:
        save_path = Path(args.save).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(report.to_json() + "\n", encoding="utf-8")
        print(f"saved baseline: {save_path}")

    exit_code = 0 if all(s.ok for s in results) else 1

    if args.compare:
        baseline_path = Path(args.compare).resolve()
        if not baseline_path.exists():
            print(f"compare baseline not found: {baseline_path}", file=sys.stderr)
            return max(exit_code, 2)
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        compare_payload = _compare_payload(
            report,
            baseline,
            threshold=regression_threshold,
        )
        compare_md = _format_compare(
            report,
            baseline,
            threshold=regression_threshold,
        )
        (out_dir / "compare.json").write_text(
            json.dumps(compare_payload, indent=2, sort_keys=False, default=str) + "\n",
            encoding="utf-8",
        )
        (out_dir / "compare.md").write_text(compare_md, encoding="utf-8")
        print()
        print(compare_md, end="")
        if compare_payload.get("any_regression"):
            exit_code = max(exit_code, 1)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
