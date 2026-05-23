"""Tier-2 deep-dive module registry.

Each module is registered with a ``build`` callable that consumes the same
Tier-1 inputs and produces a JSON-serialisable payload. The registry is
intentionally thin; the heavy lifting still lives in the original primitives
(``grouped_evaluation``, ``validation.purged_kfold``, etc).

Adding a module
---------------
Implement a build function that accepts a :class:`Tier1Inputs` and returns a
``dict``. Register it in ``_REGISTRY`` below with a short module key and a
user-facing label. Keep the build function pure — do not touch disk; the
runner handles persistence and status capture.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd

from .artifacts import save_tier2_module
from .contracts import MetricStatus, Tier2ModuleStatus
from .tier1 import Tier1Inputs


@dataclass(frozen=True)
class Tier2Module:
    """Registered Tier-2 module descriptor."""

    key: str
    label: str
    build: Callable[[Tier1Inputs], dict[str, Any]]
    estimated_seconds: int = 60  # UI hint only


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    # Keep payloads portable; normalize timestamps to ISO date strings.
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].apply(lambda ts: ts.date().isoformat() if pd.notna(ts) else None)
    return cast(list[dict[str, Any]], out.where(pd.notna(out), None).to_dict(orient="records"))


def _build_conditional_ic(inputs: Tier1Inputs) -> dict[str, Any]:
    from alpha_lab.grouped_evaluation import (
        conditional_ic_by_cross_section_size,
        conditional_ic_by_factor_magnitude,
    )
    from alpha_lab.labels import forward_return

    labels = forward_return(inputs.prices, horizon=inputs.horizon)
    by_mag = conditional_ic_by_factor_magnitude(factor_df=inputs.factor_df, labels_df=labels)
    by_size = conditional_ic_by_cross_section_size(factor_df=inputs.factor_df, labels_df=labels)
    return {
        "by_factor_magnitude": _df_to_records(by_mag),
        "by_cross_section_size": _df_to_records(by_size),
    }


def _build_random_null(inputs: Tier1Inputs) -> dict[str, Any]:
    from alpha_lab.evaluation import compute_mean_rank_ic_permutation_null
    from alpha_lab.labels import forward_return

    labels = forward_return(inputs.prices, horizon=inputs.horizon)
    n_trials = 200
    actual, null_samples = compute_mean_rank_ic_permutation_null(
        inputs.factor_df,
        labels,
        n_permutations=n_trials,
        seed=20260419,
        min_assets_per_date=3,
    )
    s = pd.Series(null_samples, dtype=float).dropna()
    if s.empty or not pd.notna(actual):
        pct = float("nan")
    elif actual >= 0:
        pct = float((s >= actual).mean())
    else:
        pct = float((s <= actual).mean())
    return {
        "n_trials": n_trials,
        "null_mean_rank_ic_samples": s.tolist(),
        "actual_mean_rank_ic": actual,
        "p_value_one_sided": pct,
    }


def _build_purged_kfold(inputs: Tier1Inputs) -> dict[str, Any]:
    from alpha_lab.experiment import run_factor_experiment
    from alpha_lab.reporting.purged_kfold_diagnostics import build_purged_kfold_diagnostics

    # Tier-2 purged k-fold is full-sample at the experiment level; the
    # k-fold structure is applied downstream. Explicit opt-in silences
    # OPT-P0-2 warning.
    experiment_result = run_factor_experiment(
        inputs.prices,
        lambda _p: inputs.factor_df.copy(),
        horizon=inputs.horizon,
        n_quantiles=inputs.n_quantiles,
        allow_full_sample_evaluation=True,
    )
    diagnostics = build_purged_kfold_diagnostics(
        experiment_result=experiment_result,
        label_horizon=inputs.horizon,
        n_splits=5,
        embargo_pct=0.01,
    )
    return {
        "summary": diagnostics.summary,
        "folds": _df_to_records(diagnostics.folds),
    }


def _build_turnover_ts(inputs: Tier1Inputs) -> dict[str, Any]:
    from alpha_lab.quantile import quantile_assignments
    from alpha_lab.turnover import long_short_turnover, quantile_turnover

    assignments = quantile_assignments(inputs.factor_df, n_quantiles=inputs.n_quantiles)
    qto = quantile_turnover(assignments)
    lsto = long_short_turnover(qto)
    return {"turnover_ts": _df_to_records(lsto)}


def _build_coverage_ts(inputs: Tier1Inputs) -> dict[str, Any]:
    if inputs.factor_df.empty:
        return {"coverage_ts": []}
    summary = (
        inputs.factor_df.groupby("date", sort=True)
        .agg(
            n_assets=("asset", "nunique"),
            n_non_null=("value", lambda s: int(s.notna().sum())),
        )
        .reset_index()
    )
    summary["coverage"] = summary["n_non_null"] / summary["n_assets"].replace(0, pd.NA)
    return {"coverage_ts": _df_to_records(summary)}


def _build_integrity_full(inputs: Tier1Inputs) -> dict[str, Any]:
    # Re-run the static leakage checks available in Tier-1 but at full depth:
    # temporal order of factor vs labels, cross-section scope check, etc.
    from alpha_lab.labels import forward_return
    from alpha_lab.research_integrity.leakage_checks import (
        check_cross_section_transform_scope,
        check_factor_label_temporal_order,
        check_no_future_dates_in_input,
    )

    labels = forward_return(inputs.prices, horizon=inputs.horizon)
    max_price_date = pd.Timestamp(inputs.prices["date"].max())
    checks = [
        check_no_future_dates_in_input(
            inputs.factor_df,
            max_allowed_date=max_price_date,
            date_col="date",
            object_name="tier2_factor",
        ),
        check_cross_section_transform_scope(
            inputs.prices[["date", "asset"]],
            inputs.factor_df[["date", "asset", "value"]],
            date_col="date",
            asset_col="asset",
            object_name="tier2_factor_scope",
        ),
        check_factor_label_temporal_order(
            inputs.factor_df,
            labels,
            join_keys=("date", "asset"),
            factor_date_col="date",
            label_date_col="date",
            object_name="tier2_factor_label_alignment",
        ),
    ]
    out = []
    for c in checks:
        out.append(
            {
                "name": getattr(c, "name", ""),
                "status": getattr(c, "status", ""),
                "message": getattr(c, "message", ""),
                "severity": getattr(c, "severity", ""),
            }
        )
    return {"checks": out}


_REGISTRY: dict[str, Tier2Module] = {
    m.key: m
    for m in (
        Tier2Module(
            "conditional_ic", "Conditional IC", _build_conditional_ic, estimated_seconds=30
        ),
        Tier2Module(
            "random_null", "Random Null (N=200)", _build_random_null, estimated_seconds=180
        ),
        Tier2Module("purged_kfold", "Purged K-Fold", _build_purged_kfold, estimated_seconds=90),
        Tier2Module(
            "turnover_ts", "Turnover Time Series", _build_turnover_ts, estimated_seconds=30
        ),
        Tier2Module(
            "coverage_ts", "Coverage Time Series", _build_coverage_ts, estimated_seconds=15
        ),
        Tier2Module(
            "integrity_full", "Integrity (Full)", _build_integrity_full, estimated_seconds=30
        ),
    )
}

TIER2_MODULES: tuple[Tier2Module, ...] = tuple(_REGISTRY.values())


def run_tier2_modules(
    inputs: Tier1Inputs,
    *,
    artifact_root: str | Path,
    factor_name: str,
    run_id: str,
    modules: Iterable[str],
    inputs_hash: str = "",
) -> dict[str, Tier2ModuleStatus]:
    """Run the requested Tier-2 modules, persist each, return status map.

    Unknown module keys produce a ``FAILED`` status with a clear message
    rather than raising, so a partial batch still succeeds for valid names.
    Each module's status is written before the next one starts — a crash
    midway does not orphan previously completed modules.
    """
    statuses: dict[str, Tier2ModuleStatus] = {}
    root = Path(artifact_root).resolve()

    for key in modules:
        mod = _REGISTRY.get(key)
        if mod is None:
            statuses[key] = Tier2ModuleStatus(
                module=key,
                status=MetricStatus.FAILED,
                message=f"unknown module: {key}",
                inputs_hash=inputs_hash,
            )
            save_tier2_module(
                root,
                factor_name,
                run_id,
                key,
                result_payload={"error": f"unknown module: {key}"},
                status=statuses[key],
            )
            continue

        start = time.perf_counter()
        try:
            payload = mod.build(inputs)
            status = Tier2ModuleStatus(
                module=key,
                status=MetricStatus.COMPUTED,
                computed_at=datetime.now(UTC).isoformat(timespec="seconds"),
                duration_sec=round(time.perf_counter() - start, 3),
                inputs_hash=inputs_hash,
                stale=False,
            )
            save_tier2_module(
                root,
                factor_name,
                run_id,
                key,
                result_payload=payload,
                status=status,
            )
            statuses[key] = status
        except Exception as exc:  # noqa: BLE001 — we want to record and continue
            tb = traceback.format_exc(limit=3)
            status = Tier2ModuleStatus(
                module=key,
                status=MetricStatus.FAILED,
                computed_at=datetime.now(UTC).isoformat(timespec="seconds"),
                duration_sec=round(time.perf_counter() - start, 3),
                inputs_hash=inputs_hash,
                message=f"{type(exc).__name__}: {exc}\n{tb}",
            )
            save_tier2_module(
                root,
                factor_name,
                run_id,
                key,
                result_payload={"error": str(exc)},
                status=status,
            )
            statuses[key] = status
    return statuses
