from __future__ import annotations

import datetime
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class MetricUnavailable:
    """One metric that could not be computed from available replay artifacts."""

    metric: str
    reason: str


@dataclass(frozen=True)
class ExecutionImpactFlag:
    """Machine-readable execution-impact quality flag."""

    name: str
    triggered: bool | None
    observed: float | str | None
    threshold: float | None
    description: str


@dataclass(frozen=True)
class ExecutionImpactThresholds:
    """Explicit, deterministic thresholds used by execution-impact flags."""

    high_execution_deviation_mean_abs: float = 0.01
    severe_tradability_skipped_ratio: float = 0.30
    price_limit_reason_ratio: float = 0.10
    liquidity_reason_ratio: float = 0.30
    reentry_reason_ratio: float = 0.10
    material_turnover_reduction_ratio: float = 0.20


@dataclass(frozen=True)
class LoadedExecutionArtifacts:
    """Optional replay artifacts loaded from one adapter output directory."""

    run_path: Path
    adapter_run_metadata: dict[str, object] | None
    backtest_summary: dict[str, object] | None
    target_weights_df: pd.DataFrame | None
    executed_weights_df: pd.DataFrame | None
    turnover_df: pd.DataFrame | None
    orders_df: pd.DataFrame | None
    trades_df: pd.DataFrame | None
    skipped_orders_df: pd.DataFrame | None
    returns_df: pd.DataFrame | None
    equity_curve_df: pd.DataFrame | None
    missing_artifacts: tuple[str, ...]

    @property
    def engine_name(self) -> str | None:
        if self.backtest_summary is None:
            return None
        raw = self.backtest_summary.get("engine")
        return str(raw) if raw is not None else None


@dataclass(frozen=True)
class ExecutionImpactReport:
    """Research-facing diagnostic summary of execution-aware replay effects."""

    run_path: Path
    comparison_run_path: Path | None
    generated_at_utc: str
    unavailable_metrics: tuple[MetricUnavailable, ...]
    dominant_execution_blocker: str | None
    reason_summary_df: pd.DataFrame
    execution_deviation_summary: dict[str, object]
    turnover_effect_summary: dict[str, object]
    performance_context: dict[str, object]
    flags: tuple[ExecutionImpactFlag, ...]
    warnings: tuple[dict[str, object], ...]
    timeseries_df: pd.DataFrame | None
    comparison_summary: dict[str, object] | None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a deterministic JSON-ready dictionary."""
        return {
            "run_path": str(self.run_path),
            "comparison_run_path": (
                str(self.comparison_run_path) if self.comparison_run_path is not None else None
            ),
            "generated_at_utc": self.generated_at_utc,
            "unavailable_metrics": [x.__dict__ for x in self.unavailable_metrics],
            "dominant_execution_blocker": self.dominant_execution_blocker,
            "reason_summary": _df_records(self.reason_summary_df),
            "execution_deviation_summary": self.execution_deviation_summary,
            "turnover_effect_summary": self.turnover_effect_summary,
            "performance_context": self.performance_context,
            "flags": [x.__dict__ for x in self.flags],
            "warnings": list(self.warnings),
            "timeseries_summary": _df_records(self.timeseries_df),
            "comparison_summary": self.comparison_summary,
            "interpretation_note": (
                "Execution impact report is descriptive research diagnostics; "
                "it is not causal attribution and not a strategy-optimization signal."
            ),
        }


def load_execution_artifacts(run_path: str | Path) -> LoadedExecutionArtifacts:
    """Load replay artifacts from one adapter output directory."""
    path = Path(run_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"replay output directory does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"replay output path is not a directory: {path}")

    frames: dict[str, pd.DataFrame | None] = {
        "target_weights_df": _read_csv_optional(path / "target_weights.csv"),
        "executed_weights_df": _read_csv_optional(path / "executed_weights.csv"),
        "turnover_df": _read_csv_optional(path / "turnover.csv"),
        "orders_df": _read_csv_optional(path / "orders.csv"),
        "trades_df": _read_csv_optional(path / "trades.csv"),
        "skipped_orders_df": _read_csv_optional(path / "skipped_orders.csv"),
        "returns_df": _read_csv_optional(path / "portfolio_returns.csv"),
        "equity_curve_df": _read_csv_optional(path / "equity_curve.csv"),
    }
    for filename, frame in frames.items():
        if frame is not None and "date" in frame.columns:
            frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
            if frame["date"].isna().any():
                raise ValueError(f"{filename} contains invalid date values in 'date' column")
            frames[filename] = _sort_if_present(frame)

    missing = []
    for required in ("adapter_run_metadata.json", "backtest_summary.json"):
        if not (path / required).exists():
            missing.append(required)

    return LoadedExecutionArtifacts(
        run_path=path,
        adapter_run_metadata=_read_json_optional(path / "adapter_run_metadata.json"),
        backtest_summary=_read_json_optional(path / "backtest_summary.json"),
        target_weights_df=frames["target_weights_df"],
        executed_weights_df=frames["executed_weights_df"],
        turnover_df=frames["turnover_df"],
        orders_df=frames["orders_df"],
        trades_df=frames["trades_df"],
        skipped_orders_df=frames["skipped_orders_df"],
        returns_df=frames["returns_df"],
        equity_curve_df=frames["equity_curve_df"],
        missing_artifacts=tuple(sorted(missing)),
    )


def build_execution_impact_report(
    run_path: str | Path,
    *,
    comparison_run_path: str | Path | None = None,
    thresholds: ExecutionImpactThresholds | None = None,
) -> ExecutionImpactReport:
    """Build a descriptive execution-impact report from replay artifacts."""
    primary = load_execution_artifacts(run_path)
    comparison = (
        load_execution_artifacts(comparison_run_path) if comparison_run_path is not None else None
    )
    resolved_thresholds = thresholds or ExecutionImpactThresholds()

    unavailable: list[MetricUnavailable] = []
    reason_summary, dominant_blocker = _reason_distribution(primary, unavailable)
    deviation_summary = _execution_deviation_summary(primary, unavailable)
    turnover_summary = _turnover_effect_summary(primary, comparison, unavailable)
    performance_context = _performance_context(primary, comparison)
    flags = _build_flags(
        reason_summary_df=reason_summary,
        deviation_summary=deviation_summary,
        turnover_summary=turnover_summary,
        thresholds=resolved_thresholds,
    )
    comparison_summary = _comparison_summary(primary, comparison)
    timeseries = _timeseries_summary(primary, comparison)

    warnings: list[dict[str, object]] = []
    if primary.backtest_summary is not None:
        raw = primary.backtest_summary.get("warnings")
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    warnings.append(item)

    return ExecutionImpactReport(
        run_path=primary.run_path,
        comparison_run_path=comparison.run_path if comparison is not None else None,
        generated_at_utc=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        unavailable_metrics=_dedupe_unavailable(unavailable),
        dominant_execution_blocker=dominant_blocker,
        reason_summary_df=reason_summary,
        execution_deviation_summary=deviation_summary,
        turnover_effect_summary=turnover_summary,
        performance_context=performance_context,
        flags=tuple(flags),
        warnings=tuple(warnings),
        timeseries_df=timeseries,
        comparison_summary=comparison_summary,
    )


def export_execution_impact_report(
    report: ExecutionImpactReport,
    output_dir: str | Path,
    *,
    export_reason_csv: bool = True,
    export_timeseries_csv: bool = True,
) -> dict[str, Path]:
    """Export report JSON and optional CSV slices."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}
    report_path = out_dir / "execution_impact_report.json"
    report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    files["report_json"] = report_path

    if export_reason_csv and not report.reason_summary_df.empty:
        reason_path = out_dir / "execution_impact_by_reason.csv"
        report.reason_summary_df.to_csv(reason_path, index=False)
        files["reason_csv"] = reason_path

    if (
        export_timeseries_csv
        and report.timeseries_df is not None
        and not report.timeseries_df.empty
    ):
        ts_path = out_dir / "execution_impact_timeseries.csv"
        report.timeseries_df.to_csv(ts_path, index=False)
        files["timeseries_csv"] = ts_path
    return files


def _read_json_optional(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_csv_optional(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _sort_if_present(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        cols = ["date"] + [c for c in ("asset",) if c in df.columns]
        return df.sort_values(cols, kind="mergesort").reset_index(drop=True)
    return df.copy()


def _reason_distribution(
    artifacts: LoadedExecutionArtifacts,
    unavailable: list[MetricUnavailable],
) -> tuple[pd.DataFrame, str | None]:
    if artifacts.skipped_orders_df is None:
        unavailable.append(
            MetricUnavailable("reason_distribution", "skipped_orders.csv is missing")
        )
        return pd.DataFrame(
            columns=["reason_code", "skipped_order_count", "skipped_order_ratio"]
        ), None
    frame = artifacts.skipped_orders_df.copy()
    reason_col = (
        "reason_code"
        if "reason_code" in frame.columns
        else ("reason" if "reason" in frame.columns else None)
    )
    if reason_col is None:
        unavailable.append(
            MetricUnavailable(
                "reason_distribution",
                "skipped_orders.csv has no reason_code/reason column",
            )
        )
        return pd.DataFrame(
            columns=["reason_code", "skipped_order_count", "skipped_order_ratio"]
        ), None

    frame["reason_code"] = frame[reason_col].astype(str).str.strip()
    frame = frame.loc[frame["reason_code"] != ""].copy()
    if frame.empty:
        unavailable.append(
            MetricUnavailable(
                "reason_distribution",
                "skipped_orders.csv has empty reason codes only",
            )
        )
        return pd.DataFrame(
            columns=["reason_code", "skipped_order_count", "skipped_order_ratio"]
        ), None

    counts = (
        frame.groupby("reason_code", as_index=False)
        .size()
        .rename(columns={"size": "skipped_order_count"})
    )
    total = int(counts["skipped_order_count"].sum())
    counts["skipped_order_ratio"] = counts["skipped_order_count"] / total if total > 0 else 0.0
    counts = counts.sort_values(
        "skipped_order_count", ascending=False, kind="mergesort"
    ).reset_index(drop=True)
    dominant = str(counts.iloc[0]["reason_code"]) if not counts.empty else None
    return counts, dominant


def _execution_deviation_summary(
    artifacts: LoadedExecutionArtifacts,
    unavailable: list[MetricUnavailable],
) -> dict[str, object]:
    if artifacts.target_weights_df is None:
        unavailable.append(
            MetricUnavailable("execution_deviation", "target_weights.csv is missing")
        )
        return {}
    if artifacts.executed_weights_df is None:
        unavailable.append(
            MetricUnavailable("execution_deviation", "executed_weights.csv is missing")
        )
        return {}

    merged = artifacts.target_weights_df.merge(
        artifacts.executed_weights_df,
        on=["date", "asset"],
        how="outer",
        suffixes=("_target", "_executed"),
    ).fillna({"target_weight_target": 0.0, "target_weight_executed": 0.0})
    merged["abs_diff"] = (merged["target_weight_target"] - merged["target_weight_executed"]).abs()

    target_gross = merged.groupby("date", as_index=False)["target_weight_target"].apply(
        lambda s: s.abs().sum()
    )
    realized_gross = merged.groupby("date", as_index=False)["target_weight_executed"].apply(
        lambda s: s.abs().sum()
    )
    gross = target_gross.merge(realized_gross, on="date", suffixes=("_target", "_realized"))
    gross["gross_abs_diff"] = (
        gross["target_weight_target"] - gross["target_weight_executed"]
    ).abs()

    return {
        "mean_abs_weight_diff": float(merged["abs_diff"].mean()),
        "max_abs_weight_diff": float(merged["abs_diff"].max()),
        "target_gross_mean": float(gross["target_weight_target"].mean()),
        "realized_gross_mean": float(gross["target_weight_executed"].mean()),
        "gross_abs_diff_mean": float(gross["gross_abs_diff"].mean()),
        "gross_abs_diff_max": float(gross["gross_abs_diff"].max()),
    }


def _turnover_effect_summary(
    primary: LoadedExecutionArtifacts,
    comparison: LoadedExecutionArtifacts | None,
    unavailable: list[MetricUnavailable],
) -> dict[str, object]:
    primary_turnover = _safe_float(
        ((primary.backtest_summary or {}).get("summary") or {}).get("mean_turnover")
    )
    if comparison is None:
        return {"primary_mean_turnover": primary_turnover}
    comparison_turnover = _safe_float(
        ((comparison.backtest_summary or {}).get("summary") or {}).get("mean_turnover")
    )
    if primary_turnover is None or comparison_turnover is None:
        unavailable.append(
            MetricUnavailable(
                "turnover_effect",
                "mean_turnover missing from backtest_summary",
            )
        )
        return {}
    ratio = (
        (comparison_turnover - primary_turnover) / abs(comparison_turnover)
        if comparison_turnover != 0.0
        else None
    )
    return {
        "primary_mean_turnover": primary_turnover,
        "comparison_mean_turnover": comparison_turnover,
        "turnover_reduction_ratio_vs_comparison": ratio,
    }


def _performance_context(
    primary: LoadedExecutionArtifacts,
    comparison: LoadedExecutionArtifacts | None,
) -> dict[str, object]:
    out: dict[str, object] = {
        "primary_engine": primary.engine_name,
        "primary_summary": ((primary.backtest_summary or {}).get("summary") or {}),
    }
    if comparison is not None:
        out["comparison_engine"] = comparison.engine_name
        out["comparison_summary"] = (comparison.backtest_summary or {}).get("summary") or {}
    return out


def _build_flags(
    *,
    reason_summary_df: pd.DataFrame,
    deviation_summary: dict[str, object],
    turnover_summary: dict[str, object],
    thresholds: ExecutionImpactThresholds,
) -> list[ExecutionImpactFlag]:
    flags: list[ExecutionImpactFlag] = []

    mean_abs = _safe_float(deviation_summary.get("mean_abs_weight_diff"))
    flags.append(
        ExecutionImpactFlag(
            name="high_execution_deviation_mean_abs",
            triggered=(
                mean_abs is not None and mean_abs >= thresholds.high_execution_deviation_mean_abs
            ),
            observed=mean_abs,
            threshold=thresholds.high_execution_deviation_mean_abs,
            description="Mean absolute target-vs-executed weight deviation is high.",
        )
    )

    reason_ratio = _reason_ratio(reason_summary_df, {"not_tradable", "min_adv_filter"})
    flags.append(
        ExecutionImpactFlag(
            name="severe_tradability_skipped_ratio",
            triggered=(
                reason_ratio is not None
                and reason_ratio >= thresholds.severe_tradability_skipped_ratio
            ),
            observed=reason_ratio,
            threshold=thresholds.severe_tradability_skipped_ratio,
            description="A high ratio of skipped orders is due to tradability/liquidity limits.",
        )
    )

    price_limit_ratio = _reason_ratio(
        reason_summary_df, {"price_limit_locked", "price_limit_policy"}
    )
    flags.append(
        ExecutionImpactFlag(
            name="price_limit_reason_ratio",
            triggered=(
                price_limit_ratio is not None
                and price_limit_ratio >= thresholds.price_limit_reason_ratio
            ),
            observed=price_limit_ratio,
            threshold=thresholds.price_limit_reason_ratio,
            description="Price-limit constraints materially impacted order execution.",
        )
    )

    reentry_ratio = _reason_ratio(reason_summary_df, {"same_day_reentry_blocked"})
    flags.append(
        ExecutionImpactFlag(
            name="reentry_reason_ratio",
            triggered=(
                reentry_ratio is not None and reentry_ratio >= thresholds.reentry_reason_ratio
            ),
            observed=reentry_ratio,
            threshold=thresholds.reentry_reason_ratio,
            description="Same-day re-entry policy blocked a material share of trades.",
        )
    )

    turnover_reduction = _safe_float(turnover_summary.get("turnover_reduction_ratio_vs_comparison"))
    flags.append(
        ExecutionImpactFlag(
            name="material_turnover_reduction_ratio",
            triggered=(
                turnover_reduction is not None
                and turnover_reduction >= thresholds.material_turnover_reduction_ratio
            ),
            observed=turnover_reduction,
            threshold=thresholds.material_turnover_reduction_ratio,
            description="Primary engine exhibits materially lower turnover than comparison engine.",
        )
    )

    return flags


def _comparison_summary(
    primary: LoadedExecutionArtifacts,
    comparison: LoadedExecutionArtifacts | None,
) -> dict[str, object] | None:
    if comparison is None:
        return None
    primary_summary = (primary.backtest_summary or {}).get("summary") or {}
    comparison_summary = (comparison.backtest_summary or {}).get("summary") or {}
    metrics = ("total_return", "sharpe_annualized", "max_drawdown", "mean_turnover")
    out: dict[str, object] = {}
    for metric in metrics:
        p = _safe_float(primary_summary.get(metric))
        c = _safe_float(comparison_summary.get(metric))
        out[metric] = {
            "primary": p,
            "comparison": c,
            "delta_primary_minus_comparison": (p - c)
            if (p is not None and c is not None)
            else None,
        }
    return out


def _timeseries_summary(
    primary: LoadedExecutionArtifacts,
    comparison: LoadedExecutionArtifacts | None,
) -> pd.DataFrame | None:
    if primary.turnover_df is None and primary.returns_df is None:
        return None
    series: pd.DataFrame | None = None
    if primary.returns_df is not None:
        series = primary.returns_df.rename(columns={"portfolio_return": "primary_return"})
    if primary.turnover_df is not None:
        turn = primary.turnover_df.rename(columns={"turnover": "primary_turnover"})
        series = turn if series is None else series.merge(turn, on="date", how="outer")
    if comparison is not None:
        if comparison.returns_df is not None:
            comp_ret = comparison.returns_df.rename(
                columns={"portfolio_return": "comparison_return"}
            )
            series = comp_ret if series is None else series.merge(comp_ret, on="date", how="outer")
        if comparison.turnover_df is not None:
            comp_turn = comparison.turnover_df.rename(columns={"turnover": "comparison_turnover"})
            series = (
                comp_turn if series is None else series.merge(comp_turn, on="date", how="outer")
            )
    if series is None:
        return None
    return series.sort_values("date", kind="mergesort").reset_index(drop=True)


def _reason_ratio(reason_summary_df: pd.DataFrame, reason_codes: set[str]) -> float | None:
    if reason_summary_df.empty:
        return None
    mask = reason_summary_df["reason_code"].astype(str).isin(reason_codes)
    if not mask.any():
        return 0.0
    return float(reason_summary_df.loc[mask, "skipped_order_ratio"].sum())


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(out):
        return None
    return out


def _dedupe_unavailable(values: list[MetricUnavailable]) -> tuple[MetricUnavailable, ...]:
    seen: set[tuple[str, str]] = set()
    out: list[MetricUnavailable] = []
    for item in values:
        key = (item.metric, item.reason)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return tuple(out)


def _df_records(df: pd.DataFrame | None) -> list[dict[str, object]] | None:
    if df is None:
        return None
    return [{k: _jsonable(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def _jsonable(value: Any) -> object:
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return None if pd.isna(value) else value
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)
