from __future__ import annotations

import datetime
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
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

    high_execution_deviation_mean_abs: float = 0.02
    severe_tradability_skipped_ratio: float = 0.20
    price_limit_reason_ratio: float = 0.05
    liquidity_reason_ratio: float = 0.10
    reentry_reason_ratio: float = 0.02
    material_turnover_reduction_ratio: float = 0.10


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
        if self.adapter_run_metadata is not None:
            raw = self.adapter_run_metadata.get("engine")
            if raw is not None:
                return str(raw)
        if self.backtest_summary is not None:
            raw = self.backtest_summary.get("engine")
            if raw is not None:
                return str(raw)
        return None


@dataclass(frozen=True)
class ExecutionImpactReport:
    """Research-facing diagnostic summary of execution-aware replay effects."""

    run_path: Path
    comparison_run_path: Path | None
    generated_at_utc: str
    engine: str | None
    missing_artifacts: tuple[str, ...]
    unavailable_metrics: tuple[MetricUnavailable, ...]
    dominant_execution_blocker: str | None
    reason_summary_df: pd.DataFrame
    execution_deviation_summary: dict[str, object]
    turnover_effect_summary: dict[str, object]
    performance_context: dict[str, object]
    warnings: tuple[dict[str, object], ...]
    flags: tuple[ExecutionImpactFlag, ...]
    timeseries_df: pd.DataFrame
    comparison_summary: dict[str, object] | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize to a deterministic JSON-ready dictionary."""

        return {
            "run_path": str(self.run_path),
            "comparison_run_path": (
                str(self.comparison_run_path) if self.comparison_run_path is not None else None
            ),
            "generated_at_utc": self.generated_at_utc,
            "engine": self.engine,
            "missing_artifacts": list(self.missing_artifacts),
            "unavailable_metrics": [asdict(item) for item in self.unavailable_metrics],
            "dominant_execution_blocker": self.dominant_execution_blocker,
            "reason_summary": _df_records(self.reason_summary_df),
            "execution_deviation_summary": self.execution_deviation_summary,
            "turnover_effect_summary": self.turnover_effect_summary,
            "performance_context": self.performance_context,
            "warnings": list(self.warnings),
            "flags": [asdict(flag) for flag in self.flags],
            "timeseries_summary": _df_records(self.timeseries_df),
            "comparison_summary": self.comparison_summary,
            "interpretation_note": (
                "Execution impact report is descriptive research diagnostics; it is not "
                "causal attribution and not a strategy-optimization signal."
            ),
        }


def load_execution_artifacts(run_path: str | Path) -> LoadedExecutionArtifacts:
    """Load replay artifacts from one adapter output directory.

    Missing optional artifacts are recorded and do not raise by default.
    Malformed files still raise explicit errors.
    """

    path = Path(run_path).resolve()
    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(f"replay output directory does not exist: {path}")

    missing: list[str] = []
    adapter_run_metadata = _read_json_optional(path, "adapter_run_metadata.json", missing)
    backtest_summary = _read_json_optional(path, "backtest_summary.json", missing)
    target_weights_df = _read_csv_optional(path, "target_weights.csv", missing)
    executed_weights_df = _read_csv_optional(path, "executed_weights.csv", missing)
    turnover_df = _read_csv_optional(path, "turnover.csv", missing)
    orders_df = _read_csv_optional(path, "orders.csv", missing)
    trades_df = _read_csv_optional(path, "trades.csv", missing)
    skipped_orders_df = _read_csv_optional(path, "skipped_orders.csv", missing)
    returns_df = _read_csv_optional(path, "portfolio_returns.csv", missing)
    equity_curve_df = _read_csv_optional(path, "equity_curve.csv", missing)

    frames: dict[str, pd.DataFrame | None] = {
        "target_weights.csv": target_weights_df,
        "executed_weights.csv": executed_weights_df,
        "turnover.csv": turnover_df,
        "orders.csv": orders_df,
        "trades.csv": trades_df,
        "skipped_orders.csv": skipped_orders_df,
        "portfolio_returns.csv": returns_df,
        "equity_curve.csv": equity_curve_df,
    }
    for filename, frame in frames.items():
        if frame is None or "date" not in frame.columns:
            continue
        parsed = pd.to_datetime(frame["date"], errors="coerce")
        if parsed.isna().any():
            raise ValueError(f"{filename} contains invalid date values in 'date' column")
        frame["date"] = parsed

    return LoadedExecutionArtifacts(
        run_path=path,
        adapter_run_metadata=adapter_run_metadata,
        backtest_summary=backtest_summary,
        target_weights_df=_sort_if_present(target_weights_df, by=("date", "asset")),
        executed_weights_df=_sort_if_present(executed_weights_df, by=("date", "asset")),
        turnover_df=_sort_if_present(turnover_df, by=("date",)),
        orders_df=_sort_if_present(orders_df, by=("date", "asset")),
        trades_df=_sort_if_present(trades_df, by=("date", "asset")),
        skipped_orders_df=_sort_if_present(skipped_orders_df, by=("date", "asset")),
        returns_df=_sort_if_present(returns_df, by=("date",)),
        equity_curve_df=_sort_if_present(equity_curve_df, by=("date",)),
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
    resolved_thresholds = thresholds or ExecutionImpactThresholds()
    unavailable: list[MetricUnavailable] = []

    reason_summary_df, dominant_blocker = _reason_distribution(
        primary.skipped_orders_df,
        unavailable=unavailable,
    )
    deviation_summary, timeseries_df = _execution_deviation_summary(
        primary.target_weights_df,
        primary.executed_weights_df,
        unavailable=unavailable,
    )
    turnover_summary = _turnover_effect_summary(
        target_weights_df=primary.target_weights_df,
        executed_weights_df=primary.executed_weights_df,
        turnover_df=primary.turnover_df,
        orders_df=primary.orders_df,
        skipped_orders_df=primary.skipped_orders_df,
        unavailable=unavailable,
        thresholds=resolved_thresholds,
    )
    performance_context, warnings = _performance_context(
        primary.backtest_summary,
        primary.adapter_run_metadata,
        unavailable=unavailable,
    )
    flags = _build_flags(
        reason_summary_df=reason_summary_df,
        dominant_execution_blocker=dominant_blocker,
        deviation_summary=deviation_summary,
        turnover_summary=turnover_summary,
        thresholds=resolved_thresholds,
    )

    comparison_summary: dict[str, object] | None = None
    resolved_comparison_path: Path | None = None
    if comparison_run_path is not None:
        comparison = load_execution_artifacts(comparison_run_path)
        comparison_summary = _comparison_summary(primary=primary, comparison=comparison)
        resolved_comparison_path = comparison.run_path

    return ExecutionImpactReport(
        run_path=primary.run_path,
        comparison_run_path=resolved_comparison_path,
        generated_at_utc=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        engine=primary.engine_name,
        missing_artifacts=primary.missing_artifacts,
        unavailable_metrics=tuple(_dedupe_unavailable(unavailable)),
        dominant_execution_blocker=dominant_blocker,
        reason_summary_df=reason_summary_df,
        execution_deviation_summary=deviation_summary,
        turnover_effect_summary=turnover_summary,
        performance_context=performance_context,
        warnings=warnings,
        flags=tuple(flags),
        timeseries_df=timeseries_df,
        comparison_summary=comparison_summary,
    )


def export_execution_impact_report(
    report: ExecutionImpactReport,
    *,
    output_dir: str | Path | None = None,
    export_reason_csv: bool = True,
    export_timeseries_csv: bool = True,
) -> dict[str, Path]:
    """Export report JSON and optional CSV slices."""

    out_dir = Path(output_dir) if output_dir is not None else report.run_path
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files: dict[str, Path] = {}
    report_path = out_dir / "execution_impact_report.json"
    report_path.write_text(
        json.dumps(report.to_dict(), sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    files["report_json"] = report_path

    if export_reason_csv and not report.reason_summary_df.empty:
        reason_path = out_dir / "execution_impact_by_reason.csv"
        report.reason_summary_df.to_csv(reason_path, index=False)
        files["reason_csv"] = reason_path

    if export_timeseries_csv and not report.timeseries_df.empty:
        ts_path = out_dir / "execution_impact_timeseries.csv"
        report.timeseries_df.to_csv(ts_path, index=False)
        files["timeseries_csv"] = ts_path

    return files


def _read_json_optional(
    run_path: Path,
    filename: str,
    missing: list[str],
) -> dict[str, object] | None:
    path = run_path / filename
    if not path.exists():
        missing.append(filename)
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{filename} must contain a JSON object")
    return payload


def _read_csv_optional(
    run_path: Path,
    filename: str,
    missing: list[str],
) -> pd.DataFrame | None:
    path = run_path / filename
    if not path.exists():
        missing.append(filename)
        return None
    return pd.read_csv(path)


def _sort_if_present(df: pd.DataFrame | None, *, by: tuple[str, ...]) -> pd.DataFrame | None:
    if df is None:
        return None
    if not all(col in df.columns for col in by):
        return df.reset_index(drop=True)
    return df.sort_values(list(by), kind="mergesort").reset_index(drop=True)


def _reason_distribution(
    skipped_orders_df: pd.DataFrame | None,
    *,
    unavailable: list[MetricUnavailable],
) -> tuple[pd.DataFrame, str | None]:
    cols = [
        "reason_code",
        "skipped_order_count",
        "skipped_order_ratio",
        "source_reason_examples",
    ]
    if skipped_orders_df is None:
        unavailable.append(
            MetricUnavailable(
                metric="constraint_reason_distribution",
                reason="skipped_orders.csv is missing",
            )
        )
        return pd.DataFrame(columns=cols), None
    if skipped_orders_df.empty:
        return pd.DataFrame(columns=cols), None

    frame = skipped_orders_df.copy()
    reason_col = "reason_code" if "reason_code" in frame.columns else "reason"
    if reason_col not in frame.columns:
        unavailable.append(
            MetricUnavailable(
                metric="constraint_reason_distribution",
                reason="skipped_orders.csv has no reason_code/reason column",
            )
        )
        return pd.DataFrame(columns=cols), None

    frame["reason_code"] = frame[reason_col].astype(str).str.strip()
    frame = frame[frame["reason_code"] != ""].copy()
    if frame.empty:
        unavailable.append(
            MetricUnavailable(
                metric="constraint_reason_distribution",
                reason="skipped_orders.csv has empty reason codes only",
            )
        )
        return pd.DataFrame(columns=cols), None

    total = int(len(frame))
    counts = frame.groupby("reason_code", sort=True).size().rename("skipped_order_count")
    out = counts.reset_index()
    out["skipped_order_ratio"] = out["skipped_order_count"] / float(total)

    if "source_reason" in frame.columns:
        examples = (
            frame.groupby("reason_code", sort=True)["source_reason"]
            .apply(_collect_examples)
            .rename("source_reason_examples")
            .reset_index()
        )
        out = out.merge(examples, on="reason_code", how="left")
    else:
        out["source_reason_examples"] = out["reason_code"]

    out = out.sort_values(
        ["skipped_order_count", "reason_code"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    dominant = str(out.iloc[0]["reason_code"]) if not out.empty else None
    return out[cols], dominant


def _collect_examples(values: pd.Series) -> str:
    text = values.astype(str).str.strip()
    uniq = sorted({item for item in text.tolist() if item})
    return ";".join(uniq[:5])


def _execution_deviation_summary(
    target_weights_df: pd.DataFrame | None,
    executed_weights_df: pd.DataFrame | None,
    *,
    unavailable: list[MetricUnavailable],
) -> tuple[dict[str, object], pd.DataFrame]:
    summary: dict[str, object] = {
        "mean_abs_weight_diff": None,
        "max_abs_weight_diff": None,
        "target_gross_mean": None,
        "realized_gross_mean": None,
        "gross_abs_diff_mean": None,
        "gross_abs_diff_max": None,
    }
    ts_cols = [
        "date",
        "target_gross",
        "realized_gross",
        "gross_abs_diff",
        "mean_abs_weight_diff",
        "max_abs_weight_diff",
    ]
    if target_weights_df is None or executed_weights_df is None:
        if target_weights_df is None:
            unavailable.append(
                MetricUnavailable(
                    metric="execution_deviation",
                    reason="target_weights.csv is missing",
                )
            )
        if executed_weights_df is None:
            unavailable.append(
                MetricUnavailable(
                    metric="execution_deviation",
                    reason="executed_weights.csv is missing",
                )
            )
        return summary, pd.DataFrame(columns=ts_cols)

    left, left_err = _prepare_weight_frame(target_weights_df)
    right, right_err = _prepare_weight_frame(executed_weights_df)
    if left_err is not None:
        unavailable.append(MetricUnavailable(metric="execution_deviation", reason=left_err))
        return summary, pd.DataFrame(columns=ts_cols)
    if right_err is not None:
        unavailable.append(MetricUnavailable(metric="execution_deviation", reason=right_err))
        return summary, pd.DataFrame(columns=ts_cols)

    if left is None or right is None:
        return summary, pd.DataFrame(columns=ts_cols)

    merged = left.merge(
        right,
        on=["date", "asset"],
        how="outer",
        suffixes=("_target", "_realized"),
    )
    merged["target_weight_target"] = merged["target_weight_target"].fillna(0.0)
    merged["target_weight_realized"] = merged["target_weight_realized"].fillna(0.0)
    merged["abs_weight_diff"] = (
        merged["target_weight_target"] - merged["target_weight_realized"]
    ).abs()

    summary["mean_abs_weight_diff"] = _maybe_float(merged["abs_weight_diff"].mean())
    summary["max_abs_weight_diff"] = _maybe_float(merged["abs_weight_diff"].max())

    by_date = merged.groupby("date", sort=True).agg(
        target_gross=("target_weight_target", lambda x: float(np.abs(x).sum())),
        realized_gross=("target_weight_realized", lambda x: float(np.abs(x).sum())),
        mean_abs_weight_diff=("abs_weight_diff", "mean"),
        max_abs_weight_diff=("abs_weight_diff", "max"),
    )
    by_date = by_date.reset_index()
    by_date["gross_abs_diff"] = (by_date["target_gross"] - by_date["realized_gross"]).abs()
    by_date = by_date[ts_cols]

    summary["target_gross_mean"] = _maybe_float(by_date["target_gross"].mean())
    summary["realized_gross_mean"] = _maybe_float(by_date["realized_gross"].mean())
    summary["gross_abs_diff_mean"] = _maybe_float(by_date["gross_abs_diff"].mean())
    summary["gross_abs_diff_max"] = _maybe_float(by_date["gross_abs_diff"].max())
    by_date = by_date.sort_values("date", kind="mergesort").reset_index(drop=True)
    return summary, by_date


def _prepare_weight_frame(df: pd.DataFrame) -> tuple[pd.DataFrame | None, str | None]:
    required = {"date", "asset", "target_weight"}
    missing = required - set(df.columns)
    if missing:
        return None, f"weight table missing columns {sorted(missing)}"
    out = df[["date", "asset", "target_weight"]].copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        return None, "weight table contains invalid dates"
    out["target_weight"] = pd.to_numeric(out["target_weight"], errors="coerce")
    if out["target_weight"].isna().any():
        return None, "weight table contains non-numeric target_weight values"
    grouped = (
        out.groupby(["date", "asset"], sort=True, as_index=False)["target_weight"]
        .sum()
        .sort_values(["date", "asset"], kind="mergesort")
        .reset_index(drop=True)
    )
    return grouped, None


def _turnover_effect_summary(
    *,
    target_weights_df: pd.DataFrame | None,
    executed_weights_df: pd.DataFrame | None,
    turnover_df: pd.DataFrame | None,
    orders_df: pd.DataFrame | None,
    skipped_orders_df: pd.DataFrame | None,
    unavailable: list[MetricUnavailable],
    thresholds: ExecutionImpactThresholds,
) -> dict[str, object]:
    out: dict[str, object] = {
        "target_mean_turnover": None,
        "realized_mean_turnover": None,
        "turnover_reduction": None,
        "turnover_reduction_ratio": None,
        "material_turnover_reduction": None,
        "skipped_order_ratio": None,
        "n_orders": None,
        "n_skipped_orders": None,
    }

    target_turnover: float | None = None
    if target_weights_df is None:
        unavailable.append(MetricUnavailable(metric="target_turnover", reason="target missing"))
    else:
        try:
            target_series = _turnover_from_weights(target_weights_df)
            target_turnover = _maybe_float(target_series.mean())
            out["target_mean_turnover"] = target_turnover
        except ValueError as exc:
            unavailable.append(
                MetricUnavailable(
                    metric="target_turnover",
                    reason=f"cannot compute from target_weights.csv: {exc}",
                )
            )

    realized_turnover: float | None = None
    if turnover_df is not None and {"date", "turnover"}.issubset(turnover_df.columns):
        realized_turnover = _maybe_float(
            pd.to_numeric(turnover_df["turnover"], errors="coerce").mean()
        )
    elif executed_weights_df is not None:
        try:
            realized_turnover = _maybe_float(_turnover_from_weights(executed_weights_df).mean())
        except ValueError as exc:
            unavailable.append(
                MetricUnavailable(
                    metric="realized_turnover",
                    reason=f"cannot compute from executed_weights.csv: {exc}",
                )
            )
    else:
        unavailable.append(
            MetricUnavailable(
                metric="realized_turnover",
                reason="turnover.csv and executed_weights.csv are both missing",
            )
        )
    out["realized_mean_turnover"] = realized_turnover

    if target_turnover is not None and realized_turnover is not None:
        reduction = target_turnover - realized_turnover
        out["turnover_reduction"] = reduction
        if abs(target_turnover) > 1e-12:
            reduction_ratio = reduction / target_turnover
            out["turnover_reduction_ratio"] = reduction_ratio
            out["material_turnover_reduction"] = (
                reduction_ratio >= thresholds.material_turnover_reduction_ratio
            )

    n_orders = int(len(orders_df)) if orders_df is not None else None
    n_skipped = int(len(skipped_orders_df)) if skipped_orders_df is not None else None
    out["n_orders"] = n_orders
    out["n_skipped_orders"] = n_skipped
    if n_orders is not None and n_skipped is not None:
        total_attempts = n_orders + n_skipped
        out["skipped_order_ratio"] = 0.0 if total_attempts == 0 else n_skipped / total_attempts
    elif n_orders is None and n_skipped is None:
        unavailable.append(
            MetricUnavailable(
                metric="skipped_order_ratio",
                reason="orders.csv and skipped_orders.csv are both missing",
            )
        )
    return out


def _turnover_from_weights(weights_df: pd.DataFrame) -> pd.Series:
    frame, error = _prepare_weight_frame(weights_df)
    if frame is None:
        raise ValueError(error or "invalid weights table")
    wide = frame.pivot(index="date", columns="asset", values="target_weight")
    wide = wide.sort_index().sort_index(axis=1).fillna(0.0)
    return (wide.diff().abs().sum(axis=1).fillna(0.0) / 2.0).astype(float)


def _performance_context(
    backtest_summary: dict[str, object] | None,
    adapter_run_metadata: dict[str, object] | None,
    *,
    unavailable: list[MetricUnavailable],
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    context: dict[str, object] = {
        "engine": None,
        "total_return": None,
        "sharpe_annualized": None,
        "max_drawdown": None,
        "mean_turnover": None,
        "n_periods": None,
        "descriptive_only": True,
    }
    if adapter_run_metadata is not None:
        raw = adapter_run_metadata.get("engine")
        if raw is not None:
            context["engine"] = str(raw)

    warnings = _collect_warning_dicts(backtest_summary, adapter_run_metadata)
    if backtest_summary is None:
        unavailable.append(
            MetricUnavailable(metric="performance_context", reason="backtest_summary.json missing")
        )
        return context, warnings

    summary = backtest_summary.get("summary")
    if not isinstance(summary, dict):
        unavailable.append(
            MetricUnavailable(
                metric="performance_context",
                reason="backtest_summary.json.summary missing or invalid",
            )
        )
    else:
        context["total_return"] = _maybe_float(summary.get("total_return"))
        context["sharpe_annualized"] = _maybe_float(summary.get("sharpe_annualized"))
        context["max_drawdown"] = _maybe_float(summary.get("max_drawdown"))
        context["mean_turnover"] = _maybe_float(summary.get("mean_turnover"))
        context["n_periods"] = _coerce_int(summary.get("n_periods"))

    raw_engine = backtest_summary.get("engine")
    if raw_engine is not None:
        context["engine"] = str(raw_engine)
    return context, warnings


def _collect_warning_dicts(
    backtest_summary: dict[str, object] | None,
    adapter_run_metadata: dict[str, object] | None,
) -> tuple[dict[str, object], ...]:
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, object]] = []

    def _append(raw: object) -> None:
        if not isinstance(raw, list):
            return
        for item in raw:
            if not isinstance(item, dict):
                continue
            code = str(item.get("code", "unknown"))
            message = str(item.get("message", ""))
            key = (code, message)
            if key in seen:
                continue
            seen.add(key)
            rows.append({"code": code, "message": message})

    if backtest_summary is not None:
        _append(backtest_summary.get("warnings"))
    if adapter_run_metadata is not None:
        _append(adapter_run_metadata.get("warnings"))
    return tuple(rows)


def _build_flags(
    *,
    reason_summary_df: pd.DataFrame,
    dominant_execution_blocker: str | None,
    deviation_summary: dict[str, object],
    turnover_summary: dict[str, object],
    thresholds: ExecutionImpactThresholds,
) -> list[ExecutionImpactFlag]:
    reason_ratio = {
        str(row.reason_code): float(row.skipped_order_ratio)
        for row in reason_summary_df.itertuples(index=False)
    }
    mean_abs_diff = _coerce_float_or_none(deviation_summary.get("mean_abs_weight_diff"))
    skipped_ratio = _coerce_float_or_none(turnover_summary.get("skipped_order_ratio"))
    price_limit_ratio = reason_ratio.get("price_limit_locked", 0.0)
    liquidity_ratio = reason_ratio.get("min_adv_filter", 0.0)
    reentry_ratio = reason_ratio.get("same_day_reentry_blocked", 0.0)

    return [
        ExecutionImpactFlag(
            name="dominant_execution_blocker",
            triggered=dominant_execution_blocker is not None,
            observed=dominant_execution_blocker,
            threshold=None,
            description="Most frequent normalized reason among skipped orders",
        ),
        ExecutionImpactFlag(
            name="high_execution_deviation",
            triggered=None
            if mean_abs_diff is None
            else mean_abs_diff >= thresholds.high_execution_deviation_mean_abs,
            observed=mean_abs_diff,
            threshold=thresholds.high_execution_deviation_mean_abs,
            description="Mean absolute target-vs-realized weight diff exceeds threshold",
        ),
        ExecutionImpactFlag(
            name="severe_tradability_constraints",
            triggered=None
            if skipped_ratio is None
            else skipped_ratio >= thresholds.severe_tradability_skipped_ratio,
            observed=skipped_ratio,
            threshold=thresholds.severe_tradability_skipped_ratio,
            description="Skipped orders share among attempted orders exceeds threshold",
        ),
        ExecutionImpactFlag(
            name="price_limit_sensitive",
            triggered=price_limit_ratio >= thresholds.price_limit_reason_ratio,
            observed=price_limit_ratio,
            threshold=thresholds.price_limit_reason_ratio,
            description="price_limit_locked ratio in skipped orders exceeds threshold",
        ),
        ExecutionImpactFlag(
            name="liquidity_sensitive",
            triggered=liquidity_ratio >= thresholds.liquidity_reason_ratio,
            observed=liquidity_ratio,
            threshold=thresholds.liquidity_reason_ratio,
            description="min_adv_filter ratio in skipped orders exceeds threshold",
        ),
        ExecutionImpactFlag(
            name="reentry_constraint_sensitive",
            triggered=reentry_ratio >= thresholds.reentry_reason_ratio,
            observed=reentry_ratio,
            threshold=thresholds.reentry_reason_ratio,
            description="same_day_reentry_blocked ratio in skipped orders exceeds threshold",
        ),
    ]


def _comparison_summary(
    *,
    primary: LoadedExecutionArtifacts,
    comparison: LoadedExecutionArtifacts,
) -> dict[str, object]:
    unavailable: list[MetricUnavailable] = []
    target_equal: bool | None = None
    if primary.target_weights_df is not None and comparison.target_weights_df is not None:
        left, left_err = _prepare_weight_frame(primary.target_weights_df)
        right, right_err = _prepare_weight_frame(comparison.target_weights_df)
        if left_err is not None or right_err is not None or left is None or right is None:
            unavailable.append(
                MetricUnavailable(
                    metric="comparison.target_weights_equal",
                    reason=left_err or right_err or "cannot normalize weight tables",
                )
            )
        else:
            target_equal = left.equals(right)
    else:
        unavailable.append(
            MetricUnavailable(
                metric="comparison.target_weights_equal",
                reason="target_weights.csv missing on one or both sides",
            )
        )

    primary_dev, _ = _execution_deviation_summary(
        primary.target_weights_df,
        primary.executed_weights_df,
        unavailable=[],
    )
    comparison_dev, _ = _execution_deviation_summary(
        comparison.target_weights_df,
        comparison.executed_weights_df,
        unavailable=[],
    )
    primary_perf, primary_warnings = _performance_context(
        primary.backtest_summary,
        primary.adapter_run_metadata,
        unavailable=[],
    )
    comparison_perf, comparison_warnings = _performance_context(
        comparison.backtest_summary,
        comparison.adapter_run_metadata,
        unavailable=[],
    )

    primary_engine = primary.engine_name
    comparison_engine = comparison.engine_name
    notes: list[str] = []
    if {primary_engine, comparison_engine} == {"backtrader", "vectorbt"}:
        notes.append(
            "Backtrader replay applies stricter execution gating (tradability/price-limit/"
            "reentry) than vectorbt v1; realized exposure and turnover may diverge."
        )

    return {
        "comparison_run_path": str(comparison.run_path),
        "primary_engine": primary_engine,
        "comparison_engine": comparison_engine,
        "target_weights_equal": target_equal,
        "key_metric_comparison": {
            "total_return": {
                "primary": primary_perf.get("total_return"),
                "comparison": comparison_perf.get("total_return"),
            },
            "sharpe_annualized": {
                "primary": primary_perf.get("sharpe_annualized"),
                "comparison": comparison_perf.get("sharpe_annualized"),
            },
            "max_drawdown": {
                "primary": primary_perf.get("max_drawdown"),
                "comparison": comparison_perf.get("max_drawdown"),
            },
            "mean_turnover": {
                "primary": primary_perf.get("mean_turnover"),
                "comparison": comparison_perf.get("mean_turnover"),
            },
        },
        "execution_deviation_comparison": {
            "mean_abs_weight_diff": {
                "primary": primary_dev.get("mean_abs_weight_diff"),
                "comparison": comparison_dev.get("mean_abs_weight_diff"),
            },
            "max_abs_weight_diff": {
                "primary": primary_dev.get("max_abs_weight_diff"),
                "comparison": comparison_dev.get("max_abs_weight_diff"),
            },
        },
        "warning_codes": {
            "primary": sorted({str(item.get("code")) for item in primary_warnings}),
            "comparison": sorted({str(item.get("code")) for item in comparison_warnings}),
        },
        "notes": notes,
        "unavailable_metrics": [asdict(item) for item in _dedupe_unavailable(unavailable)],
    }


def _dedupe_unavailable(items: list[MetricUnavailable]) -> list[MetricUnavailable]:
    seen: set[tuple[str, str]] = set()
    out: list[MetricUnavailable] = []
    for item in items:
        key = (item.metric, item.reason)
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _df_records(df: pd.DataFrame) -> list[dict[str, object]]:
    if df.empty:
        return []
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%Y-%m-%d")
    records = out.to_dict(orient="records")
    return [_clean_dict(record) for record in records]


def _clean_dict(payload: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, (np.integer, int)):
            out[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            out[key] = _maybe_float(value)
        else:
            out[key] = value
    return out


def _maybe_float(value: object) -> float | None:
    try:
        coerced = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if np.isnan(coerced) or np.isinf(coerced):
        return None
    return coerced


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return None


def _coerce_float_or_none(value: object) -> float | None:
    if value is None:
        return None
    return _maybe_float(value)
