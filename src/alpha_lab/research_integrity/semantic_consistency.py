"""Experimental semantic-audit helpers for research-vs-replay comparison.

Core research integrity checks remain in `asof.py`, `leakage_checks.py`, and
`reporting.py`. This module is reserved for optional Level 3 replay auditing.
It compares declared semantics between research and replay artifacts; it does
not enforce Level 1/2 temporal leakage rules.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from alpha_lab.research_integrity.contracts import (
    IntegrityCheckResult,
    IntegrityReport,
    IntegritySeverity,
    IntegrityStatus,
    utc_now_iso,
)
from alpha_lab.research_integrity.exceptions import raise_on_hard_failures
from alpha_lab.research_integrity.reporting import build_integrity_report

if TYPE_CHECKING:
    from alpha_lab.backtest_adapter.schema import BacktestInputBundle, BacktestRunConfig

SEMANTIC_CONSISTENCY_REPORT_SCHEMA_VERSION = "1.0.0"

SemanticMateriality = Literal["none", "minor", "material", "invalidating"]


@dataclass(frozen=True)
class ExecutionSemanticsContract:
    """Compact, comparable timing/portfolio/execution semantics summary."""

    source_layer: str
    signal_bar_timestamp: str | None = None
    signal_computed_at: str | None = None
    bar_close_known_at: str | None = None
    order_submitted_at: str | None = None
    execution_bar_timestamp: str | None = None
    execution_price_rule: str | None = None
    signal_timestamp_convention: str | None = None
    portfolio_formation_timestamp: str | None = None
    rebalance_interval_bars: int | None = None
    holding_period_bars: int | None = None
    rebalance_frequency: int | None = None
    rebalance_calendar: str | None = None
    holding_period: int | None = None
    uses_closed_bar_only: bool | None = None
    allows_incomplete_bar_features: bool | None = None
    source_bar_frequency: str | None = None
    target_bar_frequency: str | None = None
    aggregation_timestamp_convention: str | None = None
    aggregated_value_known_at: str | None = None
    completed_aggregation_only: bool | None = None
    intraday_daily_alignment_rule: str | None = None
    daily_feature_effective_time: str | None = None
    target_weight_rule_summary: str | None = None
    execution_delay_bars: int | None = None
    fill_price_rule: str | None = None
    session_boundary_policy: str | None = None
    session_gap_policy: str | None = None
    lunch_break_policy: str | None = None
    allow_eod_bar_execution: bool | None = None
    tradability_rule_summary: str | None = None
    tradability_mask_applied: bool | None = None
    capacity_model: str | None = None
    max_participation_rate: float | None = None
    min_tradable_bar_volume: float | None = None
    liquidity_assumption_summary: str | None = None
    cost_model_summary: str | None = None
    benchmark_neutrality_summary: str | None = None
    adjustment_mode: str | None = None
    target_to_execution_interpretation: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_layer": self.source_layer,
            "signal_bar_timestamp": self.signal_bar_timestamp,
            "signal_computed_at": self.signal_computed_at,
            "bar_close_known_at": self.bar_close_known_at,
            "order_submitted_at": self.order_submitted_at,
            "execution_bar_timestamp": self.execution_bar_timestamp,
            "execution_price_rule": self.execution_price_rule,
            "signal_timestamp_convention": self.signal_timestamp_convention,
            "portfolio_formation_timestamp": self.portfolio_formation_timestamp,
            "rebalance_interval_bars": self.rebalance_interval_bars,
            "holding_period_bars": self.holding_period_bars,
            "rebalance_frequency": self.rebalance_frequency,
            "rebalance_calendar": self.rebalance_calendar,
            "holding_period": self.holding_period,
            "uses_closed_bar_only": self.uses_closed_bar_only,
            "allows_incomplete_bar_features": self.allows_incomplete_bar_features,
            "source_bar_frequency": self.source_bar_frequency,
            "target_bar_frequency": self.target_bar_frequency,
            "aggregation_timestamp_convention": self.aggregation_timestamp_convention,
            "aggregated_value_known_at": self.aggregated_value_known_at,
            "completed_aggregation_only": self.completed_aggregation_only,
            "intraday_daily_alignment_rule": self.intraday_daily_alignment_rule,
            "daily_feature_effective_time": self.daily_feature_effective_time,
            "target_weight_rule_summary": self.target_weight_rule_summary,
            "execution_delay_bars": self.execution_delay_bars,
            "fill_price_rule": self.fill_price_rule,
            "session_boundary_policy": self.session_boundary_policy,
            "session_gap_policy": self.session_gap_policy,
            "lunch_break_policy": self.lunch_break_policy,
            "allow_eod_bar_execution": self.allow_eod_bar_execution,
            "tradability_rule_summary": self.tradability_rule_summary,
            "tradability_mask_applied": self.tradability_mask_applied,
            "capacity_model": self.capacity_model,
            "max_participation_rate": self.max_participation_rate,
            "min_tradable_bar_volume": self.min_tradable_bar_volume,
            "liquidity_assumption_summary": self.liquidity_assumption_summary,
            "cost_model_summary": self.cost_model_summary,
            "benchmark_neutrality_summary": self.benchmark_neutrality_summary,
            "adjustment_mode": self.adjustment_mode,
            "target_to_execution_interpretation": self.target_to_execution_interpretation,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExecutionSemanticsContract:
        return cls(
            source_layer=_safe_str(payload.get("source_layer")) or "unknown",
            signal_bar_timestamp=_safe_str(payload.get("signal_bar_timestamp")),
            signal_computed_at=_safe_str(payload.get("signal_computed_at")),
            bar_close_known_at=_safe_str(payload.get("bar_close_known_at")),
            order_submitted_at=_safe_str(payload.get("order_submitted_at")),
            execution_bar_timestamp=_safe_str(payload.get("execution_bar_timestamp")),
            execution_price_rule=_safe_str(payload.get("execution_price_rule")),
            signal_timestamp_convention=_safe_str(payload.get("signal_timestamp_convention")),
            portfolio_formation_timestamp=_safe_str(payload.get("portfolio_formation_timestamp")),
            rebalance_interval_bars=_coerce_int(payload.get("rebalance_interval_bars")),
            holding_period_bars=_coerce_int(payload.get("holding_period_bars")),
            rebalance_frequency=_coerce_int(payload.get("rebalance_frequency")),
            rebalance_calendar=_safe_str(payload.get("rebalance_calendar")),
            holding_period=_coerce_int(payload.get("holding_period")),
            uses_closed_bar_only=_coerce_bool(payload.get("uses_closed_bar_only")),
            allows_incomplete_bar_features=_coerce_bool(
                payload.get("allows_incomplete_bar_features")
            ),
            source_bar_frequency=_safe_str(payload.get("source_bar_frequency")),
            target_bar_frequency=_safe_str(payload.get("target_bar_frequency")),
            aggregation_timestamp_convention=_safe_str(
                payload.get("aggregation_timestamp_convention")
            ),
            aggregated_value_known_at=_safe_str(payload.get("aggregated_value_known_at")),
            completed_aggregation_only=_coerce_bool(payload.get("completed_aggregation_only")),
            intraday_daily_alignment_rule=_safe_str(payload.get("intraday_daily_alignment_rule")),
            daily_feature_effective_time=_safe_str(payload.get("daily_feature_effective_time")),
            target_weight_rule_summary=_safe_str(payload.get("target_weight_rule_summary")),
            execution_delay_bars=_coerce_int(payload.get("execution_delay_bars")),
            fill_price_rule=_safe_str(payload.get("fill_price_rule")),
            session_boundary_policy=_safe_str(payload.get("session_boundary_policy")),
            session_gap_policy=_safe_str(payload.get("session_gap_policy")),
            lunch_break_policy=_safe_str(payload.get("lunch_break_policy")),
            allow_eod_bar_execution=_coerce_bool(payload.get("allow_eod_bar_execution")),
            tradability_rule_summary=_safe_str(payload.get("tradability_rule_summary")),
            tradability_mask_applied=_coerce_bool(payload.get("tradability_mask_applied")),
            capacity_model=_safe_str(payload.get("capacity_model")),
            max_participation_rate=_coerce_float(payload.get("max_participation_rate")),
            min_tradable_bar_volume=_coerce_float(payload.get("min_tradable_bar_volume")),
            liquidity_assumption_summary=_safe_str(payload.get("liquidity_assumption_summary")),
            cost_model_summary=_safe_str(payload.get("cost_model_summary")),
            benchmark_neutrality_summary=_safe_str(payload.get("benchmark_neutrality_summary")),
            adjustment_mode=_safe_str(payload.get("adjustment_mode")),
            target_to_execution_interpretation=_safe_str(
                payload.get("target_to_execution_interpretation")
            ),
            metadata=dict(_coerce_mapping(payload.get("metadata"))),
        )


@dataclass(frozen=True)
class SemanticFieldComparison:
    """One field-level research-vs-replay comparison result."""

    field_name: str
    research_value: object | None
    replay_value: object | None
    status: IntegrityStatus
    severity: IntegritySeverity
    materiality: SemanticMateriality
    message: str
    remediation: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "field_name": self.field_name,
            "research_value": _jsonable(self.research_value),
            "replay_value": _jsonable(self.replay_value),
            "status": self.status,
            "severity": self.severity,
            "materiality": self.materiality,
            "message": self.message,
            "remediation": self.remediation,
        }


@dataclass(frozen=True)
class SemanticConsistencyReport:
    """Serializable semantic consistency audit payload."""

    schema_version: str
    generated_at_utc: str
    research_semantics: ExecutionSemanticsContract
    replay_semantics: ExecutionSemanticsContract
    field_comparisons: tuple[SemanticFieldComparison, ...]
    integrity_report: IntegrityReport
    context: dict[str, object] = field(default_factory=dict)

    @property
    def is_compatible(self) -> bool:
        return self.integrity_report.summary.n_fail == 0

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_at_utc": self.generated_at_utc,
            "context": dict(self.context),
            "research_semantics": self.research_semantics.to_dict(),
            "replay_semantics": self.replay_semantics.to_dict(),
            "field_comparisons": [x.to_dict() for x in self.field_comparisons],
            "integrity_report": self.integrity_report.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SemanticConsistencyReport:
        report_payload = _coerce_mapping(payload.get("integrity_report"))
        checks_payload = report_payload.get("checks")
        checks: tuple[IntegrityCheckResult, ...]
        if isinstance(checks_payload, list):
            parsed_checks: list[IntegrityCheckResult] = []
            for raw in checks_payload:
                row = _coerce_mapping(raw)
                if not row:
                    continue
                parsed_checks.append(
                    IntegrityCheckResult(
                        check_name=_safe_str(row.get("check_name"))
                        or "semantic_consistency_unknown_check",
                        status=_safe_status(row.get("status")),
                        severity=_safe_severity(row.get("severity")),
                        message=_safe_str(row.get("message")) or "semantic check result",
                        object_name=_safe_str(row.get("object_name")),
                        module_name=_safe_str(row.get("module_name")),
                        remediation=_safe_str(row.get("remediation")),
                        metrics=dict(_coerce_mapping(row.get("metrics"))),
                    )
                )
            checks = tuple(parsed_checks)
        else:
            checks = ()

        integrity_report = build_integrity_report(
            checks,
            context=dict(_coerce_mapping(report_payload.get("context"))),
        )

        field_rows = payload.get("field_comparisons")
        comparisons: list[SemanticFieldComparison] = []
        if isinstance(field_rows, list):
            for raw in field_rows:
                row = _coerce_mapping(raw)
                if not row:
                    continue
                comparisons.append(
                    SemanticFieldComparison(
                        field_name=_safe_str(row.get("field_name")) or "unknown",
                        research_value=row.get("research_value"),
                        replay_value=row.get("replay_value"),
                        status=_safe_status(row.get("status")),
                        severity=_safe_severity(row.get("severity")),
                        materiality=_safe_materiality(row.get("materiality")),
                        message=_safe_str(row.get("message")) or "comparison result",
                        remediation=_safe_str(row.get("remediation")),
                    )
                )

        return cls(
            schema_version=_safe_str(payload.get("schema_version"))
            or SEMANTIC_CONSISTENCY_REPORT_SCHEMA_VERSION,
            generated_at_utc=_safe_str(payload.get("generated_at_utc")) or utc_now_iso(),
            research_semantics=ExecutionSemanticsContract.from_dict(
                _coerce_mapping(payload.get("research_semantics"))
            ),
            replay_semantics=ExecutionSemanticsContract.from_dict(
                _coerce_mapping(payload.get("replay_semantics"))
            ),
            field_comparisons=tuple(comparisons),
            integrity_report=integrity_report,
            context=dict(_coerce_mapping(payload.get("context"))),
        )


@dataclass(frozen=True)
class _FieldRule:
    mismatch_status: IntegrityStatus
    mismatch_severity: IntegritySeverity
    mismatch_materiality: SemanticMateriality
    remediation: str
    unknown_status: IntegrityStatus = "warn"
    unknown_severity: IntegritySeverity = "warning"
    unknown_materiality: SemanticMateriality = "minor"


_FIELD_ORDER: tuple[str, ...] = (
    "signal_bar_timestamp",
    "signal_computed_at",
    "bar_close_known_at",
    "order_submitted_at",
    "execution_bar_timestamp",
    "execution_price_rule",
    "signal_timestamp_convention",
    "portfolio_formation_timestamp",
    "rebalance_interval_bars",
    "holding_period_bars",
    "rebalance_frequency",
    "rebalance_calendar",
    "holding_period",
    "uses_closed_bar_only",
    "allows_incomplete_bar_features",
    "source_bar_frequency",
    "target_bar_frequency",
    "aggregation_timestamp_convention",
    "aggregated_value_known_at",
    "completed_aggregation_only",
    "intraday_daily_alignment_rule",
    "daily_feature_effective_time",
    "target_weight_rule_summary",
    "execution_delay_bars",
    "fill_price_rule",
    "session_boundary_policy",
    "session_gap_policy",
    "lunch_break_policy",
    "allow_eod_bar_execution",
    "tradability_rule_summary",
    "tradability_mask_applied",
    "capacity_model",
    "max_participation_rate",
    "min_tradable_bar_volume",
    "liquidity_assumption_summary",
    "cost_model_summary",
    "benchmark_neutrality_summary",
    "adjustment_mode",
    "target_to_execution_interpretation",
)

_FIELD_RULES: dict[str, _FieldRule] = {
    "signal_bar_timestamp": _FieldRule(
        mismatch_status="fail",
        mismatch_severity="error",
        mismatch_materiality="material",
        remediation="Align signal bar timestamp semantics across research and replay.",
    ),
    "signal_computed_at": _FieldRule(
        mismatch_status="fail",
        mismatch_severity="error",
        mismatch_materiality="material",
        remediation="Use one explicit signal computation time convention.",
    ),
    "bar_close_known_at": _FieldRule(
        mismatch_status="fail",
        mismatch_severity="error",
        mismatch_materiality="material",
        remediation="Ensure bar-close known-at timing is consistent across layers.",
    ),
    "order_submitted_at": _FieldRule(
        mismatch_status="fail",
        mismatch_severity="error",
        mismatch_materiality="material",
        remediation="Align order submission timing semantics between research and replay.",
    ),
    "execution_bar_timestamp": _FieldRule(
        mismatch_status="fail",
        mismatch_severity="error",
        mismatch_materiality="material",
        remediation="Align execution bar timestamp semantics (t+1, t+2, etc.).",
    ),
    "execution_price_rule": _FieldRule(
        mismatch_status="fail",
        mismatch_severity="error",
        mismatch_materiality="invalidating",
        remediation="Use one explicit execution price rule across research and replay.",
    ),
    "signal_timestamp_convention": _FieldRule(
        mismatch_status="fail",
        mismatch_severity="error",
        mismatch_materiality="material",
        remediation="Use one explicit signal timestamp convention across research and replay.",
    ),
    "portfolio_formation_timestamp": _FieldRule(
        mismatch_status="fail",
        mismatch_severity="error",
        mismatch_materiality="material",
        remediation="Align portfolio formation timestamp semantics between layers.",
    ),
    "rebalance_interval_bars": _FieldRule(
        mismatch_status="fail",
        mismatch_severity="error",
        mismatch_materiality="material",
        remediation="Align rebalance interval in bars across research and replay.",
    ),
    "holding_period_bars": _FieldRule(
        mismatch_status="fail",
        mismatch_severity="error",
        mismatch_materiality="material",
        remediation="Align holding period in bars across research and replay.",
    ),
    "rebalance_frequency": _FieldRule(
        mismatch_status="fail",
        mismatch_severity="error",
        mismatch_materiality="material",
        remediation="Ensure research and replay use the same rebalance frequency.",
    ),
    "rebalance_calendar": _FieldRule(
        mismatch_status="fail",
        mismatch_severity="error",
        mismatch_materiality="material",
        remediation="Ensure rebalance calendar semantics match exactly.",
    ),
    "holding_period": _FieldRule(
        mismatch_status="fail",
        mismatch_severity="error",
        mismatch_materiality="material",
        remediation="Align holding-period assumptions or document intentional differences.",
    ),
    "uses_closed_bar_only": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="minor",
        remediation=(
            "Align declared closed-bar assumptions to keep replay comparison interpretable."
        ),
    ),
    "allows_incomplete_bar_features": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="minor",
        remediation=(
            "Align declared incomplete-bar usage assumptions before interpreting replay drift."
        ),
    ),
    "source_bar_frequency": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation="Record source bar frequency consistently for cross-timeframe auditing.",
    ),
    "target_bar_frequency": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation="Record target/feature bar frequency consistently for aggregation checks.",
    ),
    "aggregation_timestamp_convention": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation="Align declared aggregation timestamp convention (bar_start vs bar_end).",
    ),
    "aggregated_value_known_at": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation="Align declared aggregation-known-at semantics across layers.",
    ),
    "completed_aggregation_only": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation="Align declared completed-aggregation assumptions across layers.",
    ),
    "intraday_daily_alignment_rule": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation="Align declared intraday-to-daily effective-time rules across layers.",
    ),
    "daily_feature_effective_time": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation="Record daily feature effective-time semantics explicitly.",
    ),
    "target_weight_rule_summary": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation=(
            "Audit target-weight construction assumptions before interpreting replay drift."
        ),
    ),
    "execution_delay_bars": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation=(
            "Align execution_delay_bars or treat replay as a delayed-execution stress test."
        ),
    ),
    "fill_price_rule": _FieldRule(
        mismatch_status="fail",
        mismatch_severity="error",
        mismatch_materiality="invalidating",
        remediation=(
            "Replay fill rule must match research fill intent for valid "
            "apples-to-apples comparisons."
        ),
    ),
    "session_boundary_policy": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation="Align session-boundary execution handling between research and replay.",
    ),
    "session_gap_policy": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation="Align session gap handling assumptions across layers.",
    ),
    "lunch_break_policy": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="minor",
        remediation="Record lunch-break handling explicitly when markets have midday breaks.",
    ),
    "allow_eod_bar_execution": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation="Align end-of-day bar tradability assumptions between research and replay.",
    ),
    "tradability_rule_summary": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation=(
            "Align tradability masking and skip policies or label output as "
            "non-equivalent."
        ),
    ),
    "tradability_mask_applied": _FieldRule(
        mismatch_status="fail",
        mismatch_severity="error",
        mismatch_materiality="material",
        remediation="Replay must respect tradability masking assumptions used in research.",
    ),
    "capacity_model": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation="Align and declare capacity model assumptions explicitly.",
    ),
    "max_participation_rate": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation="Align max participation assumptions for comparable execution realism.",
    ),
    "min_tradable_bar_volume": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation="Align minimum tradable bar-volume thresholds.",
    ),
    "liquidity_assumption_summary": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation=(
            "Declare whether execution assumes unlimited or capacity-constrained liquidity."
        ),
    ),
    "cost_model_summary": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation="Align commission/slippage modeling assumptions before comparing PnL levels.",
    ),
    "benchmark_neutrality_summary": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation="Align benchmark/neutrality assumptions to avoid interpretation drift.",
    ),
    "adjustment_mode": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="minor",
        remediation="Record one adjustment mode explicitly in both research and replay artifacts.",
    ),
    "target_to_execution_interpretation": _FieldRule(
        mismatch_status="warn",
        mismatch_severity="warning",
        mismatch_materiality="material",
        remediation="Ensure target-vs-executed interpretation is explicit and consistent.",
    ),
}


def extract_research_semantics_from_bundle(
    bundle: BacktestInputBundle,
) -> ExecutionSemanticsContract:
    """Extract research-layer execution semantics from one handoff bundle."""

    timing = _coerce_mapping(bundle.timing_payload)
    delay_spec = _coerce_mapping(timing.get("delay_spec"))
    exp_meta = _coerce_mapping(bundle.experiment_metadata_payload)
    validation = _coerce_mapping(bundle.validation_context_payload)
    strategy_meta = _coerce_mapping(exp_meta.get("strategy"))

    signal_timestamp = _first_non_empty(
        timing.get("signal_timestamp_convention"),
        timing.get("signal_timestamp"),
        timing.get("signal_time_convention"),
        validation.get("signal_timestamp_convention"),
    )
    formation_timestamp = _first_non_empty(
        timing.get("portfolio_formation_timestamp"),
        timing.get("formation_timestamp"),
        validation.get("portfolio_formation_timestamp"),
    )
    holding_period = _coerce_int(
        _first_non_empty_obj(
            timing.get("holding_period"),
            timing.get("holding_period_bars"),
            delay_spec.get("holding_period"),
            exp_meta.get("holding_period"),
            strategy_meta.get("holding_period"),
        )
    )
    uses_closed_bar_only = _coerce_bool(
        _first_non_empty_obj(
            timing.get("uses_closed_bar_only"),
            validation.get("uses_closed_bar_only"),
        )
    )
    allows_incomplete_bar_features = _coerce_bool(
        _first_non_empty_obj(
            timing.get("allows_incomplete_bar_features"),
            validation.get("allows_incomplete_bar_features"),
        )
    )
    if uses_closed_bar_only is None:
        uses_closed_bar_only = True
    if allows_incomplete_bar_features is None:
        allows_incomplete_bar_features = False

    pc = bundle.portfolio_construction
    ea = bundle.execution_assumptions

    benchmark = _first_non_empty(exp_meta.get("benchmark"), validation.get("benchmark"))
    neutrality_summary = f"neutralization_required={pc.neutralization_required}"
    if benchmark is not None:
        neutrality_summary = f"{neutrality_summary}; benchmark={benchmark}"

    adjustment_mode = _first_non_empty(
        timing.get("adjustment_mode"),
        _coerce_mapping(bundle.dataset_fingerprint_payload).get("adjustment_mode"),
        exp_meta.get("adjustment_mode"),
    )
    signal_bar_timestamp = _first_non_empty(
        timing.get("signal_bar_timestamp"),
        timing.get("signal_timestamp_convention"),
        timing.get("signal_timestamp"),
    )
    signal_computed_at = _first_non_empty(
        timing.get("signal_computed_at"),
        timing.get("portfolio_formation_timestamp"),
    )
    execution_bar_timestamp = _first_non_empty(
        timing.get("execution_bar_timestamp"),
        delay_spec.get("execution_bar_timestamp"),
    )
    execution_price_rule = _first_non_empty(
        timing.get("execution_price_rule"),
        ea.fill_price_rule,
    )
    source_bar_frequency = _first_non_empty(
        timing.get("source_bar_frequency"),
        validation.get("source_bar_frequency"),
    )
    target_bar_frequency = _first_non_empty(
        timing.get("target_bar_frequency"),
        validation.get("target_bar_frequency"),
    )
    capacity_mode = _safe_str(ea.capacity_model)
    liquidity_assumption = _liquidity_assumption_summary(
        capacity_model=capacity_mode,
        max_participation_rate=ea.max_participation_rate,
        min_tradable_bar_volume=ea.min_tradable_bar_volume,
    )
    completed_aggregation_only = _coerce_bool(timing.get("completed_aggregation_only"))
    if completed_aggregation_only is None:
        completed_aggregation_only = True

    return ExecutionSemanticsContract(
        source_layer="research",
        signal_bar_timestamp=signal_bar_timestamp,
        signal_computed_at=signal_computed_at,
        bar_close_known_at=_first_non_empty(timing.get("bar_close_known_at")),
        order_submitted_at=_first_non_empty(timing.get("order_submitted_at")),
        execution_bar_timestamp=execution_bar_timestamp,
        execution_price_rule=execution_price_rule,
        signal_timestamp_convention=signal_timestamp,
        portfolio_formation_timestamp=formation_timestamp,
        rebalance_interval_bars=int(pc.rebalance_frequency),
        holding_period_bars=holding_period,
        rebalance_frequency=int(pc.rebalance_frequency),
        rebalance_calendar=pc.rebalance_calendar,
        holding_period=holding_period,
        uses_closed_bar_only=uses_closed_bar_only,
        allows_incomplete_bar_features=allows_incomplete_bar_features,
        source_bar_frequency=source_bar_frequency,
        target_bar_frequency=target_bar_frequency,
        aggregation_timestamp_convention=_first_non_empty(
            timing.get("aggregation_timestamp_convention")
        ),
        aggregated_value_known_at=_first_non_empty(timing.get("aggregated_value_known_at")),
        completed_aggregation_only=completed_aggregation_only,
        intraday_daily_alignment_rule=_first_non_empty(
            timing.get("intraday_daily_alignment_rule")
        ),
        daily_feature_effective_time=_first_non_empty(
            timing.get("daily_feature_effective_time")
        ),
        target_weight_rule_summary=(
            f"construction_method={pc.construction_method};"
            f"weight_method={pc.weight_method};"
            f"long_short={pc.long_short};"
            f"top_k={pc.top_k};"
            f"bottom_k={pc.bottom_k};"
            f"gross_limit={pc.gross_limit};"
            f"net_limit={pc.net_limit};"
            f"cash_buffer={pc.cash_buffer}"
        ),
        execution_delay_bars=int(ea.execution_delay_bars),
        fill_price_rule=ea.fill_price_rule,
        session_boundary_policy=ea.session_boundary_policy,
        session_gap_policy=ea.session_gap_policy,
        lunch_break_policy=ea.lunch_break_policy,
        allow_eod_bar_execution=ea.allow_eod_bar_execution,
        tradability_rule_summary=(
            f"trade_when_not_tradable={ea.trade_when_not_tradable};"
            f"suspension_policy={ea.suspension_policy};"
            f"price_limit_policy={ea.price_limit_policy};"
            f"partial_fill_policy={ea.partial_fill_policy}"
        ),
        tradability_mask_applied=not bool(ea.trade_when_not_tradable),
        capacity_model=capacity_mode,
        max_participation_rate=ea.max_participation_rate,
        min_tradable_bar_volume=ea.min_tradable_bar_volume,
        liquidity_assumption_summary=liquidity_assumption,
        cost_model_summary=(
            f"commission_model={ea.commission_model};"
            f"slippage_model={ea.slippage_model};"
            f"lot_size_rule={ea.lot_size_rule};"
            f"cash_buffer={ea.cash_buffer}"
        ),
        benchmark_neutrality_summary=neutrality_summary,
        adjustment_mode=adjustment_mode,
        target_to_execution_interpretation=(
            "target_weights_from_signal_snapshot_then_shift_by_execution_delay_bars"
        ),
        metadata={
            "artifact_path": str(bundle.artifact_path),
            "experiment_id": bundle.experiment_id,
            "dataset_fingerprint": bundle.dataset_fingerprint,
        },
    )


def extract_replay_semantics_from_runtime(
    bundle: BacktestInputBundle,
    config: BacktestRunConfig,
    *,
    engine: str,
    mapping_assumptions: Mapping[str, object] | None = None,
) -> ExecutionSemanticsContract:
    """Extract replay-layer semantics from runtime adapter inputs."""

    mapping = _coerce_mapping(mapping_assumptions)
    research = extract_research_semantics_from_bundle(bundle)

    execution_delay = _coerce_int(mapping.get("execution_delay_bars_applied"))
    if execution_delay is None:
        execution_delay = int(bundle.execution_assumptions.execution_delay_bars)

    fill_price_source = _safe_str(mapping.get("fill_price_source"))
    replay_fill_rule = _effective_fill_price_rule(
        requested=bundle.execution_assumptions.fill_price_rule,
        fill_price_source=fill_price_source,
    )

    tradability_prefix = (
        "engine_enforcement=pre_trade_mask"
        if engine == "vectorbt"
        else "engine_enforcement=bar_replay"
    )
    replay_tradability = (
        f"{tradability_prefix};"
        f"trade_when_not_tradable={bundle.execution_assumptions.trade_when_not_tradable};"
        f"suspension_policy={bundle.execution_assumptions.suspension_policy};"
        f"price_limit_policy={bundle.execution_assumptions.price_limit_policy};"
        f"partial_fill_policy={bundle.execution_assumptions.partial_fill_policy}"
    )

    replay_target_interp = _first_non_empty(
        mapping.get("portfolio_construction_engine_input"),
        "target_weights",
    )
    if replay_target_interp is not None:
        replay_target_interp = (
            f"engine_input={replay_target_interp}; delay_applied={execution_delay}"
        )

    replay_target_rule = (
        f"engine={engine};"
        f"construction_method={bundle.portfolio_construction.construction_method};"
        f"weight_method={bundle.portfolio_construction.weight_method};"
        f"long_short={bundle.portfolio_construction.long_short};"
        f"top_k={bundle.portfolio_construction.top_k};"
        f"bottom_k={bundle.portfolio_construction.bottom_k}"
    )

    replay_cost_model = (
        f"commission_model={bundle.execution_assumptions.commission_model};"
        f"slippage_model={bundle.execution_assumptions.slippage_model};"
        f"lot_size_rule={bundle.execution_assumptions.lot_size_rule};"
        f"cash_buffer={bundle.execution_assumptions.cash_buffer}"
    )
    replay_tradability_mask_applied = _coerce_bool(mapping.get("tradability_mask_applied"))
    if replay_tradability_mask_applied is None:
        replay_tradability_mask_applied = not bool(
            bundle.execution_assumptions.trade_when_not_tradable
        )

    capacity_model = _first_non_empty(
        mapping.get("capacity_model_applied"),
        bundle.execution_assumptions.capacity_model,
    )
    max_participation_rate = _coerce_float(
        _first_non_empty_obj(
            mapping.get("max_participation_rate_applied"),
            bundle.execution_assumptions.max_participation_rate,
        )
    )
    min_tradable_bar_volume = _coerce_float(
        _first_non_empty_obj(
            mapping.get("min_tradable_bar_volume_applied"),
            bundle.execution_assumptions.min_tradable_bar_volume,
        )
    )
    liquidity_assumption = _liquidity_assumption_summary(
        capacity_model=_safe_str(capacity_model),
        max_participation_rate=max_participation_rate,
        min_tradable_bar_volume=min_tradable_bar_volume,
    )
    completed_aggregation_only = _coerce_bool(mapping.get("completed_aggregation_only"))
    if completed_aggregation_only is None:
        completed_aggregation_only = research.completed_aggregation_only

    return ExecutionSemanticsContract(
        source_layer="replay",
        signal_bar_timestamp=research.signal_bar_timestamp,
        signal_computed_at=research.signal_computed_at,
        bar_close_known_at=research.bar_close_known_at,
        order_submitted_at=_first_non_empty(
            mapping.get("order_submitted_at"),
            research.order_submitted_at,
        ),
        execution_bar_timestamp=_first_non_empty(
            mapping.get("execution_bar_timestamp"),
            research.execution_bar_timestamp,
        ),
        execution_price_rule=_first_non_empty(
            mapping.get("execution_price_rule"),
            replay_fill_rule,
        ),
        signal_timestamp_convention=research.signal_timestamp_convention,
        portfolio_formation_timestamp=research.portfolio_formation_timestamp,
        rebalance_interval_bars=int(bundle.portfolio_construction.rebalance_frequency),
        holding_period_bars=research.holding_period_bars,
        rebalance_frequency=int(bundle.portfolio_construction.rebalance_frequency),
        rebalance_calendar=bundle.portfolio_construction.rebalance_calendar,
        holding_period=research.holding_period,
        uses_closed_bar_only=research.uses_closed_bar_only,
        allows_incomplete_bar_features=research.allows_incomplete_bar_features,
        source_bar_frequency=_first_non_empty(
            mapping.get("source_bar_frequency"),
            research.source_bar_frequency,
        ),
        target_bar_frequency=_first_non_empty(
            mapping.get("target_bar_frequency"),
            research.target_bar_frequency,
        ),
        aggregation_timestamp_convention=_first_non_empty(
            mapping.get("aggregation_timestamp_convention"),
            research.aggregation_timestamp_convention,
        ),
        aggregated_value_known_at=_first_non_empty(
            mapping.get("aggregated_value_known_at"),
            research.aggregated_value_known_at,
        ),
        completed_aggregation_only=completed_aggregation_only,
        intraday_daily_alignment_rule=_first_non_empty(
            mapping.get("intraday_daily_alignment_rule"),
            research.intraday_daily_alignment_rule,
        ),
        daily_feature_effective_time=_first_non_empty(
            mapping.get("daily_feature_effective_time"),
            research.daily_feature_effective_time,
        ),
        target_weight_rule_summary=replay_target_rule,
        execution_delay_bars=execution_delay,
        fill_price_rule=replay_fill_rule,
        session_boundary_policy=_first_non_empty(
            mapping.get("session_boundary_policy"),
            bundle.execution_assumptions.session_boundary_policy,
        ),
        session_gap_policy=_first_non_empty(
            mapping.get("session_gap_policy"),
            bundle.execution_assumptions.session_gap_policy,
        ),
        lunch_break_policy=_first_non_empty(
            mapping.get("lunch_break_policy"),
            bundle.execution_assumptions.lunch_break_policy,
        ),
        allow_eod_bar_execution=_coerce_bool(
            _first_non_empty_obj(
                mapping.get("allow_eod_bar_execution"),
                bundle.execution_assumptions.allow_eod_bar_execution,
            )
        ),
        tradability_rule_summary=replay_tradability,
        tradability_mask_applied=replay_tradability_mask_applied,
        capacity_model=capacity_model,
        max_participation_rate=max_participation_rate,
        min_tradable_bar_volume=min_tradable_bar_volume,
        liquidity_assumption_summary=liquidity_assumption,
        cost_model_summary=replay_cost_model,
        benchmark_neutrality_summary=research.benchmark_neutrality_summary,
        adjustment_mode=research.adjustment_mode,
        target_to_execution_interpretation=replay_target_interp,
        metadata={
            "engine": engine,
            "freq": config.freq,
            "initial_cash": float(config.initial_cash),
            "commission_bps": float(config.commission_bps),
            "slippage_bps": float(config.slippage_bps),
            "fill_price_source": fill_price_source,
            "mapping_assumptions": dict(mapping),
        },
    )


def extract_semantic_consistency_report_from_metadata(
    adapter_run_metadata: Mapping[str, object] | None,
) -> SemanticConsistencyReport | None:
    """Read semantic report object from adapter metadata when present."""

    if adapter_run_metadata is None:
        return None
    payload = _coerce_mapping(adapter_run_metadata.get("semantic_consistency"))
    if not payload:
        return None
    try:
        return SemanticConsistencyReport.from_dict(payload)
    except (ValueError, TypeError, KeyError):
        return None


def compare_research_vs_replay_semantics(
    research_semantics: ExecutionSemanticsContract,
    replay_semantics: ExecutionSemanticsContract,
) -> tuple[SemanticFieldComparison, ...]:
    """Compare two semantics contracts and classify differences."""

    comparisons: list[SemanticFieldComparison] = []
    for field_name in _FIELD_ORDER:
        rule = _FIELD_RULES[field_name]
        research_value = getattr(research_semantics, field_name)
        replay_value = getattr(replay_semantics, field_name)

        if field_name in {"max_participation_rate", "min_tradable_bar_volume"}:
            if _is_missing(research_value) and _is_missing(replay_value):
                research_mode = (_safe_str(research_semantics.capacity_model) or "").lower()
                replay_mode = (_safe_str(replay_semantics.capacity_model) or "").lower()
                if field_name == "max_participation_rate":
                    needed_modes = {"participation_capped", "participation_capped_with_min_volume"}
                else:
                    needed_modes = {"min_bar_volume_gate", "participation_capped_with_min_volume"}
                if research_mode not in needed_modes and replay_mode not in needed_modes:
                    comparisons.append(
                        SemanticFieldComparison(
                            field_name=field_name,
                            research_value=research_value,
                            replay_value=replay_value,
                            status="pass",
                            severity="info",
                            materiality="none",
                            message=(
                                "field is intentionally omitted because active capacity model "
                                "does not require it"
                            ),
                        )
                    )
                    continue

        if _is_missing(research_value) or _is_missing(replay_value):
            comparisons.append(
                SemanticFieldComparison(
                    field_name=field_name,
                    research_value=research_value,
                    replay_value=replay_value,
                    status=rule.unknown_status,
                    severity=rule.unknown_severity,
                    materiality=rule.unknown_materiality,
                    message=(
                        "field is missing from one side; comparison is advisory until "
                        "both semantics are explicit"
                    ),
                    remediation=rule.remediation,
                )
            )
            continue

        if _equivalent_values(research_value, replay_value):
            comparisons.append(
                SemanticFieldComparison(
                    field_name=field_name,
                    research_value=research_value,
                    replay_value=replay_value,
                    status="pass",
                    severity="info",
                    materiality="none",
                    message="research and replay semantics are aligned for this field",
                    remediation=None,
                )
            )
            continue

        status = rule.mismatch_status
        severity = rule.mismatch_severity
        materiality = rule.mismatch_materiality

        if field_name == "execution_delay_bars":
            rv = _coerce_int(research_value)
            pv = _coerce_int(replay_value)
            if rv is not None and pv is not None and abs(rv - pv) >= 2:
                status = "fail"
                severity = "error"
                materiality = "material"
        if field_name == "tradability_rule_summary":
            rv = _safe_str(research_value) or ""
            pv = _safe_str(replay_value) or ""
            if (
                "trade_when_not_tradable=true" in rv.lower()
                or "trade_when_not_tradable=true" in pv.lower()
            ) and rv.lower() != pv.lower():
                status = "fail"
                severity = "error"
                materiality = "material"
        if field_name == "capacity_model":
            rv = (_safe_str(research_value) or "").lower()
            pv = (_safe_str(replay_value) or "").lower()
            if {
                rv,
                pv,
            } <= {"unbounded", "participation_capped", "participation_capped_with_min_volume"}:
                status = "warn"
                severity = "warning"
                materiality = "material"
        if field_name == "tradability_mask_applied":
            rv = _coerce_bool(research_value)
            pv = _coerce_bool(replay_value)
            if rv is True and pv is False:
                status = "fail"
                severity = "error"
                materiality = "material"

        comparisons.append(
            SemanticFieldComparison(
                field_name=field_name,
                research_value=research_value,
                replay_value=replay_value,
                status=status,
                severity=severity,
                materiality=materiality,
                message="research and replay semantics diverge for this field",
                remediation=rule.remediation,
            )
        )

    comparisons.extend(_derived_semantic_checks(research_semantics, replay_semantics))
    return tuple(comparisons)


def build_semantic_consistency_report(
    research_semantics: ExecutionSemanticsContract,
    replay_semantics: ExecutionSemanticsContract,
    *,
    context: Mapping[str, object] | None = None,
) -> SemanticConsistencyReport:
    """Build one semantic consistency report from two contracts."""

    comparisons = compare_research_vs_replay_semantics(research_semantics, replay_semantics)
    checks = tuple(_to_integrity_check(x) for x in comparisons)
    report_context = {
        "audit_layer": "research_vs_replay_semantic_consistency",
        "research_source": research_semantics.source_layer,
        "replay_source": replay_semantics.source_layer,
    }
    if context is not None:
        report_context.update(dict(context))
    integrity_report = build_integrity_report(checks, context=report_context)

    return SemanticConsistencyReport(
        schema_version=SEMANTIC_CONSISTENCY_REPORT_SCHEMA_VERSION,
        generated_at_utc=utc_now_iso(),
        research_semantics=research_semantics,
        replay_semantics=replay_semantics,
        field_comparisons=comparisons,
        integrity_report=integrity_report,
        context=dict(report_context),
    )


def assert_semantic_compatibility(
    report_or_checks: SemanticConsistencyReport
    | Iterable[IntegrityCheckResult]
    | Iterable[SemanticFieldComparison],
) -> None:
    """Raise when report/checks contain hard semantic incompatibilities."""

    if isinstance(report_or_checks, SemanticConsistencyReport):
        raise_on_hard_failures(report_or_checks.integrity_report.checks)
        return

    sample = list(report_or_checks)
    if not sample:
        return

    first = sample[0]
    if isinstance(first, IntegrityCheckResult):
        raise_on_hard_failures(sample)  # type: ignore[arg-type]
        return

    if isinstance(first, SemanticFieldComparison):
        checks = [_to_integrity_check(x) for x in sample]  # type: ignore[arg-type]
        raise_on_hard_failures(checks)


def summarize_semantic_report_payload(
    payload: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Extract compact semantic status summary from report payload."""

    if payload is None:
        return None
    integrity = _coerce_mapping(payload.get("integrity_report"))
    summary = _coerce_mapping(integrity.get("summary"))
    n_fail = _coerce_int(summary.get("n_fail")) or 0
    n_warn = _coerce_int(summary.get("n_warn")) or 0
    n_pass = _coerce_int(summary.get("n_pass")) or 0
    if n_fail > 0:
        overall = "fail"
    elif n_warn > 0:
        overall = "warn"
    else:
        overall = "pass"
    return {
        "status": overall,
        "n_checks": _coerce_int(summary.get("n_checks")) or (n_pass + n_warn + n_fail),
        "n_pass": n_pass,
        "n_warn": n_warn,
        "n_fail": n_fail,
        "highest_severity": _safe_str(summary.get("highest_severity")),
        "hard_failure_checks": list(summary.get("hard_failure_checks") or []),
    }


def write_semantic_consistency_report_json(
    report: SemanticConsistencyReport,
    output_path: str | Path,
) -> Path:
    """Write semantic consistency report JSON artifact."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def render_semantic_consistency_report_markdown(report: SemanticConsistencyReport) -> str:
    """Render semantic consistency report as reviewer-facing markdown."""

    lines: list[str] = [
        "# Semantic Consistency Report",
        "",
        "## Summary",
        "",
        f"- Generated (UTC): `{report.generated_at_utc}`",
        f"- Checks: `{report.integrity_report.summary.n_checks}`",
        f"- Pass: `{report.integrity_report.summary.n_pass}`",
        f"- Warn: `{report.integrity_report.summary.n_warn}`",
        f"- Fail: `{report.integrity_report.summary.n_fail}`",
        f"- Highest Severity: `{report.integrity_report.summary.highest_severity}`",
        "",
        "## Research Semantics",
        "",
    ]
    lines.extend(_contract_markdown_lines(report.research_semantics))

    lines.extend(
        [
            "",
            "## Replay Semantics",
            "",
        ]
    )
    lines.extend(_contract_markdown_lines(report.replay_semantics))

    lines.extend(
        [
            "",
            "## Field Comparison",
            "",
            "| Field | Status | Severity | Materiality | Research | Replay | Message |",
            "|---|---|---|---|---|---|---|",
        ]
    )

    for row in report.field_comparisons:
        lines.append(
            "| "
            f"`{row.field_name}` | `{row.status}` | `{row.severity}` | `{row.materiality}` | "
            f"`{_value_for_table(row.research_value)}` | `{_value_for_table(row.replay_value)}` | "
            f"{row.message} |"
        )

    return "\n".join(lines).rstrip() + "\n"


def write_semantic_consistency_report_markdown(
    report: SemanticConsistencyReport,
    output_path: str | Path,
) -> Path:
    """Write semantic consistency report markdown artifact."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_semantic_consistency_report_markdown(report), encoding="utf-8")
    return path


def export_semantic_consistency_report(
    report: SemanticConsistencyReport,
    output_dir: str | Path,
    *,
    json_name: str = "semantic_consistency_report.json",
    markdown_name: str = "semantic_consistency_report.md",
) -> dict[str, Path]:
    """Write semantic consistency report artifacts (JSON + markdown)."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_semantic_consistency_report_json(report, out_dir / json_name)
    md_path = write_semantic_consistency_report_markdown(report, out_dir / markdown_name)
    return {
        "json": json_path,
        "markdown": md_path,
    }


def _derived_semantic_checks(
    research: ExecutionSemanticsContract,
    replay: ExecutionSemanticsContract,
) -> list[SemanticFieldComparison]:
    out: list[SemanticFieldComparison] = []

    research_mode = (_safe_str(research.capacity_model) or "").lower()
    replay_mode = (_safe_str(replay.capacity_model) or "").lower()
    if research_mode and replay_mode and research_mode != replay_mode:
        out.append(
            SemanticFieldComparison(
                field_name="capacity_regime_consistency",
                research_value=research.capacity_model,
                replay_value=replay.capacity_model,
                status="warn",
                severity="warning",
                materiality="material",
                message=(
                    "research and replay use different liquidity-capacity regimes "
                    "(for example unbounded vs participation-capped)"
                ),
                remediation=(
                    "Label this as a stress test or align both sides to one explicit capacity "
                    "regime."
                ),
            )
        )
    else:
        out.append(
            SemanticFieldComparison(
                field_name="capacity_regime_consistency",
                research_value=research.capacity_model,
                replay_value=replay.capacity_model,
                status="pass",
                severity="info",
                materiality="none",
                message="capacity regime semantics are aligned",
            )
        )

    return out


def _to_integrity_check(row: SemanticFieldComparison) -> IntegrityCheckResult:
    return IntegrityCheckResult(
        check_name=f"semantic_{row.field_name}",
        status=row.status,
        severity=row.severity,
        object_name="research_vs_replay_semantics",
        module_name="research_integrity.semantic_consistency",
        message=row.message,
        remediation=row.remediation,
        metrics={
            "field": row.field_name,
            "materiality": row.materiality,
            "research_value": _jsonable(row.research_value),
            "replay_value": _jsonable(row.replay_value),
        },
    )


def _effective_fill_price_rule(requested: str | None, fill_price_source: str | None) -> str | None:
    req = _safe_str(requested)
    src = _safe_str(fill_price_source)
    if src is None:
        return req
    lowered = src.lower()
    if lowered == "open":
        return "next_open"
    if lowered == "close":
        return "next_close"
    return lowered


def _equivalent_values(left: object, right: object) -> bool:
    lb = _coerce_bool(left)
    rb = _coerce_bool(right)
    if lb is not None and rb is not None:
        return lb == rb

    if isinstance(left, str) and isinstance(right, str):
        return left.strip().lower() == right.strip().lower()
    return left == right


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _contract_markdown_lines(contract: ExecutionSemanticsContract) -> list[str]:
    rows = contract.to_dict()
    lines: list[str] = []
    for key in _FIELD_ORDER:
        lines.append(f"- {key}: `{_value_for_table(rows.get(key))}`")
    if contract.metadata:
        lines.append(f"- metadata: `{_value_for_table(contract.metadata)}`")
    return lines


def _value_for_table(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, dict):
        if not value:
            return "{}"
        flat = ", ".join(f"{k}={v}" for k, v in sorted(value.items(), key=lambda x: str(x[0])))
        return flat[:220]
    return str(value)


def _liquidity_assumption_summary(
    *,
    capacity_model: str | None,
    max_participation_rate: float | None,
    min_tradable_bar_volume: float | None,
) -> str:
    model = (capacity_model or "unbounded").strip().lower()
    if model == "unbounded":
        return "unlimited_liquidity"
    chunks = [f"capacity_model={model}"]
    if max_participation_rate is not None:
        chunks.append(f"max_participation_rate={max_participation_rate}")
    if min_tradable_bar_volume is not None:
        chunks.append(f"min_tradable_bar_volume={min_tradable_bar_volume}")
    return ";".join(chunks)


def _jsonable(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return str(value)


def _first_non_empty(*values: object) -> str | None:
    for value in values:
        text = _safe_str(value)
        if text is not None:
            return text
    return None


def _first_non_empty_obj(*values: object) -> object | None:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        return value
    return None


def _safe_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_bool(value: object) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = _safe_str(value)
    if text is None:
        return None
    lowered = text.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False
    return None


def _coerce_mapping(value: object) -> dict[str, object]:
    return dict(value) if isinstance(value, Mapping) else {}


def _safe_status(value: object) -> IntegrityStatus:
    raw = (_safe_str(value) or "").lower()
    if raw in {"pass", "warn", "fail"}:
        return raw  # type: ignore[return-value]
    return "warn"


def _safe_severity(value: object) -> IntegritySeverity:
    raw = (_safe_str(value) or "").lower()
    if raw in {"info", "warning", "error"}:
        return raw  # type: ignore[return-value]
    return "warning"


def _safe_materiality(value: object) -> SemanticMateriality:
    raw = (_safe_str(value) or "").lower()
    if raw in {"none", "minor", "material", "invalidating"}:
        return raw  # type: ignore[return-value]
    return "minor"
