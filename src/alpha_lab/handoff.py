from __future__ import annotations

import datetime
import hashlib
import json
import platform
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.experiment import ExperimentResult
from alpha_lab.research_contracts import validate_tradability_table, validate_universe_table
from alpha_lab.walk_forward import WalkForwardResult

HANDOFF_SCHEMA_VERSION = "2.0.0"
_LEGACY_HANDOFF_SCHEMA_VERSION = "1.0.0"
PORTFOLIO_CONSTRUCTION_SCHEMA_VERSION = "1.0.0"
EXECUTION_ASSUMPTIONS_SCHEMA_VERSION = "1.0.0"

_ALLOWED_PORTFOLIO_CONSTRUCTION_METHODS: frozenset[str] = frozenset(
    {"top_bottom_k", "full_universe"}
)
_ALLOWED_REBALANCE_CALENDARS: frozenset[str] = frozenset(
    {"business_day", "week_end", "month_end", "custom"}
)
_ALLOWED_WEIGHT_METHODS: frozenset[str] = frozenset({"equal", "rank", "score"})

_ALLOWED_FILL_PRICE_RULES: frozenset[str] = frozenset(
    {"next_open", "next_close", "vwap_next_bar"}
)
_ALLOWED_COMMISSION_MODELS: frozenset[str] = frozenset({"bps", "per_share", "flat"})
_ALLOWED_SLIPPAGE_MODELS: frozenset[str] = frozenset(
    {"none", "fixed_bps", "spread_plus_impact_proxy"}
)
_ALLOWED_LOT_SIZE_RULES: frozenset[str] = frozenset({"none", "round_to_lot"})
_ALLOWED_PARTIAL_FILL_POLICIES: frozenset[str] = frozenset(
    {"allow_partial", "cancel_unfilled", "fill_or_kill"}
)
_ALLOWED_SUSPENSION_POLICIES: frozenset[str] = frozenset(
    {"skip_trade", "defer_trade", "error"}
)
_ALLOWED_PRICE_LIMIT_POLICIES: frozenset[str] = frozenset(
    {"skip_trade", "defer_trade", "error"}
)

_SIGNAL_COLUMNS: tuple[str, ...] = ("date", "asset", "signal_name", "signal_value")
_UNIVERSE_COLUMNS: tuple[str, ...] = ("date", "asset", "in_universe")
_TRADABILITY_COLUMNS: tuple[str, ...] = ("date", "asset", "is_tradable")
_LABEL_COLUMNS: tuple[str, ...] = ("date", "asset", "label_name", "label_value")
_EXCLUSION_COLUMNS: tuple[str, ...] = ("date", "asset", "reason")

_REQUIRED_ARTIFACT_FILES_V1: tuple[str, ...] = (
    "signal_snapshot.csv",
    "universe_mask.csv",
    "tradability_mask.csv",
    "timing.json",
    "experiment_metadata.json",
    "validation_context.json",
    "dataset_fingerprint.json",
    "manifest.json",
)

_REQUIRED_ARTIFACT_FILES_V2: tuple[str, ...] = (
    "signal_snapshot.csv",
    "universe_mask.csv",
    "tradability_mask.csv",
    "timing.json",
    "experiment_metadata.json",
    "validation_context.json",
    "dataset_fingerprint.json",
    "portfolio_construction.json",
    "execution_assumptions.json",
    "manifest.json",
)

_MANIFEST_COMPONENT_KEYS_V2: tuple[str, ...] = (
    "research_artifact",
    "portfolio_construction",
    "execution_assumptions",
)


@dataclass(frozen=True)
class PortfolioConstructionSpec:
    """Contract describing signal-to-portfolio intent for external replay."""

    schema_version: str = PORTFOLIO_CONSTRUCTION_SCHEMA_VERSION
    construction_method: str = "top_bottom_k"
    signal_name: str | None = None
    rebalance_frequency: int = 1
    rebalance_calendar: str = "business_day"
    long_short: bool = True
    top_k: int | None = 20
    bottom_k: int | None = 20
    weight_method: str = "rank"
    weight_clip: float | None = None
    max_weight: float = 0.10
    min_weight: float | None = None
    gross_limit: float = 1.0
    net_limit: float = 0.0
    cash_buffer: float = 0.0
    neutralization_required: bool = False
    post_construction_constraints: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.schema_version != PORTFOLIO_CONSTRUCTION_SCHEMA_VERSION:
            raise ValueError(
                "portfolio construction schema_version must be "
                f"{PORTFOLIO_CONSTRUCTION_SCHEMA_VERSION!r}"
            )
        if self.construction_method not in _ALLOWED_PORTFOLIO_CONSTRUCTION_METHODS:
            raise ValueError(
                "construction_method must be one of "
                f"{sorted(_ALLOWED_PORTFOLIO_CONSTRUCTION_METHODS)}"
            )
        if self.rebalance_frequency <= 0:
            raise ValueError("rebalance_frequency must be > 0")
        if self.rebalance_calendar not in _ALLOWED_REBALANCE_CALENDARS:
            raise ValueError(
                "rebalance_calendar must be one of "
                f"{sorted(_ALLOWED_REBALANCE_CALENDARS)}"
            )
        if self.weight_method not in _ALLOWED_WEIGHT_METHODS:
            raise ValueError(
                f"weight_method must be one of {sorted(_ALLOWED_WEIGHT_METHODS)}"
            )
        if self.weight_clip is not None and (
            self.weight_clip <= 0.0 or self.weight_clip > 1.0
        ):
            raise ValueError("weight_clip must be in (0, 1] when provided")
        if self.max_weight <= 0.0:
            raise ValueError("max_weight must be > 0")
        if self.min_weight is not None and self.min_weight > self.max_weight:
            raise ValueError("min_weight must be <= max_weight")
        if self.gross_limit <= 0.0:
            raise ValueError("gross_limit must be > 0")
        if abs(self.net_limit) > self.gross_limit:
            raise ValueError("abs(net_limit) must be <= gross_limit")
        if self.cash_buffer < 0.0 or self.cash_buffer >= 1.0:
            raise ValueError("cash_buffer must be in [0, 1)")
        if self.construction_method == "top_bottom_k":
            if self.top_k is None or self.top_k <= 0:
                raise ValueError("top_k must be > 0 for top_bottom_k construction")
            if self.long_short:
                if self.bottom_k is None or self.bottom_k <= 0:
                    raise ValueError(
                        "bottom_k must be > 0 when long_short=True for top_bottom_k"
                    )
            elif self.bottom_k not in (None, 0):
                raise ValueError("bottom_k must be omitted for long-only construction")
        if self.construction_method == "full_universe":
            if self.top_k is not None or self.bottom_k is not None:
                raise ValueError(
                    "top_k/bottom_k must be omitted when construction_method='full_universe'"
                )
        if not self.long_short and self.net_limit < 0:
            raise ValueError("net_limit must be >= 0 for long-only construction")
        if self.signal_name is not None and not self.signal_name.strip():
            raise ValueError("signal_name must be non-empty when provided")
        for idx, constraint in enumerate(self.post_construction_constraints):
            if not isinstance(constraint, str) or not constraint.strip():
                raise ValueError(
                    f"post_construction_constraints[{idx}] must be a non-empty string"
                )

    def with_signal_name(self, signal_name: str) -> PortfolioConstructionSpec:
        if not signal_name.strip():
            raise ValueError("signal_name must be non-empty")
        return PortfolioConstructionSpec(
            schema_version=self.schema_version,
            construction_method=self.construction_method,
            signal_name=signal_name,
            rebalance_frequency=self.rebalance_frequency,
            rebalance_calendar=self.rebalance_calendar,
            long_short=self.long_short,
            top_k=self.top_k,
            bottom_k=self.bottom_k,
            weight_method=self.weight_method,
            weight_clip=self.weight_clip,
            max_weight=self.max_weight,
            min_weight=self.min_weight,
            gross_limit=self.gross_limit,
            net_limit=self.net_limit,
            cash_buffer=self.cash_buffer,
            neutralization_required=self.neutralization_required,
            post_construction_constraints=self.post_construction_constraints,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "construction_method": self.construction_method,
            "signal_name": self.signal_name,
            "rebalance_frequency": self.rebalance_frequency,
            "rebalance_calendar": self.rebalance_calendar,
            "long_short": self.long_short,
            "top_k": self.top_k,
            "bottom_k": self.bottom_k,
            "weight_method": self.weight_method,
            "weight_clip": self.weight_clip,
            "max_weight": self.max_weight,
            "min_weight": self.min_weight,
            "gross_limit": self.gross_limit,
            "net_limit": self.net_limit,
            "cash_buffer": self.cash_buffer,
            "neutralization_required": self.neutralization_required,
            "post_construction_constraints": list(self.post_construction_constraints),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PortfolioConstructionSpec:
        kwargs = dict(payload)
        constraints = kwargs.get("post_construction_constraints", ())
        if constraints is None:
            constraints = ()
        if not isinstance(constraints, (list, tuple)):
            raise ValueError("post_construction_constraints must be a list of strings")
        parsed_constraints: list[str] = []
        for idx, item in enumerate(constraints):
            if not isinstance(item, str):
                raise ValueError(
                    "post_construction_constraints must contain only strings; "
                    f"found {type(item).__name__} at index {idx}"
                )
            parsed_constraints.append(item)
        kwargs["post_construction_constraints"] = tuple(parsed_constraints)
        return cls(**kwargs)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ExecutionAssumptionsSpec:
    """Contract describing execution replay assumptions for external engines."""

    schema_version: str = EXECUTION_ASSUMPTIONS_SCHEMA_VERSION
    fill_price_rule: str = "next_open"
    execution_delay_bars: int = 1
    commission_model: str = "bps"
    slippage_model: str = "fixed_bps"
    lot_size_rule: str = "none"
    lot_size: int | None = None
    cash_buffer: float = 0.0
    partial_fill_policy: str = "allow_partial"
    suspension_policy: str = "skip_trade"
    price_limit_policy: str = "skip_trade"
    trade_when_not_tradable: bool = False
    allow_same_day_reentry: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != EXECUTION_ASSUMPTIONS_SCHEMA_VERSION:
            raise ValueError(
                "execution assumptions schema_version must be "
                f"{EXECUTION_ASSUMPTIONS_SCHEMA_VERSION!r}"
            )
        if self.fill_price_rule not in _ALLOWED_FILL_PRICE_RULES:
            raise ValueError(
                f"fill_price_rule must be one of {sorted(_ALLOWED_FILL_PRICE_RULES)}"
            )
        if self.execution_delay_bars < 0:
            raise ValueError("execution_delay_bars must be >= 0")
        if self.commission_model not in _ALLOWED_COMMISSION_MODELS:
            raise ValueError(
                f"commission_model must be one of {sorted(_ALLOWED_COMMISSION_MODELS)}"
            )
        if self.slippage_model not in _ALLOWED_SLIPPAGE_MODELS:
            raise ValueError(
                f"slippage_model must be one of {sorted(_ALLOWED_SLIPPAGE_MODELS)}"
            )
        if self.lot_size_rule not in _ALLOWED_LOT_SIZE_RULES:
            raise ValueError(
                f"lot_size_rule must be one of {sorted(_ALLOWED_LOT_SIZE_RULES)}"
            )
        if self.lot_size_rule == "none":
            if self.lot_size is not None:
                raise ValueError("lot_size must be omitted when lot_size_rule='none'")
        elif self.lot_size is None or self.lot_size <= 0:
            raise ValueError("lot_size must be > 0 when lot_size_rule is enabled")
        if self.cash_buffer < 0.0 or self.cash_buffer >= 1.0:
            raise ValueError("cash_buffer must be in [0, 1)")
        if self.partial_fill_policy not in _ALLOWED_PARTIAL_FILL_POLICIES:
            raise ValueError(
                "partial_fill_policy must be one of "
                f"{sorted(_ALLOWED_PARTIAL_FILL_POLICIES)}"
            )
        if self.suspension_policy not in _ALLOWED_SUSPENSION_POLICIES:
            raise ValueError(
                f"suspension_policy must be one of {sorted(_ALLOWED_SUSPENSION_POLICIES)}"
            )
        if self.price_limit_policy not in _ALLOWED_PRICE_LIMIT_POLICIES:
            raise ValueError(
                "price_limit_policy must be one of "
                f"{sorted(_ALLOWED_PRICE_LIMIT_POLICIES)}"
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "fill_price_rule": self.fill_price_rule,
            "execution_delay_bars": self.execution_delay_bars,
            "commission_model": self.commission_model,
            "slippage_model": self.slippage_model,
            "lot_size_rule": self.lot_size_rule,
            "lot_size": self.lot_size,
            "cash_buffer": self.cash_buffer,
            "partial_fill_policy": self.partial_fill_policy,
            "suspension_policy": self.suspension_policy,
            "price_limit_policy": self.price_limit_policy,
            "trade_when_not_tradable": self.trade_when_not_tradable,
            "allow_same_day_reentry": self.allow_same_day_reentry,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ExecutionAssumptionsSpec:
        return cls(**dict(payload))  # type: ignore[arg-type]


@dataclass(frozen=True)
class HandoffManifestFile:
    """Manifest metadata for one artifact file."""

    path: str
    sha256: str
    rows: int | None
    columns: tuple[str, ...] | None
    content_fingerprint: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "rows": self.rows,
            "columns": list(self.columns) if self.columns is not None else None,
            "content_fingerprint": self.content_fingerprint,
        }


@dataclass(frozen=True)
class HandoffExportResult:
    """Result of one handoff export."""

    artifact_path: Path
    manifest_path: Path
    dataset_fingerprint: str


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Return a deterministic content hash for a DataFrame.

    Hash is order-insensitive across both rows and columns.
    """
    normalised = _normalise_for_hash(df)
    payload = normalised.to_csv(
        index=False,
        lineterminator="\n",
        na_rep="<NA>",
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compute_dataset_fingerprint(
    tables: dict[str, pd.DataFrame],
    *,
    context: dict[str, object] | None = None,
) -> dict[str, object]:
    """Compute a deterministic dataset fingerprint over named tables."""
    if not tables:
        raise ValueError("tables must contain at least one DataFrame")
    table_entries: list[dict[str, object]] = []
    for table_name in sorted(tables):
        df = tables[table_name]
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f"tables[{table_name!r}] must be a DataFrame, got {type(df).__name__}"
            )
        table_entries.append(
            {
                "name": table_name,
                "rows": int(len(df)),
                "columns": sorted(df.columns.tolist()),
                "content_fingerprint": dataframe_fingerprint(df),
            }
        )

    payload = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "tables": table_entries,
        "context": context or {},
    }
    packed = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(packed.encode("utf-8")).hexdigest()
    return {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "tables": table_entries,
        "context": context or {},
    }


def export_handoff_artifact(
    result: ExperimentResult,
    *,
    output_dir: str | Path,
    universe_df: pd.DataFrame,
    tradability_df: pd.DataFrame,
    artifact_name: str | None = None,
    experiment_id: str | None = None,
    fold_id: int | None = None,
    validation_context: dict[str, object] | None = None,
    exclusion_reasons_df: pd.DataFrame | None = None,
    include_label_snapshot: bool = False,
    portfolio_construction: PortfolioConstructionSpec | None = None,
    execution_assumptions: ExecutionAssumptionsSpec | None = None,
    overwrite: bool = False,
) -> HandoffExportResult:
    """Export one experiment result into a deterministic handoff bundle.

    Bundle schema v2 includes three explicit components:
    1. research artifact
    2. portfolio construction contract
    3. execution assumptions contract
    """
    validate_universe_table(universe_df)
    validate_tradability_table(tradability_df)
    if result.delay_spec is None:
        raise ValueError("result.delay_spec is required for handoff export")

    signal_df = _signal_snapshot_from_result(result)
    signal_dates = set(pd.to_datetime(signal_df["date"]))
    signal_keys = set(zip(pd.to_datetime(signal_df["date"]), signal_df["asset"], strict=False))

    universe_slice = _slice_mask_to_signal_dates(
        universe_df,
        signal_dates=signal_dates,
        signal_keys=signal_keys,
        value_col="in_universe",
        table_name="universe",
    )
    tradability_slice = _slice_mask_to_signal_dates(
        tradability_df,
        signal_dates=signal_dates,
        signal_keys=signal_keys,
        value_col="is_tradable",
        table_name="tradability",
    )
    _assert_universe_tradability_alignment(universe_slice, tradability_slice)
    signal_name = _signal_name_from_snapshot(signal_df)

    label_snapshot: pd.DataFrame | None
    if include_label_snapshot:
        label_snapshot = _label_snapshot_from_result(result, signal_dates=signal_dates)
    else:
        label_snapshot = None

    exclusion_slice: pd.DataFrame | None
    if exclusion_reasons_df is not None:
        exclusion_slice = _prepare_exclusion_reasons(
            exclusion_reasons_df,
            signal_dates=signal_dates,
        )
    else:
        exclusion_slice = None
    _assert_exclusion_reason_coverage(
        tradability_df=tradability_slice,
        exclusion_df=exclusion_slice,
    )

    resolved_name = artifact_name or _default_artifact_name(result, fold_id=fold_id)
    resolved_experiment_id = experiment_id or _default_experiment_id(result, fold_id=fold_id)
    artifact_path = Path(output_dir) / resolved_name
    _prepare_output_dir(artifact_path, overwrite=overwrite)

    portfolio_spec = _resolve_portfolio_construction_spec(
        signal_name=signal_name,
        portfolio_construction=portfolio_construction,
    )
    execution_spec = _resolve_execution_assumptions_spec(
        result=result,
        portfolio_spec=portfolio_spec,
        execution_assumptions=execution_assumptions,
    )
    delay_spec_payload = result.delay_spec.to_dict()

    timing_payload = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "delay_spec": delay_spec_payload,
        "label_metadata": (
            result.label_metadata.to_dict() if result.label_metadata is not None else None
        ),
    }
    _validate_timing_execution_consistency(
        delay_spec_payload=delay_spec_payload,
        portfolio_spec=portfolio_spec,
        execution_spec=execution_spec,
    )

    portfolio_payload = portfolio_spec.to_dict()
    execution_payload = execution_spec.to_dict()

    metadata_payload = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "experiment_id": resolved_experiment_id,
        "fold_id": fold_id,
        "experiment_metadata": (
            result.metadata.to_dict() if result.metadata is not None else None
        ),
        "provenance": asdict(result.provenance),
        "export_timestamp_utc": _utc_now(),
    }

    validation_payload = _build_validation_context_payload(
        result=result,
        fold_id=fold_id,
        overrides=validation_context,
    )

    fingerprint_payload = compute_dataset_fingerprint(
        _dataset_tables_for_fingerprint(
            signal_df=signal_df,
            universe_df=universe_slice,
            tradability_df=tradability_slice,
            label_df=label_snapshot,
            exclusion_df=exclusion_slice,
        ),
        context={
            "artifact_name": resolved_name,
            "experiment_id": resolved_experiment_id,
            "fold_id": fold_id,
            "dataset_id": result.metadata.dataset_id
            if result.metadata is not None
            else None,
            "dataset_hash": result.metadata.dataset_hash
            if result.metadata is not None
            else None,
            "portfolio_construction": portfolio_payload,
            "execution_assumptions": execution_payload,
        },
    )

    file_entries: dict[str, HandoffManifestFile] = {}
    _write_table(signal_df, artifact_path / "signal_snapshot.csv", file_entries)
    _write_table(universe_slice, artifact_path / "universe_mask.csv", file_entries)
    _write_table(tradability_slice, artifact_path / "tradability_mask.csv", file_entries)
    if exclusion_slice is not None:
        _write_table(exclusion_slice, artifact_path / "exclusion_reasons.csv", file_entries)
    if label_snapshot is not None:
        _write_table(label_snapshot, artifact_path / "label_snapshot.csv", file_entries)

    _write_json(
        artifact_path / "timing.json",
        timing_payload,
        file_entries=file_entries,
    )
    _write_json(
        artifact_path / "experiment_metadata.json",
        metadata_payload,
        file_entries=file_entries,
    )
    _write_json(
        artifact_path / "validation_context.json",
        validation_payload,
        file_entries=file_entries,
    )
    _write_json(
        artifact_path / "dataset_fingerprint.json",
        fingerprint_payload,
        file_entries=file_entries,
    )
    _write_json(
        artifact_path / "portfolio_construction.json",
        portfolio_payload,
        file_entries=file_entries,
    )
    _write_json(
        artifact_path / "execution_assumptions.json",
        execution_payload,
        file_entries=file_entries,
    )

    manifest_payload = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "bundle_type": "alpha_lab_handoff_bundle",
        "artifact_name": resolved_name,
        "experiment_id": resolved_experiment_id,
        "fold_id": fold_id,
        "dataset_fingerprint": fingerprint_payload["fingerprint"],
        "bundle_components": {
            "research_artifact": {
                "schema_version": HANDOFF_SCHEMA_VERSION,
                "required_files": list(_REQUIRED_ARTIFACT_FILES_V2),
            },
            "portfolio_construction": {
                "schema_version": portfolio_spec.schema_version,
                "file": "portfolio_construction.json",
            },
            "execution_assumptions": {
                "schema_version": execution_spec.schema_version,
                "file": "execution_assumptions.json",
            },
        },
        "export_timestamp_utc": _utc_now(),
        "environment": {
            "python_version": platform.python_version(),
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
        "files": {name: entry.to_dict() for name, entry in sorted(file_entries.items())},
    }
    manifest_path = artifact_path / "manifest.json"
    _write_json(manifest_path, manifest_payload, file_entries=None)

    validate_handoff_artifact(artifact_path)
    return HandoffExportResult(
        artifact_path=artifact_path,
        manifest_path=manifest_path,
        dataset_fingerprint=str(fingerprint_payload["fingerprint"]),
    )


def export_walk_forward_handoff_artifacts(
    result: WalkForwardResult,
    *,
    output_dir: str | Path,
    universe_df: pd.DataFrame,
    tradability_df: pd.DataFrame,
    artifact_prefix: str = "walk_forward_handoff",
    fold_ids: list[int] | None = None,
    exclusion_reasons_df: pd.DataFrame | None = None,
    include_label_snapshot: bool = False,
    portfolio_construction: PortfolioConstructionSpec | None = None,
    execution_assumptions: ExecutionAssumptionsSpec | None = None,
    overwrite: bool = False,
) -> list[HandoffExportResult]:
    """Export one handoff artifact per selected walk-forward fold."""
    selected_ids = set(fold_ids) if fold_ids is not None else None
    if selected_ids is not None:
        invalid = [fid for fid in selected_ids if fid < 0 or fid >= len(result.per_fold_results)]
        if invalid:
            raise ValueError(f"fold_ids contain out-of-range values: {sorted(invalid)}")

    exports: list[HandoffExportResult] = []
    for fold_id, fold_result in enumerate(result.per_fold_results):
        if selected_ids is not None and fold_id not in selected_ids:
            continue
        validation_context = _walk_forward_context(result, fold_id=fold_id)
        exports.append(
            export_handoff_artifact(
                fold_result,
                output_dir=output_dir,
                artifact_name=f"{artifact_prefix}_fold_{fold_id:03d}",
                experiment_id=f"{artifact_prefix}_fold_{fold_id:03d}",
                fold_id=fold_id,
                universe_df=universe_df,
                tradability_df=tradability_df,
                validation_context=validation_context,
                exclusion_reasons_df=exclusion_reasons_df,
                include_label_snapshot=include_label_snapshot,
                portfolio_construction=portfolio_construction,
                execution_assumptions=execution_assumptions,
                overwrite=overwrite,
            )
        )
    return exports


def validate_handoff_artifact(path: str | Path) -> None:
    """Validate on-disk handoff package structure, schema, and hashes."""
    artifact_path = Path(path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"artifact path does not exist: {artifact_path}")
    if not artifact_path.is_dir():
        raise NotADirectoryError(f"artifact path is not a directory: {artifact_path}")

    manifest_path = artifact_path / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("artifact is missing manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version")
    if schema_version not in {HANDOFF_SCHEMA_VERSION, _LEGACY_HANDOFF_SCHEMA_VERSION}:
        raise ValueError(
            f"unsupported schema_version {schema_version!r}; "
            f"expected one of {[_LEGACY_HANDOFF_SCHEMA_VERSION, HANDOFF_SCHEMA_VERSION]}"
        )

    required_files = (
        _REQUIRED_ARTIFACT_FILES_V2
        if schema_version == HANDOFF_SCHEMA_VERSION
        else _REQUIRED_ARTIFACT_FILES_V1
    )
    for required in required_files:
        if not (artifact_path / required).exists():
            raise ValueError(f"artifact is missing required file {required!r}")

    file_entries = manifest.get("files")
    if not isinstance(file_entries, dict):
        raise ValueError("manifest.files must be a dictionary")

    for required in required_files:
        if required == "manifest.json":
            continue
        if required not in file_entries:
            raise ValueError(f"manifest.files missing required file entry {required!r}")

    for rel_name, info in file_entries.items():
        abs_path = artifact_path / rel_name
        if not abs_path.exists():
            raise ValueError(f"manifest references missing file {rel_name!r}")
        expected_sha = info.get("sha256")
        actual_sha = _sha256_bytes(abs_path.read_bytes())
        if expected_sha != actual_sha:
            raise ValueError(
                f"hash mismatch for {rel_name!r}: expected {expected_sha}, got {actual_sha}"
            )

    signal = pd.read_csv(artifact_path / "signal_snapshot.csv")
    universe = pd.read_csv(artifact_path / "universe_mask.csv")
    tradability = pd.read_csv(artifact_path / "tradability_mask.csv")
    _assert_columns(signal, _SIGNAL_COLUMNS, "signal_snapshot.csv")
    _assert_columns(universe, _UNIVERSE_COLUMNS, "universe_mask.csv")
    _assert_columns(tradability, _TRADABILITY_COLUMNS, "tradability_mask.csv")
    _assert_universe_tradability_alignment(universe, tradability)
    exclusion_path = artifact_path / "exclusion_reasons.csv"
    exclusion_df: pd.DataFrame | None = None
    if exclusion_path.exists():
        exclusion_df = pd.read_csv(exclusion_path)
        _assert_columns(exclusion_df, _EXCLUSION_COLUMNS, "exclusion_reasons.csv")
    _assert_exclusion_reason_coverage(
        tradability_df=tradability,
        exclusion_df=exclusion_df,
    )

    timing = json.loads((artifact_path / "timing.json").read_text(encoding="utf-8"))
    delay_spec = timing.get("delay_spec")
    if not isinstance(delay_spec, dict):
        raise ValueError("timing.json must contain delay_spec object")
    for key in (
        "decision_timestamp",
        "execution_delay_periods",
        "return_horizon_periods",
        "label_start_offset_periods",
        "label_end_offset_periods",
    ):
        if key not in delay_spec:
            raise ValueError(f"timing.delay_spec missing required key {key!r}")

    portfolio_spec: PortfolioConstructionSpec | None = None
    execution_spec: ExecutionAssumptionsSpec | None = None
    if schema_version == HANDOFF_SCHEMA_VERSION:
        components = manifest.get("bundle_components")
        if not isinstance(components, dict):
            raise ValueError("manifest.bundle_components must be a dictionary for schema v2")
        for key in _MANIFEST_COMPONENT_KEYS_V2:
            if key not in components:
                raise ValueError(
                    f"manifest.bundle_components missing required key {key!r}"
                )
        portfolio_component = components["portfolio_construction"]
        execution_component = components["execution_assumptions"]
        if not isinstance(portfolio_component, dict) or not isinstance(execution_component, dict):
            raise ValueError(
                "manifest.bundle_components portfolio/execution entries must be objects"
            )
        if portfolio_component.get("file") != "portfolio_construction.json":
            raise ValueError(
                "manifest.bundle_components.portfolio_construction.file must be "
                "'portfolio_construction.json'"
            )
        if execution_component.get("file") != "execution_assumptions.json":
            raise ValueError(
                "manifest.bundle_components.execution_assumptions.file must be "
                "'execution_assumptions.json'"
            )
        if manifest.get("bundle_type") != "alpha_lab_handoff_bundle":
            raise ValueError("manifest.bundle_type must be 'alpha_lab_handoff_bundle'")

        portfolio_payload = json.loads(
            (artifact_path / "portfolio_construction.json").read_text(encoding="utf-8")
        )
        execution_payload = json.loads(
            (artifact_path / "execution_assumptions.json").read_text(encoding="utf-8")
        )
        portfolio_spec = PortfolioConstructionSpec.from_dict(portfolio_payload)
        execution_spec = ExecutionAssumptionsSpec.from_dict(execution_payload)
        _validate_signal_name_consistency(signal, portfolio_spec)
        _validate_timing_execution_consistency(
            delay_spec_payload=delay_spec,
            portfolio_spec=portfolio_spec,
            execution_spec=execution_spec,
        )

    fingerprint = json.loads(
        (artifact_path / "dataset_fingerprint.json").read_text(encoding="utf-8")
    )
    if fingerprint.get("fingerprint") != manifest.get("dataset_fingerprint"):
        raise ValueError(
            "dataset fingerprint mismatch between manifest and "
            "dataset_fingerprint.json"
        )


def _signal_name_from_snapshot(signal_df: pd.DataFrame) -> str:
    if signal_df.empty:
        raise ValueError("signal_snapshot is empty")
    names = pd.unique(signal_df["signal_name"].astype(str))
    if len(names) != 1:
        raise ValueError(
            "signal snapshot must contain exactly one signal_name for handoff export"
        )
    signal_name = str(names[0]).strip()
    if not signal_name:
        raise ValueError("signal_name must be non-empty")
    return signal_name


def _resolve_portfolio_construction_spec(
    *,
    signal_name: str,
    portfolio_construction: PortfolioConstructionSpec | None,
) -> PortfolioConstructionSpec:
    if portfolio_construction is None:
        return PortfolioConstructionSpec(signal_name=signal_name)
    if portfolio_construction.signal_name is None:
        return portfolio_construction.with_signal_name(signal_name)
    if portfolio_construction.signal_name != signal_name:
        raise ValueError(
            "portfolio_construction.signal_name does not match exported signal name "
            f"({portfolio_construction.signal_name!r} vs {signal_name!r})"
        )
    return portfolio_construction


def _resolve_execution_assumptions_spec(
    *,
    result: ExperimentResult,
    portfolio_spec: PortfolioConstructionSpec,
    execution_assumptions: ExecutionAssumptionsSpec | None,
) -> ExecutionAssumptionsSpec:
    if execution_assumptions is not None:
        return execution_assumptions
    delay = result.delay_spec
    assert delay is not None
    return ExecutionAssumptionsSpec(
        execution_delay_bars=int(delay.execution_delay_periods),
        cash_buffer=float(portfolio_spec.cash_buffer),
    )


def _validate_signal_name_consistency(
    signal_df: pd.DataFrame,
    portfolio_spec: PortfolioConstructionSpec,
) -> None:
    signal_name = _signal_name_from_snapshot(signal_df)
    if portfolio_spec.signal_name != signal_name:
        raise ValueError(
            "portfolio_construction.signal_name does not match signal_snapshot.csv"
        )


def _validate_timing_execution_consistency(
    *,
    delay_spec_payload: Mapping[str, object],
    portfolio_spec: PortfolioConstructionSpec,
    execution_spec: ExecutionAssumptionsSpec,
) -> None:
    research_delay = _coerce_non_negative_int(
        delay_spec_payload.get("execution_delay_periods"),
        field_name="timing.delay_spec.execution_delay_periods",
    )
    if execution_spec.execution_delay_bars < research_delay:
        raise ValueError(
            "execution_assumptions.execution_delay_bars must be >= "
            "timing.delay_spec.execution_delay_periods"
        )
    if abs(float(portfolio_spec.cash_buffer) - float(execution_spec.cash_buffer)) > 1e-12:
        raise ValueError(
            "portfolio_construction.cash_buffer must equal "
            "execution_assumptions.cash_buffer to avoid ambiguity"
        )


def _coerce_non_negative_int(value: object, *, field_name: str) -> int:
    if isinstance(value, (np.integer, int)):
        parsed = int(value)
    elif isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"{field_name} must be an integer")
        parsed = int(value)
    else:
        raise ValueError(f"{field_name} must be an integer")
    if parsed < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return parsed


def _signal_snapshot_from_result(result: ExperimentResult) -> pd.DataFrame:
    if result.factor_df.empty:
        raise ValueError("result.factor_df is empty")
    eval_dates = pd.to_datetime(result.quantile_assignments_df["date"]).unique()
    if len(eval_dates) == 0:
        raise ValueError(
            "result.quantile_assignments_df is empty; cannot infer evaluation signal snapshot"
        )

    factor_df = result.factor_df.copy()
    factor_df["date"] = pd.to_datetime(factor_df["date"], errors="coerce")
    factor_df = factor_df[factor_df["date"].isin(eval_dates)].copy()
    factor_df = factor_df[factor_df["value"].notna()].copy()
    if factor_df.empty:
        raise ValueError("no non-null factor values available for evaluation dates")

    factor_names = pd.unique(factor_df["factor"].astype(str))
    if len(factor_names) != 1:
        raise ValueError(
            "handoff export requires exactly one factor in result.factor_df; "
            f"got {factor_names!r}"
        )

    out = factor_df[["date", "asset", "value"]].copy()
    out = out.rename(columns={"value": "signal_value"})
    out["signal_name"] = str(factor_names[0])
    _assert_unique_keys(out, ("date", "asset"), "signal_snapshot")
    return _finalise_table(out, key_columns=("date", "asset"), column_order=_SIGNAL_COLUMNS)


def _label_snapshot_from_result(
    result: ExperimentResult,
    *,
    signal_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    labels = result.label_df.copy()
    labels["date"] = pd.to_datetime(labels["date"], errors="coerce")
    labels = labels[labels["date"].isin(signal_dates)].copy()
    if labels.empty:
        raise ValueError("label snapshot requested but no label rows overlap signal dates")
    out = labels[["date", "asset", "factor", "value"]].copy()
    out = out.rename(columns={"factor": "label_name", "value": "label_value"})
    _assert_unique_keys(out, ("date", "asset", "label_name"), "label_snapshot")
    return _finalise_table(
        out,
        key_columns=("date", "asset", "label_name"),
        column_order=_LABEL_COLUMNS,
    )


def _slice_mask_to_signal_dates(
    df: pd.DataFrame,
    *,
    signal_dates: set[pd.Timestamp],
    signal_keys: set[tuple[pd.Timestamp, object]],
    value_col: str,
    table_name: str,
) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out[out["date"].isin(signal_dates)].copy()
    if out.empty:
        raise ValueError(f"{table_name} has no rows on signal dates")

    out_keys = set(zip(pd.to_datetime(out["date"]), out["asset"], strict=False))
    missing = signal_keys - out_keys
    if missing:
        raise ValueError(
            f"{table_name} is missing {len(missing)} (date, asset) keys required by signal"
        )
    _assert_unique_keys(out, ("date", "asset"), table_name)

    column_order = _UNIVERSE_COLUMNS if value_col == "in_universe" else _TRADABILITY_COLUMNS
    return _finalise_table(
        out[["date", "asset", value_col]],
        key_columns=("date", "asset"),
        column_order=column_order,
    )


def _prepare_exclusion_reasons(
    exclusion_reasons_df: pd.DataFrame,
    *,
    signal_dates: set[pd.Timestamp],
) -> pd.DataFrame:
    missing = set(_EXCLUSION_COLUMNS) - set(exclusion_reasons_df.columns)
    if missing:
        raise ValueError(f"exclusion_reasons_df missing required columns: {sorted(missing)}")
    out = exclusion_reasons_df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("exclusion_reasons_df contains invalid dates")
    out = out[out["date"].isin(signal_dates)].copy()
    if (out["reason"].astype(str).str.strip() == "").any():
        raise ValueError("exclusion_reasons_df contains empty reason values")
    return _finalise_table(
        out,
        key_columns=("date", "asset", "reason"),
        column_order=tuple(out.columns),
    )


def _assert_exclusion_reason_coverage(
    *,
    tradability_df: pd.DataFrame,
    exclusion_df: pd.DataFrame | None,
) -> None:
    tradability = tradability_df.copy()
    tradability["date"] = pd.to_datetime(tradability["date"], errors="coerce")
    if tradability["date"].isna().any():
        raise ValueError("tradability table contains invalid date values")
    non_tradable = tradability.loc[~tradability["is_tradable"].astype(bool), ["date", "asset"]]
    non_tradable = non_tradable.drop_duplicates().reset_index(drop=True)
    if non_tradable.empty:
        return

    if exclusion_df is None:
        raise ValueError(
            "exclusion_reasons.csv is required when tradability contains non-tradable keys"
        )
    reasons = exclusion_df.copy()
    reasons["date"] = pd.to_datetime(reasons["date"], errors="coerce")
    if reasons["date"].isna().any():
        raise ValueError("exclusion_reasons contains invalid date values")
    reason_keys = set(zip(reasons["date"], reasons["asset"], strict=False))
    missing = [
        (pd.Timestamp(row.date), str(row.asset))
        for row in non_tradable.itertuples(index=False)
        if (pd.Timestamp(row.date), str(row.asset)) not in reason_keys
    ]
    if missing:
        sample = ", ".join(
            f"({d.date().isoformat()}, {a})" for d, a in missing[:3]
        )
        raise ValueError(
            "exclusion reasons are missing for non-tradable keys; "
            f"missing_count={len(missing)} sample={sample}"
        )


def _build_validation_context_payload(
    *,
    result: ExperimentResult,
    fold_id: int | None,
    overrides: dict[str, object] | None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "fold_id": fold_id,
    }
    if result.metadata is not None and result.metadata.validation is not None:
        payload["validation"] = result.metadata.validation.to_dict()
    else:
        payload["validation"] = None
    if overrides:
        payload.update(overrides)
    return payload


def _dataset_tables_for_fingerprint(
    *,
    signal_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    tradability_df: pd.DataFrame,
    label_df: pd.DataFrame | None,
    exclusion_df: pd.DataFrame | None,
) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {
        "signal_snapshot": signal_df,
        "universe_mask": universe_df,
        "tradability_mask": tradability_df,
    }
    if label_df is not None:
        tables["label_snapshot"] = label_df
    if exclusion_df is not None:
        tables["exclusion_reasons"] = exclusion_df
    return tables


def _write_table(
    df: pd.DataFrame,
    path: Path,
    file_entries: dict[str, HandoffManifestFile],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path,
        index=False,
        lineterminator="\n",
        float_format="%.17g",
        date_format="%Y-%m-%dT%H:%M:%S",
    )
    rel = path.name
    file_entries[rel] = HandoffManifestFile(
        path=rel,
        sha256=_sha256_bytes(path.read_bytes()),
        rows=int(len(df)),
        columns=tuple(df.columns.tolist()),
        content_fingerprint=dataframe_fingerprint(df),
    )


def _write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    file_entries: dict[str, HandoffManifestFile] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), sort_keys=True, indent=2) + "\n"
    path.write_text(text, encoding="utf-8")
    if file_entries is not None:
        rel = path.name
        file_entries[rel] = HandoffManifestFile(
            path=rel,
            sha256=_sha256_bytes(path.read_bytes()),
            rows=None,
            columns=None,
            content_fingerprint=None,
        )


def _normalise_for_hash(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.reindex(sorted(out.columns), axis=1)
    for col in out.columns:
        series = out[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            out[col] = pd.to_datetime(series, errors="coerce").dt.strftime(
                "%Y-%m-%dT%H:%M:%S.%f"
            )
            out[col] = out[col].fillna("<NA>")
        elif pd.api.types.is_bool_dtype(series):
            out[col] = (
                series.astype("boolean")
                .map({True: "true", False: "false"})
                .astype("string")
                .fillna("<NA>")
            )
        elif pd.api.types.is_numeric_dtype(series):
            out[col] = series.map(_format_numeric)
        else:
            out[col] = series.astype("string").fillna("<NA>")
    out = out.astype("string")
    if len(out.columns) > 0:
        out = out.sort_values(by=list(out.columns), kind="mergesort").reset_index(drop=True)
    return out


def _format_numeric(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "<NA>"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, (np.floating, float)):
        if np.isnan(float(value)):
            return "<NA>"
        return format(float(value), ".17g")
    return str(value)


def _finalise_table(
    df: pd.DataFrame,
    *,
    key_columns: tuple[str, ...],
    column_order: tuple[str, ...],
) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    if out["date"].isna().any():
        raise ValueError("table contains invalid date values")
    out = out.sort_values(list(key_columns), kind="mergesort").reset_index(drop=True)
    if set(column_order).issubset(out.columns):
        out = out[list(column_order)]
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    return out


def _assert_unique_keys(
    df: pd.DataFrame,
    key_columns: tuple[str, ...],
    table_name: str,
) -> None:
    if df.duplicated(subset=list(key_columns)).any():
        raise ValueError(
            f"{table_name} contains duplicate rows for keys {list(key_columns)}"
        )


def _assert_columns(df: pd.DataFrame, expected: tuple[str, ...], table_name: str) -> None:
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(f"{table_name} missing required columns: {sorted(missing)}")


def _assert_universe_tradability_alignment(
    universe_df: pd.DataFrame,
    tradability_df: pd.DataFrame,
) -> None:
    u_keys = set(zip(pd.to_datetime(universe_df["date"]), universe_df["asset"], strict=False))
    t_keys = set(
        zip(pd.to_datetime(tradability_df["date"]), tradability_df["asset"], strict=False)
    )
    if u_keys != t_keys:
        raise ValueError("universe and tradability keys must match exactly in handoff artifact")


def _prepare_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"artifact path already exists: {path}. Pass overwrite=True to replace files."
        )
    path.mkdir(parents=True, exist_ok=True)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _default_artifact_name(result: ExperimentResult, *, fold_id: int | None) -> str:
    factor = result.provenance.factor_name
    horizon = result.provenance.horizon
    suffix = f"_fold_{fold_id:03d}" if fold_id is not None else ""
    return f"handoff_{factor}_h{horizon}{suffix}"


def _default_experiment_id(result: ExperimentResult, *, fold_id: int | None) -> str:
    base = result.metadata.trial_id if result.metadata is not None else None
    if not base:
        base = f"{result.provenance.factor_name}_h{result.provenance.horizon}"
    if fold_id is not None:
        return f"{base}_fold_{fold_id:03d}"
    return base


def _walk_forward_context(result: WalkForwardResult, *, fold_id: int) -> dict[str, object]:
    payload: dict[str, object] = {}
    if result.fold_windows_df is not None and not result.fold_windows_df.empty:
        row = result.fold_windows_df.loc[result.fold_windows_df["fold_id"] == fold_id]
        if not row.empty:
            payload = {
                "train_start": _maybe_iso(row["train_start"].iloc[0]),
                "train_end": _maybe_iso(row["train_end"].iloc[0]),
                "val_start": _maybe_iso(row["val_start"].iloc[0]),
                "val_end": _maybe_iso(row["val_end"].iloc[0]),
                "test_start": _maybe_iso(row["test_start"].iloc[0]),
                "test_end": _maybe_iso(row["test_end"].iloc[0]),
            }
    if result.validation_spec is not None:
        payload["walk_forward_spec"] = result.validation_spec.to_dict()
    return payload


def _maybe_iso(value: object) -> str | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    return str(ts.isoformat())
