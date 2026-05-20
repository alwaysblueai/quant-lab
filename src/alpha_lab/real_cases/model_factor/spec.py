from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

from alpha_lab.model_factor import (
    FeatureImportanceConfig,
    FeaturePreprocessConfig,
    ModelSelectionSpec,
    ModelSpec,
    TrainingSpec,
)
from alpha_lab.real_cases.common_spec import (
    FactorDirection,
    NeutralizationSpec,
    OutputSpec,
    TargetSpec,
    TransactionCostSpec,
    UniverseSpec,
    mapping_kwargs,
    parse_long_short_direction,
    parse_mapping_payload,
    required_str,
    resolve_optional_path,
    resolve_required_path,
)

FeatureAvailabilityMode = Literal["required_timestamp", "safety_lag"]

_FUNDAMENTAL_FEATURE_HINT_TOKENS: frozenset[str] = frozenset(
    {
        "roe",
        "roa",
        "eps",
        "bp",
        "pb",
        "pe",
        "ps",
        "pcf",
        "book",
        "dividend",
        "yield",
        "margin",
        "profit",
        "revenue",
        "cash",
        "flow",
    }
)


def infer_fundamental_feature_columns(feature_columns: tuple[str, ...]) -> tuple[str, ...]:
    """Heuristic detector for likely fundamentals-based feature names."""

    matched: list[str] = []
    for column in feature_columns:
        name = str(column).strip()
        if not name:
            continue
        lower = name.lower()
        tokens = [token for token in re.split(r"[^a-z0-9]+", lower) if token]
        token_set = set(tokens)
        if token_set & _FUNDAMENTAL_FEATURE_HINT_TOKENS:
            matched.append(name)
            continue
        if lower.startswith(("roe", "roa", "eps", "bp", "pb", "pe", "ps", "pcf")):
            matched.append(name)
    return tuple(matched)


@dataclass(frozen=True)
class FeatureAvailabilitySpec:
    """Point-in-time contract for feature availability."""

    mode: FeatureAvailabilityMode = "required_timestamp"
    column: str | None = None
    safety_lag_days: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"required_timestamp", "safety_lag"}:
            raise ValueError(
                "feature_availability.mode must be one of "
                "['required_timestamp', 'safety_lag']"
            )
        if self.column is not None and not self.column.strip():
            raise ValueError("feature_availability.column must be non-empty when provided")
        if self.mode == "required_timestamp":
            if self.safety_lag_days not in {None, 0}:
                raise ValueError(
                    "feature_availability.safety_lag_days must be omitted when "
                    "feature_availability.mode='required_timestamp'"
                )
            return
        if self.column is not None:
            raise ValueError(
                "feature_availability.column is not allowed when "
                "feature_availability.mode='safety_lag'"
            )
        if self.safety_lag_days is None or self.safety_lag_days <= 0:
            raise ValueError(
                "feature_availability.safety_lag_days must be > 0 when "
                "feature_availability.mode='safety_lag'"
            )


@dataclass(frozen=True)
class ModelFactorCaseSpec:
    """Typed schema for one real-case model-generated factor study."""

    name: str
    factor_name: str
    features_path: str
    feature_columns: tuple[str, ...]
    prices_path: str
    universe: UniverseSpec
    target: TargetSpec
    direction: FactorDirection
    feature_availability: FeatureAvailabilitySpec
    feature_preprocess: FeaturePreprocessConfig
    feature_importance: FeatureImportanceConfig
    model: ModelSpec
    model_selection: ModelSelectionSpec
    training: TrainingSpec
    neutralization: NeutralizationSpec
    rebalance_frequency: str
    n_quantiles: int
    transaction_cost: TransactionCostSpec
    output: OutputSpec
    archive_identity: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.factor_name.strip():
            raise ValueError("factor_name must be non-empty")
        if not self.features_path.strip():
            raise ValueError("features_path must be non-empty")
        if not self.prices_path.strip():
            raise ValueError("prices_path must be non-empty")
        if not self.feature_columns:
            raise ValueError("feature_columns must be a non-empty list")
        seen: set[str] = set()
        for column in self.feature_columns:
            if not column.strip():
                raise ValueError("feature_columns must contain non-empty strings")
            if column in {"date", "asset", "factor", "value"}:
                raise ValueError(f"feature_columns may not include reserved column {column!r}")
            if column in seen:
                raise ValueError(f"feature_columns must be unique; duplicate: {column!r}")
            seen.add(column)
        if self.direction not in {"long", "short"}:
            raise ValueError("direction must be one of ['long', 'short']")
        if not self.rebalance_frequency.strip():
            raise ValueError("rebalance_frequency must be non-empty")
        if self.n_quantiles < 2:
            raise ValueError("n_quantiles must be >= 2")


def load_model_factor_case_spec(path: str | Path) -> ModelFactorCaseSpec:
    """Load and validate a model-factor case spec from JSON/YAML."""

    spec_path = Path(path).resolve()
    if not spec_path.exists() or not spec_path.is_file():
        raise FileNotFoundError(f"spec file does not exist: {spec_path}")

    text = spec_path.read_text(encoding="utf-8")
    parsed = parse_mapping_payload(text, suffix=spec_path.suffix.lower())
    spec = model_factor_case_spec_from_mapping(parsed)
    return resolve_spec_paths(spec, base_dir=spec_path.parent)


def model_factor_case_spec_from_mapping(data: Mapping[str, object]) -> ModelFactorCaseSpec:
    """Build typed model-factor spec from raw mapping payload."""

    name = required_str(data, "name")
    factor_name = required_str(data, "factor_name")
    archive_identity = _optional_text(data.get("archive_identity"))
    features_path = required_str(data, "features_path")
    prices_path = required_str(data, "prices_path")
    rebalance_frequency = required_str(data, "rebalance_frequency")
    feature_columns = _parse_feature_columns(data.get("feature_columns"))

    universe = UniverseSpec(**mapping_kwargs(data.get("universe", {}), field_name="universe"))
    target = TargetSpec(**mapping_kwargs(data.get("target", {}), field_name="target"))
    feature_availability = FeatureAvailabilitySpec(
        **mapping_kwargs(
            data.get("feature_availability", {}),
            field_name="feature_availability",
        )
    )
    feature_preprocess = FeaturePreprocessConfig(
        **mapping_kwargs(
            data.get("feature_preprocess", {}),
            field_name="feature_preprocess",
        )
    )
    feature_importance = FeatureImportanceConfig(
        **mapping_kwargs(
            data.get("feature_importance", {}),
            field_name="feature_importance",
        )
    )
    model = ModelSpec(**mapping_kwargs(data.get("model", {}), field_name="model"))
    model_selection = _parse_model_selection_spec(data.get("model_selection", {}))
    training = TrainingSpec(**mapping_kwargs(data.get("training", {}), field_name="training"))
    neutralization = NeutralizationSpec(
        **mapping_kwargs(data.get("neutralization", {}), field_name="neutralization")
    )
    transaction_cost = TransactionCostSpec(
        **mapping_kwargs(data.get("transaction_cost", {}), field_name="transaction_cost")
    )
    output = OutputSpec(**mapping_kwargs(data.get("output", {}), field_name="output"))

    raw_n_quantiles = data.get("n_quantiles", 5)
    if not isinstance(raw_n_quantiles, int):
        raise ValueError("n_quantiles must be an integer")

    return ModelFactorCaseSpec(
        name=name,
        factor_name=factor_name,
        archive_identity=archive_identity,
        features_path=features_path,
        feature_columns=feature_columns,
        prices_path=prices_path,
        universe=universe,
        target=target,
        direction=parse_long_short_direction(data.get("direction", "long")),
        feature_availability=feature_availability,
        feature_preprocess=feature_preprocess,
        feature_importance=feature_importance,
        model=model,
        model_selection=model_selection,
        training=training,
        neutralization=neutralization,
        rebalance_frequency=rebalance_frequency,
        n_quantiles=raw_n_quantiles,
        transaction_cost=transaction_cost,
        output=output,
    )


def resolve_spec_paths(spec: ModelFactorCaseSpec, *, base_dir: Path) -> ModelFactorCaseSpec:
    """Resolve relative file paths in spec against config directory."""

    universe = spec.universe
    if universe.path is not None:
        universe = replace(universe, path=resolve_optional_path(universe.path, base_dir=base_dir))

    neutralization = spec.neutralization
    if neutralization.exposures_path is not None:
        neutralization = replace(
            neutralization,
            exposures_path=resolve_optional_path(
                neutralization.exposures_path,
                base_dir=base_dir,
            ),
        )

    output = replace(
        spec.output,
        root_dir=resolve_required_path(spec.output.root_dir, base_dir=base_dir),
    )

    return replace(
        spec,
        features_path=resolve_required_path(spec.features_path, base_dir=base_dir),
        prices_path=resolve_required_path(spec.prices_path, base_dir=base_dir),
        universe=universe,
        neutralization=neutralization,
        output=output,
    )


def spec_to_dict(spec: ModelFactorCaseSpec) -> dict[str, object]:
    """Convert typed spec to JSON/YAML-serializable dict."""

    payload = cast(dict[str, object], asdict(spec))
    if payload.get("archive_identity") is None:
        payload.pop("archive_identity", None)
    payload["feature_columns"] = list(spec.feature_columns)
    model_selection = cast(dict[str, object], payload.get("model_selection", {}))
    model_selection["candidates"] = [
        asdict(candidate) for candidate in spec.model_selection.candidates
    ]
    payload["model_selection"] = model_selection
    return payload


def dump_spec_yaml(spec: ModelFactorCaseSpec) -> str:
    """Serialize spec as YAML text."""

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError("PyYAML is required to serialize YAML specs") from exc

    return str(
        yaml.safe_dump(
            spec_to_dict(spec),
            sort_keys=False,
            allow_unicode=False,
        )
    )


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _parse_feature_columns(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("feature_columns must be a non-empty list")
    out: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ValueError("feature_columns must contain non-empty strings")
        out.append(item.strip())
    return tuple(out)


def _parse_model_selection_spec(value: object) -> ModelSelectionSpec:
    mapping = mapping_kwargs(value, field_name="model_selection")
    raw_candidates = mapping.pop("candidates", [])
    if not isinstance(raw_candidates, list):
        raise ValueError("model_selection.candidates must be a list when provided")
    candidates: list[ModelSpec] = []
    for idx, raw in enumerate(raw_candidates):
        if not isinstance(raw, Mapping):
            raise ValueError(f"model_selection.candidates[{idx}] must be an object")
        candidates.append(
            ModelSpec(
                **mapping_kwargs(
                    raw,
                    field_name=f"model_selection.candidates[{idx}]",
                )
            )
        )
    return ModelSelectionSpec(
        candidates=tuple(candidates),
        **mapping,
    )
