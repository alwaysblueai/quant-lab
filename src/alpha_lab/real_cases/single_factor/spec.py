from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabExperimentError
from alpha_lab.real_cases.common_spec import (
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

Direction = Literal["long", "short"]
Standardization = Literal["zscore", "rank", "none"]


@dataclass(frozen=True)
class FactorInputSpec:
    """Optional in-spec factor generation config."""

    mode: str = "file"
    recipe: Mapping[str, object] | None = None
    disable_pipeline_preprocess: bool = True

    def __post_init__(self) -> None:
        normalized = self.mode.strip().lower()
        if normalized not in {"file", "recipe"}:
            raise AlphaLabConfigError("factor_input.mode must be one of ['file', 'recipe']")
        if normalized == "recipe" and self.recipe is None:
            raise AlphaLabConfigError("factor_input.recipe is required when mode='recipe'")


@dataclass(frozen=True)
class PreprocessSpec:
    """Cross-sectional preprocessing controls for the single factor."""

    winsorize: bool = True
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    standardization: Standardization = "zscore"
    min_group_size: int = 3
    min_coverage: float | None = None

    def __post_init__(self) -> None:
        if self.winsorize_lower < 0 or self.winsorize_upper > 1:
            raise AlphaLabConfigError("preprocess winsorize bounds must be within [0, 1]")
        if self.winsorize_lower >= self.winsorize_upper:
            raise AlphaLabConfigError("preprocess.winsorize_lower must be < winsorize_upper")
        if self.standardization not in {"zscore", "rank", "none"}:
            raise AlphaLabConfigError(
                "preprocess.standardization must be one of ['zscore', 'rank', 'none']"
            )
        if self.min_group_size <= 0:
            raise AlphaLabConfigError("preprocess.min_group_size must be > 0")
        if self.min_coverage is not None and (self.min_coverage <= 0 or self.min_coverage > 1):
            raise AlphaLabConfigError("preprocess.min_coverage must be in (0, 1] when provided")


@dataclass(frozen=True)
class CapacitySpec:
    """Optional research-level capacity estimation controls."""

    enabled: bool = True
    participation_rate: float = 0.05
    adv_lookback: int = 20

    def __post_init__(self) -> None:
        if self.participation_rate <= 0 or self.participation_rate > 1:
            raise AlphaLabConfigError("capacity.participation_rate must be in (0, 1]")
        if self.adv_lookback <= 0:
            raise AlphaLabConfigError("capacity.adv_lookback must be > 0")


@dataclass(frozen=True)
class SingleFactorCaseSpec:
    """Typed schema for one real-case single-factor study."""

    name: str
    factor_name: str
    factor_path: str
    prices_path: str
    factor_input: FactorInputSpec | None
    universe: UniverseSpec
    target: TargetSpec
    direction: Direction
    preprocess: PreprocessSpec
    neutralization: NeutralizationSpec
    capacity: CapacitySpec
    rebalance_frequency: str
    n_quantiles: int
    transaction_cost: TransactionCostSpec
    output: OutputSpec
    archive_identity: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise AlphaLabConfigError("name must be non-empty")
        if not self.factor_name.strip():
            raise AlphaLabConfigError("factor_name must be non-empty")
        if not self.factor_path.strip():
            raise AlphaLabConfigError("factor_path must be non-empty")
        if not self.prices_path.strip():
            raise AlphaLabConfigError("prices_path must be non-empty")
        if self.direction not in {"long", "short"}:
            raise AlphaLabConfigError("direction must be one of ['long', 'short']")
        if not self.rebalance_frequency.strip():
            raise AlphaLabConfigError("rebalance_frequency must be non-empty")
        if self.n_quantiles < 2:
            raise AlphaLabConfigError("n_quantiles must be >= 2")


def load_single_factor_case_spec(path: str | Path) -> SingleFactorCaseSpec:
    """Load and validate a single-factor case spec from JSON/YAML."""

    spec_path = Path(path).resolve()
    if not spec_path.exists() or not spec_path.is_file():
        raise FileNotFoundError(f"spec file does not exist: {spec_path}")

    text = spec_path.read_text(encoding="utf-8")
    parsed = parse_mapping_payload(text, suffix=spec_path.suffix.lower())
    spec = single_factor_case_spec_from_mapping(parsed)
    return resolve_spec_paths(spec, base_dir=spec_path.parent)


def single_factor_case_spec_from_mapping(data: Mapping[str, object]) -> SingleFactorCaseSpec:
    """Build typed single-factor spec from raw mapping payload."""

    name = required_str(data, "name")
    factor_name = required_str(data, "factor_name")
    archive_identity = _optional_text(data.get("archive_identity"))
    factor_path = required_str(data, "factor_path")
    prices_path = required_str(data, "prices_path")
    rebalance_frequency = required_str(data, "rebalance_frequency")
    factor_input = _parse_factor_input_spec(data.get("factor_input"))

    universe = UniverseSpec(**mapping_kwargs(data.get("universe", {}), field_name="universe"))
    target = TargetSpec(**mapping_kwargs(data.get("target", {}), field_name="target"))
    preprocess = PreprocessSpec(
        **mapping_kwargs(data.get("preprocess", {}), field_name="preprocess")
    )
    neutralization = NeutralizationSpec(
        **mapping_kwargs(data.get("neutralization", {}), field_name="neutralization")
    )
    capacity = CapacitySpec(**mapping_kwargs(data.get("capacity", {}), field_name="capacity"))
    transaction_cost = TransactionCostSpec(
        **mapping_kwargs(data.get("transaction_cost", {}), field_name="transaction_cost")
    )
    output = OutputSpec(**mapping_kwargs(data.get("output", {}), field_name="output"))

    raw_n_quantiles = data.get("n_quantiles", 5)
    if not isinstance(raw_n_quantiles, int):
        raise AlphaLabConfigError("n_quantiles must be an integer")

    return SingleFactorCaseSpec(
        name=name,
        factor_name=factor_name,
        archive_identity=archive_identity,
        factor_path=factor_path,
        prices_path=prices_path,
        factor_input=factor_input,
        universe=universe,
        target=target,
        direction=parse_long_short_direction(data.get("direction", "long")),
        preprocess=preprocess,
        neutralization=neutralization,
        capacity=capacity,
        rebalance_frequency=rebalance_frequency,
        n_quantiles=raw_n_quantiles,
        transaction_cost=transaction_cost,
        output=output,
    )


def resolve_spec_paths(spec: SingleFactorCaseSpec, *, base_dir: Path) -> SingleFactorCaseSpec:
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
        factor_path=resolve_required_path(spec.factor_path, base_dir=base_dir),
        prices_path=resolve_required_path(spec.prices_path, base_dir=base_dir),
        universe=universe,
        neutralization=neutralization,
        output=output,
    )


def spec_to_dict(spec: SingleFactorCaseSpec) -> dict[str, object]:
    """Convert typed spec to JSON/YAML-serializable dict."""

    payload = cast(dict[str, object], asdict(spec))
    if payload.get("archive_identity") is None:
        payload.pop("archive_identity", None)
    return payload


def dump_spec_yaml(spec: SingleFactorCaseSpec) -> str:
    """Serialize spec as YAML text."""

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - import guard
        raise AlphaLabExperimentError("PyYAML is required to serialize YAML specs") from exc

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


def _parse_factor_input_spec(value: object) -> FactorInputSpec | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise AlphaLabConfigError("factor_input must be an object")

    mode_raw = value.get("mode", "recipe")
    if not isinstance(mode_raw, str):
        raise AlphaLabConfigError("factor_input.mode must be a string")
    mode = mode_raw.strip().lower()

    recipe: Mapping[str, object] | None = None
    recipe_raw = value.get("recipe")
    if recipe_raw is not None:
        if not isinstance(recipe_raw, Mapping):
            raise AlphaLabConfigError("factor_input.recipe must be an object when provided")
        recipe = cast(Mapping[str, object], recipe_raw)

    disable_pipeline_preprocess_raw = value.get("disable_pipeline_preprocess", True)
    if not isinstance(disable_pipeline_preprocess_raw, bool):
        raise AlphaLabConfigError("factor_input.disable_pipeline_preprocess must be a boolean")

    return FactorInputSpec(
        mode=mode,
        recipe=recipe,
        disable_pipeline_preprocess=disable_pipeline_preprocess_raw,
    )
