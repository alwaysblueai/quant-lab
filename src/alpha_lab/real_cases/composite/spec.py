from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

Direction = Literal["positive", "negative"]
ComponentTransform = Literal["zscore", "rank", "none"]
TargetKind = Literal["forward_return"]


@dataclass(frozen=True)
class UniverseSpec:
    """Optional universe filter for PIT-safe row selection."""

    name: str = "default"
    path: str | None = None
    in_universe_column: str = "in_universe"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("universe.name must be non-empty")
        if not self.in_universe_column.strip():
            raise ValueError("universe.in_universe_column must be non-empty")


@dataclass(frozen=True)
class TargetSpec:
    """Target/label configuration."""

    kind: TargetKind = "forward_return"
    horizon: int = 5

    def __post_init__(self) -> None:
        if self.kind != "forward_return":
            raise ValueError("target.kind currently supports only 'forward_return'")
        if self.horizon <= 0:
            raise ValueError("target.horizon must be > 0")


@dataclass(frozen=True)
class ComponentSpec:
    """One component factor in the composite."""

    name: str
    path: str
    factor: str | None = None
    weight: float = 1.0
    direction: Direction = "positive"
    transform: ComponentTransform = "zscore"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("component.name must be non-empty")
        if not self.path.strip():
            raise ValueError(f"component[{self.name}].path must be non-empty")
        if self.factor is not None and not self.factor.strip():
            raise ValueError(f"component[{self.name}].factor must be non-empty when provided")
        if self.weight == 0:
            raise ValueError(f"component[{self.name}].weight must be non-zero")
        if self.direction not in {"positive", "negative"}:
            raise ValueError(
                f"component[{self.name}].direction must be 'positive' or 'negative'"
            )
        if self.transform not in {"zscore", "rank", "none"}:
            raise ValueError(
                f"component[{self.name}].transform must be one of ['zscore', 'rank', 'none']"
            )


@dataclass(frozen=True)
class PreprocessSpec:
    """Global preprocessing options applied before per-component transforms."""

    winsorize: bool = True
    winsorize_lower: float = 0.01
    winsorize_upper: float = 0.99
    min_group_size: int = 3
    min_coverage: float | None = None

    def __post_init__(self) -> None:
        if self.winsorize_lower < 0 or self.winsorize_upper > 1:
            raise ValueError("preprocess winsorize bounds must be within [0, 1]")
        if self.winsorize_lower >= self.winsorize_upper:
            raise ValueError("preprocess.winsorize_lower must be < winsorize_upper")
        if self.min_group_size <= 0:
            raise ValueError("preprocess.min_group_size must be > 0")
        if self.min_coverage is not None and (
            self.min_coverage <= 0 or self.min_coverage > 1
        ):
            raise ValueError("preprocess.min_coverage must be in (0, 1] when provided")


@dataclass(frozen=True)
class NeutralizationSpec:
    """Optional exposure neutralization controls."""

    enabled: bool = False
    exposures_path: str | None = None
    size_col: str | None = None
    industry_col: str | None = None
    beta_col: str | None = None
    min_obs: int = 20
    ridge: float = 1e-8

    def __post_init__(self) -> None:
        if self.min_obs <= 0:
            raise ValueError("neutralization.min_obs must be > 0")
        if self.ridge < 0:
            raise ValueError("neutralization.ridge must be >= 0")
        if self.enabled:
            if self.exposures_path is None or not self.exposures_path.strip():
                raise ValueError(
                    "neutralization.exposures_path is required when neutralization.enabled=True"
                )
            if self.size_col is None and self.industry_col is None and self.beta_col is None:
                raise ValueError(
                    "neutralization requires at least one of size_col/industry_col/beta_col"
                )


@dataclass(frozen=True)
class TransactionCostSpec:
    """Simple one-way transaction cost assumption."""

    one_way_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.one_way_rate < 0:
            raise ValueError("transaction_cost.one_way_rate must be >= 0")


@dataclass(frozen=True)
class OutputSpec:
    """Output root for case artifacts."""

    root_dir: str = "outputs/real_cases"

    def __post_init__(self) -> None:
        if not self.root_dir.strip():
            raise ValueError("output.root_dir must be non-empty")


@dataclass(frozen=True)
class CompositeCaseSpec:
    """Typed schema for one real-case composite factor study."""

    name: str
    prices_path: str
    universe: UniverseSpec
    rebalance_frequency: str
    target: TargetSpec
    n_quantiles: int
    components: tuple[ComponentSpec, ...]
    preprocess: PreprocessSpec
    neutralization: NeutralizationSpec
    transaction_cost: TransactionCostSpec
    output: OutputSpec

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.prices_path.strip():
            raise ValueError("prices_path must be non-empty")
        if not self.rebalance_frequency.strip():
            raise ValueError("rebalance_frequency must be non-empty")
        if self.n_quantiles < 2:
            raise ValueError("n_quantiles must be >= 2")
        if len(self.components) < 2:
            raise ValueError("composite spec requires at least 2 components")

        names = [c.name for c in self.components]
        if len(set(names)) != len(names):
            raise ValueError("component names must be unique")
        if sum(abs(c.weight) for c in self.components) <= 0:
            raise ValueError("sum(abs(component weights)) must be > 0")


def load_composite_case_spec(path: str | Path) -> CompositeCaseSpec:
    """Load and validate a composite-case spec from JSON/YAML."""

    spec_path = Path(path).resolve()
    if not spec_path.exists() or not spec_path.is_file():
        raise FileNotFoundError(f"spec file does not exist: {spec_path}")

    text = spec_path.read_text(encoding="utf-8")
    parsed = _parse_mapping_payload(text, suffix=spec_path.suffix.lower())
    spec = composite_case_spec_from_mapping(parsed)
    return resolve_spec_paths(spec, base_dir=spec_path.parent)


def composite_case_spec_from_mapping(data: Mapping[str, object]) -> CompositeCaseSpec:
    """Build a typed spec from a raw mapping payload."""

    name = _required_str(data, "name")
    prices_path = _required_str(data, "prices_path")
    rebalance_frequency = _required_str(data, "rebalance_frequency")

    universe = UniverseSpec(**_mapping_kwargs(data.get("universe", {}), field_name="universe"))
    target = TargetSpec(**_mapping_kwargs(data.get("target", {}), field_name="target"))
    preprocess = PreprocessSpec(
        **_mapping_kwargs(data.get("preprocess", {}), field_name="preprocess")
    )
    neutralization = NeutralizationSpec(
        **_mapping_kwargs(data.get("neutralization", {}), field_name="neutralization")
    )
    transaction_cost = TransactionCostSpec(
        **_mapping_kwargs(data.get("transaction_cost", {}), field_name="transaction_cost")
    )
    output = OutputSpec(**_mapping_kwargs(data.get("output", {}), field_name="output"))

    raw_components = data.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise ValueError("components must be a non-empty list")
    components: list[ComponentSpec] = []
    for idx, raw in enumerate(raw_components):
        kwargs = _mapping_kwargs(raw, field_name=f"components[{idx}]")
        direction = _parse_direction(kwargs.get("direction", "positive"), idx=idx)
        transform = _parse_transform(kwargs.get("transform", "zscore"), idx=idx)
        kwargs["direction"] = direction
        kwargs["transform"] = transform
        components.append(ComponentSpec(**kwargs))

    n_quantiles_raw = data.get("n_quantiles", 5)
    if not isinstance(n_quantiles_raw, int):
        raise ValueError("n_quantiles must be an integer")

    return CompositeCaseSpec(
        name=name,
        prices_path=prices_path,
        universe=universe,
        rebalance_frequency=rebalance_frequency,
        target=target,
        n_quantiles=n_quantiles_raw,
        components=tuple(components),
        preprocess=preprocess,
        neutralization=neutralization,
        transaction_cost=transaction_cost,
        output=output,
    )


def resolve_spec_paths(spec: CompositeCaseSpec, *, base_dir: Path) -> CompositeCaseSpec:
    """Resolve relative file paths in spec against config directory."""

    def _resolve(path_value: str) -> str:
        p = Path(path_value)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        else:
            p = p.resolve()
        return str(p)

    components = tuple(
        replace(component, path=_resolve(component.path)) for component in spec.components
    )

    universe = spec.universe
    if universe.path is not None:
        universe = replace(universe, path=_resolve(universe.path))

    neutralization = spec.neutralization
    if neutralization.exposures_path is not None:
        neutralization = replace(
            neutralization,
            exposures_path=_resolve(neutralization.exposures_path),
        )

    output = replace(output := spec.output, root_dir=_resolve(output.root_dir))

    return replace(
        spec,
        prices_path=_resolve(spec.prices_path),
        components=components,
        universe=universe,
        neutralization=neutralization,
        output=output,
    )


def spec_to_dict(spec: CompositeCaseSpec) -> dict[str, object]:
    """Convert typed spec to JSON/YAML-serializable dict."""

    payload = asdict(spec)
    payload["components"] = [asdict(component) for component in spec.components]
    return cast(dict[str, object], payload)


def dump_spec_yaml(spec: CompositeCaseSpec) -> str:
    """Serialize spec as YAML text."""

    try:
        import yaml
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("PyYAML is required to serialize YAML specs") from exc

    return str(
        yaml.safe_dump(
            spec_to_dict(spec),
            sort_keys=False,
            allow_unicode=False,
        )
    )


def _parse_mapping_payload(text: str, *, suffix: str) -> Mapping[str, object]:
    parsed: object
    if suffix == ".json":
        parsed = json.loads(text)
    elif suffix in {".yml", ".yaml"}:
        parsed = _yaml_load(text)
    else:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = _yaml_load(text)

    if not isinstance(parsed, Mapping):
        raise ValueError("spec root must be a mapping/object")
    return cast(Mapping[str, object], parsed)


def _yaml_load(text: str) -> object:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "PyYAML is required for YAML specs; use JSON or install PyYAML"
        ) from exc

    return yaml.safe_load(text)


def _mapping_kwargs(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    out: dict[str, object] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings")
        out[key] = raw
    return out


def _required_str(data: Mapping[str, object], key: str) -> str:
    raw = data.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return raw


def _parse_direction(value: object, *, idx: int) -> Direction:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"positive", "+", "pos", "long"}:
            return "positive"
        if normalized in {"negative", "-", "neg", "short"}:
            return "negative"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if float(value) == 1.0:
            return "positive"
        if float(value) == -1.0:
            return "negative"
    raise ValueError(
        f"components[{idx}].direction must be positive/negative (or +/-1), got {value!r}"
    )


def _parse_transform(value: object, *, idx: int) -> ComponentTransform:
    if not isinstance(value, str):
        raise ValueError(f"components[{idx}].transform must be a string")
    normalized = value.strip().lower()
    if normalized not in {"zscore", "rank", "none"}:
        raise ValueError(
            f"components[{idx}].transform must be one of ['zscore', 'rank', 'none']"
        )
    return cast(ComponentTransform, normalized)
