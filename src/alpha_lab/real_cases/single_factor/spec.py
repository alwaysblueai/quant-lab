from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast

Direction = Literal["long", "short"]
Standardization = Literal["zscore", "rank", "none"]
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
    """Target/label definition."""

    kind: TargetKind = "forward_return"
    horizon: int = 5

    def __post_init__(self) -> None:
        if self.kind != "forward_return":
            raise ValueError("target.kind currently supports only 'forward_return'")
        if self.horizon <= 0:
            raise ValueError("target.horizon must be > 0")


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
            raise ValueError("preprocess winsorize bounds must be within [0, 1]")
        if self.winsorize_lower >= self.winsorize_upper:
            raise ValueError("preprocess.winsorize_lower must be < winsorize_upper")
        if self.standardization not in {"zscore", "rank", "none"}:
            raise ValueError(
                "preprocess.standardization must be one of ['zscore', 'rank', 'none']"
            )
        if self.min_group_size <= 0:
            raise ValueError("preprocess.min_group_size must be > 0")
        if self.min_coverage is not None and (
            self.min_coverage <= 0 or self.min_coverage > 1
        ):
            raise ValueError("preprocess.min_coverage must be in (0, 1] when provided")


@dataclass(frozen=True)
class NeutralizationSpec:
    """Optional exposure neutralization controls (size/industry)."""

    enabled: bool = False
    exposures_path: str | None = None
    size_col: str | None = None
    industry_col: str | None = None
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
            if self.size_col is None and self.industry_col is None:
                raise ValueError(
                    "neutralization requires at least one of size_col/industry_col"
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
class SingleFactorCaseSpec:
    """Typed schema for one real-case single-factor study."""

    name: str
    factor_name: str
    factor_path: str
    prices_path: str
    universe: UniverseSpec
    target: TargetSpec
    direction: Direction
    preprocess: PreprocessSpec
    neutralization: NeutralizationSpec
    rebalance_frequency: str
    n_quantiles: int
    transaction_cost: TransactionCostSpec
    output: OutputSpec

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be non-empty")
        if not self.factor_name.strip():
            raise ValueError("factor_name must be non-empty")
        if not self.factor_path.strip():
            raise ValueError("factor_path must be non-empty")
        if not self.prices_path.strip():
            raise ValueError("prices_path must be non-empty")
        if self.direction not in {"long", "short"}:
            raise ValueError("direction must be one of ['long', 'short']")
        if not self.rebalance_frequency.strip():
            raise ValueError("rebalance_frequency must be non-empty")
        if self.n_quantiles < 2:
            raise ValueError("n_quantiles must be >= 2")


def load_single_factor_case_spec(path: str | Path) -> SingleFactorCaseSpec:
    """Load and validate a single-factor case spec from JSON/YAML."""

    spec_path = Path(path).resolve()
    if not spec_path.exists() or not spec_path.is_file():
        raise FileNotFoundError(f"spec file does not exist: {spec_path}")

    text = spec_path.read_text(encoding="utf-8")
    parsed = _parse_mapping_payload(text, suffix=spec_path.suffix.lower())
    spec = single_factor_case_spec_from_mapping(parsed)
    return resolve_spec_paths(spec, base_dir=spec_path.parent)


def single_factor_case_spec_from_mapping(data: Mapping[str, object]) -> SingleFactorCaseSpec:
    """Build typed single-factor spec from raw mapping payload."""

    name = _required_str(data, "name")
    factor_name = _required_str(data, "factor_name")
    factor_path = _required_str(data, "factor_path")
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

    direction = _parse_direction(data.get("direction", "long"))

    raw_n_quantiles = data.get("n_quantiles", 5)
    if not isinstance(raw_n_quantiles, int):
        raise ValueError("n_quantiles must be an integer")

    return SingleFactorCaseSpec(
        name=name,
        factor_name=factor_name,
        factor_path=factor_path,
        prices_path=prices_path,
        universe=universe,
        target=target,
        direction=direction,
        preprocess=preprocess,
        neutralization=neutralization,
        rebalance_frequency=rebalance_frequency,
        n_quantiles=raw_n_quantiles,
        transaction_cost=transaction_cost,
        output=output,
    )


def resolve_spec_paths(spec: SingleFactorCaseSpec, *, base_dir: Path) -> SingleFactorCaseSpec:
    """Resolve relative file paths in spec against config directory."""

    def _resolve(path_value: str) -> str:
        p = Path(path_value)
        if not p.is_absolute():
            p = (base_dir / p).resolve()
        else:
            p = p.resolve()
        return str(p)

    universe = spec.universe
    if universe.path is not None:
        universe = replace(universe, path=_resolve(universe.path))

    neutralization = spec.neutralization
    if neutralization.exposures_path is not None:
        neutralization = replace(
            neutralization,
            exposures_path=_resolve(neutralization.exposures_path),
        )

    output = replace(spec.output, root_dir=_resolve(spec.output.root_dir))

    return replace(
        spec,
        factor_path=_resolve(spec.factor_path),
        prices_path=_resolve(spec.prices_path),
        universe=universe,
        neutralization=neutralization,
        output=output,
    )


def spec_to_dict(spec: SingleFactorCaseSpec) -> dict[str, object]:
    """Convert typed spec to JSON/YAML-serializable dict."""

    return cast(dict[str, object], asdict(spec))


def dump_spec_yaml(spec: SingleFactorCaseSpec) -> str:
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


def _required_str(data: Mapping[str, object], key: str) -> str:
    raw = data.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return raw


def _mapping_kwargs(value: object, *, field_name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    out: dict[str, object] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            raise ValueError(f"{field_name} keys must be strings")
        out[key] = raw
    return out


def _parse_direction(value: object) -> Direction:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"long", "positive", "+", "1"}:
            return "long"
        if normalized in {"short", "negative", "-", "-1"}:
            return "short"
    raise ValueError("direction must be one of ['long', 'short']")
