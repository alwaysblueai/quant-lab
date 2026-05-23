from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabExperimentError

FactorDirection = Literal["long", "short"]
TargetKind = Literal["forward_return"]
ExecutionPriceMode = Literal["close", "next_open"]
_ALLOWED_EXECUTION_PRICE_MODES: tuple[str, ...] = ("close", "next_open")


@dataclass(frozen=True)
class UniverseSpec:
    """Optional universe filter for PIT-safe row selection."""

    name: str = "default"
    path: str | None = None
    in_universe_column: str = "in_universe"

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise AlphaLabConfigError("universe.name must be non-empty")
        if not self.in_universe_column.strip():
            raise AlphaLabConfigError("universe.in_universe_column must be non-empty")


@dataclass(frozen=True)
class TargetSpec:
    """Target/label definition.

    ``execution_price_mode`` selects the entry-price convention used to build
    the forward-return label.  ``"next_open"`` (default) models a
    trader who sees the signal at the close of day ``t`` and executes at the
    next open, which is the realistic entry for after-close or full-day
    signals.  ``"close"`` remains available for explicit legacy close-to-close
    studies.  The
    string is normalized to lowercase; values outside the allowed set fail
    fast.
    """

    kind: TargetKind = "forward_return"
    horizon: int = 5
    price_column: str = "close"
    max_abs_forward_return: float | None = None
    winsorize_zscore: float | None = None
    execution_price_mode: ExecutionPriceMode = "next_open"

    def __post_init__(self) -> None:
        if self.kind != "forward_return":
            raise AlphaLabConfigError("target.kind currently supports only 'forward_return'")
        if self.horizon <= 0:
            raise AlphaLabConfigError("target.horizon must be > 0")
        if not isinstance(self.price_column, str) or not self.price_column.strip():
            raise AlphaLabConfigError("target.price_column must be a non-empty string")
        if self.max_abs_forward_return is not None and self.max_abs_forward_return <= 0:
            raise AlphaLabConfigError(
                "target.max_abs_forward_return must be > 0 when provided"
            )
        if self.winsorize_zscore is not None and self.winsorize_zscore <= 0:
            raise AlphaLabConfigError(
                "target.winsorize_zscore must be > 0 when provided (e.g. 3.0)"
            )
        if not isinstance(self.execution_price_mode, str):
            raise AlphaLabConfigError(
                "target.execution_price_mode must be a string; "
                f"got {type(self.execution_price_mode).__name__}"
            )
        normalized = self.execution_price_mode.strip().lower()
        if normalized not in _ALLOWED_EXECUTION_PRICE_MODES:
            raise AlphaLabConfigError(
                "target.execution_price_mode must be one of "
                f"{list(_ALLOWED_EXECUTION_PRICE_MODES)}; got "
                f"{self.execution_price_mode!r}"
            )
        if normalized != self.execution_price_mode:
            object.__setattr__(self, "execution_price_mode", cast(ExecutionPriceMode, normalized))


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
            raise AlphaLabConfigError("neutralization.min_obs must be > 0")
        if self.ridge < 0:
            raise AlphaLabConfigError("neutralization.ridge must be >= 0")
        if self.enabled:
            if self.exposures_path is None or not self.exposures_path.strip():
                raise AlphaLabConfigError(
                    "neutralization.exposures_path is required when neutralization.enabled=True"
                )
            if self.size_col is None and self.industry_col is None:
                raise AlphaLabConfigError(
                    "neutralization requires at least one of size_col/industry_col"
                )


@dataclass(frozen=True)
class TransactionCostSpec:
    """Simple one-way transaction cost assumption."""

    one_way_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.one_way_rate < 0:
            raise AlphaLabConfigError("transaction_cost.one_way_rate must be >= 0")


@dataclass(frozen=True)
class OutputSpec:
    """Output root for case artifacts."""

    root_dir: str = "outputs/real_cases"

    def __post_init__(self) -> None:
        if not self.root_dir.strip():
            raise AlphaLabConfigError("output.root_dir must be non-empty")


def parse_mapping_payload(text: str, *, suffix: str) -> Mapping[str, Any]:
    parsed: object
    if suffix == ".json":
        parsed = json.loads(text)
    elif suffix in {".yml", ".yaml"}:
        parsed = yaml_load(text)
    elif suffix == ".md":
        parsed = yaml_load(_extract_yaml_from_markdown(text))
    else:
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = yaml_load(text)

    if not isinstance(parsed, Mapping):
        raise AlphaLabConfigError("spec root must be a mapping/object")
    return cast(Mapping[str, Any], parsed)


def _extract_yaml_from_markdown(text: str) -> str:
    """Extract the YAML content from a Markdown code fence (```yaml ... ```)."""
    marker = "```yaml"
    start = text.find(marker)
    if start < 0:
        return text  # no fence found, try raw
    start += len(marker)
    end = text.find("```", start)
    if end < 0:
        return text[start:]
    return text[start:end]


def yaml_load(text: str) -> object:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - import guard
        raise AlphaLabExperimentError(
            "PyYAML is required for YAML specs; use JSON or install PyYAML"
        ) from exc

    return yaml.safe_load(text)


def required_str(data: Mapping[str, object], key: str) -> str:
    raw = data.get(key)
    if not isinstance(raw, str) or not raw.strip():
        raise AlphaLabConfigError(f"{key} must be a non-empty string")
    return raw


def mapping_kwargs(value: object, *, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise AlphaLabConfigError(f"{field_name} must be an object")
    out: dict[str, Any] = {}
    for key, raw in value.items():
        if not isinstance(key, str):
            raise AlphaLabConfigError(f"{field_name} keys must be strings")
        out[key] = raw
    return out


def parse_long_short_direction(value: object) -> FactorDirection:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"long", "positive", "+", "1"}:
            return "long"
        if normalized in {"short", "negative", "-", "-1"}:
            return "short"
    raise AlphaLabConfigError("direction must be one of ['long', 'short']")


def resolve_optional_path(path_value: str | None, *, base_dir: Path) -> str | None:
    if path_value is None:
        return None
    return resolve_required_path(path_value, base_dir=base_dir)


def resolve_required_path(path_value: str, *, base_dir: Path) -> str:
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    else:
        path = path.resolve()
    return str(path)
