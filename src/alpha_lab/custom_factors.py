from __future__ import annotations

import hashlib
import json
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpha_lab.factor_recipe import factor_registry

CUSTOM_FACTOR_SCOPES: tuple[str, ...] = ("promoted", "research")
BUILTIN_FACTOR_NAMES: frozenset[str] = frozenset(
    {"momentum", "reversal", "low_volatility", "amplitude", "downside_volatility", "vcimom"}
)


class BrokenCustomFactorWarning(UserWarning):
    """Emitted when a persisted custom factor metadata file cannot be loaded.

    The loader keeps its long-standing "skip and continue" behavior so a single
    broken draft does not block an otherwise valid research run. The warning
    exists so that the skip is observable — Stage 3 ``validate-draft-factor``
    is still the primary gate for draft authoring (AGENTS §6).
    """


def _warn_broken_custom_factor(path: Path, exc: BaseException) -> None:
    warnings.warn(
        (
            f"skipping broken custom factor metadata at {path}: "
            f"{type(exc).__name__}: {exc}"
        ),
        BrokenCustomFactorWarning,
        stacklevel=3,
    )


@dataclass(frozen=True)
class CustomFactorSource:
    """Auditable metadata for one persisted custom factor definition."""

    name: str
    scope: str
    path: Path
    code: str
    code_sha256: str
    factor_json_sha256: str
    description: str = ""
    created_at: str = ""
    required_columns: tuple[str, ...] = ()
    optional_columns: tuple[str, ...] = ()
    archive_identity: str = ""
    frequency: str | None = None
    unavailable_data_policy: str | None = None
    pit_assumption: str | None = None
    provenance: dict[str, Any] | None = None

    def to_audit_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "name": self.name,
            "scope": self.scope,
            "path": str(self.path),
            "code_sha256": self.code_sha256,
            "factor_json_sha256": self.factor_json_sha256,
        }
        if self.description:
            payload["description"] = self.description
        if self.created_at:
            payload["created_at"] = self.created_at
        if self.required_columns:
            payload["required_columns"] = list(self.required_columns)
        if self.optional_columns:
            payload["optional_columns"] = list(self.optional_columns)
        if self.archive_identity:
            payload["archive_identity"] = self.archive_identity
        if self.frequency:
            payload["frequency"] = self.frequency
        if self.unavailable_data_policy:
            payload["unavailable_data_policy"] = self.unavailable_data_policy
        if self.pit_assumption:
            payload["pit_assumption"] = self.pit_assumption
        if self.provenance:
            payload["provenance"] = dict(self.provenance)
        return payload


def compile_custom_factor(name: str, code: str) -> Any:
    """Compile user-provided Python code into a callable factor builder.

    Preferred Stage 3 draft code defines ``build_factor(frame)`` and returns a
    Series aligned to the input index.  Legacy research drafts may still define
    ``builder(prices, ...)`` and return a canonical factor-like frame.
    """
    import numpy as np  # noqa: F401
    import pandas as pd  # noqa: F401

    namespace: dict[str, Any] = {"np": np, "pd": pd}
    try:
        compiled = compile(code, f"<custom_factor:{name}>", "exec")
    except SyntaxError as exc:
        raise ValueError(f"syntax error in custom factor code: {exc}") from exc
    exec(compiled, namespace)  # noqa: S102
    builder = namespace.get("builder")
    if builder is not None and callable(builder):
        return builder

    build_factor = namespace.get("build_factor")
    if build_factor is not None and callable(build_factor):
        return _wrap_series_builder(build_factor)

    raise ValueError(
        "custom factor code must define a callable named 'build_factor' or 'builder'; "
        "preferred: def build_factor(frame): ... return series; "
        "legacy: def builder(prices, *, window=20, **kwargs): ..."
    )


def _wrap_series_builder(build_factor: Any) -> Any:
    """Adapt ``build_factor(frame) -> Series`` to the recipe builder protocol."""

    def builder(prices: Any, **_: Any) -> Any:
        import numpy as np
        import pandas as pd

        if not isinstance(prices, pd.DataFrame):
            raise ValueError("custom build_factor input must be a pandas DataFrame")
        missing = {"date", "asset"}.difference(prices.columns)
        if missing:
            raise ValueError(f"custom build_factor wrapper missing columns: {sorted(missing)}")
        raw_value = build_factor(prices)
        if isinstance(raw_value, pd.DataFrame):
            if {"date", "asset", "value"}.issubset(raw_value.columns):
                out = raw_value[["date", "asset", "value"]].copy()
                out["value"] = pd.to_numeric(out["value"], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                return out
            if "value" in raw_value.columns:
                raw_value = raw_value["value"]
            else:
                raise ValueError(
                    "build_factor returned a DataFrame without a 'value' column or "
                    "date/asset/value columns"
                )
        value = pd.Series(raw_value, index=getattr(raw_value, "index", prices.index))
        value = pd.to_numeric(value.reindex(prices.index), errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        out = prices[["date", "asset"]].copy()
        out["value"] = value.to_numpy()
        return out

    return builder


def find_custom_factor_workspace_root(start: str | Path | None = None) -> Path:
    """Find the nearest workspace root that contains ``custom_factors``."""

    for anchor in _iter_anchor_candidates(start):
        if (anchor / "custom_factors").exists():
            return anchor
    return Path.cwd().resolve()


def iter_custom_factor_meta_paths(
    workspace_root: str | Path,
    *,
    scopes: Iterable[str] = CUSTOM_FACTOR_SCOPES,
) -> list[Path]:
    """Return persisted custom-factor metadata files in priority order."""

    root = Path(workspace_root).expanduser().resolve() / "custom_factors"
    paths: list[Path] = []
    for raw_scope in scopes:
        scope = str(raw_scope).strip().lower()
        if scope not in CUSTOM_FACTOR_SCOPES:
            continue
        base = root / scope
        if not base.exists():
            continue
        paths.extend(sorted(base.glob("*/factor.json")))
        paths.extend(sorted(path for path in base.glob("*.json") if path.name != "factor.json"))
    return paths


def custom_factor_write_path(workspace_root: str | Path, name: str) -> Path:
    normalized = name.strip().lower()
    return (
        Path(workspace_root).expanduser().resolve()
        / "custom_factors"
        / "research"
        / normalized
        / "factor.json"
    )


def custom_factor_meta_path(workspace_root: str | Path, name: str) -> Path:
    normalized = name.strip().lower()
    for path in iter_custom_factor_meta_paths(workspace_root):
        try:
            source = read_custom_factor_source(path)
        except Exception as exc:  # noqa: BLE001 — see BrokenCustomFactorWarning
            _warn_broken_custom_factor(path, exc)
            continue
        if source.name == normalized:
            return path
    return custom_factor_write_path(workspace_root, normalized)


def read_custom_factor_source(path: str | Path) -> CustomFactorSource:
    meta_path = Path(path).expanduser().resolve()
    raw_text = meta_path.read_text(encoding="utf-8")
    payload = json.loads(raw_text)
    if not isinstance(payload, dict):
        raise ValueError(f"custom factor metadata must be an object: {meta_path}")

    inferred_name = meta_path.parent.name if meta_path.name == "factor.json" else meta_path.stem
    name = str(payload.get("name") or inferred_name).strip().lower()
    if not name:
        raise ValueError(f"custom factor metadata missing name: {meta_path}")
    code = str(payload.get("code") or "")
    if not code.strip():
        raise ValueError(f"custom factor metadata missing code: {meta_path}")

    provenance_raw = payload.get("provenance")
    provenance: dict[str, Any] | None
    if isinstance(provenance_raw, dict):
        provenance = dict(provenance_raw)
    else:
        provenance = None

    return CustomFactorSource(
        name=name,
        scope=_infer_scope(meta_path),
        path=meta_path,
        code=code,
        code_sha256=sha256_text(code),
        factor_json_sha256=sha256_text(raw_text),
        description=str(payload.get("description") or "").strip(),
        created_at=str(payload.get("created_at") or "").strip(),
        required_columns=_string_tuple(payload.get("required_columns")),
        optional_columns=_string_tuple(payload.get("optional_columns")),
        archive_identity=str(payload.get("archive_identity") or "").strip(),
        frequency=_optional_text(payload.get("frequency")),
        unavailable_data_policy=_optional_text(payload.get("unavailable_data_policy")),
        pit_assumption=_optional_text(payload.get("pit_assumption")),
        provenance=provenance,
    )


def get_custom_factor_source(
    name: str,
    *,
    workspace_root: str | Path,
    scopes: Iterable[str] = CUSTOM_FACTOR_SCOPES,
) -> CustomFactorSource | None:
    normalized = name.strip().lower()
    if not normalized:
        return None
    for path in iter_custom_factor_meta_paths(workspace_root, scopes=scopes):
        try:
            source = read_custom_factor_source(path)
        except Exception as exc:  # noqa: BLE001 — see BrokenCustomFactorWarning
            _warn_broken_custom_factor(path, exc)
            continue
        if source.name == normalized:
            return source
    return None


def load_persisted_custom_factors(
    workspace_root: str | Path,
    *,
    scopes: Iterable[str] = CUSTOM_FACTOR_SCOPES,
    overwrite_existing: bool = False,
    ignore_errors: bool = True,
) -> dict[str, CustomFactorSource]:
    """Load custom-factor builders from disk into ``factor_registry``.

    Broken metadata files are skipped by default so an unrelated draft does not
    block a standard research run.
    """

    sources: dict[str, CustomFactorSource] = {}
    for meta_path in iter_custom_factor_meta_paths(workspace_root, scopes=scopes):
        try:
            source = read_custom_factor_source(meta_path)
            if source.name in sources:
                continue
            already_registered = source.name in factor_registry
            if source.name in BUILTIN_FACTOR_NAMES and not overwrite_existing:
                continue
            fn = compile_custom_factor(source.name, source.code)
            if overwrite_existing or not already_registered:
                factor_registry.register(source.name, fn)
            sources[source.name] = source
        except Exception as exc:  # noqa: BLE001 — see BrokenCustomFactorWarning
            if not ignore_errors:
                raise
            _warn_broken_custom_factor(meta_path, exc)
    return sources


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _iter_anchor_candidates(start: str | Path | None) -> Iterable[Path]:
    seen: set[Path] = set()
    raw_candidates = []
    if start is not None:
        raw_candidates.append(Path(start).expanduser())
    raw_candidates.append(Path.cwd())

    for raw in raw_candidates:
        try:
            resolved = raw.resolve()
        except OSError:
            resolved = raw.absolute()
        anchor = resolved if resolved.is_dir() else resolved.parent
        for candidate in (anchor, *anchor.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            yield candidate


def _infer_scope(path: Path) -> str:
    parts = path.parts
    try:
        idx = parts.index("custom_factors")
    except ValueError:
        return "unknown"
    if idx + 1 >= len(parts):
        return "unknown"
    scope = parts[idx + 1].lower()
    return scope if scope in CUSTOM_FACTOR_SCOPES else "unknown"


def _string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    out = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return tuple(out)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
