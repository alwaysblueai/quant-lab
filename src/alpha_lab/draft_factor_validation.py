from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_lab.custom_factors import BUILTIN_FACTOR_NAMES, compile_custom_factor, sha256_text

_FACTOR_NAME_RE = re.compile(r"^[a-z][a-z0-9_]{2,63}$")
_REQUIRED_KEYS = frozenset(
    {
        "name",
        "description",
        "required_columns",
        "optional_columns",
        "frequency",
        "unavailable_data_policy",
        "pit_assumption",
        "code",
    }
)
_FORBIDDEN_IMPORT_ROOTS = frozenset(
    {
        "glob",
        "httpx",
        "os",
        "pathlib",
        "pickle",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "urllib",
    }
)
_FORBIDDEN_CALL_NAMES = frozenset(
    {
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "input",
        "open",
    }
)
_FORBIDDEN_ATTR_CALLS = frozenset(
    {
        "mkdir",
        "popen",
        "read_csv",
        "read_excel",
        "read_feather",
        "read_hdf",
        "read_json",
        "read_parquet",
        "read_pickle",
        "read_sql",
        "read_table",
        "read_text",
        "remove",
        "rename",
        "request",
        "rmdir",
        "run",
        "system",
        "to_csv",
        "to_excel",
        "to_feather",
        "to_hdf",
        "to_json",
        "to_parquet",
        "to_pickle",
        "to_sql",
        "to_stata",
        "unlink",
        "urlopen",
        "write_text",
    }
)
_LEAKAGE_IDENTIFIER_FRAGMENTS = (
    "future_return",
    "forward_return",
    "fwd_return",
    "next_return",
    "label_return",
    "target_return",
)


@dataclass(frozen=True)
class DraftFactorIssue:
    severity: str
    code: str
    message: str


@dataclass(frozen=True)
class DraftFactorValidationResult:
    path: Path
    factor_name: str | None
    code_sha256: str | None
    factor_json_sha256: str | None
    errors: tuple[DraftFactorIssue, ...] = field(default_factory=tuple)
    warnings: tuple[DraftFactorIssue, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_payload(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "path": str(self.path),
            "factor_name": self.factor_name,
            "code_sha256": self.code_sha256,
            "factor_json_sha256": self.factor_json_sha256,
            "errors": [issue.__dict__ for issue in self.errors],
            "warnings": [issue.__dict__ for issue in self.warnings],
        }


def validate_draft_factor_file(
    path: str | Path,
    *,
    available_fields: set[str] | None = None,
    allow_non_research_path: bool = False,
) -> DraftFactorValidationResult:
    factor_path = Path(path).expanduser().resolve()
    errors: list[DraftFactorIssue] = []
    warnings: list[DraftFactorIssue] = []
    factor_name: str | None = None
    code_sha256: str | None = None
    factor_json_sha256: str | None = None

    try:
        raw_text = factor_path.read_text(encoding="utf-8")
    except OSError as exc:
        return _result(
            factor_path,
            factor_name,
            code_sha256,
            factor_json_sha256,
            errors=[_error("file_read", f"cannot read factor.json: {exc}")],
            warnings=warnings,
        )

    factor_json_sha256 = sha256_text(raw_text)
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return _result(
            factor_path,
            factor_name,
            code_sha256,
            factor_json_sha256,
            errors=[_error("json_parse", f"factor.json is not valid JSON: {exc}")],
            warnings=warnings,
        )
    if not isinstance(payload, dict):
        return _result(
            factor_path,
            factor_name,
            code_sha256,
            factor_json_sha256,
            errors=[_error("json_schema", "factor.json root must be an object")],
            warnings=warnings,
        )

    missing_keys = sorted(_REQUIRED_KEYS.difference(payload))
    if missing_keys:
        errors.append(_error("missing_keys", f"factor.json missing keys: {missing_keys}"))

    factor_name = _optional_text(payload.get("name"))
    if factor_name is None:
        errors.append(_error("name_missing", "name must be a non-empty string"))
    elif not _FACTOR_NAME_RE.match(factor_name):
        errors.append(
            _error(
                "name_format",
                "name must be snake_case, start with a letter, and use 3-64 lowercase chars",
            )
        )
    elif factor_name in BUILTIN_FACTOR_NAMES:
        errors.append(
            _error("name_builtin_collision", f"name collides with builtin: {factor_name}")
        )

    if factor_name and not allow_non_research_path:
        expected_tail = ("custom_factors", "research", factor_name, "factor.json")
        if not _path_has_tail(factor_path, expected_tail):
            errors.append(
                _error(
                    "path_scope",
                    "draft factor must live at "
                    f"custom_factors/research/{factor_name}/factor.json",
                )
            )

    description = _optional_text(payload.get("description"))
    if description is None:
        errors.append(_error("description_missing", "description must be non-empty"))

    required_columns = _string_list(payload.get("required_columns"))
    optional_columns = _string_list(payload.get("optional_columns"))
    if required_columns is None:
        errors.append(_error("required_columns_type", "required_columns must be a string list"))
        required_columns = []
    if optional_columns is None:
        errors.append(_error("optional_columns_type", "optional_columns must be a string list"))
        optional_columns = []
    if not required_columns:
        errors.append(_error("required_columns_empty", "required_columns must not be empty"))
    overlap = sorted(set(required_columns).intersection(optional_columns))
    if overlap:
        errors.append(
            _error("column_overlap", f"columns cannot be both required and optional: {overlap}")
        )
    if available_fields is not None:
        unavailable = sorted(set(required_columns).difference(available_fields))
        if unavailable:
            errors.append(
                _error(
                    "required_columns_unavailable",
                    f"required columns unavailable in provided field list: {unavailable}",
                )
            )

    frequency = _optional_text(payload.get("frequency"))
    if frequency != "daily":
        errors.append(_error("frequency", "frequency must be 'daily' for Stage3 draft factors"))

    unavailable_data_policy = _optional_text(payload.get("unavailable_data_policy"))
    if unavailable_data_policy is None:
        errors.append(
            _error("unavailable_data_policy_missing", "unavailable_data_policy must be non-empty")
        )
    pit_assumption = _optional_text(payload.get("pit_assumption"))
    if pit_assumption is None:
        errors.append(_error("pit_assumption_missing", "pit_assumption must be non-empty"))

    code = _optional_text(payload.get("code"))
    if code is None:
        errors.append(_error("code_missing", "code must be a non-empty string"))
    else:
        code_sha256 = sha256_text(code)
        _validate_code_contract(
            code,
            required_columns=required_columns,
            errors=errors,
            warnings=warnings,
        )
        if not errors:
            _validate_toy_run(
                factor_name=factor_name or "draft_factor",
                code=code,
                required_columns=required_columns,
                optional_columns=optional_columns,
                errors=errors,
            )

    return _result(
        factor_path,
        factor_name,
        code_sha256,
        factor_json_sha256,
        errors=errors,
        warnings=warnings,
    )


def _validate_code_contract(
    code: str,
    *,
    required_columns: list[str],
    errors: list[DraftFactorIssue],
    warnings: list[DraftFactorIssue],
) -> None:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        errors.append(_error("code_syntax", f"code has syntax error: {exc}"))
        return

    top_level_allowed = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Import, ast.ImportFrom)
    for node in tree.body:
        if not isinstance(node, top_level_allowed):
            errors.append(
                _error(
                    "top_level_side_effect",
                    "code may only contain imports and function definitions at top level",
                )
            )

    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in {"build_factor", "builder"}
    }
    if "build_factor" not in functions and "builder" not in functions:
        errors.append(
            _error(
                "missing_entrypoint",
                "code must define build_factor(frame) or legacy builder(prices, ...)",
            )
        )
    if "builder" in functions and "build_factor" not in functions:
        warnings.append(
            _warning(
                "legacy_builder",
                "legacy builder(prices, ...) is accepted, but new Stage2 output should "
                "prefer build_factor(frame)",
            )
        )
    if "build_factor" in functions:
        args = functions["build_factor"].args.args
        if not args or args[0].arg != "frame":
            errors.append(
                _error("build_factor_signature", "build_factor first argument must be frame")
            )
    if "builder" in functions:
        args = functions["builder"].args.args
        if not args or args[0].arg != "prices":
            warnings.append(
                _warning("builder_signature", "legacy builder first argument should be prices")
            )

    for required in required_columns:
        if repr(required) not in code and f'"{required}"' not in code:
            errors.append(
                _error(
                    "required_column_not_checked",
                    f"required column {required!r} does not appear in code",
                )
            )

    if (".rolling(" in code or ".expanding(" in code) and "groupby(" not in code:
        errors.append(
            _error(
                "rolling_without_groupby",
                "rolling/expanding operations must be grouped by asset in draft factors",
            )
        )

    uses_plain_mean_or_std = ".mean()" in code or ".std()" in code
    if uses_plain_mean_or_std and "rolling(" not in code and "groupby(" not in code:
        warnings.append(
            _warning(
                "possible_full_sample_stat",
                "mean/std appears without rolling/groupby; confirm this is not "
                "full-sample normalization",
            )
        )

    for walk_node in ast.walk(tree):
        if isinstance(walk_node, ast.Import):
            for alias in walk_node.names:
                _validate_import(alias.name, errors)
        elif isinstance(walk_node, ast.ImportFrom):
            _validate_import(walk_node.module or "", errors)
        elif isinstance(walk_node, ast.Call):
            _validate_call(walk_node, errors)
        elif isinstance(walk_node, ast.Name):
            _validate_identifier(walk_node.id, errors)
        elif isinstance(walk_node, ast.Attribute):
            _validate_identifier(walk_node.attr, errors)


def _validate_toy_run(
    *,
    factor_name: str,
    code: str,
    required_columns: list[str],
    optional_columns: list[str],
    errors: list[DraftFactorIssue],
) -> None:
    try:
        builder = compile_custom_factor(factor_name, code)
        toy = _toy_price_frame([*required_columns, *optional_columns])
        out = builder(toy)
    except Exception as exc:
        errors.append(_error("toy_run_failed", f"factor code failed on toy frame: {exc}"))
        return
    if not isinstance(out, pd.DataFrame):
        errors.append(_error("toy_output_type", "compiled factor must return a pandas DataFrame"))
        return
    missing = {"date", "asset", "value"}.difference(out.columns)
    if missing:
        errors.append(_error("toy_output_schema", f"toy output missing columns: {sorted(missing)}"))
        return
    if out.empty:
        errors.append(_error("toy_output_empty", "toy output must not be empty"))
        return
    values = pd.to_numeric(out["value"], errors="coerce")
    if np.isinf(values.to_numpy(dtype=float, na_value=np.nan)).any():
        errors.append(_error("toy_output_inf", "toy output value contains inf"))


def _toy_price_frame(columns: list[str]) -> pd.DataFrame:
    requested = set(columns)
    base_columns = {
        "amount",
        "asset",
        "close",
        "date",
        "high",
        "low",
        "open",
        "pre_close",
        "volume",
    }
    all_columns = sorted(requested.union(base_columns))
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2024-01-01", periods=48, freq="D")
    for asset_idx, asset in enumerate(("AAA", "BBB")):
        for day_idx, date in enumerate(dates):
            base = 10.0 + asset_idx * 3.0 + day_idx * 0.1
            row: dict[str, object] = {
                "date": date,
                "asset": asset,
                "open": base,
                "high": base * 1.02,
                "low": base * 0.98,
                "close": base * 1.01,
                "pre_close": base * 0.995,
                "volume": 1000.0 + asset_idx * 100.0 + day_idx * 10.0,
                "amount": (1000.0 + day_idx * 10.0) * base,
            }
            for column in all_columns:
                if column in row:
                    continue
                if column.startswith("is_") or column.startswith("has_"):
                    row[column] = False
                elif column in {"industry", "sector"}:
                    row[column] = "demo"
                else:
                    row[column] = base + float(day_idx % 7)
            rows.append(row)
    return pd.DataFrame(rows, columns=all_columns)


def _validate_import(module: str, errors: list[DraftFactorIssue]) -> None:
    root = module.split(".", 1)[0]
    if root in _FORBIDDEN_IMPORT_ROOTS:
        errors.append(_error("forbidden_import", f"forbidden import in draft factor: {module}"))


def _validate_call(node: ast.Call, errors: list[DraftFactorIssue]) -> None:
    call_name = _call_name(node.func)
    attr_name = call_name.rsplit(".", 1)[-1]
    if call_name in _FORBIDDEN_CALL_NAMES or attr_name in _FORBIDDEN_ATTR_CALLS:
        errors.append(_error("forbidden_call", f"forbidden call in draft factor: {call_name}"))
    if attr_name in {"diff", "pct_change", "shift"} and node.args:
        if _is_negative_number(node.args[0]):
            errors.append(
                _error(
                    "future_shift",
                    f"{attr_name} with a negative period can introduce future data",
                )
            )


def _validate_identifier(identifier: str, errors: list[DraftFactorIssue]) -> None:
    normalized = identifier.lower()
    for fragment in _LEAKAGE_IDENTIFIER_FRAGMENTS:
        if fragment in normalized:
            errors.append(
                _error(
                    "future_label_identifier",
                    f"identifier suggests future label usage inside feature code: {identifier}",
                )
            )


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _is_negative_number(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value < 0
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        operand = node.operand
        return isinstance(operand, ast.Constant) and isinstance(operand.value, (int, float))
    return False


def _string_list(value: object) -> list[str] | None:
    if not isinstance(value, list):
        return None
    out: list[str] = []
    for item in value:
        if not isinstance(item, str):
            return None
        text = item.strip()
        if not text:
            return None
        out.append(text)
    return out


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _path_has_tail(path: Path, expected_tail: tuple[str, ...]) -> bool:
    normalized_parts = tuple(part.lower() for part in path.parts)
    expected = tuple(part.lower() for part in expected_tail)
    return len(normalized_parts) >= len(expected) and normalized_parts[-len(expected) :] == expected


def _error(code: str, message: str) -> DraftFactorIssue:
    return DraftFactorIssue(severity="error", code=code, message=message)


def _warning(code: str, message: str) -> DraftFactorIssue:
    return DraftFactorIssue(severity="warning", code=code, message=message)


def _result(
    path: Path,
    factor_name: str | None,
    code_sha256: str | None,
    factor_json_sha256: str | None,
    *,
    errors: list[DraftFactorIssue],
    warnings: list[DraftFactorIssue],
) -> DraftFactorValidationResult:
    return DraftFactorValidationResult(
        path=path,
        factor_name=factor_name,
        code_sha256=code_sha256,
        factor_json_sha256=factor_json_sha256,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
