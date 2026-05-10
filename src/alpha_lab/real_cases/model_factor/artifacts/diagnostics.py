from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

# Cross-module imports (auto-added)
from ._utils import _write_json


def write_diagnostics_artifact(
    *,
    output_dir: str | Path,
    diagnostics_payload: Mapping[str, object],
    pretty: bool = False,
) -> Path:
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_path = out_dir / "diagnostics.json"
    _write_json(diagnostics_path, diagnostics_payload, pretty=pretty)
    return diagnostics_path
