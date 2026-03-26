from __future__ import annotations

import tomllib
from pathlib import Path


def test_console_script_entrypoint_declared() -> None:
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    scripts = payload.get("project", {}).get("scripts", {})
    assert scripts.get("alpha-lab") == "alpha_lab.cli:main"
