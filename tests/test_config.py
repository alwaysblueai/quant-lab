"""Tests for alpha_lab.config path resolution."""
from __future__ import annotations

from pathlib import Path

import pytest


def test_project_root_resolves_correctly() -> None:
    """PROJECT_ROOT should point to the directory containing pyproject.toml."""
    from alpha_lab.config import PROJECT_ROOT

    assert (PROJECT_ROOT / "pyproject.toml").exists(), (
        f"PROJECT_ROOT={PROJECT_ROOT} does not contain pyproject.toml"
    )


def test_data_dirs_are_under_project_root() -> None:
    from alpha_lab.config import DATA_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR

    assert DATA_DIR == PROJECT_ROOT / "data"
    assert RAW_DATA_DIR == DATA_DIR / "raw"
    assert PROCESSED_DATA_DIR == DATA_DIR / "processed"


def test_env_var_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """ALPHA_LAB_PROJECT_ROOT env var overrides __file__-based resolution."""
    # Create a fake project root with pyproject.toml so the integrity check passes.
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'fake'\n")

    monkeypatch.setenv("ALPHA_LAB_PROJECT_ROOT", str(tmp_path))

    import importlib

    import alpha_lab.config as cfg

    importlib.reload(cfg)
    try:
        assert cfg.PROJECT_ROOT == tmp_path.resolve()
        assert cfg.DATA_DIR == tmp_path.resolve() / "data"
    finally:
        # Reload back to actual state so other tests are not affected.
        monkeypatch.delenv("ALPHA_LAB_PROJECT_ROOT", raising=False)
        importlib.reload(cfg)


def test_invalid_env_var_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A project root without pyproject.toml should raise RuntimeError."""
    # tmp_path exists but has no pyproject.toml
    monkeypatch.setenv("ALPHA_LAB_PROJECT_ROOT", str(tmp_path))

    import importlib

    import alpha_lab.config as cfg

    with pytest.raises(RuntimeError, match="pyproject.toml"):
        importlib.reload(cfg)

    # Clean up — reload with correct state
    monkeypatch.delenv("ALPHA_LAB_PROJECT_ROOT", raising=False)
    importlib.reload(cfg)
