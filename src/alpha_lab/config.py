from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root resolution
# ---------------------------------------------------------------------------
# This package is designed to be used as an editable install (``pip install -e .``
# or ``uv sync``).  In that layout ``__file__`` resolves to
# ``<project_root>/src/alpha_lab/config.py``, so ``parents[2]`` is the project
# root.
#
# For non-editable installs the file ends up inside the venv site-packages and
# ``parents[2]`` resolves incorrectly.  In that case set the
# ``ALPHA_LAB_PROJECT_ROOT`` environment variable to the project root directory.
# ---------------------------------------------------------------------------

_env_root = os.environ.get("ALPHA_LAB_PROJECT_ROOT")

if _env_root:
    PROJECT_ROOT = Path(_env_root).resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Integrity check: verify the resolved root contains pyproject.toml.  This
# catches incorrect resolution early (e.g. non-editable install without
# ALPHA_LAB_PROJECT_ROOT set) rather than silently writing artifacts to a
# wrong location.
if not (PROJECT_ROOT / "pyproject.toml").exists():
    raise RuntimeError(
        f"alpha_lab.config: PROJECT_ROOT resolved to {PROJECT_ROOT!s} but "
        "'pyproject.toml' was not found there.  If the package was installed "
        "non-editablely, set the ALPHA_LAB_PROJECT_ROOT environment variable "
        "to the project root directory."
    )

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
