"""Layered exception hierarchy for alpha-lab.

All alpha-lab specific exceptions inherit from :class:`AlphaLabError` so
callers can catch the entire family with a single ``except AlphaLabError``.

CLI layers should catch these and translate to appropriate exit codes /
user-facing messages rather than leaking raw tracebacks.
"""

from __future__ import annotations


class AlphaLabError(Exception):
    """Base exception for all alpha-lab errors."""


class AlphaLabDataError(AlphaLabError, ValueError):
    """Raised when input data violates a contract.

    Examples: missing columns, NaN in required fields, duplicate rows,
    empty DataFrames where non-empty is required.

    Inherits from ``ValueError`` for backward compatibility with existing
    ``except ValueError`` handlers.
    """


class AlphaLabConfigError(AlphaLabError, ValueError):
    """Raised when configuration or spec validation fails.

    Examples: invalid spec fields, missing required paths, conflicting
    parameter combinations.
    """


class AlphaLabIOError(AlphaLabError, OSError):
    """Raised when file I/O operations fail.

    Examples: missing input files, unreadable CSV, write permission errors.
    Inherits from ``OSError`` (parent of ``FileNotFoundError``) for
    backward compatibility.
    """


class AlphaLabExperimentError(AlphaLabError, RuntimeError):
    """Raised when an experiment pipeline encounters an unrecoverable error.

    Examples: no walk-forward folds produced, factor_fn returns invalid
    output, integrity hard-failure during execution.
    """


class VaultWriteError(AlphaLabError, PermissionError):
    """Raised when a vault write would land outside the authorized root.

    The sole authorized vault write path is ``export_to_vault()``, which
    always writes under ``50_experiments/``.  Any attempt to write elsewhere
    inside the vault via ``write_obsidian_note(restricted_root=...)`` is
    blocked with this error.
    """


class LifecyclePromotionError(AlphaLabError, ValueError):
    """Raised when a lifecycle-gated write lacks the required backlink.

    Cards with ``lifecycle: validated-backtest`` (or higher) must contain at
    least one ``[[50_experiments/...]]`` wikilink so the promotion is
    traceable to an experiment artifact.
    """
