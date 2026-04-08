from __future__ import annotations

import re
from pathlib import Path

from alpha_lab.exceptions import AlphaLabDataError, LifecyclePromotionError, VaultWriteError

# Lifecycle values that do NOT require a backlink gate.  Any other value
# (specifically "validated-backtest" and anything more advanced) triggers the
# lifecycle backlink check.
_UNGATED_LIFECYCLE_VALUES: frozenset[str] = frozenset(
    {"", "draft", "active", "theoretical"}
)


def write_obsidian_note(
    markdown: str,
    output_path: str | Path,
    *,
    overwrite: bool = False,
    restricted_root: Path | None = None,
) -> Path:
    """Write a markdown string to disk as an Obsidian note.

    Parameters
    ----------
    markdown:
        Note content to write.  Must be a :class:`str`.
    output_path:
        Destination file path.  Parent directories are created automatically.
        Must not resolve to an existing directory.
    overwrite:
        If ``False`` (default), raises :exc:`FileExistsError` when the
        destination file already exists.  Set to ``True`` to replace it.
    restricted_root:
        When provided, ``output_path`` must resolve to a location inside this
        directory.  Also enables the lifecycle backlink gate: if the markdown
        frontmatter contains ``lifecycle: validated-backtest`` (or any
        non-draft/non-theoretical value) and no ``[[50_experiments/...]]``
        wikilink is present, :exc:`LifecyclePromotionError` is raised.  Pass
        ``None`` (default) to disable both guards.

    Returns
    -------
    Path
        Resolved path of the written file.

    Raises
    ------
    TypeError
        If ``markdown`` is not a :class:`str`.
    ValueError
        If ``output_path`` resolves to an existing directory.
    FileExistsError
        If the destination file already exists and ``overwrite`` is ``False``.
    VaultWriteError
        If ``restricted_root`` is set and ``output_path`` falls outside it.
    LifecyclePromotionError
        If ``restricted_root`` is set and the markdown contains a gated
        lifecycle value without a ``[[50_experiments/...]]`` backlink.
    """
    if not isinstance(markdown, str):
        raise TypeError(f"markdown must be str, got {type(markdown).__name__}")

    path = Path(output_path).resolve()

    if path.is_dir():
        raise ValueError(
            f"output_path {path!s} is an existing directory; provide a file path."
        )

    # --- Vault write boundary guard -----------------------------------------
    if restricted_root is not None:
        resolved_root = restricted_root.resolve()
        try:
            path.relative_to(resolved_root)
        except ValueError:
            raise VaultWriteError(
                f"Vault write rejected: {path!s} is outside the authorized root "
                f"{resolved_root!s}.  Use export_to_vault() for vault writes."
            )

        # --- Lifecycle backlink gate -----------------------------------------
        lifecycle_value = _parse_frontmatter_field(markdown, "lifecycle")
        if lifecycle_value.strip().lower() not in _UNGATED_LIFECYCLE_VALUES:
            if not re.search(r"\[\[50_experiments/", markdown):
                raise LifecyclePromotionError(
                    f"lifecycle: {lifecycle_value!r} requires at least one "
                    "[[50_experiments/...]] backlink to an experiment artifact. "
                    "Add the backlink before promoting this card."
                )

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Note already exists at {path!s}.  "
            "Pass overwrite=True to replace it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path


def _parse_frontmatter_field(markdown: str, field: str) -> str:
    """Extract a scalar YAML frontmatter field value, or '' if not present."""
    if not markdown.startswith("---"):
        return ""
    end = markdown.find("\n---", 3)
    if end == -1:
        return ""
    frontmatter = markdown[3:end]
    pattern = re.compile(
        r"^" + re.escape(field) + r"\s*:\s*(.+)$",
        re.MULTILINE,
    )
    match = pattern.search(frontmatter)
    if not match:
        return ""
    return match.group(1).strip().strip('"').strip("'")
