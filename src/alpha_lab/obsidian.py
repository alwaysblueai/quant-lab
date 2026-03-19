from __future__ import annotations

from pathlib import Path


def write_obsidian_note(
    markdown: str,
    output_path: str | Path,
    *,
    overwrite: bool = False,
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
    """
    if not isinstance(markdown, str):
        raise TypeError(f"markdown must be str, got {type(markdown).__name__}")

    path = Path(output_path)

    if path.is_dir():
        raise ValueError(
            f"output_path {path!s} is an existing directory; provide a file path."
        )

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Note already exists at {path!s}.  "
            "Pass overwrite=True to replace it."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(markdown, encoding="utf-8")
    return path
