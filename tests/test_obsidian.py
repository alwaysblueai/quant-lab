from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.cli import main
from alpha_lab.obsidian import write_obsidian_note
from alpha_lab.registry import load_registry

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _write_prices_csv(path: Path, n_assets: int = 6, n_days: int = 30, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows = []
    for asset in [f"A{i}" for i in range(n_assets)]:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date.strftime("%Y-%m-%d"), "asset": asset, "close": price})
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


_SAMPLE_MD = "# Test\n\nSome content.\n"


# ---------------------------------------------------------------------------
# 1. write_obsidian_note — basic correctness
# ---------------------------------------------------------------------------


def test_write_obsidian_note_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "note.md"
    write_obsidian_note(_SAMPLE_MD, out)
    assert out.exists()


def test_write_obsidian_note_returns_path(tmp_path: Path) -> None:
    out = tmp_path / "note.md"
    result = write_obsidian_note(_SAMPLE_MD, out)
    assert isinstance(result, Path)
    assert result == out


def test_write_obsidian_note_content_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "note.md"
    write_obsidian_note(_SAMPLE_MD, out)
    assert out.read_text(encoding="utf-8") == _SAMPLE_MD


def test_write_obsidian_note_accepts_string_path(tmp_path: Path) -> None:
    out = str(tmp_path / "note.md")
    write_obsidian_note(_SAMPLE_MD, out)
    assert Path(out).exists()


# ---------------------------------------------------------------------------
# 2. write_obsidian_note — parent directory creation
# ---------------------------------------------------------------------------


def test_write_obsidian_note_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "a" / "b" / "c" / "note.md"
    assert not out.parent.exists()
    write_obsidian_note(_SAMPLE_MD, out)
    assert out.exists()


# ---------------------------------------------------------------------------
# 3. write_obsidian_note — overwrite protection
# ---------------------------------------------------------------------------


def test_write_obsidian_note_raises_on_existing_file(tmp_path: Path) -> None:
    out = tmp_path / "note.md"
    write_obsidian_note(_SAMPLE_MD, out)
    with pytest.raises(FileExistsError, match="already exists"):
        write_obsidian_note(_SAMPLE_MD, out)


def test_write_obsidian_note_overwrite_false_does_not_modify(tmp_path: Path) -> None:
    out = tmp_path / "note.md"
    original = "# Original\n"
    write_obsidian_note(original, out)
    with pytest.raises(FileExistsError):
        write_obsidian_note("# New\n", out, overwrite=False)
    assert out.read_text(encoding="utf-8") == original


def test_write_obsidian_note_overwrite_true_replaces_file(tmp_path: Path) -> None:
    out = tmp_path / "note.md"
    write_obsidian_note("# Old\n", out)
    write_obsidian_note("# New\n", out, overwrite=True)
    assert out.read_text(encoding="utf-8") == "# New\n"


# ---------------------------------------------------------------------------
# 4. write_obsidian_note — input validation
# ---------------------------------------------------------------------------


def test_write_obsidian_note_rejects_non_string_markdown(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="must be str"):
        write_obsidian_note(123, tmp_path / "note.md")  # type: ignore[arg-type]


def test_write_obsidian_note_rejects_directory_as_output(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="existing directory"):
        write_obsidian_note(_SAMPLE_MD, tmp_path)


# ---------------------------------------------------------------------------
# 5. Markdown structure sanity (from to_obsidian_markdown)
# ---------------------------------------------------------------------------


def test_obsidian_markdown_has_yaml_frontmatter(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    md_path = tmp_path / "note.md"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
            "--obsidian-markdown-path", str(md_path),
        ]
    )
    content = md_path.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "factor:" in content
    assert "quantiles:" in content
    assert "date:" in content
    assert "tags:" in content


def test_obsidian_markdown_has_horizon_in_frontmatter(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    md_path = tmp_path / "note.md"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "3",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
            "--obsidian-markdown-path", str(md_path),
        ]
    )
    content = md_path.read_text(encoding="utf-8")
    assert "horizon: 3" in content


def test_obsidian_markdown_has_required_sections(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    md_path = tmp_path / "note.md"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
            "--obsidian-markdown-path", str(md_path),
        ]
    )
    content = md_path.read_text(encoding="utf-8")
    for section in ("## Experiment", "## Summary Metrics", "## Interpretation", "## Next Steps"):
        assert section in content


# ---------------------------------------------------------------------------
# 6. CLI — directory path auto-generates filename
# ---------------------------------------------------------------------------


def test_cli_obsidian_dir_path_creates_file_in_dir(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    notes_dir = tmp_path / "notes"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
            "--experiment-name", "dir_test",
            "--obsidian-markdown-path", str(notes_dir) + "/",
        ]
    )
    md_files = list(notes_dir.glob("*.md"))
    assert len(md_files) == 1


def test_cli_obsidian_dir_path_filename_contains_experiment_name(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    notes_dir = tmp_path / "notes"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
            "--experiment-name", "dir_name_test",
            "--obsidian-markdown-path", str(notes_dir) + "/",
        ]
    )
    md_files = list(notes_dir.glob("*.md"))
    assert "dir_name_test" in md_files[0].name


def test_cli_obsidian_dir_path_filename_starts_with_date(tmp_path: Path) -> None:
    import datetime

    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    notes_dir = tmp_path / "notes"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
            "--experiment-name", "date_test",
            "--obsidian-markdown-path", str(notes_dir) + "/",
        ]
    )
    md_files = list(notes_dir.glob("*.md"))
    today = datetime.date.today().isoformat()
    assert md_files[0].name.startswith(today)


def test_cli_obsidian_existing_dir_auto_generates_filename(tmp_path: Path) -> None:
    """When --obsidian-markdown-path points to an already-existing directory,
    auto-filename generation must still trigger."""
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    notes_dir = tmp_path / "notes"
    notes_dir.mkdir()
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
            "--experiment-name", "existing_dir_test",
            "--obsidian-markdown-path", str(notes_dir),
        ]
    )
    md_files = list(notes_dir.glob("*.md"))
    assert len(md_files) == 1


# ---------------------------------------------------------------------------
# 7. CLI — overwrite protection
# ---------------------------------------------------------------------------


def test_cli_obsidian_refuses_overwrite_by_default(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    md_path = tmp_path / "note.md"
    args = [
        "--input-path", str(prices_csv),
        "--factor", "momentum",
        "--label-horizon", "5",
        "--quantiles", "5",
        "--output-dir", str(tmp_path / "out"),
        "--obsidian-markdown-path", str(md_path),
    ]
    main(args)
    with pytest.raises(SystemExit):
        main(args)  # second run must fail — file exists


def test_cli_obsidian_overwrite_flag_allows_rewrite(tmp_path: Path) -> None:
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    md_path = tmp_path / "note.md"
    args = [
        "--input-path", str(prices_csv),
        "--factor", "momentum",
        "--label-horizon", "5",
        "--quantiles", "5",
        "--output-dir", str(tmp_path / "out"),
        "--obsidian-markdown-path", str(md_path),
        "--obsidian-overwrite",
    ]
    assert main(args) == 0
    assert main(args) == 0  # second run must succeed


# ---------------------------------------------------------------------------
# 8. Registry path propagation
# ---------------------------------------------------------------------------


def test_cli_registry_stores_resolved_file_path_not_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a directory path is supplied, the registry must store the resolved
    file path, not the directory."""
    import alpha_lab.registry as reg_module

    registry_path = tmp_path / "registry.csv"
    monkeypatch.setattr(reg_module, "DEFAULT_REGISTRY_PATH", registry_path)
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    notes_dir = tmp_path / "notes"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
            "--experiment-name", "path_test",
            "--obsidian-markdown-path", str(notes_dir) + "/",
            "--append-registry",
        ]
    )
    df = load_registry(registry_path)
    stored_path = str(df["obsidian_path"].iloc[0])
    # Must be a .md file path, not the directory
    assert stored_path.endswith(".md")
    assert Path(stored_path).exists()


def test_cli_registry_obsidian_path_matches_written_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import alpha_lab.registry as reg_module

    registry_path = tmp_path / "registry.csv"
    monkeypatch.setattr(reg_module, "DEFAULT_REGISTRY_PATH", registry_path)
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    md_path = tmp_path / "my_note.md"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
            "--experiment-name", "path_match_test",
            "--obsidian-markdown-path", str(md_path),
            "--append-registry",
        ]
    )
    df = load_registry(registry_path)
    stored = str(df["obsidian_path"].iloc[0])
    assert stored == str(md_path)
    assert Path(stored).exists()


# ---------------------------------------------------------------------------
# 9. NaN metrics render correctly in written note
# ---------------------------------------------------------------------------


def test_obsidian_note_does_not_contain_literal_nan(tmp_path: Path) -> None:
    """No metric should appear as the literal string 'nan' in the note.
    NaN values are rendered as an em dash by the reporting layer."""
    prices_csv = _write_prices_csv(tmp_path / "prices.csv")
    md_path = tmp_path / "note.md"
    main(
        [
            "--input-path", str(prices_csv),
            "--factor", "momentum",
            "--label-horizon", "5",
            "--quantiles", "5",
            "--output-dir", str(tmp_path / "out"),
            "--obsidian-markdown-path", str(md_path),
        ]
    )
    content = md_path.read_text(encoding="utf-8")
    # Strip known occurrences of "nan" as part of legitimate words before checking
    # (e.g. "quantiles", "quant" in frontmatter tags do not contain "nan").
    # The simplest check: "nan" must not appear as a standalone cell value.
    assert "| nan |" not in content
    assert "| NaN |" not in content


def test_obsidian_note_nan_metric_renders_as_em_dash_unit() -> None:
    """Unit test: NaN metrics from a constant factor (no cross-section IC)
    are rendered as em dash, not as the literal string 'nan'."""
    import numpy as np
    import pandas as pd

    from alpha_lab.experiment import run_factor_experiment
    from alpha_lab.reporting import to_obsidian_markdown

    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    rows = []
    for asset in ["A0", "A1", "A2", "A3"]:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": price})
    prices = pd.DataFrame(rows)

    # Constant factor → zero cross-sectional variance → IC is always NaN
    def constant_fn(p: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            [{"date": d, "asset": a, "factor": "const", "value": 1.0}
             for d in p["date"].unique() for a in p["asset"].unique()]
        )

    result = run_factor_experiment(prices, constant_fn)
    md = to_obsidian_markdown(result)
    assert "| nan |" not in md
    assert "| NaN |" not in md
    assert "\u2014" in md  # em dash present for NaN IC
