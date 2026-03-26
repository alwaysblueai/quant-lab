"""Tests for export_experiment_card — the alpha-lab -> quant-knowledge integration.

Covers:
  1. Overwrite protection (safe default)
  2. Valid path creation (50_experiments subdir auto-created)
  3. Directory collision / invalid target
  4. Deterministic filename generation
  5. Final written path return value
  6. Markdown contains auto-generated / manual-edit guidance
  7. Vault existence requirement
  8. Empty / whitespace name rejection
  9. Name with path separators rejected
"""

from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_lab.experiment import run_factor_experiment
from alpha_lab.experiment_metadata import ExperimentMetadata
from alpha_lab.factors.momentum import momentum
from alpha_lab.reporting import export_experiment_card
from alpha_lab.timing import DelaySpec

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_prices(n_assets: int = 6, n_days: int = 30, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    rows = []
    for asset in [f"A{i}" for i in range(n_assets)]:
        price = 100.0
        for date in dates:
            price *= 1.0 + rng.normal(0.0, 0.01)
            rows.append({"date": date, "asset": asset, "close": price})
    return pd.DataFrame(rows)


def _momentum_fn(prices: pd.DataFrame) -> pd.DataFrame:
    return momentum(prices, window=5)


@pytest.fixture()
def result():
    return run_factor_experiment(_make_prices(), _momentum_fn)


@pytest.fixture()
def vault(tmp_path: Path) -> Path:
    """A temporary directory that acts as the quant-knowledge vault root."""
    v = tmp_path / "quant-knowledge"
    v.mkdir()
    return v


# ---------------------------------------------------------------------------
# 1. Overwrite protection — safe default
# ---------------------------------------------------------------------------


def test_export_raises_on_existing_file(vault: Path, result) -> None:
    export_experiment_card(result, name="test-factor", vault_path=vault)
    with pytest.raises(FileExistsError, match="already exists"):
        export_experiment_card(result, name="test-factor", vault_path=vault)


def test_export_overwrite_false_does_not_modify(vault: Path, result) -> None:
    path = export_experiment_card(result, name="test-factor", vault_path=vault)
    original = path.read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        export_experiment_card(result, name="test-factor", vault_path=vault, overwrite=False)
    assert path.read_text(encoding="utf-8") == original


def test_export_overwrite_true_replaces_file(vault: Path, result) -> None:
    path = export_experiment_card(result, name="test-factor", vault_path=vault)
    first_content = path.read_text(encoding="utf-8")
    path2 = export_experiment_card(result, name="test-factor", vault_path=vault, overwrite=True)
    assert path2 == path
    # File must have been written (content identical for same result — just check it exists)
    assert path.read_text(encoding="utf-8") == first_content


# ---------------------------------------------------------------------------
# 2. Valid path creation — 50_experiments subdir is created on demand
# ---------------------------------------------------------------------------


def test_export_creates_experiments_subdir(vault: Path, result) -> None:
    subdir = vault / "50_experiments"
    assert not subdir.exists()
    export_experiment_card(result, name="test-factor", vault_path=vault)
    assert subdir.is_dir()


def test_export_creates_markdown_file(vault: Path, result) -> None:
    path = export_experiment_card(result, name="test-factor", vault_path=vault)
    assert path.exists()
    assert path.suffix == ".md"


def test_export_file_is_inside_50_experiments(vault: Path, result) -> None:
    path = export_experiment_card(result, name="test-factor", vault_path=vault)
    assert path.parent.name == "50_experiments"
    assert path.parent.parent == vault


# ---------------------------------------------------------------------------
# 3. Vault existence requirement
# ---------------------------------------------------------------------------


def test_export_raises_when_vault_does_not_exist(tmp_path: Path, result) -> None:
    nonexistent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        export_experiment_card(result, name="test-factor", vault_path=nonexistent)


def test_export_raises_when_vault_is_a_file(tmp_path: Path, result) -> None:
    file_path = tmp_path / "vault_as_file.txt"
    file_path.write_text("oops")
    with pytest.raises(NotADirectoryError):
        export_experiment_card(result, name="test-factor", vault_path=file_path)


# ---------------------------------------------------------------------------
# 4. Deterministic filename generation
# ---------------------------------------------------------------------------


def test_export_filename_contains_name(vault: Path, result) -> None:
    path = export_experiment_card(result, name="momentum-5d", vault_path=vault)
    assert "momentum-5d" in path.name


def test_export_filename_contains_yyyymm(vault: Path, result) -> None:
    yyyymm = datetime.date.today().strftime("%Y%m")
    path = export_experiment_card(result, name="momentum-5d", vault_path=vault)
    assert yyyymm in path.name


def test_export_filename_starts_with_exp(vault: Path, result) -> None:
    path = export_experiment_card(result, name="momentum-5d", vault_path=vault)
    assert path.name.startswith("Exp - ")


def test_export_filename_is_deterministic(vault: Path, result) -> None:
    """Two calls with the same name on the same day produce the same filename."""
    path = export_experiment_card(result, name="deterministic-test", vault_path=vault)
    yyyymm = datetime.date.today().strftime("%Y%m")
    expected_name = f"Exp - {yyyymm} - deterministic-test.md"
    assert path.name == expected_name


# ---------------------------------------------------------------------------
# 5. Return value — final written path
# ---------------------------------------------------------------------------


def test_export_returns_path_instance(vault: Path, result) -> None:
    path = export_experiment_card(result, name="return-type-test", vault_path=vault)
    assert isinstance(path, Path)


def test_export_returned_path_exists(vault: Path, result) -> None:
    path = export_experiment_card(result, name="return-exists-test", vault_path=vault)
    assert path.exists()


def test_export_returned_path_matches_written_file(vault: Path, result) -> None:
    path = export_experiment_card(result, name="path-match-test", vault_path=vault)
    yyyymm = datetime.date.today().strftime("%Y%m")
    expected = vault / "50_experiments" / f"Exp - {yyyymm} - path-match-test.md"
    assert path == expected.resolve() or path == expected


# ---------------------------------------------------------------------------
# 6. Markdown content — auto-generated / manual-edit guidance
# ---------------------------------------------------------------------------


def test_export_markdown_has_autogenerated_notice(vault: Path, result) -> None:
    path = export_experiment_card(result, name="notice-test", vault_path=vault)
    content = path.read_text(encoding="utf-8")
    assert "Auto-generated" in content or "auto-generated" in content


def test_export_markdown_indicates_manual_sections(vault: Path, result) -> None:
    path = export_experiment_card(result, name="manual-notice-test", vault_path=vault)
    content = path.read_text(encoding="utf-8")
    # Note must distinguish what the researcher should fill in
    assert "manual" in content.lower() or "Manual" in content


def test_export_markdown_has_autogen_and_manual_in_same_note(vault: Path, result) -> None:
    path = export_experiment_card(result, name="both-notice-test", vault_path=vault)
    content = path.read_text(encoding="utf-8")
    lower = content.lower()
    assert "auto-generated" in lower
    assert "manual" in lower


def test_export_markdown_includes_delay_and_validation_fields(vault: Path) -> None:
    rich_result = run_factor_experiment(
        _make_prices(),
        _momentum_fn,
        horizon=5,
        delay_spec=DelaySpec.for_horizon(5, purge_periods=2, embargo_periods=1),
        metadata=ExperimentMetadata(
            dataset_id="dataset-v1",
            dataset_hash="abc123",
            trial_id="trial-7",
        ),
    )
    path = export_experiment_card(rich_result, name="delay-validation-test", vault_path=vault)
    content = path.read_text(encoding="utf-8")
    assert "Decision timestamp" in content
    assert "Validation scheme" in content
    assert "dataset_id: dataset-v1" in content
    assert "dataset_hash: abc123" in content


def test_export_markdown_notice_appears_before_setup(vault: Path, result) -> None:
    path = export_experiment_card(result, name="notice-order-test", vault_path=vault)
    content = path.read_text(encoding="utf-8")
    notice_pos = content.lower().find("auto-generated")
    setup_pos = content.find("## Setup")
    assert notice_pos < setup_pos


# ---------------------------------------------------------------------------
# 7. Input validation — name
# ---------------------------------------------------------------------------


def test_export_raises_on_empty_name(vault: Path, result) -> None:
    with pytest.raises(ValueError, match="empty"):
        export_experiment_card(result, name="", vault_path=vault)


def test_export_raises_on_whitespace_name(vault: Path, result) -> None:
    with pytest.raises(ValueError, match="empty"):
        export_experiment_card(result, name="   ", vault_path=vault)


def test_export_raises_on_name_with_forward_slash(vault: Path, result) -> None:
    with pytest.raises(ValueError, match="path separators"):
        export_experiment_card(result, name="bad/name", vault_path=vault)


def test_export_raises_on_name_with_backslash(vault: Path, result) -> None:
    with pytest.raises(ValueError, match="path separators"):
        export_experiment_card(result, name="bad\\name", vault_path=vault)


# ---------------------------------------------------------------------------
# 8. Markdown content — structural correctness
# ---------------------------------------------------------------------------


def test_export_markdown_has_yaml_frontmatter(vault: Path, result) -> None:
    path = export_experiment_card(result, name="frontmatter-test", vault_path=vault)
    content = path.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "type: experiment" in content


def test_export_markdown_has_required_sections(vault: Path, result) -> None:
    path = export_experiment_card(result, name="sections-test", vault_path=vault)
    content = path.read_text(encoding="utf-8")
    for section in ("## Setup", "## Results", "## Interpretation", "## Next Steps"):
        assert section in content, f"Missing section: {section}"


def test_export_markdown_ends_with_newline(vault: Path, result) -> None:
    path = export_experiment_card(result, name="newline-test", vault_path=vault)
    content = path.read_text(encoding="utf-8")
    assert content.endswith("\n")
