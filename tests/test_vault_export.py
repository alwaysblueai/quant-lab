from __future__ import annotations

from pathlib import Path

from alpha_lab.vault_export import export_to_vault


def _write_sources(tmp_path: Path, *, content: str = "card") -> dict[str, Path]:
    src_dir = tmp_path / "source"
    src_dir.mkdir(parents=True, exist_ok=True)

    card = src_dir / "experiment_card.md"
    summary = src_dir / "summary.md"
    manifest = src_dir / "run_manifest.json"

    card.write_text(content, encoding="utf-8")
    summary.write_text("summary", encoding="utf-8")
    manifest.write_text('{"ok": true}', encoding="utf-8")

    return {
        "experiment_card_path": card,
        "summary_path": summary,
        "manifest_path": manifest,
    }


def test_vault_export_versioned_creates_timestamp_and_latest(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    vault_root.mkdir(parents=True, exist_ok=True)
    source_paths = _write_sources(tmp_path)

    result = export_to_vault(
        source_paths,
        case_name="value_quality_lowvol_v1",
        vault_root=vault_root,
        mode="versioned",
    )

    assert result.success
    assert result.status == "success"
    case_dir = vault_root / "50_experiments" / "value_quality_lowvol_v1"
    assert (case_dir / "latest.md").exists()
    timestamped = list(case_dir.glob("*__experiment_card.md"))
    assert len(timestamped) == 1


def test_vault_export_overwrite_replaces_latest(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    vault_root.mkdir(parents=True, exist_ok=True)

    source_a = _write_sources(tmp_path / "a", content="first")
    source_b = _write_sources(tmp_path / "b", content="second")

    result_a = export_to_vault(
        source_a,
        case_name="case_a",
        vault_root=vault_root,
        mode="overwrite",
    )
    result_b = export_to_vault(
        source_b,
        case_name="case_a",
        vault_root=vault_root,
        mode="overwrite",
    )

    assert result_a.success and result_b.success
    latest = vault_root / "50_experiments" / "case_a" / "latest.md"
    assert latest.read_text(encoding="utf-8") == "second"


def test_vault_export_skip_mode_writes_nothing(tmp_path: Path) -> None:
    vault_root = tmp_path / "vault"
    vault_root.mkdir(parents=True, exist_ok=True)
    source_paths = _write_sources(tmp_path)

    result = export_to_vault(
        source_paths,
        case_name="case_skip",
        vault_root=vault_root,
        mode="skip",
    )

    assert result.success
    assert result.status == "skipped"
    assert result.mode_used == "skip"
    assert not (vault_root / "50_experiments").exists()


def test_vault_export_missing_vault_path_is_skip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("OBSIDIAN_VAULT_PATH", raising=False)
    source_paths = _write_sources(tmp_path)

    result = export_to_vault(
        source_paths,
        case_name="case_no_vault",
        vault_root=None,
        mode="versioned",
    )

    assert result.success
    assert result.status == "skipped"
    assert result.mode_used == "skip"


def test_vault_export_invalid_path_is_handled_gracefully(tmp_path: Path) -> None:
    source_paths = _write_sources(tmp_path)

    result = export_to_vault(
        source_paths,
        case_name="case_bad_path",
        vault_root=tmp_path / "does_not_exist",
        mode="versioned",
    )

    assert not result.success
    assert result.status == "failed"
    assert result.error is not None
