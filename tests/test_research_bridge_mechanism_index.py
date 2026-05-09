from __future__ import annotations

import json
from pathlib import Path

from alpha_lab.research_bridge.mechanism_index import (
    _vault_hash,
    build_mechanism_index,
    load_mechanism_embeddings,
)


def _build_vault(tmp_path: Path, name: str = "quant-knowledge") -> Path:
    vault = tmp_path / name
    (vault / "30_factors").mkdir(parents=True, exist_ok=True)
    (vault / "90_moc").mkdir(parents=True, exist_ok=True)
    (vault / "90_moc" / "CARD-INDEX.tsv").write_text(
        "path\ttype\tname\tdomain\tlifecycle\ttags\tparent_moc\tsummary\n"
        "30_factors/Factor - Demo.md\tfactor\tFactor - Demo\tprice_action\t"
        "theoretical\tdemo\tMOC - Factors\tDemo summary.\n",
        encoding="utf-8",
    )
    (vault / "30_factors" / "Factor - Demo.md").write_text(
        "# Demo\n\nA demo card.\n",
        encoding="utf-8",
    )
    return vault


def test_build_no_api_key_returns_fallback(tmp_path: Path, monkeypatch) -> None:
    vault = _build_vault(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    outcome = build_mechanism_index(
        vault_root=vault,
        workspace_root=tmp_path,
    )

    assert outcome.cards_total == 1
    assert outcome.cards_regenerated == 0
    assert outcome.cards_cached == 0
    assert outcome.fallback_reason == "no_api_key"


def test_load_mechanism_embeddings_returns_none_when_missing(tmp_path: Path) -> None:
    vault = _build_vault(tmp_path)

    assert load_mechanism_embeddings(workspace_root=tmp_path, vault_root=vault) is None


def test_load_mechanism_embeddings_rejects_vault_mismatch(tmp_path: Path) -> None:
    vault_a = _build_vault(tmp_path, "vault-a")
    vault_b = _build_vault(tmp_path, "vault-b")
    index_root = (
        tmp_path
        / ".research_bridge_cache"
        / "mechanism_index"
        / _vault_hash(vault_a)
    )
    index_root.mkdir(parents=True)
    (index_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "vault_root": str(vault_a.resolve()),
                "vault_hash": _vault_hash(vault_a),
                "card_hashes": {},
            }
        ),
        encoding="utf-8",
    )

    assert load_mechanism_embeddings(workspace_root=tmp_path, vault_root=vault_b) is None
