from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[a-z0-9_]+|[\u4e00-\u9fff]+", re.IGNORECASE)
DEFAULT_SIMILARITY_THRESHOLD = 0.35


@dataclass(frozen=True, slots=True)
class SearchResult:
    name: str
    score: float
    type: str
    path: str
    summary: str


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    normalized = re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", lowered)
    return SPACE_RE.sub(" ", normalized).strip()


def _hash_index(feature: str, dimension: int) -> int:
    digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dimension


def encode_text(text: str, *, dimension: int, idf: np.ndarray) -> np.ndarray:
    normalized = _normalize_text(text)
    counts: dict[int, float] = {}
    for token in WORD_RE.findall(normalized):
        idx = _hash_index(f"w:{token}", dimension)
        counts[idx] = counts.get(idx, 0.0) + 1.0
    compact = normalized.replace(" ", "")
    for gram_size in (3, 4):
        if len(compact) < gram_size:
            continue
        for idx_offset in range(len(compact) - gram_size + 1):
            gram = compact[idx_offset : idx_offset + gram_size]
            idx = _hash_index(f"c:{gram}", dimension)
            counts[idx] = counts.get(idx, 0.0) + 0.35

    vector = np.zeros(dimension, dtype=np.float32)
    for idx, raw_tf in counts.items():
        vector[idx] = (1.0 + np.log(raw_tf)) * float(idf[idx])
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector /= norm
    return vector


class VaultEmbeddings:
    def __init__(self, embeddings_path: Path):
        self.embeddings_path = Path(embeddings_path).expanduser().resolve()
        self._names: list[str] = []
        self._types: list[str] = []
        self._paths: list[str] = []
        self._summaries: list[str] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._idf = np.zeros(0, dtype=np.float32)
        self._dimension = 0
        self._name_to_index: dict[str, int] = {}
        self._loaded = False

    @classmethod
    def from_vault_root(cls, vault_root: Path) -> VaultEmbeddings:
        return cls(Path(vault_root).expanduser().resolve() / "90_computed" / "embeddings.npz")

    def build(
        self,
        card_index_tsv: Path | None = None,
        vault_root: Path | None = None,
        model: str = "hash-tfidf-v1",
    ) -> None:
        del model
        if not self.embeddings_path.exists():
            self.rebuild_cache(card_index_tsv=card_index_tsv, vault_root=vault_root)
        self.load()

    def rebuild_cache(
        self,
        card_index_tsv: Path | None = None,
        vault_root: Path | None = None,
    ) -> Path:
        resolved_vault = self._resolve_vault_root(
            vault_root=vault_root, card_index_tsv=card_index_tsv
        )
        script_path = resolved_vault / "00_protocols" / "rebuild-embeddings.py"
        if not script_path.exists():
            raise FileNotFoundError(f"rebuild-embeddings.py not found: {script_path}")

        proc = subprocess.run(
            [sys.executable, str(script_path), str(resolved_vault)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "unknown rebuild-embeddings failure").strip()
            raise RuntimeError(f"failed to rebuild embeddings cache: {detail}")
        return self.embeddings_path

    def load(self) -> None:
        payload = np.load(self.embeddings_path, allow_pickle=False)
        names = payload["names"]
        types = payload["types"]
        paths = payload["paths"]
        summaries = payload["summaries"]
        matrix = payload["matrix"]
        idf = payload["idf"]
        dimension = int(payload["dimension"][0])
        if matrix.ndim != 2:
            raise ValueError(f"embeddings matrix must be 2-D: {self.embeddings_path}")
        if matrix.shape[0] != len(names):
            raise ValueError(f"embeddings rows mismatch names: {self.embeddings_path}")
        if matrix.shape[1] != dimension or len(idf) != dimension:
            raise ValueError(f"embeddings dimension mismatch: {self.embeddings_path}")

        self._names = [str(item) for item in names.tolist()]
        self._types = [str(item) for item in types.tolist()]
        self._paths = [str(item) for item in paths.tolist()]
        self._summaries = [str(item) for item in summaries.tolist()]
        self._matrix = matrix.astype(np.float32)
        self._idf = idf.astype(np.float32)
        self._dimension = dimension
        self._name_to_index = {name: idx for idx, name in enumerate(self._names)}
        self._loaded = True

    def encode_query(self, query: str) -> np.ndarray:
        self._require_loaded()
        return encode_text(query, dimension=self._dimension, idf=self._idf)

    def search(
        self, query: str, top_k: int = 10, type_filter: str | None = None
    ) -> list[SearchResult]:
        self._require_loaded()
        query_vec = self.encode_query(query)
        scores = self._matrix @ query_vec
        ranked = np.argsort(scores)[::-1]
        results: list[SearchResult] = []
        for idx in ranked:
            score = float(scores[int(idx)])
            if score <= 0.0:
                break
            if type_filter and self._types[int(idx)] != type_filter:
                continue
            results.append(
                SearchResult(
                    name=self._names[int(idx)],
                    score=score,
                    type=self._types[int(idx)],
                    path=self._paths[int(idx)],
                    summary=self._summaries[int(idx)],
                )
            )
            if len(results) >= top_k:
                break
        return results

    def pairwise_similarity(self, name_a: str, name_b: str) -> float:
        self._require_loaded()
        idx_a = self._name_to_index.get(name_a)
        idx_b = self._name_to_index.get(name_b)
        if idx_a is None or idx_b is None:
            return 0.0
        return float(self._matrix[idx_a] @ self._matrix[idx_b])

    def get_entry(self, name: str) -> SearchResult | None:
        self._require_loaded()
        idx = self._name_to_index.get(name)
        if idx is None:
            return None
        return SearchResult(
            name=self._names[idx],
            score=1.0,
            type=self._types[idx],
            path=self._paths[idx],
            summary=self._summaries[idx],
        )

    def iter_entries(self, *, type_filter: str | None = None) -> list[SearchResult]:
        self._require_loaded()
        entries: list[SearchResult] = []
        for idx, name in enumerate(self._names):
            if type_filter and self._types[idx] != type_filter:
                continue
            entries.append(
                SearchResult(
                    name=name,
                    score=1.0,
                    type=self._types[idx],
                    path=self._paths[idx],
                    summary=self._summaries[idx],
                )
            )
        return entries

    def max_similarity_to(self, name: str, others: list[str]) -> float:
        self._require_loaded()
        idx = self._name_to_index.get(name)
        if idx is None:
            return 0.0
        other_indices = [
            self._name_to_index[item] for item in others if item in self._name_to_index
        ]
        if not other_indices:
            return 0.0
        scores = self._matrix[other_indices] @ self._matrix[idx]
        return float(np.max(scores))

    def suggest_similar(
        self, name: str, threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> list[str]:
        self._require_loaded()
        idx = self._name_to_index.get(name)
        if idx is None:
            return []
        row = self._matrix @ self._matrix[idx]
        ranked = np.argsort(row)[::-1]
        suggestions: list[str] = []
        for target_idx in ranked:
            target_idx = int(target_idx)
            if target_idx == idx:
                continue
            score = float(row[target_idx])
            if score < threshold:
                break
            suggestions.append(self._names[target_idx])
        return suggestions

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("VaultEmbeddings.load() must be called before querying embeddings")

    def _resolve_vault_root(
        self,
        *,
        vault_root: Path | None,
        card_index_tsv: Path | None,
    ) -> Path:
        if vault_root is not None:
            return Path(vault_root).expanduser().resolve()
        if card_index_tsv is not None:
            card_index_path = Path(card_index_tsv).expanduser().resolve()
            return card_index_path.parent.parent
        embeddings_path = self.embeddings_path
        if (
            embeddings_path.name == "embeddings.npz"
            and embeddings_path.parent.name == "90_computed"
        ):
            return embeddings_path.parent.parent
        raise ValueError("vault_root is required when embeddings path is not under 90_computed/")
