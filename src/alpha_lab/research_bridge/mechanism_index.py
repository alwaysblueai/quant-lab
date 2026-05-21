from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from alpha_lab.research_bridge import loaders as bridge_loaders
from alpha_lab.research_bridge._llm_usage import read_attr, usage_int
from alpha_lab.research_bridge.embeddings import (
    WORD_RE,
    VaultEmbeddings,
    _hash_index,
    _normalize_text,
    encode_text,
)
from alpha_lab.research_bridge.llm_rerank import (
    DEFAULT_MODEL,
    anthropic_client_kwargs,
    extract_json_object_from_response,
)

LOG = logging.getLogger(__name__)
SCHEMA_VERSION = 1
DEFAULT_DIMENSION = 4096


@dataclass(frozen=True, slots=True)
class CardFingerprint:
    path: str
    name: str
    type: str
    core_mechanism: list[str]
    transferable_principle: str
    applicable_scenarios: list[str]
    similar_problems: list[str]
    failure_conditions: list[str]


@dataclass(frozen=True, slots=True)
class IndexBuildOutcome:
    cards_total: int
    cards_regenerated: int
    cards_cached: int
    cards_failed: list[tuple[str, str]]
    tokens_input: int
    tokens_output: int
    fallback_reason: str | None


def build_mechanism_index(
    *,
    vault_root: Path,
    workspace_root: Path,
    model: str | None = None,
    force: bool = False,
    api_key_env: str = "ANTHROPIC_API_KEY",
) -> IndexBuildOutcome:
    """Build the offline mechanism fingerprint sidecar for a vault."""
    resolved_vault = Path(vault_root).expanduser().resolve()
    resolved_workspace = Path(workspace_root).expanduser().resolve()
    model_name = str(model or DEFAULT_MODEL)
    rows = [
        row
        for row in bridge_loaders.load_card_index_rows(resolved_vault)
        if str(row.get("path") or "").strip()
    ]
    if not os.environ.get(api_key_env):
        return IndexBuildOutcome(
            cards_total=len(rows),
            cards_regenerated=0,
            cards_cached=0,
            cards_failed=[],
            tokens_input=0,
            tokens_output=0,
            fallback_reason="no_api_key",
        )

    index_root = _index_root(
        workspace_root=resolved_workspace,
        vault_root=resolved_vault,
    )
    cards_root = index_root / "cards"
    cards_root.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(index_root)
    previous_hashes = _manifest_card_hashes(manifest)

    try:
        import anthropic  # type: ignore[import-not-found]

        client_kwargs: Any = anthropic_client_kwargs(str(os.environ[api_key_env]))
        client: Any = anthropic.Anthropic(**client_kwargs)
    except Exception as exc:
        reason = f"api_error: {type(exc).__name__}: {exc}"
        LOG.warning("mechanism index client setup failed: %s", reason)
        return IndexBuildOutcome(
            cards_total=len(rows),
            cards_regenerated=0,
            cards_cached=0,
            cards_failed=[],
            tokens_input=0,
            tokens_output=0,
            fallback_reason=reason,
        )

    regenerated = 0
    cached = 0
    failed: list[tuple[str, str]] = []
    tokens_input = 0
    tokens_output = 0
    fingerprints: list[tuple[CardFingerprint, str]] = []
    next_hashes: dict[str, str] = {}

    for row in rows:
        rel_path = str(row.get("path") or "").strip()
        name = str(row.get("name") or "").strip()
        card_type = str(row.get("type") or "").strip()
        body = _read_card_body(resolved_vault, rel_path)
        content_hash = _content_hash(body)
        next_hashes[rel_path] = content_hash
        cached_payload = _load_fingerprint_payload(cards_root, rel_path)
        if (
            not force
            and previous_hashes.get(rel_path) == content_hash
            and cached_payload is not None
            and cached_payload.get("content_hash") == content_hash
        ):
            cached_fp = _fingerprint_from_payload(cached_payload)
            if cached_fp is not None:
                cached += 1
                fingerprints.append((cached_fp, content_hash))
                continue

        try:
            request_payload = {
                "path": rel_path,
                "name": name,
                "type": card_type,
                "body": body[:6000],
            }
            try:
                response = client.messages.create(
                    model=model_name,
                    system=[
                        {
                            "type": "text",
                            "text": _fingerprint_rubric(),
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[
                        {
                            "role": "user",
                            "content": json.dumps(
                                request_payload,
                                ensure_ascii=False,
                            ),
                        }
                    ],
                    max_tokens=1024,
                    temperature=0,
                    tools=[_submit_fingerprint_tool_schema()],
                    tool_choice={"type": "tool", "name": "submit_card_fingerprint"},
                )
                tool_input = _extract_tool_input(
                    response,
                    tool_name="submit_card_fingerprint",
                )
            except Exception as tool_exc:
                LOG.info(
                    "mechanism fingerprint tool call failed; trying JSON text mode: %s",
                    tool_exc,
                )
                response = client.messages.create(
                    model=model_name,
                    system=_json_text_system(_fingerprint_rubric()),
                    messages=[
                        {
                            "role": "user",
                            "content": json.dumps(
                                request_payload,
                                ensure_ascii=False,
                            ),
                        }
                    ],
                    max_tokens=1024,
                    temperature=0,
                )
                tool_input = extract_json_object_from_response(response)
            payload = {
                "path": rel_path,
                "name": name,
                "type": card_type,
                "content_hash": content_hash,
                "core_mechanism": _clean_str_list(tool_input.get("core_mechanism")),
                "transferable_principle": str(
                    tool_input.get("transferable_principle") or ""
                ).strip(),
                "applicable_scenarios": _clean_str_list(
                    tool_input.get("applicable_scenarios")
                ),
                "similar_problems": _clean_str_list(tool_input.get("similar_problems")),
                "failure_conditions": _clean_str_list(
                    tool_input.get("failure_conditions")
                ),
                "model": model_name,
                "generated_at": _utc_now_iso(),
            }
            fingerprint = _fingerprint_from_payload(payload)
            if fingerprint is None:
                failed.append((rel_path, "invalid_fingerprint"))
                continue
            _write_fingerprint_payload(cards_root, rel_path, payload)
            usage = read_attr(response, "usage", {})
            tokens_input += (
                usage_int(usage, "input_tokens")
                + usage_int(usage, "cache_creation_input_tokens")
                + usage_int(usage, "cache_read_input_tokens")
            )
            tokens_output += usage_int(usage, "output_tokens")
            regenerated += 1
            fingerprints.append((fingerprint, content_hash))
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            LOG.warning("mechanism fingerprint failed for %s: %s", rel_path, reason)
            failed.append((rel_path, reason))

    _write_manifest(
        index_root=index_root,
        vault_root=resolved_vault,
        model=model_name,
        card_hashes=next_hashes,
    )
    _write_mechanism_embeddings(
        embeddings_path=index_root / "mechanism_embeddings.npz",
        fingerprints=[item[0] for item in fingerprints],
    )
    return IndexBuildOutcome(
        cards_total=len(rows),
        cards_regenerated=regenerated,
        cards_cached=cached,
        cards_failed=failed,
        tokens_input=tokens_input,
        tokens_output=tokens_output,
        fallback_reason=None,
    )


def load_mechanism_embeddings(
    *,
    workspace_root: Path,
    vault_root: Path,
) -> VaultEmbeddings | None:
    index_root = _index_root(workspace_root=workspace_root, vault_root=vault_root)
    manifest = _load_manifest(index_root)
    if not _manifest_matches_vault(manifest, vault_root):
        return None
    embeddings_path = index_root / "mechanism_embeddings.npz"
    if not embeddings_path.exists():
        return None
    embeddings = VaultEmbeddings(embeddings_path)
    try:
        embeddings.load()
    except (OSError, ValueError):
        return None
    return embeddings


def load_card_fingerprint(
    *,
    workspace_root: Path,
    vault_root: Path,
    path: str,
) -> CardFingerprint | None:
    index_root = _index_root(workspace_root=workspace_root, vault_root=vault_root)
    manifest = _load_manifest(index_root)
    if not _manifest_matches_vault(manifest, vault_root):
        return None
    payload = _load_fingerprint_payload(index_root / "cards", path)
    return _fingerprint_from_payload(payload or {})


def mechanism_index_status(
    *,
    workspace_root: Path,
    vault_root: Path,
) -> dict[str, object]:
    """Return a cheap UI-facing status snapshot for the mechanism sidecar."""
    index_root = _index_root(workspace_root=workspace_root, vault_root=vault_root)
    manifest_path = index_root / "manifest.json"
    embeddings_path = index_root / "mechanism_embeddings.npz"
    manifest = _load_manifest(index_root)
    vault_hash = _vault_hash(Path(vault_root).expanduser().resolve())
    base: dict[str, object] = {
        "ok": True,
        "ready": False,
        "status": "not_built",
        "card_count": 0,
        "generated_at": "",
        "vault_hash": vault_hash,
        "manifest_vault_root": "",
        "manifest_path": str(manifest_path),
        "embeddings_path": str(embeddings_path),
    }
    if not manifest:
        return base

    card_hashes = _manifest_card_hashes(manifest)
    base.update(
        {
            "card_count": len(card_hashes),
            "generated_at": str(manifest.get("generated_at") or ""),
            "vault_hash": str(manifest.get("vault_hash") or vault_hash),
            "manifest_vault_root": str(manifest.get("vault_root") or ""),
        }
    )
    if not _manifest_matches_vault(manifest, vault_root):
        return {**base, "status": "vault_mismatch"}
    if not embeddings_path.exists():
        return {**base, "status": "missing_embeddings"}
    try:
        with np.load(embeddings_path) as payload:
            required = {"names", "paths", "matrix", "idf", "dimension"}
            if not required.issubset(set(payload.files)):
                return {**base, "status": "invalid_embeddings"}
    except (OSError, ValueError):
        return {**base, "status": "invalid_embeddings"}
    return {**base, "ready": True, "status": "ready"}


def _vault_hash(vault_root: Path) -> str:
    canonical = str(Path(vault_root).expanduser().resolve())
    digest = hashlib.blake2b(canonical.encode("utf-8"), digest_size=8).hexdigest()
    return digest[:16]


def _path_slug(path: str) -> str:
    encoded = base64.urlsafe_b64encode(path.encode("utf-8")).decode("ascii")
    return encoded.rstrip("=") or "empty"


def _index_root(*, workspace_root: Path, vault_root: Path) -> Path:
    return (
        Path(workspace_root).expanduser().resolve()
        / ".research_bridge_cache"
        / "mechanism_index"
        / _vault_hash(Path(vault_root).expanduser().resolve())
    )


def _load_manifest(index_root: Path) -> dict[str, Any]:
    path = index_root / "manifest.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return raw if isinstance(raw, dict) else {}


def _write_manifest(
    *,
    index_root: Path,
    vault_root: Path,
    model: str,
    card_hashes: dict[str, str],
) -> None:
    index_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model,
        "schema_version": SCHEMA_VERSION,
        "vault_root": str(Path(vault_root).expanduser().resolve()),
        "vault_hash": _vault_hash(Path(vault_root).expanduser().resolve()),
        "generated_at": _utc_now_iso(),
        "card_hashes": dict(sorted(card_hashes.items())),
    }
    (index_root / "manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _manifest_card_hashes(manifest: dict[str, Any]) -> dict[str, str]:
    raw = manifest.get("card_hashes", {})
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items()}


def _manifest_matches_vault(manifest: dict[str, Any], vault_root: Path) -> bool:
    manifest_root = str(manifest.get("vault_root") or "").strip()
    if not manifest_root:
        return False
    try:
        return Path(manifest_root).expanduser().resolve() == Path(
            vault_root
        ).expanduser().resolve()
    except OSError:
        return False


def _read_card_body(vault_root: Path, rel_path: str) -> str:
    normalized = str(rel_path or "").strip()
    if not normalized:
        return ""
    root = Path(vault_root).expanduser().resolve()
    card_path = (root / normalized).resolve()
    if not str(card_path).startswith(str(root)):
        return ""
    if not card_path.exists() or not card_path.is_file():
        return ""
    try:
        return card_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _content_hash(text: str) -> str:
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
    return f"blake2b:{digest}"


def _load_fingerprint_payload(cards_root: Path, rel_path: str) -> dict[str, Any] | None:
    path = cards_root / f"{_path_slug(rel_path)}.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _write_fingerprint_payload(
    cards_root: Path,
    rel_path: str,
    payload: dict[str, Any],
) -> None:
    cards_root.mkdir(parents=True, exist_ok=True)
    path = cards_root / f"{_path_slug(rel_path)}.json"
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _fingerprint_from_payload(payload: dict[str, Any]) -> CardFingerprint | None:
    rel_path = str(payload.get("path") or "").strip()
    name = str(payload.get("name") or "").strip()
    if not rel_path or not name:
        return None
    return CardFingerprint(
        path=rel_path,
        name=name,
        type=str(payload.get("type") or "").strip(),
        core_mechanism=_clean_str_list(payload.get("core_mechanism")),
        transferable_principle=str(
            payload.get("transferable_principle") or ""
        ).strip(),
        applicable_scenarios=_clean_str_list(payload.get("applicable_scenarios")),
        similar_problems=_clean_str_list(payload.get("similar_problems")),
        failure_conditions=_clean_str_list(payload.get("failure_conditions")),
    )


def _fingerprint_text(fingerprint: CardFingerprint) -> str:
    parts = [
        *fingerprint.core_mechanism,
        fingerprint.transferable_principle,
        *fingerprint.similar_problems,
        *fingerprint.applicable_scenarios,
    ]
    return " ".join(part for part in parts if part)


def _write_mechanism_embeddings(
    *,
    embeddings_path: Path,
    fingerprints: list[CardFingerprint],
) -> None:
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    names = [item.name for item in fingerprints]
    types = [item.type for item in fingerprints]
    paths = [item.path for item in fingerprints]
    summaries = [item.transferable_principle for item in fingerprints]
    texts = [_fingerprint_text(item) for item in fingerprints]
    idf = _build_idf(texts, DEFAULT_DIMENSION)
    matrix = np.vstack(
        [encode_text(text, dimension=DEFAULT_DIMENSION, idf=idf) for text in texts]
    ).astype(np.float32) if texts else np.zeros((0, DEFAULT_DIMENSION), dtype=np.float32)
    np.savez_compressed(
        embeddings_path,
        names=np.array(names),
        types=np.array(types),
        paths=np.array(paths),
        summaries=np.array(summaries),
        matrix=matrix,
        idf=idf.astype(np.float32),
        dimension=np.array([DEFAULT_DIMENSION], dtype=np.int32),
    )


def _build_idf(texts: list[str], dimension: int) -> np.ndarray:
    df = np.zeros(dimension, dtype=np.float32)
    for text in texts:
        for idx in _feature_indices(text, dimension):
            df[idx] += 1.0
    doc_count = float(len(texts))
    return np.asarray(
        np.log((1.0 + doc_count) / (1.0 + df)) + 1.0,
        dtype=np.float32,
    )


def _feature_indices(text: str, dimension: int) -> set[int]:
    normalized = _normalize_text(text)
    indices: set[int] = set()
    for token in WORD_RE.findall(normalized):
        indices.add(_hash_index(f"w:{token}", dimension))
    compact = normalized.replace(" ", "")
    for gram_size in (3, 4):
        if len(compact) < gram_size:
            continue
        for idx_offset in range(len(compact) - gram_size + 1):
            gram = compact[idx_offset : idx_offset + gram_size]
            indices.add(_hash_index(f"c:{gram}", dimension))
    return indices


def _fingerprint_rubric() -> str:
    return "\n".join(
        [
            "You are a quant knowledge engineer.",
            "Extract a mechanism fingerprint from one research card.",
            "Focus only on causal mechanism, transferable principle, scenarios, "
            "similar cross-domain problems, and failure conditions.",
            "Do not restate the card title.",
            "core_mechanism should contain 3-6 abstract phrases.",
            "transferable_principle should be one sentence.",
            "similar_problems must include cross-domain analogies, not only synonyms.",
        ]
    )


def _json_text_system(rubric: str) -> str:
    return "\n".join(
        [
            rubric,
            "The API gateway may not support tool calls.",
            "Return raw valid JSON only. Do not wrap it in markdown fences.",
            (
                'Return exactly {"core_mechanism":[],"transferable_principle":"",'
                '"applicable_scenarios":[],"similar_problems":[],'
                '"failure_conditions":[]}.'
            ),
        ]
    )


def _submit_fingerprint_tool_schema() -> dict[str, Any]:
    array_schema = {
        "type": "array",
        "items": {"type": "string", "maxLength": 80},
        "minItems": 0,
        "maxItems": 8,
    }
    return {
        "name": "submit_card_fingerprint",
        "description": "Submit the mechanism fingerprint for one card.",
        "input_schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "core_mechanism": array_schema,
                "transferable_principle": {"type": "string", "maxLength": 240},
                "applicable_scenarios": array_schema,
                "similar_problems": array_schema,
                "failure_conditions": array_schema,
            },
            "required": [
                "core_mechanism",
                "transferable_principle",
                "applicable_scenarios",
                "similar_problems",
                "failure_conditions",
            ],
        },
    }


def _extract_tool_input(response: object, *, tool_name: str) -> dict[str, Any]:
    content = read_attr(response, "content", [])
    if not isinstance(content, list):
        raise ValueError("response content is not a list")
    for block in content:
        block_type = read_attr(block, "type", "")
        block_name = read_attr(block, "name", "")
        if block_type == "tool_use" and block_name == tool_name:
            raw_input = read_attr(block, "input", {})
            if isinstance(raw_input, dict):
                return raw_input
            raise ValueError("tool input is not a dict")
    raise ValueError(f"{tool_name} tool call not found")


def _clean_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    rows: list[str] = []
    seen: set[str] = set()
    for item in value:
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        rows.append(text[:120])
    return rows


def _utc_now_iso() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build research_bridge mechanism index")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--vault", required=True, type=Path)
    build_parser.add_argument("--workspace-root", default=Path.cwd(), type=Path)
    build_parser.add_argument("--force", action="store_true")
    build_parser.add_argument("--model", default=None)
    args = parser.parse_args(argv)

    if args.command == "build":
        outcome = build_mechanism_index(
            vault_root=args.vault,
            workspace_root=args.workspace_root,
            model=args.model,
            force=bool(args.force),
        )
        print(json.dumps(_outcome_to_payload(outcome), ensure_ascii=False, indent=2))
        return 0 if outcome.fallback_reason is None else 1
    return 1


def _outcome_to_payload(outcome: IndexBuildOutcome) -> dict[str, object]:
    return {
        "cards_total": outcome.cards_total,
        "cards_regenerated": outcome.cards_regenerated,
        "cards_cached": outcome.cards_cached,
        "cards_failed": list(outcome.cards_failed),
        "tokens_input": outcome.tokens_input,
        "tokens_output": outcome.tokens_output,
        "fallback_reason": outcome.fallback_reason,
    }


if __name__ == "__main__":
    raise SystemExit(main())
