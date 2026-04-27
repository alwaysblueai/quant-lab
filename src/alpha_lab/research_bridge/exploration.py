from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class FrontierEntry:
    direction: str
    factor_family: str
    mechanism: str
    reason: str
    suggested_by: str
    priority: str


@dataclass(frozen=True, slots=True)
class ExploredRegion:
    hypothesis: str
    mechanism: str
    factor_family: str
    data_types: list[str]
    projects: list[str]
    experiments: list[str]
    best_verdict: str
    exhaustion_level: str
    last_explored: str


@dataclass(frozen=True, slots=True)
class FailureKnowledgeRef:
    failure_id: str
    title: str
    status: str
    failure_class: str
    failure_statement: str
    prevention_rule: str


class ExplorationMap:
    def __init__(self, exploration_map_path: Path):
        self.exploration_map_path = Path(exploration_map_path).expanduser().resolve()
        self._frontier: list[FrontierEntry] = []
        self._explored_regions: list[ExploredRegion] = []
        self._failure_refs: list[FailureKnowledgeRef] = []
        self._meta: dict[str, Any] = {}
        self._loaded = False

    @classmethod
    def from_vault_root(cls, vault_root: Path) -> ExplorationMap:
        return cls(Path(vault_root).expanduser().resolve() / "90_computed" / "exploration_map.json")

    def build(self, vault_root: Path | None = None) -> None:
        if not self.exploration_map_path.exists():
            self.rebuild_cache(vault_root=vault_root)
        self.load()

    def rebuild_cache(self, vault_root: Path | None = None) -> Path:
        resolved_vault = self._resolve_vault_root(vault_root=vault_root)
        script_path = resolved_vault / "00_protocols" / "rebuild-exploration-map.py"
        if not script_path.exists():
            raise FileNotFoundError(f"rebuild-exploration-map.py not found: {script_path}")
        proc = subprocess.run(
            [sys.executable, str(script_path), str(resolved_vault)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            detail = (
                proc.stderr or proc.stdout or "unknown rebuild-exploration-map failure"
            ).strip()
            raise RuntimeError(f"failed to rebuild exploration map: {detail}")
        return self.exploration_map_path

    def load(self) -> None:
        payload = json.loads(self.exploration_map_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"exploration_map root must be an object: {self.exploration_map_path}")
        self._frontier = [
            FrontierEntry(
                direction=str(item.get("direction") or "").strip(),
                factor_family=str(item.get("factor_family") or "").strip(),
                mechanism=str(item.get("mechanism") or "").strip(),
                reason=str(item.get("reason") or "").strip(),
                suggested_by=str(item.get("suggested_by") or "").strip(),
                priority=str(item.get("priority") or "").strip(),
            )
            for item in payload.get("frontier", [])
            if isinstance(item, dict)
        ]
        self._explored_regions = [
            ExploredRegion(
                hypothesis=str(item.get("hypothesis") or "").strip(),
                mechanism=str(item.get("mechanism") or "").strip(),
                factor_family=str(item.get("factor_family") or "").strip(),
                data_types=[str(v).strip() for v in item.get("data_types", []) if str(v).strip()],
                projects=[str(v).strip() for v in item.get("projects", []) if str(v).strip()],
                experiments=[str(v).strip() for v in item.get("experiments", []) if str(v).strip()],
                best_verdict=str(item.get("best_verdict") or "").strip(),
                exhaustion_level=str(item.get("exhaustion_level") or "").strip(),
                last_explored=str(item.get("last_explored") or "").strip(),
            )
            for item in payload.get("explored_regions", [])
            if isinstance(item, dict)
        ]
        self._failure_refs = [
            FailureKnowledgeRef(
                failure_id=str(item.get("failure_id") or "").strip(),
                title=str(item.get("title") or "").strip(),
                status=str(item.get("status") or "").strip(),
                failure_class=str(item.get("failure_class") or "").strip(),
                failure_statement=str(item.get("failure_statement") or "").strip(),
                prevention_rule=str(item.get("prevention_rule") or "").strip(),
            )
            for item in payload.get("failure_registry_refs", [])
            if isinstance(item, dict)
        ]
        self._meta = dict(payload.get("meta", {})) if isinstance(payload.get("meta"), dict) else {}
        self._loaded = True

    def frontier(
        self,
        *,
        priority: str | None = None,
        factor_family: str | None = None,
        mechanism: str | None = None,
    ) -> list[FrontierEntry]:
        self._require_loaded()
        items = self._frontier
        if priority:
            items = [item for item in items if item.priority == priority]
        if factor_family:
            items = [item for item in items if item.factor_family == factor_family]
        if mechanism:
            items = [item for item in items if item.mechanism == mechanism]
        return list(items)

    def explored_regions(self) -> list[ExploredRegion]:
        self._require_loaded()
        return list(self._explored_regions)

    def related_failures(
        self,
        *,
        factor_family: str = "",
        mechanism: str = "",
        text_query: str = "",
        max_items: int = 5,
    ) -> list[FailureKnowledgeRef]:
        self._require_loaded()
        keywords = {
            part
            for part in [
                factor_family.strip().lower(),
                mechanism.strip().lower(),
                text_query.strip().lower(),
            ]
            if part
        }
        if not keywords:
            return self._failure_refs[:max_items]
        matches: list[tuple[int, FailureKnowledgeRef]] = []
        for item in self._failure_refs:
            haystack = " ".join(
                [
                    item.title.lower(),
                    item.failure_class.lower(),
                    item.failure_statement.lower(),
                    item.prevention_rule.lower(),
                ]
            )
            score = sum(1 for keyword in keywords if keyword in haystack)
            if score > 0:
                matches.append((score, item))
        matches.sort(key=lambda pair: (-pair[0], pair[1].failure_id))
        return [item for _, item in matches[:max_items]]

    def record_exploration(
        self,
        *,
        hypothesis: str,
        mechanism: str,
        factor_family: str,
        data_types: list[str],
        project_slug: str,
        experiment_name: str,
        best_verdict: str,
        exhaustion_level: str,
        last_explored: str,
    ) -> None:
        self._require_loaded()
        regions = list(self._explored_regions)
        for idx, region in enumerate(regions):
            if (
                region.hypothesis == hypothesis
                and region.mechanism == mechanism
                and region.factor_family == factor_family
            ):
                merged_projects = sorted(set(region.projects + [project_slug]))
                merged_experiments = sorted(set(region.experiments + [experiment_name]))
                merged_data = sorted(set(region.data_types + data_types))
                regions[idx] = ExploredRegion(
                    hypothesis=hypothesis,
                    mechanism=mechanism,
                    factor_family=factor_family,
                    data_types=merged_data,
                    projects=merged_projects,
                    experiments=merged_experiments,
                    best_verdict=best_verdict or region.best_verdict,
                    exhaustion_level=exhaustion_level or region.exhaustion_level,
                    last_explored=last_explored or region.last_explored,
                )
                self._explored_regions = regions
                self.save()
                return
        regions.append(
            ExploredRegion(
                hypothesis=hypothesis,
                mechanism=mechanism,
                factor_family=factor_family,
                data_types=sorted(set(data_types)),
                projects=[project_slug] if project_slug else [],
                experiments=[experiment_name] if experiment_name else [],
                best_verdict=best_verdict,
                exhaustion_level=exhaustion_level,
                last_explored=last_explored,
            )
        )
        self._explored_regions = regions
        self.save()

    def save(self) -> None:
        self._require_loaded()
        payload = {
            "meta": self._meta,
            "explored_regions": [
                {
                    "hypothesis": item.hypothesis,
                    "mechanism": item.mechanism,
                    "factor_family": item.factor_family,
                    "data_types": item.data_types,
                    "projects": item.projects,
                    "experiments": item.experiments,
                    "best_verdict": item.best_verdict,
                    "exhaustion_level": item.exhaustion_level,
                    "last_explored": item.last_explored,
                }
                for item in self._explored_regions
            ],
            "frontier": [
                {
                    "direction": item.direction,
                    "factor_family": item.factor_family,
                    "mechanism": item.mechanism,
                    "reason": item.reason,
                    "suggested_by": item.suggested_by,
                    "priority": item.priority,
                }
                for item in self._frontier
            ],
            "failure_registry_refs": [
                {
                    "failure_id": item.failure_id,
                    "title": item.title,
                    "status": item.status,
                    "failure_class": item.failure_class,
                    "failure_statement": item.failure_statement,
                    "prevention_rule": item.prevention_rule,
                }
                for item in self._failure_refs
            ],
        }
        self.exploration_map_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _require_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "ExplorationMap.load() must be called before querying exploration state"
            )

    def _resolve_vault_root(self, *, vault_root: Path | None) -> Path:
        if vault_root is not None:
            return Path(vault_root).expanduser().resolve()
        path = self.exploration_map_path
        if path.name == "exploration_map.json" and path.parent.name == "90_computed":
            return path.parent.parent
        raise ValueError("vault_root is required when path is not under 90_computed/")
