from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from alpha_lab.exceptions import AlphaLabConfigError, AlphaLabExperimentError


def _require_yaml() -> Any:
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - dependency is project-wide.
        raise AlphaLabExperimentError(
            "PyYAML is required to use research bridge project files"
        ) from exc
    return yaml


@dataclass(slots=True)
class AlphaLabDefaults:
    data_source: str = "tushare"
    slice_preset: str = "standard"
    universe: str = "listed_90d"
    adjustment: str = "qfq"
    evaluation_profile: str = "exploratory_screening"


@dataclass(slots=True)
class WritebackPolicy:
    manual_review_required: bool = True
    formal_experiment_dir: str = "50_experiments"


@dataclass(slots=True)
class ProjectStatus:
    lifecycle: str = "active"
    current_hypothesis: str = "待定义"
    current_focus: str = "待开始第一轮讨论"
    next_action: str = "刷新项目包并启动第一轮讨论"
    current_case: str = ""
    latest_run: str = ""
    last_verdict: str = ""


CURRENT_SCHEMA_VERSION = 2


@dataclass(slots=True)
class ProjectConfig:
    slug: str
    title_zh: str
    category: str
    owner: str
    market: str
    frequency: str
    chatgpt_project_name: str
    schema_version: int = CURRENT_SCHEMA_VERSION
    max_research_level: int = 2
    source_priority: list[str] = field(
        default_factory=lambda: [
            "quant_knowledge",
            "alpha_lab_artifacts",
            "web_search",
        ]
    )
    origin_cards: list[str] = field(default_factory=list)
    supporting_cards: list[str] = field(default_factory=list)
    failure_cards: list[str] = field(default_factory=list)
    related_experiment_cards: list[str] = field(default_factory=list)
    preferred_web_sources: list[str] = field(default_factory=list)
    alpha_lab_defaults: AlphaLabDefaults = field(default_factory=AlphaLabDefaults)
    writeback_policy: WritebackPolicy = field(default_factory=WritebackPolicy)
    status: ProjectStatus = field(default_factory=ProjectStatus)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ProjectConfig:
        source_priority = _string_list(payload.get("source_priority")) or [
            "quant_knowledge",
            "alpha_lab_artifacts",
            "web_search",
        ]
        raw_schema_version = payload.get("schema_version")
        schema_version = (
            int(raw_schema_version)
            if raw_schema_version is not None
            else 1  # v1 projects predate this field
        )
        raw_max_level = payload.get("max_research_level")
        max_research_level = int(raw_max_level) if raw_max_level is not None else 2
        return cls(
            slug=str(payload.get("slug") or "").strip(),
            title_zh=str(payload.get("title_zh") or "").strip(),
            category=str(payload.get("category") or "").strip(),
            owner=str(payload.get("owner") or "").strip(),
            market=str(payload.get("market") or "").strip(),
            frequency=str(payload.get("frequency") or "").strip(),
            chatgpt_project_name=str(payload.get("chatgpt_project_name") or "").strip(),
            schema_version=schema_version,
            max_research_level=max_research_level,
            source_priority=source_priority,
            origin_cards=_string_list(payload.get("origin_cards")),
            supporting_cards=_string_list(payload.get("supporting_cards")),
            failure_cards=_string_list(payload.get("failure_cards")),
            related_experiment_cards=_string_list(payload.get("related_experiment_cards")),
            preferred_web_sources=_string_list(payload.get("preferred_web_sources")),
            alpha_lab_defaults=AlphaLabDefaults(
                **_mapping(payload.get("alpha_lab_defaults")),
            ),
            writeback_policy=WritebackPolicy(
                **_mapping(payload.get("writeback_policy")),
            ),
            status=ProjectStatus(
                **_mapping(payload.get("status")),
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def load_yaml_document(path: str | Path) -> dict[str, Any]:
    yaml = _require_yaml()
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8")
    payload: Any
    if resolved.suffix.lower() == ".md":
        payload = yaml.safe_load(_extract_yaml_code_block(text, path=resolved))
    else:
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise AlphaLabConfigError(f"YAML document root must be an object: {resolved}")
    return payload


def save_yaml_document(
    payload: dict[str, Any], path: str | Path, *, title: str | None = None
) -> Path:
    yaml = _require_yaml()
    resolved = Path(path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).strip()
    if resolved.suffix.lower() == ".md":
        heading = f"# {title}\n\n" if title else ""
        content = f"{heading}```yaml\n{rendered}\n```\n"
    else:
        content = f"{rendered}\n"
    resolved.write_text(content, encoding="utf-8")
    return resolved


def _extract_yaml_code_block(text: str, *, path: Path) -> str:
    marker = "```yaml"
    start = text.find(marker)
    if start < 0:
        raise AlphaLabConfigError(f"markdown YAML document is missing ```yaml block: {path}")
    start += len(marker)
    end = text.find("```", start)
    if end < 0:
        raise AlphaLabConfigError(f"markdown YAML document has unterminated code block: {path}")
    return text[start:end].strip()


def load_project_config(path: str | Path) -> ProjectConfig:
    payload = load_yaml_document(path)
    project = ProjectConfig.from_dict(payload)
    if not project.slug:
        raise AlphaLabConfigError(f"project.yaml missing slug: {path}")
    if project.schema_version > CURRENT_SCHEMA_VERSION:
        raise AlphaLabConfigError(
            f"project.yaml schema_version {project.schema_version} is newer than "
            f"supported version {CURRENT_SCHEMA_VERSION}. "
            f"Upgrade alpha-lab or run 'alpha-lab bridge migrate-project'."
        )
    return project


def save_project_config(project: ProjectConfig, path: str | Path) -> Path:
    return save_yaml_document(project.to_dict(), path, title=f"Project - {project.slug}")
