from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

from alpha_lab.reporting.renderers import write_campaign_report
from alpha_lab.real_cases.composite.pipeline import run_composite_case
from alpha_lab.real_cases.single_factor.pipeline import run_single_factor_case

PackageType = Literal["single_factor", "composite"]
Status = Literal["success", "failed", "skipped"]

_ALLOWED_PACKAGE_TYPES: frozenset[str] = frozenset({"single_factor", "composite"})
_ALLOWED_VAULT_MODES: frozenset[str] = frozenset({"skip", "overwrite", "versioned"})
_REQUIRED_CASE_NAMES: tuple[str, str, str] = (
    "bp_single_factor_v1",
    "roe_ttm_single_factor_v1",
    "value_quality_lowvol_v1",
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CampaignCase:
    case_name: str
    package_type: PackageType
    spec_path: str


@dataclass(frozen=True)
class CampaignConfig:
    campaign_name: str
    campaign_description: str
    output_root_dir: str
    cases: tuple[CampaignCase, ...]
    execution_order: tuple[str, ...]
    case_output_root_dir: str | None = None
    vault_root: str | None = None
    vault_export_mode: str = "versioned"


@dataclass(frozen=True)
class CampaignCaseResult:
    case_name: str
    package_type: PackageType
    status: Status
    output_dir: Path | None
    summary_path: Path | None
    experiment_card_path: Path | None
    run_manifest_path: Path | None
    metrics_path: Path | None
    key_metrics: dict[str, object]
    vault_export: dict[str, object]
    error: str | None = None


@dataclass(frozen=True)
class CampaignRunResult:
    config: CampaignConfig
    output_dir: Path
    case_results: tuple[CampaignCaseResult, ...]
    artifact_paths: dict[str, Path]


def load_research_campaign_1_config(path: str | Path) -> CampaignConfig:
    config_path = Path(path).resolve()
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"campaign config does not exist: {config_path}")

    text = config_path.read_text(encoding="utf-8")
    parsed = _parse_mapping_payload(text, suffix=config_path.suffix.lower())
    config = campaign_config_from_mapping(parsed)
    return resolve_campaign_paths(config, base_dir=config_path.parent)


def campaign_config_from_mapping(data: dict[str, object]) -> CampaignConfig:
    campaign_name = _required_str(data, "campaign_name")
    campaign_description = _required_str(data, "campaign_description")
    output_root_dir = _required_str(data, "output_root_dir")

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or len(raw_cases) == 0:
        raise ValueError("cases must be a non-empty list")

    cases: list[CampaignCase] = []
    for idx, raw in enumerate(raw_cases):
        if not isinstance(raw, dict):
            raise ValueError(f"cases[{idx}] must be an object")
        case_name = _required_str(raw, "case_name")
        raw_package = _required_str(raw, "package_type").lower()
        if raw_package not in _ALLOWED_PACKAGE_TYPES:
            raise ValueError(
                f"cases[{idx}].package_type must be one of {sorted(_ALLOWED_PACKAGE_TYPES)}"
            )
        spec_path = _required_str(raw, "spec_path")
        cases.append(
            CampaignCase(
                case_name=case_name,
                package_type=cast(PackageType, raw_package),
                spec_path=spec_path,
            )
        )

    raw_order = data.get("execution_order")
    if not isinstance(raw_order, list) or len(raw_order) == 0:
        raise ValueError("execution_order must be a non-empty list")
    execution_order: tuple[str, ...] = tuple(_required_list_item_str(raw_order, "execution_order"))

    case_names = [case.case_name for case in cases]
    if len(set(case_names)) != len(case_names):
        raise ValueError("case names in campaign config must be unique")

    if set(execution_order) != set(case_names):
        raise ValueError("execution_order must contain exactly the configured case names")

    if set(case_names) != set(_REQUIRED_CASE_NAMES):
        raise ValueError(
            "research_campaign_1 must include exactly: "
            f"{list(_REQUIRED_CASE_NAMES)}"
        )

    case_output_root_dir = _optional_str(data.get("case_output_root_dir"))

    vault_root: str | None = None
    vault_export_mode = "versioned"
    raw_vault = data.get("vault_export", {})
    if raw_vault is not None:
        if not isinstance(raw_vault, dict):
            raise ValueError("vault_export must be an object when provided")
        vault_root = _optional_str(raw_vault.get("vault_root"))
        raw_mode = _optional_str(raw_vault.get("mode"))
        if raw_mode is not None:
            mode = raw_mode.lower()
            if mode not in _ALLOWED_VAULT_MODES:
                raise ValueError(
                    f"vault_export.mode must be one of {sorted(_ALLOWED_VAULT_MODES)}"
                )
            vault_export_mode = mode

    return CampaignConfig(
        campaign_name=campaign_name,
        campaign_description=campaign_description,
        output_root_dir=output_root_dir,
        cases=tuple(cases),
        execution_order=execution_order,
        case_output_root_dir=case_output_root_dir,
        vault_root=vault_root,
        vault_export_mode=vault_export_mode,
    )


def resolve_campaign_paths(config: CampaignConfig, *, base_dir: Path) -> CampaignConfig:
    def _resolve(path_value: str) -> str:
        path = Path(path_value)
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        else:
            path = path.resolve()
        return str(path)

    cases = tuple(
        CampaignCase(
            case_name=case.case_name,
            package_type=case.package_type,
            spec_path=_resolve(case.spec_path),
        )
        for case in config.cases
    )

    case_output_root = (
        _resolve(config.case_output_root_dir)
        if config.case_output_root_dir is not None
        else None
    )
    vault_root = _resolve(config.vault_root) if config.vault_root is not None else None

    return CampaignConfig(
        campaign_name=config.campaign_name,
        campaign_description=config.campaign_description,
        output_root_dir=_resolve(config.output_root_dir),
        cases=cases,
        execution_order=config.execution_order,
        case_output_root_dir=case_output_root,
        vault_root=vault_root,
        vault_export_mode=config.vault_export_mode,
    )


def run_research_campaign_1(
    config_or_path: CampaignConfig | str | Path,
    *,
    output_root_dir: str | Path | None = None,
    case_output_root_dir: str | Path | None = None,
    vault_root: str | Path | None = None,
    vault_export_mode: str | None = None,
) -> CampaignRunResult:
    config = (
        config_or_path
        if isinstance(config_or_path, CampaignConfig)
        else load_research_campaign_1_config(config_or_path)
    )

    campaign_out = (
        Path(output_root_dir).resolve()
        if output_root_dir is not None
        else Path(config.output_root_dir).resolve()
    )
    campaign_out.mkdir(parents=True, exist_ok=True)

    case_out = (
        Path(case_output_root_dir).resolve()
        if case_output_root_dir is not None
        else (
            Path(config.case_output_root_dir).resolve()
            if config.case_output_root_dir is not None
            else None
        )
    )

    resolved_vault_root = (
        str(Path(vault_root).resolve())
        if vault_root is not None and str(vault_root).strip()
        else config.vault_root
    )

    resolved_mode = (vault_export_mode or config.vault_export_mode).strip().lower()
    if resolved_mode not in _ALLOWED_VAULT_MODES:
        raise ValueError(f"vault_export_mode must be one of {sorted(_ALLOWED_VAULT_MODES)}")

    case_map = {case.case_name: case for case in config.cases}
    ordered_cases = [case_map[name] for name in config.execution_order]

    case_results: list[CampaignCaseResult] = []
    for case in ordered_cases:
        result = _run_case(
            case,
            case_output_root_dir=case_out,
            vault_root=resolved_vault_root,
            vault_export_mode=resolved_mode,
        )
        case_results.append(result)
        _write_case_pointer(campaign_out, result)

    now_utc = datetime.datetime.now(datetime.UTC).isoformat()
    manifest_payload = {
        "schema_version": "1.0.0",
        "campaign_name": config.campaign_name,
        "campaign_description": config.campaign_description,
        "run_timestamp_utc": now_utc,
        "execution_order": list(config.execution_order),
        "output_root_dir": str(campaign_out),
        "case_output_root_dir": str(case_out) if case_out is not None else None,
        "vault_export_defaults": {
            "vault_root": resolved_vault_root,
            "mode": resolved_mode,
        },
        "cases": [asdict(case) for case in ordered_cases],
    }

    results_payload = {
        "campaign_name": config.campaign_name,
        "run_timestamp_utc": now_utc,
        "n_cases": len(case_results),
        "n_success": sum(1 for row in case_results if row.status == "success"),
        "n_failed": sum(1 for row in case_results if row.status == "failed"),
        "n_skipped": sum(1 for row in case_results if row.status == "skipped"),
        "cases": [_case_result_to_dict(row) for row in case_results],
    }

    manifest_path = campaign_out / "campaign_manifest.json"
    results_path = campaign_out / "campaign_results.json"
    summary_path = campaign_out / "campaign_summary.md"
    index_path = campaign_out / "campaign_index.md"

    _write_json(manifest_path, manifest_payload)
    _write_json(results_path, results_payload)
    summary_path.write_text(
        render_campaign_summary(
            config=config,
            case_results=tuple(case_results),
            run_timestamp_utc=now_utc,
        ),
        encoding="utf-8",
    )
    index_path.write_text(
        render_campaign_index(
            config=config,
            case_results=tuple(case_results),
            run_timestamp_utc=now_utc,
        ),
        encoding="utf-8",
    )

    return CampaignRunResult(
        config=config,
        output_dir=campaign_out,
        case_results=tuple(case_results),
        artifact_paths={
            "campaign_manifest": manifest_path,
            "campaign_results": results_path,
            "campaign_summary": summary_path,
            "campaign_index": index_path,
        },
    )


def render_campaign_summary(
    *,
    config: CampaignConfig,
    case_results: tuple[CampaignCaseResult, ...],
    run_timestamp_utc: str,
) -> str:
    lines = [
        f"# Campaign Summary: {config.campaign_name}",
        "",
        "## 1. Campaign Objective",
        "",
        config.campaign_description,
        "",
        "## 2. Cases Included",
        "",
    ]

    for row in case_results:
        lines.append(
            f"- `{row.case_name}` ({row.package_type}) - status: `{row.status}`"
        )

    lines += [
        "",
        "## 3. Method Overview",
        "",
        "- Reused existing real-case single-factor and composite package runners.",
        "- Each case writes standardized artifacts under `outputs/real_cases/<case_name>/`.",
        "- Campaign outputs aggregate per-case manifests, metrics, and vault export status.",
        "",
        "## 4. High-Level Findings By Case",
        "",
    ]

    for row in case_results:
        if row.status != "success":
            lines.append(
                f"- `{row.case_name}`: failed ({row.error or 'no error detail'})"
            )
            continue

        m = row.key_metrics
        lines.append(
            "- "
            f"`{row.case_name}`: "
            f"IC={_fmt(m.get('mean_ic'))}, "
            f"ICIR={_fmt(m.get('ic_ir'))}, "
            f"L/S={_fmt(m.get('mean_long_short_return'))}, "
            f"Turnover={_fmt(m.get('mean_long_short_turnover'))}, "
            f"Coverage={_fmt(m.get('coverage_mean'))}"
        )

    lines += [
        "",
        "## 5. Comparative Observations",
        "",
        (
            "| Case | Factor Type | Direction | IC | ICIR | Long-Short | "
            "Turnover | Coverage | Vault Export |"
        ),
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]

    for row in case_results:
        metrics = row.key_metrics
        lines.append(
            "| "
            f"{row.case_name} | "
            f"{row.package_type} | "
            f"{metrics.get('direction', 'N/A')} | "
            f"{_fmt(metrics.get('mean_ic'))} | "
            f"{_fmt(metrics.get('ic_ir'))} | "
            f"{_fmt(metrics.get('mean_long_short_return'))} | "
            f"{_fmt(metrics.get('mean_long_short_turnover'))} | "
            f"{_fmt(metrics.get('coverage_mean'))} | "
            f"{row.vault_export.get('status', 'N/A')} "
            "|"
        )

    best_icir = _best_case_by_metric(case_results, "ic_ir")
    lines += [
        "",
        (
            "- Best ICIR case (successful runs only): "
            f"`{best_icir}`" if best_icir is not None else
            "- Best ICIR case: N/A (no successful runs with finite ICIR)."
        ),
        "",
        "## 6. Limitations",
        "",
        (
            "- Campaign-level conclusions rely only on generated metrics; "
            "no manual interpretation added."
        ),
        "- Failed/missing cases are marked explicitly and not imputed.",
        "- This v1 campaign runner is intentionally explicit to research_campaign_1 only.",
        "",
        "## 7. Recommended Next Steps",
        "",
        "- Add deeper attribution diagnostics for successful cases.",
        "- Add a follow-up campaign with robustness windows and alternate universes.",
        "- Move accepted findings into quant-knowledge interpretation sections.",
        "",
        f"_Run timestamp (UTC): `{run_timestamp_utc}`_",
        "",
    ]
    return "\n".join(lines)


def render_campaign_index(
    *,
    config: CampaignConfig,
    case_results: tuple[CampaignCaseResult, ...],
    run_timestamp_utc: str,
) -> str:
    lines = [
        f"# Campaign Index: {config.campaign_name}",
        "",
        config.campaign_description,
        "",
        "## Cases",
        "",
    ]

    for row in case_results:
        pointer_rel = f"{row.case_name}/case_output_pointer.json"
        lines.append(
            "- "
            f"`{row.case_name}` ({row.package_type}) - {row.status} - "
            f"[pointer]({pointer_rel})"
        )
        if row.experiment_card_path is not None:
            lines.append(f"  card: `{row.experiment_card_path}`")

    lines += [
        "",
        "## Notes",
        "",
        "- This index is campaign-level and vault-friendly.",
        "- Use each case pointer file to locate canonical run artifacts.",
        "",
        f"_Run timestamp (UTC): `{run_timestamp_utc}`_",
        "",
    ]
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="research-campaign-1",
        description="Run research_campaign_1 (3 standard real-cases) end-to-end.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "campaign_config",
        nargs="?",
        default="configs/campaigns/research_campaign_1/campaign.yaml",
        help="Path to campaign manifest YAML/JSON.",
    )
    parser.add_argument(
        "--output-root-dir",
        default=None,
        help="Override campaign output root directory.",
    )
    parser.add_argument(
        "--case-output-root-dir",
        default=None,
        help="Optional override for per-case output root passed to case runners.",
    )
    parser.add_argument(
        "--vault-root",
        default=None,
        help="Optional vault root override passed to case runners.",
    )
    parser.add_argument(
        "--vault-export-mode",
        choices=["skip", "overwrite", "versioned"],
        default=None,
        help="Optional vault export mode override passed to case runners.",
    )
    parser.add_argument(
        "--render-report",
        action="store_true",
        help="Render campaign_report.md after a successful campaign run.",
    )
    parser.add_argument(
        "--render-overwrite",
        action="store_true",
        help="Overwrite existing campaign_report.md when rendering is enabled.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        result = run_research_campaign_1(
            args.campaign_config,
            output_root_dir=args.output_root_dir,
            case_output_root_dir=args.case_output_root_dir,
            vault_root=args.vault_root,
            vault_export_mode=args.vault_export_mode,
        )
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        parser.error(str(exc))

    render_meta = _render_campaign_report(
        output_dir=result.output_dir,
        enabled=bool(args.render_report),
        overwrite=bool(args.render_overwrite),
    )
    _update_campaign_results(
        result.artifact_paths["campaign_results"],
        render_meta,
    )

    success = sum(1 for row in result.case_results if row.status == "success")
    failed = sum(1 for row in result.case_results if row.status == "failed")

    print("")
    print(f"  Campaign : {result.config.campaign_name}")
    print("  Status   : completed")
    print(f"  Cases    : {len(result.case_results)} total / {success} success / {failed} failed")
    print(f"  Output   : {result.output_dir}")
    print(f"  Manifest : {result.artifact_paths['campaign_manifest']}")
    print(f"  Results  : {result.artifact_paths['campaign_results']}")
    print(f"  Summary  : {result.artifact_paths['campaign_summary']}")
    print(f"  Index    : {result.artifact_paths['campaign_index']}")
    print(f"  Report Render Status: {render_meta.get('render_status')}")
    print(f"  Report Path         : {render_meta.get('rendered_report_path')}")

    return 0


def _render_campaign_report(
    *,
    output_dir: Path,
    enabled: bool,
    overwrite: bool,
) -> dict[str, object]:
    if not enabled:
        return {
            "rendered_report": False,
            "rendered_report_path": None,
            "render_status": "skipped",
            "render_error": None,
        }

    try:
        report_path = write_campaign_report(output_dir, overwrite=overwrite)
        return {
            "rendered_report": True,
            "rendered_report_path": str(report_path),
            "render_status": "success",
            "render_error": None,
        }
    except Exception as exc:
        logger.warning(
            "Campaign report rendering failed for %s: %s",
            output_dir,
            exc,
        )
        return {
            "rendered_report": False,
            "rendered_report_path": None,
            "render_status": "failed",
            "render_error": str(exc),
        }


def _update_campaign_results(
    campaign_results_path: Path,
    render_meta: dict[str, object],
) -> None:
    try:
        payload = _load_json(campaign_results_path)
        payload.update(render_meta)
        _write_json(campaign_results_path, payload)
    except Exception as exc:
        logger.warning(
            "Failed to persist campaign render metadata into %s: %s",
            campaign_results_path,
            exc,
        )


def _run_case(
    case: CampaignCase,
    *,
    case_output_root_dir: Path | None,
    vault_root: str | None,
    vault_export_mode: str,
) -> CampaignCaseResult:
    try:
        if case.package_type == "single_factor":
            run = run_single_factor_case(
                case.spec_path,
                output_root_dir=case_output_root_dir,
                vault_root=vault_root,
                vault_export_mode=vault_export_mode,
            )
        else:
            run = run_composite_case(
                case.spec_path,
                output_root_dir=case_output_root_dir,
                vault_root=vault_root,
                vault_export_mode=vault_export_mode,
            )

        manifest_path = run.artifact_paths["run_manifest"]
        metrics_path = run.artifact_paths["metrics"]
        summary_path = run.artifact_paths["summary"]
        card_path = run.artifact_paths["experiment_card"]

        manifest = _load_json(manifest_path)
        key_metrics = _extract_key_metrics(metrics_path)

        direction = key_metrics.get("direction")
        if direction is None:
            direction = _extract_direction_from_manifest(manifest, package_type=case.package_type)
        if direction is not None:
            key_metrics["direction"] = direction

        vault_export = cast(
            dict[str, object],
            manifest.get(
                "vault_export",
                {"enabled": False, "mode": "skip", "status": "skipped", "error": None},
            ),
        )

        return CampaignCaseResult(
            case_name=case.case_name,
            package_type=case.package_type,
            status="success",
            output_dir=run.output_dir,
            summary_path=summary_path,
            experiment_card_path=card_path,
            run_manifest_path=manifest_path,
            metrics_path=metrics_path,
            key_metrics=key_metrics,
            vault_export=vault_export,
            error=None,
        )
    except Exception as exc:
        return CampaignCaseResult(
            case_name=case.case_name,
            package_type=case.package_type,
            status="failed",
            output_dir=None,
            summary_path=None,
            experiment_card_path=None,
            run_manifest_path=None,
            metrics_path=None,
            key_metrics={},
            vault_export={
                "enabled": bool(vault_root),
                "mode": vault_export_mode,
                "status": "failed",
                "error": str(exc),
            },
            error=str(exc),
        )


def _extract_key_metrics(metrics_path: Path) -> dict[str, object]:
    payload = _load_json(metrics_path)
    raw_metrics = payload.get("metrics", {})
    if not isinstance(raw_metrics, dict):
        return {}

    keys = (
        "direction",
        "mean_ic",
        "mean_rank_ic",
        "ic_ir",
        "mean_long_short_return",
        "mean_long_short_turnover",
        "coverage_mean",
        "missingness_mean",
        "n_dates_used",
    )
    return {key: raw_metrics.get(key) for key in keys if key in raw_metrics}


def _extract_direction_from_manifest(
    manifest: dict[str, object],
    *,
    package_type: PackageType,
) -> str | None:
    spec = manifest.get("spec", {})
    if not isinstance(spec, dict):
        return None

    if package_type == "single_factor":
        value = spec.get("direction")
        return str(value) if value is not None else None

    raw_components = spec.get("components", [])
    if not isinstance(raw_components, list):
        return None

    directions = sorted(
        {
            str(row.get("direction"))
            for row in raw_components
            if isinstance(row, dict) and row.get("direction") is not None
        }
    )
    if not directions:
        return None
    return "/".join(directions)


def _write_case_pointer(campaign_output_dir: Path, result: CampaignCaseResult) -> None:
    case_dir = campaign_output_dir / result.case_name
    case_dir.mkdir(parents=True, exist_ok=True)
    pointer_path = case_dir / "case_output_pointer.json"

    payload = {
        "case_name": result.case_name,
        "package_type": result.package_type,
        "status": result.status,
        "output_dir": str(result.output_dir) if result.output_dir is not None else None,
        "summary_path": (
            str(result.summary_path)
            if result.summary_path is not None
            else None
        ),
        "experiment_card_path": (
            str(result.experiment_card_path)
            if result.experiment_card_path is not None
            else None
        ),
        "run_manifest_path": (
            str(result.run_manifest_path)
            if result.run_manifest_path is not None
            else None
        ),
        "metrics_path": (
            str(result.metrics_path)
            if result.metrics_path is not None
            else None
        ),
        "vault_export": result.vault_export,
        "error": result.error,
    }
    _write_json(pointer_path, payload)


def _case_result_to_dict(result: CampaignCaseResult) -> dict[str, object]:
    return {
        "case_name": result.case_name,
        "package_type": result.package_type,
        "status": result.status,
        "output_dir": str(result.output_dir) if result.output_dir is not None else None,
        "summary_path": (
            str(result.summary_path)
            if result.summary_path is not None
            else None
        ),
        "experiment_card_path": (
            str(result.experiment_card_path)
            if result.experiment_card_path is not None
            else None
        ),
        "run_manifest_path": (
            str(result.run_manifest_path)
            if result.run_manifest_path is not None
            else None
        ),
        "metrics_path": (
            str(result.metrics_path)
            if result.metrics_path is not None
            else None
        ),
        "key_metrics": result.key_metrics,
        "vault_export": result.vault_export,
        "error": result.error,
    }


def _best_case_by_metric(
    case_results: tuple[CampaignCaseResult, ...],
    metric_name: str,
) -> str | None:
    best_case: str | None = None
    best_value: float | None = None
    for row in case_results:
        if row.status != "success":
            continue
        raw = row.key_metrics.get(metric_name)
        if not isinstance(raw, (int, float)):
            continue
        value = float(raw)
        if not math.isfinite(value):
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_case = row.case_name
    return best_case


def _fmt(value: object) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return "N/A"
        return f"{value:.6f}"
    return str(value)


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, object], payload)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _parse_mapping_payload(text: str, *, suffix: str) -> dict[str, object]:
    parsed: object
    if suffix == ".json":
        parsed = json.loads(text)
    else:
        parsed = _yaml_load(text)

    if not isinstance(parsed, dict):
        raise ValueError("campaign config root must be an object")
    return cast(dict[str, object], parsed)


def _yaml_load(text: str) -> object:
    try:
        import yaml
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("PyYAML is required to load campaign YAML configs") from exc

    return yaml.safe_load(text)


def _required_str(data: dict[str, object], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("expected a string or null")
    text = value.strip()
    return text if text else None


def _required_list_item_str(values: list[object], field_name: str) -> list[str]:
    out: list[str] = []
    for idx, value in enumerate(values):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name}[{idx}] must be a non-empty string")
        out.append(value)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
