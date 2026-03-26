from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import cast

from .templates import (
    CAMPAIGN_SECTION_TITLES,
    COMPARISON_TABLE_COLUMNS,
    PLACEHOLDER_NEXT_STEPS,
    PLACEHOLDER_OBJECTIVE,
    format_metric,
    format_text,
    markdown_table,
    section_lines,
)


def render_campaign_report(campaign_output_dir: str | Path) -> str:
    """Render a standardized campaign-level markdown report."""

    campaign_dir = Path(campaign_output_dir).resolve()
    manifest = _load_optional_json(campaign_dir / "campaign_manifest.json")
    results = _load_required_json(campaign_dir / "campaign_results.json")

    campaign_name = format_text(
        (manifest or {}).get("campaign_name", results.get("campaign_name")),
        na=campaign_dir.name,
    )
    objective = format_text((manifest or {}).get("campaign_description"), na=PLACEHOLDER_OBJECTIVE)

    cases = _as_list(results.get("cases"))
    rows = [_case_row(raw, campaign_dir=campaign_dir) for raw in cases]

    lines: list[str] = [f"# Campaign Report: {campaign_name}", ""]

    lines.extend(
        section_lines(
            CAMPAIGN_SECTION_TITLES[0],
            [f"- Campaign name: `{campaign_name}`"],
        )
    )
    lines.extend(section_lines(CAMPAIGN_SECTION_TITLES[1], [objective]))

    included = [
        f"- `{row['case_name']}` ({row['package_type']}) - status: `{row['status']}`"
        for row in rows
    ]
    lines.extend(
        section_lines(
            CAMPAIGN_SECTION_TITLES[2],
            included if included else ["- N/A"],
        )
    )

    table_rows = [
        (
            row["case_name"],
            _display_case_type(row["package_type"]),
            _metric_pair(row["metrics"].get("mean_ic"), row["metrics"].get("ic_ir")),
            format_metric(row["metrics"].get("mean_long_short_return")),
            format_metric(row["metrics"].get("mean_long_short_turnover")),
            format_metric(row["metrics"].get("coverage_mean")),
            row["status"],
        )
        for row in rows
    ]
    lines.extend(
        section_lines(
            CAMPAIGN_SECTION_TITLES[3],
            markdown_table(COMPARISON_TABLE_COLUMNS, table_rows),
        )
    )

    highlight_lines = [_highlight_line(row) for row in rows]
    lines.extend(section_lines(CAMPAIGN_SECTION_TITLES[4], highlight_lines or ["- N/A"]))

    lines.extend(section_lines(CAMPAIGN_SECTION_TITLES[5], _cross_case_insights(rows)))

    failed = [row for row in rows if row["status"] != "success"]
    if failed:
        failure_lines = [
            f"- `{row['case_name']}` ({row['status']}): {format_text(row['error'])}"
            for row in failed
        ]
    else:
        failure_lines = ["- No failed cases recorded in campaign_results.json."]
    lines.extend(section_lines(CAMPAIGN_SECTION_TITLES[6], failure_lines))

    n_success = sum(1 for row in rows if row["status"] == "success")
    conclusions = [
        f"- Total cases: {len(rows)}",
        f"- Successful cases: {n_success}",
        f"- Failed/skipped cases: {len(rows) - n_success}",
    ]
    best_case = _best_case(rows, metric="ic_ir")
    if best_case is not None:
        conclusions.append(
            "- Best ICIR among successful cases: "
            f"`{best_case}`"
        )
    lines.extend(section_lines(CAMPAIGN_SECTION_TITLES[7], conclusions))

    lines.extend(
        section_lines(
            CAMPAIGN_SECTION_TITLES[8],
            [PLACEHOLDER_NEXT_STEPS],
        )
    )

    return "\n".join(lines).rstrip() + "\n"


def write_campaign_report(
    campaign_output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Render and write ``campaign_report.md`` in the campaign output directory."""

    campaign_dir = Path(campaign_output_dir).resolve()
    campaign_dir.mkdir(parents=True, exist_ok=True)
    report_path = campaign_dir / "campaign_report.md"
    if report_path.exists() and not overwrite:
        raise FileExistsError(
            f"{report_path} already exists. Pass overwrite=True to replace it."
        )
    report_path.write_text(render_campaign_report(campaign_dir), encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="campaign-report-renderer",
        description="Render campaign_report.md from campaign artifacts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("campaign_output_dir", help="Campaign output directory.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing campaign_report.md when present.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report_path = write_campaign_report(args.campaign_output_dir, overwrite=args.overwrite)
    except (FileNotFoundError, ValueError, KeyError, FileExistsError) as exc:
        parser.error(str(exc))

    print(f"Campaign report written: {report_path}")
    return 0


def _case_row(raw: object, *, campaign_dir: Path) -> dict[str, object]:
    payload = _as_dict(raw)
    case_name = format_text(payload.get("case_name"))
    package_type = format_text(payload.get("package_type"))
    status = format_text(payload.get("status"))
    error = payload.get("error")

    metrics = _as_dict(payload.get("key_metrics")).copy()
    if not metrics:
        loaded = _load_case_metrics(payload, campaign_dir=campaign_dir)
        metrics.update(loaded)

    return {
        "case_name": case_name,
        "package_type": package_type,
        "status": status,
        "metrics": metrics,
        "error": error,
    }


def _load_case_metrics(payload: dict[str, object], *, campaign_dir: Path) -> dict[str, object]:
    metrics_path = payload.get("metrics_path")
    output_dir = payload.get("output_dir")

    candidates: list[Path] = []
    if isinstance(metrics_path, str) and metrics_path.strip():
        candidates.append(_resolve_path(metrics_path, base=campaign_dir))
    if isinstance(output_dir, str) and output_dir.strip():
        candidates.append(_resolve_path(output_dir, base=campaign_dir) / "metrics.json")

    for path in candidates:
        loaded = _load_optional_json(path)
        if loaded is None:
            continue
        metrics = _as_dict(loaded.get("metrics"))
        if metrics:
            return metrics
    return {}


def _cross_case_insights(rows: list[dict[str, object]]) -> list[str]:
    lines: list[str] = []

    value_case = _best_case(rows, metric="ic_ir", name_tokens=("value", "bp"))
    quality_case = _best_case(rows, metric="ic_ir", name_tokens=("quality", "roe"))
    composite_case = _best_case(rows, metric="ic_ir", package_type="composite")

    lines.append(
        "- Value proxy (name contains value/bp): "
        f"{_insight_entry(value_case, rows, metric='ic_ir')}"
    )
    lines.append(
        "- Quality proxy (name contains quality/roe): "
        f"{_insight_entry(quality_case, rows, metric='ic_ir')}"
    )
    lines.append(
        "- Composite cases: "
        f"{_insight_entry(composite_case, rows, metric='ic_ir')}"
    )

    overall_best = _best_case(rows, metric="mean_long_short_return")
    if overall_best is not None:
        value = _case_metric(rows, overall_best, "mean_long_short_return")
        lines.append(
            "- Best long-short performance among successful cases: "
            f"`{overall_best}` ({format_metric(value)})"
        )
    else:
        lines.append("- Long-short comparison: N/A (no successful cases with finite returns).")

    return lines


def _highlight_line(row: dict[str, object]) -> str:
    case_name = format_text(row.get("case_name"))
    status = format_text(row.get("status"))
    metrics = _as_dict(row.get("metrics"))

    if status != "success":
        return f"- `{case_name}` did not complete successfully ({status})."

    return (
        f"- `{case_name}`: IC/ICIR={_metric_pair(metrics.get('mean_ic'), metrics.get('ic_ir'))}, "
        f"L/S={format_metric(metrics.get('mean_long_short_return'))}, "
        f"turnover={format_metric(metrics.get('mean_long_short_turnover'))}, "
        f"coverage={format_metric(metrics.get('coverage_mean'))}."
    )


def _best_case(
    rows: list[dict[str, object]],
    *,
    metric: str,
    name_tokens: tuple[str, ...] | None = None,
    package_type: str | None = None,
) -> str | None:
    best_case: str | None = None
    best_value: float | None = None
    for row in rows:
        if format_text(row.get("status")) != "success":
            continue
        case_name = format_text(row.get("case_name")).lower()
        row_package_type = format_text(row.get("package_type")).lower()

        if name_tokens is not None and not any(token in case_name for token in name_tokens):
            continue
        if package_type is not None and package_type.lower() != row_package_type:
            continue

        value = _to_float(_as_dict(row.get("metrics")).get(metric))
        if value is None:
            continue
        if best_value is None or value > best_value:
            best_value = value
            best_case = format_text(row.get("case_name"))
    return best_case


def _case_metric(rows: list[dict[str, object]], case_name: str, metric: str) -> float | None:
    for row in rows:
        if format_text(row.get("case_name")) == case_name:
            return _to_float(_as_dict(row.get("metrics")).get(metric))
    return None


def _insight_entry(case_name: str | None, rows: list[dict[str, object]], *, metric: str) -> str:
    if case_name is None:
        return "N/A"
    value = _case_metric(rows, case_name, metric)
    return f"`{case_name}` ({format_metric(value)})"


def _display_case_type(value: object) -> str:
    text = format_text(value).lower()
    if text == "single_factor":
        return "single"
    if text == "composite":
        return "composite"
    return "N/A"


def _metric_pair(left: object, right: object) -> str:
    return f"{format_metric(left)} / {format_metric(right)}"


def _resolve_path(raw: str, *, base: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _load_required_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"required artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, object], payload)


def _load_optional_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return cast(dict[str, object], payload)


def _as_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return cast(dict[str, object], value)
    return {}


def _as_list(value: object) -> list[object]:
    if isinstance(value, list):
        return cast(list[object], value)
    return []


def _to_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        if math.isfinite(out):
            return out
    return None


if __name__ == "__main__":
    raise SystemExit(main())
