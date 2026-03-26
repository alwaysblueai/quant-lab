from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import cast

import pandas as pd

from .templates import (
    CASE_SECTION_TITLES,
    PLACEHOLDER_INTERPRETATION,
    PLACEHOLDER_NEXT_STEPS,
    PLACEHOLDER_OBJECTIVE,
    format_metric,
    format_text,
    markdown_table,
    section_lines,
)


def render_case_report(case_output_dir: str | Path) -> str:
    """Render a standardized markdown report for one case output directory."""

    case_dir = Path(case_output_dir).resolve()
    manifest = _load_required_json(case_dir / "run_manifest.json")
    metrics_payload = _load_required_json(case_dir / "metrics.json")
    metrics = _extract_metrics(metrics_payload)
    spec = _as_dict(manifest.get("spec"))
    package_type = _resolve_package_type(manifest.get("artifact_type"))
    case_name = format_text(manifest.get("case_name"), na=case_dir.name)

    summary_text = _read_text_if_exists(case_dir / "summary.md")

    group_returns_path = case_dir / "group_returns.csv"
    turnover_path = case_dir / "turnover.csv"
    coverage_path = case_dir / "coverage.csv"
    group_summary = _group_return_summary(group_returns_path)
    mean_turnover = _mean_csv_metric(turnover_path, "turnover")
    mean_coverage = _mean_csv_metric(coverage_path, "coverage")

    objective_text = _extract_objective(summary_text)
    preprocess = _as_dict(spec.get("preprocess"))
    target = _as_dict(spec.get("target"))
    universe = _as_dict(spec.get("universe"))
    neutralization = _as_dict(spec.get("neutralization"))

    lines: list[str] = [f"# Case Report: {case_name}", ""]

    lines.extend(
        section_lines(
            CASE_SECTION_TITLES[0],
            [
                f"- Case name: `{case_name}`",
                f"- Package type: `{package_type}`",
            ],
        )
    )

    lines.extend(
        section_lines(
            CASE_SECTION_TITLES[1],
            [objective_text or PLACEHOLDER_OBJECTIVE],
        )
    )

    lines.extend(
        section_lines(
            CASE_SECTION_TITLES[2],
            _signal_definition_lines(
                package_type=package_type,
                spec=spec,
                metrics=metrics,
                preprocess=preprocess,
            ),
        )
    )

    target_kind = target.get("kind", metrics.get("target_kind"))
    target_horizon = target.get("horizon", metrics.get("target_horizon"))
    data_setup_lines = [
        f"- Universe: `{format_text(universe.get('name'))}`",
        (
            "- Label/target: "
            f"`{format_text(target_kind)}` (horizon={format_text(target_horizon)})"
        ),
        "- Rebalance frequency: "
        f"`{format_text(spec.get('rebalance_frequency', metrics.get('rebalance_frequency')))}`",
    ]
    lines.extend(section_lines(CASE_SECTION_TITLES[3], data_setup_lines))

    lines.extend(
        section_lines(
            CASE_SECTION_TITLES[4],
            _methodology_lines(
                package_type=package_type,
                preprocess=preprocess,
                neutralization=neutralization,
            ),
        )
    )

    diagnostics_rows = [
        ("IC / ICIR", _metric_pair(metrics.get("mean_ic"), metrics.get("ic_ir"))),
        (
            "Long-Short Performance",
            format_metric(metrics.get("mean_long_short_return")),
        ),
        (
            "Group Return Summary",
            group_summary if group_summary is not None else "N/A",
        ),
        (
            "Turnover",
            format_metric(metrics.get("mean_long_short_turnover"))
            if metrics.get("mean_long_short_turnover") is not None
            else format_metric(mean_turnover),
        ),
        (
            "Coverage",
            format_metric(metrics.get("coverage_mean"))
            if metrics.get("coverage_mean") is not None
            else format_metric(mean_coverage),
        ),
    ]
    lines.extend(
        section_lines(
            CASE_SECTION_TITLES[5],
            markdown_table(("Metric", "Value"), diagnostics_rows),
        )
    )

    lines.extend(
        section_lines(CASE_SECTION_TITLES[6], [PLACEHOLDER_INTERPRETATION])
    )

    lines.extend(
        section_lines(
            CASE_SECTION_TITLES[7],
            _risks_and_limitations_lines(
                metrics=metrics,
                neutralization=neutralization,
                transaction_cost=_as_dict(spec.get("transaction_cost")),
            ),
        )
    )

    lines.extend(section_lines(CASE_SECTION_TITLES[8], [PLACEHOLDER_NEXT_STEPS]))
    return "\n".join(lines).rstrip() + "\n"


def write_case_report(case_output_dir: str | Path, *, overwrite: bool = False) -> Path:
    """Render and write ``case_report.md`` in the provided case output directory."""

    case_dir = Path(case_output_dir).resolve()
    case_dir.mkdir(parents=True, exist_ok=True)
    report_path = case_dir / "case_report.md"
    if report_path.exists() and not overwrite:
        raise FileExistsError(
            f"{report_path} already exists. Pass overwrite=True to replace it."
        )
    report_path.write_text(render_case_report(case_dir), encoding="utf-8")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="case-report-renderer",
        description="Render case_report.md from an existing case artifact directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("case_output_dir", help="Case output directory containing manifests.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing case_report.md when present.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report_path = write_case_report(args.case_output_dir, overwrite=args.overwrite)
    except (FileNotFoundError, ValueError, KeyError, FileExistsError) as exc:
        parser.error(str(exc))

    print(f"Case report written: {report_path}")
    return 0


def _signal_definition_lines(
    *,
    package_type: str,
    spec: dict[str, object],
    metrics: dict[str, object],
    preprocess: dict[str, object],
) -> list[str]:
    lines: list[str] = []
    if package_type == "single_factor":
        lines.append(
            f"- Factor name: `{format_text(spec.get('factor_name', metrics.get('factor_name')))}`"
        )
        lines.append(
            f"- Direction: `{format_text(spec.get('direction', metrics.get('direction')))}`"
        )
    else:
        components = _as_list(spec.get("components"))
        if components:
            lines.append("- Components:")
            for component in components:
                row = _as_dict(component)
                lines.append(
                    "- "
                    f"`{format_text(row.get('name'))}` "
                    f"(weight={format_metric(_to_float(row.get('weight')))}, "
                    f"direction={format_text(row.get('direction'))}, "
                    f"transform={format_text(row.get('transform'))})"
                )
        else:
            lines.append("- Components: N/A")

    lines.append(
        "- Transformation steps: "
        f"{_render_preprocess_description(preprocess, include_standardization=True)}"
    )
    return lines


def _methodology_lines(
    *,
    package_type: str,
    preprocess: dict[str, object],
    neutralization: dict[str, object],
) -> list[str]:
    lines = [
        (
            "- Preprocessing: "
            f"{_render_preprocess_description(preprocess, include_standardization=True)}"
        ),
        f"- Neutralization: {_render_neutralization_description(neutralization)}",
        (
            "- Timestamp alignment: factor values at `t` are evaluated against "
            "forward returns after `t`."
        ),
    ]

    if package_type == "composite":
        lines.append(
            "- Combination logic: weighted linear blend of transformed component factors."
        )
    else:
        lines.append("- Combination logic: single-factor signal (no component blending).")
    return lines


def _risks_and_limitations_lines(
    *,
    metrics: dict[str, object],
    neutralization: dict[str, object],
    transaction_cost: dict[str, object],
) -> list[str]:
    lines: list[str] = []

    coverage = _to_float(metrics.get("coverage_mean"))
    if coverage is not None and coverage < 0.6:
        lines.append(
            f"- Coverage is low ({format_metric(coverage)}), so signals may be thin on some dates."
        )

    n_dates = _to_float(metrics.get("n_dates_used"))
    if n_dates is not None and n_dates < 30:
        lines.append(
            f"- Evaluation sample is short ({int(n_dates)} dates), increasing estimate uncertainty."
        )

    missingness = _to_float(metrics.get("missingness_mean"))
    if missingness is not None and missingness > 0.4:
        lines.append(
            f"- Missingness is elevated ({format_metric(missingness)}), which can bias diagnostics."
        )

    if neutralization.get("enabled") is False:
        lines.append("- Neutralization is disabled; factor may retain style/sector exposures.")

    cost = _to_float(transaction_cost.get("one_way_rate"))
    if cost is not None and math.isclose(cost, 0.0, rel_tol=0.0, abs_tol=1e-12):
        lines.append("- Transaction cost is set to 0.0; live frictions are not reflected.")

    if not lines:
        lines.append(
            "- Auto-generated from recorded artifacts; manual review is still "
            "required for deployment."
        )
    return lines


def _metric_pair(left: object, right: object) -> str:
    return f"{format_metric(left)} / {format_metric(right)}"


def _extract_objective(summary_text: str | None) -> str | None:
    if summary_text is None:
        return None
    lines = [line.strip() for line in summary_text.splitlines()]
    if not lines:
        return None

    for line in lines:
        lowered = line.lower()
        if lowered.startswith("objective:"):
            return line.split(":", 1)[1].strip() or None

    for idx, line in enumerate(lines):
        if line.lower().startswith("## objective"):
            for candidate in lines[idx + 1 :]:
                if candidate and not candidate.startswith("#"):
                    return candidate
            break

    return None


def _group_return_summary(path: Path) -> str | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or "group" not in frame.columns or "group_return" not in frame.columns:
        return None
    grouped = frame.groupby("group", dropna=True)["group_return"].mean()
    if grouped.empty:
        return None
    spread = float(grouped.max() - grouped.min())
    return (
        f"mean top-bottom spread={format_metric(spread)} "
        f"(groups={int(grouped.index.nunique())})"
    )


def _mean_csv_metric(path: Path, column: str) -> float | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or column not in frame.columns:
        return None
    value = float(frame[column].mean())
    if not math.isfinite(value):
        return None
    return value


def _render_preprocess_description(
    preprocess: dict[str, object],
    *,
    include_standardization: bool,
) -> str:
    if not preprocess:
        return "N/A"
    winsorize = bool(preprocess.get("winsorize", False))
    lower = format_metric(_to_float(preprocess.get("winsorize_lower")))
    upper = format_metric(_to_float(preprocess.get("winsorize_upper")))
    parts = [f"winsorize={winsorize} ({lower}, {upper})"]
    if include_standardization:
        parts.append(f"standardization={format_text(preprocess.get('standardization'))}")
    min_group = preprocess.get("min_group_size")
    if min_group is not None:
        parts.append(f"min_group_size={format_text(min_group)}")
    min_coverage = preprocess.get("min_coverage")
    if min_coverage is not None:
        parts.append(f"min_coverage={format_metric(_to_float(min_coverage))}")
    return ", ".join(parts)


def _render_neutralization_description(neutralization: dict[str, object]) -> str:
    if not neutralization:
        return "N/A"
    enabled = bool(neutralization.get("enabled", False))
    if not enabled:
        return "disabled"

    fields: list[str] = ["enabled"]
    for name in ("size_col", "industry_col", "beta_col"):
        value = neutralization.get(name)
        if value is not None:
            fields.append(f"{name}={value}")
    min_obs = neutralization.get("min_obs")
    if min_obs is not None:
        fields.append(f"min_obs={min_obs}")
    ridge = neutralization.get("ridge")
    if ridge is not None:
        fields.append(f"ridge={ridge}")
    return ", ".join(fields)


def _extract_metrics(payload: dict[str, object]) -> dict[str, object]:
    raw = payload.get("metrics")
    if isinstance(raw, dict):
        return cast(dict[str, object], raw)
    return {}


def _resolve_package_type(artifact_type: object) -> str:
    if not isinstance(artifact_type, str):
        return "N/A"
    text = artifact_type.lower()
    if "single_factor" in text:
        return "single_factor"
    if "composite" in text:
        return "composite"
    return "N/A"


def _read_text_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def _load_required_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"required artifact not found: {path}")
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
