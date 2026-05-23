"""Research tearsheet payload builder for Level 1/2 factor evaluation."""

from __future__ import annotations

import base64
import datetime as dt
import io
import json
import math
import re
import tempfile
import warnings
from collections import defaultdict
from collections.abc import Mapping
from html import escape as _escape_html
from pathlib import Path
from typing import Any

import pandas as pd

from alpha_lab.reporting.display_helpers import parse_text_list, safe_text, to_finite_float
from alpha_lab.reporting.factor_verdict import build_factor_verdict
from alpha_lab.reporting.level2_promotion import build_level2_promotion

RESEARCH_TEARSHEET_SCHEMA_VERSION = "1.0.0"
RESEARCH_TEARSHEET_ARTIFACT_TYPE = "alpha_lab_research_tearsheet"

_KNOWN_CSV_ARTIFACTS: dict[str, str] = {
    "ic_decay": "ic_decay.csv",
    "group_returns": "group_returns.csv",
    "turnover": "turnover.csv",
    "rolling_stability": "rolling_stability.csv",
    "ic_timeseries": "ic_timeseries.csv",
    "coverage": "coverage.csv",
}

_KNOWN_JSON_ARTIFACTS: dict[str, str] = {
    "backtest_result_json": "backtest_result.json",
}


def build_research_tearsheet_payload(
    *,
    metrics_path: str | Path,
    artifact_paths: Mapping[str, str | Path] | None = None,
    meta: Mapping[str, object] | None = None,
    schema_version: str = RESEARCH_TEARSHEET_SCHEMA_VERSION,
    artifact_type: str = RESEARCH_TEARSHEET_ARTIFACT_TYPE,
) -> dict[str, object]:
    """Build a robust tearsheet payload from metrics and optional artifacts.

    Parameters
    ----------
    metrics_path:
        Path to ``metrics.json``.
    artifact_paths:
        Optional artifact path hints. Missing artifacts are handled gracefully.
    meta:
        Optional metadata overrides/extensions for the top-level ``meta`` block.
    schema_version:
        Schema version string.
    artifact_type:
        Artifact type identifier.
    """

    metrics_file = Path(metrics_path).expanduser().resolve()
    case_dir = metrics_file.parent
    root_payload = _read_json_object(metrics_file)
    metrics = _extract_metrics(root_payload)

    artifacts = _load_artifacts(case_dir=case_dir, artifact_paths=artifact_paths or {})
    alias_values, alias_sources = _resolve_metric_aliases(
        metrics=metrics,
        root_payload=root_payload,
    )

    verdict_layer = _build_verdict_layer(metrics=metrics)

    setup_section = _build_setup_section(
        metrics=metrics,
        alias_values=alias_values,
        artifacts=artifacts,
        meta=meta or {},
    )
    signal_section = _build_signal_section(
        metrics=metrics,
        alias_values=alias_values,
        artifacts=artifacts,
    )
    stability_section = _build_stability_section(
        metrics=metrics,
        alias_values=alias_values,
        artifacts=artifacts,
    )
    conversion_risk_section = _build_conversion_risk_section(
        metrics=metrics,
        alias_values=alias_values,
    )
    appendix = _build_appendix(
        metrics=metrics,
        artifacts=artifacts,
    )

    meta_payload: dict[str, object] = {
        "metrics_path": str(metrics_file),
        "case_dir": str(case_dir),
        "field_aliases": alias_sources,
    }
    if meta:
        meta_payload.update(dict(meta))

    return {
        "schema_version": schema_version,
        "artifact_type": artifact_type,
        "meta": meta_payload,
        "verdict_layer": verdict_layer,
        "sections": {
            "setup": setup_section,
            "signal": signal_section,
            "stability": stability_section,
            "conversion_risk": conversion_risk_section,
        },
        "appendix": appendix,
    }


def export_research_tearsheet_pdf(
    *,
    payload: Mapping[str, object],
    output_path: str | Path,
) -> Path:
    """Render a report-style PDF from tearsheet payload.

    Prefer HTML+Chromium (Playwright) to preserve report layout and visual style.
    Fallback to matplotlib multi-page export when Playwright runtime is unavailable.
    """

    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    if _export_research_tearsheet_pdf_html(payload=payload, output_path=target):
        return target

    warnings.warn(
        "research_tearsheet PDF fallback: Playwright/Chromium unavailable; "
        "using matplotlib exporter with reduced visual fidelity.",
        stacklevel=2,
    )
    _export_research_tearsheet_pdf_matplotlib(payload=payload, output_path=target)
    return target


_PDF_SECTION_LABELS: dict[str, str] = {
    "setup": "Section 1 · Setup",
    "signal": "Section 2 · Signal",
    "stability": "Section 3 · Stability",
    "conversion_risk": "Section 4 · Conversion & Risk",
    "appendix": "附录 · Appendix",
}

_PDF_METRIC_LABELS: dict[str, str] = {
    "factor_name": "Factor",
    "direction": "Direction",
    "universe": "Universe",
    "target": "Target",
    "sample_window": "Sample Window",
    "coverage_mean": "Coverage Mean",
    "mean_rank_ic": "Mean RankIC",
    "ic_ir": "ICIR",
    "ic_positive_rate": "IC Hit Rate",
    "mean_long_short_return": "L/S Mean Return",
    "group_monotonicity_share": "Monotonicity Share",
    "qtop_qbottom_spread_mean": "QTop-QBottom Spread",
    "rolling_ic_positive_share": "Rolling IC Positive Share",
    "subperiod_positive_share_min": "Subperiod Positive Share Min",
    "ic_half_life_horizon": "IC Half-Life Horizon",
    "ic_decay_rebalance_ratio": "IC Decay/Rebalance Ratio",
    "base_portfolio_return": "Base Portfolio Return",
    "base_portfolio_turnover": "Base Portfolio Turnover",
    "ls_max_drawdown": "L/S Max Drawdown",
    "ls_cvar_5": "L/S CVaR 5%",
    "long_short_return_per_turnover": "L/S Return per Turnover",
}

_PDF_PERCENT_KEYS: set[str] = {
    "coverage_mean",
    "ic_positive_rate",
    "mean_long_short_return",
    "group_monotonicity_share",
    "qtop_qbottom_spread_mean",
    "rolling_ic_positive_share",
    "subperiod_positive_share_min",
    "base_portfolio_return",
    "base_portfolio_turnover",
    "ls_max_drawdown",
    "ls_cvar_5",
    "long_short_return_per_turnover",
}


def _export_research_tearsheet_pdf_html(
    *,
    payload: Mapping[str, object],
    output_path: Path,
) -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return False

    html_doc = _render_research_tearsheet_pdf_html(payload=payload)
    try:
        with tempfile.TemporaryDirectory(prefix="alpha_lab_tearsheet_pdf_") as tmp_dir:
            html_path = Path(tmp_dir) / "research_tearsheet_export.html"
            html_path.write_text(html_doc, encoding="utf-8")
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                try:
                    page = browser.new_page(viewport={"width": 1560, "height": 1100})
                    page.goto(html_path.as_uri(), wait_until="networkidle")
                    page.wait_for_timeout(200)
                    page.pdf(
                        path=str(output_path),
                        format="A4",
                        print_background=True,
                        margin={
                            "top": "10mm",
                            "right": "10mm",
                            "bottom": "12mm",
                            "left": "10mm",
                        },
                        prefer_css_page_size=True,
                    )
                finally:
                    browser.close()
    except Exception:
        return False
    return output_path.exists() and output_path.stat().st_size > 0


def _render_research_tearsheet_pdf_html(*, payload: Mapping[str, object]) -> str:
    verdict_layer = _as_mapping(payload.get("verdict_layer"))
    sections = _as_mapping(payload.get("sections"))
    appendix = _as_mapping(payload.get("appendix"))
    setup_metrics = _as_mapping(_as_mapping(sections.get("setup")).get("metrics"))
    factor_name = safe_text(setup_metrics.get("factor_name")) or safe_text(
        _get_nested(payload, "meta.factor_name")
    )
    universe = safe_text(setup_metrics.get("universe"))
    sample_window = safe_text(setup_metrics.get("sample_window"))
    summary_sentence = safe_text(verdict_layer.get("summary_sentence")) or "-"

    header_meta_chunks = []
    if factor_name:
        header_meta_chunks.append(f"Factor: {_h(factor_name)}")
    if universe:
        header_meta_chunks.append(f"Universe: {_h(universe)}")
    if sample_window:
        header_meta_chunks.append(f"Window: {_h(sample_window)}")
    header_meta = " | ".join(header_meta_chunks) if header_meta_chunks else "-"

    strip_items = _build_pdf_summary_strip_items(sections=sections)
    strip_html = (
        "".join(
            (
                "<div class='summary-chip'>"
                f"<span class='k'>{_h(label)}</span>"
                f"<span class='v'>{_h(value)}</span>"
                "</div>"
            )
            for label, value in strip_items
        )
        if strip_items
        else "<div class='summary-empty'>暂无可展示摘要指标</div>"
    )

    section_order = ("setup", "signal", "stability", "conversion_risk")
    section_html_chunks: list[str] = []
    for section_key in section_order:
        section = _as_mapping(sections.get(section_key))
        rendered = _render_pdf_section_html(section_key=section_key, section=section)
        if rendered:
            section_html_chunks.append(rendered)

    appendix_rendered = _render_pdf_section_html(section_key="appendix", section=appendix)
    if appendix_rendered:
        section_html_chunks.append(appendix_rendered)

    content = "\n".join(section_html_chunks)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Alpha-Lab Research Tearsheet</title>
  <style>
    :root {{
      --bg: #f5f4ed;
      --panel: #f6f4ea;
      --panel-white: #fffefb;
      --line: #d8d4c5;
      --line-soft: #e7e3d6;
      --rule-cream: #ded8c6;
      --ink: #1f2937;
      --muted: #6b7280;
      --brand: #577a4f;
      --brand-dark: #3f5c3a;
      --brand-soft: #e6efde;
      --shadow: 0 2px 8px rgba(49, 55, 33, 0.06);
    }}
    @page {{ size: A4; margin: 10mm; }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background: var(--bg);
      font-family: "PingFang SC", "Noto Sans SC", "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      font-size: 12px;
      line-height: 1.45;
    }}
    .report {{ display: grid; gap: 12px; }}
    .card {{
      border: 1px solid var(--rule-cream);
      border-radius: 12px;
      background: linear-gradient(180deg, var(--panel-white) 0%, var(--panel) 100%);
      box-shadow: var(--shadow);
      padding: 10px 12px;
    }}
    .head-title {{
      margin: 0 0 4px 0;
      color: var(--brand-dark);
      font-size: 19px;
      font-weight: 800;
      letter-spacing: 0.01em;
    }}
    .head-meta {{
      margin: 0;
      color: var(--muted);
      font-size: 12px;
    }}
    .head-summary {{
      margin-top: 8px;
      color: var(--ink);
      font-size: 12px;
    }}
    .summary-strip {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }}
    .summary-chip {{
      border: 1px solid var(--line-soft);
      border-radius: 10px;
      background: #fff;
      padding: 8px 10px;
      display: grid;
      gap: 3px;
    }}
    .summary-chip .k {{
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 700;
    }}
    .summary-chip .v {{
      color: var(--ink);
      font-size: 14px;
      font-weight: 800;
      font-variant-numeric: tabular-nums;
    }}
    .summary-empty {{ color: var(--muted); }}
    .section {{
      border: 1px solid var(--line-soft);
      border-radius: 12px;
      background: #fff;
      padding: 10px;
      break-inside: avoid-page;
      page-break-inside: avoid;
    }}
    .section-title {{
      margin: 0 0 8px 0;
      color: var(--brand-dark);
      font-size: 15px;
      font-weight: 800;
      letter-spacing: 0.01em;
    }}
    .section-summary {{
      margin: 0 0 8px 0;
      color: var(--muted);
      font-size: 12px;
    }}
    .metrics-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px 12px;
      margin-bottom: 10px;
    }}
    .metric-item {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 5px 0;
      border-bottom: 1px solid var(--line-soft);
    }}
    .metric-item:last-child {{ border-bottom: 0; }}
    .metric-k {{
      color: var(--muted);
      font-size: 11px;
      letter-spacing: 0.02em;
    }}
    .metric-v {{
      color: var(--ink);
      font-weight: 700;
      font-variant-numeric: tabular-nums;
      text-align: right;
      word-break: break-word;
    }}
    .charts-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 10px;
    }}
    .chart-card {{
      border: 1px solid var(--line-soft);
      border-radius: 10px;
      padding: 8px;
      background: var(--panel-white);
      break-inside: avoid-page;
      page-break-inside: avoid;
    }}
    .chart-title {{
      margin: 0 0 6px 0;
      color: var(--brand-dark);
      font-size: 12px;
      font-weight: 800;
    }}
    .chart-card img {{
      width: 100%;
      height: auto;
      border-radius: 8px;
      border: 1px solid var(--line-soft);
      display: block;
      background: #fff;
    }}
    .muted {{ color: var(--muted); font-size: 12px; }}
    .table-wrap {{
      margin-top: 8px;
      border: 1px solid var(--line-soft);
      border-radius: 10px;
      overflow: hidden;
      break-inside: avoid-page;
      page-break-inside: avoid;
    }}
    .table-title {{
      margin: 0;
      padding: 6px 8px;
      background: #f8f7f2;
      color: var(--brand-dark);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      font-weight: 800;
      border-bottom: 1px solid var(--line-soft);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 11px;
      font-variant-numeric: tabular-nums;
    }}
    thead th {{
      text-align: left;
      color: var(--muted);
      font-size: 10px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      background: #fbfaf6;
      padding: 6px 8px;
      border-bottom: 1px solid var(--line-soft);
    }}
    tbody td {{
      padding: 6px 8px;
      border-bottom: 1px solid var(--line-soft);
      vertical-align: top;
    }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    @media print {{
      .summary-strip, .charts-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
    }}
  </style>
</head>
<body>
  <div class="report">
    <section class="card">
      <h1 class="head-title">Alpha-Lab 单因子全面评价报告</h1>
      <p class="head-meta">{header_meta}</p>
      <p class="head-summary">{_h(summary_sentence)}</p>
    </section>
    <section class="card">
      <div class="summary-strip">{strip_html}</div>
    </section>
    {content}
  </div>
</body>
</html>
"""


def _build_pdf_summary_strip_items(
    *,
    sections: Mapping[str, object],
) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    summary_specs = (
        ("signal", "mean_rank_ic"),
        ("signal", "ic_ir"),
        ("signal", "ic_positive_rate"),
        ("signal", "mean_long_short_return"),
        ("stability", "ic_half_life_horizon"),
        ("stability", "ic_decay_rebalance_ratio"),
        ("conversion_risk", "ls_max_drawdown"),
        ("conversion_risk", "long_short_return_per_turnover"),
    )
    for section_key, metric_key in summary_specs:
        metrics = _as_mapping(_as_mapping(sections.get(section_key)).get("metrics"))
        if metric_key not in metrics:
            continue
        value = _format_html_metric_value(metric_key, metrics.get(metric_key))
        if value == "-":
            continue
        label = _PDF_METRIC_LABELS.get(metric_key, metric_key)
        items.append((label, value))
    return items[:8]


def _render_pdf_section_html(
    *,
    section_key: str,
    section: Mapping[str, object],
) -> str:
    if not section:
        return ""
    title = _PDF_SECTION_LABELS.get(section_key, section_key)
    summary = safe_text(section.get("summary")) or ""
    metrics = _as_mapping(section.get("metrics"))
    charts = _as_chart_list(section.get("charts"))
    tables = _as_table_list(section.get("tables"))

    if section_key == "appendix":
        availability = _as_mapping(section.get("artifact_availability"))
        notes = section.get("notes")
        notes_items = parse_text_list(notes)
        availability_metrics = {
            f"artifact::{key}": ("available" if bool(value) else "missing")
            for key, value in availability.items()
        }
        if availability_metrics:
            metrics = {**availability_metrics, **metrics}
        if notes_items:
            metrics = {
                **metrics,
                "appendix_notes": "；".join(item for item in notes_items if item),
            }

    metrics_html = _render_pdf_metrics_grid_html(metrics=metrics)
    charts_html = _render_pdf_charts_html(charts=charts, section_key=section_key)
    tables_html = _render_pdf_tables_html(tables=tables)

    body_parts = [metrics_html, charts_html, tables_html]
    body_html = "".join(chunk for chunk in body_parts if chunk)
    if not body_html:
        return ""
    summary_html = f"<p class='section-summary'>{_h(summary)}</p>" if summary else ""
    return (
        "<section class='section'>"
        f"<h2 class='section-title'>{_h(title)}</h2>"
        f"{summary_html}"
        f"{body_html}"
        "</section>"
    )


def _render_pdf_metrics_grid_html(*, metrics: Mapping[str, object]) -> str:
    if not metrics:
        return ""
    items: list[str] = []
    for key, value in metrics.items():
        rendered = _format_html_metric_value(str(key), value)
        if rendered == "-":
            continue
        label = _PDF_METRIC_LABELS.get(str(key), str(key))
        items.append(
            "<div class='metric-item'>"
            f"<span class='metric-k'>{_h(label)}</span>"
            f"<span class='metric-v'>{_h(rendered)}</span>"
            "</div>"
        )
    if not items:
        return ""
    return f"<div class='metrics-grid'>{''.join(items)}</div>"


def _render_pdf_charts_html(*, charts: list[dict[str, object]], section_key: str) -> str:
    if not charts:
        return ""
    cards: list[str] = []
    for chart in charts:
        title = safe_text(chart.get("title")) or "Chart"
        img_src = _render_chart_data_uri(chart=chart, section_key=section_key)
        if img_src is None:
            cards.append(
                "<article class='chart-card'>"
                f"<h3 class='chart-title'>{_h(title)}</h3>"
                "<div class='muted'>图表数据不足。</div>"
                "</article>"
            )
            continue
        cards.append(
            "<article class='chart-card'>"
            f"<h3 class='chart-title'>{_h(title)}</h3>"
            f"<img src='{img_src}' alt='{_h(title)}' />"
            "</article>"
        )
    if not cards:
        return ""
    return f"<div class='charts-grid'>{''.join(cards)}</div>"


def _render_chart_data_uri(*, chart: Mapping[str, object], section_key: str) -> str | None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    fig = _render_tearsheet_chart_figure(chart=chart, section_key=section_key, plt=plt)
    if fig is None:
        return None
    fig.patch.set_facecolor("#fffefb")
    png = io.BytesIO()
    try:
        fig.savefig(
            png,
            format="png",
            dpi=220,
            bbox_inches="tight",
            facecolor=fig.get_facecolor(),
        )
    finally:
        plt.close(fig)
    encoded = base64.b64encode(png.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _render_pdf_tables_html(*, tables: list[dict[str, object]]) -> str:
    if not tables:
        return ""
    chunks: list[str] = []
    for table in tables:
        title = safe_text(table.get("title")) or "Table"
        columns_raw = table.get("columns")
        columns = (
            [str(item) for item in columns_raw if safe_text(item)]
            if isinstance(columns_raw, (list, tuple))
            else []
        )
        rows = table.get("rows")
        row_items = rows if isinstance(rows, list) else []
        if not columns or not row_items:
            continue

        header_html = "".join(f"<th>{_h(col)}</th>" for col in columns)
        body_rows: list[str] = []
        for row in row_items:
            values: list[str] = []
            if isinstance(row, Mapping):
                for col in columns:
                    values.append(_format_html_table_value(row.get(col)))
            elif isinstance(row, (list, tuple)):
                row_list = list(row)
                for idx in range(len(columns)):
                    values.append(
                        _format_html_table_value(row_list[idx] if idx < len(row_list) else None)
                    )
            else:
                continue
            body_rows.append(
                "<tr>" + "".join(f"<td>{_h(value)}</td>" for value in values) + "</tr>"
            )
        if not body_rows:
            continue

        chunks.append(
            "<div class='table-wrap'>"
            f"<p class='table-title'>{_h(title)}</p>"
            "<table>"
            f"<thead><tr>{header_html}</tr></thead>"
            f"<tbody>{''.join(body_rows)}</tbody>"
            "</table>"
            "</div>"
        )
    return "".join(chunks)


def _format_html_table_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "true" if value else "false"
    numeric = to_finite_float(value)
    if numeric is not None:
        return f"{numeric:.6g}"
    text = safe_text(value)
    return text or "-"


def _format_html_metric_value(key: str, value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, Mapping):
        rendered_items: list[str] = []
        for sub_key, sub_value in value.items():
            if sub_value is None:
                continue
            text = _format_html_metric_value(str(sub_key), sub_value)
            if text == "-":
                continue
            rendered_items.append(f"{sub_key}={text}")
        return ", ".join(rendered_items) if rendered_items else "-"
    numeric = to_finite_float(value)
    if numeric is not None:
        if key in _PDF_PERCENT_KEYS:
            return f"{numeric * 100:.2f}%"
        if key.endswith("_horizon"):
            return f"{numeric:.1f}"
        if key.startswith("n_"):
            return str(int(round(numeric)))
        return f"{numeric:.6g}"
    text_value = safe_text(value)
    return text_value or "-"


def _h(text: object) -> str:
    return _escape_html(str(text), quote=True)


def _as_table_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _export_research_tearsheet_pdf_matplotlib(
    *,
    payload: Mapping[str, object],
    output_path: Path,
) -> None:
    """Fallback PDF exporter based on matplotlib figures."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    sections = _as_mapping(payload.get("sections"))
    appendix = _as_mapping(payload.get("appendix"))

    with PdfPages(output_path) as pdf:
        cover_fig = _render_tearsheet_cover_figure(payload=payload, plt=plt)
        pdf.savefig(cover_fig, bbox_inches="tight")
        plt.close(cover_fig)

        for section_key in ("signal", "stability", "conversion_risk", "setup"):
            section = _as_mapping(sections.get(section_key))
            for chart in _as_chart_list(section.get("charts")):
                fig = _render_tearsheet_chart_figure(
                    chart=chart,
                    section_key=section_key,
                    plt=plt,
                )
                if fig is None:
                    continue
                pdf.savefig(fig, bbox_inches="tight")
                plt.close(fig)

        for chart in _as_chart_list(appendix.get("charts")):
            fig = _render_tearsheet_chart_figure(
                chart=chart,
                section_key="appendix",
                plt=plt,
            )
            if fig is None:
                continue
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)


def _render_tearsheet_cover_figure(*, payload: Mapping[str, object], plt: Any) -> Any:
    fig, ax = plt.subplots(figsize=(11.69, 8.27))  # A4 landscape
    ax.axis("off")

    verdict = _as_mapping(payload.get("verdict_layer"))
    sections = _as_mapping(payload.get("sections"))
    meta = _as_mapping(payload.get("meta"))
    summary = safe_text(verdict.get("summary_sentence")) or "No summary available."

    lines: list[str] = [
        "Alpha-Lab Research Tearsheet",
        "",
        f"Factor Verdict: {safe_text(verdict.get('factor_verdict')) or '-'}",
        f"Promotion Decision: {safe_text(verdict.get('promotion_decision')) or '-'}",
        f"Summary: {summary}",
        "",
    ]

    factor_name = safe_text(_get_nested(meta, "factor_name")) or safe_text(
        _get_nested(_as_mapping(sections.get("setup")), "metrics.factor_name")
    )
    if factor_name is not None:
        lines.append(f"Factor: {factor_name}")

    for section_key in ("signal", "stability", "conversion_risk"):
        section = _as_mapping(sections.get(section_key))
        metrics = _as_mapping(section.get("metrics"))
        if not metrics:
            continue
        lines.append("")
        lines.append(f"[{section_key}]")
        for key, value in list(metrics.items())[:6]:
            lines.append(f"- {key}: {_format_pdf_metric_value(value)}")

    y = 0.97
    for idx, line in enumerate(lines[:48]):
        size = 15 if idx == 0 else 10
        ax.text(
            0.02,
            y,
            line,
            fontsize=size,
            ha="left",
            va="top",
            transform=ax.transAxes,
        )
        y -= 0.035
    return fig


def _render_tearsheet_chart_figure(
    *,
    chart: Mapping[str, object],
    section_key: str,
    plt: Any,
) -> Any | None:
    chart_type = (safe_text(chart.get("type")) or "").strip().lower()
    title = safe_text(chart.get("title")) or "Chart"
    series_items = _as_chart_series(chart.get("series"))

    if not series_items:
        return None

    fig, ax = plt.subplots(figsize=(11.69, 6.5))
    ax.set_title(f"{title} ({section_key})", fontsize=12, loc="left")
    ax.grid(alpha=0.2, linestyle="--", linewidth=0.7)

    if chart_type in {"line", "timeseries"}:
        rendered = _plot_tearsheet_line(ax=ax, series_items=series_items)
    elif chart_type in {"bar", "bars"}:
        rendered = _plot_tearsheet_bar(ax=ax, series_items=series_items)
    elif chart_type in {"hist", "histogram"}:
        rendered = _plot_tearsheet_histogram(ax=ax, series_items=series_items)
    else:
        rendered = _plot_tearsheet_line(ax=ax, series_items=series_items)

    if not rendered:
        plt.close(fig)
        return None
    return fig


def _plot_tearsheet_line(*, ax: Any, series_items: list[dict[str, object]]) -> bool:
    has_data = False
    date_ticks: list[tuple[float, str]] = []
    for series in series_items:
        name = safe_text(series.get("name")) or "series"
        points = series.get("points")
        parsed = _parse_line_series_points(points)
        if parsed is None:
            continue
        x_values, y_values, x_labels = parsed
        if len(x_values) < 2:
            continue
        has_data = True
        ax.plot(x_values, y_values, linewidth=1.6, label=name)
        if not date_ticks and x_labels is not None and len(x_values) >= 2:
            date_ticks = _build_annual_axis_ticks(x_values=x_values, x_labels=x_labels)
    if has_data:
        if date_ticks:
            ax.set_xticks([tick[0] for tick in date_ticks])
            ax.set_xticklabels(
                [tick[1] for tick in date_ticks],
                rotation=30 if len(date_ticks) > 6 else 0,
                ha="right" if len(date_ticks) > 6 else "center",
            )
            ax.set_xlabel("Date", fontsize=9)
        ax.legend(loc="best", fontsize=9)
    return has_data


def _plot_tearsheet_bar(*, ax: Any, series_items: list[dict[str, object]]) -> bool:
    for series in series_items:
        bars = series.get("bars")
        if not isinstance(bars, list):
            continue
        labels: list[str] = []
        values: list[float] = []
        for item in bars:
            if not isinstance(item, Mapping):
                continue
            label = safe_text(item.get("group"))
            value = to_finite_float(item.get("value"))
            if label is None or value is None:
                continue
            labels.append(label)
            values.append(value)
        if not labels:
            continue
        x_axis = list(range(len(labels)))
        ax.bar(x_axis, values, width=0.65)
        ax.set_xticks(x_axis)
        ax.set_xticklabels(labels, rotation=0 if len(labels) <= 8 else 30)
        return True
    return False


def _plot_tearsheet_histogram(*, ax: Any, series_items: list[dict[str, object]]) -> bool:
    for series in series_items:
        bins = series.get("bins")
        if not isinstance(bins, list):
            continue
        lefts: list[float] = []
        widths: list[float] = []
        counts: list[float] = []
        for item in bins:
            if not isinstance(item, Mapping):
                continue
            left = to_finite_float(item.get("left"))
            right = to_finite_float(item.get("right"))
            count = to_finite_float(item.get("count"))
            if left is None or right is None or count is None:
                continue
            width = right - left
            if width <= 0.0:
                continue
            lefts.append(left)
            widths.append(width)
            counts.append(count)
        if not lefts:
            continue
        ax.bar(lefts, counts, width=widths, align="edge", alpha=0.8)
        return True
    return False


def _parse_line_series_points(
    points: object,
) -> tuple[list[float], list[float], list[str] | None] | None:
    if not isinstance(points, list):
        return None
    x_values: list[float] = []
    y_values: list[float] = []
    text_x: list[str] = []
    has_text_x = False
    row_idx = 0
    for item in points:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        y_val = to_finite_float(item[1])
        if y_val is None:
            continue
        raw_x = item[0]
        x_val = to_finite_float(raw_x)
        if x_val is None:
            has_text_x = True
            text = safe_text(raw_x) or str(row_idx)
            x_values.append(float(row_idx))
            text_x.append(text)
        else:
            x_values.append(x_val)
            text_x.append(str(raw_x))
        y_values.append(y_val)
        row_idx += 1
    if len(x_values) < 2:
        return None
    return x_values, y_values, text_x if has_text_x else None


def _build_annual_axis_ticks(
    *,
    x_values: list[float],
    x_labels: list[str],
) -> list[tuple[float, str]]:
    dated: list[tuple[int, float, dt.date]] = []
    for idx, (x_value, label) in enumerate(zip(x_values, x_labels, strict=False)):
        parsed = _parse_axis_date(label)
        if parsed is not None:
            dated.append((idx, x_value, parsed))
    if len(dated) < 2:
        return []

    start = dated[0][2]
    end = dated[-1][2]
    if end < start:
        return []

    interval = _axis_tick_interval(start=start, end=end)
    ticks: list[tuple[float, str, dt.date]] = []
    used_indices: set[int] = set()
    last_idx = -1

    def append_first_on_or_after(target: dt.date) -> None:
        nonlocal last_idx
        candidate: tuple[int, float, dt.date] | None = None
        for item in dated:
            idx, _x_value, item_date = item
            if idx <= last_idx or item_date < target:
                continue
            candidate = item
            break
        if candidate is None:
            return
        idx, x_value, item_date = candidate
        label = _format_axis_date(item_date, include_day=interval[2])
        if idx in used_indices or (ticks and ticks[-1][1] == label):
            return
        used_indices.add(idx)
        last_idx = idx
        ticks.append((x_value, label, item_date))

    target = start
    while target <= end:
        append_first_on_or_after(target)
        target = _add_axis_tick_interval(target, months=interval[0], days=interval[1])

    last_idx_value, last_x, last_date = dated[-1]
    if (
        last_idx_value not in used_indices
        and (not ticks or (last_date - ticks[-1][2]).days >= interval[3])
    ):
        ticks.append((last_x, _format_axis_date(last_date, include_day=interval[2]), last_date))
    if len(ticks) < 2 and last_idx_value not in used_indices:
        ticks.append((last_x, _format_axis_date(last_date, include_day=interval[2]), last_date))
    return [(x_value, label) for x_value, label, _date in ticks]


def _parse_axis_date(value: object) -> dt.date | None:
    text = safe_text(value)
    if text is None:
        return None
    normalized = _normalize_date_token(text)
    if normalized is None:
        return None
    try:
        parts = normalized.replace("/", "-").replace(".", "-").split("-")
        year = int(parts[0])
        month = int(parts[1]) if len(parts) >= 2 else 1
        day = int(parts[2]) if len(parts) >= 3 else 1
        return dt.date(year, month, day)
    except (TypeError, ValueError, IndexError):
        return None


def _axis_tick_interval(*, start: dt.date, end: dt.date) -> tuple[int, int, bool, int]:
    span_days = max(0, (end - start).days)
    if span_days <= 45:
        return 0, 7, True, 4
    if span_days <= 120:
        return 0, 14, True, 7
    if span_days <= 550:
        return 1, 0, False, 15
    if span_days <= 1095:
        return 3, 0, False, 45
    if span_days <= 2190:
        return 6, 0, False, 90
    return 12, 0, False, 180


def _add_axis_tick_interval(value: dt.date, *, months: int, days: int) -> dt.date:
    if days > 0:
        return value + dt.timedelta(days=days)
    return _add_months_clamped(value, months=max(1, months))


def _add_months_clamped(value: dt.date, *, months: int) -> dt.date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    if month == 12:
        next_month = dt.date(year + 1, 1, 1)
    else:
        next_month = dt.date(year, month + 1, 1)
    last_day = (next_month - dt.timedelta(days=1)).day
    return dt.date(year, month, min(value.day, last_day))


def _format_axis_date(value: dt.date, *, include_day: bool = False) -> str:
    if include_day:
        return f"{value.year}.{value.month}.{value.day}"
    return f"{value.year}.{value.month}"


def _as_chart_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _as_chart_series(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _format_pdf_metric_value(value: object) -> str:
    numeric = to_finite_float(value)
    if numeric is not None:
        return f"{numeric:.6g}"
    text = safe_text(value)
    return text or "-"


def _build_verdict_layer(
    *,
    metrics: Mapping[str, object],
) -> dict[str, object]:
    verdict_obj = build_factor_verdict(metrics)
    promotion_obj = build_level2_promotion(metrics)

    factor_verdict = safe_text(metrics.get("factor_verdict")) or verdict_obj.label
    verdict_reasons = parse_text_list(metrics.get("factor_verdict_reasons"))
    if not verdict_reasons:
        verdict_reasons = [item for item in verdict_obj.reasons if safe_text(item)]

    promotion_decision = safe_text(metrics.get("promotion_decision")) or promotion_obj.label
    promotion_reasons = parse_text_list(metrics.get("promotion_reasons"))
    if not promotion_reasons:
        promotion_reasons = [item for item in promotion_obj.reasons if safe_text(item)]

    blockers = parse_text_list(metrics.get("promotion_blockers"))
    if not blockers:
        blockers = [item for item in promotion_obj.blockers if safe_text(item)]

    caveats = [
        item for item in promotion_reasons if item not in blockers and "blocked" not in item.lower()
    ]
    if not caveats:
        caveats = [item for item in verdict_reasons if item not in blockers]
    caveats = caveats[:3]

    primary_support_summary = (
        _first_non_empty(verdict_reasons)
        or _first_non_empty(promotion_reasons)
        or "no clear supportive evidence found"
    )
    primary_blocker_summary = (
        _first_non_empty(blockers) or _first_non_empty(caveats) or "no explicit blocker reported"
    )
    summary_sentence = (
        f"{factor_verdict}; {promotion_decision}. "
        f"Primary support: {primary_support_summary}. "
        f"Primary blocker/caveat: {primary_blocker_summary}."
    )

    return {
        "factor_verdict": factor_verdict,
        "promotion_decision": promotion_decision,
        "summary_sentence": summary_sentence,
        "verdict_reasons": verdict_reasons,
        "blockers": blockers,
        "caveats": caveats,
        "primary_support_summary": primary_support_summary,
        "primary_blocker_summary": primary_blocker_summary,
    }


def _build_setup_section(
    *,
    metrics: Mapping[str, object],
    alias_values: Mapping[str, object],
    artifacts: Mapping[str, object],
    meta: Mapping[str, object],
) -> dict[str, object]:
    ic_df = _as_df(artifacts.get("ic_timeseries"))
    split_contract = _resolve_split_contract(metrics=metrics, meta=meta)
    split_description = (
        _format_split_contract_description(split_contract)
        or safe_text(metrics.get("split_description"))
    )
    sample_window = _resolve_sample_window(
        ic_df=ic_df,
        split_description=split_description,
    )
    target_kind = (
        safe_text(metrics.get("target_kind"))
        or safe_text(meta.get("target_kind"))
        or safe_text(meta.get("label_name"))
    )
    target_horizon = (
        to_finite_float(metrics.get("target_horizon"))
        or to_finite_float(meta.get("target_horizon"))
        or to_finite_float(metrics.get("rebalance_step_dates"))
    )
    universe = (
        safe_text(metrics.get("universe"))
        or safe_text(metrics.get("universe_name"))
        or safe_text(meta.get("universe"))
        or safe_text(meta.get("universe_name"))
    )
    factor_name = (
        safe_text(metrics.get("factor_name"))
        or safe_text(meta.get("factor_name"))
        or safe_text(meta.get("name"))
    )
    direction = safe_text(metrics.get("direction")) or safe_text(meta.get("direction"))

    setup_metrics = {
        "factor_name": factor_name,
        "direction": direction,
        "universe": universe,
        "target": {
            "kind": target_kind,
            "horizon": target_horizon,
        },
        "sample_window": sample_window,
        "coverage_mean": alias_values.get("coverage_mean"),
    }

    return {
        "summary": "实验设定与样本边界",
        "metrics": setup_metrics,
        "charts": [],
        "tables": [],
    }


def _build_signal_section(
    *,
    metrics: Mapping[str, object],
    alias_values: Mapping[str, object],
    artifacts: Mapping[str, object],
) -> dict[str, object]:
    signal_metrics = {
        "mean_rank_ic": to_finite_float(metrics.get("mean_rank_ic")),
        "ic_ir": to_finite_float(metrics.get("ic_ir")),
        "ic_positive_rate": to_finite_float(metrics.get("ic_positive_rate")),
        "mean_long_short_return": alias_values.get("long_short_mean_return"),
        "group_monotonicity_share": to_finite_float(metrics.get("group_monotonicity_share")),
        "qtop_qbottom_spread_mean": to_finite_float(metrics.get("group_monotonicity_qtop_qbottom")),
    }
    charts: list[dict[str, object]] = []
    chart = _build_cumulative_long_short_nav_chart(artifacts=artifacts)
    if chart is not None:
        charts.append(chart)
    chart = _build_ic_timeseries_with_cumulative_chart(artifacts=artifacts)
    if chart is not None:
        charts.append(chart)
    chart = _build_quantile_cumulative_returns_chart(
        artifacts=artifacts,
        metrics=metrics,
    )
    if chart is not None:
        charts.append(chart)
    chart = _build_group_mean_return_chart(artifacts=artifacts)
    if chart is not None:
        charts.append(chart)

    table_rows = [
        {"metric": "mean_rank_ic", "value": signal_metrics["mean_rank_ic"]},
        {"metric": "ic_ir", "value": signal_metrics["ic_ir"]},
        {"metric": "ic_positive_rate", "value": signal_metrics["ic_positive_rate"]},
        {
            "metric": "mean_long_short_return",
            "value": signal_metrics["mean_long_short_return"],
        },
        {
            "metric": "group_monotonicity_share",
            "value": signal_metrics["group_monotonicity_share"],
        },
        {
            "metric": "qtop_qbottom_spread_mean",
            "value": signal_metrics["qtop_qbottom_spread_mean"],
        },
    ]
    table_rows = [row for row in table_rows if row["value"] is not None]
    tables = [
        {
            "title": "IC Summary",
            "columns": ["metric", "value"],
            "rows": table_rows,
        }
    ]

    return {
        "summary": "核心信号证据",
        "metrics": signal_metrics,
        "charts": charts,
        "tables": tables,
    }


def _build_stability_section(
    *,
    metrics: Mapping[str, object],
    alias_values: Mapping[str, object],
    artifacts: Mapping[str, object],
) -> dict[str, object]:
    subperiod_min = _min_finite(
        to_finite_float(metrics.get("subperiod_positive_share_min")),
        to_finite_float(metrics.get("subperiod_ic_positive_share")),
        to_finite_float(metrics.get("subperiod_long_short_positive_share")),
    )
    stability_metrics = {
        "rolling_ic_positive_share": to_finite_float(metrics.get("rolling_ic_positive_share")),
        "subperiod_positive_share_min": subperiod_min,
        "ic_half_life_horizon": to_finite_float(metrics.get("ic_half_life_horizon")),
        "ic_decay_rebalance_ratio": to_finite_float(metrics.get("ic_decay_rebalance_ratio")),
        "long_short_return_per_turnover": alias_values.get("long_short_return_per_turnover"),
    }
    charts: list[dict[str, object]] = []
    chart = _build_rolling_ic_rankic_chart(artifacts=artifacts)
    if chart is not None:
        charts.append(chart)
    chart = _build_ic_decay_chart(artifacts=artifacts)
    if chart is not None:
        charts.append(chart)
    chart = _build_ic_distribution_chart(artifacts=artifacts)
    if chart is not None:
        charts.append(chart)

    return {
        "summary": "稳定性与衰减",
        "metrics": stability_metrics,
        "charts": charts,
        "tables": [],
    }


def _build_conversion_risk_section(
    *,
    metrics: Mapping[str, object],
    alias_values: Mapping[str, object],
) -> dict[str, object]:
    conversion_metrics = {
        "base_portfolio_return": alias_values.get("base_portfolio_return"),
        "base_portfolio_turnover": alias_values.get("base_portfolio_turnover"),
        "ls_max_drawdown": alias_values.get("ls_max_drawdown"),
        "ls_cvar_5": alias_values.get("ls_cvar_5"),
        "long_short_return_per_turnover": alias_values.get("long_short_return_per_turnover"),
    }
    return {
        "summary": "组合转化与风险",
        "metrics": conversion_metrics,
        "charts": [],
        "tables": [],
    }


def _build_appendix(
    *,
    metrics: Mapping[str, object],
    artifacts: Mapping[str, object],
) -> dict[str, object]:
    charts: list[dict[str, object]] = []
    turnover_chart = _build_turnover_timeseries_chart(artifacts=artifacts)
    if turnover_chart is not None:
        charts.append(turnover_chart)
    coverage_chart = _build_coverage_timeseries_chart(artifacts=artifacts)
    if coverage_chart is not None:
        charts.append(coverage_chart)

    tables: list[dict[str, object]] = []
    yearly_rows = _build_yearly_breakdown_rows(artifacts=artifacts)
    if yearly_rows:
        tables.append(
            {
                "title": "Yearly Breakdown",
                "columns": [
                    "year",
                    "mean_long_short_return",
                ],
                "rows": yearly_rows,
            }
        )
    decay_rows = _build_ic_decay_table_rows(artifacts=artifacts)
    if decay_rows:
        tables.append(
            {
                "title": "IC Decay by Lag",
                "columns": ["lag", "mean_ic", "mean_rank_ic"],
                "rows": decay_rows,
            }
        )

    availability = {key: bool(_artifact_present(value)) for key, value in artifacts.items()}
    return {
        "artifact_availability": availability,
        "charts": charts,
        "tables": tables,
        "notes": parse_text_list(metrics.get("campaign_triage_reasons")),
    }


def _resolve_metric_aliases(
    *,
    metrics: Mapping[str, object],
    root_payload: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, str]]:
    alias_map: dict[str, tuple[str, ...]] = {
        "coverage_mean": (
            "coverage_mean",
            "eval_coverage_ratio_mean",
            "coverage_by_date_summary.mean_coverage",
        ),
        "coverage_min": (
            "coverage_min",
            "eval_coverage_ratio_min",
            "coverage_by_date_summary.min_coverage",
        ),
        "long_short_mean_return": (
            "long_short_mean_return",
            "mean_long_short_return",
        ),
        "long_short_turnover_mean": (
            "long_short_turnover_mean",
            "mean_long_short_turnover",
        ),
        "long_short_return_per_turnover": (
            "long_short_return_per_turnover",
            "mean_long_short_return_per_turnover",
        ),
        "base_portfolio_return": (
            "base_portfolio_return",
            "portfolio_validation_base_mean_portfolio_return",
        ),
        "base_portfolio_turnover": (
            "base_portfolio_turnover",
            "portfolio_validation_base_mean_turnover",
        ),
        "base_portfolio_cost_adjusted_return": (
            "base_portfolio_cost_adjusted_return",
            "portfolio_validation_base_cost_adjusted_return_review_rate",
        ),
        "ls_max_drawdown": (
            "ls_max_drawdown",
            "max_drawdown",
        ),
        "ls_cvar_5": (
            "ls_cvar_5",
            "cvar_5",
        ),
    }

    values: dict[str, object] = {}
    sources: dict[str, str] = {}
    for canonical, candidates in alias_map.items():
        chosen_value: object = None
        chosen_source = ""
        for source in candidates:
            resolved = _resolve_alias_source(
                source=source,
                metrics=metrics,
                root_payload=root_payload,
            )
            if resolved is None:
                continue
            chosen_value = resolved
            chosen_source = source
            break
        values[canonical] = chosen_value
        if chosen_source:
            sources[canonical] = chosen_source
    return values, sources


def _resolve_alias_source(
    *,
    source: str,
    metrics: Mapping[str, object],
    root_payload: Mapping[str, object],
) -> object | None:
    if "." in source:
        value = _get_nested(root_payload, source)
        if to_finite_float(value) is not None:
            return to_finite_float(value)
        return value if value is not None else None

    value = metrics.get(source)
    if value is None:
        return None
    numeric = to_finite_float(value)
    if numeric is not None:
        return numeric
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return None
    return value


def _load_artifacts(
    *,
    case_dir: Path,
    artifact_paths: Mapping[str, str | Path],
) -> dict[str, object]:
    loaded: dict[str, object] = {}
    for key, fallback_name in _KNOWN_CSV_ARTIFACTS.items():
        path = _resolve_artifact_path(
            key=key,
            fallback_name=fallback_name,
            case_dir=case_dir,
            artifact_paths=artifact_paths,
        )
        loaded[key] = _read_csv(path)

    group_df = _as_df(loaded.get("group_returns"))
    if group_df.empty:
        quantile_path = _resolve_artifact_path(
            key="quantile_returns",
            fallback_name="quantile_returns.csv",
            case_dir=case_dir,
            artifact_paths=artifact_paths,
        )
        loaded["group_returns"] = _read_csv(quantile_path)

    for key, fallback_name in _KNOWN_JSON_ARTIFACTS.items():
        path = _resolve_artifact_path(
            key=key,
            fallback_name=fallback_name,
            case_dir=case_dir,
            artifact_paths=artifact_paths,
        )
        loaded[key] = _read_json_object(path)

    return loaded


def _build_cumulative_long_short_nav_chart(
    *,
    artifacts: Mapping[str, object],
) -> dict[str, object] | None:
    backtest = _as_mapping(artifacts.get("backtest_result_json"))
    summary = _as_mapping(backtest.get("summary"))
    nav_points = summary.get("nav_points")
    parsed = _parse_nav_points(nav_points)
    if parsed:
        return {
            "title": "Cumulative Long-Short NAV",
            "type": "line",
            "series": [{"name": "nav", "points": parsed}],
        }

    group_df = _as_df(artifacts.get("group_returns"))
    long_short = _long_short_series(group_df)
    if not long_short:
        return None
    nav = _series_to_cumulative(long_short)
    points = [[date, value] for date, value in nav]
    return {
        "title": "Cumulative Long-Short NAV",
        "type": "line",
        "series": [{"name": "nav", "points": points}],
    }


def _build_ic_timeseries_with_cumulative_chart(
    *,
    artifacts: Mapping[str, object],
) -> dict[str, object] | None:
    ic_df = _as_df(artifacts.get("ic_timeseries"))
    if ic_df.empty:
        return None
    date_col = _pick_col(ic_df, "date", "trade_date", "trading_date", "dt")
    if date_col is None:
        return None

    value_col = None
    points: list[list[object]] = []
    for candidate in ("rank_ic", "ic"):
        candidate_col = _pick_col(ic_df, candidate)
        if candidate_col is None:
            continue
        candidate_points = _time_series_points(
            ic_df,
            date_col=date_col,
            value_col=candidate_col,
        )
        if candidate_points:
            value_col = candidate_col
            points = candidate_points
            break
    if not points:
        return None

    cumulative: list[list[object]] = []
    acc = 0.0
    for date, value in points:
        numeric = to_finite_float(value)
        if numeric is None:
            continue
        acc += numeric
        cumulative.append([date, acc])

    return {
        "title": "IC Time Series + Cumulative IC",
        "type": "line",
        "series": [
            {"name": value_col, "points": points},
            {"name": f"cumulative_{value_col}", "points": cumulative},
        ],
    }


def _build_quantile_cumulative_returns_chart(
    *,
    artifacts: Mapping[str, object],
    metrics: Mapping[str, object],
) -> dict[str, object] | None:
    group_df = _as_df(artifacts.get("group_returns"))
    sample_step = _resolve_group_return_sample_step(artifacts=artifacts, metrics=metrics)
    series = _group_cumulative_series(group_df, sample_step=sample_step)
    if not series:
        return None
    return {
        "title": f"Quantile Cumulative NAV (non-overlapping {sample_step}D)",
        "type": "line",
        "series": series,
    }


def _build_group_mean_return_chart(
    *,
    artifacts: Mapping[str, object],
) -> dict[str, object] | None:
    group_df = _as_df(artifacts.get("group_returns"))
    if group_df.empty:
        return None
    group_col = _pick_col(group_df, "group", "quantile", "bucket", "group_id")
    value_col = _pick_col(group_df, "group_return", "mean_return", "return")
    if group_col is None or value_col is None:
        return None

    grouped = defaultdict(list)
    for _, row in group_df.iterrows():
        group = _normalize_group_token(row.get(group_col))
        value = to_finite_float(row.get(value_col))
        if group is None or value is None:
            continue
        grouped[group].append(value)
    bars = []
    for group in sorted(grouped.keys(), key=_group_sort_key):
        values = grouped[group]
        if not values:
            continue
        bars.append({"group": group, "value": float(sum(values) / len(values))})
    if not bars:
        return None
    return {
        "title": "Group Mean Return",
        "type": "bar",
        "series": [{"name": "mean_group_return", "bars": bars}],
    }


def _build_rolling_ic_rankic_chart(
    *,
    artifacts: Mapping[str, object],
) -> dict[str, object] | None:
    rolling_df = _as_df(artifacts.get("rolling_stability"))
    if rolling_df.empty:
        return None
    date_col = _pick_col(rolling_df, "date", "trade_date", "trading_date", "dt")
    ic_col = _pick_col(rolling_df, "rolling_mean_ic", "rolling_ic", "mean_ic")
    rank_col = _pick_col(
        rolling_df,
        "rolling_mean_rank_ic",
        "rolling_rank_ic",
        "mean_rank_ic",
    )
    if date_col is None:
        return None

    series: list[dict[str, object]] = []
    if ic_col is not None:
        points = _time_series_points(rolling_df, date_col=date_col, value_col=ic_col)
        if points:
            series.append({"name": "rolling_ic", "points": points})
    if rank_col is not None:
        points = _time_series_points(rolling_df, date_col=date_col, value_col=rank_col)
        if points:
            series.append({"name": "rolling_rank_ic", "points": points})
    if not series:
        return None

    return {
        "title": "Rolling IC / RankIC",
        "type": "line",
        "series": series,
    }


def _build_ic_decay_chart(
    *,
    artifacts: Mapping[str, object],
) -> dict[str, object] | None:
    decay_df = _as_df(artifacts.get("ic_decay"))
    if decay_df.empty:
        return None
    lag_col = _pick_col(decay_df, "horizon", "lag")
    if lag_col is None:
        return None

    series: list[dict[str, object]] = []
    for col_name, label in (("mean_ic", "mean_ic"), ("mean_rank_ic", "mean_rank_ic")):
        col = _pick_col(decay_df, col_name)
        if col is None:
            continue
        points = []
        for _, row in decay_df.iterrows():
            lag = to_finite_float(row.get(lag_col))
            value = to_finite_float(row.get(col))
            if lag is None or value is None:
                continue
            points.append([lag, value])
        if points:
            series.append({"name": label, "points": points})
    if not series:
        return None

    return {
        "title": "IC Decay",
        "type": "line",
        "series": series,
    }


def _build_ic_distribution_chart(
    *,
    artifacts: Mapping[str, object],
) -> dict[str, object] | None:
    ic_df = _as_df(artifacts.get("ic_timeseries"))
    if ic_df.empty:
        return None

    value_col = None
    finite: list[float] = []
    for candidate in ("rank_ic", "ic"):
        candidate_col = _pick_col(ic_df, candidate)
        if candidate_col is None:
            continue
        values = [to_finite_float(item) for item in ic_df[candidate_col].tolist()]
        candidate_finite = [item for item in values if item is not None]
        if len(candidate_finite) >= 3:
            value_col = candidate_col
            finite = candidate_finite
            break
    if not finite:
        return None

    bins = _histogram(finite, n_bins=12)
    if not bins:
        return None
    return {
        "title": "IC Distribution",
        "type": "histogram",
        "series": [{"name": value_col, "bins": bins}],
    }


def _build_turnover_timeseries_chart(
    *,
    artifacts: Mapping[str, object],
) -> dict[str, object] | None:
    turnover_df = _as_df(artifacts.get("turnover"))
    if turnover_df.empty:
        return None
    date_col = _pick_col(turnover_df, "date", "trade_date", "trading_date", "dt")
    value_col = _pick_col(turnover_df, "turnover", "long_short_turnover")
    if date_col is None or value_col is None:
        return None
    points = _time_series_points(turnover_df, date_col=date_col, value_col=value_col)
    if not points:
        return None
    return {
        "title": "Turnover Time Series",
        "type": "line",
        "series": [{"name": "turnover", "points": points}],
    }


def _build_coverage_timeseries_chart(
    *,
    artifacts: Mapping[str, object],
) -> dict[str, object] | None:
    coverage_df = _as_df(artifacts.get("coverage"))
    if coverage_df.empty:
        return None
    date_col = _pick_col(coverage_df, "date", "trade_date", "trading_date", "dt")
    value_col = _pick_col(coverage_df, "coverage")
    if date_col is None or value_col is None:
        return None
    points = _time_series_points(coverage_df, date_col=date_col, value_col=value_col)
    if not points:
        return None
    return {
        "title": "Coverage Time Series",
        "type": "line",
        "series": [{"name": "coverage", "points": points}],
    }


def _build_yearly_breakdown_rows(*, artifacts: Mapping[str, object]) -> list[dict[str, object]]:
    group_df = _as_df(artifacts.get("group_returns"))
    long_short = _long_short_series(group_df)
    if not long_short:
        return []
    by_year: dict[str, list[float]] = defaultdict(list)
    for date, value in long_short:
        year = date[:4]
        by_year[year].append(value)
    rows: list[dict[str, object]] = []
    for year in sorted(by_year.keys()):
        values = by_year[year]
        if not values:
            continue
        rows.append(
            {
                "year": year,
                "mean_long_short_return": float(sum(values) / len(values)),
            }
        )
    return rows


def _build_ic_decay_table_rows(*, artifacts: Mapping[str, object]) -> list[dict[str, object]]:
    decay_df = _as_df(artifacts.get("ic_decay"))
    if decay_df.empty:
        return []
    lag_col = _pick_col(decay_df, "horizon", "lag")
    mean_ic_col = _pick_col(decay_df, "mean_ic")
    mean_rank_col = _pick_col(decay_df, "mean_rank_ic")
    if lag_col is None:
        return []
    rows: list[dict[str, object]] = []
    for _, row in decay_df.iterrows():
        lag = to_finite_float(row.get(lag_col))
        if lag is None:
            continue
        rows.append(
            {
                "lag": lag,
                "mean_ic": (
                    to_finite_float(row.get(mean_ic_col)) if mean_ic_col is not None else None
                ),
                "mean_rank_ic": (
                    to_finite_float(row.get(mean_rank_col)) if mean_rank_col is not None else None
                ),
            }
        )
    return rows


def _resolve_sample_window(
    *,
    ic_df: pd.DataFrame,
    split_description: str | None,
) -> str | None:
    date_col = _pick_col(ic_df, "date", "trade_date", "trading_date", "dt")
    if date_col is not None:
        dates = [_normalize_date_token(item) for item in ic_df[date_col].tolist()]
        finite = [item for item in dates if item is not None]
        if finite:
            return f"{min(finite)} -> {max(finite)}"
    return split_description


def _resolve_split_contract(
    *,
    metrics: Mapping[str, object],
    meta: Mapping[str, object],
) -> Mapping[str, object]:
    raw = metrics.get("split_contract")
    if isinstance(raw, Mapping):
        return raw
    raw = meta.get("split_contract")
    if isinstance(raw, Mapping):
        return raw
    return {}


def _format_split_contract_description(contract: Mapping[str, object]) -> str | None:
    if not contract:
        return None
    is_start = safe_text(contract.get("is_start"))
    is_end = safe_text(contract.get("is_end"))
    oos_start = safe_text(contract.get("oos_start"))
    oos_end = safe_text(contract.get("oos_end"))
    if not (is_start and is_end and oos_start and oos_end):
        return None
    embargo = safe_text(contract.get("embargo_days"))
    suffix = f" / embargo={embargo}" if embargo else ""
    return f"IS {is_start} -> {is_end} / OOS {oos_start} -> {oos_end}{suffix}"


def _resolve_rebalance_step(value: object) -> int:
    numeric = to_finite_float(value)
    if numeric is not None and numeric > 0:
        return max(1, int(numeric))
    text = str(value or "").strip().lower()
    if not text or text in {"d", "daily", "1d"}:
        return 1
    if "week" in text or text == "w":
        return 5
    if "month" in text or text == "m":
        return 21
    if "quarter" in text or text == "q":
        return 63
    match = re.search(r"(\d+)", text)
    return max(1, int(match.group(1))) if match else 1


def _resolve_group_return_sample_step(
    *,
    artifacts: Mapping[str, object],
    metrics: Mapping[str, object],
) -> int:
    backtest = _as_mapping(artifacts.get("backtest_result_json"))
    summary = _as_mapping(backtest.get("summary"))
    horizon_candidates = [
        backtest.get("target_horizon"),
        summary.get("target_horizon"),
        summary.get("label_horizon"),
        metrics.get("target_horizon"),
    ]
    horizon = 1
    for raw in horizon_candidates:
        value = to_finite_float(raw)
        if value is not None and value > 0:
            horizon = max(1, int(value))
            break
    rebalance_step = _resolve_rebalance_step(
        backtest.get("rebalance_frequency")
        or summary.get("rebalance_frequency")
        or metrics.get("rebalance_frequency")
    )
    return max(1, horizon, rebalance_step)


def _group_cumulative_series(
    group_df: pd.DataFrame,
    *,
    sample_step: int = 1,
) -> list[dict[str, object]]:
    if group_df.empty:
        return []
    date_col = _pick_col(group_df, "date", "trade_date", "trading_date", "dt")
    group_col = _pick_col(group_df, "group", "quantile", "bucket", "group_id")
    value_col = _pick_col(group_df, "group_return", "mean_return", "return")
    if date_col is None or group_col is None or value_col is None:
        return []

    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for _, row in group_df.iterrows():
        date = _normalize_date_token(row.get(date_col))
        group = _normalize_group_token(row.get(group_col))
        value = to_finite_float(row.get(value_col))
        if date is None or group is None or value is None:
            continue
        grouped[group].append((date, value))

    series: list[dict[str, object]] = []
    step = max(1, int(sample_step))
    for group in sorted(grouped.keys(), key=_group_sort_key):
        rows = sorted(grouped[group], key=lambda item: item[0])
        acc = 1.0
        points: list[list[object]] = []
        for idx, (date, value) in enumerate(rows):
            if idx % step != 0:
                continue
            acc *= 1.0 + value
            points.append([date, acc])
        if points:
            series.append({"name": group, "points": points})
    return series


def _long_short_series(group_df: pd.DataFrame) -> list[tuple[str, float]]:
    if group_df.empty:
        return []
    date_col = _pick_col(group_df, "date", "trade_date", "trading_date", "dt")
    group_col = _pick_col(group_df, "group", "quantile", "bucket", "group_id")
    value_col = _pick_col(group_df, "group_return", "mean_return", "return")
    if date_col is None or group_col is None or value_col is None:
        return []

    per_date: dict[str, dict[float, float]] = defaultdict(dict)
    for _, row in group_df.iterrows():
        date = _normalize_date_token(row.get(date_col))
        group_num = _normalize_group_number(row.get(group_col))
        value = to_finite_float(row.get(value_col))
        if date is None or group_num is None or value is None:
            continue
        per_date[date][group_num] = value

    out: list[tuple[str, float]] = []
    for date in sorted(per_date.keys()):
        bucket = per_date[date]
        if len(bucket) < 2:
            continue
        top = max(bucket.keys())
        bottom = min(bucket.keys())
        out.append((date, bucket[top] - bucket[bottom]))
    return out


def _series_to_cumulative(series: list[tuple[str, float]]) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    acc = 1.0
    for date, value in sorted(series, key=lambda item: item[0]):
        acc *= 1.0 + value
        out.append((date, acc))
    return out


def _time_series_points(
    frame: pd.DataFrame,
    *,
    date_col: str,
    value_col: str,
) -> list[list[object]]:
    points: list[list[object]] = []
    for _, row in frame.iterrows():
        date = _normalize_date_token(row.get(date_col))
        value = to_finite_float(row.get(value_col))
        if date is None or value is None:
            continue
        points.append([date, value])
    return points


def _histogram(values: list[float], *, n_bins: int) -> list[dict[str, object]]:
    if not values:
        return []
    v_min = min(values)
    v_max = max(values)
    if not math.isfinite(v_min) or not math.isfinite(v_max):
        return []
    if v_min == v_max:
        width = max(abs(v_min) * 0.01, 1e-6)
        return [
            {
                "left": v_min - width / 2.0,
                "right": v_min + width / 2.0,
                "count": len(values),
            }
        ]

    width = (v_max - v_min) / float(n_bins)
    counts = [0] * n_bins
    for value in values:
        idx = int((value - v_min) / width)
        if idx >= n_bins:
            idx = n_bins - 1
        if idx < 0:
            idx = 0
        counts[idx] += 1

    out: list[dict[str, object]] = []
    for idx, count in enumerate(counts):
        left = v_min + idx * width
        right = left + width
        out.append({"left": left, "right": right, "count": count})
    return out


def _parse_nav_points(value: object) -> list[list[object]]:
    if not isinstance(value, list):
        return []
    out: list[list[object]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        date = _normalize_date_token(item[0])
        nav = to_finite_float(item[1])
        if date is None or nav is None:
            continue
        out.append([date, nav])
    return out


def _artifact_present(value: object) -> bool:
    if isinstance(value, pd.DataFrame):
        return not value.empty
    if isinstance(value, Mapping):
        return bool(value)
    if isinstance(value, list):
        return bool(value)
    return value is not None


def _read_csv(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists() or not path.is_file():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _read_json_object(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists() or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return _as_mapping(payload)


def _extract_metrics(payload: Mapping[str, object]) -> dict[str, object]:
    metrics = payload.get("metrics")
    if isinstance(metrics, Mapping):
        return dict(metrics)
    return dict(payload)


def _resolve_artifact_path(
    *,
    key: str,
    fallback_name: str,
    case_dir: Path,
    artifact_paths: Mapping[str, str | Path],
) -> Path | None:
    raw = artifact_paths.get(key)
    if raw is None:
        raw = artifact_paths.get(f"{key}_path")
    if raw is not None:
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = case_dir / path
        return path.resolve()
    candidate = (case_dir / fallback_name).resolve()
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def _get_nested(payload: Mapping[str, object], dotted_path: str) -> object | None:
    cursor: object = payload
    for token in dotted_path.split("."):
        if not isinstance(cursor, Mapping):
            return None
        cursor = cursor.get(token)
    return cursor


def _pick_col(frame: pd.DataFrame, *candidates: str) -> str | None:
    if frame.empty:
        return None
    cols = {str(col): str(col) for col in frame.columns}
    lowered = {str(col).strip().lower(): str(col) for col in frame.columns}
    for name in candidates:
        if name in cols:
            return cols[name]
        found = lowered.get(name.strip().lower())
        if found is not None:
            return found
    return None


def _normalize_date_token(value: object) -> str | None:
    text = safe_text(value)
    if text is None:
        return None
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    if len(text) >= 10 and text[4] == "/" and text[7] == "/":
        return text[:10].replace("/", "-")
    compact_ymd = re.search(r"(?<!\d)(\d{4})(\d{2})(\d{2})(?!\d)", text)
    if compact_ymd:
        return f"{compact_ymd.group(1)}-{compact_ymd.group(2)}-{compact_ymd.group(3)}"
    compact_ym = re.search(r"(?<!\d)(\d{4})(\d{2})(?!\d)", text)
    if compact_ym:
        return f"{compact_ym.group(1)}-{compact_ym.group(2)}-01"
    compact_year = re.search(r"(?<!\d)(\d{4})(?!\d)", text)
    if compact_year:
        return f"{compact_year.group(1)}-01-01"
    return text


def _normalize_group_number(value: object) -> float | None:
    if isinstance(value, str):
        text = value.strip().upper()
        if text.startswith("Q"):
            return to_finite_float(text[1:])
    return to_finite_float(value)


def _normalize_group_token(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        if text.upper().startswith("Q"):
            num = to_finite_float(text[1:])
            if num is not None and float(int(num)) == num:
                return f"Q{int(num)}"
        num = to_finite_float(text)
        if num is not None and float(int(num)) == num:
            return f"Q{int(num)}"
        return text
    num = to_finite_float(value)
    if num is None:
        return None
    if float(int(num)) == num:
        return f"Q{int(num)}"
    return f"Q{num}"


def _group_sort_key(token: str) -> tuple[int, float, str]:
    text = token.strip().upper()
    if text.startswith("Q"):
        number = to_finite_float(text[1:])
        if number is not None:
            return (0, number, text)
    return (1, float("inf"), text)


def _first_non_empty(values: list[str]) -> str | None:
    for value in values:
        text = safe_text(value)
        if text is not None:
            return text
    return None


def _min_finite(*values: float | None) -> float | None:
    finite = [value for value in values if value is not None]
    return min(finite) if finite else None


def _as_mapping(value: object) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _as_df(value: object) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value
    return pd.DataFrame()
