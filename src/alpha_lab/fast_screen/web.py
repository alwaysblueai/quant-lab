"""HTTP service layer + HTML page for the fast-screen UI.

The main ``web_unified`` server delegates a handful of routes here so this
file can own every URL under ``/fast-screen`` and ``/api/fast-screen`` without
bloating the SPA file.

Conventions
-----------
- Service functions raise ``FileNotFoundError`` / ``ValueError`` for common
  problems; the caller translates these into HTTP 404 / 400.
- The HTML page is self-contained (no build step) and uses vanilla fetch +
  tiny inline SVG charts. Heavy charting libraries are intentionally avoided
  so the page stays <15KB gzipped.
"""
# ruff: noqa: E501

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import (
    load_tier1_result,
    load_tier2_index,
    tier1_dir,
    tier2_module_dir,
)
from .tier1 import Tier1Inputs, run_tier1
from .tier2 import TIER2_MODULES, run_tier2_modules


@dataclass(frozen=True)
class WebConfig:
    """Runtime configuration for the fast-screen web surface."""

    artifact_root: Path


def list_runs(cfg: WebConfig) -> list[dict[str, Any]]:
    """Scan the artifact root and return one summary dict per run.

    Cheap enough to call on every page load: reads only ``tier1/result.json``
    header fields, never chart payloads.
    """
    rows: list[dict[str, Any]] = []
    root = cfg.artifact_root
    if not root.exists():
        return rows
    for factor_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for run_dir in sorted(p for p in factor_dir.iterdir() if p.is_dir()):
            result_path = run_dir / "tier1" / "result.json"
            if not result_path.exists():
                continue
            try:
                data = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            tier2_index = {}
            idx_path = run_dir / "tier2" / "index.json"
            if idx_path.exists():
                try:
                    tier2_index = json.loads(idx_path.read_text(encoding="utf-8")).get(
                        "modules", {}
                    )
                except (OSError, json.JSONDecodeError):
                    tier2_index = {}
            rows.append(
                {
                    "factor_name": data.get("factor_name"),
                    "run_id": data.get("run_id"),
                    "universe": data.get("universe"),
                    "frequency": data.get("frequency"),
                    "verdict": (data.get("verdict") or {}).get("status"),
                    "generated_at": data.get("generated_at"),
                    "tier2_modules_done": sorted(tier2_index.keys()),
                }
            )
    return rows


def latest_run_for_factor(cfg: WebConfig, factor_name: str) -> dict[str, Any] | None:
    """Return ``{factor_name, run_id, generated_at}`` for the newest run, or None.

    Used by the UI to bridge run-store run_ids (per-project) to fast-screen
    run_ids (per-factor) without forcing the two systems to share ids.
    """
    root = cfg.artifact_root
    factor_dir = root / _sanitize(factor_name)
    if not factor_dir.exists():
        return None
    best: tuple[str, str] | None = None  # (generated_at, run_id)
    for run_dir in factor_dir.iterdir():
        if not run_dir.is_dir():
            continue
        result_path = run_dir / "tier1" / "result.json"
        if not result_path.exists():
            continue
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        generated_at = str(data.get("generated_at") or "")
        run_id = str(data.get("run_id") or run_dir.name)
        if best is None or generated_at > best[0]:
            best = (generated_at, run_id)
    if best is None:
        return None
    return {"factor_name": factor_name, "run_id": best[1], "generated_at": best[0]}


def _sanitize(name: str) -> str:
    # Mirror artifacts._sanitize so lookups work for the same factor names.
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("._") or "unnamed"
    return out[:120]


def get_run_result(cfg: WebConfig, factor_name: str, run_id: str) -> dict[str, Any]:
    result = load_tier1_result(cfg.artifact_root, factor_name, run_id)
    tier2_index = load_tier2_index(cfg.artifact_root, factor_name, run_id)
    return {
        "tier1": result.to_dict(),
        "tier2_index": {k: v.to_dict() for k, v in tier2_index.items()},
        "available_modules": [
            {"key": m.key, "label": m.label, "estimated_seconds": m.estimated_seconds}
            for m in TIER2_MODULES
        ],
    }


def get_tier2_module_payload(
    cfg: WebConfig, factor_name: str, run_id: str, module: str
) -> dict[str, Any]:
    path = tier2_module_dir(cfg.artifact_root, factor_name, run_id, module) / "result.json"
    if not path.exists():
        raise FileNotFoundError(f"tier2 module result not found: {module}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"tier2 module result payload must be object: {path}")
    return {str(key): value for key, value in payload.items()}


def backfill_existing_runs(
    cfg: WebConfig,
    records: list[Any],
    *,
    overwrite: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Emit Tier-1 artifacts for prior succeeded single-factor runs.

    ``records`` is an iterable of objects exposing ``.run_id``, ``.status``,
    ``.spec_path``, ``.case_name`` (duck-typed against ``_RunRecord``). Runs
    whose spec cannot be parsed as a single-factor spec are skipped.
    """
    from alpha_lab.real_cases.single_factor.spec import load_single_factor_case_spec

    from .artifacts import save_tier1_result
    from .cli import _prepare_inputs

    succeeded: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for rec in records:
        if getattr(rec, "status", None) != "succeeded":
            skipped.append({"run_id": getattr(rec, "run_id", None), "reason": "not_succeeded"})
            continue
        run_id = str(getattr(rec, "run_id", ""))
        spec_path = str(getattr(rec, "spec_path", ""))
        if not spec_path:
            skipped.append({"run_id": run_id, "reason": "no_spec_path"})
            continue
        try:
            spec = load_single_factor_case_spec(Path(spec_path).resolve())
        except Exception as exc:
            skipped.append({"run_id": run_id, "reason": f"spec_load_failed: {exc}"})
            continue
        try:
            existing = tier1_dir(cfg.artifact_root, spec.factor_name, run_id) / "result.json"
            if existing.exists() and not overwrite:
                skipped.append({"run_id": run_id, "reason": "already_exists"})
                continue
            inputs = _prepare_inputs(spec)
            result = run_tier1(inputs, run_id=run_id)
            save_tier1_result(cfg.artifact_root, result)
            succeeded.append(
                {
                    "run_id": run_id,
                    "factor_name": result.factor_name,
                    "verdict": result.verdict.status,
                }
            )
        except Exception as exc:
            failed.append({"run_id": run_id, "error": f"{type(exc).__name__}: {exc}"})

    return {"succeeded": succeeded, "skipped": skipped, "failed": failed}


def trigger_deep_dive(
    cfg: WebConfig,
    *,
    factor_name: str,
    run_id: str,
    modules: list[str],
    inputs: Tier1Inputs,
    inputs_hash: str = "",
) -> dict[str, Any]:
    """Synchronously run Tier-2 modules. A real deployment would queue these.

    Returns the per-module status map. For now synchronous is acceptable
    because the web layer is single-user and the modules are bounded.
    """
    statuses = run_tier2_modules(
        inputs,
        artifact_root=cfg.artifact_root,
        factor_name=factor_name,
        run_id=run_id,
        modules=modules,
        inputs_hash=inputs_hash,
    )
    return {k: v.to_dict() for k, v in statuses.items()}


def page_html() -> str:
    """Serve the minimal fast-screen SPA. Self-contained HTML + JS + CSS."""
    return _PAGE_HTML


_PAGE_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Fast Screen — Alpha Lab</title>
<style>
  :root {
    --bg: #f5f4ed;
    --panel: #faf9f5;
    --panel-hover: #ede8db;
    --ink: #141413;
    --muted: #5e5d59;
    --line: #d9d5ca;
    --line-soft: #e7e3d6;
    --brand: #c96442;
    --brand-dark: #a84f35;
    --brand-soft: #fce6d6;
    --ok: #4f6b3e;
    --ok-bg: #e8efe0;
    --warn: #a07628;
    --warn-bg: #f5edd0;
    --bad: #8b3a2b;
    --bad-bg: #f4e4df;
    --sub: #87867f;
    --mono: "JetBrains Mono", SFMono-Regular, Menlo, Monaco, Consolas, "Courier New", monospace;
  }
  * { box-sizing: border-box; }
  body { margin: 0; font-family: "MiSans", "PingFang SC", "Microsoft YaHei UI", system-ui, sans-serif;
         background: var(--bg); color: var(--ink); line-height: 1.6; font-size: 14px; }
  .wrap { max-width: 1280px; margin: 0 auto; padding: 24px; }
  h1 { font-size: 20px; margin: 0 0 4px 0; }
  .muted { color: var(--muted); font-size: 13px; }
  .toolbar { margin: 12px 0; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }
  .row { display: grid; gap: 12px; }
  .row-metrics { grid-template-columns: repeat(9, minmax(0, 1fr)); }
  @media (max-width: 1100px) { .row-metrics { grid-template-columns: repeat(3, 1fr); } }
  .row-charts { grid-template-columns: repeat(2, 1fr); margin-top: 16px; }
  @media (max-width: 900px) { .row-charts { grid-template-columns: 1fr; } }
  .card { background: var(--panel); border: 1px solid var(--line); border-radius: 12px; padding: 12px; box-shadow: 0 0 0 1px #f0eee6; }
  .metric { display: flex; flex-direction: column; gap: 2px; }
  .metric .label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
  .metric .value { font-size: 20px; font-weight: 700; font-variant-numeric: tabular-nums; }
  .metric .secondary { font-size: 11px; color: var(--muted); }
  .metric.partial .value { color: var(--warn); }
  .metric.na .value, .metric.skipped .value, .metric.missing .value { color: var(--sub); font-weight: 500; font-size: 13px; }
  .badge { display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 999px; text-transform: uppercase; letter-spacing: .04em; font-weight: 700; }
  .badge.pass, .badge.computed { background: var(--ok-bg); color: var(--ok); }
  .badge.warn, .badge.partial, .badge.missing { background: var(--warn-bg); color: var(--warn); }
  .badge.fail, .badge.failed { background: var(--bad-bg); color: var(--bad); }
  .badge.skipped, .badge.na, .badge.notcomputed { background: var(--panel-hover); color: var(--muted); }
  .chart { height: 180px; }
  .chart-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
  .factor-title { font-weight: 700; font-size: 15px; }
  .section-title { margin: 24px 0 8px 0; font-size: 14px; letter-spacing: .03em; text-transform: uppercase; color: var(--muted); }
  .panel-status { margin-left: auto; }
  header.bar { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 12px 16px; border: 1px solid var(--line); border-radius: 12px; background: var(--panel); margin-bottom: 16px; }
  .verdict { padding: 12px 16px; border-radius: 12px; background: var(--panel); border: 1px solid var(--line); margin-top: 16px; }
  .verdict.pass { border-color: var(--ok); }
  .verdict.warn { border-color: var(--warn); }
  .verdict.fail { border-color: var(--bad); }
  .verdict-rules { margin: 6px 0 0 16px; padding: 0; }
  details.panel { background: var(--panel); border: 1px solid var(--line); border-radius: 10px; padding: 10px 14px; margin-bottom: 8px; }
  details.panel summary { cursor: pointer; display: flex; align-items: center; gap: 8px; font-weight: 600; }
  button {
    font: inherit;
    border: 1px solid var(--line);
    background: var(--bg);
    color: var(--ink);
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-weight: 600;
    font-size: 14px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.6);
    transition: all 0.2s;
  }
  button:hover { background: var(--panel-hover); border-color: #d1cfc5; }
  button:active { transform: translateY(1px); box-shadow: none; }
  button.action {
    border-color: var(--brand-dark);
    background: var(--brand);
    color: #ffffff;
    box-shadow: 0 0 0 1px #d9d5ca, inset 0 1px 0 rgba(255,255,255,0.15);
  }
  button.action:hover { background: var(--brand-dark); }
  button.ghost { background: var(--bg); color: var(--ink); border: 1px solid var(--line); }
  button.small { padding: 5px 10px; font-size: 13px; }
  button.accent { color: var(--brand); }
  button.accent:hover { background: var(--brand-soft); border-color: #d1cfc5; }
  button:disabled { opacity: .5; cursor: not-allowed; }
  .queue { border: 1px solid var(--line); border-radius: 12px; background: var(--panel); margin-bottom: 24px; overflow: hidden; }
  .queue table { width: 100%; border-collapse: collapse; font-size: 13px; }
  .queue th, .queue td { padding: 10px 12px; text-align: left; border-bottom: 1px solid var(--line-soft); vertical-align: middle; }
  .queue th { background: var(--bg); font-weight: 600; color: var(--muted); text-transform: uppercase; font-size: 11px; letter-spacing: .08em; }
  .queue tr.selected { background: var(--brand-soft); }
  .queue tr:hover { background: var(--panel-hover); cursor: pointer; }
  .queue code { font-family: var(--mono); font-size: 12px; }
  .t2grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; }
  @media (max-width: 900px) { .t2grid { grid-template-columns: 1fr; } }
  .note { font-size: 11px; color: var(--muted); margin-top: 2px; }
  dialog { border: 1px solid var(--line); border-radius: 12px; padding: 20px; box-shadow: 0 10px 25px rgb(0 0 0 / .2); min-width: 360px; background: var(--panel); color: var(--ink); }
  dialog::backdrop { background: rgb(20 20 19 / .35); }
  label.modbox { display: flex; align-items: center; gap: 8px; padding: 8px; border: 1px solid var(--line); border-radius: 8px; margin-bottom: 6px; cursor: pointer; background: var(--bg); }
  label.modbox:hover { background: var(--panel-hover); border-color: #d1cfc5; }
  .modbox-title { font-weight: 600; }
  .modbox-time { font-size: 11px; }
  svg.spark { width: 100%; height: 100%; }
  .spark .axis { stroke: var(--line); stroke-width: 1; }
  .spark .line { fill: none; stroke: var(--brand); stroke-width: 1.5; }
  .spark .bar { fill: var(--brand); }
</style>
</head>
<body>
<div class="wrap">
  <h1>Fast Screen <span class="muted">single-factor quick-screening</span></h1>
  <div class="muted">Tier-1 metrics &amp; verdict. Deep-dive modules are on-demand.</div>

  <div class="toolbar">
    <button id="backfillBtn" class="ghost small accent">Backfill prior runs</button>
    <span class="muted" id="backfillMsg"></span>
  </div>

  <div class="queue" id="queue"></div>
  <div id="detail"></div>
</div>

<dialog id="ddDialog">
  <form method="dialog" id="ddForm">
    <h3 style="margin-top:0">Deep Dive</h3>
    <div class="muted" id="ddSubtitle"></div>
    <div id="ddMods" style="margin: 12px 0"></div>
    <div style="display:flex; justify-content: flex-end; gap: 8px;">
      <button value="cancel" class="ghost small">Cancel</button>
      <button value="ok" class="action small" id="ddRun">Run</button>
    </div>
  </form>
</dialog>

<script>
const api = {
  runs: () => fetch('/api/fast-screen/runs').then(r => r.json()),
  run: (factor, id) => fetch(`/api/fast-screen/runs/${encodeURIComponent(factor)}/${encodeURIComponent(id)}`).then(r => r.json()),
  deepDive: (factor, id, modules, spec) => fetch(
    `/api/fast-screen/runs/${encodeURIComponent(factor)}/${encodeURIComponent(id)}/deep-dive`,
    { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ modules, spec }) }
  ).then(r => r.json()),
};

let state = { selected: null };

function fmtNum(v, unit) {
  if (v === null || v === undefined || Number.isNaN(v)) return '—';
  const abs = Math.abs(v);
  let s;
  if (unit === 'ratio' || abs < 1) s = (v * 100).toFixed(2) + '%';
  else if (abs >= 1000) s = v.toFixed(0);
  else s = v.toFixed(3);
  return s;
}
function metricClass(m) {
  const st = m.status;
  if (st === 'computed') return 'metric';
  if (st === 'partial') return 'metric partial';
  if (st === 'missing_input') return 'metric missing';
  if (st === 'not_applicable') return 'metric na';
  if (st === 'skipped') return 'metric skipped';
  return 'metric';
}
function metricDisplay(m) {
  if (m.status === 'computed' || m.status === 'partial') return fmtNum(m.value, m.unit);
  if (m.status === 'skipped') return 'skipped';
  if (m.status === 'not_applicable') return 'n/a';
  if (m.status === 'missing_input') return 'missing input';
  return '—';
}
function secondaryText(m) {
  const parts = [];
  for (const [k, v] of Object.entries(m.secondary || {})) {
    if (v === null || v === undefined) continue;
    if (typeof v === 'number') parts.push(`${k}: ${fmtNum(v, '')}`);
    else parts.push(`${k}: ${v}`);
  }
  return parts.join(' · ');
}

function sparkline(svg, xs, ys, kind) {
  const w = 320, h = 160, pad = 20;
  const data = ys.map((y, i) => ({ x: i, y })).filter(d => typeof d.y === 'number' && isFinite(d.y));
  if (!data.length) { svg.innerHTML = `<text x="${w/2}" y="${h/2}" text-anchor="middle" fill="#94a3b8" font-size="12">no data</text>`; svg.setAttribute('viewBox', `0 0 ${w} ${h}`); return; }
  const minY = Math.min(...data.map(d => d.y));
  const maxY = Math.max(...data.map(d => d.y));
  const span = (maxY - minY) || 1;
  const xScale = i => pad + (i / Math.max(1, data.length - 1)) * (w - 2 * pad);
  const yScale = y => h - pad - ((y - minY) / span) * (h - 2 * pad);
  let inner = '';
  const zeroY = yScale(0);
  if (minY < 0 && maxY > 0) inner += `<line class="axis" x1="${pad}" y1="${zeroY}" x2="${w-pad}" y2="${zeroY}"/>`;
  if (kind === 'bar') {
    const bw = Math.max(4, (w - 2 * pad) / data.length - 4);
    inner += data.map((d, i) => {
      const x = xScale(i) - bw / 2;
      const y0 = yScale(0), y1 = yScale(d.y);
      const yy = Math.min(y0, y1); const hh = Math.abs(y1 - y0);
      return `<rect class="bar" x="${x}" y="${yy}" width="${bw}" height="${hh || 1}" />`;
    }).join('');
    inner += data.map((d, i) => `<text x="${xScale(i)}" y="${h - 4}" text-anchor="middle" font-size="10" fill="#94a3b8">${xs[i]}</text>`).join('');
  } else {
    const d = data.map((p, i) => `${i === 0 ? 'M' : 'L'}${xScale(i)},${yScale(p.y)}`).join(' ');
    inner += `<path class="line" d="${d}"/>`;
  }
  svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
  svg.innerHTML = inner;
}

async function renderQueue() {
  const data = await api.runs();
  const rows = (data.runs || []).slice().reverse();
  const q = document.getElementById('queue');
  if (!rows.length) {
    q.innerHTML = `<div class="card muted">No runs yet. Run <code>alpha-lab fast-screen run --spec ...</code> to create one.</div>`;
    return;
  }
  q.innerHTML = `<table><thead><tr>
    <th>Factor</th><th>Run</th><th>Universe</th><th>Verdict</th>
    <th>Tier-2 done</th><th>Generated</th></tr></thead>
    <tbody>${rows.map(r => `
      <tr data-factor="${r.factor_name}" data-run="${r.run_id}"
          class="${state.selected && state.selected.factor === r.factor_name && state.selected.run === r.run_id ? 'selected' : ''}">
        <td><strong>${r.factor_name}</strong></td>
        <td><code>${r.run_id}</code></td>
        <td>${r.universe || '—'}</td>
        <td><span class="badge ${r.verdict}">${r.verdict || '—'}</span></td>
        <td class="muted">${(r.tier2_modules_done || []).join(', ') || '—'}</td>
        <td class="muted">${r.generated_at || ''}</td>
      </tr>`).join('')}</tbody></table>`;
  q.querySelectorAll('tr[data-factor]').forEach(tr => {
    tr.onclick = () => selectRun(tr.dataset.factor, tr.dataset.run);
  });
  if (!state.selected && rows.length) selectRun(rows[0].factor_name, rows[0].run_id);
}

async function selectRun(factor, run) {
  state.selected = { factor, run };
  await renderQueue();
  const { tier1, tier2_index, available_modules } = await api.run(factor, run);
  renderDetail(tier1, tier2_index, available_modules);
}

function renderDetail(tier1, tier2_index, available_modules) {
  const d = document.getElementById('detail');
  const v = tier1.verdict || {};
  d.innerHTML = `
    <header class="bar">
      <div>
        <div class="factor-title">${tier1.factor_name}</div>
        <div class="muted">${tier1.universe} · ${tier1.frequency} · ${tier1.window.start} → ${tier1.window.end}</div>
      </div>
      <div><span class="badge ${v.status}">${v.status}</span></div>
    </header>
    <div class="row row-metrics" id="metricRow"></div>
    <div class="row row-charts" id="chartRow"></div>
    <div class="verdict ${v.status}">
      <strong>${v.status.toUpperCase()}</strong> &middot; ${v.next_step || ''}
      ${v.triggered_rules && v.triggered_rules.length ? `<ul class="verdict-rules">${v.triggered_rules.map(r => `<li>${r}</li>`).join('')}</ul>` : ''}
    </div>
    <h3 class="section-title">Deep Dive (Tier-2)</h3>
    <div class="t2grid" id="t2grid"></div>
  `;
  const mr = d.querySelector('#metricRow');
  mr.innerHTML = tier1.metrics.map(m => `
    <div class="card ${metricClass(m)}">
      <div class="label">${m.label}</div>
      <div class="value">${metricDisplay(m)}</div>
      <div class="secondary">${secondaryText(m) || '&nbsp;'}</div>
      ${m.note ? `<div class="note">${m.note}</div>` : ''}
    </div>`).join('');
  const cr = d.querySelector('#chartRow');
  cr.innerHTML = tier1.charts.map(c => `
    <div class="card">
      <div class="chart-label">${c.label}</div>
      <div class="chart"><svg class="spark" id="sv-${c.key}"></svg></div>
    </div>`).join('');
  tier1.charts.forEach(c => {
    const svg = document.getElementById(`sv-${c.key}`);
    if (svg) sparkline(svg, c.x, c.y, c.kind);
  });
  renderTier2Panels(tier2_index, available_modules, tier1);
}

function renderTier2Panels(tier2_index, modules, tier1) {
  const box = document.getElementById('t2grid');
  box.innerHTML = modules.map(m => {
    const st = tier2_index[m.key];
    const status = st ? st.status : 'not_computed';
    const label = status === 'computed' ? `computed ${(st.computed_at || '').slice(0, 16)}`
                : status === 'failed' ? `failed` : status;
    const canRun = (tier1.verdict.status !== 'fail');
    return `<details class="panel" data-module="${m.key}">
      <summary>
        <span>${m.label}</span>
        <span class="badge ${status.replace('_', '')} panel-status">${label}</span>
      </summary>
      <div style="margin-top:8px">
        <button class="ghost small accent" data-run-module="${m.key}" ${!canRun && status !== 'computed' ? 'disabled title="Tier-1 FAIL — cannot run"' : ''}>
          ${status === 'computed' ? 'Re-run' : 'Run'}
        </button>
        ${status === 'failed' ? `<div class="note">${(st && st.message || '').slice(0, 400)}</div>` : ''}
      </div>
    </details>`;
  }).join('');
  box.querySelectorAll('button[data-run-module]').forEach(btn => {
    btn.onclick = () => openDeepDive([btn.dataset.runModule], modules);
  });
}

function openDeepDive(preselected, modules) {
  const dlg = document.getElementById('ddDialog');
  const sub = document.getElementById('ddSubtitle');
  sub.textContent = `${state.selected.factor} · ${state.selected.run}`;
  const host = document.getElementById('ddMods');
  host.innerHTML = modules.map(m => `
    <label class="modbox">
      <input type="checkbox" name="mod" value="${m.key}" ${preselected.includes(m.key) ? 'checked' : ''} />
      <div>
        <div class="modbox-title">${m.label}</div>
        <div class="muted modbox-time">~${m.estimated_seconds}s</div>
      </div>
    </label>`).join('');
  dlg.returnValue = '';
  dlg.showModal();
  dlg.onclose = async () => {
    if (dlg.returnValue !== 'ok') return;
    const picked = Array.from(host.querySelectorAll('input[name=mod]:checked')).map(i => i.value);
    if (!picked.length) return;
    const spec = window.prompt('Spec path for this factor (required to reconstruct inputs):');
    if (!spec) return;
    const btn = document.getElementById('ddRun'); btn.disabled = true;
    try {
      await api.deepDive(state.selected.factor, state.selected.run, picked, spec);
      await selectRun(state.selected.factor, state.selected.run);
    } finally { btn.disabled = false; }
  };
}

document.getElementById('backfillBtn').onclick = async () => {
  const btn = document.getElementById('backfillBtn');
  const msg = document.getElementById('backfillMsg');
  btn.disabled = true; msg.textContent = 'Backfilling…';
  try {
    const r = await fetch('/api/fast-screen/backfill', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}'
    });
    const data = await r.json();
    const s = (data.succeeded || []).length;
    const k = (data.skipped || []).length;
    const f = (data.failed || []).length;
    msg.textContent = `Done · ${s} added · ${k} skipped · ${f} failed`;
    await renderQueue();
  } catch (e) {
    msg.textContent = `Error: ${e}`;
  } finally { btn.disabled = false; }
};

renderQueue();
</script>
</body>
</html>"""
