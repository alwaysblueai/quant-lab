/* Thin data layer — talks to the Python server at /api/projects/:slug/runs/:id.
 * Components themselves remain data-agnostic; swap this module to change source. */

export interface RunPayload {
  run_id: string;
  project_slug?: string;
  case_name?: string;
  factor_name?: string;
  status?: string;
  created_at?: string;
  updated_at?: string;
  summary?: Record<string, unknown>;
  artifacts?: Record<string, string>;
}

export function getContextFromQuery(): { project: string | null; runId: string | null } {
  const p = new URLSearchParams(window.location.search);
  return {
    project: p.get("project"),
    runId: p.get("run_id"),
  };
}

export async function fetchRun(project: string, runId: string): Promise<RunPayload> {
  const url = `/api/projects/${encodeURIComponent(project)}/runs/${encodeURIComponent(runId)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetchRun ${res.status}`);
  return (await res.json()) as RunPayload;
}

export function artifactUrl(project: string, runId: string, key: string): string {
  return `/api/projects/${encodeURIComponent(project)}/runs/${encodeURIComponent(runId)}/artifact/${encodeURIComponent(key)}`;
}

export async function fetchArtifactJson<T = unknown>(
  project: string,
  runId: string,
  key: string,
): Promise<T | null> {
  try {
    const res = await fetch(artifactUrl(project, runId, key));
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

export async function fetchArtifactCsv(
  project: string,
  runId: string,
  key: string,
): Promise<Record<string, string>[] | null> {
  try {
    const res = await fetch(artifactUrl(project, runId, key));
    if (!res.ok) return null;
    const text = await res.text();
    return parseCsv(text);
  } catch {
    return null;
  }
}

function parseCsv(text: string): Record<string, string>[] {
  const lines = text.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = splitCsvLine(lines[0]);
  const rows: Record<string, string>[] = [];
  for (let i = 1; i < lines.length; i++) {
    const cells = splitCsvLine(lines[i]);
    const row: Record<string, string> = {};
    for (let j = 0; j < headers.length; j++) row[headers[j]] = cells[j] ?? "";
    rows.push(row);
  }
  return rows;
}

function splitCsvLine(line: string): string[] {
  const out: string[] = [];
  let cur = "";
  let inQ = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQ) {
      if (c === '"' && line[i + 1] === '"') { cur += '"'; i++; }
      else if (c === '"') { inQ = false; }
      else cur += c;
    } else {
      if (c === '"') inQ = true;
      else if (c === ",") { out.push(cur); cur = ""; }
      else cur += c;
    }
  }
  out.push(cur);
  return out;
}

/* ---------- Formatting helpers ---------- */

export const num = (v: unknown): number | null => {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};
export const fmt = (v: unknown, d = 3): string => {
  const n = num(v);
  return n === null ? "—" : n.toFixed(d);
};
export const fmtPct = (v: unknown, d = 2): string => {
  const n = num(v);
  return n === null ? "—" : `${(n * 100).toFixed(d)}%`;
};
