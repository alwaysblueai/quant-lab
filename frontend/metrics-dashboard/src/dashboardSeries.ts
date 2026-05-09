export type SplitPhase = "IS" | "OOS" | "EMBARGO";

export type TimedValue = {
  date: string;
  value: number;
  sourceColumn: string;
  splitPhase?: SplitPhase;
};

export type SplitSummary = { overall: number | null; split: number | null };

export type SplitSeriesRow = {
  date: string;
  rank_ic?: number;
  rank_ic_is?: number;
  rank_ic_oos?: number;
};

export function aggregateGroupMeans(rows: Record<string, string>[]): { group: string; mean: number }[] {
  const sums = new Map<number, { s: number; n: number }>();
  for (const r of rows) {
    const g = parseFiniteCell(r.group);
    const v = parseFiniteCell(r.group_return);
    if (g === null || v === null) continue;
    const cur = sums.get(g) ?? { s: 0, n: 0 };
    cur.s += v;
    cur.n += 1;
    sums.set(g, cur);
  }
  return Array.from(sums.entries())
    .sort((a, b) => a[0] - b[0])
    .map(([g, { s, n }]) => ({ group: `Q${g}`, mean: s / n }));
}

export function parseFlags(v: unknown): string[] | null {
  if (Array.isArray(v)) return v.map((x) => String(x)).filter(Boolean);
  if (typeof v === "string" && v.trim()) {
    try {
      const parsed = JSON.parse(v);
      if (Array.isArray(parsed)) return parsed.map(String);
    } catch {
      /* pass */
    }
    return v.split(/[|;,]/).map((x) => x.trim()).filter(Boolean);
  }
  return null;
}

export function parseSplitStartLegacy(v: unknown): string | null {
  const raw = String(v ?? "").trim();
  if (!raw) return null;
  const match = raw.match(/test\s*[>=：:]\s*(\d{4}-\d{2}-\d{2})/i);
  if (!match) return null;
  return normalizeDate(match[1]);
}

export function parseSplitStart(...values: unknown[]): string | null {
  for (const value of values) {
    const parsed = parseOneSplitStart(value);
    if (parsed) return parsed;
  }
  return null;
}

export function parseOneSplitStart(v: unknown): string | null {
  if (v && typeof v === "object") {
    const record = v as Record<string, unknown>;
    return parseOneSplitStart(record.oos_start ?? record.test_start ?? record.testStart);
  }
  const raw = String(v ?? "").trim();
  if (!raw) return null;
  const directDate = normalizeDate(raw);
  if (directDate) return directDate;
  if (raw.startsWith("{")) {
    try {
      return parseOneSplitStart(JSON.parse(raw) as unknown);
    } catch {
      /* pass */
    }
  }
  const fromLegacy = parseSplitStartLegacy(raw);
  if (fromLegacy) return fromLegacy;
  const m = raw.match(/test\s*>?=?\s*(\d{4}-\d{2}-\d{2})/i);
  if (m) {
    return normalizeDate(m[1]);
  }

  return null;
}

export function parseTimeseriesRows(
  rows: Record<string, string>[] | null,
  valueColumns: readonly string[],
): TimedValue[] {
  if (!rows) return [];
  const selectedColumn = selectTimeseriesColumn(rows, valueColumns);
  if (!selectedColumn) return [];
  const out: TimedValue[] = [];
  for (const row of rows) {
    const date = String(row.date ?? "").slice(0, 10);
    const normalizedDate = normalizeDate(date);
    if (!normalizedDate) continue;
    const value = parseFiniteCell(row[selectedColumn]);
    if (value === null) continue;
    const splitPhase = normalizeSplitPhase(row.split_phase ?? row.phase ?? row.sample_phase);
    out.push({
      date: normalizedDate,
      value,
      sourceColumn: selectedColumn,
      ...(splitPhase ? { splitPhase } : {}),
    });
  }
  return out;
}

export function selectTimeseriesColumn(
  rows: Record<string, string>[],
  valueColumns: readonly string[],
): string | null {
  for (const column of valueColumns) {
    if (rows.some((row) => parseFiniteCell(row[column]) !== null)) {
      return column;
    }
  }
  return null;
}

export function buildSplitSeries(
  rows: TimedValue[],
  splitStart: string | null,
): SplitSeriesRow[] {
  if (!hasSplitBoundary(rows, splitStart)) {
    return rows.map((r) => ({ date: r.date, rank_ic: r.value }));
  }
  const out: SplitSeriesRow[] = [];
  for (const row of rows) {
    const bucket = splitBucket(row, splitStart);
    if (bucket === "EMBARGO") continue;
    out.push(
      bucket === "OOS"
        ? { date: row.date, rank_ic_oos: row.value }
        : { date: row.date, rank_ic_is: row.value },
    );
  }
  return out;
}

export function splitMean(rows: TimedValue[], splitStart: string | null): SplitSummary {
  const overall = mean(nonEmbargoRows(rows).map((r) => r.value));
  if (!hasSplitBoundary(rows, splitStart)) return { overall, split: null };
  return { overall, split: mean(oosRows(rows, splitStart).map((r) => r.value)) };
}

export function splitIr(rows: TimedValue[], splitStart: string | null): SplitSummary {
  return {
    overall: ir(nonEmbargoRows(rows)),
    split: hasSplitBoundary(rows, splitStart) ? ir(oosRows(rows, splitStart)) : null,
  };
}

export function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  const n = values.length;
  return values.reduce((acc, value) => acc + value, 0) / n;
}

export function ir(values: TimedValue[]): number | null;
export function ir(values: number[]): number | null;
export function ir(values: TimedValue[] | number[]): number | null {
  const xs = values.length === 0
    ? []
    : typeof values[0] === "object"
      ? (values as TimedValue[]).map((v) => v.value)
      : values.map((v) => Number(v));
  if (xs.length < 2) return null;
  const m = mean(xs);
  if (m === null) return null;
  const varVal = xs.reduce((acc, value) => acc + Math.pow(value - m, 2), 0) / (xs.length - 1);
  if (!Number.isFinite(varVal) || varVal <= 0) return null;
  return m / Math.sqrt(varVal);
}

export function parseFiniteCell(value: unknown): number | null {
  const raw = String(value ?? "").trim();
  if (!raw || /^nan$/i.test(raw) || /^null$/i.test(raw)) return null;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? parsed : null;
}

export function normalizeSplitPhase(value: unknown): SplitPhase | null {
  const raw = String(value ?? "").trim().toUpperCase();
  if (raw === "IS" || raw === "IN_SAMPLE") return "IS";
  if (raw === "OOS" || raw === "OUT_OF_SAMPLE") return "OOS";
  if (raw === "EMBARGO") return "EMBARGO";
  return null;
}

export function hasSplitBoundary(rows: TimedValue[], splitStart: string | null): boolean {
  return Boolean(splitStart) || rows.some((row) => row.splitPhase === "IS" || row.splitPhase === "OOS");
}

export function splitBucket(row: TimedValue, splitStart: string | null): SplitPhase | "ALL" {
  if (row.splitPhase) return row.splitPhase;
  if (!splitStart) return "ALL";
  return row.date >= splitStart ? "OOS" : "IS";
}

export function nonEmbargoRows(rows: TimedValue[]): TimedValue[] {
  return rows.filter((row) => row.splitPhase !== "EMBARGO");
}

export function oosRows(rows: TimedValue[], splitStart: string | null): TimedValue[] {
  return rows.filter((row) => splitBucket(row, splitStart) === "OOS");
}

export function primaryMetricLabel(rows: TimedValue[]): string {
  const source = rows[0]?.sourceColumn;
  if (source === "rank_ic") return "RankIC";
  if (source === "ic") return "IC";
  return "Metric";
}

export function normalizeDate(value: string): string | null {
  const trimmed = String(value ?? "").trim().slice(0, 10);
  if (!/^\d{4}-\d{2}-\d{2}$/.test(trimmed)) return null;
  return trimmed;
}

export function formatSplitStat(v: number | null, digits: number, isPct = false): string | null {
  if (v === null) return null;
  return isPct ? `${(v * 100).toFixed(digits)}%` : v.toFixed(digits);
}

export function formatMetricWithOos(overall: string, oos: string | null): string {
  return oos ? `${overall} (OOS: ${oos})` : overall;
}
