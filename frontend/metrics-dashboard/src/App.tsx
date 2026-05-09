import { useEffect, useState } from "react";
import {
  DashboardShell,
  ReportHeader,
  SummaryStrip,
  Section,
  MetricGrid,
  DataTable,
  ChartPanel,
  LineSeriesChart,
  BarSeriesChart,
  Callout,
  PressedLeaf,
} from "./components";
import { MetricItem, TableColumn, Tone } from "./types";
import {
  RunPayload,
  fetchArtifactCsv,
  fetchRun,
  fmt,
  fmtPct,
  getContextFromQuery,
  num,
} from "./data";
import {
  aggregateGroupMeans,
  buildSplitSeries,
  formatMetricWithOos,
  formatSplitStat,
  hasSplitBoundary,
  parseFlags,
  parseSplitStart,
  parseTimeseriesRows,
  primaryMetricLabel,
  splitIr,
  splitMean,
} from "./dashboardSeries";

/* ------------------------------------------------------------------ *
 *  App shell: context + data load + compose generic sections.        *
 *  Sections are declarative; swap the builder functions to change      *
 *  which metrics get surfaced without touching layout code.           *
 * ------------------------------------------------------------------ */

export function App() {
  const [ctx] = useState(getContextFromQuery);
  const [run, setRun] = useState<RunPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [ic, setIc] = useState<Record<string, string>[] | null>(null);
  const [groupReturns, setGroupReturns] = useState<Record<string, string>[] | null>(null);
  const [turnover, setTurnover] = useState<Record<string, string>[] | null>(null);
  const [coverage, setCoverage] = useState<Record<string, string>[] | null>(null);

  useEffect(() => {
    if (!ctx.project || !ctx.runId) return;
    setLoading(true);
    (async () => {
      try {
        const r = await fetchRun(ctx.project!, ctx.runId!);
        setRun(r);
        const [icRows, grRows, turnoverRows, coverageRows] = await Promise.all([
          fetchArtifactWithFallback(ctx.project!, ctx.runId!, [
            "ic_timeseries",
            "rank_ic_timeseries",
          ]),
          fetchArtifactCsv(ctx.project!, ctx.runId!, "group_returns"),
          fetchArtifactCsv(ctx.project!, ctx.runId!, "turnover"),
          fetchArtifactCsv(ctx.project!, ctx.runId!, "coverage"),
        ]);
        setIc(icRows);
        setGroupReturns(grRows);
        setTurnover(turnoverRows);
        setCoverage(coverageRows);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    })();
  }, [ctx.project, ctx.runId]);

  if (!ctx.project || !ctx.runId) return <EmptyState message="Missing project or run_id query parameter." />;
  if (loading && !run) return <EmptyState message="Loading evaluation report…" />;
  if (error) return <EmptyState message={`Load failed: ${error}`} tone="neg" />;
  if (!run) return <EmptyState message="Run not found." tone="warn" />;

  const s = (run.summary ?? {}) as Record<string, unknown>;
  const splitStart = parseSplitStart(s.split_contract, s.oos_start, s.split_description, s.split_desc);

  const parsedIcRowsForChart = parseTimeseriesRows(ic, ["rank_ic", "ic"]);
  const parsedIcRowsForIr = parseTimeseriesRows(ic, ["ic"]);
  const parsedTurnoverRows = parseTimeseriesRows(turnover, ["turnover", "mean_turnover", "value"]);
  const parsedCoverageRows = parseTimeseriesRows(coverage, ["coverage", "mean_coverage", "value"]);

  const icChartData = buildSplitSeries(parsedIcRowsForChart, splitStart);
  const icMetricLabel = primaryMetricLabel(parsedIcRowsForChart);
  const meanIcSummaryValue = icMetricLabel === "IC" ? s.mean_ic : s.mean_rank_ic;
  const chartSeries = hasSplitBoundary(parsedIcRowsForChart, splitStart)
    ? [
        { key: "rank_ic_is", label: `${icMetricLabel} (IS)` },
        { key: "rank_ic_oos", label: `${icMetricLabel} (OOS)` },
      ]
    : [{ key: "rank_ic", label: icMetricLabel }];

  const oosRankIc = splitMean(parsedIcRowsForChart, splitStart);
  const oosIcIr = splitIr(parsedIcRowsForIr, splitStart);
  const oosTurnover = splitMean(parsedTurnoverRows, splitStart);
  const oosCoverage = splitMean(parsedCoverageRows, splitStart);

  /* ------- Summary strip -------- */
  const summary: MetricItem[] = [
    {
      label: `Mean ${icMetricLabel === "Metric" ? "RankIC" : icMetricLabel}`,
      value: formatMetricWithOos(fmt(meanIcSummaryValue, 4), formatSplitStat(oosRankIc.split, 4)),
      tone: toneByThreshold(num(meanIcSummaryValue), 0.03, 0.015),
    },
    {
      label: "ICIR",
      value: formatMetricWithOos(fmt(s.ic_ir, 2), formatSplitStat(oosIcIr.split, 2)),
      tone: toneByThreshold(num(s.ic_ir), 1.0, 0.5),
    },
    { label: "DSR p-value", value: fmt(s.dsr_pvalue, 3), tone: toneByReverse(num(s.dsr_pvalue), 0.05, 0.1) },
    {
      label: "Turnover",
      value: formatMetricWithOos(fmtPct(s.mean_long_short_turnover, 1), formatSplitStat(oosTurnover.split, 1, true)),
    },
    {
      label: "Coverage",
      value: formatMetricWithOos(fmtPct(s.coverage_mean, 1), formatSplitStat(oosCoverage.split, 1, true)),
    },
    { label: "N Dates", value: String(s.n_dates_used ?? "—") },
  ];

  /* ------- Verdict line -------- */
  const verdict = String(s.factor_verdict ?? "").trim();
  const triage = String(s.campaign_triage ?? "").trim();
  const promo = String(s.promotion_decision ?? "").trim();

  /* ------- Section 02 : Diagnostics scalar grid -------- */
  const diagnostics: MetricItem[] = [
    { label: "Haircut Ratio", value: fmt(s.haircut_sharpe_ratio, 2), tone: toneByThreshold(num(s.haircut_sharpe_ratio), 0.6, 0.3), threshold: "≥0.6 good" },
    { label: "Random-baseline p", value: fmt(s.random_baseline_p_value, 3), tone: toneByReverse(num(s.random_baseline_p_value), 0.05, 0.1), threshold: "≤0.05 good" },
    { label: "Random z-score", value: fmt(s.random_baseline_observed_z_score, 2) },
    { label: "Fama-MacBeth t", value: fmt(s.fama_macbeth_t_statistic, 2), tone: toneByAbs(num(s.fama_macbeth_t_statistic), 2.0, 1.0) },
    { label: "Untradable %", value: fmtPct(s.tradability_untradable_rate, 1), tone: toneByReverse(num(s.tradability_untradable_rate), 0.02, 0.05) },
    { label: "Cost Drag Share", value: fmtPct(s.daily_pnl_cost_drag_share, 1) },
    { label: "Mean MI", value: fmt(s.mean_mutual_information, 4) },
    { label: "IC Half-life", value: s.ic_half_life_horizon ? `${s.ic_half_life_horizon}` : "—", unit: "periods" },
  ];

  /* ------- Section 04 : Group returns monotonicity -------- */
  const groupAgg = aggregateGroupMeans(groupReturns ?? []);

  /* ------- Section 05 : Flags / callouts -------- */
  const flags = parseFlags(s.rolling_instability_flags) ?? [];
  const promoReasons = parseFlags(s.promotion_reasons) ?? [];
  const pvRisks = parseFlags(s.portfolio_validation_major_risks) ?? [];

  return (
    <DashboardShell>
      <ReportHeader
        kicker={`Evaluation Report · ${run.case_name ?? run.factor_name ?? "run"}`}
        title={run.factor_name || run.case_name || "Signal Evaluation"}
        subtitle={
          verdict
            ? `Verdict: ${verdict}. ${triage ? `Triage — ${triage}. ` : ""}${promo ? `L2 — ${promo}.` : ""}`
            : "Structured walkthrough of performance, stability, and attribution."
        }
        meta={[
          { label: "Run", value: run.run_id.slice(0, 12) },
          { label: "Project", value: run.project_slug ?? ctx.project! },
          { label: "Status", value: run.status ?? "—" },
          { label: "Updated", value: (run.updated_at ?? run.created_at ?? "").slice(0, 10) || "—" },
          { label: "Split", value: String(s.split_description ?? "—") },
        ]}
      />

      <SummaryStrip items={summary} />

      <Section
        id="timeseries"
        kicker="Section 01"
        heading="Primary Time Series"
        note="Rank information coefficient over the evaluation window. If split is available, we separate IS and OOS series."
      >
        {icChartData.length > 0 ? (
          <ChartPanel
            title={`${icMetricLabel} — daily`}
            subtitle={`n = ${icChartData.length} observations · source: ic_timeseries.csv${splitStart ? ` · split >= ${splitStart}` : ""}`}
            footnote="If OOS split is present, blue and green paths show IS and OOS separately."
          >
            <LineSeriesChart
              data={icChartData}
              xKey="date"
              series={chartSeries}
              yFormat={(v) => v.toFixed(3)}
            />
          </ChartPanel>
        ) : (
          <Callout tone="neutral">Time series artifact not available for this run.</Callout>
        )}
      </Section>

      <Section
        id="diagnostics"
        kicker="Section 02"
        heading="Diagnostics Grid"
        note="Scalar diagnostic surface. Tiles are generated from the run summary dictionary; add or remove entries without touching the layout."
      >
        <MetricGrid items={diagnostics} cols={4} />
      </Section>

      {groupAgg.length > 0 && (
        <Section
          id="groups"
          kicker="Section 03"
          heading="Group Response"
          note="Mean return by quantile group aggregated over the evaluation window."
        >
          <ChartPanel title="Mean group return" height={240}>
            <BarSeriesChart
              data={groupAgg}
              xKey="group"
              series={[{ key: "mean", label: "Mean Return" }]}
              yFormat={(v) => v.toFixed(3)}
            />
          </ChartPanel>
        </Section>
      )}

      <Section
        id="table"
        kicker="Section 04"
        heading="Run Context"
        note="Point-in-time context that accompanies the evaluation."
      >
        <DataTable
          columns={CONTEXT_COLUMNS}
          rows={buildContextRows(run, s)}
          caption="Context"
        />
      </Section>

      {(flags.length > 0 || promoReasons.length > 0 || pvRisks.length > 0) && (
        <Section
          id="observations"
          kicker="Section 05"
          heading="Observations & Flags"
          note="Automated callouts emitted by the evaluation pipeline."
        >
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {flags.length > 0 && (
              <Callout tone="warn" title="Rolling instability flags">
                <ul className="list-disc ml-5 space-y-1">
                  {flags.map((f, i) => <li key={i}>{f}</li>)}
                </ul>
              </Callout>
            )}
            {pvRisks.length > 0 && (
              <Callout tone="neg" title="Portfolio validation — major risks">
                <ul className="list-disc ml-5 space-y-1">
                  {pvRisks.map((f, i) => <li key={i}>{f}</li>)}
                </ul>
              </Callout>
            )}
            {promoReasons.length > 0 && (
              <Callout tone="pos" title="Promotion reasons">
                <ul className="list-disc ml-5 space-y-1">
                  {promoReasons.map((f, i) => <li key={i}>{f}</li>)}
                </ul>
              </Callout>
            )}
          </div>
        </Section>
      )}
    </DashboardShell>
  );
}

/* ------------------------------------------------------------------ */

async function fetchArtifactWithFallback(
  project: string,
  runId: string,
  keys: string[],
): Promise<Record<string, string>[] | null> {
  for (const key of keys) {
    const rows = await fetchArtifactCsv(project, runId, key);
    if (rows !== null) {
      return rows;
    }
  }
  return null;
}

interface ContextRow extends Record<string, unknown> {
  field: string;
  value: string;
}

const CONTEXT_COLUMNS: TableColumn<ContextRow>[] = [
  { key: "field", header: "Field", width: "30%" },
  { key: "value", header: "Value", kind: "mono" },
];

function buildContextRows(run: RunPayload, s: Record<string, unknown>): ContextRow[] {
  const out: ContextRow[] = [
    { field: "Run ID", value: run.run_id },
    { field: "Factor", value: run.factor_name ?? "—" },
    { field: "Case", value: run.case_name ?? "—" },
    { field: "Split", value: String(s.split_description ?? "—") },
    { field: "Rolling Window", value: String(s.rolling_window_size ?? "—") },
    { field: "Mean IC", value: fmt(s.mean_ic, 4) },
    { field: "Mean Long-Short Return", value: fmtPct(s.mean_long_short_return, 3) },
    { field: "Mean Turnover", value: fmtPct(s.mean_long_short_turnover, 2) },
  ];
  return out;
}

function toneByThreshold(v: number | null, good: number, warn: number): Tone {
  if (v === null) return "neutral";
  if (v >= good) return "pos";
  if (v >= warn) return "warn";
  return "neg";
}
function toneByReverse(v: number | null, good: number, warn: number): Tone {
  if (v === null) return "neutral";
  if (v <= good) return "pos";
  if (v <= warn) return "warn";
  return "neg";
}
function toneByAbs(v: number | null, good: number, warn: number): Tone {
  if (v === null) return "neutral";
  const a = Math.abs(v);
  if (a >= good) return "pos";
  if (a >= warn) return "warn";
  return "neg";
}

function EmptyState({ message, tone = "neutral" }: { message: string; tone?: Tone }) {
  const color = tone === "neg" ? "text-tone-neg" : tone === "warn" ? "text-tone-warn" : "text-ink-muted";
  return (
    <DashboardShell>
      <div className="py-24 flex flex-col items-center gap-5">
        <PressedLeaf className="w-24 h-24 opacity-75" />
        <div className={`text-center font-serif italic text-lede ${color}`}>{message}</div>
      </div>
    </DashboardShell>
  );
}
