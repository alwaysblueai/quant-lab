import { PropsWithChildren } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  AreaChart,
  Area,
} from "recharts";

interface ChartPanelProps {
  title: string;
  subtitle?: string;
  footnote?: string;
  height?: number;
  actions?: React.ReactNode;
}

/* Figure panel.
 * Uses the .surface-card language: ivory fill, cream 1px border,
 * 8px radius, no drop shadow. Header is a serif sub-sm caption
 * styled more like a plate caption than a dashboard card title. */
export function ChartPanel({
  title,
  subtitle,
  footnote,
  height = 280,
  actions,
  children,
}: PropsWithChildren<ChartPanelProps>) {
  return (
    <figure className="surface-card overflow-hidden">
      <header className="flex items-start justify-between gap-4 px-5 sm:px-6 py-4 rule-bottom">
        <div className="min-w-0">
          <div className="kicker mb-1.5">{title}</div>
          {subtitle && (
            <div className="text-caption text-ink-muted font-serif italic">
              {subtitle}
            </div>
          )}
        </div>
        {actions && <div className="shrink-0">{actions}</div>}
      </header>
      <div style={{ height }} className="px-4 sm:px-5 pt-5 pb-4">
        <ResponsiveContainer width="100%" height="100%">
          {children as React.ReactElement}
        </ResponsiveContainer>
      </div>
      {footnote && (
        <footer className="px-5 sm:px-6 py-3 border-t border-rule-cream text-footnote text-ink-faint font-serif italic">
          {footnote}
        </footer>
      )}
    </figure>
  );
}

/* -------- Shared chart primitives (muted editorial palette) --------
 *
 * Values mirror the CSS custom properties declared in index.css so
 * changing a palette token propagates to Recharts automatically. */

const chartTokens = {
  axis:          "var(--color-ink-faint)",
  grid:          "var(--color-rule-soft)",
  tooltipBg:     "var(--color-paper-soft)",
  tooltipBorder: "var(--color-rule-cream)",
  series: [
    "#3f5d56", // accent.teal
    "#c96442", // accent.DEFAULT (terracotta — lead series)
    "#6b7a4f", // accent.olive
    "#9b6a48", // accent.clay
    "#5e5d59", // ink.muted
    "#8b3a2b", // tone.neg
  ],
} as const;

/* Recharts reads SVG color props; CSS vars resolve in SVG since Recharts
 * renders into the DOM and inherits var() from :root. Fallbacks kept
 * literal for tick fonts which Recharts inlines as style objects. */
const axisProps = {
  stroke: chartTokens.axis,
  tick: { fill: "#87867f", fontSize: 10.5 },
  tickLine: false,
  axisLine: { stroke: "#e7e3d6" },
} as const;

const tooltipStyle = {
  contentStyle: {
    background: "#faf9f5",
    border: "1px solid #f0eee6",
    borderRadius: 8,
    fontSize: 12,
    fontFamily: "var(--font-mono)",
    boxShadow: "0 4px 24px 0 rgba(20,20,19,0.06)",
  },
  cursor: { stroke: "#e7e3d6", strokeWidth: 1 },
};

export interface SeriesSpec {
  key: string;
  label?: string;
  color?: string;
}

interface SeriesChartProps {
  data: Record<string, number | string>[];
  xKey: string;
  series: SeriesSpec[];
  yFormat?: (v: number) => string;
}

export function LineSeriesChart({
  data,
  xKey,
  series,
  yFormat,
}: SeriesChartProps) {
  return (
    <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
      <CartesianGrid stroke="#e7e3d6" strokeDasharray="2 4" vertical={false} />
      <XAxis dataKey={xKey} {...axisProps} />
      <YAxis {...axisProps} tickFormatter={yFormat} width={48} />
      <Tooltip {...tooltipStyle} />
      {series.map((s, i) => (
        <Line
          key={s.key}
          type="monotone"
          dataKey={s.key}
          name={s.label ?? s.key}
          stroke={s.color ?? chartTokens.series[i % chartTokens.series.length]}
          strokeWidth={1.5}
          dot={false}
          activeDot={{ r: 3 }}
        />
      ))}
    </LineChart>
  );
}

export function AreaSeriesChart({
  data,
  xKey,
  series,
  yFormat,
}: SeriesChartProps) {
  return (
    <AreaChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
      <defs>
        {series.map((s, i) => (
          <linearGradient
            key={s.key}
            id={`fill-${s.key}`}
            x1="0"
            y1="0"
            x2="0"
            y2="1"
          >
            <stop
              offset="0%"
              stopColor={s.color ?? chartTokens.series[i % chartTokens.series.length]}
              stopOpacity={0.22}
            />
            <stop
              offset="100%"
              stopColor={s.color ?? chartTokens.series[i % chartTokens.series.length]}
              stopOpacity={0.02}
            />
          </linearGradient>
        ))}
      </defs>
      <CartesianGrid stroke="#e7e3d6" strokeDasharray="2 4" vertical={false} />
      <XAxis dataKey={xKey} {...axisProps} />
      <YAxis {...axisProps} tickFormatter={yFormat} width={48} />
      <Tooltip {...tooltipStyle} />
      {series.map((s, i) => (
        <Area
          key={s.key}
          type="monotone"
          dataKey={s.key}
          name={s.label ?? s.key}
          stroke={s.color ?? chartTokens.series[i % chartTokens.series.length]}
          strokeWidth={1.3}
          fill={`url(#fill-${s.key})`}
        />
      ))}
    </AreaChart>
  );
}

export function BarSeriesChart({
  data,
  xKey,
  series,
  yFormat,
}: SeriesChartProps) {
  return (
    <BarChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 4 }}>
      <CartesianGrid stroke="#e7e3d6" strokeDasharray="2 4" vertical={false} />
      <XAxis dataKey={xKey} {...axisProps} />
      <YAxis {...axisProps} tickFormatter={yFormat} width={48} />
      <Tooltip {...tooltipStyle} />
      {series.map((s, i) => (
        <Bar
          key={s.key}
          dataKey={s.key}
          name={s.label ?? s.key}
          fill={s.color ?? chartTokens.series[i % chartTokens.series.length]}
          radius={[3, 3, 0, 0]}
        />
      ))}
    </BarChart>
  );
}
