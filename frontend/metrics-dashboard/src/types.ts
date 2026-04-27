export type Tone = "pos" | "warn" | "neg" | "neutral";

export interface MetricItem {
  label: string;
  value: string | number;
  unit?: string;
  delta?: string | number;
  tone?: Tone;
  help?: string;
  threshold?: string;
}

export interface TableColumn<R = Record<string, unknown>> {
  key: string;
  header: string;
  align?: "left" | "right" | "center";
  width?: string;
  render?: (row: R) => React.ReactNode;
  kind?: "text" | "num" | "mono";
}

export interface ReportMeta {
  label?: string;
  value: string;
}

export interface SectionSpec {
  id: string;
  kicker?: string;
  heading: string;
  note?: string;
}
