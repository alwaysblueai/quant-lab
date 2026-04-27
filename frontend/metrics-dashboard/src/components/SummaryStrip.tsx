import { MetricItem, Tone } from "../types";

const toneClass: Record<Tone, string> = {
  pos: "text-tone-pos",
  warn: "text-tone-warn",
  neg: "text-tone-neg",
  neutral: "text-ink",
};

/* Masthead numerics strip.
 * A single ruled band under the report header — no container, just
 * top and bottom hairlines — so the report's first visual beat after
 * the serif title is the page-level stat line. */
export function SummaryStrip({ items }: { items: MetricItem[] }) {
  return (
    <div
      className="
        mt-10 sm:mt-12
        grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6
        divide-y sm:divide-y-0 sm:divide-x divide-rule-cream
        border-y border-rule-cream
      "
    >
      {items.map((it, i) => {
        const tone = it.tone ?? "neutral";
        return (
          <div key={i} className="px-5 sm:px-6 py-5 sm:py-6" title={it.help}>
            <div className="kicker">{it.label}</div>
            <div className="mt-2 flex items-baseline gap-1.5">
              <span className={`num text-metric-lg ${toneClass[tone]}`}>
                {it.value}
              </span>
              {it.unit && (
                <span className="text-caption text-ink-faint">{it.unit}</span>
              )}
            </div>
            {it.delta !== undefined && (
              <div className={`num mt-2 text-micro ${toneClass[tone]}`}>
                {it.delta}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
