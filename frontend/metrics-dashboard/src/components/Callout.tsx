import { PropsWithChildren } from "react";
import { Tone } from "../types";

/* Pull-quote style callout.
 * Tone is carried by a left chapter-rail (::before) on a cream-bordered
 * tinted surface — matches MetricCard's rail language so toned elements
 * read as one family across the page. */
const toneRail: Record<Tone, string> = {
  pos:     "before:bg-tone-pos",
  warn:    "before:bg-tone-warn",
  neg:     "before:bg-tone-neg",
  neutral: "before:bg-ink-faint",
};

const toneSurface: Record<Tone, string> = {
  pos:     "bg-callout-pos",
  warn:    "bg-callout-warn",
  neg:     "bg-callout-neg",
  neutral: "bg-callout-neutral",
};

export function Callout({
  tone = "neutral",
  title,
  children,
}: PropsWithChildren<{ tone?: Tone; title?: string }>) {
  return (
    <aside
      className={`
        relative overflow-hidden
        border border-rule-cream rounded-md
        ${toneSurface[tone]}
        px-5 sm:px-6 py-4 sm:py-5
        before:content-[''] before:absolute before:inset-y-0 before:left-0
        before:w-[3px] ${toneRail[tone]}
      `}
    >
      {title && (
        <div className="kicker mb-1.5">{title}</div>
      )}
      <div className="text-body-lg text-ink-body font-serif leading-relaxed">
        {children}
      </div>
    </aside>
  );
}
