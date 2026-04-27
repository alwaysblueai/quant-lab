import { MetricItem, Tone } from "../types";

const toneClass: Record<Tone, string> = {
  pos: "text-tone-pos",
  warn: "text-tone-warn",
  neg: "text-tone-neg",
  neutral: "text-ink",
};

/* Left accent strip — quiet chapter-mark instead of a heavy left border.
 * A tinted ::before rail keeps the card's 8px radius intact. */
const toneRail: Record<Tone, string> = {
  pos:     "before:bg-tone-pos",
  warn:    "before:bg-tone-warn",
  neg:     "before:bg-tone-neg",
  neutral: "before:bg-rule-hover",
};

export function MetricCard({ item }: { item: MetricItem }) {
  const tone = item.tone ?? "neutral";
  return (
    <div
      className={`
        relative overflow-hidden
        bg-paper-soft border border-rule-cream rounded-sm
        px-5 py-4
        transition-shadow duration-150
        hover:shadow-whisper
        before:content-[''] before:absolute before:inset-y-0 before:left-0
        before:w-[3px] ${toneRail[tone]}
      `}
      title={item.help}
    >
      <div className="kicker mb-2.5">{item.label}</div>
      <div className="flex items-baseline gap-1.5">
        <span className={`num text-metric ${toneClass[tone]}`}>
          {item.value}
        </span>
        {item.unit && (
          <span className="text-caption text-ink-faint">{item.unit}</span>
        )}
      </div>
      <div className="mt-2.5 flex items-center justify-between min-h-[16px]">
        {item.delta !== undefined ? (
          <span className={`num text-micro ${toneClass[tone]}`}>
            {item.delta}
          </span>
        ) : (
          <span />
        )}
        {item.threshold && (
          <span className="text-micro text-ink-faint italic font-serif">
            {item.threshold}
          </span>
        )}
      </div>
    </div>
  );
}

export function MetricGrid({
  items,
  cols = 4,
}: {
  items: MetricItem[];
  cols?: 2 | 3 | 4 | 5 | 6;
}) {
  /* Responsive: always stack on very narrow, expand through the tiers. */
  const gridCls = {
    2: "grid-cols-1 sm:grid-cols-2",
    3: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3",
    4: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-4",
    5: "grid-cols-1 sm:grid-cols-2 lg:grid-cols-5",
    6: "grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-6",
  }[cols];
  return (
    <div className={`grid ${gridCls} gap-4`}>
      {items.map((it, i) => (
        <MetricCard key={i} item={it} />
      ))}
    </div>
  );
}
