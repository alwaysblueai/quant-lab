import { ReportMeta } from "../types";

interface Props {
  kicker?: string;
  title: string;
  subtitle?: string;
  meta?: ReportMeta[];
}

/* Article-style masthead.
 * Kicker (terracotta) -> serif title (sub-lg, medium) -> italic serif
 * lede -> tabular meta dl. Everything sits on parchment with a bottom
 * rule, not a container, so the report feels like the opening spread
 * of a journal rather than a card. */
export function ReportHeader({ kicker, title, subtitle, meta = [] }: Props) {
  return (
    <header className="pb-10 sm:pb-12 rule-bottom">
      {kicker && (
        <div className="kicker mb-4 text-accent">{kicker}</div>
      )}
      <h1 className="text-sub-lg sm:text-[40px] font-medium text-ink">
        {title}
      </h1>
      {subtitle && (
        <p className="mt-4 max-w-prose text-lede text-ink-muted font-serif italic">
          {subtitle}
        </p>
      )}
      {meta.length > 0 && (
        <dl className="mt-8 flex flex-wrap gap-x-10 gap-y-3 text-caption text-ink-muted">
          {meta.map((m, i) => (
            <div key={i} className="flex gap-2 items-baseline">
              {m.label && (
                <dt className="kicker">{m.label}</dt>
              )}
              <dd className="num text-ink">{m.value}</dd>
            </div>
          ))}
        </dl>
      )}
    </header>
  );
}
