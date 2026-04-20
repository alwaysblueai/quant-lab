import { PropsWithChildren } from "react";

interface Props {
  id: string;
  kicker?: string;
  heading: string;
  note?: string;
  actions?: React.ReactNode;
}

/* Section = a chapter break.
 * Vertical rhythm is generous — pt-20 on desktop creates the "new
 * chapter, new spread" feeling. The header is a bottom-ruled band,
 * not a container; content lives on parchment directly. */
export function Section({
  id,
  kicker,
  heading,
  note,
  actions,
  children,
}: PropsWithChildren<Props>) {
  return (
    <section
      id={id}
      className="pt-14 sm:pt-20 lg:pt-24 border-t border-rule-cream [&:first-of-type]:pt-0 [&:first-of-type]:border-t-0"
    >
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 sm:gap-6 pb-5 rule-bottom">
        <div className="min-w-0">
          {kicker && <div className="kicker mb-2">{kicker}</div>}
          <h2 className="text-sub-sm sm:text-sub font-medium text-ink">
            {heading}
          </h2>
          {note && (
            <p className="mt-3 max-w-prose text-body text-ink-muted font-serif">
              {note}
            </p>
          )}
        </div>
        {actions && (
          <div className="flex items-center gap-2 text-caption text-ink-muted shrink-0">
            {actions}
          </div>
        )}
      </div>
      <div className="pt-8 sm:pt-10">{children}</div>
    </section>
  );
}
