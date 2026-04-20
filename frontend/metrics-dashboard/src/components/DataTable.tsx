import { TableColumn } from "../types";

interface Props<R extends Record<string, unknown>> {
  columns: TableColumn<R>[];
  rows: R[];
  caption?: string;
  footnote?: string;
  dense?: boolean;
}

/* Editorial data table.
 * Lives on ivory with a cream border — no box shadow. Hairlines between
 * rows stay in the rule.cream family so the table reads as a paragraph
 * of numerics, not a grid of cells. */
export function DataTable<R extends Record<string, unknown>>({
  columns,
  rows,
  caption,
  footnote,
  dense,
}: Props<R>) {
  const rowPad = dense ? "py-2" : "py-3";
  return (
    <figure className="surface-card overflow-hidden">
      {caption && (
        <figcaption className="px-5 sm:px-6 py-3.5 rule-bottom">
          <span className="kicker">{caption}</span>
        </figcaption>
      )}
      <div className="overflow-x-auto">
        <table className="w-full text-body-sm">
          <thead>
            <tr className="border-b border-rule-cream bg-paper-deep/30">
              {columns.map((c) => (
                <th
                  key={c.key}
                  style={{ width: c.width, textAlign: c.align ?? "left" }}
                  className={`kicker px-4 sm:px-5 ${rowPad} font-medium`}
                >
                  {c.header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr
                key={i}
                className="border-b border-rule-cream last:border-b-0 hover:bg-paper-deep/40 transition-colors"
              >
                {columns.map((c) => {
                  const align =
                    c.align ??
                    (c.kind === "num" || c.kind === "mono" ? "right" : "left");
                  const cls =
                    c.kind === "num" || c.kind === "mono"
                      ? "num text-ink-strong"
                      : "text-ink-body";
                  const content = c.render
                    ? c.render(row)
                    : (row[c.key] as React.ReactNode);
                  return (
                    <td
                      key={c.key}
                      style={{ textAlign: align }}
                      className={`px-4 sm:px-5 ${rowPad} ${cls}`}
                    >
                      {content ?? <span className="text-ink-faint">—</span>}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {footnote && (
        <div className="px-5 sm:px-6 py-3 border-t border-rule-cream text-footnote text-ink-faint font-serif italic">
          {footnote}
        </div>
      )}
    </figure>
  );
}
