import { PropsWithChildren } from "react";

/* Page-level container.
 * Responsive padding widens with viewport so the parchment margin feels
 * like a book gutter on desktop and a comfortable touch-safe margin on
 * mobile. Max-width caps at 1240px to keep prose measure readable. */
export function DashboardShell({ children }: PropsWithChildren) {
  return (
    <div className="min-h-screen bg-paper">
      <div className="mx-auto max-w-report px-5 sm:px-8 lg:px-12 py-10 sm:py-14 lg:py-20">
        {children}
      </div>
    </div>
  );
}
