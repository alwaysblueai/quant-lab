# Metrics Dashboard — Reusable Research Skeleton

Editorial, institution-grade dashboard shell for evaluation pages.

## Run

### Development (standalone)

```bash
cd frontend/metrics-dashboard
npm install
npm run dev   # http://localhost:5174
# The dev server proxies /api → http://127.0.0.1:8765 (the alpha-lab Python server).
# Start the Python server separately, then open:
#   http://localhost:5174/?project=<slug>&run_id=<run_id>
```

### Production (mounted inside web_unified.py)

```bash
cd frontend/metrics-dashboard
npm install
npm run build   # emits frontend/metrics-dashboard/dist with base=/metrics-app/
```

Then start `start_unified_server()`. The Python handler serves the built SPA at
`/metrics-app/*` (see `_serve_metrics_app` in `src/alpha_lab/web_unified.py`).
From the workbench, clicking **查看评价台** on a succeeded run now loads:

```
/metrics-app/?project=<slug>&run_id=<run_id>
```

inside an iframe in the evaluation view. A "use legacy overview" link remains
as fallback. If the build output is missing, the handler renders a build-hint
page instead of 404.

## Structure

```
src/
  components/
    DashboardShell.tsx   # paper-tone outer shell, max-width report container
    ReportHeader.tsx     # title / subtitle / meta row
    SummaryStrip.tsx     # 6-tile inline KPI strip
    Section.tsx          # kicker + H2 + note + optional actions
    MetricCard.tsx       # MetricCard + MetricGrid (2..6 cols)
    DataTable.tsx        # numeric-aligned research table
    ChartPanel.tsx       # chart shell + Line/Area/Bar primitives (Recharts)
    Callout.tsx          # toned commentary block
  types.ts               # MetricItem, TableColumn, SectionSpec, Tone
  App.tsx                # demo wiring; replace with your payload
```

## Replacing the demo

- Swap the arrays in `App.tsx` with your own `MetricItem[]`, table rows, and chart data.
- `TableColumn<R>` is generic — define your own row type.
- Chart components accept a neutral `{ data, xKey, series }` contract so you can
  point them at any long-form numeric series.
- `Section` is content-agnostic — compose any mix of `MetricGrid`, `ChartPanel`,
  `DataTable`, and `Callout`.

## Design tokens (tailwind.config.ts)

- `paper` / `paper.soft` / `paper.deep` — warm off-white surface tiers
- `ink` / `ink.muted` / `ink.faint` — typographic ink
- `rule` / `rule.soft` — hairline dividers
- `accent.teal|olive|clay` — muted accents
- `tone.pos|warn|neg` — low-saturation status colors
- `font.serif` — headings & editorial commentary
- `font.sans` — body
- `font.mono` — numeric values (tabular)
