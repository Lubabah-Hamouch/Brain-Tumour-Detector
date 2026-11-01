# Frontend

Next.js (App Router, React, TypeScript) site for the two models. It reads everything it shows from
the FastAPI backend in `../backend`.

| Route | Status |
|---|---|
| `/` Overview | built: headline test results from `GET /api/results/summary`, hero render, index, pipeline |
| `/segment` | placeholder; next to build, on `POST /api/segment` + `GET /api/jobs/{id}` |
| `/classify` | built: upload or test-set sample → checked preview → prediction with Grad-CAM viewer and PNG download (`POST /api/classify`, `GET /api/samples`) |
| `/model-card` | placeholder |

## Running locally

Start the API first, from the project root, in the project virtual environment:

```bash
uvicorn backend.main:app --port 8010
```

Then, in this folder:

```bash
npm install
npm run dev          # http://localhost:3000, hot reload
```

or as a production build:

```bash
npm run build
npm run start        # http://localhost:3000
```

Other commands: `npm run lint`.

The API address comes from `API_URL` (server side) or `NEXT_PUBLIC_API_URL`, defaulting to
`http://127.0.0.1:8010`. If the API is not running, the Overview still renders and the results
section says so, with the command to start it.

## Hero image

The Overview hero is a 3D render of a held-out test patient's brain and the ensemble's predicted
tumour regions. It is patient-derived, so it is not committed: `public/generated/` is ignored.
Generate it with `src/visualization/render_hero.py` (instructions in its docstring; needs
`requirements-render.txt`). Without it the page shows a short note in its place.

## Structure

```
src/
  app/                  routes (App Router)
    layout.tsx          fonts, design tokens, header and footer
    globals.css         base styles and shared primitives (.label, .mono, .caption, .u)
    page.tsx            Overview
    segment/ classify/ model-card/
  components/           SiteHeader, SiteFooter, ButtonLink, Button, Metric, HeroVisual, PlannedScreen
    classify/           Classify screen: upload view, result view, shared styles
  design/
    tokens.ts           the design system: colour, type scale, motion, spacing
    useEntryProgress.ts eased entry animation (count-ups, bars), honours reduced motion
  lib/api.ts            typed API client
  lib/imageCheck.ts     browser-side upload checks (format by content, size, dimensions)
```

## Design system

`src/design/tokens.ts` is the single source of the design tokens. The root layout turns them into
CSS custom properties (`--color-*`, `--region-*`, `--text-12` … `--text-128`, `--lh-*`, `--track-*`,
`--ease`, `--dur-*`, `--page-margin`, `--section-gap`, `--dot-grid`), so CSS Modules use the
variables and TypeScript imports the same values directly (for example, animation timings).

- **Type:** Geist for text, JetBrains Mono for numbers. Both are self-hosted at build time with
  `next/font`, so there are no runtime font requests.
- **Scale:** 12 / 14 / 16 / 20 / 28 / 40 / 64 / 128 px.
- **Colour:** a graphite grey ramp. Red is reserved for findings: edema `#7A111B`, necrotic core
  `#D7263D`, enhancing tumour `#FFC2CA`.
- **Motion:** one easing curve, `cubic-bezier(0.2, 0.7, 0.2, 1)`. With `prefers-reduced-motion`,
  animations and transitions are switched off and count-ups show their final value immediately.
