# Frontend

Next.js (App Router, React, TypeScript) site for the two models. It reads everything it shows from
the FastAPI backend in `../backend`.

| Route | Status |
|---|---|
| `/` Overview | built: headline test results from `GET /api/results/summary`, hero render, index, pipeline |
| `/segment` | built: four NIfTI files (or the sample study) with sequence detection and header checks → background job with live stages → axial, coronal, sagittal and interactive 3D views, volumes and downloads (`POST /api/segment`, `GET /api/jobs/{id}`, `GET /api/sample-study`, `GET /api/segment/{id}/mesh.glb`) |
| `/classify` | built: upload or test-set sample → checked preview → prediction with Grad-CAM viewer (soft heatmap with a half-peak outline; the plain image first for "No tumour") and PNG download (`POST /api/classify`, `GET /api/samples`) |
| `/model-card` | built: specification, test tables with paired comparisons, stress-test chart and limitations, all from `GET /api/results/model-card` |

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

The API address comes from `API_URL` (server side) or `NEXT_PUBLIC_API_URL`. `npm run dev`
defaults to `http://127.0.0.1:8010`. A production build has no default: set `NEXT_PUBLIC_API_URL`
before `npm run build` (it is fixed into the client bundle), or every API request fails with a
message naming the variable. If the API is not running, the Overview still renders and the results
section says so, with the command to start it.

**Vercel:** see [DEPLOYMENT.md](../DEPLOYMENT.md). On Vercel, `next.config.ts` stops the build if
`NEXT_PUBLIC_API_URL` is missing, and skips the standalone output, which only the Docker image uses.
`HERO_IMAGE_URL` (read at request time) points the Overview's 3D image at a hosted copy, because
the render is not committed and so is not in a deployment built from git.

## Docker

`Dockerfile` builds the production site as a standalone Next.js server (`output: "standalone"` in
`next.config.ts`) on Node 22. Normally it runs through `docker-compose.yml` at the project root,
together with the API (see "Running with Docker" in the root README).

- **`NEXT_PUBLIC_API_URL`** (build argument, default `http://localhost:8010`): where the browser
  reaches the API. It is fixed into the client bundle at build time.
- **`API_URL`** (runtime variable): where the server reaches the API. Compose sets it to
  `http://api:8010` on the internal network.
- **Hero image:** mount it at `/app/public/generated` (compose mounts `public/generated/`). It is
  not copied into the image.

```bash
docker build -t brain-tumour-web .
docker run --rm -p 3000:3000 -e API_URL=http://host.docker.internal:8010 brain-tumour-web
```

## 3D view

The Segment results have a fourth view, **3D**, next to Axial, Coronal and Sagittal
(`src/components/segment/Volume3D.tsx`). It loads the result's `mesh.glb` and draws:

- the brain as glass, with a small fresnel shader;
- the tumour layers in the finding colours, with a glow on the tumour only (selective bloom).

**Controls:** drag to rotate and scroll or pinch to zoom, with damped, non-bouncing orbit
controls. The label toggles and the overlay opacity slider act on the 3D layers as on the slices.

**Libraries:** three.js with `@react-three/fiber`, `@react-three/drei` and
`@react-three/postprocessing`. They load only when the 3D view is opened.

**Performance:** the canvas renders only when something changes, and the pixel ratio is capped at
1.5. The glass is a cheap shader rather than a transmission pass. It runs smoothly on an ordinary
laptop GPU.

**Failure handling:** without WebGL, the view shows a message and the rest of the screen is
unchanged. If the surfaces fail to load, it shows the reason and a retry.

The Overview's hero image links to `/segment?demo=3d`. That loads the sample study, runs it through
the real upload and job, and opens the result in 3D.

## Hero image

The Overview hero is a 3D render of a held-out test patient's brain and the ensemble's predicted
tumour regions. It is generated from the patient's volumes, so it is not committed:
`public/generated/` is ignored. (It does appear in the Overview screenshot in `docs/screenshots/`.)
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
  components/           SiteHeader, SiteFooter, ButtonLink, Button, icons, Metric, HeroVisual
    classify/           Classify screen: upload view, result view
    modelcard/          Model card: stress-test headline and chart, table styles
    segment/            Segment screen: upload, processing (job polling), results viewer, 3D view
    workspace/          layout and controls shared by both screens
  design/
    tokens.ts           the design system: colour, type scale, motion, spacing
    useEntryProgress.ts eased entry animation (count-ups, bars), honours reduced motion
  lib/api.ts            typed API client
  lib/imageCheck.ts     browser-side upload checks (format by content, size, dimensions)
  lib/nifti.ts          NIfTI header and single-slice reading in the browser (streams gzip)
  lib/studyFiles.ts     sequence detection from file names, per-file and cross-file checks
```

## Glossary

Domain terms carry a small "?" (`src/components/Term.tsx`) that shows a plain-language definition
from `src/lib/glossary.ts` on hover, keyboard focus or tap. The MRI sequences get extra help, since
every Segment screen depends on them:

- the FLAIR, T1, T1ce and T2 toggles show their own definition on hover, focus or tap (the
  `useDefinition` hook from `Term.tsx`), and a line under the toolbar always says what the
  sequence on screen shows;
- on the upload screen, each file's sequence menu and each sequence in the sidebar has its own "?".

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
