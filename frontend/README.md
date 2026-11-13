# React results explorer

A read-only React + TypeScript + Vite frontend over the existing Python report.
The Message/Bubble and expandable Trace UI uses actual copied shadcn/ui Radix
Nova components, not CSS lookalikes or an AI SDK service. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream sources and license.

## Start

Requires Node **22.12+** and npm. From the repository root:

```sh
make ui-install
make ui-check
make ui-dev
```

Open the URL Vite prints (normally http://127.0.0.1:5173). Stop with Ctrl-C.
No API keys, Python server, or model calls are needed to view the bundled results.
First-time dependency installation requires network access; dependencies are
pinned in `package-lock.json`. Use `npm ci`, not an unreviewed dependency update.

Without Make, including Windows, run these inside `frontend/`:

```sh
npm ci
npm run lint
npm test
npm run build
npm run dev
```

## Data contract

`src/main.tsx` imports `../docs/data/benchmark_summary.json` relative to the
frontend directory (the canonical file at the repository root). There is no
second manually maintained dataset and no browser recalculation of benchmark
metrics, confidence intervals, or p-values. TypeScript checks the imported
report against the frontend's expected structure.

The checked-in dataset is synthetic. A high score or tiny Page identity residual
is not a real-model finding. Response specimens are sampled final-round records,
not full conversation history, hidden reasoning, or live tool activity. The
response view labels the actual recorded rounds; the overview round filter does
not select historical responses that were never exported.

When intentionally updating the shared demo, run `make demo` at the repo root,
review the changed data, then rebuild the frontend. Development mode reloads
when the imported JSON changes. Do not overwrite the shared report for a private
experiment; use the existing CLI's separate output paths instead.

## What to exercise

- Dashboard: compact laptop layout with linked chart selection, all condition rows,
  six selected-condition metrics, and a topic trajectory. Views live in the sidebar.
- Overview: family/search filters, reset, round selection, keyboard-selectable
  chart points, trajectory metric, sortable scores, and export of visible scores to CSV.
- Response traces: choose a condition, read both recorded outputs, expand context,
  extracted topics, and raw JSON. Baseline settings and a baseline shortcut are included.
  System prompts and tool/skill events are explicitly marked not recorded in the
  current export. Text is rendered literally, never as HTML.
- Numeric core: C1/C2/C3 final scores, metric trajectories, and recorded contrasts.
  These controls are independent of Track 4's filters.
- Narrow viewports: charts/tables scroll locally; the page should not overflow.

## Tests

```sh
npm test
npm run lint
npm run build
npx playwright install chromium
npm run test:e2e
```

Linux may need `npx playwright install --with-deps chromium`. Browser tests start
an isolated production preview on port 4173 and check desktop/mobile interaction,
keyboard disclosure, console errors, overflow, and screenshots. That port must
be free. Artifacts go into ignored `test-results/`. Python Playwright and npm
Playwright may require different browser versions; install the frontend's binary
from this directory. Browser launch failure is not a passing E2E run. Restricted
macOS environments can reject Chromium's MachPortRendezvous/bootstrap setup.

`npm run format` formats first-party frontend files. Vendored components are
excluded to preserve their upstream source. `npm run lint` checks TypeScript and
formatting; it is not an ESLint/security-audit substitute.

## Build and publish

`npm run build` creates `frontend/dist/`, including the untouched standalone
Python report under `legacy/`. `make ui-preview` serves the production build
(normally http://127.0.0.1:4173). The build's footer links to the legacy report.
The original `make serve` command still serves `docs/` directly with no Node step.

The Pages workflow builds and tests this frontend before publishing `dist/`.
Relative asset URLs support a repository subpath; no SPA route rewrite is needed.
Changing files locally does not publish them. Existing README screenshots depict
the standalone report, not a verified screenshot of this React frontend.

## Ownership

- `src/components/ui/`: copied official components; review upstream changes.
- `src/components/response-inspector.tsx`: Message + Trace compositions.
- `src/components/track-overview.tsx`, `numeric-panel.tsx`, `plots.tsx`: report views.
- `src/lib/results.ts`: report types, labels, formatting, and round lookup.
- `src/index.css`: theme tokens, Tailwind variants, responsive layout.
- `src/App.test.tsx`: DOM interaction and edge-case tests.
- `e2e/explorer.spec.ts`: real browser acceptance checks.

The Python engine, schema generation, and raw results remain authoritative.
