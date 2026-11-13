# Hackathon teammate guide

## What is ready to test?

The offline simulator, metric calculations, CLI, saved-run analysis, and static
interactive dashboard. No API keys, local LLM, or paid account are required.
Dependencies need an internet connection on first install; subsequent offline
simulation and tests do not need model-provider access.

**The checked-in dashboard/results are synthetic demonstration data.** Differences
between conditions reflect simulator assumptions, not measured differences between
real models or countries. Do not present them as live empirical evidence.

## 1. Install (macOS/Linux)

Prerequisites: Git, Python 3.11 or 3.12, and Make. From the repo root:

```sh
python3 --version
make install
make check
```

If your default Python is older, use `make install BOOTSTRAP_PYTHON=python3.12`.
The installer reuses `.venv`; it does not intentionally wipe an existing environment.
For a fresh checkout, use the clone command in README.md. No shell activation or
`uv` is required. Python versions beyond 3.12 are not currently in the CI matrix.

`make check` must finish successfully: Ruff lint/format checks, fast tests, then a
small synthetic run with 11 conditions and numeric C1/C2/C3 analysis. Look for
`Demo complete`, a nonzero record count, and a tiny `max_identity_residual` (below
1e-11). Generated files go to ignored `tmp/`, not published `docs/` or `reports/`.

### Without Make / Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -e ".[dev]"
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m ruff format --check .
.venv\Scripts\python -m pytest -m "not e2e" -v
.venv\Scripts\python -m apart_epistemics setup --out tmp/smoke.yaml --agents 8 --rounds 2 --tasks 2
.venv\Scripts\python -m apart_epistemics demo --config tmp/smoke.yaml --out tmp/smoke-results --site tmp/smoke-site
.venv\Scripts\python -m apart_epistemics analyze --input tmp/smoke-results
.venv\Scripts\python -m http.server 8000 --bind 127.0.0.1 --directory docs
```

On macOS/Linux without Make, substitute `.venv/bin/python` and create the venv
with `python3 -m venv .venv`. Windows instructions are provided but not locally
verified on Windows; CI currently runs Linux.

## 2. Try the dashboard

### New React / shadcn/ui explorer

Requires Node 22.12+ in addition to the Python tools:

```sh
make ui-install
make ui-check
make ui-dev
```

Open the URL Vite prints (normally http://127.0.0.1:5173). Try the Overview,
Numeric core, and Response traces tabs. Expand the recorded context/topics/raw
JSON rows; these are actual saved fields, not a live agent's reasoning. No API
keys are required. The source dataset remains `docs/data/benchmark_summary.json`.
See [frontend/README.md](frontend/README.md) for npm-only/Windows commands,
browser setup, and the data contract. Run `make ui-check` for frontend changes.

Frontend browser tests use their own Playwright dependency:

```sh
cd frontend
npx playwright install chromium
cd ..
make ui-e2e
```

### Standalone Python report (no Node build)

```sh
make serve
```

Open http://127.0.0.1:8000. This serves the checked-in full dashboard, not the
small smoke run. Stop with Ctrl-C. Use `make serve PORT=8001` if the port is busy.
To inspect your newly generated small run instead:

```sh
.venv/bin/python -m http.server 8001 --bind 127.0.0.1 --directory tmp/smoke-site
```

Manual acceptance checklist:

- Switch Track 4 / Numeric Core tabs; both show charts and metrics.
- Filter by nationality, then reset to All; the full dashboard has 11 conditions.
- Change the round and trajectory metric; the chart updates.
- Sort the score table; inspect a condition and close its trace drawer.
- Try a narrow/mobile window: no horizontal page overflow or clipped controls.
- Check the browser console for errors; report missing plots, NaN, or undefined.

## 3. Run browser tests

```sh
make install-browser
make test-e2e
make cov
```

On Linux, Chromium may also need system packages:
` .venv/bin/python -m playwright install --with-deps chromium`.
On Windows use `.venv\Scripts\python -m playwright install chromium` and
`.venv\Scripts\python -m pytest -m e2e -v`.

Browser tests target the checked-in full `docs/` dashboard. A browser launch
failure is not a passing check: keep the failure output and run fast tests
separately. In restricted macOS execution environments, Chromium can fail with
`bootstrap_check_in` / `MachPortRendezvousServer`; retry from a normal Terminal
session or use the Linux CI runner. Do not disable/skip the test to make CI green.

## 4. Work together without trampling results

- Create a feature branch and agree who owns overlapping files.
- Engine/metrics: `src/apart_epistemics/engine/`, `metrics.py`, corresponding tests.
- Backend integration: `backends/`, `parsers.py`, `tests/test_backends.py`.
- Dashboard: `templates/dashboard.html`, `report/`, `tests/e2e/`.
- Research/config: `configs/`, `datasets/`, `SPEC.md`, `LITERATURE_REVIEW.md`.
- Run `make check` before a PR; dashboard changes also need browser checks.
- `make demo` deliberately rewrites tracked full results and `docs/`. Use it
  only when updating the shared demonstration, and review the resulting diff.
- Give each experiment a separate output directory under `tmp/`. Numeric runs
  resume existing JSONL cells: do not reuse an output directory after changing
  config, seed, model, or backend, or you may mix old and new results.
- `make clean` deletes `tmp/` including your local experiments. Archive anything
  you need outside that directory first.

## 5. Live-run limitations — agree with the team before spending

- Cloud keys are read from exported environment variables. Merely creating a
  `.env` file does not load it. Never paste credentials into issues or commits.
- There is no `--backend live`. See `apart-epistemics run --help` for adapters.
- The CLI creates one backend/model per invocation; condition provenance labels
  do not automatically route requests to a pool of real models. Implement and
  verify routing before claiming a multi-model live experiment.
- `--budget` is an approximate numeric-track cost guard, not a provider-enforced
  or whole-run spending limit. Track 4 is not covered, and a completed call may
  exceed the estimate. Use provider-side limits and small, explicitly approved runs.
- Start live numeric experimentation with `--numeric-only`, an explicit real
  model ID, a small config, and a new output directory. Availability of default
  model IDs is not guaranteed. No cloud inference is part of onboarding.
- `audit-selection` currently uses built-in example scores, not a live evaluation.
- Synthetic topic-label agreement does not replace independent human validation.
- Dependency ranges are not a lockfile. Include `python -m pip freeze` output
  (reviewed for local/private URLs) when reporting environment-specific failures.

## Bug report / handoff template

```text
Commit and branch:
OS and Python version:
Install command:
Exact failing command:
Expected behavior:
Actual behavior + exit code:
Config / seed / backend (no keys):
Output directory (new or resumed?):
Relevant traceback / screenshot (redact secrets and private prompts):
Checks passed / failed / not run:
```

CI runs Python 3.11/3.12 lint, full tests, the smoke workflow, and a wheel-install
smoke test outside the source checkout. Local success is not a substitute for a
green CI run after changes are pushed.
