# Implementation Plan: Epistemic Prism (`apart-epistemics`)

Ordered walking-skeleton-first. Every task follows Red -> Green -> Refactor TDD.

---

## Task 1: Project Packaging, Config & Repo Hygiene (`2 min`)
- **Files:** `pyproject.toml`, `requirements.txt`, `.gitignore`, `.env.example`, `LICENSE`, `configs/default_suite.yaml`
- **Action:** Configure Python `>=3.11` package (`apart_epistemics`), CLI entry point `apart-epistemics`, `ruff` + `pytest` dev dependencies, MIT license, and default 6-arm + $K$-sweep + `T=0` ablation YAML config.
- **Verify:** `uv venv && uv pip install -e ".[dev]" && pytest --version && ruff --version`

## Task 2: Domain Data Models & Dual-Track Datasets (`5 min`)
- **Files:** `tests/test_models_and_datasets.py`, `src/apart_epistemics/models.py`, `src/apart_epistemics/datasets.py`
- **Action (TDD):** Write failing tests validating Pydantic constraints (`stance_prob` in `[0,1]`, `num_models >= 0`, `temperature` bounds), 8 ground-truth tasks, 8 open-debate tasks with canonical topic taxonomies, and stratified regional agent generation (`N=20..50`). Implement models and datasets to pass.
- **Verify:** `pytest tests/test_models_and_datasets.py -q`

## Task 3: Epistemic, Calibration, Goldman & Lexical Metrics (`6 min`)
- **Files:** `tests/test_metrics.py`, `src/apart_epistemics/metrics.py`
- **Action (TDD):** Write failing unit tests for perfect vs. miscalibrated Brier/ECE, confident-wrong rate, Shannon topic entropy & Hill number $\exp(H)$, bimodality coefficient, Goldman effective independent sources $N_{\text{eff}}$ under correlated vs. independent agents, and `distinct_2` / pairwise Jaccard lexical distance. Implement pure vectorized-free Python math functions.
- **Verify:** `pytest tests/test_metrics.py -q`

## Task 4: Simulation Backends (`synthetic`, `replay`, `ollama`, `live_http`) (`8 min`)
- **Files:** `tests/test_backends.py`, `src/apart_epistemics/backends.py`
- **Action (TDD):** Write failing tests verifying:
  - `SyntheticEpistemicBackend` reproduces the `T=0` vs. `T=0.8` lexical jump while preserving shared-model belief collapse, `personalized_peer` vs. `personalized_advisor` calibration split, and $K$ scaling.
  - `ReplayBackend` replays recorded `RoundRecord` traces bit-for-bit.
  - `OllamaBackend` and `HTTPProviderBackend` build valid JSON payloads (passing `temperature` and `top_p`), parse structured/fallback responses, and raise safe errors without leaking env vars.
- **Verify:** `pytest tests/test_backends.py -q`

## Task 5: Multi-Round Engine & JSONL/CSV Runner (`6 min`)
- **Files:** `tests/test_engine.py`, `src/apart_epistemics/engine.py`
- **Action (TDD):** Write failing integration test for multi-round belief updates (`Round 0 -> Round R`), peer neighborhood message passing, Goldman source-overlap tracking, and deterministic JSONL + `summary.json` + `summary.csv` persistence. Implement engine.
- **Verify:** `pytest tests/test_engine.py -q`

## Task 6: SVG Figures & Interactive Editorial HTML Explorer (`8 min`)
- **Files:** `tests/test_report.py`, `src/apart_epistemics/report.py`, `src/apart_epistemics/templates/dashboard.html`
- **Action (TDD):** Write failing tests asserting `generate_report` writes valid SVGs (`dissociation_decoding.svg`, `epistemic_quadrant.svg`, `round_trajectories.svg`, `k_scaling.svg`) and self-contained `index.html` containing all serialized condition data, interactive controls, and zero broken placeholders. Implement clean SVG generators + Editorial Quantitative Lab Notebook dashboard per `DESIGN.md`.
- **Verify:** `pytest tests/test_report.py -q`

## Task 7: CLI Entrypoint & End-to-End Smoke Run (`4 min`)
- **Files:** `tests/test_cli.py`, `src/apart_epistemics/cli.py`, `src/apart_epistemics/__main__.py`
- **Action (TDD):** Write failing CLI tests for `demo`, `run`, `analyze`, `report`, and error paths. Implement CLI and run `apart-epistemics demo` to generate live outputs in `docs/` and `reports/figures/`.
- **Verify:** `pytest -q && ruff check . && apart-epistemics demo --out reports/results --site docs`

## Task 8: Playwright E2E + Visual Verification (1440px & 390px) (`5 min`)
- **Files:** `tests/e2e/test_dashboard_playwright.py`, `docs/screenshots/`
- **Action:** Install Playwright Chromium, run E2E tests against the built `docs/index.html` verifying interactive filters, round scrubber, track toggles, zero console/page errors, zero horizontal overflow at `1440x900` and `390x844`, and save screenshots (`desktop-1440.png`, `mobile-390.png`) for visual inspection and PR inclusion.
- **Verify:** `pytest tests/e2e/ -q`

## Task 9: CI/CD, GitHub Pages Deployment, PR Review & Repo Polish (`5 min`)
- **Files:** `.github/workflows/ci.yml`, `.github/workflows/pages.yml`, `.github/dependabot.yml`, `README.md`, `reports/SUMMARY.md`, `reports/DECISIONS.md`
- **Action:** Add CI & GitHub Pages workflows, push feature branch, open PR, run `pr-review` audit, fix any Critical/High issues, verify CI green, squash-merge PR, verify live GitHub Pages URL with Playwright, update README & GitHub metadata (`description`, `topics`, `homepage`), and flip repo public once all quality gates pass.

