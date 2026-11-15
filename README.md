# Sapere Aude: harnessing the wisdom of the crowd (`apart-epistemics`)

[![CI](https://github.com/vikramwish/apart-epistemics/actions/workflows/ci.yml/badge.svg)](https://github.com/vikramwish/apart-epistemics/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-245A7A.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-2F6F4E.svg)](LICENSE)
[![Live Explorer](https://img.shields.io/badge/live-dashboard-C84B31.svg)](https://vikramwish.github.io/apart-epistemics/)

> Multi-agent simulation and QA testing suite measuring how assistant provenance (`nationality_type`), model count ($K$), decoding temperature, and personalization regimes reshape agenda diversity, epistemic calibration, Goldman source independence, and Page's Diversity Prediction Theorem.

**Live React explorer:** [https://vikramwish.github.io/apart-epistemics/](https://vikramwish.github.io/apart-epistemics/)

---

## About

'Sapere Aude' (_Latin: 'Dare to use your own wisdom'_) is a research workbench for investigating how groups of
AI-assisted agents form beliefs, share topics, and make quantitative estimates.
It pairs a Python simulation and evaluation pipeline with a React interface for
researchers to compare conditions and inspect the records behind each score.

The workspace covers two tracks: **Track 4** compares model diversity, decoding,
provenance, and personalization; the **numeric core** measures crowd error and
estimate diversity using Page's diversity identity. The bundled results are a
seeded synthetic demonstration, not evidence of real-world model effects.

The frontend uses React, TypeScript, Vite, Tailwind, and shadcn/ui components.
It works from saved report data, so exploring the demo needs no API keys or live
model connection.

## Why it exists

Multi-agent deployments increasingly wire multiple simulated humans to a single underlying foundation model or regional model family. Evaluating such ecosystems solely with embedding similarity or surface phrasing hides critical failure modes identified in recent literature (Kim et al. 2025/2026; Qiu et al. 2026; Sourati et al. 2025):

1. **Decoding Veil vs. Epistemic Monoculture**: Stochastic sampling (`T=0.8`) restores lexical variety over greedy decoding (`T=0.0, top_p=0.0`) without diversifying underlying concepts, arguments, or correlated sources.
2. **National & Post-Training Provenance (`nationality_type x K`)**: Scaling $K$ within one frontier ecosystem differs sharply from cross-national pools (US + CN + EU) or same-base divergent SFT models (e.g., `Qwen3.8-27B` paired with `Arete-27B`).
3. **Sycophantic Personalization Split**: Peer-mirroring assistants can preserve diverse topical agendas while silently degrading calibration (`Brier`, confident-wrong errors).
4. **Page's Diversity Prediction Theorem on Quantitative Estimates**: Under social feedback (`C2` population median sharing), crowd spread collapses mathematically via `Crowd Error = Avg Individual Error - Diversity`.

Designed for **Apart Research Track 4 (Open Track: Deployment, Adoption, and Impact)** with lightweight peer-exchange interactions (avoiding bloated orchestration frameworks) plus a crash-resumable numeric core.

---

## Key Empirical Highlights




---

## Features

- **Dual-Track Simulation Engine** ([`engine/`](src/apart_epistemics/engine/)): Lightweight peer-exchange communication graph supporting both categorical calibration/open-debate rounds (`GROUND_TRUTH`, `OPEN_DEBATE`) and damped `log10` quantitative belief updating (`C1/C2/C3`), with shared Holm-Bonferroni contrast analysis ([`contrasts.py`](src/apart_epistemics/engine/contrasts.py)).
- **Blinded Dual-Judge Topic Summarizer** ([`summarizer.py`](src/apart_epistemics/summarizer.py)): Strips condition, model family, nationality, and round metadata, hashes deterministic `blind_id`s, consolidates a shared global codebook, enforces Cohen's $\kappa \ge 0.70$, and exports stratified `audit_human_check.csv`.
- **Protocol-Clean Multi-Backend Adapters** ([`backends/`](src/apart_epistemics/backends/)): Deterministic `SyntheticEpistemicBackend`, offline `ReplayBackend`, local `OllamaBackend`, OpenRouter with automatic `:online` suffix stripping and `allow_fallbacks: false` provider pinning, plus OpenAI, Anthropic, and Gemini adapters built over a shared `_HTTPChatBackend` base with credential masking.
- **Crash-Resumable JSONL & Budget Guard**: Skips completed `(condition_id, question_id, replicate, round_idx, agent_id)` cells on restart, retries once on unparseable replies, logs `miss=True` safely without corrupting beliefs, and halts cleanly on `BudgetExceededError`.
- **Self-Contained Interactive Explorer & SVG Pipeline** ([`report/`](src/apart_epistemics/report/), [`dashboard.html`](src/apart_epistemics/templates/dashboard.html)): Zero external CDN dependencies; keyboard-accessible `1440px` desktop and `390px` mobile views with live round scrubber, family filters, sortable ledger (`aria-sort`), and slide-over trace drawer.

---

## Architecture & Project Structure

```mermaid
flowchart LR
    CFG["YAML Config / Datasets<br/>(datasets/, models.py)"] --> ENG["Simulation Engine<br/>(engine/)"]
    BE["Backends<br/>(backends/: synthetic / ollama / openrouter / replay)"] --> ENG
    ENG --> PAR["Strict Parsers<br/>(parsers.py)"]
    ENG --> SUM["Blinded Dual-Judge Summarizer<br/>(summarizer.py)"]
    PAR --> MET["Epistemic & Statistical Metrics<br/>(Page, Hill N1, Goldman, Holm)<br/>(metrics.py)"]
    SUM --> MET
    MET --> REP["SVG Figures & Lab Notebook<br/>(report/, dashboard.html)"]
    REP --> DOCS["docs/index.html & reports/figures/*.svg"]
```

- **Library code ([`src/apart_epistemics/`](src/apart_epistemics/))**: Modular subpackages for [`datasets/`](src/apart_epistemics/datasets/), [`backends/`](src/apart_epistemics/backends/), [`engine/`](src/apart_epistemics/engine/), and [`report/`](src/apart_epistemics/report/), alongside [`models.py`](src/apart_epistemics/models.py), [`parsers.py`](src/apart_epistemics/parsers.py), [`metrics.py`](src/apart_epistemics/metrics.py), [`summarizer.py`](src/apart_epistemics/summarizer.py), [`cli.py`](src/apart_epistemics/cli.py), and PEP 561 [`py.typed`](src/apart_epistemics/py.typed).
- **Testing & Automation ([`tests/`](tests/), [`Makefile`](Makefile), [`CONTRIBUTING.md`](CONTRIBUTING.md))**: Shared pytest fixtures in [`tests/conftest.py`](tests/conftest.py), per-module unit/integration files, strict `e2e` Playwright marker, Ruff complexity/format gates (`C90 <= 16`), and GitHub Actions branch-coverage CI.

---

## Python quick start

**Hackathon teammates:** see [HACKATHON.md](HACKATHON.md) for the full onboarding
and testing checklist, Windows commands, troubleshooting, and live-run caveats.
No API keys or model downloads are needed for the offline demo.

```bash
git clone https://github.com/vikramwish/apart-epistemics.git
cd apart-epistemics

make install     # Python 3.11+; no uv required
make check       # Lint, fast tests, and small synthetic demo in tmp/
make serve       # Open http://127.0.0.1:8000; Ctrl-C to stop

# Optional: browser tests (separate browser download)
make install-browser
make test

# Optional: rebuild FULL benchmark and tracked dashboard (zero API cost)
make demo
```

Live cloud runs read exported environment variables; `.env` is not automatically
loaded. Read the safety limitations in [HACKATHON.md](HACKATHON.md) first.

---

## Testing

The Python suite includes regression coverage for actual model routing, prompted JSON parsing,
manifest-validated resume, incomplete analysis, persistent request budgets, secret
redaction, trace preservation, and independent-judge availability, alongside the
math, provider, CLI, and browser checks. Tests run offline with zero API charges. See [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`Makefile`](Makefile) for workflows:

```bash
make lint       # ruff check . && ruff format --check .
make test-fast  # pytest -m "not e2e" -v (~2s)
make cov        # pytest --cov=apart_epistemics --cov-report=term-missing -v
```

The React frontend adds 20 unit tests and 10 desktop/mobile browser checks:

```bash
make ui-check   # TypeScript, formatting, unit tests, production build
make ui-e2e     # Browser checks; install frontend Playwright Chromium first
```

Use the CI badge and your local test output for current verification status.
Browser tests require a working Playwright Chromium installation and permission
to launch subprocesses; fast tests do not require a browser binary.

---

## Roadmap

- Execute multi-seed live OpenRouter runs across `US_ONLY`, `CN_ONLY`, `DIVERGENT_SFT`, and `CROSS_NATIONAL` pools using `--budget` caps and deterministic `ReplayBackend` archival.
- Collect 3-rater human verification annotations against `reports/results/audit_human_check.csv` to calibrate automated dual-judge topic labeling on domain debates.
- Extend `audit-selection` to ingest cold-start OpenRouter evaluation CSVs automatically prior to benchmark sweeps.

## What I learned

- Decoupling surface phrasing variance (`pairwise_jaccard_distance`) from conceptual agenda diversity (`effective_topic_count`, `goldman_effective_sources`) immediately flags temperature-induced diversity illusions.
- Treating LLM numeric/JSON parsing and resume state as first-class crash-safe contracts eliminates expensive lost runs before multi-model API budgets are touched.
- Inspecting actual Playwright screenshots at both `1440px` and `390px` catches SVG label overlap and right-edge button clipping that pure DOM unit tests miss.


## Experiment integrity

Live multi-model conditions require explicit provider-compatible `models` lists;
`--model` selects the agent simulator and numeric assistant. Actual model/provider
identities, parse status, and prompts are saved in new live traces.

Numeric resumes validate a saved manifest. Track 4 refuses to overwrite existing
traces and flushes each completed observation. Incomplete or legacy unverified
numeric runs do not publish final comparisons. Default seed labels do not qualify
as independent topic judging: reliability is unavailable until two judges run.
Source-diversity and persuasion metrics remain explicitly labeled assumption-based
proxies; pooled H4 correlation is descriptive, not a causal effect.

`--budget` reserves `--request-allowance` USD before every HTTP attempt across both
tracks (default allowance: 1 USD). Reservations persist in `budget.json` and include
failed requests. Set the allowance above the largest possible per-request charge
for your selected models. This controls authorized request allowances, not actual
provider billing. See [live-run details](HACKATHON.md#5-live-run-limitations--agree-with-the-team-before-spending).

The [audit and resolution notes](docs/codebase-audit.md) explain the failures and
regression coverage. Gemini credentials now use the API header documented in the
[official reference](https://ai.google.dev/api), rather than request URLs.
