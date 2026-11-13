# Autopilot Summary (`apart-epistemics`)

- **Repository:** `https://github.com/vikramwish/apart-epistemics`
- **Branch:** `autopilot/feat-simulation-harness` -> PR #1 ready for squash-merge to `main`
- **Live Explorer (post-Pages deploy):** `https://vikramwish.github.io/apart-epistemics/`

## Shipped Artifacts & Capabilities
1. **Verified Research Foundation:** 30-citation literature review (`LITERATURE_REVIEW.md`), specification (`SPEC.md`), implementation plan (`PLAN.md`), and Editorial Quantitative Lab Notebook design spec (`DESIGN.md`).
2. **Track 4 & Numeric Core Package (`src/apart_epistemics/`):**
   - `models.py`, `datasets.py` (8 Ground-Truth tasks, 8 Open-Debate tasks, 10 Numeric Core tasks, 24 global agents, selection-bias audit), `parsers.py` (adversarial LaTeX/sci/currency/JSON extractors).
   - `metrics.py` (Page's Theorem decomposition, Pearson H4, Brier/ECE/Confident-Wrong, Shannon/Hill $N_1$, minority topic survival, SAS bimodality, Goldman Kish $N_{\text{eff}}$, distinct-2 & Jaccard distance, Cohen's $\kappa$, exact/normal paired Wilcoxon, percentile bootstrap CI, Holm-Bonferroni).
   - `summarizer.py` (deterministic SHA-256 `blind_id` metadata stripping, global codebook consolidation, dual-judge $\kappa$ reliability gate, CSV audit exporter).
   - `backends.py` (`SyntheticEpistemicBackend`, `ReplayBackend`, local `OllamaBackend`, pinned `:online`-stripped `OpenRouterBackend`, plus OpenAI/Anthropic/Gemini HTTP adapters with credential masking & `BudgetExceededError` guard).
   - `engine.py` (`run_suite`, `build_numeric_prompts`, crash-resumable `run_numeric_core` with trailing-truncated line recovery, `analyze_numeric_core`).
   - `report.py` & `templates/dashboard.html` (4 static SVGs + self-contained interactive Lab Notebook dashboard).
   - `cli.py` & `__main__.py` (`setup`, `demo`, `run`, `analyze`, `report`, `audit-selection`).
3. **Verification Results (Observed):**
   - **Ruff Lint:** `ruff check .` -> `All checks passed!`
   - **Test Suite (`pytest -v`):** `37 passed` across unit, integration, cross-process `PYTHONHASHSEED` determinism, crash-truncated JSONL recovery, CLI, SVG/HTML XSS report safety, and Playwright E2E tests (`1440x900` & `390x844`), `93%` total coverage.
   - **Benchmark Demo (`apart-epistemics demo`):** `11 conditions`, `19,008` Track 4 records, dual-judge `kappa = 1.0`, `3,456` numeric records, max Page identity residual `1.09e-16` (`C2 vs C1` diversity `mean_diff = -0.013698`, `p_holm = 0.001953`; `C3 vs C1` crowd error `mean_diff = -0.002464`, `p_holm = 0.00293`).
   - **Local Ollama Live Smoke (`llama3.1:8b`):** Extracted `Final estimate: 6853.` (`log10 = 3.8359`, `58` prompt / `104` eval tokens) with zero cloud API spend.
4. **CI/CD & Documentation:**
   - `.github/workflows/ci.yml` (`ruff check .`, Chromium Playwright install, `pytest -v` on Python `3.11` & `3.12`), `.github/workflows/pages.yml` (`actions/deploy-pages@v4`), `.github/dependabot.yml`.
   - Portfolio-grade `README.md` with Mermaid architecture diagram, verified metrics table, and desktop/mobile screenshots.

## Open Risks / Owner Optional Actions (see `reports/NEEDS_OWNER.md`)
- All cloud provider keys (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) are unset in this environment; all synthetic & local Ollama validations passed cleanly with zero paid API calls. When keys are funded/exported, live sweeps can run immediately via `apart-epistemics run --backend openrouter --budget 10`.
- Optional human validation pass: annotate 40 sampled blinded traces in `reports/results/audit_human_check.csv`.
