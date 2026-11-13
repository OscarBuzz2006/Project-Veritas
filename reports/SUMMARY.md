# Autopilot Summary (`apart-epistemics`)

- **Repository:** `https://github.com/vikramwish/apart-epistemics`
- **Active Branch:** `autopilot/refactor-optimize-readability-testing` (PR #2 into `main`, following merged PR #1)
- **Live Explorer:** `https://vikramwish.github.io/apart-epistemics/`

## Shipped Artifacts & Capabilities
1. **Verified Research Foundation:** 30-citation literature review (`LITERATURE_REVIEW.md`), specification (`SPEC.md`), implementation plan (`PLAN.md`), and Editorial Quantitative Lab Notebook design spec (`DESIGN.md`).
2. **Modular Track 4 & Numeric Core Package (`src/apart_epistemics/`):**
   - `__init__.py`, `py.typed`, `models.py`, `cli.py` (`setup`, `demo`, `run`, `analyze`, `report`, `audit-selection`).
   - `datasets/` (`task_bank.py`, `population.py`): 8 Ground-Truth tasks, 8 Open-Debate tasks, 10 Numeric Core tasks, 24 global agents, and selection-bias filter audit.
   - `parsers.py`: Strict JSON and adversarial LaTeX/scientific/currency extractors with overflow safety (`1e9999` -> `miss=True`) and shared `clamp`.
   - `metrics.py`: Page's Theorem decomposition, Pearson H4, Brier/ECE/Confident-Wrong, Shannon/Hill $N_1$, minority topic survival, SAS bimodality, Goldman Kish $N_{\text{eff}}$, $O(U^2)$ frequency-weighted distinct-2 & Jaccard distance, Cohen's $\kappa$, exact/normal paired Wilcoxon, percentile bootstrap CI, Holm-Bonferroni, and shared `clamp`/`quantile_sorted` helpers.
   - `summarizer.py`: Deterministic SHA-256 `blind_id` metadata stripping, global codebook consolidation, dual-judge $\kappa$ reliability gate, CSV audit exporter.
   - `backends/` (`base.py`, `providers.py`, `replay.py`, `synthetic.py`): Shared `_HTTPChatBackend` base with credential masking and `BudgetExceededError` guard, pinned `:online`-stripped `OpenRouterBackend`, `OllamaBackend`, OpenAI/Anthropic/Gemini HTTP adapters, `ReplayBackend`, and decomposed `SyntheticEpistemicBackend`.
   - `engine/` (`track4.py`, `numeric_core.py`, `contrasts.py`): Pre-grouped task aggregation in `run_suite`, non-finite `NaN` fallback guard in live stepping, crash-resumable `run_numeric_core`, and shared `compute_paired_holm_contrasts`.
   - `report/` (`figures.py`, `site.py`) & `templates/dashboard.html`: 4 native SVG figures, `importlib.resources` template bundling, and keyboard-accessible interactive Lab Notebook (`tabindex="0"`, `aria-label`, `aria-sort`).
3. **Developer Ergonomics & Test Suite (`CONTRIBUTING.md`, `Makefile`, `tests/`):**
   - `Makefile` targets (`install`, `lint`, `format`, `test-fast`, `test`, `cov`, `demo`, `clean`) and `CONTRIBUTING.md` contributor/testing guide.
   - `tests/conftest.py` shared fixtures and focused test modules (`test_backends.py`, `test_engine_track4.py`, `test_engine_numeric.py`, `test_models_datasets_parsers.py`, `test_metrics_and_summarizer.py`, `test_report.py`, `test_cli.py`, `tests/e2e/test_dashboard_playwright.py` with strict `e2e` marker).
4. **Verification Results (Observed):**
   - **Ruff Lint & Format:** `ruff check .` -> `All checks passed!`; `ruff format --check .` -> clean (`0` diffs).
   - **Test Suite:** `pytest -m "not e2e" -v` -> `43 passed, 1 deselected`; full `pytest --cov=apart_epistemics --cov-report=term-missing -v` -> `44 passed in 6.81s`, `93%` total statement + branch coverage.
   - **Benchmark Demo (`apart-epistemics demo`):** `11 conditions`, `19,008` Track 4 records, dual-judge `kappa = 1.0`, `3,456` numeric records, max Page identity residual `1.09e-16` (`git diff reports/results/summary.csv reports/results/numeric_analysis.json` empty after refactor).
5. **CI/CD & Documentation:**
   - `.github/workflows/ci.yml` (`ruff check .`, `ruff format --check .`, Chromium Playwright install, `pytest --cov=apart_epistemics --cov-report=term-missing -v` on Python `3.11` & `3.12`), `.github/workflows/pages.yml` (`actions/deploy-pages@v4`), `.github/dependabot.yml`.

## Open Risks / Owner Optional Actions (see `reports/NEEDS_OWNER.md`)
- All cloud provider keys (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) remain unset in this environment; all synthetic, offline, and Playwright validations run with zero paid API calls. When keys are funded/exported, live sweeps can run immediately via `apart-epistemics run --backend openrouter --budget 10`.
- Optional human validation pass: annotate 40 sampled blinded traces in `reports/results/audit_human_check.csv`.
