# Contributing & Developer Testing Guide

`apart-epistemics` (**Epistemic Prism**) is structured around small, focused subpackages so both researchers and engineers can read, test, and extend each stage independently with zero paid API calls.

## Package Layout

```text
src/apart_epistemics/
  __init__.py          Public API exports and `__all__`
  py.typed             PEP 561 typing marker
  models.py            Pydantic schemas and YAML suite config dataclasses
  parsers.py           Strict JSON + adversarial LaTeX/scientific/currency extractors
  metrics.py           Page decomposition, Pearson H4, Hill N1, Goldman N_eff, Holm-Bonferroni
  summarizer.py        SHA-256 blinded topic labeling and dual-judge Cohen's kappa gate
  cli.py               CLI (`setup`, `demo`, `run`, `analyze`, `report`, `audit-selection`)
  datasets/
    task_bank.py       Ground-truth, open-debate, and numeric core tasks
    population.py      Agent population generator, model pools, selection-bias audit
  backends/
    base.py            Shared `_HTTPChatBackend`, credential masking, `BudgetExceededError`
    providers.py       OpenRouter, Ollama, OpenAI-compatible, Anthropic, Gemini adapters
    replay.py          Deterministic JSONL cassette playback (`ReplayBackend`)
    synthetic.py       Seeded deterministic Track 4 + Numeric Core simulator
  engine/
    track4.py          Multi-round peer graph runner and task-averaged metric aggregator
    numeric_core.py    Crash-resumable `log10` C1/C2/C3 runner and Page residual analyzer
    contrasts.py       Shared paired Wilcoxon + bootstrap CI + Holm-Bonferroni contrast helper
  report/
    figures.py         Native SVG chart generator (4 vector plots)
    site.py            Self-contained HTML lab notebook builder (`importlib.resources`)
  templates/
    dashboard.html     Zero-CDN interactive explorer UI
```

## Developer Commands

Use the `Makefile` shortcuts for local iteration:

```bash
make install     # Create .venv (Python 3.12), install .[dev], install Playwright Chromium
make lint        # Run `ruff check .` and `ruff format --check .`
make format      # Auto-format code and tests with Ruff
make test-fast   # Run unit and integration tests without Playwright (`-m "not e2e"`, ~2s)
make test        # Run all tests including Playwright desktop/mobile E2E
make cov         # Run full suite with statement + branch coverage (`--cov-report=term-missing`)
make demo        # Regenerate offline benchmark summaries, SVGs, and `docs/index.html`
make clean       # Clear local caches and temporary directories
```

## Test Suite Architecture

- **`tests/conftest.py`**: Centralizes reusable fixtures (`sample_agent`, `record_factory`, `core_suite_config`, `mini_demo_bundle`) so each test stays concise.
- **Unit & Integration Tests**: Split by module (`test_models_datasets_parsers.py`, `test_metrics_and_summarizer.py`, `test_backends.py`, `test_engine_track4.py`, `test_engine_numeric.py`, `test_report.py`, `test_cli.py`).
- **Browser E2E (`@pytest.mark.e2e`)**: `tests/e2e/test_dashboard_playwright.py` serves `docs/index.html`, checks interactive filters, keyboard accessibility (`Tab`/`Enter`/`aria-sort`), console/page error counts, unclipped action buttons at `1440x900`, and zero horizontal overflow at `390x844`. Set `SAVE_DASHBOARD_SCREENSHOTS=1` to refresh `docs/screenshots/`.

## Code Quality & PR Checklist

1. **Zero live API calls in tests**: Always mock HTTP transports (`transport=...`), use `ReplayBackend`, or use `SyntheticEpistemicBackend`.
2. **Cross-process determinism**: Never seed or hash from unordered `set` iteration; sort keys before building RNG inputs. Re-verify under varying `PYTHONHASHSEED` values.
3. **Complexity & typing**: Keep cyclomatic complexity at or below 16 (`C901`), pass `ruff check .` and `ruff format --check .`, and reuse shared math/statistics helpers rather than duplicating logic.
4. **No secrets or raw traces**: Never commit `.env` files, API tokens, or generated `reports/results/*.jsonl` files.
