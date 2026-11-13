.PHONY: help install install-browser lint format test test-fast test-e2e check cov smoke serve demo clean ui-install ui-dev ui-check ui-build ui-preview ui-e2e

PYTHON := .venv/bin/python
BOOTSTRAP_PYTHON ?= python3
PORT ?= 8000
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff
CLI := .venv/bin/apart-epistemics

help: ## Show available developer targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create/reuse .venv and install package + test tools (Python 3.11+)
	$(BOOTSTRAP_PYTHON) -m venv .venv
	$(PYTHON) -m pip install -e ".[dev]"

install-browser: ## Install Chromium for browser E2E (Linux: add --with-deps manually)
	$(PYTHON) -m playwright install chromium

lint: ## Run Ruff static analysis and format check
	$(RUFF) check .
	$(RUFF) format --check .

format: ## Format source and test files with Ruff
	$(RUFF) format .

test-fast: ## Run unit and integration tests without browser E2E (~2s)
	$(PYTEST) -m "not e2e" -v

test: ## Run full test suite including Playwright E2E
	$(PYTEST) -v

test-e2e: ## Run only browser checks (requires make install-browser)
	$(PYTEST) -m e2e -v

check: lint test-fast smoke ## Run the offline teammate acceptance checks

smoke: ## Build a small offline demo in ignored tmp/ (never overwrite published results)
	$(CLI) setup --out tmp/smoke.yaml --agents 8 --rounds 2 --tasks 2
	$(CLI) demo --config tmp/smoke.yaml --out tmp/smoke-results --site tmp/smoke-site
	$(CLI) analyze --input tmp/smoke-results

serve: ## Preview the checked-in full dashboard at http://127.0.0.1:8000
	$(PYTHON) -m http.server $(PORT) --bind 127.0.0.1 --directory docs

ui-install: ## Install the React explorer dependencies (Node 22.12+)
	cd frontend && npm ci

ui-dev: ## Start the shadcn/ui React results explorer (see printed URL)
	cd frontend && npm run dev

ui-check: ## Typecheck, format-check, test, and build the React explorer
	cd frontend && npm run lint && npm test && npm run build

ui-build: ## Build frontend/dist without changing the Python report
	cd frontend && npm run build

ui-preview: ## Serve the production React build (see printed URL)
	cd frontend && npm run preview

ui-e2e: ui-build ## Test React desktop/mobile UI (requires frontend Playwright Chromium)
	cd frontend && npm run test:e2e

cov: ## Run full test suite with statement + branch coverage report
	$(PYTEST) --cov=apart_epistemics --cov-report=term-missing -v

demo: ## Regenerate offline benchmark results, SVG figures, and docs site
	$(CLI) demo --out reports/results --site docs

clean: ## Remove local cache and temporary test artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov tmp build dist
	find src tests -type d -name "__pycache__" -exec rm -rf {} +
