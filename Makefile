.PHONY: help install lint format test test-fast cov demo clean

PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff
CLI := .venv/bin/apart-epistemics

help: ## Show available developer targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install package and dev/test tools into .venv
	uv venv .venv --python 3.12
	uv pip install -e ".[dev]"
	.venv/bin/playwright install chromium

lint: ## Run Ruff static analysis and format check
	$(RUFF) check .
	$(RUFF) format --check .

format: ## Format source and test files with Ruff
	$(RUFF) format .

test-fast: ## Run unit and integration tests without browser E2E (~2s)
	$(PYTEST) -m "not e2e" -v

test: ## Run full test suite including Playwright E2E
	$(PYTEST) -v

cov: ## Run full test suite with statement + branch coverage report
	$(PYTEST) --cov=apart_epistemics --cov-report=term-missing -v

demo: ## Regenerate offline benchmark results, SVG figures, and docs site
	$(CLI) demo --out reports/results --site docs

clean: ## Remove local cache and temporary test artifacts
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov tmp build dist
	find src tests -type d -name "__pycache__" -exec rm -rf {} +
