SHELL := /bin/sh

UV ?= uv
PACKAGE ?= researchpilot
TEST_DIR ?= tests/unit

.DEFAULT_GOAL := help

.PHONY: help install sync test lint format format-check typecheck check run build clean

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*## "; printf "Usage: make <target>\n\nTargets:\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: sync ## Create or update the virtual environment

sync: ## Synchronize dependencies from uv.lock
	$(UV) sync

test: ## Run the unit test suite
	$(UV) run python -m unittest discover -s $(TEST_DIR) -p 'test*.py' -v

lint: ## Run Ruff lint checks
	$(UV) run --with ruff ruff check .

format: ## Format Python files with Ruff
	$(UV) run --with ruff ruff format .

format-check: ## Check Python formatting without changing files
	$(UV) run --with ruff ruff format --check .

typecheck: ## Run the Pyright type checker
	$(UV) run --with pyright pyright

check: format-check lint typecheck test ## Run all quality checks and tests

run: ## Run the application
	$(UV) run $(PACKAGE)

build: ## Build the distributable package
	$(UV) build

clean: ## Remove generated build and tool artifacts
	rm -rf build dist .pytest_cache .ruff_cache .mypy_cache .pyright
	rm -rf *.egg-info