.PHONY: help lint format fix test typecheck install clean

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies with uv
	uv sync --all-extras
	uv pip install -e .

lint:  ## Run ruff linter
	uv run ruff check src/ examples/

format:  ## Format Python, TOML, and Markdown files
	uv run ruff format src/ examples/
	find . -name "*.toml" -not -path "./.venv/*" -not -path "*/.venv/*" -not -path "./.git/*" -not -path "./.conductor/*" -exec taplo format {} +
	find . -name "*.md" -not -path "./.venv/*" -not -path "*/.venv/*" -not -path "./.git/*" -not -path "./.conductor/*" -exec uv run mdformat --wrap 100 {} +

fix:  ## Fix linting issues and format all code
	uv run ruff check --fix src/ examples/
	uv run ruff format src/ examples/
	find . -name "*.toml" -not -path "./.venv/*" -not -path "*/.venv/*" -not -path "./.git/*" -not -path "./.conductor/*" -exec taplo format {} +
	find . -name "*.md" -not -path "./.venv/*" -not -path "*/.venv/*" -not -path "./.git/*" -not -path "./.conductor/*" -exec uv run mdformat --wrap 100 {} +

test:  ## Run tests with pytest
	uv run pytest
	cd examples/autogen-dev-team && uv run --extra test pytest tests/ -v

typecheck:  ## Run type checking with mypy
	uv run mypy src/ examples/

typecheck-src:  ## Run type checking on src/ only
	uv run mypy src/

ci-check:  ## Run all CI checks (lint, format check, typecheck)
	@echo "🔍 Running linting checks..."
	uv run ruff check src/ examples/
	@echo "✅ Linting passed"
	
	@echo "🎨 Checking code formatting..."
	@if ! git diff --exit-code > /dev/null 2>&1; then \
		echo "❌ There are uncommitted changes. Please commit or stash them first."; \
		exit 1; \
	fi
	uv run ruff format src/ examples/
	find . -name "*.toml" -not -path "./.venv/*" -not -path "*/.venv/*" -not -path "./.git/*" -not -path "./.conductor/*" -exec taplo format {} +
	@if ! git diff --exit-code > /dev/null 2>&1; then \
		echo "❌ Code formatting issues found. Please run 'make format' and commit the changes."; \
		git diff --name-only; \
		exit 1; \
	fi
	@echo "✅ Code formatting is correct"
	
	@echo "🔍 Running type checking..."
	uv run mypy src/
	@echo "✅ Type checking passed"
	
	@echo "🎉 All CI checks passed!"

clean:  ## Clean build artifacts and cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	rm -rf build/ dist/ 2>/dev/null || true