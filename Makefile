# Makefile
#
# Why this file exists:
#   Every command below wraps something you'd otherwise have to remember
#   (the exact uv/docker/pytest incantation). This is the single reference
#   for "how do I run X" — for you, and for anyone reviewing the repo.

.PHONY: help install sync lint format typecheck test cov up down logs migrate clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create venv and install all dependency groups (dev)
	uv venv
	uv pip install -e ".[dev,ingestion,eval]"

sync: ## Sync environment to lockfile exactly
	uv sync --all-extras

lint: ## Run ruff lint checks
	uv run ruff check src tests

format: ## Auto-format code with ruff
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck: ## Run mypy static type checking
	uv run mypy src

test: ## Run test suite
	uv run pytest

cov: ## Run tests with coverage report
	uv run pytest --cov-report=html
	@echo "Coverage report: htmlcov/index.html"

up: ## Start local dev stack (Postgres + API) via Docker Compose
	docker compose --env-file .env -f docker/docker-compose.yml up -d

down: ## Stop local dev stack
	docker compose --env-file .env -f docker/docker-compose.yml down

logs: ## Tail logs from the dev stack
	docker compose --env-file .env -f docker/docker-compose.yml logs -f

migrate: ## Apply Alembic migrations (Phase 3+)
	uv run alembic upgrade head

clean: ## Remove caches, build artifacts, and coverage output
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
