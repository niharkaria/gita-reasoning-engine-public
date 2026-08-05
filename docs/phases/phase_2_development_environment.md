# Phase 2: Development Environment

## What was built

- **Git repository** initialized at project root.
- **uv-managed Python project** (`pyproject.toml`): Python 3.12, core deps
  (FastAPI, SQLAlchemy, pgvector, LangGraph, structlog, etc.) plus three
  optional dependency groups — `ingestion` (PyMuPDF, PaddleOCR), `eval`
  (Ragas, DeepEval, Langfuse), `dev` (ruff, mypy, pytest, pre-commit).
- **Typed configuration** (`src/gita_engine/core/config.py`): a single
  `pydantic-settings` `Settings` class loading from `.env`, validated at
  startup, cached via `lru_cache`.
- **Structured logging** (`src/gita_engine/core/logging.py`): `structlog`
  configured for human-readable console output in dev, JSON in
  staging/production.
- **Docker**: `docker/Dockerfile.api` (multi-stage build, slim runtime
  image, explicitly excludes the heavy `ingestion` extra) and
  `docker/docker-compose.yml` (Postgres + pgvector, API service).
- **VS Code workspace config**: format-on-save via ruff, recommended
  extensions, pytest integration.
- **Pre-commit hooks**: ruff lint/format + standard hygiene checks
  (trailing whitespace, large files, private key detection). mypy
  deliberately excluded from pre-commit (runs in CI/Makefile instead —
  see "Decisions" below).
- **CI** (`.github/workflows/ci.yml`): lint, typecheck, test on every push/PR.
- **Makefile**: single reference for install/lint/format/typecheck/test/
  up/down/migrate/clean.
- **First real test** (`tests/unit/test_config.py`): validates `Settings`
  loads and `get_settings()` caches correctly. Ran it — passes.
- **ADR 0001**: formalized the Phase 0/1 decision to defer Neo4j.
- **README + CHANGELOG**: project overview, quickstart, and a running log
  of what's landed.

## Why it exists

This phase exists so every subsequent phase has somewhere real to plug
into: config and logging are used by literally every other module we'll
write, Docker gives us environment parity between your machine and
Kaggle/eventual deployment, and CI/pre-commit catch mistakes before they
compound. None of this is generic scaffolding — I validated it by
actually installing the package, running the test suite, and confirming
logging emits real output (not just "the code looks right").

## How it connects to the rest of the system

- Every future module (`ingestion`, `retrieval`, `reasoning`, etc.) will
  import `get_settings()` and `get_logger(__name__)` rather than reaching
  for `os.environ` or `print()`.
- Phase 3 (Database) will add SQLAlchemy models under `src/gita_engine/db/`
  and Alembic migrations, using `settings.postgres_dsn` from this phase.
- The `ingestion` and `eval` extras defined here are what Phase 4/7 will
  actually install and use.
- `docker-compose.yml` currently only has `postgres` + `api` — Neo4j (or
  anything else) gets added here only if/when a later ADR reverses
  ADR 0001.

## Decisions made this phase (worth flagging explicitly)

1. **mypy excluded from pre-commit, kept in CI/Makefile.** mypy pre-commit
   hooks often lag on third-party stub resolution and slow down every
   commit. Running it via `make typecheck` and in CI gives the same
   guarantee without the commit-time friction.
2. **`ingestion` and `eval` deps are optional extras, not core deps.** The
   API container doesn't need PyMuPDF/PaddleOCR (ingestion runs on Kaggle)
   or Ragas/DeepEval/Langfuse (only needed during eval runs) — keeps the
   production image smaller and avoids unnecessary system-level deps
   (OCR needs Poppler etc.) in the serving container.
3. **Dockerfile references `gita_engine.api.main:app`, which doesn't exist
   yet.** This is intentional forward-definition — the infra is defined
   now, the app code lands when we build the API in later phases. The
   image won't build/run correctly until then, and that's expected.

## Suggested improvements / things to revisit later

- Once Phase 3 lands, add an Alembic `env.py` wired to `settings.postgres_dsn`
  and wire `make migrate` to something real (currently a stub target).
- Consider adding `ruff`'s `S` (bandit-equivalent security) rule set once
  the API layer exists and there's real attack surface to lint against.
- `docker-compose.yml`'s `api` service will fail to start until Phase 6+
  (no `main.py` yet) — fine for now, just don't `make up` expecting the
  API container to actually come up until then. Postgres alone will work.
