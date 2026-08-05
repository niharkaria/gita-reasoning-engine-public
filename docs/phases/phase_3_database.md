# Phase 3: Database

## What was built

- **SQLAlchemy models** (`src/gita_engine/db/models.py`): `Verse`, `Source`,
  `Translation`, `Commentary`, `QueryLog` — five tables covering canonical
  verse text, translator/commentator metadata, embeddable translation and
  commentary text (pgvector, 1024-dim for BGE-M3), and query history.
- **Session management** (`src/gita_engine/db/session.py`): single
  `get_session()` context manager used by every future module that touches
  the DB.
- **Alembic migrations**: `alembic.ini`, `alembic/env.py` (wired to our
  `Settings.postgres_dsn`, not a hardcoded URL), and the hand-written
  initial migration `alembic/versions/0001_initial_schema.py`.

## Why it exists

Every later phase depends on this shape: ingestion (Phase 4) writes into
`verse`/`translation`/`commentary`, retrieval (Phase 5) queries the
`embedding` columns via the HNSW indexes, and `query_log` is what
evaluation (Phase 7) will eventually score.

## Validation performed

This was not just written and assumed correct — I built pgvector from
source, stood up a real local Postgres 16 instance, and ran the actual
migration against it:

- `alembic upgrade head` — succeeded, created all 5 tables
- Verified via `\dt` / `\di` / `\d translation` that both HNSW vector
  indexes, all foreign keys, and all unique constraints landed exactly as
  designed
- `alembic downgrade base` — cleanly removed everything
- Re-ran `alembic upgrade head` to leave things in a known-good state

## How it connects to the rest of the system

- `Source.is_accepted` is the mechanism that will enforce FR7 (never
  answer from outside the accepted corpus) at the retrieval layer in
  Phase 5 — queries will filter to `is_accepted = True` sources only.
- `Commentary.chunk_index` exists because Phase 4's ingestion pipeline
  will need to split long commentary text before embedding; the schema
  is ready for that even though chunking logic isn't written yet.
- `QueryLog` is intentionally unused by any code yet — it'll be written to
  once Phase 6 (reasoning/generation) produces real answers.

## Open items carried forward

- `Verse.variant_note` is nullable and unused until we know whether any
  Pushtimarg source includes non-standard verses (flagged, not blocking).
- Embedding dimension (1024, BGE-M3) is hardcoded in both `models.py` and
  the migration. If the embedding model ever changes, that's a schema
  migration, not a config toggle — worth remembering before Phase 4.
