# ADR 0001: Defer Neo4j Until a Concrete Graph-Shaped Need Exists

**Status:** Accepted
**Date:** 2026-08-05

## Context

The original stack list included Neo4j for modeling relationships between verses,
commentaries, and philosophical concepts. At Phase 0 planning time, no concrete
query pattern had been identified that requires graph traversal rather than
relational joins or vector similarity search.

## Decision

Neo4j is **not** included in the Phase 3 core database design. PostgreSQL (with
pgvector) will handle both structured relational data (verse/commentary metadata)
and semantic retrieval for v1.

Neo4j will be reconsidered if, during Phase 5/6 (retrieval/reasoning), we hit a
concrete need such as multi-hop reasoning across thematically linked verses that
relational queries handle poorly.

## Consequences

- One fewer service to run, configure, and back up during early phases.
- Faster path to a working end-to-end system.
- If a graph-shaped need does emerge later, we accept the cost of a schema
  migration/backfill at that point — deemed acceptable given corpus size
  (thousands, not millions, of records).
- `NEO4J_*` variables remain in `.env.example` as a placeholder so the
  config surface doesn't need rework if/when this is revisited.
