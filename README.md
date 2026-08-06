# Gita Reasoning Engine

A retrieval-grounded reasoning system that answers questions about the Bhagavad Gita
strictly according to a defined corpus of Pushtimarg-accepted translations and
commentaries — never introducing outside philosophical content.

Built as a production-style AI engineering project: ingestion → embedding → retrieval
→ reasoning → generation → evaluation, with full observability.

## Status

Actively in development, phase by phase. See `docs/phases/` for a written record of
what was built at each stage, why, and how it connects to the rest of the system.

## Architecture

See `docs/architecture/` for Architecture Decision Records (ADRs) covering key
technical decisions (e.g. why Neo4j is deferred, inference hosting strategy).

## Quickstart (local dev)

```bash
# 1. Install dependencies
make install

# 2. Configure environment
cp .env.example .env
# edit .env with real values, including HF_API_TOKEN for the generation model

# 3. Start the full stack (Postgres + API + frontend)
make up

# 4. Open the app
# Frontend: http://localhost:3000
# API docs: http://localhost:8000/docs

# 5. Run tests
make test
```

Run `make help` for the full list of available commands.

Note: the API answers questions by embedding queries with BGE-M3
(runs locally, fine for one query at a time) and generating with a
hosted Qwen model via Hugging Face's Inference API (needs `HF_API_TOKEN`
in `.env`). Bulk corpus embeddings are generated separately on Kaggle
(GPU required) — see `docs/phases/phase_5_6_7_retrieval_reasoning_evaluation.md`
for that one-time round trip.

## Tech Stack

Python 3.12 · uv · FastAPI · PostgreSQL + pgvector · LangGraph · BGE-M3 / BGE
Reranker · Qwen Instruct · Ragas · DeepEval · Langfuse · Docker · Next.js (Phase 8+)

## Project Structure

```
src/gita_engine/     Core Python package (ingestion, retrieval, reasoning, ...)
frontend/             Next.js chat UI
kaggle/               Heavy-compute notebooks (embedding jobs)
tests/                Unit and integration tests
data/                 Raw/processed corpus (gitignored — not committed)
docs/                 Phase writeups and architecture decision records
docker/               Dockerfile + Compose stack for local dev
```

## License

MIT (portfolio/personal project).
