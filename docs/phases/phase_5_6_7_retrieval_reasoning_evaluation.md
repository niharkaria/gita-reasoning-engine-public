# Phases 5-7: Retrieval, Reasoning, and Evaluation

## What was built

**Phase 5 — Retrieval**
- `embedding/export_for_embedding.py`: queries Postgres for every
  translation/commentary row still missing an embedding, writes them to
  JSONL for upload to Kaggle.
- `embedding/kaggle_generate_embeddings.py`: runs BGE-M3 on Kaggle's GPU
  to encode that JSONL, writes vectors back out.
- `embedding/import_embeddings.py`: reads Kaggle's output, writes the
  vectors into the `translation.embedding` / `commentary.embedding`
  pgvector columns.
- `retrieval/retriever.py`: `embed_query()` (BGE-M3, run locally — fine
  for one short query at a time, unlike bulk corpus encoding) and
  `retrieve()` (pgvector cosine-distance search, filtered to
  `is_accepted=True` sources, merged across translation+commentary,
  top-k by similarity).

**Phase 6 — Reasoning**
- `generation/llm_client.py`: wraps a Hugging Face Inference API call to
  the configured Qwen model.
- `reasoning/graph.py`: LangGraph pipeline (`embed -> retrieve ->
  generate`), a system prompt that enforces citation-grounded,
  corpus-only answers, and `answer_question()` as the main entry point,
  which also writes to `query_log`.

**Phase 7 — Evaluation**
- `evaluation/golden_set.json`: hand-curated question -> expected-verse
  pairs.
- `evaluation/evaluate_retrieval.py`: computes Hit Rate@k and Mean
  Reciprocal Rank against the golden set — fully local, no LLM judge
  needed.

## What's actually validated vs. what needs your machine

Being direct about this rather than implying everything was proven:

**Tested end-to-end against real Postgres, in this session:**
- `export_for_embedding.py` — ran against real DB, produced correct JSONL
- `import_embeddings.py` — ran with placeholder (random) vectors,
  confirmed they land correctly in the pgvector columns with correct
  dimensions
- `retriever.py`'s `retrieve()` — ran real queries against real Postgres,
  confirmed: correct joins, correct `is_accepted` filtering (verified by
  flipping a source's flag and confirming its rows disappear from
  results), correct top-k merging across both tables
- `reasoning/graph.py`'s full pipeline wiring — ran end-to-end with
  `embed_query` and `generate` mocked (since this sandbox has no network
  access to Hugging Face and no GPU), confirmed: correct prompt
  construction (system prompt rules present, passages formatted with
  citations, question included), correct state flow through all three
  nodes, correct `query_log` write with accurate retrieved-verse-ID
  tracking
- `evaluate_retrieval.py` — ran end-to-end with `embed_query` mocked,
  confirmed correct hit/miss detection, correct MRR computation, correct
  golden-set JSON loading

**NOT tested — genuinely needs your machine/accounts:**
- `kaggle_generate_embeddings.py` — never executed; written carefully
  against documented BGE-M3/sentence-transformers usage, but Kaggle's
  exact environment (package versions, dataset mount paths) can only be
  confirmed by actually running it there
- `embed_query()`'s real BGE-M3 behavior — the caching pattern and call
  shape are correct Python, but actual model download/inference timing
  on your 8GB/i3 laptop is unverified
- `llm_client.py`'s real HTTP call — the request/response shape follows
  HF's documented chat-completions-style Inference API, but hosted-API
  shapes change; verify against current HF docs and your actual
  `HF_API_TOKEN` before trusting it
- Real embedding quality / retrieval accuracy — everything above was
  validated with **random placeholder vectors**, which by definition
  can't demonstrate real semantic retrieval quality. The Hit Rate/MRR
  numbers you'll see with random vectors are meaningless; re-run
  `evaluate_retrieval.py` once real BGE-M3 embeddings are loaded to get
  a real signal.

## How to actually run this for real

1. Load the real corpus if not already done:
   `uv run python src/gita_engine/ingestion/load_to_db.py full_book.txt`
2. Export for embedding:
   `uv run python src/gita_engine/embedding/export_for_embedding.py embedding_export.jsonl`
3. Upload `embedding_export.jsonl` to Kaggle as a dataset, run
   `kaggle_generate_embeddings.py` there (GPU accelerator on), download
   `embedding_output.jsonl`.
4. Import back locally:
   `uv run python src/gita_engine/embedding/import_embeddings.py embedding_output.jsonl`
5. Get a Hugging Face API token, add it to `.env` as `HF_API_TOKEN`.
6. Try a real question:
   ```python
   from gita_engine.reasoning.graph import answer_question
   result = answer_question("What does Krishna say about the eternal soul?")
   print(result["answer"])
   ```
7. Run real evaluation:
   `uv run python src/gita_engine/evaluation/evaluate_retrieval.py`

## Known gaps / next steps

- `evaluate_retrieval.py` only measures retrieval, not answer quality.
  Faithfulness/groundedness scoring (does the generated answer actually
  stick to the retrieved passages?) needs an LLM-judge metric — Ragas is
  already in `pyproject.toml`'s `eval` extra for this, but wiring it to
  our HF-hosted Qwen client (rather than Ragas' default OpenAI
  assumption) is unbuilt.
- The golden set has only 3 questions, covering 3 of 18 chapters. Worth
  expanding before trusting the retrieval metrics as representative.
- No reranking step yet (BGE Reranker was in the original Phase 0 stack
  list) — currently ranking is embedding-similarity only.
