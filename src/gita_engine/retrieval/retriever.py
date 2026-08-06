"""Retrieve the most relevant translations and commentaries for a query.

Why this file exists:
    The actual retrieval step of the RAG pipeline: given a user's question,
    find the passages (translations/commentaries) most semantically similar
    to it, using pgvector's cosine-distance search over the embeddings
    generated on Kaggle and imported via import_embeddings.py.

Design notes:
    - Only retrieves from Source rows where is_accepted=True — this is the
      mechanism that enforces FR7 (never answer from outside the accepted
      corpus), decided back in the Phase 3 schema design.
    - pgvector's `<=>` operator computes cosine DISTANCE (0 = identical,
      2 = opposite), not similarity — we convert to a similarity score
      (1 - distance) for a more intuitive 0..1-ish scale in results.
    - Embedding the query itself happens via BGE-M3 loaded locally
      (see embed_query below). This is fine for a SINGLE short query at
      request time — it's bulk-encoding ~1200 corpus texts that needed
      Kaggle's GPU, not one-off query embedding.
"""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from gita_engine.db.models import Commentary, Source, Translation, Verse

TOP_K_DEFAULT = 5


@dataclass
class RetrievedPassage:
    """One retrieved passage, ready to feed into the reasoning/generation step."""

    verse_id: int
    chapter: int
    verse_number: int
    sanskrit_text: str
    passage_type: str  # 'translation' | 'commentary'
    text: str
    source_citation: str
    similarity: float


def embed_query(query: str) -> list[float]:
    """Embed a single query string using BGE-M3, loaded locally.

    NOT tested in this sandbox — no GPU/network access to download model
    weights here. Encoding ONE short query on CPU is expected to take low
    seconds, which is fine for interactive use; this is a fundamentally
    different workload from bulk-encoding the ~1200-text corpus (which is
    why that step runs on Kaggle instead). Verify timing on your machine
    once you have the `sentence-transformers` package installed.
    """
    from sentence_transformers import SentenceTransformer

    # Cached at module level so repeated calls (e.g. multiple queries in
    # one process) don't reload the model from disk every time.
    if not hasattr(embed_query, "_model"):
        embed_query._model = SentenceTransformer("BAAI/bge-m3", device="cpu")  # type: ignore[attr-defined]

    model = embed_query._model  # type: ignore[attr-defined]
    embedding = model.encode(query, normalize_embeddings=True)
    return list(embedding.tolist())


def retrieve(
    session: Session, query_embedding: list[float], top_k: int = TOP_K_DEFAULT
) -> list[RetrievedPassage]:
    """Find the top-k most similar translation/commentary passages to the
    query embedding, restricted to accepted sources, across both tables."""
    results: list[RetrievedPassage] = []

    translation_rows = session.execute(
        select(
            Translation.id,
            Translation.text,
            Verse.id.label("verse_id"),
            Verse.chapter,
            Verse.verse_number,
            Verse.sanskrit_text,
            Source.citation_label,
            (1 - Translation.embedding.cosine_distance(query_embedding)).label("similarity"),
        )
        .join(Verse, Verse.id == Translation.verse_id)
        .join(Source, Source.id == Translation.source_id)
        .where(Source.is_accepted.is_(True), Translation.embedding.is_not(None))
        .order_by(Translation.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    ).all()

    for row in translation_rows:
        results.append(
            RetrievedPassage(
                verse_id=row.verse_id,
                chapter=row.chapter,
                verse_number=row.verse_number,
                sanskrit_text=row.sanskrit_text,
                passage_type="translation",
                text=row.text,
                source_citation=row.citation_label,
                similarity=float(row.similarity),
            )
        )

    commentary_rows = session.execute(
        select(
            Commentary.id,
            Commentary.text,
            Verse.id.label("verse_id"),
            Verse.chapter,
            Verse.verse_number,
            Verse.sanskrit_text,
            Source.citation_label,
            (1 - Commentary.embedding.cosine_distance(query_embedding)).label("similarity"),
        )
        .join(Verse, Verse.id == Commentary.verse_id)
        .join(Source, Source.id == Commentary.source_id)
        .where(Source.is_accepted.is_(True), Commentary.embedding.is_not(None))
        .order_by(Commentary.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    ).all()

    for row in commentary_rows:
        results.append(
            RetrievedPassage(
                verse_id=row.verse_id,
                chapter=row.chapter,
                verse_number=row.verse_number,
                sanskrit_text=row.sanskrit_text,
                passage_type="commentary",
                text=row.text,
                source_citation=row.citation_label,
                similarity=float(row.similarity),
            )
        )

    # Merge translation+commentary candidates and keep the overall top_k by
    # similarity, so a verse whose commentary is a great match isn't
    # crowded out just because we queried two tables separately.
    results.sort(key=lambda r: r.similarity, reverse=True)
    return results[:top_k]
