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
    - Embedding the query happens via Cloudflare Workers AI's hosted
      @cf/baai/bge-m3 (see embed_query below), NOT loaded locally. Local
      loading caused an OOM crash on Render's free tier (512MB total RAM,
      nowhere near enough for BGE-M3's ~2.2GB weights). It is the SAME model
      as the Kaggle-generated corpus embeddings; a real compatibility check
      showed cosine similarity of 0.991-1.000 between Cloudflare's vectors
      and the stored ones, so no re-embedding or schema change was needed.
      (History: this used Hugging Face's router until HF removed the free
      credit for free accounts, which made every call fail with a 402.)
"""

from dataclasses import dataclass

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from gita_engine.core.config import get_settings
from gita_engine.core.logging import get_logger
from gita_engine.db.models import EMBEDDING_DIM, Commentary, Source, Translation, Verse

logger = get_logger(__name__)

TOP_K_DEFAULT = 5

CF_EMBEDDING_MODEL = "@cf/baai/bge-m3"


class EmbeddingError(Exception):
    """Raised when the embedding API call fails or returns an unusable response."""


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


def embed_query(query: str, *, prefix: str | None = None) -> list[float]:
    """Embed a single query string using BGE-M3, via Cloudflare Workers AI's
    hosted REST API rather than loading the model locally.

    Uses the SAME model (BAAI/bge-m3) as the corpus embeddings already
    stored in Supabase, so query and corpus vectors stay comparable.

    prefix: optional instruction text prepended to the query before
    embedding. None by default (unchanged behavior); used by
    evaluate_retrieval.py's --prefix flag to A/B test this against the
    audited golden set.

    NOTE on response shape (confirmed via a real call): the vector is at
    response["result"]["data"][0] — a list of 1024 floats.
    """
    settings = get_settings()
    if not settings.cloudflare_account_id or not settings.cloudflare_api_token:
        raise EmbeddingError(
            "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must be set — "
            "required to call the embedding API."
        )

    text_to_embed = f"{prefix}{query}" if prefix else query

    url = (
        "https://api.cloudflare.com/client/v4/accounts/"
        f"{settings.cloudflare_account_id}/ai/run/{CF_EMBEDDING_MODEL}"
    )
    headers = {"Authorization": f"Bearer {settings.cloudflare_api_token}"}
    payload = {"text": [text_to_embed]}

    logger.info("calling_embedding_model", model=CF_EMBEDDING_MODEL)

    try:
        response = httpx.post(url, headers=headers, json=payload, timeout=30.0)
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.error("embedding_request_failed", error=str(e))
        raise EmbeddingError(f"Embedding API call failed: {e}") from e

    try:
        data = response.json()
        if data.get("success") is False:
            raise ValueError(f"API reported failure: {data.get('errors')}")
        embedding = data["result"]["data"][0]
        if not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIM:
            raise ValueError(
                f"expected a list of {EMBEDDING_DIM} floats, got "
                f"{type(embedding).__name__} of length "
                f"{len(embedding) if hasattr(embedding, '__len__') else 'n/a'}"
            )
    except (AttributeError, KeyError, IndexError, TypeError, ValueError) as e:
        logger.error("embedding_response_unparseable", error=str(e))
        raise EmbeddingError(f"Unexpected response from embedding API: {e}") from e

    return [float(x) for x in embedding]


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