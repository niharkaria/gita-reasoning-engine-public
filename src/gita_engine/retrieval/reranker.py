"""Re-rank a wider pool of retrieved candidates for better precision.

Why this file exists:
    Plain embedding-similarity search (retriever.py) is fast but
    approximate — it can pull in passages that are topically nearby but
    not actually relevant to the specific question (observed directly:
    asking "who is the true enjoyer of sacrifices" pulled in 4 of 5 top
    candidates that didn't address the question at all). A reranker is a
    more accurate (but slower/costlier) cross-encoder model that scores
    a (query, passage) pair directly, rather than comparing independent
    embeddings — so it's much better at judging true relevance.

    Standard pattern: cast a WIDE net with cheap embedding search (e.g.
    top 20), then use the reranker to re-score and keep only the
    genuinely best few (e.g. top 5) to actually feed the generation
    model. This file implements the re-scoring step; retriever.py still
    does the initial wide-net embedding search.

Design notes:
    - Uses Hugging Face's hosted Inference API (router.huggingface.co),
      same hosting approach as embed_query in retriever.py — keeps the
      API process from needing to load the reranker model's weights
      locally (same OOM concern that motivated the embedding move).
    - Unlike embedding failures, a failed rerank call is treated as a
      hard error (RerankError), not a silent fallback to embedding-only
      ranking — this was a deliberate choice: consistency with how
      EmbeddingError/GenerationError already behave (fail loud, not
      silently degrade), rather than introducing a third failure
      handling pattern.
    - Response shape confirmed via real curl/live tests: for N input
      pairs, HF returns a SINGLE outer list containing ONE inner list of
      N result dicts, each with a "score" field — e.g. for N=1:
      [[{"label": "LABEL_0", "score": 0.5}]]; for N=20:
      [[{...}, {...}, ... 20 dicts]]. The N=1 case is ambiguous (both
      "N single-item lists" and "1 list of N items" look identical when
      N=1) — this was initially misread from the N=1 test alone and
      only caught by testing with a real N=20 batch. data[0] is the
      list of N results, in input order, zipped back with the original
      passages by position.
"""

import httpx

from gita_engine.core.config import get_settings
from gita_engine.core.logging import get_logger
from gita_engine.retrieval.retriever import RetrievedPassage

logger = get_logger(__name__)


class RerankError(Exception):
    """Raised when the reranker API call fails or returns an unusable response."""


def _build_rerank_url(model: str) -> str:
    return f"https://router.huggingface.co/hf-inference/models/{model}/pipeline/text-classification"


def rerank(query: str, passages: list[RetrievedPassage], top_k: int) -> list[RetrievedPassage]:
    """Re-score a candidate pool of passages against the query using a
    cross-encoder reranker, and return the best top_k.

    passages should already be a wider candidate pool (e.g. top 20 from
    embedding search), not just the final top_k — reranking narrows a
    wide pool down, it doesn't widen a narrow one.
    """
    if not passages:
        return []

    settings = get_settings()
    if not settings.hf_api_token:
        raise RerankError("HF_API_TOKEN is not set in .env — required to call the reranker API.")

    headers = {"Authorization": f"Bearer {settings.hf_api_token}"}
    payload = {
        "inputs": [{"text": query, "text_pair": f"{p.sanskrit_text}\n{p.text}"} for p in passages]
    }

    logger.info(
        "calling_reranker_model", model=settings.reranker_model, num_candidates=len(passages)
    )

    try:
        response = httpx.post(
            _build_rerank_url(settings.reranker_model),
            headers=headers,
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.error("rerank_request_failed", error=str(e))
        raise RerankError(f"Rerank API call failed: {e}") from e

    data = response.json()
    try:
        results = data[0]
        if len(results) != len(passages):
            raise ValueError(f"Expected {len(passages)} rerank results, got {len(results)}")
        scores = [float(item["score"]) for item in results]
    except (KeyError, IndexError, TypeError, ValueError) as e:
        logger.error("rerank_response_unparseable", response=data)
        raise RerankError(f"Unexpected response shape from rerank API: {data}") from e

    scored = list(zip(scores, passages, strict=True))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [passage for _score, passage in scored[:top_k]]
