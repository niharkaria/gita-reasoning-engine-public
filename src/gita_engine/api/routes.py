"""API routes.

Why this file exists:
    Translates HTTP requests into calls on the reasoning pipeline
    (answer_question) and shapes the result into the API's response
    schema, with proper HTTP error handling for the failure modes we
    already know about (missing GROQ_API_KEY, DB connection issues,
    embedding API unavailability, and -- as of 2026-09-14, item #4 --
    Groq rate-limit exhaustion under concurrent load).

RATE-LIMIT HANDLING (2026-09-14, item #4): fail-fast design, agreed
after the OTPM=1000 discovery in llm_client.py made this genuinely
urgent -- one heavy real call can eat the whole per-minute output
budget, so two real users close together is confirmed fragile, not
just theoretical. Queueing (Celery/Redis) would need real
infrastructure this free-tier portfolio project doesn't otherwise
need. Polling is worse UX here specifically, given the real observed
worst-case wait was 15-30+ minutes -- a spinner that never resolves
reads as broken. Fail-fast with Groq's own real retry-after value
(RateLimitExhaustedError.retry_after, not a guessed number) is the
honestly-scoped choice: real HTTP 429 + a real wait estimate via the
standard Retry-After header.
"""

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text as sql_text

from gita_engine.api.rate_limit import limiter
from gita_engine.api.schemas import AskRequest, AskResponse, CitedPassage, HealthResponse
from gita_engine.core.logging import get_logger
from gita_engine.db.session import get_session
from gita_engine.generation.llm_client import GenerationError, RateLimitExhaustedError
from gita_engine.reasoning.graph import answer_question
from gita_engine.retrieval.retriever import EmbeddingError

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness check — confirms the API is up AND can reach Postgres."""
    db_ok = False
    try:
        with get_session() as session:
            session.execute(sql_text("SELECT 1"))
        db_ok = True
    except Exception:
        logger.error("health_check_db_failed", exc_info=True)

    return HealthResponse(status="ok" if db_ok else "degraded", database_connected=db_ok)


@router.post("/ask", response_model=AskResponse)
@limiter.limit("8/minute")
def ask(request: Request, body: AskRequest) -> AskResponse:
    """Answer a question about the Gita, grounded in the accepted corpus.

    Rate-limited per IP (8/minute) — this endpoint triggers a real Groq
    API call each time it's hit, and the goal is just to stop a bot or
    crawler from burning through Groq's free-tier rate limits, not to
    build a sophisticated abuse-prevention system.
    """
    try:
        result = answer_question(body.question)
    except EmbeddingError as e:
        logger.error("ask_embedding_failed", error=str(e))
        raise HTTPException(
            status_code=502,
            detail=f"The embedding model is unavailable: {e}",
        ) from e
    except RateLimitExhaustedError as e:
        # MUST be caught before the general GenerationError below --
        # this is a subclass of it, and except-clause matching is
        # top-to-bottom, so a preceding `except GenerationError` would
        # silently swallow this and lose the retry_after info.
        wait = round(e.retry_after)
        logger.error("ask_rate_limit_exhausted", retry_after=wait)
        raise HTTPException(
            status_code=429,
            detail=f"The generation model is temporarily rate-limited. Try again in about {wait} seconds.",
            headers={"Retry-After": str(wait)},
        ) from e
    except GenerationError as e:
        logger.error("ask_generation_failed", error=str(e))
        raise HTTPException(
            status_code=502,
            detail=f"The generation model is unavailable: {e}",
        ) from e
    except Exception as e:
        logger.error("ask_unexpected_failure", exc_info=True)
        raise HTTPException(status_code=500, detail="Something went wrong answering that.") from e

    citations = [
        CitedPassage(
            chapter=p.chapter,
            verse_number=p.verse_number,
            passage_type=p.passage_type,
            sanskrit_text=p.sanskrit_text,
            text=p.text,
            similarity=p.similarity,
        )
        for p in result["retrieved"]
    ]

    return AskResponse(question=body.question, answer=result["answer"], citations=citations)
