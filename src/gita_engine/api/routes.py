"""API routes.

Why this file exists:
    Translates HTTP requests into calls on the reasoning pipeline
    (answer_question) and shapes the result into the API's response
    schema, with proper HTTP error handling for the failure modes we
    already know about (missing GROQ_API_KEY, DB connection issues).
"""

from fastapi import APIRouter, HTTPException, Request
from sqlalchemy import text as sql_text

from gita_engine.api.rate_limit import limiter
from gita_engine.api.schemas import AskRequest, AskResponse, CitedPassage, HealthResponse
from gita_engine.core.logging import get_logger
from gita_engine.db.session import get_session
from gita_engine.generation.llm_client import GenerationError
from gita_engine.reasoning.graph import answer_question

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
            text=p.text,
            similarity=p.similarity,
        )
        for p in result["retrieved"]
    ]

    return AskResponse(question=body.question, answer=result["answer"], citations=citations)
