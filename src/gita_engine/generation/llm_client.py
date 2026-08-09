"""Client for calling the generation LLM (Qwen) via Groq's free tier.

Why this file exists:
    Per the Phase 0 architecture decision, generation can't run locally
    (no GPU). Originally wired to Hugging Face's Inference API, but that
    free tier turned out to only give $0.10/month in credit — not
    "generous," just enough for a handful of test calls. Switched to
    Groq, which hosts open Qwen models on a genuinely free tier (real
    rate limits, not a spending cap that silently runs out).

Note on qwen3.6-27b specifically: it's a reasoning model. Groq's API
returns its internal chain-of-thought inline in the `content` field,
wrapped in <think>...</think> tags, followed by the real answer — there
is no separate "reasoning" field to read instead. Confirmed directly
against a real Groq response on 2026-08-09. We strip the <think> block
before returning, since raw internal reasoning isn't meant to be shown
to end users. max_tokens is set higher than a typical model needs,
because the <think> block itself can consume a large chunk of the
budget before the model ever reaches the real answer — confirmed via a
real call that got cut off (finish_reason="length") entirely inside the
<think> block at only 100 tokens.
"""

import re

import httpx

from gita_engine.core.config import get_settings
from gita_engine.core.logging import get_logger

logger = get_logger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


class GenerationError(Exception):
    """Raised when the LLM API call fails or returns an unusable response."""


def generate(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 3072,
    temperature: float = 0.2,
) -> str:
    """Call the configured generation model (via Groq) and return its text
    response, with any internal <think> reasoning block stripped out.

    Low temperature (0.2) by default — this is a grounded-answer task
    where we want the model to stick closely to retrieved source text,
    not be creative.
    """
    settings = get_settings()
    if not settings.groq_api_key:
        raise GenerationError(
            "GROQ_API_KEY is not set in .env — required to call the generation model."
        )

    headers = {"Authorization": f"Bearer {settings.groq_api_key}"}
    payload = {
        "model": settings.generation_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    logger.info("calling_generation_model", model=settings.generation_model)

    try:
        response = httpx.post(GROQ_API_URL, headers=headers, json=payload, timeout=60.0)
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.error("generation_request_failed", error=str(e))
        raise GenerationError(f"Generation API call failed: {e}") from e

    data = response.json()
    try:
        raw_content = str(data["choices"][0]["message"]["content"])
        finish_reason = data["choices"][0].get("finish_reason")
    except (KeyError, IndexError) as e:
        logger.error("generation_response_unparseable", response=data)
        raise GenerationError(f"Unexpected response shape from generation API: {data}") from e

    answer = _THINK_BLOCK_RE.sub("", raw_content).strip()

    if not answer:
        # Most likely cause: the response got cut off by max_tokens while
        # still inside the <think> block, so nothing real-answer-shaped
        # was ever produced. Surface this clearly instead of returning an
        # empty string silently.
        logger.error("generation_truncated_no_answer", finish_reason=finish_reason)
        raise GenerationError(
            "Generation model returned only reasoning, no final answer — "
            f"finish_reason={finish_reason!r}. Try increasing max_tokens."
        )

    if finish_reason == "length":
        # It produced a real answer, but Groq still cut it off before the
        # model was naturally done. Not fatal, but worth knowing about.
        logger.warning("generation_possibly_truncated", finish_reason=finish_reason)

    return answer
