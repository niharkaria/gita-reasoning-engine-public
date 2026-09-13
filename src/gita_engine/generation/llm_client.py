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

KNOWN FAILURE MODE (found 2026-08-11, real production bug): for broad
questions requiring long reasoning, the model can burn through the
ENTIRE max_tokens budget while still inside <think>...</think>, meaning
no closing </think> tag is ever emitted. The old strip-a-matched-pair
regex found nothing to strip in that case, so the raw internal
reasoning (thousands of words of scratchpad) was returned directly to
the user as if it were the real answer — a real leak of internal
reasoning, not just a cosmetic issue. Fixed by explicitly detecting an
UNCLOSED <think> tag and treating it the same as the existing
reasoning-only-no-answer case: raise GenerationError with a clear
message, rather than silently returning garbage.

FOLLOW-UP (found same day): even with the leak fixed, max_tokens=3072
was still too tight after the system prompt grew (added an instruction
to preserve Sanskrit/Gujarati terms) — the model would sometimes close
</think> properly but then get cut off mid-sentence while writing the
real answer (finish_reason="length", answer silently truncated, e.g.
'The atman is નિત્ય (' with no closing).

First attempted fix: raised max_tokens to 6144. This immediately hit
Groq's real constraint for qwen/qwen3.6-27b on the free tier: TPM
(tokens per minute) = 8000, and this covers INPUT + OUTPUT combined,
not just the generated answer. Retrieved passages can include long
Gujarati commentary blocks (some several hundred words), so a single
request's system prompt + retrieved passages + question can already
be 1500-3000+ input tokens — at max_tokens=6144, a single request
could exceed the entire 8000 TPM budget by itself, causing an
immediate 413 Payload Too Large regardless of other traffic (waiting
between requests did not help, confirming this).

Settled on max_tokens=4096 as the first fix for that specific issue.

SECOND FOLLOW-UP (found 2026-09-13, during Phase 7d faithfulness eval
build): the SAME truncation failure mode resurfaced at max_tokens=4096
on a real question ("What happens to the soul when the body dies?") —
this particular question triggered an unusually long <think> block
(thousands of words), which closed properly (so the unclosed-think
check didn't fire) but left too little budget for the real answer,
which got cut off mid-sentence, mid-citation (finish_reason="length",
completion_tokens=4096 exactly). Real measured prompt size for this
question was 2416 input tokens. Raised max_tokens to 5120 (2416 + 5120
= 7536, still under the 8000 TPM ceiling with real margin) to give more
headroom. This is a mitigation, not a guarantee — a question that
triggers even longer internal reasoning could still hit this. If it
recurs again, the more robust fix would be constraining reasoning
length directly (if Groq exposes such a parameter for this model) or
detecting truncation and automatically retrying with a tighter
system-prompt instruction to reason more concisely, rather than keeps
inching max_tokens upward and eating into TPM headroom.

ALSO ADDED 2026-09-13: retry-with-backoff on HTTP 429 (rate limit)
responses. Found during the same eval build: a single real generation
call uses ~6500 of the 8000 TPM budget by itself (confirmed via a real
call's usage.total_tokens), so two real calls landing within the same
rolling rate-limit window reliably 429 the second one. Groq's 429
response includes a `retry-after` header telling us exactly how long to
wait — so rather than a blind sleep or manual re-run, we read that
header and retry automatically, bounded to a small number of attempts
so a persistent problem still surfaces as a real error instead of
hanging forever.
"""

import re
import time

import httpx

from gita_engine.core.config import get_settings
from gita_engine.core.logging import get_logger

logger = get_logger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_UNCLOSED_THINK_RE = re.compile(r"<think>(?!.*</think>)", flags=re.DOTALL)

MAX_RATE_LIMIT_RETRIES = 5
# Fallback wait if Groq doesn't send a retry-after header for some reason.
DEFAULT_RATE_LIMIT_WAIT_SECONDS = 10.0


class GenerationError(Exception):
    """Raised when the LLM API call fails or returns an unusable response."""


def generate(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 5120,
    temperature: float = 0.2,
) -> str:
    """Call the configured generation model (via Groq) and return its text
    response, with any internal <think> reasoning block stripped out.

    Low temperature (0.2) by default — this is a grounded-answer task
    where we want the model to stick closely to retrieved source text,
    not be creative.

    Retries automatically on HTTP 429 (rate limited), up to
    MAX_RATE_LIMIT_RETRIES times, using Groq's own retry-after header to
    know how long to wait. All other HTTP errors raise immediately, same
    as before.
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

    response = None
    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 2):  # +1 for the initial try
        try:
            response = httpx.post(GROQ_API_URL, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429 and attempt <= MAX_RATE_LIMIT_RETRIES:
                retry_after = e.response.headers.get("retry-after")
                try:
                    wait_seconds = (
                        float(retry_after) if retry_after else DEFAULT_RATE_LIMIT_WAIT_SECONDS
                    )
                except ValueError:
                    wait_seconds = DEFAULT_RATE_LIMIT_WAIT_SECONDS
                logger.warning(
                    "generation_rate_limited_retrying",
                    attempt=attempt,
                    max_retries=MAX_RATE_LIMIT_RETRIES,
                    wait_seconds=wait_seconds,
                )
                time.sleep(wait_seconds)
                continue
            logger.error("generation_request_failed", error=str(e))
            raise GenerationError(f"Generation API call failed: {e}") from e
        except httpx.HTTPError as e:
            logger.error("generation_request_failed", error=str(e))
            raise GenerationError(f"Generation API call failed: {e}") from e
    else:
        raise GenerationError(
            f"Generation API call rate-limited after {MAX_RATE_LIMIT_RETRIES} retries."
        )

    data = response.json()
    try:
        raw_content = str(data["choices"][0]["message"]["content"])
        finish_reason = data["choices"][0].get("finish_reason")
    except (KeyError, IndexError) as e:
        logger.error("generation_response_unparseable", response=data)
        raise GenerationError(f"Unexpected response shape from generation API: {data}") from e

    # Detect an UNCLOSED <think> tag first — this means the model burned
    # its entire max_tokens budget still inside internal reasoning and
    # never got to a real answer. Must check this BEFORE stripping,
    # because the strip regex would simply find no match and silently
    # pass the raw reasoning straight through as if it were the answer.
    if _UNCLOSED_THINK_RE.search(raw_content):
        logger.error(
            "generation_unclosed_think_block",
            finish_reason=finish_reason,
            content_length=len(raw_content),
        )
        raise GenerationError(
            "Generation model ran out of tokens while still reasoning internally "
            f"(no closing </think> tag found) — finish_reason={finish_reason!r}. "
            "Try increasing max_tokens or narrowing the question."
        )

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
