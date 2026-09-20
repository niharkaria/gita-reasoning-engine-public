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

THIRD FOLLOW-UP (2026-09-13, hardening pass after Phase 7e): previously,
a non-fatal truncation (finish_reason == "length" but a real, non-empty
answer was still produced) only logged a warning and then silently
RETURNED THE TRUNCATED ANSWER to the caller -- meaning a real user could
receive an answer cut off mid-sentence with zero indication anything
went wrong, and no way for a caller to detect or retry it. Fixed by
changing generate()'s return type from a plain str to a GenerationResult
NamedTuple carrying both the cleaned answer AND finish_reason, so
callers (graph.py, eval scripts) can explicitly check for this and
decide what to do -- log it, retry, or surface it -- rather than it
being invisible past this module.

FOURTH FOLLOW-UP (2026-09-13, same day, real root-cause fix): a stress
test across the full 18-question golden set (using the THIRD FOLLOW-UP's
new visibility) showed the truncation problem was NOT actually solved by
raising max_tokens alone -- 2 of 10 answered questions were still flagged
[POSSIBLY TRUNCATED], and 1 question failed completely with an unclosed
<think> block, on questions never previously flagged as risky. This
directly contradicted an earlier "tested clean on 2 known triggers"
result from Phase 7e -- too thin a sample, same overconfidence mistake
the reranker experiment made earlier in this project.

Root-caused via direct testing against Groq's raw API (bypassing this
module) on two fronts:

1. reasoning_effort: Groq's own docs page only lists reasoning_effort as
   supported for qwen3-32b, qwen-qwq-32b, and deepseek-r1-distill-
   llama-70b -- NOT qwen3.6-27b. A third-party model registry (pi.dev)
   claimed it WAS supported for this model regardless, with two values:
   "none" and "default". Confirmed via real API calls: reasoning_effort
   ="none" IS honored for this model -- content came back with ZERO
   <think> tags at all, i.e. the entire max_tokens budget goes toward
   the real, visible answer instead of being at risk of being consumed
   by internal reasoning first. This directly eliminates the unclosed-
   <think> failure mode, since there's no <think> block to leave
   unclosed.

2. A real, undocumented-on-the-friendly-dashboard OTPM (output tokens
   per minute) sub-limit was discovered: Limit 1000, confirmed via
   repeated real 429 responses on this exact org/model/service-tier.
   This is a PRE-FLIGHT check against the requested max_tokens itself,
   not just actual usage -- confirmed by a real request with max_tokens
   =1500 rejected instantly even with a completely full 8000/8000 TPM
   window (remaining-tokens header showed 8000, reset in 1ms). This
   means max_tokens must stay under ~1000 for EVERY call, unconditionally
   -- not situational headroom-tuning. This directly conflicts with this
   file's prior history of raising max_tokens for more truncation
   headroom (4096 -> 5120) -- that direction is now a dead end.

Given both findings, max_tokens default was LOWERED (not raised again)
to 950 -- just under the confirmed OTPM=1000 ceiling -- paired with
reasoning_effort="none" so the full 950-token budget goes to the real
answer, not reasoning overhead.

Real batch testing (4 real questions: the previously-failing unclosed-
think question, the zero-citations question, one more previously-
flagged question, and one ordinary control) with reasoning_effort=
"none" + max_tokens=950 showed 3/4 completed cleanly (finish_reason=
"stop", real citations, no <think> tags). The 4th (a question requiring
three separate causal threads to answer fully) still hit finish_reason=
"length" at exactly 950 tokens -- proving reasoning_effort="none" alone
is NOT fully sufficient; some real (not reasoning-related) answers are
just long enough to need more room than a hard OTPM-safe ceiling allows.

To handle that remaining case, added a genuine retry-on-truncation
fallback: if finish_reason=="length" even with reasoning already
disabled, retry ONCE with a stricter brevity instruction appended to the
system prompt for that retry only, asking the model to answer more
concisely and prioritize its strongest citation. This is a real
functional retry (not just a backoff/retry on errors like the 429
logic above) -- it changes what's asked of the model on the second
attempt, since simply repeating the same prompt would very likely
produce the same length again.

FIFTH FOLLOW-UP (2026-09-14, item #4 -- concurrent-user rate-limit
handling): found that the 429-exhaustion path below used to fall
through to the generic "raise GenerationError(f'Generation API call
failed: {e}')" line -- indistinguishable from any other HTTP failure.
The dedicated `for...else` clause that appeared to handle "rate-limited
after N retries" as a distinct case was real, confirmed DEAD CODE: an
exception raised inside a for-loop body propagates immediately and
never reaches that loop's `else` (which only runs if the loop completes
without break AND without an exception escaping). So rate-limit
exhaustion previously carried no retry_after info anywhere.

Fixed by raising a dedicated RateLimitExhaustedError (subclass of
GenerationError, so existing `except GenerationError` call sites are
unaffected) directly inside the 429-handling branch once retries are
exhausted, carrying the real retry_after value just computed (Groq's
own header when present on that final attempt, else our exponential
backoff estimate -- never a made-up number). This is what lets
routes.py return an honest HTTP 429 with a real wait estimate instead
of a generic 502, per the fail-fast design agreed for item #4.
"""

import re
import time
from typing import NamedTuple

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

# Confirmed via real testing 2026-09-13: Groq enforces a pre-flight OTPM
# (output tokens per minute) ceiling of 1000 for this model/org/service-tier,
# checked against the REQUESTED max_tokens itself, not just actual usage.
# Must stay safely under this on every call, unconditionally.
DEFAULT_MAX_TOKENS = 950

# One retry, with a stricter brevity instruction, if the model still runs
# out of room even with reasoning disabled. Not unbounded -- a persistent
# problem should still surface as a real truncated/error result rather
# than silently retrying forever.
MAX_TRUNCATION_RETRIES = 1

BREVITY_RETRY_SUFFIX = (
    "\n\nIMPORTANT: Your previous attempt at this exact answer ran out of "
    "space and was cut off before finishing. This time, answer "
    "significantly more concisely -- aim for roughly half the length, "
    "lead with your single strongest supporting citation, and make sure "
    "you reach a complete conclusion within the available space, even if "
    "that means covering fewer sub-points."
)


class GenerationError(Exception):
    """Raised when the LLM API call fails or returns an unusable response."""


class RateLimitExhaustedError(GenerationError):
    """Raised specifically when Groq's 429 rate limit is still in effect
    after MAX_RATE_LIMIT_RETRIES retries -- a distinct failure mode from
    every other GenerationError (bad key, malformed response, unclosed
    <think>, etc.), because it's the one case where we have a real,
    honest wait estimate to hand back to a caller instead of a flat
    failure. Carries retry_after (seconds): Groq's own retry-after header
    value when it sent one on the final attempt, else our exponential
    backoff estimate -- never a made-up number.

    Subclasses GenerationError (not a standalone exception) so existing
    `except GenerationError` call sites keep working unchanged; callers
    that want to special-case rate-limit exhaustion (e.g. routes.py,
    for a real HTTP 429 with Retry-After) should catch this subclass
    FIRST, since except-clause matching is top-to-bottom and a subclass
    will otherwise be swallowed by a preceding `except GenerationError`.
    """

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class GenerationResult(NamedTuple):
    """Return type for generate() -- carries the cleaned answer plus the
    raw finish_reason, so callers can detect a non-fatal truncation
    (finish_reason == "length" but answer non-empty) instead of it being
    silently swallowed after only a log line."""

    answer: str
    finish_reason: str | None


def _call_groq_once(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    reasoning_effort: str,
) -> tuple[str, str | None]:
    """Make one real call to Groq (with its own internal retry-on-429
    loop), and return the raw (unprocessed) content string plus
    finish_reason. Raises GenerationError (or its RateLimitExhaustedError
    subclass, specifically when 429 retries are exhausted) on any
    unrecoverable failure.

    This is intentionally separate from generate() so the truncation-retry
    logic in generate() can call it twice (original attempt, then a
    brevity-instructed retry) without duplicating the 429-retry machinery.
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
        "reasoning_effort": reasoning_effort,
    }

    logger.info("calling_generation_model", model=settings.generation_model)

    response = None
    for attempt in range(1, MAX_RATE_LIMIT_RETRIES + 2):  # +1 for the initial try
        try:
            response = httpx.post(GROQ_API_URL, headers=headers, json=payload, timeout=60.0)
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                retry_after_header = e.response.headers.get("retry-after")
                if retry_after_header:
                    try:
                        wait_seconds = float(retry_after_header)
                    except ValueError:
                        wait_seconds = DEFAULT_RATE_LIMIT_WAIT_SECONDS * (2 ** (attempt - 1))
                else:
                    # No retry-after header this time (observed to happen
                    # inconsistently -- Groq doesn't always send one).
                    # Exponential backoff as a real fallback: flat 10s
                    # retries were found (2026-09-13) to be insufficient --
                    # 5 retries at 10s each only totals 50s, but the token
                    # window has been observed needing 100s+ to clear after
                    # a heavy call. 10/20/40/80/160s gives real headroom.
                    wait_seconds = DEFAULT_RATE_LIMIT_WAIT_SECONDS * (2 ** (attempt - 1))

                if attempt <= MAX_RATE_LIMIT_RETRIES:
                    logger.warning(
                        "generation_rate_limited_retrying",
                        attempt=attempt,
                        max_retries=MAX_RATE_LIMIT_RETRIES,
                        wait_seconds=wait_seconds,
                    )
                    time.sleep(wait_seconds)
                    continue

                # Retries exhausted. Found 2026-09-14: this branch used to
                # fall through to the generic "raise GenerationError(...)"
                # below, indistinguishable from any other HTTP failure --
                # a for/else clause further down that looked like it
                # handled this case was confirmed dead code (an exception
                # raised inside a for-loop body propagates immediately; it
                # never reaches that loop's else). Raising the dedicated
                # subclass HERE, with the real wait_seconds just computed
                # above, is what actually makes retry_after available to
                # callers -- routes.py needs this to return an honest
                # HTTP 429 with a real wait estimate instead of a generic
                # 502.
                logger.error(
                    "generation_rate_limit_exhausted",
                    max_retries=MAX_RATE_LIMIT_RETRIES,
                    retry_after=wait_seconds,
                )
                raise RateLimitExhaustedError(
                    f"Generation API rate-limited after {MAX_RATE_LIMIT_RETRIES} retries.",
                    retry_after=wait_seconds,
                ) from e

            logger.error("generation_request_failed", error=str(e))
            raise GenerationError(f"Generation API call failed: {e}") from e
        except httpx.HTTPError as e:
            logger.error("generation_request_failed", error=str(e))
            raise GenerationError(f"Generation API call failed: {e}") from e

    assert response is not None
    data = response.json()
    try:
        raw_content = str(data["choices"][0]["message"]["content"])
        finish_reason = data["choices"][0].get("finish_reason")
    except (KeyError, IndexError) as e:
        logger.error("generation_response_unparseable", response=data)
        raise GenerationError(f"Unexpected response shape from generation API: {data}") from e

    return raw_content, finish_reason


def _process_raw_content(raw_content: str, finish_reason: str | None) -> str:
    """Shared processing for a single raw Groq response: detect an
    unclosed <think> block, strip any closed one, and detect an
    empty-after-stripping answer. Raises GenerationError for the fatal
    cases. Returns the cleaned answer string on success.
    """
    # Detect an UNCLOSED <think> tag first — this means the model burned
    # its entire max_tokens budget still inside internal reasoning and
    # never got to a real answer. Must check this BEFORE stripping,
    # because the strip regex would simply find no match and silently
    # pass the raw reasoning straight through as if it were the answer.
    # With reasoning_effort="none" this should no longer occur in
    # practice (confirmed via real testing 2026-09-13 -- zero <think>
    # tags across every test call), but the check is kept as a real
    # safety net rather than assumed away.
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

    return answer


def generate(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.2,
    reasoning_effort: str = "none",
) -> GenerationResult:
    """Call the configured generation model (via Groq) and return its text
    response (with any internal <think> reasoning block stripped out) plus
    the raw finish_reason, wrapped in a GenerationResult.

    Low temperature (0.2) by default — this is a grounded-answer task
    where we want the model to stick closely to retrieved source text,
    not be creative.

    reasoning_effort defaults to "none" -- confirmed via real testing
    2026-09-13 to be honored by qwen/qwen3.6-27b on Groq despite not
    being listed on Groq's own docs page for this model, and to
    eliminate the internal-reasoning-eats-the-budget failure mode
    entirely (zero <think> tags observed across every real test call).

    max_tokens defaults to DEFAULT_MAX_TOKENS (950) -- kept safely under
    Groq's confirmed real OTPM (output tokens per minute) pre-flight
    ceiling of 1000 for this model/org/service-tier. This is a hard
    ceiling, not situational headroom -- do not raise this without first
    re-confirming the OTPM limit hasn't changed.

    Retries automatically on HTTP 429 (rate limited), up to
    MAX_RATE_LIMIT_RETRIES times, using Groq's own retry-after header to
    know how long to wait. If retries are exhausted, raises
    RateLimitExhaustedError (a GenerationError subclass) carrying the
    real retry_after value, rather than a generic GenerationError, so a
    caller can return honest backpressure instead of a flat failure. All
    other HTTP errors raise GenerationError immediately.

    Separately, if the model still produces a real (non-empty) answer
    that gets cut off (finish_reason == "length") even with reasoning
    disabled, this function retries ONCE with a stricter brevity
    instruction appended to the system prompt for that retry only --
    confirmed via real testing 2026-09-13 that reasoning_effort="none"
    alone does not guarantee every real answer fits under the OTPM-safe
    max_tokens ceiling for questions needing to cover several distinct
    points.
    """
    raw_content, finish_reason = _call_groq_once(
        system_prompt,
        user_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
    )
    answer = _process_raw_content(raw_content, finish_reason)

    retries_used = 0
    while finish_reason == "length" and retries_used < MAX_TRUNCATION_RETRIES:
        retries_used += 1
        logger.warning(
            "generation_retrying_for_truncation",
            attempt=retries_used,
            max_retries=MAX_TRUNCATION_RETRIES,
        )
        retry_system_prompt = system_prompt + BREVITY_RETRY_SUFFIX
        raw_content, finish_reason = _call_groq_once(
            retry_system_prompt,
            user_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )
        answer = _process_raw_content(raw_content, finish_reason)

    if finish_reason == "length":
        # It produced a real answer, but Groq still cut it off before the
        # model was naturally done -- even after the brevity retry above.
        # Not fatal on its own -- as of the 2026-09-13 hardening pass,
        # this is no longer swallowed after just a log line. finish_reason
        # is returned to the caller via GenerationResult so it can be
        # detected and acted on explicitly.
        logger.warning(
            "generation_possibly_truncated",
            finish_reason=finish_reason,
            retries_used=retries_used,
        )

    return GenerationResult(answer=answer, finish_reason=finish_reason)

def translate_query(question: str, *, target_language: str = "Gujarati") -> str:
    """Translate an English question into `target_language` before it gets
    embedded -- an experiment to test whether a same-language match against
    the corpus (which is entirely Gujarati commentary/translation text)
    improves retrieval over embedding the raw English query directly.

    Reuses generate() rather than a separate HTTP call, so this inherits
    the same Groq error handling, 429 retry behavior, and <think>-block
    stripping already relied on elsewhere -- not a second code path to
    maintain.

    Low max_tokens (a translated question is short) and temperature 0.0
    (translation should be as literal/consistent as possible, not
    creative). Not yet proven to help this corpus -- that's exactly what
    the --translate flag on evaluate_retrieval.py is for.
    """
    system_prompt = (
        f"Translate the user's question into {target_language}. "
        "Return ONLY the translation itself -- no explanation, no quotation "
        "marks, no additional commentary."
    )
    result = generate(
        system_prompt,
        question,
        max_tokens=200,
        temperature=0.0,
        reasoning_effort="none",
    )
    return result.answer.strip()
