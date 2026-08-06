"""Client for calling the generation LLM (Qwen Instruct) via a hosted API.

Why this file exists:
    Per the Phase 0 architecture decision, generation can't run locally
    (no GPU). This wraps the Hugging Face Inference API call so the rest
    of the reasoning pipeline doesn't need to know the HTTP details.

NOT independently tested in this sandbox — no network access to
huggingface.co here. The HTTP shape below follows HF's documented
chat-completion-style Inference API; verify against current HF docs and
your actual HF_API_TOKEN before relying on this in production, since
hosted-API request/response shapes do change over time.
"""

import httpx

from gita_engine.core.config import get_settings
from gita_engine.core.logging import get_logger

logger = get_logger(__name__)

HF_INFERENCE_URL = "https://api-inference.huggingface.co/models/{model}/v1/chat/completions"


class GenerationError(Exception):
    """Raised when the LLM API call fails or returns an unusable response."""


def generate(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.2,
) -> str:
    """Call the configured generation model and return its text response.

    Low temperature (0.2) by default — this is a grounded-answer task
    where we want the model to stick closely to retrieved source text,
    not be creative.
    """
    settings = get_settings()
    if not settings.hf_api_token:
        raise GenerationError(
            "HF_API_TOKEN is not set in .env — required to call the generation model."
        )

    url = HF_INFERENCE_URL.format(model=settings.generation_model)
    headers = {"Authorization": f"Bearer {settings.hf_api_token}"}
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
        response = httpx.post(url, headers=headers, json=payload, timeout=60.0)
        response.raise_for_status()
    except httpx.HTTPError as e:
        logger.error("generation_request_failed", error=str(e))
        raise GenerationError(f"Generation API call failed: {e}") from e

    data = response.json()
    try:
        return str(data["choices"][0]["message"]["content"])
    except (KeyError, IndexError) as e:
        logger.error("generation_response_unparseable", response=data)
        raise GenerationError(f"Unexpected response shape from generation API: {data}") from e
