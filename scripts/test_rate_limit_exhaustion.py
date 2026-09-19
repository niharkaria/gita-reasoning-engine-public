"""Throwaway test script -- NOT part of the real codebase, do not commit.
Verifies RateLimitExhaustedError is actually raised (with the real
retry_after value) when Groq returns 429 on every attempt, by mocking
httpx.post directly rather than trusting the logic by inspection alone.
"""

from unittest.mock import patch, MagicMock
import httpx

from gita_engine.generation.llm_client import (
    _call_groq_once,
    RateLimitExhaustedError,
    MAX_RATE_LIMIT_RETRIES,
)


def make_429_response(retry_after_value: str = "7"):
    resp = MagicMock()
    resp.status_code = 429
    resp.headers = {"retry-after": retry_after_value}
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "429 rate limited", request=MagicMock(), response=resp
    )
    return resp


call_count = 0


def fake_post(*args, **kwargs):
    global call_count
    call_count += 1
    return make_429_response("7")


with patch("gita_engine.generation.llm_client.httpx.post", side_effect=fake_post), \
     patch("gita_engine.generation.llm_client.time.sleep") as mock_sleep, \
     patch("gita_engine.generation.llm_client.get_settings") as mock_settings:

    mock_settings.return_value.groq_api_key = "fake-key-for-test"
    mock_settings.return_value.generation_model = "fake-model"

    try:
        _call_groq_once(
            "system prompt",
            "user prompt",
            max_tokens=950,
            temperature=0.2,
            reasoning_effort="none",
        )
        print("FAIL: expected RateLimitExhaustedError, nothing was raised")
    except RateLimitExhaustedError as e:
        expected_calls = MAX_RATE_LIMIT_RETRIES + 1
        assert call_count == expected_calls, (
            f"FAIL: expected {expected_calls} real calls to httpx.post, got {call_count}"
        )
        assert e.retry_after == 7.0, f"FAIL: expected retry_after=7.0, got {e.retry_after!r}"
        assert mock_sleep.call_count == MAX_RATE_LIMIT_RETRIES, (
            f"FAIL: expected {MAX_RATE_LIMIT_RETRIES} sleep calls, got {mock_sleep.call_count}"
        )
        print(f"PASS: RateLimitExhaustedError raised after {call_count} real calls, "
              f"retry_after={e.retry_after}, slept {mock_sleep.call_count} times")
    except Exception as e:
        print(f"FAIL: wrong exception type raised: {type(e).__name__}: {e}")
