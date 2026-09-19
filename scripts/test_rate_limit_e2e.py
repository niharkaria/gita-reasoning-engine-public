"""Throwaway test -- NOT part of the real codebase, do not commit.

Verifies the FULL real path (routes.py -> graph.py -> llm_client.py)
turns a rate-limit exhaustion into a real HTTP 429 with a real
Retry-After header, rather than trusting the unit-level mock test
(which only proved RateLimitExhaustedError raises correctly in
isolation) or inspection of graph.py (which only proved no try/except
swallows it -- LangGraph's own invoke() behavior on a node exception
was still an unconfirmed unknown before this test).

Mocks httpx.post at the same boundary the earlier unit test used, but
this time drives it through the real FastAPI app via TestClient, so
it exercises routes.py -> answer_question() -> graph.py's _generate_node
-> llm_client.py's _call_groq_once(), including whatever LangGraph
does internally with a node exception -- without spending real Groq
OTPM quota or waiting on real backoff sleeps.
"""

from unittest.mock import patch, MagicMock

import httpx
from fastapi.testclient import TestClient

from gita_engine.api.main import create_app

app = create_app()
client = TestClient(app)


def make_429_response(retry_after_value: str = "9"):
    resp = MagicMock()
    resp.status_code = 429
    resp.headers = {"retry-after": retry_after_value}
    resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "429 rate limited", request=MagicMock(), response=resp
    )
    return resp


# First real attempt at this test hit a REAL embedding-API 429 (BAAI/bge-m3
# via Hugging Face) before ever reaching generation -- embed_query() makes
# its own separate real httpx.post call that our llm_client-only mock
# didn't touch. Mocking embed_query/retrieve directly here so the test
# isolates the generation rate-limit path specifically, rather than
# depending on retriever.py's internal HTTP shape.
with patch(
    "gita_engine.reasoning.graph.embed_query",
    return_value=[0.0] * 1024,  # dummy embedding vector, shape doesn't matter -- retrieve() is also mocked below
), patch(
    "gita_engine.reasoning.graph.retrieve",
    return_value=[],  # empty passages list -- fine, we only care whether generation's 429 propagates correctly
), patch(
    "gita_engine.generation.llm_client.httpx.post",
    return_value=make_429_response("9"),
), patch("gita_engine.generation.llm_client.time.sleep"):  # skip real waiting

    response = client.post("/ask", json={"question": "What is dharma?"})

    print(f"Real status_code: {response.status_code}")
    print(f"Real headers: {dict(response.headers)}")
    print(f"Real body: {response.json()}")

    assert response.status_code == 429, f"FAIL: expected 429, got {response.status_code}"
    assert response.headers.get("retry-after") == "9", (
        f"FAIL: expected Retry-After=9, got {response.headers.get('retry-after')!r}"
    )
    print("PASS: real end-to-end 429 with correct Retry-After header confirmed.")
